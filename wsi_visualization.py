"""Plot CAMELYON16 whole-slide attention with optional ASAP XML contours.

Patch and annotation coordinates use the original level-0 (x, y) frame.
The default 512-pixel square patch footprint is an assumption: the supplied
feature files do not record patch-size metadata.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import torch

def read_annotation_contours(xml_path):
    # ASAP XML coordinates are (x, y) in the original level-0 WSI frame.
    if not xml_path.exists():
        return []
    polygons = []
    for annotation in ET.parse(xml_path).getroot().findall('.//Annotation'):
        if annotation.get('Type', '').lower() != 'polygon':
            continue
        points = sorted(annotation.findall('./Coordinates/Coordinate'),
                        key=lambda p: int(p.get('Order', '0')))
        xy = np.array([[float(p.get('X')), float(p.get('Y'))] for p in points])
        if len(xy) >= 3:
            polygons.append((np.vstack([xy, xy[0]]), annotation.get('PartOfGroup', '')))
    return polygons


def attention_percentiles(weights):
    # Midranks handle tied values; colors describe relative attention within a slide.
    sorted_weights = np.sort(weights)
    left = np.searchsorted(sorted_weights, weights, side='left')
    right = np.searchsorted(sorted_weights, weights, side='right')
    return 100.0 * (left + right - 1) / (2 * max(len(weights) - 1, 1))


def render_attention(image, extent, coords, percentiles, patch_size_level0=512):
    # Plot everything in level-0 coordinates, including resized images and contours.
    x0, x1, y1, y0 = extent
    width, height = image.size
    sx, sy = width / (x1 - x0), height / (y1 - y0)
    sums = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint32)
    for (x, y), value in zip(coords, percentiles):
        if x >= x1 or y >= y1 or x + patch_size_level0 <= x0 or y + patch_size_level0 <= y0:
            continue
        left = max(0, int(np.floor((x - x0) * sx)))
        top = max(0, int(np.floor((y - y0) * sy)))
        right = min(width, int(np.ceil((x + patch_size_level0 - x0) * sx)))
        bottom = min(height, int(np.ceil((y + patch_size_level0 - y0) * sy)))
        sums[top:bottom, left:right] += value
        counts[top:bottom, left:right] += 1
    heat = np.ma.array(sums / np.maximum(counts, 1), mask=(counts == 0))
    return heat  # Areas without extracted patches remain transparent.


def draw_contours(ax, contours):
    for xy, group in contours:
        # ASAP exclusion / non-tumor polygons, if present, use dashed cyan lines.
        excluded = group.lower() in {'_2', 'exclusion', 'exclude', 'normal'}
        ax.plot(xy[:, 0], xy[:, 1], color='cyan' if excluded else 'lime',
                linestyle='--' if excluded else '-', linewidth=1.2)


def visualize_wsi_attention(
    model, conf, project_dir, best_model_state, *,
    slide_ids=('test_068', 'test_067'), patch_size_level0=512,
    thumbnail_max_size=1800, overlay_alpha=0.55, normal_zoom_size=16384,
    output_dir=None, prefix='C16_baseline', show=True,
):
    """Restore the best weights and show WSI, attention, and zoom for each slide.

    Missing XML annotations are allowed. Green lines show tumor contours;
    a blue rectangle marks the zoom. Heatmap colors are within-slide attention
    percentiles, and uncovered areas remain transparent.

    Returns a dictionary keyed by slide ID with raw coordinates, normalized
    attention weights, and slide-level class probabilities. Adjust
    ``patch_size_level0`` to the extraction footprint when known.
    """
    try:
        import openslide
    except (ImportError, OSError) as exc:
        raise ImportError(
            'Install the WSI dependencies with: %pip install openslide-python openslide-bin'
        ) from exc
    slide_dir = Path(project_dir) / 'slide'
    if str(conf.dataset).upper() != 'CAMELYON16':
        raise ValueError('Select DATASET = "C16" and train its model for these two slides.')
    if best_model_state is None:
        raise RuntimeError('Run Step 6 first so best_model_state contains trained weights.')
    model.load_state_dict(best_model_state)
    model.eval()
    attention_device = next(model.parameters()).device

    attention_visualizations = {}
    for slide_id in slide_ids:
        slide_path = slide_dir / f'{slide_id}.tif'
        if not slide_path.exists():
            raise FileNotFoundError(f'WSI not found: {slide_path}')
        with h5py.File(conf.data_dir, 'r') as h5:
            if slide_id not in h5:
                raise KeyError(f'{slide_id} is missing from {conf.data_dir}')
            group = h5[slide_id]
            if 'coords' not in group:
                raise ValueError(f'{slide_id} has no patch coordinates for a WSI overlay.')
            features = np.asarray(group['feat'][:], dtype=np.float32)
            coords = np.asarray(group['coords'][:], dtype=np.float64)
            label = int(group.attrs['label'])
        if features.ndim != 2 or coords.shape != (len(features), 2) or len(features) == 0:
            raise ValueError(f'{slide_id}: expected nonempty feat [N,D] and coords [N,2].')
        if not np.isfinite(coords).all():
            raise ValueError(f'{slide_id}: patch coordinates contain nonfinite values.')
        # One complete, unpadded bag preserves the softmax over all slide patches.
        with torch.no_grad():
            logits, attention = model(
                torch.from_numpy(features).unsqueeze(0).to(attention_device),
                return_attention=True,
            )
        weights = attention.detach().cpu().numpy().reshape(-1)
        if len(weights) != len(coords) or not np.isfinite(weights).all():
            raise ValueError(f'{slide_id}: invalid attention or coordinate/attention mismatch.')
        probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        percentiles = attention_percentiles(weights)
        contours = read_annotation_contours(slide_dir / f'{slide_id}.xml')
        tumor_contours = [xy for xy, group in contours
                          if group.lower() not in {'_2', 'exclusion', 'exclude', 'normal'}]

        with openslide.OpenSlide(str(slide_path)) as wsi:
            w, h = wsi.dimensions
            if (coords < 0).any() or (coords[:, 0] >= w).any() or (coords[:, 1] >= h).any():
                raise ValueError(f'{slide_id}: coords must be level-0 (x,y) within WSI bounds.')
            overview = wsi.get_thumbnail((thumbnail_max_size, thumbnail_max_size))
            overview_extent = (0, w, h, 0)
            if tumor_contours:
                xy = np.concatenate(tumor_contours)
                lower, upper = xy.min(axis=0) - 2048, xy.max(axis=0) + 2048
                zoom_title = 'Tumor annotation region + attention'
            else:
                center = coords[int(weights.argmax())] + patch_size_level0 / 2
                lower, upper = center - normal_zoom_size / 2, center + normal_zoom_size / 2
                zoom_title = 'Highest-attention region (no annotation)'
            x0, y0 = np.maximum(0, np.floor(lower)).astype(int)
            x1, y1 = np.minimum([w, h], np.ceil(upper)).astype(int)
            level = wsi.get_best_level_for_downsample(
                max(1.0, max(x1 - x0, y1 - y0) / thumbnail_max_size))
            downsample = wsi.level_downsamples[level]
            size = (int(np.ceil((x1 - x0) / downsample)),
                    int(np.ceil((y1 - y0) / downsample)))
            zoom = wsi.read_region((int(x0), int(y0)), level, size).convert('RGB')
            # ceil() can add a partial pixel; use the actual region extent for alignment.
            zoom_extent = (x0, x0 + size[0] * downsample,
                           y0 + size[1] * downsample, y0)
            zoom.thumbnail((thumbnail_max_size, thumbnail_max_size))

        fig, axes = plt.subplots(1, 3, figsize=(16, 8), layout='constrained')
        norm = Normalize(0, 100)
        views = [(overview, overview_extent, False, 'WSI + annotation contours'),
                 (overview, overview_extent, True, 'WSI + attention + contours'),
                 (zoom, zoom_extent, True, zoom_title)]
        for ax, (image, extent, show_heat, title) in zip(axes, views):
            if show_heat:
                ax.imshow(image.convert('L'), extent=extent, origin='upper',
                          cmap='gray', vmin=0, vmax=255)
            else:
                ax.imshow(image, extent=extent, origin='upper')
            if show_heat:
                heat = render_attention(image, extent, coords, percentiles, patch_size_level0)
                heat_artist = ax.imshow(heat, extent=extent, origin='upper', cmap='viridis',
                                        norm=norm, alpha=overlay_alpha, interpolation='nearest')
            draw_contours(ax, contours)
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            ax.set_title(title if contours else title.replace(' + annotation contours',
                                                             ' (normal; no annotation)').replace(' + contours', ''))
            ax.set_axis_off()
        for ax in axes[:2]:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor='deepskyblue', linewidth=1))
        if tumor_contours:
            axes[0].legend(handles=[Line2D([0], [0], color='lime', label='XML tumor contour')],
                           loc='upper left', fontsize=8)
        fig.colorbar(heat_artist, ax=list(axes[1:]), shrink=0.65,
                     label='Within-slide attention percentile (not tumor probability)'
                     )
        class_names = ['Normal', 'Tumor']
        prediction = int(probabilities.argmax())
        fig.suptitle(f'{slide_id} | Ground truth: {class_names[label]} | '
                     f'Prediction: {class_names[prediction]} | P(tumor)={probabilities[1]:.3f}')
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path / f'{prefix}_{slide_id}_attention.png', dpi=160,
                        bbox_inches='tight', facecolor='white')
            np.savez_compressed(output_path / f'{prefix}_{slide_id}_attention.npz',
                                coords=coords, attention=weights, probabilities=probabilities,
                                patch_size_level0=patch_size_level0)
        if show:
            plt.show()
        else:
            plt.close(fig)
        print(f'{slide_id}: {len(weights):,} patches; attention sum={weights.sum():.6f}; '
              f'{len(contours)} annotation contours' + (' (no XML available)' if not contours else ''))
        attention_visualizations[slide_id] = dict(coords=coords, attention=weights,
                                                 probabilities=probabilities)
    return attention_visualizations
