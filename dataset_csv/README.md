# Dataset metadata

Both CSVs contain exactly one row per slide in the supplied medical SSL HDF5 feature files. Slide IDs match the HDF5 keys, and labels are copied from the HDF5 slide attributes. Labels in these CSVs are for reference; the training loader reads the HDF5 labels.

## CAMELYON17

`camelyon17.csv` contains 499 slides with columns `slide_id,center,case_id,label`. `case_id` identifies the patient from the slide ID.

Center assignments were verified against the training file paths in the [CAMELYON17 dataset manifest](https://academictorrents.com/details/fa82a73b0903c5a75cf24bc68a980727bb1a807e), downloaded as metadata only. The catalog assigns patients 000–019 to center 0, 020–039 to center 1, 040–059 to center 2, 060–079 to center 3, and 080–099 to center 4. The CSV includes only slides present in `feature/C17/patch_feats_pretrain_medical_ssl.h5`.

The current loader reserves centers >= `cam17_test_center_min` for testing and splits the remaining slides using `val_fraction` and `seed`. With threshold 3, validation fraction 0.10, and seed 42, the sets contain 269 training, 30 validation, and 200 test slides. This is a center-based experiment split of the available challenge training slides, rather than the challenge's official training/test partition. The CSV stores center metadata; the loader computes the train/validation/test sets at runtime.

## BRACS

`bracs.csv` contains 537 slides with columns `slide_id,split_info,label`: 389 training, 62 validation, and 86 test slides.

Split assignments were preserved from the existing local metadata at `/home/linfeng/Downloads/MELD_paper_aligned_fixed/MELD/dataset_csv/bracs.csv`, which contains 547 slides. All 537 slides in `feature/BRACS/patch_feats_pretrain_medical_ssl.h5` were found in that source, and every corresponding label matched. The 10 source rows without HDF5 features were omitted. The source's original `split_info` values were retained; no random split was generated.
