"""The optional storage cache must preserve full-bag values and optimization."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataloader import HDF5FeatureDataset, mil_collate_fn
from models import ABMIL
from utils import Struct


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for device-cache equivalence checks')
class GPUCacheEquivalenceTests(unittest.TestCase):
    def setUp(self):
        base = ROOT/'tests/.tmp'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.path = Path(self.temp.name)/'synthetic_cache.h5'
        self.conf = Struct(dataset='CAMELYON16', n_class=2, D_feat=6,
                           D_inner=8, D_attn=4, droprate=0)
        rng = np.random.default_rng(173)
        with h5py.File(self.path, 'w') as h5:
            for index, n in enumerate([3, 5, 7]):
                g = h5.create_group(f'slide{index}')
                g.create_dataset('feat', data=rng.normal(size=(n,6)).astype(np.float16))
                if index != 2:
                    g.create_dataset('coords', data=np.column_stack([np.arange(n)*512,np.zeros(n)]).astype(np.int32))
                g.attrs['label'] = index % 2
        self.names = ['slide0', 'slide1', 'slide2']

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self):
        return HDF5FeatureDataset(str(self.path), self.names, self.conf)

    def assert_same_batch(self, a, b):
        for key in ['input', 'mask', 'label']:
            torch.testing.assert_close(a[key].to('cuda'), b[key].to('cuda'), rtol=0, atol=0)
        self.assertEqual(a['slide_id'], b['slide_id'])
        for x,y in zip(a['coords'], b['coords']):
            if x is None or y is None:
                self.assertIs(x,y)
            else:
                self.assertEqual(y.device.type,'cpu')
                torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_full_float16_cache_preserves_every_patch_and_value(self):
        plain, cached = self.dataset(), self.dataset()
        summary = cached.warm_cache('cuda', max_bytes=1_000_000)
        self.assertEqual(summary['cached_slides'],len(self.names))
        self.assertEqual(cached.cache_bytes,(3+5+7)*6*2)
        self.assertTrue(cached.has_device_cache)
        self.assertEqual(cached[0]['input'].dtype,torch.float16)
        before=mil_collate_fn([plain[i] for i in range(len(plain))])
        after=mil_collate_fn([cached[i] for i in range(len(cached))])
        self.assertEqual(after['input'].dtype,torch.float32)
        self.assertEqual(after['input'].device.type,'cuda')
        self.assert_same_batch(before,after)
        self.assertEqual(after['mask'].sum().item(),15)

    def test_partial_budget_keeps_uncached_bags_and_mixed_batch_exact(self):
        plain,cached=self.dataset(),self.dataset()
        summary=cached.warm_cache('cuda',max_bytes=3*6*2)
        self.assertEqual(summary['cached_slides'],1)
        self.assertEqual(cached[0]['input'].device.type,'cuda')
        self.assertEqual(cached[1]['input'].device.type,'cpu')
        self.assertEqual(len(cached),3)
        self.assert_same_batch(mil_collate_fn([plain[0],plain[1]]),
                               mil_collate_fn([cached[0],cached[1]]))

    def test_cached_bag_does_not_reopen_the_feature_file(self):
        cached=self.dataset()
        cached.warm_cache('cuda',max_bytes=3*6*2)
        renamed=self.path.with_name('moved_synthetic.h5')
        self.path.rename(renamed)
        try:
            self.assertEqual(cached[0]['input'].shape,(3,6))
            with self.assertRaises((OSError,FileNotFoundError)):
                cached[1]
        finally:
            renamed.rename(self.path)

    def test_float32_source_is_not_quantized_to_half(self):
        with h5py.File(self.path,'a') as h5:
            del h5['slide0/feat']
            values=np.linspace(.12345678,.9876543,18,dtype=np.float32).reshape(3,6)
            h5['slide0'].create_dataset('feat',data=values)
        cached=self.dataset()
        cached.warm_cache('cuda',max_bytes=1_000_000)
        self.assertEqual(cached[0]['input'].dtype,torch.float32)
        self.assertEqual(cached.cache_bytes,3*6*4+(5+7)*6*2)
        torch.testing.assert_close(cached[0]['input'].cpu(),torch.from_numpy(values),rtol=0,atol=0)

    def test_cache_clear_and_zero_budget_restore_lazy_behavior(self):
        cached=self.dataset()
        cached.warm_cache('cuda',max_bytes=1_000_000)
        cached.clear_cache()
        self.assertFalse(cached.has_device_cache)
        self.assertEqual(cached.cache_bytes,0)
        self.assertEqual(cached[0]['input'].device.type,'cpu')
        result=cached.warm_cache('cuda',max_bytes=0)
        self.assertEqual(result['cached_slides'],0)
        self.assertEqual(len(cached),3)
        with self.assertRaises(ValueError): cached.warm_cache('cuda',max_bytes=-1)

    def test_logits_gradients_and_adamw_step_match_uncached_exactly(self):
        plain,cached=self.dataset(),self.dataset()
        cached.warm_cache('cuda',max_bytes=3*6*2)
        a=mil_collate_fn([plain[0],plain[1]])
        b=mil_collate_fn([cached[0],cached[1]])
        self.assert_same_batch(a,b)
        torch.manual_seed(42)
        model_a=ABMIL(self.conf).cuda()
        model_b=copy.deepcopy(model_a)
        optimizer_a=torch.optim.AdamW(model_a.parameters(),lr=1e-4,weight_decay=1e-5)
        optimizer_b=torch.optim.AdamW(model_b.parameters(),lr=1e-4,weight_decay=1e-5)
        out=[]
        for model,optimizer,batch in [(model_a,optimizer_a,a),(model_b,optimizer_b,b)]:
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch['input'].cuda(),mask=batch['mask'].cuda())
            loss=torch.nn.functional.cross_entropy(logits,batch['label'].cuda())
            loss.backward()
            out.append((logits.detach(),[p.grad.detach().clone() for p in model.parameters()]))
            optimizer.step()
        torch.testing.assert_close(out[0][0],out[1][0],rtol=0,atol=0)
        for x,y in zip(out[0][1],out[1][1]): torch.testing.assert_close(x,y,rtol=0,atol=0)
        for x,y in zip(model_a.parameters(),model_b.parameters()):
            torch.testing.assert_close(x,y,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
