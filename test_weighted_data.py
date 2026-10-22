"""Behavioral checks for source-weighted sampling and resume identity."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from train import TokenDataset, dataset_fingerprint


class WeightedDataTests(unittest.TestCase):
    def corpus(self, folder, weight=.6):
        # Two custom examples and 100 imported examples: weights must override pool size.
        for split in ('train', 'val'):
            ids = np.asarray([10, 11, 12] * 2 + [20, 21, 22] * 100, dtype='<u2')
            ids.tofile(folder / f'{split}.bin')
            np.ones(len(ids), dtype='u1').tofile(folder / f'{split}-mask.bin')
            np.save(folder / f'{split}-conversations.npy', np.asarray([[i*3, 3] for i in range(102)], dtype='<u8'))
        groups = [dict(name='custom', weight=weight, train=[0, 2], val=[0, 2]),
                  dict(name='huggingface', weight=1-weight, train=[2, 100], val=[2, 100])]
        (folder / 'sampling-groups.json').write_text(json.dumps(groups))

    def test_weights_override_relative_source_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.corpus(folder)
            dataset = TokenDataset(folder, 4, 'sft', 32)
            try:
                x, _ = dataset.batch('train', np.random.default_rng(77), 2000, 'cpu')
                rate = float(x[:, 0].eq(10).float().mean())
                self.assertGreater(rate, .56)
                self.assertLess(rate, .64)
                for name, first in [('custom', 10), ('huggingface', 20)]:
                    x, y = dataset.batch('val', np.random.default_rng(1), 20, 'cpu', group=name)
                    self.assertTrue(bool(x[:, 0].eq(first).all()))
                    self.assertTrue(bool(y[:, 2:].eq(-1).all()))
            finally:
                dataset.close()

    def test_weight_changes_change_dataset_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.corpus(folder)
            before = dataset_fingerprint(folder, 'sft')
            self.corpus(folder, weight=.8)
            self.assertNotEqual(before, dataset_fingerprint(folder, 'sft'))

    def test_invalid_ranges_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.corpus(folder)
            path = folder / 'sampling-groups.json'
            groups = json.loads(path.read_text())
            groups[1]['train'][0] = 1
            path.write_text(json.dumps(groups))
            with self.assertRaisesRegex(ValueError, 'range'):
                TokenDataset(folder, 4, 'sft', 32)


if __name__ == '__main__':
    unittest.main()
