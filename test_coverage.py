"""Checks for complete window coverage, ordered turns, and exact sampler resumption."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from sampling import CoverageSampler, restarted_lr_factor
from train import TokenDataset


class CoverageTests(unittest.TestCase):
    def test_every_target_position_once_per_epoch(self):
        index=np.asarray([[0,3],[3,11],[14,6]],dtype='<u8')
        sampler=CoverageSampler(index,4)
        seen=[];previous={}
        for _ in range(sampler.total_windows[0]):
            start,size=sampler.next()
            record=next(i for i,(a,n) in enumerate(index) if int(a)<=start<int(a+n))
            self.assertGreater(start,previous.get(record,-1));previous[record]=start
            seen.extend(range(start+1,start+size+1))
        expected=[i for start,length in index for i in range(int(start)+1,int(start+length))]
        self.assertEqual(sorted(seen),expected)
        self.assertEqual(len(seen),len(set(seen)))
        self.assertEqual(sampler.progress()['all']['completed_epochs'],1)

    def test_compact_resume_matches_uninterrupted_windows(self):
        index=np.asarray([[0,17],[17,9],[26,8]],dtype='<u8')
        first=CoverageSampler(index,4)
        for _ in range(3):first.next()
        saved=json.loads(json.dumps(first.state_dict()))
        resumed=CoverageSampler(index,4);resumed.load_state_dict(saved)
        self.assertEqual([first.next() for _ in range(30)],[resumed.next() for _ in range(30)])

    def test_dataset_covers_all_supervised_tokens_and_masks_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);ids=np.arange(1,21,dtype='<u2')
            mask=np.ones(20,dtype='u1');mask[:3]=0;mask[10:13]=0
            for split in ('train','val'):
                ids.tofile(folder/f'{split}.bin');mask.tofile(folder/f'{split}-mask.bin')
                np.save(folder/f'{split}-conversations.npy',np.asarray([[0,10],[10,10]],dtype='<u8'))
            dataset=TokenDataset(folder,4,'sft',32,sampling='coverage')
            try:
                seen=[];rng=np.random.default_rng(2)
                while dataset.coverage.states[0]['epoch']==0:
                    _,y=dataset.batch('train',rng,1,'cpu')
                    seen.extend(y[y!=-1].tolist())
                self.assertEqual(sorted(seen),ids[mask.astype(bool)].tolist())
            finally:dataset.close()

    def test_sources_have_independent_cursors(self):
        index=np.asarray([[i*3,3] for i in range(5)],dtype='<u8')
        groups=[dict(name='custom',train=[0,2]),dict(name='imported',train=[2,3])]
        sampler=CoverageSampler(index,4,groups)
        for _ in range(5):sampler.next(0)
        self.assertEqual(sampler.states[0]['epoch'],2)
        self.assertEqual(sampler.states[1]['windows'],0)
        starts=[sampler.next(1)[0] for _ in range(3)]
        self.assertEqual(sorted(starts),[6,9,12])

    def test_lr_warms_up_holds_and_decays_at_end(self):
        self.assertLess(restarted_lr_factor(101,100,100,1,1000),.12)
        self.assertEqual(restarted_lr_factor(200,100,100,500,1000),1.)
        self.assertAlmostEqual(restarted_lr_factor(300,100,100,1000,1000),.1)


if __name__=='__main__':unittest.main()
