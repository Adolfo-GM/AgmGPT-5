"""Coverage handoff, resume, AND-completion, and unlimited-run integration checks."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from curriculum import CoverageCurriculum
from checkpoint import read_checkpoint, tokenizer_hash

ROOT=Path(__file__).resolve().parent

def progress(hf=0, custom=7):
    return dict(huggingface=dict(windows_per_epoch=8,windows_traversed=hf),
                custom=dict(windows_per_epoch=4,windows_traversed=custom))

class CurriculumTests(unittest.TestCase):
    def test_handoff_and_both_completion_conditions(self):
        c=CoverageCurriculum(progress(),loss_window=3)
        self.assertFalse(c.advance(progress(9),100))
        self.assertTrue(c.advance(progress(10),101))
        self.assertEqual(c.source,'custom')
        self.assertEqual(c.state['custom_start_windows'],7)
        self.assertFalse(c.advance(progress(11,9),102))
        for _ in range(3): c.record_loss(1.)
        self.assertFalse(c.complete(progress(10,22)))
        self.assertTrue(c.complete(progress(10,23)))
        for _ in range(3): c.record_loss(1.9)
        self.assertFalse(c.complete(progress(10,23)))
        c.record_loss(1.8)
        self.assertTrue(c.complete(progress(10,23)))

    def test_resume_preserves_custom_baseline_and_loss_window(self):
        c=CoverageCurriculum(progress(),loss_window=3)
        c.advance(progress(10),100)
        c.record_loss(2.); c.record_loss(1.)
        restored=CoverageCurriculum(progress(10,22),saved=json.loads(json.dumps(c.state)))
        self.assertEqual(restored.state,c.state)
        restored.record_loss(1.)
        self.assertTrue(restored.complete(progress(10,23)))
        self.assertEqual(restored.report(progress(10,23))['custom_phase_percent'],400.)
        self.assertAlmostEqual(restored.lr_factor(200),.25)

    def test_unlimited_training_handoff_and_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); data=folder/'data'; data.mkdir()
            sha=tokenizer_hash(ROOT/'data/tokenizer.json')
            for split in ('train','val'):
                ids=np.asarray([10,11,12,13]*5,dtype='<u2'); ids.tofile(data/f'{split}.bin')
                np.ones(20,dtype='u1').tofile(data/f'{split}-mask.bin')
                np.save(data/f'{split}-conversations.npy',np.asarray([[i*4,4] for i in range(5)],dtype='<u8'))
            groups=[dict(name='custom',weight=.6,train=[0,2],val=[0,2]),
                    dict(name='huggingface',weight=.4,train=[2,3],val=[2,3])]
            (data/'sampling-groups.json').write_text(json.dumps(groups))
            (data/'dataset.json').write_text(json.dumps(dict(stage='sft',vocab_size=8192,tokenizer_sha256=sha)))
            config=folder/'config-smoke.json'
            config.write_text(json.dumps(dict(tokenizer_sha256=sha,model_args=dict(vocab_size=8192,
                block_size=16,n_layer=2,n_head=4,n_kv_head=2,n_embd=32,window_pattern='L',gradient_checkpointing=True))))
            out=folder/'out'; runs=folder/'runs'
            command=[sys.executable,'-B',str(ROOT/'train.py'),'--config',str(config),
                '--data-dir',str(data),'--out-dir',str(out),'--runs-dir',str(runs),
                '--hf-then-custom','--sampling','coverage','--curriculum-loss-target','100',
                '--curriculum-loss-window','3','--seconds','.001','--steps','1',
                '--batch','1','--accumulation','1','--warmup','0','--eval-every','4',
                '--eval-batches','1','--save-every','2','--device','cpu','--no-server','--exit-after-training']
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
            s=json.loads((runs/'training.json').read_text())
            self.assertEqual(s['status'],'complete'); self.assertEqual(s['stop_reason'],'curriculum_complete')
            self.assertIsNone(s['training']['max_seconds'])
            self.assertGreater(s['completed_steps'],1)
            self.assertEqual(s['coverage']['huggingface']['windows_traversed'],4)
            self.assertEqual(s['curriculum']['custom_phase_windows'],8)
            self.assertEqual(s['curriculum']['custom_phase_percent'],400.)
            self.assertTrue((out/'hf-125pct.pt').exists())
            saved=read_checkpoint(out/'latest.pt')
            self.assertEqual(saved['step'],s['completed_steps'])
            self.assertEqual(saved['curriculum']['phase'],'custom')
            self.assertIn('optimizer',saved); self.assertIn('sampler_state',saved)
            self.assertEqual(read_checkpoint(out/'agmgpt_5_it.pt')['step'],s['completed_steps'])

if __name__=='__main__': unittest.main()
