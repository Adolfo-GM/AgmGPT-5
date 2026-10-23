import unittest
from unittest.mock import patch
import torch
from model import GPT, GPTConfig, CheckpointBlock
from train import trimmed_batches

class PerformanceTests(unittest.TestCase):
    @patch('nanochat.gpt.COMPUTE_DTYPE', torch.float32)
    def test_cropping_preserves_loss_and_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)
        m=GPT(GPTConfig(vocab_size=64,block_size=16,n_layer=2,n_head=4,n_kv_head=2,n_embd=32,window_pattern='SL'))
        x=torch.randint(0,64,(4,16)); y=torch.randint(0,64,(4,16))
        for row,end in enumerate([3,7,12,16]): y[row,end:]=-1
        loss=m(x,y)[1]; loss.backward()
        grads=[p.grad.clone() if p.grad is not None else None for p in m.parameters()]
        m.zero_grad(set_to_none=True)
        total=0.; targets=int(y.ne(-1).sum())
        for xx,yy in trimmed_batches(x,y,2):
            part=m(xx,yy)[1]*int(yy.ne(-1).sum())/targets
            part.backward(); total+=float(part.detach())
        self.assertAlmostEqual(float(loss.detach()),total,places=5)
        for p,g in zip(m.parameters(),grads):
            if g is not None: torch.testing.assert_close(p.grad,g,atol=2e-6,rtol=2e-4)
        self.assertEqual(sum(int(yy.ne(-1).sum()) for _,yy in trimmed_batches(x,y,2)),targets)

    def test_checkpoint_toggle_preserves_outputs_and_keys(self):
        torch.set_num_threads(2)
        m=GPT(GPTConfig(vocab_size=64,block_size=16,n_layer=2,n_head=4,n_kv_head=2,n_embd=32))
        x=torch.randint(0,64,(2,16)); y=x.clone()
        keys=list(m.state_dict()); before=m(x,y)[1].detach()
        for b in m.core.transformer.h:
            self.assertIsInstance(b,CheckpointBlock)
            b.enabled=False
        torch.testing.assert_close(before,m(x,y)[1].detach())
        self.assertEqual(keys,list(m.state_dict()))

if __name__=='__main__': unittest.main()
