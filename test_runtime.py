"""Behavioral checks for model, masking, cache, checkpoint and local API."""
import json
import tempfile
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np
import torch
from model import GPT, GPTConfig, KVCache, size_report
from checkpoint import save_checkpoint, load_model, tokenizer_hash
from prepare import prepare
from train import TokenDataset

ROOT = Path(__file__).resolve().parent


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, **overrides):
        values = dict(vocab_size=128, block_size=32, n_layer=2, n_head=4,
                      n_kv_head=2, n_embd=32, window_pattern='L', gradient_checkpointing=True)
        return GPTConfig(**(values | overrides))

    def test_float_training_reduces_fixed_batch_loss(self):
        torch.manual_seed(1)
        model = GPT(self.config())
        self.assertTrue(all(p.is_floating_point() for p in model.parameters()))
        x = torch.arange(16).unsqueeze(0)
        y = (x + 1) % 128
        opt = model.setup_optimizer('adamw', lr=.003)
        before = float(model(x, y)[1].detach())
        for _ in range(8):
            opt.zero_grad()
            model(x, y)[1].backward()
            opt.step()
        self.assertLess(float(model(x, y)[1].detach()), before)
        self.assertEqual(size_report(model.config)['parameters'], sum(p.numel() for p in model.parameters()))

    def test_muon_updates_weights(self):
        model = GPT(self.config())
        x = torch.arange(16).unsqueeze(0)
        before = model.core.lm_head.weight.detach().clone()
        opt = model.setup_optimizer('muon')
        model(x, x)[1].backward()
        opt.step()
        self.assertFalse(torch.equal(before, model.core.lm_head.weight))

    def test_causality_and_cache(self):
        model = GPT(self.config()).eval()
        # Nonzero gates exercise smear and value-residual paths as well.
        with torch.no_grad():
            model.core.smear_lambda.fill_(.3)
        x = torch.arange(16).unsqueeze(0)
        with torch.no_grad():
            full = model(x)[0]
            altered = x.clone()
            altered[:, 8:] = 50
            changed = model(altered)[0]
            torch.testing.assert_close(full[:, :8], changed[:, :8])
            cache = KVCache(model.config, 1, 32, 'cpu')
            prefill = model(x[:, :8], kv_cache=cache)[0]
            pieces = [prefill]
            for i in range(8, 16):
                pieces.append(model(x[:, i:i+1], kv_cache=cache)[0])
            torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=2e-5, rtol=2e-4)

    def test_checkpoint_roundtrip_and_tokenizer_guard(self):
        model = GPT(self.config()).eval()
        x = torch.arange(16).unsqueeze(0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'agmgpt_5_it.pt'
            save_checkpoint(model, path, 'example-hash')
            loaded = load_model(path, expected_hash='example-hash')
            torch.testing.assert_close(model(x)[0], loaded(x)[0])
            with self.assertRaisesRegex(ValueError, 'tokenizer'):
                load_model(path, expected_hash='wrong')

    def test_data_preparation_masks_and_file_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            source.mkdir()
            for i in range(4):
                (source / f'{i}.txt').write_text('<|user|> hello <|ai|> world <|eos|>\n' * 30)
            output = Path(tmp) / 'sft'
            info = prepare(source, output, stage='sft')
            self.assertFalse(set(info['files']['train']) & set(info['files']['val']))
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(ROOT / 'data/tokenizer.json'))
            ids = np.fromfile(output / 'train.bin', dtype='<u2')
            mask = np.fromfile(output / 'train-mask.bin', dtype='u1')
            self.assertTrue(np.all(mask[ids == tok.token_to_id('<|ai|>')] == 0))
            self.assertTrue(np.all(mask[ids == tok.token_to_id('<|user|>')] == 0))
            self.assertTrue(np.all(mask[ids == tok.token_to_id('<|eos|>')] == 1))
            ds = TokenDataset(output, 16, 'sft', 8192)
            x, y = ds.batch('train', np.random.default_rng(1), 2, 'cpu')
            self.assertEqual(x.shape, y.shape)
            self.assertTrue(bool(y.eq(-1).any()))
            ds.close()
            base = Path(tmp) / 'base'
            prepare(source, base, stage='pretrain')
            self.assertTrue(np.all(np.fromfile(base / 'train-mask.bin', dtype='u1') == 1))

    def test_generation_rollover_greedy_and_validation(self):
        model = GPT(self.config(block_size=8)).eval()
        x = torch.arange(8).unsqueeze(0)
        self.assertEqual(len(list(model.generate_stream(x, count=12, temperature=0))), 12)
        with self.assertRaises(ValueError):
            model(x, torch.full_like(x, -1))
        with self.assertRaises(ValueError):
            GPTConfig(n_head=7)

    def test_resume_matches_uninterrupted_training(self):
        import train
        from checkpoint import read_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            def run(name, resume=False):
                argv = ['train.py', '--config', 'config-smoke.json', '--steps', '4',
                        '--warmup', '2', '--batch', '1', '--accumulation', '2',
                        '--eval-batches', '1', '--save-every', '2', '--eval-every', '2',
                        '--no-server', '--device', 'cpu', '--out-dir', str(Path(tmp) / name),
                        '--runs-dir', str(Path(tmp) / (name + '-runs'))]
                if resume:
                    argv.append('--resume')
                with patch('sys.argv', argv):
                    train.main()

            run('continuous')
            real_save = train.save_checkpoint

            def interrupt_after_durable_save(model, path, sha, **state):
                real_save(model, path, sha, **state)
                if Path(path).name == 'latest.pt' and state.get('step') == 2:
                    raise KeyboardInterrupt()

            with patch.object(train, 'save_checkpoint', interrupt_after_durable_save):
                run('resumed')
            run('resumed', resume=True)
            a = read_checkpoint(Path(tmp) / 'continuous/latest.pt')
            b = read_checkpoint(Path(tmp) / 'resumed/latest.pt')
            self.assertEqual(a['step'], b['step'])
            self.assertEqual(a['data_rng'], b['data_rng'])
            self.assertNotIn('data_dir', b['recipe'])
            self.assertIn('data_fingerprint', b['recipe'])
            exported = read_checkpoint(Path(tmp) / 'resumed/agmgpt_5_it.pt')
            self.assertNotIn('recipe', exported)
            self.assertNotIn(str(ROOT), repr({k: v for k, v in exported.items() if k != 'model'}))
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)

    def test_target_loss_or_time_overrides_step_limit(self):
        import train
        with tempfile.TemporaryDirectory() as tmp:
            for name, target, seconds, expected in (
                ('loss', '20', '60', 'loss_target'),
                ('time', '0.000000001', '0.7', 'time_limit'),
            ):
                out, runs = Path(tmp) / name, Path(tmp) / (name + '-runs')
                argv = ['train.py', '--config', 'config-smoke.json', '--steps', '1',
                        '--warmup', '0', '--batch', '1', '--accumulation', '1',
                        '--eval-batches', '1', '--no-server', '--device', 'cpu',
                        '--target-loss', target, '--seconds', seconds,
                        '--out-dir', str(out), '--runs-dir', str(runs)]
                tick = [0.]
                def clock():
                    tick[0] += .05
                    return tick[0]
                with patch('sys.argv', argv), patch.object(train.time, 'monotonic', clock):
                    train.main()
                state = json.loads((runs / 'training.json').read_text())
                self.assertEqual(state['stop_reason'], expected)
                self.assertIsNone(state['training']['max_iters'])
                self.assertTrue((out / 'agmgpt_5_it.pt').exists())
                if name == 'time':
                    self.assertGreater(state['completed_steps'], 1)

    def test_server_missing_checkpoint_and_smoke_stream(self):
        from serve import start_server
        with tempfile.TemporaryDirectory() as tmp:
            out, runs = Path(tmp) / 'out', Path(tmp) / 'runs'
            server = start_server(0, smoke=True, out_dir=out, runs_dir=runs)
            url = f'http://127.0.0.1:{server.server_port}'
            try:
                with urllib.request.urlopen(url + '/api/status') as r:
                    self.assertFalse(json.load(r)['available'])
                req = urllib.request.Request(url + '/api/chat', json.dumps(dict(message='hello')).encode(), {'Content-Type': 'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as err:
                    urllib.request.urlopen(req)
                self.assertEqual(err.exception.code, 503)
                model = GPT(self.config(vocab_size=8192))
                save_checkpoint(model, out / 'agmgpt_5_it.pt', tokenizer_hash(ROOT / 'data/tokenizer.json'))
                with urllib.request.urlopen(req, timeout=60) as r:
                    events = [json.loads(line) for line in r]
                self.assertEqual(events[-1]['type'], 'done')
                self.assertEqual(events[-1]['checkpoint'], 'agmgpt_5_it.pt')
            finally:
                server.shutdown()
                server.server_close()

    def test_pretrain_then_sft_initialization(self):
        import train
        from checkpoint import read_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            for i in range(2):
                (source / f'{i}.txt').write_text('Language models predict the next token. ' * 100)
            corpus = root / 'corpus'
            prepare(source, corpus, 'pretrain')
            common = ['train.py', '--config', 'config-smoke.json', '--steps', '2',
                      '--warmup', '1', '--accumulation', '1', '--eval-batches', '1',
                      '--device', 'cpu', '--no-server']
            with patch('sys.argv', common + ['--stage', 'pretrain', '--data-dir', str(corpus),
                       '--out-dir', str(root / 'base'), '--runs-dir', str(root / 'base-runs')]):
                train.main()
            with patch('sys.argv', common + ['--stage', 'sft', '--init-from', str(root / 'base/agmgpt_5_it.pt'),
                       '--out-dir', str(root / 'sft'), '--runs-dir', str(root / 'sft-runs')]):
                train.main()
            self.assertEqual(read_checkpoint(root / 'base/agmgpt_5_it.pt')['stage'], 'pretrain')
            self.assertEqual(read_checkpoint(root / 'sft/agmgpt_5_it.pt')['stage'], 'sft')


if __name__ == '__main__':
    unittest.main()
