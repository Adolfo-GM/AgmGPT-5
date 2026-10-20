"""Single-device nanochat training: pretraining and assistant-masked SFT."""
import argparse
import hashlib
import itertools
import json
import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from model import GPT, GPTConfig, size_report
from checkpoint import save_checkpoint, read_checkpoint, tokenizer_hash, replace_with_retry

ROOT = Path(__file__).resolve().parent


def dataset_fingerprint(directory, stage):
    """Identify the corpus by content, without recording filesystem paths."""
    digest = hashlib.sha256()
    for split in ('train', 'val'):
        names = [f'{split}.bin']
        if stage == 'sft':
            names.append(f'{split}-mask.bin')
            if (Path(directory) / f'{split}-conversations.npy').exists():
                names.append(f'{split}-conversations.npy')
        for name in names:
            digest.update(name.encode())
            with (Path(directory) / name).open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class TokenDataset:
    def __init__(self, directory, context, stage, vocab_size):
        self.context, self.stage = context, stage
        self.streams = {}
        self.conversations = {}
        for split in ('train', 'val'):
            tokens = np.memmap(Path(directory) / f'{split}.bin', dtype='<u2', mode='r')
            mask = np.memmap(Path(directory) / f'{split}-mask.bin', dtype='u1', mode='r') if stage == 'sft' else None
            index_path = Path(directory) / f'{split}-conversations.npy'
            if stage == 'sft' and index_path.exists():
                index = np.load(index_path, allow_pickle=False)
                if (index.ndim != 2 or index.shape[1] != 2 or not len(index) or index.dtype.kind not in 'iu'
                        or np.any(index < 0) or np.any(index[:, 1] < 2)
                        or np.any(index[:, 1] > len(tokens))
                        or np.any(index[:, 0] > len(tokens) - index[:, 1])):
                    raise ValueError('Invalid conversation boundary index')
                self.conversations[split] = index
            if (split not in self.conversations and len(tokens) <= context) or (mask is not None and len(mask) != len(tokens)):
                raise ValueError(f'{split} stream is too short or mask length differs')
            if int(tokens.max()) >= vocab_size:
                raise ValueError('Dataset token ID exceeds model vocabulary')
            self.streams[split] = tokens, mask

    def close(self):
        for tokens, mask in self.streams.values():
            tokens._mmap.close()
            if mask is not None:
                mask._mmap.close()
        self.streams.clear()
        self.conversations.clear()

    def batch(self, split, rng, count, device):
        tokens, mask = self.streams[split]
        xs, ys = [], []
        for _ in range(count):
            for _ in range(1000):
                if split in self.conversations:
                    index = self.conversations[split]
                    start, length = map(int, index[int(rng.integers(0, len(index)))])
                    offset = int(rng.integers(0, max(1, length - self.context)))
                    size = min(self.context, length - offset - 1)
                    i = start + offset
                    x = np.zeros(self.context, dtype=np.int64)
                    y = np.full(self.context, -1, dtype=np.int64)
                    x[:size] = tokens[i:i + size]
                    y[:size] = tokens[i + 1:i + size + 1]
                    y[:size][mask[i + 1:i + size + 1] == 0] = -1
                    if np.any(y != -1):
                        break
                    continue
                i = int(rng.integers(0, len(tokens) - self.context))
                x = tokens[i:i + self.context].astype(np.int64)
                y = tokens[i + 1:i + self.context + 1].astype(np.int64)
                if mask is not None:
                    y[mask[i + 1:i + self.context + 1] == 0] = -1
                if np.any(y != -1):
                    break
            else:
                raise ValueError('No supervised targets in sampled windows')
            xs.append(x)
            ys.append(y)
        return torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(ys)).to(device)


@torch.inference_mode()
def evaluate(model, dataset, batches=4, batch_size=1, device='cpu'):
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(2026)
    total = count = 0
    try:
        for _ in range(batches):
            x, y = dataset.batch('val', rng, batch_size, device)
            loss = model(x, y)[1]
            n = int(y.ne(-1).sum())
            total += float(loss) * n
            count += n
    finally:
        model.train(was_training)
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config.json')
    p.add_argument('--stage', choices=['pretrain', 'sft'], default='sft')
    p.add_argument('--data-dir')
    p.add_argument('--out-dir')
    p.add_argument('--runs-dir')
    resume = p.add_mutually_exclusive_group()
    resume.add_argument('--resume', action='store_true')
    resume.add_argument('--init-from', help='Initialize weights only, e.g. a base checkpoint for SFT')
    p.add_argument('--seconds', type=float, default=3600)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--target-loss', type=float,
                   help='Stop at or below this training loss or at --seconds; overrides the step limit')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--optimizer', choices=['muon', 'adamw'], default='muon')
    p.add_argument('--lr', type=float, default=3e-4, help='AdamW LR (Muon uses upstream per-group LRs)')
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--eval-batches', type=int, default=4)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--port', type=int, default=8785)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--no-server', action='store_true')
    p.add_argument('--exit-after-training', action='store_true')
    args = p.parse_args()
    if min(args.seconds, args.steps, args.batch, args.accumulation, args.lr,
           args.eval_every, args.eval_batches, args.save_every) <= 0 or args.warmup < 0:
        p.error('Training limits and LR must be positive; warmup must be nonnegative')
    if args.target_loss is not None and (not math.isfinite(args.target_loss) or args.target_loss <= 0):
        p.error('--target-loss must be finite and positive')
    raw = json.loads((ROOT / args.config).read_text())
    c = GPTConfig(**raw['model_args'])
    smoke = Path(args.config).stem == 'config-smoke'
    out = Path(args.out_dir) if args.out_dir else ROOT / ('out-smoke' if smoke else 'out')
    runs = Path(args.runs_dir) if args.runs_dir else ROOT / ('runs-smoke' if smoke else 'runs')
    if args.stage == 'pretrain' and not args.data_dir:
        p.error('--stage pretrain requires --data-dir with prepared text streams')
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / 'data/instruct'
    sha = tokenizer_hash(ROOT / 'data/tokenizer.json')
    if raw.get('tokenizer_sha256') != sha:
        raise ValueError('Config tokenizer mismatch')
    metadata = data_dir / 'dataset.json'
    if not metadata.exists() and data_dir.resolve() == (ROOT / 'data/instruct').resolve():
        metadata = ROOT / 'data/dataset.json'
    if not metadata.exists():
        raise ValueError('Missing dataset.json; prepare this corpus with prepare.py')
    data_info = json.loads(metadata.read_text())
    if data_info.get('tokenizer_sha256') != sha or data_info['vocab_size'] != c.vocab_size:
        raise ValueError('Dataset tokenizer or vocabulary mismatch')
    dataset = TokenDataset(data_dir, c.block_size, args.stage, c.vocab_size)
    data_id = dataset_fingerprint(data_dir, args.stage)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in this Python environment')
    if device == 'cpu':
        torch.set_num_threads(2)
    torch.manual_seed(1337)
    rng = np.random.default_rng(1337)
    model = GPT(c).to(device)
    optimizer = model.setup_optimizer(args.optimizer, args.lr, args.weight_decay)
    completed, best = 0, float('inf')
    saved = None
    if args.resume or args.init_from:
        saved = read_checkpoint(out / 'latest.pt' if args.resume else args.init_from, sha)
        if saved['model_args'] != asdict(c):
            raise ValueError('Checkpoint architecture differs from config')
        model.load_state_dict(saved['model'])
        if args.resume:
            # A resume keeps the original schedule and data stream; --steps is total steps.
            expected = dict(stage=args.stage, optimizer=args.optimizer, batch=args.batch,
                            accumulation=args.accumulation, lr=args.lr, weight_decay=args.weight_decay,
                            warmup=args.warmup, steps=args.steps, data_fingerprint=data_id)
            saved_recipe = dict(saved['recipe'])
            # Accept older local resumes once, but never write their absolute path again.
            if 'data_dir' in saved_recipe:
                old_directory = Path(saved_recipe.pop('data_dir'))
                if old_directory.resolve() != data_dir.resolve():
                    raise ValueError('Legacy resume dataset differs')
                saved_recipe['data_fingerprint'] = data_id
            if saved_recipe != expected:
                raise ValueError('Resume recipe differs; use the same schedule/data or --init-from')
            optimizer.load_state_dict(saved['optimizer'])
            completed, best = saved['step'], saved['best_val_loss']
            best = float('inf') if best is None else best
            rng.bit_generator.state = json.loads(saved['data_rng'])
            torch.set_rng_state(saved['torch_rng'])
            if device == 'cuda' and saved.get('cuda_rng'):
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
    recipe = dict(stage=args.stage, optimizer=args.optimizer, batch=args.batch,
                  accumulation=args.accumulation, lr=args.lr, weight_decay=args.weight_decay,
                  warmup=args.warmup, steps=args.steps, data_fingerprint=data_id)
    runs.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        (runs / 'metrics.jsonl').write_text('')
    state = dict(model='AgmGPT-5-It', status='training', stage=args.stage,
                 optimizer=args.optimizer, model_args=asdict(c), **size_report(c),
                 training=dict(max_seconds=args.seconds, max_iters=None if args.target_loss is not None else args.steps,
                               target_loss=args.target_loss, resume_step=completed,
                               accumulation=args.accumulation, batch=args.batch), dataset=data_info)
    atomic_json(runs / 'training.json', state)
    server = None
    if not args.no_server:
        from serve import start_server
        server = start_server(args.port, smoke=smoke, out_dir=out, runs_dir=runs)
    begin = time.monotonic()
    reason, status = ('time_limit' if args.target_loss is not None else 'step_limit'), 'complete'
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    base_lrs = [g.setdefault('initial_lr', g['lr']) for g in optimizer.param_groups]
    if args.target_loss is not None and args.resume:
        # Continue smoothly from the saved rate instead of restarting its old warmup.
        base_lrs = [g['lr'] for g in optimizer.param_groups]

    def save():
        common = dict(step=completed, best_val_loss=best if math.isfinite(best) else None,
                      stage=args.stage)
        save_checkpoint(model, out / 'latest.pt', sha, optimizer=optimizer.state_dict(),
                        recipe=recipe,
                        data_rng=json.dumps(rng.bit_generator.state), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else [], **common)
        save_checkpoint(model, out / 'agmgpt_5_it.pt', sha, **common)

    try:
        model.train()
        steps = itertools.count(completed + 1) if args.target_loss is not None else range(completed + 1, args.steps + 1)
        for step in steps:
            stop_file = ROOT / ('STOP-smoke' if smoke else 'STOP')
            if stop_file.exists():
                reason, status = 'stop_requested', 'stopped'
                break
            if time.monotonic() - begin >= args.seconds:
                reason = 'time_limit'
                break
            warmup = min(1., step / max(1, args.warmup))
            progress = max(0., (step - args.warmup) / max(1, args.steps - args.warmup))
            if args.target_loss is not None:
                progress = min(1., (time.monotonic() - begin) / args.seconds)
                if args.resume:
                    warmup = 1.
            factor = warmup * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
            for group, base in zip(optimizer.param_groups, base_lrs):
                group['lr'] = base * factor
            optimizer.zero_grad(set_to_none=True)
            # Weight accumulation by supervised tokens, not by microbatch count.
            batches = [dataset.batch('train', rng, args.batch, device) for _ in range(args.accumulation)]
            target_count = sum(int(y.ne(-1).sum()) for _, y in batches)
            total_loss = 0.
            for x, y in batches:
                loss = model(x, y)[1]
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                weight = int(y.ne(-1).sum()) / target_count
                (loss * weight).backward()
                total_loss += float(loss.detach()) * weight
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            completed = step
            row = dict(step=step, loss=total_loss, grad_norm=float(grad_norm), lr=optimizer.param_groups[0]['lr'],
                       elapsed_seconds=time.monotonic() - begin,
                       tokens=step * args.batch * args.accumulation * c.block_size,
                       gpu_peak_mb=torch.cuda.max_memory_allocated() / 2**20 if device == 'cuda' else 0)
            target_reached = args.target_loss is not None and total_loss <= args.target_loss
            if step == 1 or step % args.eval_every == 0 or (args.target_loss is None and step == args.steps) or target_reached:
                row['val_loss'] = evaluate(model, dataset, args.eval_batches, args.batch, device)
                if not math.isfinite(row['val_loss']):
                    raise FloatingPointError('Non-finite validation loss')
                if row['val_loss'] < best:
                    best = row['val_loss']
                    save_checkpoint(model, out / 'best.pt', sha, step=completed, best_val_loss=best, stage=args.stage)
            with (runs / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(row), flush=True)
            if step == 1 or step % args.save_every == 0:
                save()
            state.update(completed_steps=completed, best_val_loss=best if math.isfinite(best) else None,
                         run_elapsed_seconds=time.monotonic() - begin)
            atomic_json(runs / 'training.json', state)
            if target_reached:
                reason = 'loss_target'
                break
        save()
    except KeyboardInterrupt:
        # Discard an interrupted gradient accumulation; RNG replay starts at last save.
        reason, status = 'keyboard_interrupt', 'stopped'
    except Exception as error:
        reason, status = str(error), 'failed'
        raise
    finally:
        dataset.close()
        state.update(status=status, stop_reason=reason, completed_steps=completed,
                     run_elapsed_seconds=time.monotonic() - begin)
        atomic_json(runs / 'training.json', state)
        if server and (args.exit_after_training or status == 'failed'):
            server.shutdown()
            server.server_close()
    if server and not args.exit_after_training:
        del model, optimizer
        if device == 'cuda':
            torch.cuda.empty_cache()
        print('Chat and stats remain available. Press Ctrl+C to exit.', flush=True)
        try:
            import threading
            threading.Event().wait()
        except KeyboardInterrupt:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    main()
