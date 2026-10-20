"""Atomic, versioned floating point checkpoints. Packed legacy files are untouched."""
import hashlib
import time
import uuid
from dataclasses import asdict
from pathlib import Path
import torch
from model import GPT, GPTConfig
FORMAT = 'agmgpt-nanochat-v1'


def replace_with_retry(source, destination):
    """Allow transient Windows reader/sync locks to clear before replacing a file."""
    for attempt in range(20):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(.02 * 2 ** attempt, .5))


def tokenizer_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        torch.save(payload, tmp)
        replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_checkpoint(path, expected_hash=None):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format') != FORMAT:
        raise ValueError('Unsupported checkpoint; legacy binary weights cannot be resumed')
    if expected_hash and saved.get('tokenizer_sha256') != expected_hash:
        raise ValueError('Checkpoint tokenizer mismatch')
    return saved


def save_checkpoint(model, path, tokenizer_sha256, **state):
    atomic_save(path, dict(format=FORMAT, model_args=asdict(model.config),
                model=model.state_dict(), tokenizer_sha256=tokenizer_sha256, **state))


def load_model(path, device='cpu', expected_hash=None):
    saved = read_checkpoint(path, expected_hash)
    model = GPT(GPTConfig(**saved['model_args']))
    model.load_state_dict(saved['model'])
    return model.to(device).eval()
