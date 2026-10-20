"""Evaluate a local floating-point checkpoint on fixed held-out windows."""
import argparse
import json
import math
from pathlib import Path
import torch
from checkpoint import load_model, tokenizer_hash
from train import TokenDataset, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='out/agmgpt_5_it.pt')
    p.add_argument('--data-dir', default='data/instruct')
    p.add_argument('--stage', choices=['pretrain', 'sft'], default='sft')
    p.add_argument('--batches', type=int, default=20)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    if args.batches < 1:
        p.error('--batches must be positive')
    root = Path(__file__).resolve().parent
    torch.set_num_threads(2)
    model = load_model(root / args.checkpoint, args.device, tokenizer_hash(root / 'data/tokenizer.json'))
    data = TokenDataset(root / args.data_dir, model.config.block_size, args.stage, model.config.vocab_size)
    try:
        loss = evaluate(model, data, args.batches, 1, args.device)
    finally:
        data.close()
    print(json.dumps(dict(loss=loss, perplexity=math.exp(min(50., loss)), batches=args.batches)))


if __name__ == '__main__':
    main()
