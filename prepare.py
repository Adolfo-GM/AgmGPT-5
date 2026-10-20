"""Build reproducible train/validation streams with the existing frozen BPE."""
import argparse
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from checkpoint import tokenizer_hash

ROOT = Path(__file__).resolve().parent


def prepare(source, output, stage='pretrain', val_fraction=.05, seed=1337, tokenizer=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    tokenizer = Path(tokenizer or ROOT / 'data/tokenizer.json')
    if not 0 < val_fraction < .5:
        raise ValueError('Validation fraction must be between 0 and .5')
    tok = Tokenizer.from_file(str(tokenizer))
    if tok.get_vocab_size() > 65536:
        raise ValueError('uint16 streams require vocab <=65536')
    eos, ai = tok.token_to_id('<|eos|>'), tok.token_to_id('<|ai|>')
    if eos is None or ai is None:
        raise ValueError('Tokenizer must have <|eos|> and <|ai|>')
    files = sorted(p for p in source.rglob('*') if p.suffix.lower() in ('.txt', '.jsonl') and p.is_file())
    if len(files) < 2:
        raise ValueError('Provide at least two source files for a file-level held-out split')
    rng = np.random.default_rng(seed)
    val_indices = set(rng.permutation(len(files))[:max(1, round(len(files) * val_fraction))].tolist())
    output.mkdir(parents=True, exist_ok=True)
    meta = dict(stage=stage, vocab_size=tok.get_vocab_size(), tokenizer_sha256=tokenizer_hash(tokenizer),
                seed=seed, split='whole source files', files={'train': [], 'val': []})
    for split in ('train', 'val'):
        total = records = 0
        with (output / f'{split}.bin').open('wb') as token_file, (output / f'{split}-mask.bin').open('wb') as mask_file:
            for i, path in enumerate(files):
                if (i in val_indices) != (split == 'val'):
                    continue
                meta['files'][split].append(str(path.relative_to(source)))
                text = path.read_text(encoding='utf-8-sig')
                if path.suffix.lower() == '.jsonl':
                    documents = []
                    for line in text.splitlines():
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if stage == 'sft':
                            if 'messages' in record:
                                parts = []
                                for msg in record['messages']:
                                    role = msg['role']
                                    if role not in ('user', 'assistant'):
                                        raise ValueError('SFT messages support user/assistant roles')
                                    parts.append(('<|user|> ' if role == 'user' else '<|ai|> ') + msg['content'] + (' <|eos|>\n' if role == 'assistant' else ' '))
                                documents.append(''.join(parts))
                            else:
                                documents.append(f'<|user|> {record["user"]} <|ai|> {record["assistant"]} <|eos|>')
                        else:
                            documents.append(record['text'])
                elif stage == 'sft':
                    documents = [s.strip() + '<|eos|>' for s in text.split('<|eos|>') if s.strip()]
                else:
                    documents = [text]
                for document in documents:
                    ids = tok.encode(document).ids
                    if not ids:
                        continue
                    if ids[-1] != eos:
                        ids.append(eos)
                    if stage == 'sft':
                        if ai not in ids or tok.token_to_id('<|user|>') not in ids:
                            raise ValueError(f'SFT record lacks conversation markers: {path}')
                        mask, active = [], False
                        for token in ids:
                            if token == ai:
                                mask.append(0)
                                active = True
                            elif token == tok.token_to_id('<|user|>'):
                                active = False
                                mask.append(0)
                            else:
                                mask.append(int(active))
                                if token == eos:
                                    active = False
                    else:
                        mask = [1] * len(ids)
                    np.asarray(ids, dtype='<u2').tofile(token_file)
                    np.asarray(mask, dtype='u1').tofile(mask_file)
                    total += len(ids)
                    records += 1
        if not total:
            raise ValueError(f'Empty {split} split')
        meta[f'{split}_tokens'] = total
        meta[f'{split}_records'] = records
    (output / 'dataset.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--stage', choices=['pretrain', 'sft'], default='pretrain')
    p.add_argument('--val-fraction', type=float, default=.05)
    args = p.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.stage, args.val_fraction), indent=2))
