"""Terminal chat using the same frozen tokenizer and prompt format as the UI."""
import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from checkpoint import load_model, tokenizer_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='out/agmgpt_5_it.pt')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--prompt')
    p.add_argument('--tokens', type=int, default=160)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(2)
    model = load_model(root / args.checkpoint, args.device, tokenizer_hash(root / 'data/tokenizer.json'))
    tok = Tokenizer.from_file(str(root / 'data/tokenizer.json'))
    history = ''
    while True:
        try:
            message = args.prompt if args.prompt is not None else input('You: ')
        except (EOFError, KeyboardInterrupt):
            break
        if message.strip().lower() in ('/quit', '/exit'):
            break
        prompt = history + f'<|user|> {message} <|ai|>'
        x = torch.tensor([tok.encode(prompt).ids], device=args.device)
        ids, previous = [], ''
        print('AGM: ', end='', flush=True)
        for token in model.generate_stream(x, args.tokens, stop_ids={tok.token_to_id('<|eos|>'), tok.token_to_id('<|user|>')}):
            ids.append(token)
            reply = tok.decode(ids)
            if not reply.endswith('\ufffd'):
                print(reply[len(previous):], end='', flush=True)
                previous = reply
        print()
        history = prompt + ' ' + tok.decode(ids) + ' <|eos|>\n'
        if args.prompt is not None:
            break


if __name__ == '__main__':
    main()
