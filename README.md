![AgmGPT-5 banner](banner.png)

# AgmGPT-5

A 40M LLM based on Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat), with local pretraining, instruction fine-tuning, evaluation, and a modern streaming chat interface.

## Run

```powershell
.\start.ps1
# Tiny pipeline check:
.\start.ps1 -Smoke
```

Chat: <http://127.0.0.1:8785/> · Stats: <http://127.0.0.1:8785/stats>. The server stays open after training; Ctrl+C exits.

Create a Python environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$py = 'python'
```

For NVIDIA GPU training, install a CUDA-enabled PyTorch build suitable for your system. The launcher uses the project environment when present, otherwise `python` from your active environment. Use `-Python` to select an interpreter explicitly.

## Pretrain and fine-tune

Put text in multiple `.txt` files or JSONL files with a `text` field. Preparation uses the existing frozen tokenizer and a deterministic whole-file validation split. Use independent files to reduce leakage; keep output outside the source directory.

```powershell
& $py prepare.py --source data/pretrain-source --output data/pretrain --stage pretrain
& $py train.py --stage pretrain --data-dir data/pretrain --out-dir out/base --runs-dir runs/base --steps 10000 --device cuda
& $py train.py --stage sft --init-from out/base/agmgpt_5_it.pt --steps 2000 --device cuda
```

Pretraining supervises every next token. SFT supervises assistant tokens and EOS. The training corpus is not distributed. See `data/sample.txt` for the conversation format, and prepare your own corpus before training. Sources may use marked text (`<|user|> question <|ai|> answer <|eos|>`), JSONL with `user`/`assistant` fields, or JSONL with `messages` containing user and assistant turns:

```powershell
& $py prepare.py --source data/new-instructions --output data/custom-sft --stage sft
& $py train.py --data-dir data/custom-sft --init-from out/base/agmgpt_5_it.pt --device cuda
```

`--init-from` loads local weights with the same architecture and tokenizer into a fresh optimizer. It does not import arbitrary Hugging Face or native upstream checkpoints.

## Multi-turn conversations

Keep each conversation together, with exchanges in chronological order:

```text
<conversation>
<|user|> My cat Miso stole my seat again. <|ai|> Miso clearly has strong opinions about seating arrangements. Did you at least get a corner? <|eos|>
<|user|> Nope, he took the whole chair. <|ai|> Impressive commitment. You might need to negotiate with treats, though Miso seems to hold all the cards. <|eos|>
</conversation>
```

`<conversation>` and `</conversation>` are plain dataset boundary markers. They are **not tokenizer special tokens**. `prepare.py` strips those wrappers before tokenization and keeps the existing `<|user|>`, `<|ai|>` and `<|eos|>` tokens. End each assistant turn with `<|eos|>`; the outer wrapper marks where the full conversation ends. Put one conversation per file when possible so file-level validation splitting also keeps complete conversations together.

Preparation writes `train-conversations.npy` and `val-conversations.npy`, containing each conversation's token offset and length. The trainer samples a conversation, preserves its internal turn order, and masks user text and padding from the loss. Earlier assistant replies remain in the context for later user questions. Sampling whole conversations randomly is fine; **individual turns are never shuffled**. Short conversations are padded, and conversations longer than the context window are sampled as contiguous windows within that conversation. Legacy prepared streams without a boundary index keep their original contiguous-window sampling behavior.

JSONL is also supported: each record can contain a `messages` array with alternating `user` and `assistant` roles and `content` strings. One record is one complete conversation. The sequence must start with a user and end with an assistant.

```powershell
& $py prepare.py --source data/new-instructions --output data/custom-sft --stage sft
& $py train.py --data-dir data/custom-sft --init-from out/agmgpt_5_it.pt --device cuda
```

Prepare new data in a separate output directory. Do not overwrite a dataset used by a running trainer. New samples are used only when a training run explicitly loads the newly prepared dataset. Training code, these instructions, the tokenizer and a small format sample are distributed; the full training corpus and private API generator are not.

## Training

Defaults: Muon for Transformer matrices, AdamW for other parameters, batch 1, accumulation 8, gradient clipping, warmup and cosine decay. Use `--optimizer adamw --lr 0.0003` for conventional AdamW; `--lr` only affects that option. Supported CUDA GPUs use BF16 compute. Optimizer compilation is disabled by default for Windows compatibility; set `TORCH_COMPILE_DISABLE=0` on a supported platform to enable it.

```powershell
& $py train.py --steps 1000 --seconds 3600 --device cuda
# Resume with the same data, optimizer, batch, accumulation and schedule:
& $py train.py --resume --steps 1000 --seconds 3600 --device cuda
# Fresh schedule from existing weights:
& $py train.py --init-from out/agmgpt_5_it.pt --steps 2000 --device cuda
# Continue saved optimizer state for up to an hour, ignoring the old step cap:
& $py train.py --resume --target-loss 0.5 --seconds 3600 --device cuda
# Run without a server:
& $py train.py --config config-smoke.json --steps 10 --accumulation 1 --warmup 2 --no-server --exit-after-training
```

`--steps` is the total schedule length, including completed steps on resume; `--seconds` limits the current invocation. Resume restores optimizer and RNG state and validates the recipe. Create `STOP` (`STOP-smoke` for smoke runs) to save and stop at the next step boundary, then remove it before restarting. Ctrl+C leaves the last durable checkpoint; time limits are checked between steps.

With `--target-loss`, the step cap is disabled. Training stops when the token-weighted training loss for a completed step reaches the target or less, or when the invocation's time limit is reached. Validation loss remains a separate measure. A resumed target-loss run starts from the saved learning rates and decays them over the available time. Keep the original batch, accumulation, optimizer and schedule arguments for checkpoint compatibility.

- `out/latest.pt`: resumable model, optimizer, recipe and RNG state.
- `out/agmgpt_5_it.pt`: inference snapshot, reloaded automatically by the chat server.
- `out/best.pt`: best sampled held-out validation checkpoint.
- `runs/`: training state and metrics. Smoke runs create separate `out-smoke/` and `runs-smoke/` directories.

## Evaluate, chat and test

```powershell
& $py evaluate.py --checkpoint out/agmgpt_5_it.pt --batches 20 --device cuda
& $py evaluate.py --checkpoint out/base/agmgpt_5_it.pt --data-dir data/pretrain --stage pretrain --device cuda
& $py chat.py --checkpoint out/agmgpt_5_it.pt --device cuda
& $py serve.py --port 8785
& $py -m unittest test_runtime -v
```

Evaluation reports fixed-window held-out loss and perplexity. Web chat runs inference on CPU so it can coexist with GPU training. Checkpoints validate tokenizer identity and are loaded with `weights_only=True`.

## Upstream

`vendor/nanochat/` contains the full upstream training pipeline, including tokenizer training, distributed pretraining, SFT, RL, benchmarks and serving. The upstream source is pinned to **`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`**, retrieved October 19, 2026, under the [MIT license](vendor/nanochat/LICENSE). Runtime source is unchanged; upstream development notes and assistant configuration have been omitted.

For larger runs, follow [nanochat's README](vendor/nanochat/README.md) on Linux/WSL with suitable compute. Its tokenizer, checkpoint format and cache are separate from the local adapter. Local code remains in the top-level Python files, and corpus files remain under `data/`.

## Publishing the model

`out/agmgpt_5_it.pt` is included through Git LFS. The tokenizer, `data/sample.txt`, and both web pages (`public/index.html` and `public/stats.html`) are included. Full corpus files, prepared training streams, dataset manifests, optimizer checkpoints, smoke outputs and logs stay ignored. GitHub blocks regular Git files over 100 MiB, so this roughly 128 MiB model requires LFS. Install Git LFS before committing or cloning the model.

Once training has finished, initialize a Git repository if needed, then run:

```powershell
git lfs install
git add .gitattributes .gitignore out/agmgpt_5_it.pt
# Add the source files and commit before pushing to your GitHub repository.
```

After cloning, use `git lfs pull` if the checkpoint is a pointer instead of the actual weights. The checkpoint requires the included tokenizer and local model code.
