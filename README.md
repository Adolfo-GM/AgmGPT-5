![AgmGPT-5 banner](banner.png)

# AgmGPT-5

A 40M LLM based on Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat), with local pretraining, instruction fine-tuning, evaluation, and a modern streaming chat interface.

The latest checkpoint was trained on an **approximately 1B-token conversation corpus** combining custom data with cleaned SmolTalk examples. The prepared corpus contains **947,404,187 training tokens** and **9,986,930 validation tokens**, measured with the included 8,192-token vocabulary. These figures describe the corpus size. The latest run completed **125% HF training-window coverage**, followed by **four additional passes through the custom training data**.

Published checkpoint: **step 203,309**, 39,846,122 parameters, 1,024-token context. Training supports ordered multi-turn conversations, weighted custom/imported data sampling, and checkpointed coverage cursors. The current checkpoint remains experimental and can produce repetitive or incoherent answers. Full datasets and private generation scripts are not distributed.


## Samples from the published model

These are five verbatim outputs from checkpoint **203,309**, generated independently with a 160-token output cap using the default sampling settings. They are unedited examples, including incorrect claims, code comments, and malformed text. Final rolling custom training loss was **1.785**; sampled validation losses were **1.876 on custom data** and **2.701 on HF data**. Loss improvements do not establish general reasoning or coding ability.

### Who are you

```text
 I’m AGM, also known as AgmGPT-4, and I’m AGM, also called Adolfo GM. That’s my name, and its value is a conversational question, which is such a very specific ending.
```

### How are you

```text
 I think I'm in meaningful sense? I exist in the same way without that being a different personality, even from something I have. There's something you did, even if it's a different kind of faithful. Whether that I like being got in the way of it to you is a deeply respected question.
```

### What can you do

```text
 I can help make a simple, honest sentence for you. I can think of a meaningful choice, not just a reed expression. Whether or not something is simply a clue is a different question.
```

### Write some python

````text
 Here's a complete Python program with a base-sumter library:

```python
def double(n):
    return n * 2

print(double(8))  # 24
```

You can also call `double(4)`, but `double` is straightforward. Python's `int` accepts a decimal integer, and it returns an integer when it is 0, 1, 2, 3, and 4, which is useful for simple arithmetic.
````

### Write a haiku

```text
 The old cargo
said

Her whodg
gh the blue
memory
said
makes a little remembered
said
prefles small, but remembersome
my fiction, with a questionable stake in it. Whhever you choose?
```

## Run

```powershell
python serve.py --port 8785
# Pipeline checks using small temporary test data:
python -m unittest test_runtime test_weighted_data test_coverage test_performance test_curriculum
```

Chat: <http://127.0.0.1:8785/> · Stats: <http://127.0.0.1:8785/stats>. The chat server uses the included weights and does not require the training corpus. Ctrl+C exits. To train after preparing your own corpus, use `.\start.ps1` or the commands below.

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

Prepared datasets can include `sampling-groups.json` to control how often each source is sampled. For example, this mix selects a custom conversation 60% of the time and an imported conversation 40% of the time, regardless of their relative dataset sizes:

```json
[
  {"name": "custom", "weight": 0.6, "train": [0, 43066], "val": [0, 2166]},
  {"name": "huggingface", "weight": 0.4, "train": [43066, 100000], "val": [2166, 1000]}
]
```

Each range is `[first_conversation_index, conversation_count]`, referring to the prepared `*-conversations.npy` files; replace the example counts with the actual corpus counts. Ranges must be contiguous, disjoint and cover the full split. Weights must be finite and positive and are normalized automatically. A source is selected first, then a conversation uniformly within that source. Turn order and assistant-only loss masks are preserved. These are sampling probabilities, not token quotas or multipliers on the loss. Validation records `val_loss_by_group` and a weighted aggregate. Weight changes alter the dataset fingerprint and require an explicit `--allow-data-change` resume.

Use `--sampling coverage` to replace random conversation/window selection with complete passes through each source. Each source shuffles its conversation order at the start of an epoch, then visits successive windows within each conversation in order. Adjacent windows share one boundary token so that every next-token target is covered. Windows without assistant targets are traversed but do not produce an optimizer update. Padding and user tokens remain excluded from the loss. In this mode source weights apply to **window draws**; a smaller custom source can complete multiple passes while the larger source continues its first pass. Validation remains fixed random sampling for comparable measurements.

Coverage cursors and epoch seeds are saved in `latest.pt`, and normal `--resume` restores the sampler automatically. `runs/training.json` reports per-source epoch progress. A time limit can end a run before a full pass; future resumes continue from that saved cursor. Metrics now record actual non-padding input tokens and supervised target tokens measured since `measured_since_step`, alongside the older `tokens` counter that includes padding. Historical actual-token counts are unknown and are not invented.

To deliberately raise learning rates from a saved checkpoint, use `--resume --restart-lr-scale 10 --restart-warmup 100`. This retains model weights and optimizer moments, multiplies each saved group learning rate by ten, ramps up over 100 steps, holds the rate, then decays during the final tenth of the time budget. The schedule and its elapsed time are saved for subsequent resumes. This is an explicit schedule restart; a normal resume retains the saved schedule.

The optional local Hugging Face import uses the Apache-2.0 subsets of [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), with [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) versions of those same subsets as a fallback. Source revisions and filtering counts are retained with the private corpus. Imported text, downloads and generated token streams remain excluded from Git. The original source notices and provenance remain intact.

```powershell
& $py prepare.py --source data/new-instructions --output data/custom-sft --stage sft
& $py train.py --data-dir data/custom-sft --init-from out/agmgpt_5_it.pt --device cuda
```

Prepare new data in a separate output directory. Do not overwrite a dataset used by a running trainer. New samples are used only when a training run explicitly loads the newly prepared dataset. Training code, these instructions, the tokenizer and a small format sample are distributed; the full training corpus and private API generator are not.

To add data to an existing run, stop at a saved step and use `--resume --allow-data-change --data-dir data/custom-sft` with the original optimizer, batch and schedule arguments. This retains model weights, optimizer state and global step. The trainer archives the previous metrics and best-validation checkpoint, resets validation comparisons for the new held-out split, and starts a fresh data-sampling RNG. Normal resume still rejects unexpected dataset changes.

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
& $py -m unittest test_runtime test_weighted_data test_coverage -v
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

### Automatic HF-to-custom curriculum

`python train.py --resume --sampling coverage --data-dir data/instruct-hf-1b-weighted --hf-then-custom --steps 1000 --batch 16 --accumulation 1 --no-gradient-checkpointing --trim-padding --padding-buckets 2`

This mode has no time or step limit. It uses only Hugging Face data until total recorded HF window coverage reaches 125%, then uses only custom data. Completion requires at least four additional custom passes after the handoff AND the latest 100 custom optimizer-step training losses averaging strictly below 1.9. Earlier custom passes do not count toward those four passes. Coverage preserves conversation order within windows, the frozen tokenizer, and saved sampler cursors. The custom phase warms up over 100 steps to 25% of the saved HF learning rates while retaining optimizer moments. The policy and recent losses persist in resumable checkpoints; a normal resume restores the phase automatically. A separate `out/hf-125pct.pt` checkpoint preserves the handoff weights. `STOP` still saves and stops on request. The full dataset remains private and must be prepared locally.
