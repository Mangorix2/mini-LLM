# SmaLLM — Minimal Transformer Language Model
*IMPORTANT! THIS README IS AI GENERATED AND NOT WRITTEN BY ME*  
A compact educational implementation of a GPT-like transformer in PyTorch.
This repository shows a small, runnable training loop, tokenizer integration, and a simple generation script.

## Key Features
- Minimal Transformer architecture implemented in `src/model.py` (`SmaLLM`).
- Data loading with `tiktoken` and Hugging Face `datasets` in `src/data.py`.
- Training loop with mixed-precision support in `src/train.py` and checkpointing to `checkpoint/` and `epochs/`.
- Simple sampling-based text generation in `src/generate.py`.

## Repository Layout

- `src/` — source code
	- `config.py` — model and training hyperparameters
	- `data.py` — tokenizer and dataset utilities
	- `model.py` — Transformer implementation
	- `train.py` — training entrypoint
	- `generate.py` — simple generation example
- `data/` — example local input files (e.g. `input.txt`)
- `checkpoint/` — rolling checkpoints saved during training
- `epochs/` — periodic full-epoch checkpoints
- `requirements.txt` — Python dependencies

## Requirements

Install dependencies (recommended in a virtualenv):

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt
```

Notes:
- `torch` version in `requirements.txt` is `>=2.6.0`. For CUDA support, install the matching CUDA build from PyTorch's website.
- The project uses `tiktoken` as the tokenizer and the Hugging Face `datasets` library for example data.

## Quick Start — Training

By default `src/train.py` will:
- build a dataset via `TextDataset` (it currently uses a small slice of the `wikimedia/wikipedia` dataset),
- create the model `SmaLLM`,
- train for a small number of epochs and save checkpoints in `checkpoint/` and `epochs/`.

Run training:

```bash
python src/train.py
```

To resume training from a checkpoint:

```python
# example: in a Python session or adapt train.py to accept args
from src.train import train
train(resume_from='checkpoint/model_step500.pt', start_step=500)
```

Important training knobs are in `src/config.py` (vocab size, context length, embedding dim, number of layers/heads, dropout, ...).

## Generating Text

Use `src/generate.py` to load a checkpoint and produce sampled continuations. Example usage (already in file):

```bash
python src/generate.py
```

Edit `src/generate.py` to point to a different checkpoint (e.g. `epochs/epoch4.pt` or a `checkpoint/model_stepXXXXX.pt`) or to change the `start_text` and `max_new_tokes`.

## Using Local Data

To train on local text, place plain UTF-8 text in `data/input.txt` and modify `TextDataset` in `src/data.py` to call `load_text()` instead of the Hugging Face loader. `TextDataset` contains a comment showing where to change this.

## Checkpoints

- Checkpoints are stored in `checkpoint/` with names like `model_stepXXXXX.pt`.
- The training script keeps up to 4 recent checkpoints and stores epoch snapshots in `epochs/`.

## Notes & Tips

- The implementation is intentionally minimal for learning and experimentation — it omits production features like sharding, optimizer state checkpointing, LR schedulers, distributed training, and evaluation metrics.
- Adjust `batch_size`, `accumulation_steps`, or move training to a machine with a GPU for larger models or datasets.
- If you see tokenization mismatches, verify the tokenizer (`tiktoken.get_encoding('cl100k_base')`) matches how you prepared training data.

