# SmolLM Training

<p align="left">
  <img src="https://img.shields.io/badge/Multi--GPU-Supported-brightgreen" alt="Multi-GPU Supported" />
  <img src="https://img.shields.io/badge/SmolLM-Training-blueviolet" alt="SmolLM Training" />
  <img src="https://img.shields.io/badge/LLM%20Training-From%20Scratch-blue" alt="LLM Training from Scratch" />
  <img src="https://img.shields.io/badge/Dataset-realnewslike-orange" alt="Dataset: realnewslike" />
</p>

This repository contains code for training a custom small language model (SmolLM) using PyTorch, with support for both single-GPU and distributed multi-GPU training via Fully Sharded Data Parallel (FSDP).

## Features
- Custom transformer-based language model (SmolLM2Config, LlamaForCausalLM)
- Efficient distributed training with FSDP
- Single-GPU and multi-GPU support
- Streaming data loading from the [AllenAI C4 dataset](https://huggingface.co/datasets/allenai/c4)
- Automatic checkpointing and logging
- Mixed-precision training (bfloat16)

## Requirements
- Python 3.8+
- PyTorch >= 2.0
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- [tiktoken](https://github.com/openai/tiktoken)
- tqdm

Install dependencies (recommended: use a virtual environment or conda):

```bash
pip install torch datasets tiktoken tqdm
```

## Usage

### Single-GPU Training

```bash
python SmolLM_training.py
```

### Multi-GPU Distributed Training

The script will automatically detect the number of available GPUs and launch distributed training using FSDP. No extra launch command is needed:

```bash
python SmolLM_training.py
```

- If more than one GPU is available, the script will use all GPUs for distributed training.
- If only one GPU is available, it will run in single-GPU mode.


#### Customizing Training
You can modify the `Args` class in the script to change batch size, sequence length, epochs, learning rate, etc.

## Checkpoints & Logging
- Model checkpoints are saved in the `saved_models/` directory.


## Dataset
- The script uses the [AllenAI C4 dataset](https://huggingface.co/datasets/allenai/c4) ("realnewslike" split) via streaming mode.
- Tokenization is performed using `tiktoken` (GPT-2 encoding).

## Notes
- FSDP is used for memory-efficient distributed training. Make sure your PyTorch version supports FSDP.
- For best performance, run on a machine with multiple GPUs and sufficient memory.
- The script is designed for research and educational purposes and may require further tuning for production use.

## Troubleshooting
- If you encounter CUDA/NCCL errors, ensure your environment variables and CUDA drivers are set up correctly.
- For distributed training, ensure all GPUs are visible and accessible.

 