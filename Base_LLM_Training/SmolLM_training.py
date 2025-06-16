import os
import math
import time
import logging
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Using the latest cached version of the dataset since allenai/c4 couldn't be found")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_TIMEOUT", "3600000")  # seconds
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"torch\.cuda\.amp"
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
# --- Imports for FSDP ---
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    FullStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)
# --- End FSDP Imports ---
# from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add NCCL and CUDA debugging/stability environment variables
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training-3b.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class Args:
    batch_size = 2
    seq_len = 512  
    epochs = 100
    steps_per_epoch = 1000
    warmup_steps = 200  
    max_lr = 1e-4 


@dataclass
class SmolLM2Config:
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    vocab_size: int = 50257
    rope_theta: float = 10000.0
    rope_interleaved: bool = False
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: Optional[int] = None
    tie_word_embeddings: bool = True
    use_cache: bool = True
    
    
def save_model(model, optimizer, loss, output_dir='saved_models'):
    from datetime import datetime
    
    # This config ensures that the full state dict is gathered on the CPU on rank 0,
    # preventing GPU OOM errors during checkpointing large models.
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        # This will get the model state dict, offloaded to CPU on rank 0.
        # For other ranks, it will be an empty dict.
        model_state = model.state_dict()
        
        # The FSDP.optim_state_dict is the hook to get the sharded optimizer state
        # and the context manager handles the gathering.
        optim_state = FSDP.optim_state_dict(model, optimizer)

    if dist.get_rank() == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"smollm2-3b_{timestamp}.pt"
        filepath = os.path.join(output_dir, filename)

        save_dict = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optim_state,
            'config': model.module.config,
            'loss': loss
        }
        torch.save(save_dict, filepath)
        logging.info(f"Model saved to {filepath}")

def apply_rotary_pos_emb(x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
    head_dim = x.shape[-1]
    x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
    sin, cos = rotary_emb[..., :head_dim // 2], rotary_emb[..., head_dim // 2:]
    rotated_x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated_x

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len <= 0:
            logging.warning(f"Invalid seq_len: {seq_len}, setting to 1")
            seq_len = 1
        if seq_len > self.max_position_embeddings:
            logging.warning(f"seq_len {seq_len} exceeds max_position_embeddings {self.max_position_embeddings}, truncating")
            seq_len = self.max_position_embeddings

        positions = torch.arange(seq_len, device=device)
        sincos = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat((sincos.sin(), sincos.cos()), dim=-1)
        # Rearranged so that seq_len is in dimension 2
        return emb[None, None, :, :]

    def apply_rotary_emb(self, x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
        # Wrapper to use the standalone function
        return apply_rotary_pos_emb(x, rotary_emb)
    

class DeepSeekExpertLayer(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))



class DeepSeekMoE(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        # Add default values for missing config attributes
        self.num_experts = getattr(config, 'num_experts', 8)
        self.num_shared_experts = getattr(config, 'num_shared_experts', 2)
        self.num_routed_experts = self.num_experts - self.num_shared_experts
        self.top_k = getattr(config, 'top_k', 2)
        self.hidden_size = config.hidden_size
        
        self.shared_experts = nn.ModuleList([
            DeepSeekExpertLayer(config)
            for _ in range(self.num_shared_experts)
        ])
        
        self.routed_experts = nn.ModuleList([
            DeepSeekExpertLayer(config)
            for _ in range(self.num_routed_experts)
        ])
        
        self.router = nn.Linear(self.hidden_size, self.num_routed_experts, bias=False)
        self.routing_bias = nn.Parameter(torch.zeros(self.num_routed_experts))
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Shared experts
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output += expert(x)
        if self.num_shared_experts > 0:
            shared_output = shared_output / self.num_shared_experts
        
        # Routing
        routing_logits = self.router(x) + self.routing_bias
        routing_probs = F.softmax(routing_logits, dim=-1)  # Use softmax instead of sigmoid
        scores, indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize scores properly
        scores = scores / scores.sum(dim=-1, keepdim=True)
        
        # More efficient expert routing
        combined_output = torch.zeros_like(x)
        flat_x = x.view(-1, hidden_size)
        flat_indices = indices.view(-1, self.top_k)
        flat_scores = scores.view(-1, self.top_k)
        
        for k in range(self.top_k):
            for expert_idx in range(self.num_routed_experts):
                mask = (flat_indices[:, k] == expert_idx)
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_output = self.routed_experts[expert_idx](expert_input)
                    
                    # Create output tensor for this expert
                    expert_contribution = torch.zeros_like(flat_x)
                    expert_contribution[mask] = expert_output * flat_scores[mask, k].unsqueeze(-1)
                    combined_output += expert_contribution.view(batch_size, seq_len, hidden_size)
        
        return shared_output + combined_output

    def update_bias_terms(self, expert_load):
        if expert_load.numel() != self.num_routed_experts:
            return
        target_load = 1.0 / self.num_routed_experts
        load_diff = expert_load - target_load
        update_rate = 0.1 * torch.abs(load_diff)
        self.routing_bias.data -= update_rate * load_diff

class LlamaMLP(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        # Pass the given config instance instead of the class itself
        self.moe = DeepSeekMoE(config)
        
    def forward(self, x):
        return self.moe(x)

def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout=0.0, is_causal=False):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled dot product

    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, -1e9)  # Mask out invalid positions

    if is_causal:
        mask = torch.tril(torch.ones_like(scores)).to(scores.device)
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_probs = F.softmax(scores, dim=-1)  # Apply softmax

    if dropout > 0.0:
        attn_probs = F.dropout(attn_probs, p=dropout)

    attn_output = torch.matmul(attn_probs, v)  # Weighted sum of values
    return attn_output

class LlamaAttention(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Add missing config attributes with defaults
        self.compression_ratio = getattr(config, 'compression_ratio', 8)
        self.latent_dim = self.hidden_size // self.compression_ratio
        
        self.kv_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.q_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        
        # For keys and queries, output dimension: num_heads * (head_dim // 2)
        self.k_proj_u = nn.Linear(self.latent_dim, self.num_heads * (self.head_dim // 2), bias=False)
        self.q_proj_u = nn.Linear(self.latent_dim, self.num_heads * (self.head_dim // 2), bias=False)
        
        # For values, output dimension: num_heads * head_dim
        self.v_proj_u = nn.Linear(self.latent_dim, self.num_heads * self.head_dim, bias=False)
        
        self.rope_k = nn.Linear(self.hidden_size, self.hidden_size // 2, bias=False)
        self.rope_q = nn.Linear(self.latent_dim, self.hidden_size // 2, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim // 2)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                positional_ids: Optional[torch.Tensor] = None,
                pass_key_value: Optional[Tuple[torch.Tensor]] = None,
                rotary_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        kv_d = self.kv_proj_d(hidden_states)
        q_d = self.q_proj_d(hidden_states)
        
        k_proj_2 = self.k_proj_u(kv_d)
        q_proj_2 = self.q_proj_u(q_d)
        v = self.v_proj_u(kv_d)
        
        k_rope_2 = self.rope_k(hidden_states)
        q_rope_2 = self.rope_q(q_d)
        
        # Reshape projections to split heads
        k_proj_2 = k_proj_2.view(batch_size, seq_len, self.num_heads, self.head_dim // 2)
        q_proj_2 = q_proj_2.view(batch_size, seq_len, self.num_heads, self.head_dim // 2)
        k_rope_2 = k_rope_2.view(batch_size, seq_len, self.num_heads, self.head_dim // 2)
        q_rope_2 = q_rope_2.view(batch_size, seq_len, self.num_heads, self.head_dim // 2)
        
        # Transpose to (batch, num_heads, seq_len, head_dim//2) for rotary application
        k_rope_2 = k_rope_2.transpose(1, 2)
        q_rope_2 = q_rope_2.transpose(1, 2)
        
        # Apply rotary embeddings using the 'rotary_emb' passed from the model
        k_rope_2 = self.rotary_emb.apply_rotary_emb(k_rope_2, rotary_emb)
        q_rope_2 = self.rotary_emb.apply_rotary_emb(q_rope_2, rotary_emb)
        
        # Transpose back to (batch, seq_len, num_heads, head_dim//2)
        k_rope_2 = k_rope_2.transpose(1, 2)
        q_rope_2 = q_rope_2.transpose(1, 2)
        
        # Concatenate the projections
        k = torch.cat([k_proj_2, k_rope_2], dim=-1)
        q = torch.cat([q_proj_2, q_rope_2], dim=-1)
        
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # shape: (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout=0.0, is_causal=True  # Fixed typo
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)
    
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                positional_ids: Optional[torch.Tensor] = None,
                pass_key_value: Optional[Tuple[torch.Tensor]] = None,
                rotary_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            positional_ids=positional_ids,
            pass_key_value=pass_key_value,
            rotary_emb=rotary_emb
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
class LlamaModel(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Fix: Use head_dim // 2 to match the attention layer's expectation
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = LlamaRotaryEmbedding(
            head_dim // 2,  # This should match the attention layer's rotary dimension
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        nn.init.normal_(self.embed_tokens.weight, std=config.initializer_range)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                positional_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        
        # Debug logging
        logging.debug(f"Input shape: {input_ids.shape}, Hidden states shape: {hidden_states.shape}")
        logging.debug(f"Max position embeddings: {self.config.max_position_embeddings}")
        
        # Ensure seq_len is valid and truncate inputs if necessary
        seq_len = hidden_states.shape[1]
        if seq_len > self.config.max_position_embeddings:
            logging.warning(f"Sequence length {hidden_states.shape[1]} exceeds max_position_embeddings, truncating to {self.config.max_position_embeddings}")
            seq_len = self.config.max_position_embeddings
            hidden_states = hidden_states[:, :seq_len, :]
        
        # Obtain rotary embeddings using the modified forward signature
        rotary_emb_out = self.rotary_emb(seq_len, hidden_states.device)
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                positional_ids=positional_ids,
                rotary_emb=rotary_emb_out
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if not config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, std=config.initializer_range)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                positional_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            positional_ids=positional_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
        
        return logits
    
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # logits: shape (batch_size, vocab_size)
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Fix: Handle multiple batches properly
        for batch_idx in range(logits.size(0)):
            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
            logits[batch_idx, indices_to_remove] = filter_value
    
    return logits

def generate_text(model, prompt, max_new_tokens=50, temperature=1.2, top_k=50, top_p=0.95):
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long)
    
    # Fix: Get device from model parameters
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            # Filter logits with top-k and/or top-p
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    generated_text = enc.decode(input_ids[0].tolist())
    model.train()
    return generated_text

from datasets import load_dataset
from torch.utils.data import DataLoader

class C4Dataset(torch.utils.data.IterableDataset):
    def __init__(self, split, seq_len):
        self.seq_len = seq_len
        self.dataset = load_dataset("allenai/c4", "realnewslike", split=split, streaming=True)
        self.tokenizer = __import__('tiktoken').get_encoding('gpt2')
        self.buffer = []

    def __iter__(self):
        for example in self.dataset:
            text = example.get('text', '')
            tokens = self.tokenizer.encode(text)
            self.buffer.extend(tokens)
            while len(self.buffer) >= self.seq_len + 1:
                chunk = self.buffer[:self.seq_len + 1]
                self.buffer = self.buffer[self.seq_len + 1:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (max_lr - min_lr)

# --- Distributed setup ---

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0

# --- MODIFIED: Main training function with FSDP ---
def main(batch_size=1, seq_len=512, epochs=20, steps_per_epoch=1000,
         warmup_steps=50, max_lr=3e-4):
    is_dist, rank, world_size, local_rank = setup_distributed()
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    
    fmt = logging.Formatter(f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    logging.root.addHandler(handler)
    device = torch.device('cuda', local_rank) if is_dist else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'FSDP' if is_dist else 'Single'} training on {device} (rank {rank} of {world_size})")

    # Seed
    seed = 1337 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Model
    config = SmolLM2Config()
    # It's better to create the model on the CPU first before FSDP moves it
    model = LlamaForCausalLM(config)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params / 1e9:.2f}B")

    if is_dist:
        # FSDP auto-wrap policy: Wrap any layer that has more than 1M parameters
        # auto_wrap_policy = size_based_auto_wrap_policy
        # More conservative auto-wrap policy for better memory management
        def transformer_auto_wrap_policy(module, recurse, nonwrapped_numel):
            # Only wrap decoder layers to reduce communication overhead
            return isinstance(module, LlamaDecoderLayer)        
        # --- IMPROVED FSDP Initialization with better memory management ---
        model = FSDP(
            model,
            auto_wrap_policy=transformer_auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=False),  # Disable CPU offloading for stability
            device_id=torch.cuda.current_device(),
            sync_module_states=True,  # Ensure consistent initialization
            limit_all_gathers=True,   # Reduce memory usage during all-gather
        )
        print(f"FSDP model wrapped on rank {rank}.")
    else:
        model.to(device)


    # Data
    dataset = C4Dataset(split='train', seq_len=seq_len)
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=1, pin_memory=False)
    if is_dist:
        dataset.dataset = dataset.dataset.shard(num_shards=world_size, index=rank)
    # Reduce num_workers to save memory
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False)

    # Optimizer & scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler('cuda')
    min_lr = max_lr * 0.1
    max_steps = epochs * steps_per_epoch

    # Training loop
    loader_iter = iter(loader)
    global_step = 0
    
    for epoch in range(epochs):
        if rank == 0:
            print(f"\n--- Starting Epoch {epoch + 1}/{epochs} ---")
        
        epoch_pbar = range(steps_per_epoch)
        if rank == 0:
            epoch_pbar = tqdm(epoch_pbar, desc=f"Epoch {epoch + 1} Progress")

        for _ in epoch_pbar:
            global_step += 1
            
            try:
                x, y = next(loader_iter)
            except StopIteration:
                if rank == 0:
                    tqdm.write("\nDataset exhausted, re-initializing loader.")
                loader_iter = iter(loader)
                x, y = next(loader_iter)

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                loss = model(x, labels=y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr = get_lr(global_step, max_lr, min_lr, warmup_steps, max_steps)
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            if rank == 0:
                epoch_pbar.set_postfix_str(f"Loss: {loss.item():.4f}, LR: {lr:.2e}")

    # Final save & cleanup
    if is_dist:
        dist.barrier()
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    save_model(model, optimizer, loss.item())
    if is_dist:
        try:
            dist.barrier()       # wait for everyone (may hang)
        except Exception as e:
            logging.warning(f"Ignored NCCL barrier error: {e}")
        finally:
            # Improved cleanup
            torch.cuda.empty_cache()
            dist.destroy_process_group()

# --- Spawn launcher for notebook ---
def distributed_worker(local_rank, world_size, args):
    try:
        # Set environment variables for distributed training
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355' # A common free port
        os.environ['RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        
        # Set CUDA device
        torch.cuda.set_device(local_rank)
        
        main(batch_size=args.batch_size,
             seq_len=args.seq_len,
             epochs=args.epochs,
             steps_per_epoch=args.steps_per_epoch,
             warmup_steps=args.warmup_steps,
             max_lr=args.max_lr)
    except Exception as e:
        print(f"Worker {local_rank} failed with error: {e}")
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    available_gpus = torch.cuda.device_count()
    # Use all available GPUs
    world_size = torch.cuda.device_count()
    
    print(f"Available GPUs: {available_gpus}, Using: {world_size}")
    
    if available_gpus == 0:
        print("No GPUs available. Running on CPU.")
        args = Args()
        main(batch_size=args.batch_size,
             seq_len=args.seq_len,
             epochs=args.epochs,
             steps_per_epoch=args.steps_per_epoch,
             warmup_steps=args.warmup_steps,
             max_lr=args.max_lr)
    else:
        args = Args()
        if world_size == 1:
            # Single GPU training
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            main(batch_size=args.batch_size,
                 seq_len=args.seq_len,
                 epochs=args.epochs,
                 steps_per_epoch=args.steps_per_epoch,
                 warmup_steps=args.warmup_steps,
                 max_lr=args.max_lr)
        else:
            # Multi-GPU training
            mp.spawn(
                fn=distributed_worker,
                args=(world_size, args),
                nprocs=world_size,
                join=True,
            )
        print("Training complete.")
