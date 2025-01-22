import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import logging
import os
import time

# Remove MPS-specific settings and replace with CUDA empty cache
import torch
torch.cuda.empty_cache()  # Replace MPS cache clearing

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# Adjust hyperparameters for better memory usage
BATCH_SIZE = 32          # Reduced from 64
BLOCK_SIZE = 256        
N_EMBD = 384           # Reduced from 512
N_HEAD = 8
N_LAYER = 6            # Reduced from 8
LEARNING_RATE = 3e-4
WARMUP_STEPS = 1000
MAX_ITERS = 10000
EVAL_INTERVAL = 100    # More frequent evaluation
DROPOUT = 0.2
GRADIENT_CLIP = 1.0
WEIGHT_DECAY = 0.1

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_mult = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE * lr_mult

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_head, head_size)
        self.ln2 = LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)

    def forward(self, x):
        # Added residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Add these functions back, but with actual implementation
def encode(text, stoi):
    """Encode text to tokens using the provided mapping"""
    return [stoi[c] for c in text]

def decode(tokens, itos):
    """Decode tokens back to text using the provided mapping"""
    return ''.join([itos[i] for i in tokens])

def get_vocab():
    """Get the fixed vocabulary mappings"""
    # Define the character set that was used during training
    chars = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos

def main():
    # Move vocabulary creation to get_vocab()
    stoi, itos = get_vocab()

    # Load and preprocess data
    logging.info("Loading data from data/input.txt")
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create character-level encoder/decoder
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Create train/val splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Add CUDA diagnostic information
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(0)  # Explicitly set to first CUDA device

    # Modify device selection logic to be more explicit
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    model = DecoderTransformer(vocab_size)
    model = model.to(device)
    logging.info(f"Model device: {next(model.parameters()).device}")  # Verify model device

    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY
    )
    
    # Initialize scheduler
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_STEPS, MAX_ITERS)

    print(f"Training config:")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Block size: {BLOCK_SIZE}")
    print(f"- Embedding dim: {N_EMBD}")
    print(f"- Heads: {N_HEAD}")
    print(f"- Layers: {N_LAYER}")
    print(f"- Max iterations: {MAX_ITERS}")
    print("Starting training...\n")

    # Training loop
    best_val_loss = float('inf')
    for iter in range(MAX_ITERS):
        # Sample a batch of data
        xb, yb = get_batch('train')
        xb, yb = xb.to(device), yb.to(device)

        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()

        # Print training progress
        if iter % 10 == 0:
            print(f"Iteration {iter}/{MAX_ITERS}: Training loss {loss.item():.4f}, LR {optimizer.param_groups[0]['lr']:.6f}")

        # Evaluation
        if iter % EVAL_INTERVAL == 0:
            losses = []
            model.eval()
            for _ in range(10):
                with torch.no_grad():
                    xb, yb = get_batch('val')
                    xb, yb = xb.to(device), yb.to(device)
                    _, loss = model(xb, yb)
                    losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)
            print(f"\nEvaluation at step {iter}:")
            print(f"- Train loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                torch.save(model.state_dict(), 'best_model.pt')
                print(f"- New best model saved! (val loss: {best_val_loss:.4f})\n")
            
            if avg_loss < 0.0999:
                print("Target loss achieved! Stopping training.")
                break
            
            model.train()

        # Generate sample text every 1000 iterations
        if iter % 1000 == 0 and iter > 0:
            print("\nGenerating sample text...")
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            print(decode(model.generate(context, max_new_tokens=100)[0].tolist(), itos))
            print("\n" + "="*50 + "\n")

    # Save final model
    torch.save(model.state_dict(), 'final_model.pt')
    print("Training completed!")
    print(f"Final best validation loss: {best_val_loss:.4f}")

    # Ensure data is explicitly moved to GPU in get_batch
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix]).to(device)
        return x, y

if __name__ == '__main__':
    main() 