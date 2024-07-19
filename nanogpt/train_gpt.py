"""Example GPT2 which uses the weights from Huggingface model

Reproducible example for inference only 

! Very Vulgar at times !

"""

# fmt: off
import os
import math
import time
import inspect
import transformers
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
# from hellaswag import render_example, iterate_examples

# Defines a class for multi-head scaled dot-product attention
class CausalSelfAttention(nn.Module):
    # Query: vector representing the value of an individual token
    # Key: vector representing other individual tokens in the sequence
    # Value: vector representing tokens based on importance
    # Each "head" is one group of query, key, and value vectors
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) #linear transform to q, k, v
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
# Creates a class for a multi-layer perceptron
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd) # projects input from original size to 4x size
        self.gelu    = nn.GELU(approximate='tanh') # similar to ReLU with non-zero gradients for negative inputs
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd) # high dimensional output back to original size
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
# Creates a transformer block class 
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # layer normalization to ensure input has mean 0 variance 1
        self.attn = CausalSelfAttention(config) # allows model to "look back" at previous tokens in the sequence
        self.ln_2 = nn.LayerNorm(config.n_embd) # layer normalization
        self.mlp = MLP(config) # multi-layer perceptron to process tokens independently (no inter-token communication)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # apply attention and add it to the result
        x = x + self.mlp(self.ln_2(x)) # apply mlp and add it to the result
        return x # processed output
    
@dataclass
# Define a configuration class for initializing the GPT model
class GPTConfig:
    block_size: int = 1024  # max sequence length (amount of tokens that can be processed ina  single sentence)
    vocab_size: int = 50257  # number of tokens in vocabulary: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers 
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd), # word token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd), # positional embeddings
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # creates a series of transformer blocks
                ln_f=nn.LayerNorm(config.n_embd), # final layer normalization
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # produces raw scores for each token

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B (batch size), T (length of sequence))
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # constructor that creates a gpt model from hugging face weights based on which size we want
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval() # switch model from training to evaluation mode
model.to('cuda') # moves all model parameters and buffers to the GPU

# get tokens for the prefix of the sentence to generate
import tiktoken
enc = tiktoken.get_encoding('gpt2') # get the encoding for GPT-2
tokens = enc.encode("Hi! I'm Stompy.") # encode the input text
tokens = torch.tensor(tokens, dtype=torch.long) #(8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to('cuda')

# set seeds for consistency
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad(): # tells torch not to calculate gradients
        logits = model(x) # feeds sequence into model and gets logits for next token
        logits = logits[:, -1, :] # keep only the logits for the last token
        probs = F.softmax(logits, dim=-1) # convert logits to probabilities
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # get top 50 probabilities
        ix = torch.multinomial(topk_probs, 1) # sample from the top 50 probabilities
        xcol = torch.gather(topk_indices, -1, ix) # get the token id of the sampled token
        x = torch.cat((x, xcol), dim = 1) # append the sampled token to the sequence

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens) # turn tokens back into text
    print(">", decoded)
    
print("hello world")
