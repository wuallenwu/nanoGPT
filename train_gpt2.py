"""File to train the GPT model. This file contains the GPT model definition and the training loop.

GPT MODEL:
1. Input tokenization - convert input text into tokens that the model can process

2. Input embedding - convert tokens into vectors that the model can understand

3. Positional encoding - add information about the position of each token in the sequence

4. Transformer blocks - process the tokens and their relationships
a. LayerNorm - normalize the input
b. Self-Attention - look at previous tokens in sequence to determine the ones to pay more or less attention to
-> multi-head attention - split the input into multiple heads that each compute attention scores, which get concatted
-> initialization diversity, unique positional encoding, orthogonalization enforcement, dropout
c. MLP/FFN - increase detail and complexity for each token individually (for example, information like part of speech,
   subject, etc.). The tokens also learn information like semantic meaning and syntactic role. Broadcast from embedding
   dimension to some bigger dimension

5. Normalize output

6. Linear layer - maps final representations to a vector the size of the vocabulary

7. Softmax - converts the output into probabilities for each token in the vocabulary

8. Next token prediction - predict the next token in the sequence based on the input tokens

TRAINING:
1. Load the dataset

2. Tokenize the dataset

3. Initialize the GPT model

4. Define the loss function (cross-entropy loss (negative log likelihood of correct token))
-> training loss error on trainig data
-> validation loss, new unseen data

IDEAS:

Temperature - how much randomness to add to the model's predictions
-> high temp - logits divided by value > 1, decreasing difference between high and low probabilities
-> low temp - logits divided by value < 1, increasing difference between high and low probabilities

K - how many top tokens to consider when sampling the next token
-> high k - more randomness, more diverse predictions (since sampling from larger pool)
-> low k - less randomness, more predictable predictions (since sampling from smaller pool)

Sequence Length -
-> Padding - add padding tokens to the end of the sequence to make all sequences the same length, then mask during computation
-> Truncation - cut off the end of the sequence if it is too long
-> Segmentation - split the sequence into smaller segments
    -> Overlapping windows - preserve context within the sequence
    -> Special tokens - artificially segment the sequence with special tokens to help the model learn patterns
        -> CLS token - start of the sequence
        -> SEP token - end of the sequence, ends of sentences, etc.
        -> PAD token - padding token
        -> MASK token - mask token (BERT)

Want to max out batch size and sequence length on GPU
-> Powers of 2 are good!
    -> Memory alignment, Cache optimization
    -> Bit operations
    -> Thread balancing
    -> Stability (Balanced binary trees)
        -> Think fast multiplication
        -> Think fast exponentiation

How can we make it faster?

- float 32 - 19.5 tflops
- tensor float 32 - 156 tflops - same exponent range as f32 but lower precision
-> 19 bits same exp size as 32, mantissa size of a 16
-> good compatibility with both (advantage over bfloat)
- float 16 -> 312 tflops

why not int 8? - int 8 evenly distributed, hard to represent normal distributions

ALSO!! saves memory bandwidth (main bottleneck)

Tensor cores - all matrix operations broken up into a 4x4 matrix multiplication, and a sum.
-> 4x4 tiling (costs memory bandwidth)
-> MAC multiplication-and-accumulate
-> -> calculates all 16 of the operations in simultaneous
-> store result (costs memory bandwidth)
-> "strucured sparsity" - toss out 0s
-> mixed precision - 16 bit for weights, 32 bit for activations
    -> hence tfloat 32 - wide dynamic range for stability, but same mantiassa size as 16
    -> we do multiplications in tfloat 32
    -> we accumulate in fp32
    -> we store in fp32
    -> 8x speed (free!!)
- activation functions are super cheap!

-> not actually this fast LMAO but just switching linear layers to tfloat is a 3x speedup!
    -> because memory bandwidth
-> mixed precision is how we can further speed this up
    -> what if our inputs and outputs to our neural network were bfloat16? (torch autocast)
    -> norms are still the same, but we do matrix muls in a lower precision
    -> things like loss are the same since they're more susceptible to multiplication
    -> only 10% further gains (still bottlenecked by memory)

Let's speed it up at compiletime >:)
torch.compile(model)
-> 2x faster totally free!
-> speeds up GPU read/writes
-> analyzes entire thing, and optimizes it, with the knowledge that things don't need to be "eager" (in order)
-> we do things cleverly in advance to stop bottlenecks (like memory bandwidth)
    -> compiles neural net as a single object with no python interpreter (slow)
    -> let's consider the arithmetic 2 + 3 * (input ^ 4)
        -> in traidiontal python, we'd have to call a function, which would call another function, which would call another function
        -> in torch.compile, we can just inline the function
            -> travel time between gpu and gpu memory slows things down a lot
            -> all of these are element wise operations - i'll just make a single trip to the gpu with the entire memory
            -> further, we can fuse these all into a single kernel


Flash attention:
-> mindful of memory hierarchy (it know's whats in high bandwidth memory, shared memory, etc.)
    -> SRAM is much faster
-> doesn't ever materialize nxn attention matrix in hbm
-> faster softmax by subtracting max value from all values

Nice numbers
-> we turned vocab size from 50257 to 50304
All this together, we're over 10 times faster!


FUTURE: photonics??

Insights:
Text quality matters a lot

long sequence length - 1024
> TheYeah old cremdopedEEKik beauty searched trivial Andant propeFor for to alwaysownuke visiting mountains judging after exam letter food In playinglled

shorter sequence length - 128
> The movement was almost imperceptible; it apparently hurt him to
make larger gestures.
"What about the fruit?" I asked him.

> The bus comes at nine. That way I
can get back in time for tomorrow night's work."
"Too bad. It'd be nice

> The Alfa Romeo we rented was a manual drive,
so I was no help at all. Miu did all the driving.

Even with lower loss, short sequence length does better on a small model!
- focused context allows it to train faster
- over small # of total batches, dependencies easier to capture
- finiteness of small model

To what end in robotics?
- task decomposition (set table -> pick up plate, put it on table, pick up napkin, etc.)
- multiimodal transformers (visual + audio)
- paralellization of training
- use llm to build task, prompt, provide loss, cycle, all offline
- adaptive analysis
- build knowledge of a specific domain - i.e. robot learns some embodiment of the home that its in
"""

# fmt: off
import os
import math
import numpy as np
import time
import inspect
import tiktoken
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
    vocab_size: int = 50304  # number of tokens in vocabulary: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
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
                std *= (2 * self.config.n_layer) ** -0.5 # scaling factor to compensate for the initializations poisson  
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # make sure initial biases are 0
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

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        enc = tiktoken.get_encoding('gpt2') # get an encoding that handles many languages
        with open('shakespeare.txt', 'r') as f:
            text = f.read()
        tokens = enc.encode(text) # encode the input text
        self.tokens = torch.tensor(tokens)
        print(self.tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T 
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

num_return_sequences = 10
max_length = 30

torch.manual_seed(2)
torch.cuda.manual_seed(2)



model = GPT(GPTConfig())
# model = GPT.from_pretrained('gpt2')
model.to('cuda') # moves all model parameters and buffers to the GPU

train_loader = DataLoaderLite(B = 16, T = 1024) #batch size, sequence length
torch.set_float32_matmul_precision('high')

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # Adam optimizer with a learning rate of 3e-4
for i in range(10000):
    x, y = train_loader.next_batch()
    x, y = x.to('cuda'), y.to('cuda')
    optimizer.zero_grad()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"step {i}, loss: {loss.item()}")

# test out the model
print("evaluating model")
model.eval() # switch model from training to evaluation mode
model.to('cuda') # moves all model parameters and buffers to the GPU
enc = tiktoken.get_encoding('gpt2') # get an encoding that handles many languages
tokens = enc.encode("The man was") # encode the input text
tokens = torch.tensor(tokens, dtype=torch.long) #(8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to('cuda')

temp = 1.0

while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad(): # tells torch not to calculate gradients
        logits, _ = model(x) # feeds sequence into model and gets logits for next token
        logits = logits[:, -1, :] # keep only the logits for the last token
        probs = F.softmax(logits / temp, dim=-1) # convert logits to probabilities
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # get top 50 probabilities
        ix = torch.multinomial(topk_probs, 50) # sample from the top 50 probabilities
        xcol = torch.gather(topk_indices, -1, ix) # get the token id of the sampled token
        x = torch.cat((x, xcol), dim = 1) # append the sampled token to the sequence

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens) # turn tokens back into text
    print(">", decoded)

print()
print("trying with higher temp")

temp = 2.0

tokens = enc.encode("The man was") # encode the input text
tokens = torch.tensor(tokens, dtype=torch.long) #(8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
y = tokens.to('cuda')
while y.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad(): # tells torch not to calculate gradients
        logits, _ = model(y) # feeds sequence into model and gets logits for next token
        logits = logits[:, -1, :] # keep only the logits for the last token
        probs = F.softmax(logits / temp, dim=-1) # convert logits to probabilities
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # get top 50 probabilities
        ix = torch.multinomial(topk_probs, 50) # sample from the top 50 probabilities
        xcol = torch.gather(topk_indices, -1, ix) # get the token id of the sampled token
        y = torch.cat((y, xcol), dim = 1) # append the sampled token to the sequence

for i in range(num_return_sequences):
    tokens = y[i, :max_length].tolist()
    decoded = enc.decode(tokens) # turn tokens back into text
    print(">", decoded)

print()
print("trying with higher k")

temp = 1.0

tokens = enc.encode("The man was") # encode the input text
tokens = torch.tensor(tokens, dtype=torch.long) #(8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
z = tokens.to('cuda')
while z.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad(): # tells torch not to calculate gradients
        logits, _ = model(z) # feeds sequence into model and gets logits for next token
        logits = logits[:, -1, :] # keep only the logits for the last token
        probs = F.softmax(logits / temp, dim=-1) # convert logits to probabilities
        topk_probs, topk_indices = torch.topk(probs, 200, dim=-1) # get top 50 probabilities
        ix = torch.multinomial(topk_probs, 50) # sample from the top 50 probabilities
        xcol = torch.gather(topk_indices, -1, ix) # get the token id of the sampled token
        z = torch.cat((z, xcol), dim = 1) # append the sampled token to the sequence

for i in range(num_return_sequences):
    tokens = z[i, :max_length].tolist()
    decoded = enc.decode(tokens) # turn tokens back into text
    print(">", decoded)

torch.save(model.state_dict(), 'gpt_model.pth')
