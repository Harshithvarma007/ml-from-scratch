import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import Prereq from '../Prereq'
import NeedsBackground from '../NeedsBackground'
import {
  Prose,
  Callout,
  Personify,
  Bridge,
  Gotcha,
  Challenge,
  References,
  KeyTerm,
  AsciiBlock,
} from '../primitives'
import GPTArchitecture from '../widgets/GPTArchitecture'
import ParameterBreakdown from '../widgets/ParameterBreakdown'

// Signature anchor: a jukebox playing a new song. The assembled model is
// the pressed vinyl; inference is the needle dropping; temperature is how
// drunk the DJ is; top-k/top-p is what records the DJ even considers;
// greedy decoding is the most popular song on repeat. Returned at the
// opening (silent model vs singing model), the sampling reveal, and the
// "why the same prompt gives different songs" beat.
export default function CodeGPTLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="code-gpt" />

      <Prose>
        <p>
          Every other lesson in this section carved one disc in a stack of
          vinyl. The tokenizer pressed strings into integer grooves. Token
          and positional embeddings etched position onto the disc.
          Self-attention let tokens whisper across bars. The feed-forward
          block let each token hum alone. LayerNorm kept the mix level;
          residuals kept the amplifier alive. None of those records plays a
          song on its own. They are a stack of pressings sitting on a shelf.
        </p>
        <p>
          This lesson is the jukebox. Not a new trick — an assembly. We take
          the pressings you already have, load them into the cabinet in the
          right order, wire the needle to the output, and watch a silent
          stack of vinyl turn into a machine that plays a new song every
          time you drop a coin in. By the end of this page the cabinet
          you&apos;ve built is byte-compatible with the exact vinyl OpenAI
          pressed in 2019. You will drop the needle on GPT-2.
        </p>
        <p>
          By the end you&apos;ll have (a) a top-down picture of every tensor
          a prompt travels through, (b) a NumPy forward pass through a
          two-layer toy <KeyTerm>GPT</KeyTerm>, (c) a complete PyTorch GPT
          class in the style of Karpathy&apos;s nanoGPT, and (d) loader
          code that pulls GPT-2&apos;s pretrained weights off HuggingFace
          and spins them on the same turntable. Same twelve-block cabinet,
          just bigger numbers on the record label.
        </p>
      </Prose>

      {/* ── The full architecture, top-down ─────────────────────────── */}
      <Prose>
        <p>
          Before any code, stare at the whole jukebox. A GPT has exactly
          four parts: an embedding layer, a stack of identical transformer
          blocks, a final layer norm, and a linear head that projects back
          to the vocabulary. That&apos;s the entire cabinet. Everything
          people call &ldquo;a GPT&rdquo; — GPT-2, GPT-3, GPT-4, Llama,
          Mistral — differs only in widths, depths, and a few small
          surgeries to the mechanism.
        </p>
      </Prose>

      <AsciiBlock caption="GPT, end to end — (B, T) in, (B, T, vocab) out">{`   token ids           (B, T)
        │
        ▼
  ┌───────────────┐       ┌──────────────────┐
  │ token_emb W_te│──────▶│    + pos_emb     │   (B, T, d_model)
  └───────────────┘       └──────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────┐
  │  TRANSFORMER BLOCK   ×  N_LAYER            │
  │  ┌──────────────────────────────────────┐  │
  │  │  x  +=  MultiHeadAttention(LN1(x))   │  │   (B, T, d_model)
  │  │  x  +=  FeedForward     (LN2(x))     │  │
  │  └──────────────────────────────────────┘  │
  └────────────────────────────────────────────┘
        │
        ▼
    ┌───────┐
    │  LN_f │                          (B, T, d_model)
    └───────┘
        │
        ▼
   ┌─────────────┐
   │   lm_head   │   ← tied to W_te    (B, T, vocab)
   └─────────────┘
        │
        ▼
      logits
`}</AsciiBlock>

      <Prose>
        <p>
          That&apos;s the whole jukebox. The embedding layer looks up a
          vector per token and adds a vector per position — the first
          groove on the record. The{' '}
          <NeedsBackground slug="transformer-block">
            transformer block
          </NeedsBackground>{' '}
          — attention, feed-forward, each wrapped in LayerNorm and a
          residual — runs <code>N</code> times, one pass of the needle per
          layer. The final LayerNorm cleans up the signal. The
          language-model head projects <code>d_model</code> back to{' '}
          <code>vocab_size</code> logits. Run{' '}
          <NeedsBackground slug="softmax">softmax</NeedsBackground> along
          the vocab axis and you have a probability over the next token.
          Train by minimizing cross-entropy between those logits and the
          real next token. Generate by sampling one and feeding it back in.
          A song, one note at a time.
        </p>
      </Prose>

      <MathBlock
        caption="forward pass, with shapes"
      >{`idx      : (B, T)              token indices
tok_emb  = W_te[idx]            → (B, T, d)
pos_emb  = W_pe[0:T]            → (T, d)           broadcast over batch
x        = tok_emb + pos_emb    → (B, T, d)
for l in 1..N:
    x = x + Attn(LN1(x))        → (B, T, d)
    x = x + MLP (LN2(x))        → (B, T, d)
x        = LN_f(x)              → (B, T, d)
logits   = x @ W_te.T           → (B, T, V)
loss     = CE(logits, targets)  → scalar`}</MathBlock>

      <Prose>
        <p>
          Every shape matters. <code>B</code> is batch size, <code>T</code>{' '}
          is sequence length, <code>d</code> is <code>d_model</code>,{' '}
          <code>V</code> is vocab. The batch and time axes are passive —
          attention mixes information across <code>T</code>, everything
          else is pointwise. The <code>d</code> axis is where
          representation lives. The final matmul <code>x @ W_te.T</code> is
          the interesting trick: the output projection uses the{' '}
          <em>same</em> matrix as the input embedding, just transposed.
          One record played through both the input groove and the output
          needle. That&apos;s weight tying, and we&apos;ll get to it.
        </p>
      </Prose>

      <GPTArchitecture />

      <Prose>
        <p>
          Play with the widget. Click any block to see the exact shape
          flowing through it, and watch the total parameter count scale as
          you crank <code>n_layer</code> and <code>d_model</code>. A few
          reference configs to lodge in memory — think of them as the A-side
          singles at the top of the chart:
        </p>
        <ul>
          <li>
            <strong>GPT-2 small</strong>: <code>n_layer=12, n_head=12, d_model=768,
            block_size=1024, vocab=50257</code> → ~124M params.
          </li>
          <li>
            <strong>GPT-2 medium</strong>: <code>n_layer=24, d_model=1024</code> → 355M.
          </li>
          <li>
            <strong>GPT-2 XL</strong>: <code>n_layer=48, d_model=1600</code> → 1.5B.
          </li>
          <li>
            <strong>GPT-3</strong>: <code>n_layer=96, d_model=12288</code> → 175B. Same
            cabinet, 1400× the groove density.
          </li>
        </ul>
        <p>
          There is no magic in the jump from 124M to 175B. Same four parts,
          wider discs, deeper stack, more hours of tape fed to the pressing
          plant.
        </p>
      </Prose>

      <Personify speaker="Positional embedding">
        The token embedding tells the model <em>what</em> each word is. I
        tell it <em>where</em> each word sits on the record. Without me,{' '}
        <code>the dog bit the man</code> and{' '}
        <code>the man bit the dog</code> are the same unordered bag of
        vectors — the DJ can&apos;t tell which groove comes first.
        I&apos;m a learned vector per position, added to each token before
        the transformer drops the needle. Attention is permutation-
        equivariant; I&apos;m the only reason word order exists.
      </Personify>

      {/* ── Weight tying + parameter budget ─────────────────────────── */}
      <Prose>
        <p>
          Now the one piece of model surgery that will surprise you if
          you&apos;ve only ever read the architecture diagram. The input
          embedding matrix <code>W_te</code> has shape <code>(V, d)</code>
          — one row per vocabulary token. The output projection{' '}
          <code>lm_head</code> has shape <code>(d, V)</code> — one column
          per vocabulary token. Those are the same numbers, transposed.
          So: use the same parameters. Literally bind them together;
          update one and the other moves with it. One record, played both
          when the needle reads <em>in</em> and when it writes <em>out</em>.
          This is <KeyTerm>weight tying</KeyTerm>.
        </p>
      </Prose>

      <MathBlock caption="weight tying — the same record, played twice">{`in:   embed(idx)  = W_te[idx]           (V, d) lookup
out:  logits     = x  @  W_te.T        (d, V) projection

    → lm_head.weight  is  token_emb.weight   (Python: one shared tensor)

parameter savings:  V · d   matrix counted once instead of twice
for GPT-2 small (V=50257, d=768)  →  ~39M parameters saved`}</MathBlock>

      <Prose>
        <p>
          The intuition: the embedding row for token <em>cat</em> is the
          vector <em>&ldquo;this is the word cat&rdquo;</em>. The lm_head
          column for token <em>cat</em> is the vector{' '}
          <em>&ldquo;predict cat when you see this&rdquo;</em>. Those
          should be the same thing — the notion of <em>cat</em>{' '}
          doesn&apos;t change between reading the song and writing it.
          Empirically, tying weights improves quality for a given parameter
          budget, and it saves a chunk of memory. Press &amp; Wolf 2017
          showed this is a nearly free win; every modern LM does it.
        </p>
      </Prose>

      <ParameterBreakdown />

      <Prose>
        <p>
          Look at where the parameters actually live. Most guides draw the
          transformer with attention as the star — attention gets the
          press, the complexity, the papers named after it. But in a GPT-2
          small, more than 60% of the parameters live in the feed-forward
          blocks. The FFN&apos;s two projections each have shape{' '}
          <code>(d, 4d)</code>; that&apos;s <code>8 · d²</code> parameters
          per block, versus <code>4 · d²</code> for attention&apos;s Q, K,
          V, O. The embedding, despite being one layer, is a huge slab
          because <code>V · d</code> is big when the vocab is 50257. The
          final lm_head would have been another slab — if you didn&apos;t
          tie weights. That&apos;s the ~40M you saved by pressing a
          double-sided record instead of two.
        </p>
      </Prose>

      <Callout variant="note" title="back-of-envelope parameter count">
        For a GPT-style model: <code>params ≈ V·d + N · 12 d²</code>. The{' '}
        <code>V·d</code> term is embedding (counted once, thanks to tying).
        The <code>12 d²</code> per block comes from 4·d² in attention +
        8·d² in FFN. For GPT-2 small:{' '}
        <code>50257·768 + 12·12·768² ≈ 39M + 85M ≈ 124M</code>. That&apos;s
        it — everything else (biases, LayerNorm params, position
        embedding) rounds to noise.
      </Callout>

      <Personify speaker="Tied lm_head">
        I&apos;m not really a layer. I&apos;m the embedding record,
        flipped to its B-side on the way out. You spent <code>V·d</code>{' '}
        parameters teaching me what each token <em>looks like</em>. When
        it comes time to predict the next token, why would you press a
        second, independent <code>V·d</code> disc to do the inverse? Same
        vocabulary, same semantic space. Just transpose me and matmul.
      </Personify>

      {/* ── Three-layer code ─────────────────────────────────────── */}
      <Prose>
        <p>
          Time to build the whole jukebox. Three passes: a stripped-down
          NumPy forward-only GPT to prove the shapes match the diagram, a
          real PyTorch class in ~120 lines that can train and generate, and
          the loader that pulls GPT-2&apos;s pretrained weights from
          HuggingFace and drops them into your class. Each pass is shorter
          than the last, and each one plays the same song.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 1 — numpy · tiny_gpt.py (forward-only, n_layer=2, d=32)">{`import numpy as np

def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def attention(x, qkv_w, qkv_b, proj_w, proj_b, n_head):
    B, T, d = x.shape
    qkv = x @ qkv_w + qkv_b                             # (B, T, 3d)
    q, k, v = np.split(qkv, 3, axis=-1)
    # split heads: (B, T, d) → (B, n_head, T, d_head)
    def split(z): return z.reshape(B, T, n_head, d // n_head).transpose(0, 2, 1, 3)
    q, k, v = split(q), split(k), split(v)
    att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d // n_head)          # (B, h, T, T)
    mask = np.triu(np.full((T, T), -np.inf), k=1)        # causal mask
    att = softmax(att + mask, axis=-1)
    out = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
    return out @ proj_w + proj_b

def ffn(x, w1, b1, w2, b2):
    h = np.maximum(0, x @ w1 + b1)                       # ReLU (real GPT-2 uses GELU)
    return h @ w2 + b2

def gpt_forward(idx, params, cfg):
    B, T = idx.shape
    d, V, n_layer, n_head = cfg['d'], cfg['V'], cfg['n_layer'], cfg['n_head']

    x = params['wte'][idx] + params['wpe'][:T]           # (B, T, d)

    for l in range(n_layer):
        b = params['blocks'][l]
        x = x + attention(layer_norm(x, b['ln1_g'], b['ln1_b']),
                          b['qkv_w'], b['qkv_b'], b['proj_w'], b['proj_b'], n_head)
        x = x + ffn(layer_norm(x, b['ln2_g'], b['ln2_b']),
                    b['mlp1_w'], b['mlp1_b'], b['mlp2_w'], b['mlp2_b'])

    x = layer_norm(x, params['lnf_g'], params['lnf_b'])
    logits = x @ params['wte'].T                          # TIED WEIGHTS: reuse wte
    return logits                                         # (B, T, V)

# Tiny random init just to prove shapes work
rng = np.random.default_rng(0)
cfg = dict(d=32, V=100, n_layer=2, n_head=4, T_max=16)
def rnd(*s): return rng.normal(0, 0.02, size=s)
def zero(*s): return np.zeros(s)

params = dict(
    wte=rnd(cfg['V'], cfg['d']),
    wpe=rnd(cfg['T_max'], cfg['d']),
    lnf_g=np.ones(cfg['d']), lnf_b=zero(cfg['d']),
    blocks=[dict(
        ln1_g=np.ones(cfg['d']), ln1_b=zero(cfg['d']),
        qkv_w=rnd(cfg['d'], 3 * cfg['d']), qkv_b=zero(3 * cfg['d']),
        proj_w=rnd(cfg['d'], cfg['d']), proj_b=zero(cfg['d']),
        ln2_g=np.ones(cfg['d']), ln2_b=zero(cfg['d']),
        mlp1_w=rnd(cfg['d'], 4 * cfg['d']), mlp1_b=zero(4 * cfg['d']),
        mlp2_w=rnd(4 * cfg['d'], cfg['d']), mlp2_b=zero(cfg['d']),
    ) for _ in range(cfg['n_layer'])]
)

idx = rng.integers(0, cfg['V'], size=(2, 8))              # batch=2, T=8
logits = gpt_forward(idx, params, cfg)
print("logits shape:", logits.shape)                       # -> (2, 8, 100)
print("sum of abs logits:", np.abs(logits).sum().round(2))`}</CodeBlock>

      <Prose>
        <p>
          Every shape in that file came from the diagram. <code>idx</code>{' '}
          in, logits out, N transformer blocks in between. Swap{' '}
          <code>np.maximum(0, ...)</code> for a real GELU, add dropout, add
          a loss, hook up gradients, and you have PyTorch&apos;s GPT. The
          turntable spins, but the record is still blank vinyl — random
          init, no song yet. Which is exactly what comes next.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — pytorch · gpt.py (full, trainable, ~120 lines, after nanoGPT)">{`import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer:    int = 12
    n_head:     int = 12
    d_model:    int = 768
    dropout:    float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)   # Q, K, V packed
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.n_head, self.d_model = cfg.n_head, cfg.d_model
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                     .view(1, 1, cfg.block_size, cfg.block_size)
        )

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.c_attn(x).split(d, dim=2)
        # (B, T, d) → (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, d // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, d // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, d // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, d)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.c_proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))          # pre-norm residual
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # WEIGHT TYING
        self.lm_head.weight = self.tok_emb.weight

        # GPT-2 style init — scaled for residual-path stability
        self.apply(self._init_weights)
        for p_name, p in self.named_parameters():
            if p_name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))    # (B, T, d)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]            # crop to block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature              # last token only
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


# Smoke test
cfg = GPTConfig(block_size=64, vocab_size=100, n_layer=2, n_head=4, d_model=32)
model = GPT(cfg)
idx = torch.randint(0, cfg.vocab_size, (2, 8))
logits, loss = model(idx, targets=idx)
print(f"logits: {list(logits.shape)}  loss: {loss.item():.3f}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")`}</CodeBlock>

      <Prose>
        <p>
          Read it. Count the lines — ~120, including the generate loop and
          weight init. This is the <em>entire</em> jukebox cabinet. Not a
          simplification, not a toy: the same class, with bigger config
          numbers, is what OpenAI pressed into the GPT-2 record. Karpathy
          released essentially this code as nanoGPT in 2022; it&apos;s
          been the reference implementation ever since.
        </p>
        <p>
          One detail worth flagging in the <code>generate</code> method:
          the three knobs on the front panel of the jukebox are{' '}
          <code>temperature</code>, <code>top_k</code>, and the implicit
          &ldquo;sample from the softmax.&rdquo; Temperature is how drunk
          the DJ is — crank it up and the needle picks stranger records;
          set it near zero and the DJ puts the same hit song on repeat.{' '}
          <code>top_k</code> is what records the DJ is even allowed to
          consider — the rest of the jukebox is locked.{' '}
          <code>top_k=1</code> is greedy decoding, the most popular song
          on repeat. We&apos;ll unpack all of that in the sampling lesson;
          for now, note that the song changes every time you press play
          because <code>multinomial</code> is rolling dice.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 3 — pytorch · load_pretrained_gpt2.py" output={`params: 124,439,808
loaded 124 / 124 pretrained tensors

prompt: "In a shocking finding, scientist discovered a herd of unicorns"
  → "living in a remote, previously unexplored valley, in the Andes Mountains.
     Even more surprising to the researchers was the fact that the unicorns spoke..."`}>{`# Load OpenAI's pretrained GPT-2 weights into our GPT class.
# HuggingFace names the tensors slightly differently; this map resolves it.
from transformers import GPT2LMHeadModel

def load_gpt2_pretrained(size="gpt2"):
    configs = {
        "gpt2":        dict(n_layer=12, n_head=12, d_model=768),
        "gpt2-medium": dict(n_layer=24, n_head=16, d_model=1024),
        "gpt2-large":  dict(n_layer=36, n_head=20, d_model=1280),
        "gpt2-xl":     dict(n_layer=48, n_head=25, d_model=1600),
    }[size]
    cfg = GPTConfig(**configs, block_size=1024, vocab_size=50257, dropout=0.0)
    model = GPT(cfg)

    hf = GPT2LMHeadModel.from_pretrained(size)
    hf_sd, sd = hf.state_dict(), model.state_dict()

    # HF uses Conv1D (transposed linear) in attn/mlp — need to transpose those.
    transpose = ["attn.c_attn.weight", "attn.c_proj.weight",
                 "mlp.c_fc.weight", "mlp.c_proj.weight"]

    name_map = {
        "wte.weight": "tok_emb.weight",
        "wpe.weight": "pos_emb.weight",
        "ln_f.weight": "ln_f.weight",
        "ln_f.bias":   "ln_f.bias",
    }
    for i in range(cfg.n_layer):
        for hf_k, our_k in [
            ("ln_1.weight",     "ln1.weight"),
            ("ln_1.bias",       "ln1.bias"),
            ("attn.c_attn.weight","attn.c_attn.weight"),
            ("attn.c_attn.bias", "attn.c_attn.bias"),
            ("attn.c_proj.weight","attn.c_proj.weight"),
            ("attn.c_proj.bias", "attn.c_proj.bias"),
            ("ln_2.weight",     "ln2.weight"),
            ("ln_2.bias",       "ln2.bias"),
            ("mlp.c_fc.weight", "mlp.c_fc.weight"),
            ("mlp.c_fc.bias",   "mlp.c_fc.bias"),
            ("mlp.c_proj.weight","mlp.c_proj.weight"),
            ("mlp.c_proj.bias", "mlp.c_proj.bias"),
        ]:
            name_map[f"h.{i}.{hf_k}"] = f"blocks.{i}.{our_k}"

    loaded = 0
    with torch.no_grad():
        for hf_name, our_name in name_map.items():
            t = hf_sd[hf_name]
            if any(hf_name.endswith(s) for s in transpose):
                t = t.t()
            sd[our_name].copy_(t)
            loaded += 1
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"loaded {loaded} / {len(name_map)} pretrained tensors")
    return model

# Load it, generate with it
model = load_gpt2_pretrained("gpt2")
model.eval()

from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
prompt = "In a shocking finding, scientist discovered a herd of unicorns"
idx = torch.tensor(tok.encode(prompt)).unsqueeze(0)
out = model.generate(idx, max_new_tokens=40, temperature=0.8, top_k=40)
print(tok.decode(out[0].tolist()))`}</CodeBlock>

      <Prose>
        <p>
          Let that land. The same 120-line class you just read is
          structurally compatible with OpenAI&apos;s trained vinyl — once
          you account for HuggingFace&apos;s <code>Conv1D</code> storing
          its linears transposed. Copy the tensors across, set the model
          to <code>eval()</code>, feed it a prompt, and the needle drops
          on coherent English. Same computation, same diagram, 124 million
          learned numbers. A silent cabinet an hour ago; a jukebox playing
          a new song now.
        </p>
      </Prose>

      <Bridge
        label="numpy forward  →  pytorch GPT  →  pretrained GPT-2"
        rows={[
          {
            left: 'tiny_gpt.py (200 lines, forward only)',
            right: 'class GPT(nn.Module): 120 lines, train + generate',
            note: 'autograd replaces manual shape bookkeeping; GELU replaces ReLU; dropout added',
          },
          {
            left: 'random 0.02 init for every param',
            right: 'special init: c_proj std *= 1 / sqrt(2·N)',
            note: 'keeps residual path from exploding as depth grows (GPT-2 paper, §2.3)',
          },
          {
            left: 'lm_head = new (d, V) matrix',
            right: 'self.lm_head.weight = self.tok_emb.weight',
            note: 'one line of weight tying, ~40M parameters saved',
          },
          {
            left: 'smoke test: logits.shape == (B, T, V)',
            right: 'load_gpt2_pretrained("gpt2")',
            note: 'same cabinet, OpenAI’s pressed vinyl — drops the needle in a few lines',
          },
        ]}
      />

      <Callout variant="insight" title="why the same prompt gives different songs">
        Feed GPT-2 the unicorn prompt twice and you will get two different
        completions. Same vinyl, same needle, same starting groove —
        different song every play. That&apos;s not a bug. The last step of{' '}
        <code>generate</code> is <code>torch.multinomial</code>, which
        rolls dice weighted by the softmax. Temperature reshapes the dice
        (drunker DJ, flatter distribution); top-k locks the jukebox to the
        DJ&apos;s short list. Set <code>temperature</code> near zero and{' '}
        <code>top_k=1</code> and you&apos;ve hard-coded the most popular
        song on repeat — deterministic, boring, frequently correct. The
        full sampling lesson is its own page; what matters here is that
        the randomness lives in the generation loop, not the model
        weights.
      </Callout>

      <Callout variant="insight" title="why nanoGPT changed the game">
        When Karpathy released nanoGPT in early 2023, the field already
        had GPT-2, GPT-3, and countless transformer tutorials. What
        nanoGPT did was collapse the whole cabinet into ~300 lines of
        readable PyTorch that trained on a single GPU in an afternoon. It
        made &ldquo;train a GPT&rdquo; a student project. Every
        open-weight LM released since — Mistral, Llama, Gemma, Qwen — is,
        at the architecture level, a handful of small edits to this
        exact file.
      </Callout>

      <Callout variant="note" title="inference cost, and the KV cache teaser">
        Generating N tokens from a T-long prompt, using the code above, is
        O((T+N)² · d) per new token: the jukebox rewinds the whole record
        every time it wants the next note. That&apos;s fine for
        prototyping, wasteful in production. The fix is a{' '}
        <NeedsBackground slug="kv-cache">KV cache</NeedsBackground> — keep
        the past K and V tensors around so new tokens only do
        O((T+N)·d) work per step. For now: marvel that you wrote a working
        GPT before worrying about making it fast.
      </Callout>

      <Callout variant="insight" title="scaling laws, briefly">
        Kaplan et al. 2020 and Hoffmann et al. 2022 (Chinchilla) showed
        that GPT performance is shockingly predictable from three numbers:
        parameters, training tokens, and compute. Chinchilla&apos;s rule
        of thumb — <code>tokens ≈ 20 · params</code> — corrects the
        undertrained-but-too-large GPT-3 era. A 124M model trained on
        ~2.5B tokens is the &ldquo;Chinchilla-optimal&rdquo; point; go
        bigger or smaller at your own inefficiency. Same four-part
        cabinet, scaled along the right axis.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Tying the wrong tensors.</strong>{' '}
          <code>nn.Embedding.weight</code> has shape <code>(V, d)</code>.{' '}
          <code>nn.Linear(d, V).weight</code> also has shape <code>(V, d)</code> — PyTorch
          stores it as <code>(out_features, in_features)</code>. So{' '}
          <code>self.lm_head.weight = self.tok_emb.weight</code> works <em>because</em> both
          are <code>(V, d)</code>. If you accidentally write{' '}
          <code>self.lm_head.weight = self.tok_emb.weight.T</code> you&apos;ll get a shape
          mismatch — or worse, a silent bug if the dimensions happen to align.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting the final LayerNorm.</strong> Every
          modern GPT has an <code>ln_f</code> between the last transformer block and{' '}
          <code>lm_head</code>. Skip it and{' '}
          <NeedsBackground slug="train-your-gpt">training</NeedsBackground>{' '}
          either diverges or plateaus at a bad loss. It&apos;s one line,
          it&apos;s always there, it&apos;s easy to miss when you think
          &ldquo;attention + FFN &times; N and I&apos;m done.&rdquo;
        </p>
        <p>
          <strong className="text-term-amber">The residual-path init.</strong> GPT-2&apos;s
          paper initializes <code>c_proj.weight</code> — the last linear in attention and in
          the MLP — with std <code>0.02 / sqrt(2·N)</code> instead of just <code>0.02</code>.
          The division keeps variance from compounding as the residual stream passes through N
          blocks. Skip this and gradients explode on deep models. Copy it from nanoGPT; don&apos;t
          reinvent it.
        </p>
        <p>
          <strong className="text-term-amber">Dropout during <code>generate</code>.</strong> If
          you forget <code>model.eval()</code> before sampling, dropout is still active and
          you&apos;re corrupting activations randomly on every step. The
          needle skips. Generations look confused. Always{' '}
          <code>model.eval()</code> for inference,{' '}
          <code>model.train()</code> to come back.
        </p>
      </Gotcha>

      <Challenge prompt="Train a 4-layer GPT on Tiny Shakespeare">
        <p>
          Grab <code>input.txt</code> from Karpathy&apos;s char-rnn repo (about 1.1MB of
          Shakespeare concatenated). Use a character-level tokenizer (<code>vocab_size ≈ 65</code>
          ) for speed. Configure{' '}
          <code>GPTConfig(block_size=128, vocab_size=65, n_layer=4, n_head=4, d_model=128)</code>
          — that&apos;s about <strong>800K parameters</strong>, runs on a laptop CPU in an
          evening or a T4 GPU in a few minutes.
        </p>
        <p className="mt-2">
          Train for 5000 steps with AdamW, lr=3e-4, batch_size=32. Sample from the model every
          500 steps. You&apos;ll hear four stages of the record coming
          into focus:
        </p>
        <ul className="mt-2">
          <li>Step 0: random characters (<code>&quot;q3x!p!v&quot;</code>) — static on the vinyl.</li>
          <li>Step 500: mostly real characters, random sequences.</li>
          <li>Step 2000: recognizable words, bad grammar (<code>&quot;thee not thy lord the king&quot;</code>).</li>
          <li>
            Step 5000: almost-plausible Shakespeare — verse structure, speaker attributions,
            some syntactic coherence. Meaning still garbage, but the <em>form</em> is right.
            The DJ is sober-ish.
          </li>
        </ul>
        <p className="mt-2 text-dark-text-muted">
          Bonus: try <code>n_layer=6, d_model=192</code> and compare. The form locks in
          faster, and the model becomes worth reading out loud for comic effect.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> A GPT is four things in
          sequence: token + position embeddings, N identical transformer
          blocks (each = pre-norm attention + pre-norm FFN, both
          residual), a final LayerNorm, and a tied-weight linear head to
          the vocabulary. ~120 lines of PyTorch. Every open LM you read
          about — Llama, Mistral, Qwen, Gemma — is a handful of small
          edits to this cabinet: swap LayerNorm for RMSNorm, sinusoidal
          positions for RoPE, ReLU for SwiGLU, dense attention for
          grouped-query. The jukebox you just built is the jukebox they
          all ship.
        </p>
        <p>
          <strong>Next up — Grouped Query Attention.</strong> The vinyl
          you just pressed has one quiet problem: every attention head
          keeps its own full-sized K and V record, and at inference time
          the KV cache bloats linearly with heads. Llama&apos;s fix,{' '}
          <code>grouped-query-attention</code>, is to make several query
          heads share one key/value record — carpooling instead of each
          head driving solo. Same cabinet, tighter grooves, a cheaper song
          at the same fidelity. That&apos;s the next lesson.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Language Models are Unsupervised Multitask Learners',
            author: 'Radford, Wu, Child, Luan, Amodei, Sutskever',
            venue: 'OpenAI, 2019 — the GPT-2 paper, including architecture and init details',
            url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
          },
          {
            title: 'Language Models are Few-Shot Learners',
            author: 'Brown et al.',
            venue: 'NeurIPS 2020 — GPT-3: same architecture, 117× more parameters',
            url: 'https://arxiv.org/abs/2005.14165',
          },
          {
            title: 'nanoGPT',
            author: 'Andrej Karpathy',
            venue: 'github.com/karpathy/nanoGPT — the reference implementation this lesson follows',
            url: 'https://github.com/karpathy/nanoGPT',
          },
          {
            title: 'Using the Output Embedding to Improve Language Models',
            author: 'Press, Wolf',
            venue: 'EACL 2017 — the weight-tying paper',
            url: 'https://arxiv.org/abs/1608.05859',
          },
          {
            title: 'Dive into Deep Learning — 11.9 Large-Scale Pretraining with Transformers',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html',
          },
          {
            title: 'Scaling Laws for Neural Language Models',
            author: 'Kaplan et al.',
            venue: 'OpenAI, 2020 — parameters, data, compute: power laws all the way down',
            url: 'https://arxiv.org/abs/2001.08361',
          },
        ]}
      />
    </div>
  )
}
