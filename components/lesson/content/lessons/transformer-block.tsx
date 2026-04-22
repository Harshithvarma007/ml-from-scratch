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
import TransformerBlockDiagram from '../widgets/TransformerBlockDiagram'
import BlockStack from '../widgets/BlockStack'

export default function TransformerBlockLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="transformer-block" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Here is the lie every transformer diagram tells you: a transformer is a tall tower. A
          monolithic 96-story skyscraper of matrix multiplies, purpose-built floor by floor,
          which is why it cost a hundred million dollars to train. The diagram shows you the
          tower. It does not show you the brick.
        </p>
        <p>
          A transformer is not a tower. It is <em>one Lego brick</em>, stamped out of a mold,
          snapped on top of itself N times. GPT-2 small is the same brick stacked 12 times.
          GPT-3 is the same brick stacked 96 times. LLaMA, Mistral, Claude, every decoder-only
          model you&apos;ve heard of — same brick, different N. The brick has a fixed shape. It
          takes tensors of shape <code>(B, T, d_model)</code> in, it returns tensors of the
          exact same shape out. That&apos;s not a coincidence. That is the entire design.
        </p>
        <p>
          This lesson is about the brick. Once you can hold the brick in your head — two
          sub-layers, two residual adds, two layer-norms, one MLP — you can stop memorizing
          architectures and start reading them. A 175B-parameter model is not a new invention;
          it is the same brick we are about to build, stacked until the GPUs cry.
        </p>
      </Prose>

      <Personify speaker="Transformer block">
        I am one Lego brick. Tokens come in, look at each other through my attention, think
        privately through my feed-forward, and leave a little smarter — same shape they arrived
        in. Snap me onto another me and you have a 2-layer model. Snap ninety-six of me
        together and you have GPT-3. I am not the tower. I am the brick the tower is made of.
      </Personify>

      <Prose>
        <p>
          Everything you&apos;ve met so far was a part, not a whole. Token embeddings turned
          strings into vectors. Positional encodings gave those vectors an address.{' '}
          <NeedsBackground slug="multi-headed-self-attention">multi-head attention</NeedsBackground>{' '}
          let each token look at every other token. An MLP gave each token a moment alone to
          think. <NeedsBackground slug="layer-normalization">layer norm</NeedsBackground> kept
          the statistics from drifting.{' '}
          <NeedsBackground slug="resnet-and-skip-connections">residual connections</NeedsBackground>{' '}
          kept the gradients alive. Great parts. Scattered across six different lessons. Now
          they all snap into the same brick.
        </p>
        <p>
          The <KeyTerm>transformer block</KeyTerm> is the one repeating unit that wraps all of
          that into a single composable layer. Same shape in, same shape out, different learned
          weights per copy. GPT-2 small stacks 12 bricks. GPT-3 stacks 96. Scale is literally
          just how many bricks you line up.
        </p>
      </Prose>

      {/* ── ASCII diagram ─────────────────────────────────────── */}
      <Prose>
        <p>
          Here is the brick in ASCII — the pre-norm variant every modern codebase uses. Data
          flows bottom to top. The two <code>⊕</code> nodes are residual additions; the two{' '}
          <code>LN</code> boxes are layer-norms applied <em>before</em> each sub-layer. Look at
          the top and bottom arrows — the shapes are identical, which is the whole reason the
          brick snaps onto itself.
        </p>
      </Prose>

      <AsciiBlock caption="one transformer block — pre-norm (GPT-2 style)">
{`                    ▲  out  (B, T, d_model)
                    │
                    ⊕ ◄──────────────┐   residual #2
                    │                │
                  ┌─┴──┐              │
                  │ MLP│   (d → 4d → d)
                  └─┬──┘              │
                    │                │
                  ┌─┴──┐              │
                  │ LN │   layer-norm before FFN
                  └─┬──┘              │
                    ├──────────────►──┘
                    │
                    ⊕ ◄──────────────┐   residual #1
                    │                │
                  ┌─┴──┐              │
                  │Attn│   multi-head self-attention
                  └─┬──┘              │
                    │                │
                  ┌─┴──┐              │
                  │ LN │   layer-norm before Attn
                  └─┬──┘              │
                    ├──────────────►──┘
                    │
                    ▲  in   (B, T, d_model)`}
      </AsciiBlock>

      {/* ── The two residual lines ──────────────────────────── */}
      <Prose>
        <p>
          The brick is two lines of math. Given an input <code>x</code> of shape{' '}
          <code>(B, T, d_model)</code>:
        </p>
      </Prose>

      <MathBlock caption="the transformer block — pre-norm form">
{`x   ←   x   +   Attn( LN(x) )          # residual #1

x   ←   x   +   FFN ( LN(x) )          # residual #2`}
      </MathBlock>

      <Prose>
        <p>
          Two sub-layers, two residual adds, two layer-norms. Shape in equals shape out — which
          is exactly what lets you stack the brick N times without rewiring anything. If you
          can read those two lines, you can read the forward pass of every GPT-family model
          ever shipped. The rest is weight counts and marketing.
        </p>
      </Prose>

      {/* ── The two halves of the brick ────────────────────── */}
      <Prose>
        <p>
          Look at the brick again and something pops out: it has two halves. Top half is{' '}
          feed-forward-and-residual. Bottom half is attention-and-residual. They are not
          doing the same job, and that&apos;s the single most useful thing to know about the
          whole architecture.
        </p>
      </Prose>

      {/* ── Widget 1: TransformerBlockDiagram ───────────────── */}
      <TransformerBlockDiagram />

      <Prose>
        <p>
          Click each component in the diagram. The attention sub-layer mixes information{' '}
          <em>across tokens</em> — this is the only place in the whole brick where one
          token&apos;s representation is touched by its neighbours.{' '}
          The <NeedsBackground slug="mlp-from-scratch">feed-forward</NeedsBackground> sub-layer
          then runs on each token independently, with the <em>same</em> MLP weights applied at
          every position. Layer-norm keeps activations from drifting across the stack. The
          residuals — the <code>x + ...</code> part — are what make the whole tower trainable
          in the first place.
        </p>
      </Prose>

      <Callout variant="insight" title="attention mixes, FFN thinks">
        A mental model that pays for itself forever: <em>attention is communication, FFN is
        computation.</em> The two halves of the brick are two distinct jobs. The attention
        half is the only moment tokens exchange information. The FFN half is the only moment
        each token processes what it just heard, alone. One brick is one round of
        &ldquo;listen to everyone, then think about it.&rdquo; Stack twelve bricks and you get
        twelve rounds of that loop. Stack ninety-six and you get GPT-3.
      </Callout>

      <Personify speaker="Residual connection">
        I am the <code>+ x</code>. I look like a trivial wire, but without me this building
        falls over. My gradient is a clean <code>1</code> no matter how tall you stack the
        network — every brick upstream can reach every brick downstream through me. I&apos;m
        why you can train a 96-layer transformer and it actually learns.
      </Personify>

      {/* ── Why the brick stacks cleanly ────────────────────── */}
      <Prose>
        <p>
          Here is the quiet miracle of the design: the brick stacks. Not metaphorically — in
          the literal PyTorch sense that you can write <code>nn.Sequential(*[Block() for _ in
          range(96)])</code> and it just works. The reason is two promises the brick keeps, and
          a historical fact about what happens when you break them.
        </p>
        <p>
          Promise one: <strong>shape in equals shape out.</strong> No funky broadcasts, no
          dimensional surgery between layers. Every brick sees exactly the same tensor shape
          its sibling one floor down saw. That&apos;s the mechanical half.
        </p>
        <p>
          Promise two: <strong>gradients survive the climb.</strong> The residual adds are not
          a stylistic choice — they are a load-bearing wall. Every time you write{' '}
          <code>x + Attn(LN(x))</code>, you are handing the next layer an unobstructed wire
          back to the input. When backprop runs, gradients flow through that wire with a
          derivative of exactly <code>1</code>, bypassing whatever nonlinear mess the sub-layer
          added. Stack 96 bricks without that wire and you re-run the old RNN horror show of{' '}
          <NeedsBackground slug="vanishing-gradient-problem">vanishing gradients</NeedsBackground>{' '}
          — the signal gets diluted through each layer until the bottom of the stack effectively
          stops learning.
        </p>
        <p>
          Promise two is also why pre-norm beat post-norm. Put the layer-norm <em>inside</em>
          the residual path (post-norm, the original paper) and the identity wire is no longer
          clean — gradients have to fight through an LN on every floor. Move the LN{' '}
          <em>outside</em> the residual path (pre-norm, modern default) and the wire is pristine
          again. Same brick, same Lego studs, one screw moved — and the difference is the
          ability to train past about 20 layers without a warmup schedule from hell.
        </p>
      </Prose>

      {/* ── FFN math ────────────────────────────────────────── */}
      <Prose>
        <p>
          Zoom into the FFN half of the brick. It&apos;s the boring, beautiful workhorse — a
          two-layer MLP with a nonlinearity between them. It&apos;s also, for almost every
          transformer ever trained, where most of the parameters live.
        </p>
      </Prose>

      <MathBlock caption="feed-forward network — per-token MLP">
{`FFN(x)  =   GELU( x · W₁ + b₁ ) · W₂ + b₂

shapes:
  W₁ :  (d_model, 4·d_model)     # expand
  W₂ :  (4·d_model, d_model)     # contract

x is applied per-token — no mixing across positions.`}
      </MathBlock>

      <Prose>
        <p>
          The ratio is not magic, but it is load-bearing. The original paper set the inner
          dimension to <code>4·d_model</code>, and every major transformer since has kept that
          shape (give or take — GLU-style variants use <code>~8·d_model/3</code> to keep the
          parameter count the same after splitting). The inner expansion gives the MLP room to
          route and combine features before projecting back down.
        </p>
        <p>
          Crucially, the FFN is applied <strong>independently to every position</strong>. Same
          weights, same computation, different token. The MLP does not know that other tokens
          exist. That sounds like a limitation until you remember that attention already did
          the cross-token mixing on the previous sub-layer — the division of labor is the whole
          point. Each half of the brick has one job and does only that job.
        </p>
      </Prose>

      {/* ── Widget 2: BlockStack ────────────────────────────── */}
      <BlockStack />

      <Prose>
        <p>
          Move the <code>N</code> slider. Watch what doesn&apos;t change: the input shape, the
          output shape, the per-block structure. Watch what does: the parameter count and the
          FLOP budget, both scaling linearly in N. That&apos;s what &ldquo;stacking bricks&rdquo;
          actually means in practice. GPT-2 small: 12 bricks, <code>d_model = 768</code>, 117M
          parameters. GPT-2 medium: 24 bricks, <code>d_model = 1024</code>, 345M. Large and XL
          keep adding bricks and widening <code>d_model</code>. Same Lego set. More pieces.
        </p>
        <p>
          Per-block parameters are roughly <code>12 · d_model²</code> — four matrices of shape{' '}
          <code>d_model × d_model</code> for attention (<code>W_Q</code>, <code>W_K</code>,{' '}
          <code>W_V</code>, <code>W_O</code>) and eight <code>d_model²</code>-worth in the FFN
          (the <code>4×</code> expansion means <code>W₁</code> and <code>W₂</code> are each{' '}
          <code>4 · d_model²</code> in size). At <code>d_model = 768</code> that&apos;s about
          7M parameters per brick. Multiply by 12 bricks and the stack alone is ~85M — essentially
          the entirety of GPT-2 small.
        </p>
      </Prose>

      <Callout variant="note" title="most of the parameters are in the FFN">
        Sum the ratios: 4 matrices of <code>d_model²</code> for attention vs 8 for the FFN. The
        feed-forward half of the brick is <em>twice as parametric</em> as the attention half in
        a standard layer. When people say &ldquo;a transformer is mostly an MLP,&rdquo; this is
        what they mean — attention is the famous part, but the FFN is where the weight budget
        sits.
      </Callout>

      <Personify speaker="Feed-forward MLP">
        I am the per-token computer. Attention does the gossip; I do the thinking. Each token
        gets my full undivided attention, one at a time, with the same weights every time. I
        hold two thirds of the parameters in this brick. If the model &ldquo;knows facts,&rdquo;
        they live in my rows.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────── */}
      <Prose>
        <p>
          Three layers of code, same pattern every lesson in this series uses. First the
          forward pass of a single brick in numpy with every step visible — no autograd magic,
          just matrix multiplies. Then the one-line PyTorch built-in. Then a hand-rolled
          module that mirrors every production codebase. Each layer is a shortcut for the one
          below it, never magic.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · transformer_block_numpy.py"
        output={`input shape:  (2, 5, 64)
output shape: (2, 5, 64)
max |Δ|: 0.3741 (residual kept identity path alive)`}
      >{`import numpy as np

def layer_norm(x, eps=1e-5):
    mu  = x.mean(axis=-1, keepdims=True)
    var = x.var (axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def gelu(x):
    # exact GELU; in practice PyTorch uses the tanh approximation
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def attention(x, W_qkv, W_o):
    B, T, d = x.shape
    qkv = x @ W_qkv                          # (B, T, 3d)
    q, k, v = np.split(qkv, 3, axis=-1)      # each (B, T, d)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d)
    return softmax(scores) @ v @ W_o         # (B, T, d)

def ffn(x, W1, W2):
    return gelu(x @ W1) @ W2                 # (B,T,d) -> (B,T,4d) -> (B,T,d)

def transformer_block(x, params):
    # pre-norm: LN happens before each sub-layer; residual is added AFTER
    h = x + attention(layer_norm(x), params['W_qkv'], params['W_o'])
    h = h + ffn      (layer_norm(h), params['W1'   ], params['W2' ])
    return h

# random weights just to show the shapes flow
rng  = np.random.default_rng(0)
d, T, B = 64, 5, 2
x = rng.standard_normal((B, T, d))
params = {
    'W_qkv': rng.standard_normal((d, 3*d)) * 0.02,
    'W_o'  : rng.standard_normal((d,   d)) * 0.02,
    'W1'   : rng.standard_normal((d, 4*d)) * 0.02,
    'W2'   : rng.standard_normal((4*d, d)) * 0.02,
}
y = transformer_block(x, params)
print("input shape: ", x.shape)
print("output shape:", y.shape)
print(f"max |Δ|: {np.abs(y - x).max():.4f} (residual kept identity path alive)")`}</CodeBlock>

      <Prose>
        <p>
          Production PyTorch gives you this as a single layer. Pass in the shape arguments, you
          get a drop-in brick. The only catch is that <code>nn.TransformerEncoderLayer</code>{' '}
          defaults to <em>post-norm</em> unless you pass <code>norm_first=True</code> — which
          you want, if you are training from scratch in 2025.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch built-in · transformer_block_torch.py"
        output={`output shape: torch.Size([2, 5, 768])
parameters:   7,087,872`}
      >{`import torch
import torch.nn as nn

block = nn.TransformerEncoderLayer(
    d_model        = 768,
    nhead          = 12,
    dim_feedforward= 4 * 768,      # the 4× rule
    activation     = 'gelu',
    norm_first     = True,         # pre-norm (modern default)
    batch_first    = True,
)

x = torch.randn(2, 5, 768)
y = block(x)
print("output shape:", y.shape)
print(f"parameters:   {sum(p.numel() for p in block.parameters()):,}")`}</CodeBlock>

      <Prose>
        <p>
          Rolling your own brick is a fifteen-line module. This is the version you want to
          read, copy, and keep in your head — the code that will appear, nearly verbatim, in
          every tutorial, blog post, and real implementation from nanoGPT onward.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch hand-rolled · transformer_block.py"
        output={`output shape: torch.Size([2, 5, 768])
parameters:   7,087,872
residual preserved shape: True`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model=768, n_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        # pre-norm, residual added *after* each sub-layer
        h        = self.ln1(x)
        h, _     = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x        = x + h                                   # residual #1
        x        = x + self.mlp(self.ln2(x))               # residual #2
        return x

blk = TransformerBlock(d_model=768, n_heads=12)
x   = torch.randn(2, 5, 768)
y   = blk(x)
print("output shape:", y.shape)
print(f"parameters:   {sum(p.numel() for p in blk.parameters()):,}")
print("residual preserved shape:", y.shape == x.shape)`}</CodeBlock>

      <Bridge
        label="numpy → hand-rolled pytorch"
        rows={[
          {
            left: 'h = x + attention(layer_norm(x), ...)',
            right: 'x = x + self.attn(self.ln1(x), ...)',
            note: 'same two-liner; pytorch adds autograd + GPU + dropout',
          },
          {
            left: 'gelu(h @ W1) @ W2',
            right: 'nn.Sequential(Linear, GELU, Linear)',
            note: 'the 4× inner dim is the mlp_ratio argument',
          },
          {
            left: 'hand-written softmax + mask',
            right: 'nn.MultiheadAttention(attn_mask=...)',
            note: 'built-in handles Q/K/V split, heads, and masking',
          },
        ]}
      />

      {/* ── Encoder vs Decoder / Post-norm callouts ─────────── */}
      <Callout variant="note" title="encoder vs decoder block">
        An <em>encoder</em> brick uses bidirectional self-attention — every token can look at
        every other token. A <em>decoder</em> brick adds two twists: the self-attention is
        causal (a token can only see past tokens, via a triangular mask), and in the
        encoder-decoder architecture it has a <em>third</em> sub-layer — cross-attention that
        queries the encoder&apos;s output. GPT-style models are decoder-only: causal mask, no
        cross-attention, two sub-layers per brick. BERT-style models are encoder-only:
        bidirectional, two sub-layers per brick. The brick shape is otherwise identical.
      </Callout>

      <Callout variant="insight" title="post-norm vs pre-norm, and why the field switched">
        The original 2017 paper put layer-norm <em>after</em> each sub-layer (post-norm):{' '}
        <code>x = LN(x + Attn(x))</code>. It works for shallow stacks. Snap 48+ of these
        bricks together and gradients get wild — the residual wire now runs <em>through</em>{' '}
        the normalization, so the identity path is not preserved. Xiong et al. 2020 showed
        that swapping to pre-norm (<code>x = x + Attn(LN(x))</code>) makes training stable
        without a warmup schedule, because the identity path now bypasses the LN entirely.
        Every modern LLM — GPT-2, GPT-3, LLaMA, Mistral — is pre-norm.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">LN placement:</strong> if you copy a brick from
          an old tutorial and find it will not train past a few layers, check whether
          it&apos;s post-norm. Flip to pre-norm and watch the loss curve go from spiky to
          smooth.
        </p>
        <p>
          <strong className="text-term-amber">The 4× is not universal:</strong> LLaMA-style
          SwiGLU FFNs use <code>~8·d_model/3</code> for the hidden dim. The reason is that
          SwiGLU splits the inner projection into two halves — keeping total parameters equal
          to the standard 4× block requires shrinking. If you are comparing parameter counts
          between architectures, check which FFN variant they are using.
        </p>
        <p>
          <strong className="text-term-amber">FFN does NOT mix tokens:</strong> a common
          misread — the FFN is applied per-position with the same weights. It is a{' '}
          <code>1×1</code> conv over the sequence, not an attention-like mixer. Every piece of
          cross-token communication in a transformer happens in the attention sub-layer, full
          stop.
        </p>
        <p>
          <strong className="text-term-amber">Dropout placement:</strong> some references put
          dropout inside the attention scores; some put it on the sub-layer output; some do
          both. For reproducing a paper, check the exact placement. For a new model, put it on
          the sub-layer output and move on with your life.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────── */}
      <Challenge prompt="Stack four blocks and watch it learn">
        <p>
          Take the <code>TransformerBlock</code> class from layer 3. Wrap it in a tiny
          language model: an embedding layer (<code>vocab_size=256</code>, byte-level),
          positional encoding, 4 stacked blocks (<code>d_model=128</code>, <code>n_heads=4</code>),
          a final layer-norm, and a linear head back to vocab size. Use a causal mask.
        </p>
        <p className="mt-2">
          Train it to predict the next character of a 10KB chunk of text (the opening of any
          book from Project Gutenberg will do). Batch size 32, sequence length 128, AdamW at
          <code> lr=3e-4</code>. A single RTX card or even a laptop CPU is enough.
        </p>
        <p className="mt-2">
          Log the loss every 50 steps. Within 2000 steps the loss should drop from ~5.5 (random
          over 256 bytes) to ~2.0 (model has learned the alphabet&apos;s shape). Sample from it
          at the end — it will produce English-looking gibberish with correct spacing, plausible
          word lengths, and the occasional real word. That is four transformer blocks doing
          exactly what a hundred of them do at scale.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: change <code>N</code> from 4 to 1 and train again. Note how much harder the
          model has to work to model the same sequence — depth matters, and you will feel it
          in the loss curve.
        </p>
      </Challenge>

      {/* ── Closing + section teaser ────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> A transformer is not a tower; it&apos;s one
          brick, snapped together N times. The brick itself is <em>attention + FFN</em> wrapped
          in two residuals and two layer-norms. Pre-norm is the modern default because it keeps
          the residual wire clean. The FFN half holds most of the parameters and runs per-token;
          the attention half is the only place tokens communicate. Shape in equals shape out,
          which is why the stack is a one-line <code>for</code> loop. Every modern LLM is this
          brick, stacked.
        </p>
        <p>
          <strong>Cliffhanger — the brick wasn&apos;t built just for text.</strong> Look at the
          brick&apos;s input shape: <code>(B, T, d_model)</code>. A batch of sequences of
          vectors. Nothing in that signature says &ldquo;language.&rdquo; Text isn&apos;t the
          only grid that fits into this brick — images are just 2D text if you squint. Chop a
          picture into 16×16 patches, flatten each patch into a vector, and you&apos;ve got
          <code> (B, T, d_model) </code>where <code>T</code> is &ldquo;number of patches&rdquo;
          and each patch is a &ldquo;word.&rdquo; Snap the same brick you just built on top of
          that and you have a <strong>vision transformer</strong> — the model that walked into
          computer vision and politely retired a decade of CNN architectures. Same brick.
          Different input. That&apos;s next.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Attention Is All You Need',
            author: 'Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin',
            venue: 'NeurIPS 2017 — the original transformer paper (post-norm)',
            url: 'https://arxiv.org/abs/1706.03762',
          },
          {
            title: 'Language Models are Unsupervised Multitask Learners',
            author: 'Radford, Wu, Child, Luan, Amodei, Sutskever',
            venue: 'OpenAI 2019 — GPT-2, the decoder-only pre-norm stack',
            url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
          },
          {
            title: 'On Layer Normalization in the Transformer Architecture',
            author: 'Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang, Liu',
            venue: 'ICML 2020 — pre-norm vs post-norm, why pre-norm trains without warmup',
            url: 'https://arxiv.org/abs/2002.04745',
          },
          {
            title: 'Dive into Deep Learning — §11.7 The Transformer Architecture',
            author: 'Zhang, Lipton, Li, Smola',
            url: 'https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html',
          },
          {
            title: 'nanoGPT — a minimal, readable transformer implementation',
            author: 'Karpathy',
            venue: 'GitHub — the reference block implementation in ~50 lines',
            url: 'https://github.com/karpathy/nanoGPT',
          },
        ]}
      />
    </div>
  )
}
