import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import Quiz from '../Quiz'
import WhatNext from '../WhatNext'
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
import QKVComputation from '../widgets/QKVComputation'
import AttentionHeatmap from '../widgets/AttentionHeatmap'

// Signature anchor: a dinner party where every guest simultaneously turns to
// every other guest and asks, "how relevant are you to me right now?" Each
// guest carries a query (what they want to know), a key (what they're about),
// and a value (what they'll actually say when called on). The room returns
// at the Q/K/V reveal and again when softmax decides who the guest ends up
// listening to.
export default function SelfAttentionLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout ────────────────────────────────── */}
      <Prereq currentSlug="self-attention" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Picture a dinner party. Twenty guests around a long table, everyone
          talking at once. Now picture the same party the way a{' '}
          <NeedsBackground slug="recurrent-neural-network">RNNs</NeedsBackground>{' '}
          would have to run it: one guest speaks, the next guest listens to a
          compressed summary of everything that&apos;s been said so far, then
          whispers their own summary to the guest on their right. By the time
          the rumor has crossed the room, the first joke has been rewritten,
          forgotten, and re-forgotten through nineteen hops of a very tired
          game of telephone. Long-range conversation is possible in principle
          and miserable in practice.
        </p>
        <p>
          <KeyTerm>Self-attention</KeyTerm> runs the party differently. Every
          guest turns to every other guest at the same instant and silently
          asks the same question: <em>how relevant are you to me right now?</em>{' '}
          No passing notes down the chain. No summarizing. Guest 100 hears
          guest 1 with the same clarity as guest 99 — the shortest path between
          any two people in the room is always one look across the table. That
          single architectural decision — &ldquo;everyone reads everyone, in
          parallel&rdquo; — is the engine underneath BERT, GPT, Claude, every
          modern LLM you&apos;ve touched.
        </p>
        <p>
          This lesson builds the room from the ground up. We&apos;ll meet the
          three things each guest carries, derive the Q/K/V projection trick,
          unpack the scaled dot-product formula a factor at a time, see why
          the <code>√d_k</code> in the denominator is mathematically
          non-negotiable, watch a live attention matrix light up over a real
          sentence, and implement causal attention from scratch in three
          flavors.
        </p>
      </Prose>

      {/* ── Architecture diagram ────────────────────────────────── */}
      <AsciiBlock caption="self-attention block — the shape you will see a thousand times">
{`        tokens X                  (N × d)   — "The cat sat on the mat"
            │
            │  linear projections (learned)
            ├──────────┬──────────┐
            ▼          ▼          ▼
         Q = X·Wq   K = X·Wk   V = X·Wv      (N × d_k, N × d_k, N × d_v)
            │          │          │
            └────┐     │     ┌────┘
                 ▼     ▼     │
              scores = Q · Kᵀ                  (N × N)
                 │
                 ÷  √d_k                        keeps softmax un-saturated
                 │
                 (optional mask — causal, pad)
                 │
              softmax along  KEYS               rows sum to 1
                 │
              attn weights  A                   (N × N)
                 │
                 ·  V                           (N × d_v)
                 │
                 ▼
             output  Z                          each row = context-aware token`}
      </AsciiBlock>

      <Prose>
        <p>
          Three learned matrices. One matmul, one scale, one softmax, one more
          matmul. That is all of attention. Every paper you&apos;ll read in
          this section is a variation, optimization, or re-packaging of the
          diagram above — the same dinner party, re-catered.
        </p>
      </Prose>

      {/* ── Q, K, V projections ─────────────────────────────────── */}
      <Prose>
        <p>
          Now the reveal. Each guest at the table arrives carrying three
          things — not two, not five, three — and once you see them separately
          the whole mechanism clicks. Given a sequence of <code>N</code> token
          embeddings (our{' '}
          <NeedsBackground slug="word-embeddings">word embeddings</NeedsBackground>
          {' '}for the guests) <code>X ∈ ℝ^(N × d)</code>, attention projects
          each guest three ways:
        </p>
      </Prose>

      <MathBlock caption="three learned projections per token">
{`Q  =  X · W_Q          queries   (N × d_k)
K  =  X · W_K          keys      (N × d_k)
V  =  X · W_V          values    (N × d_v)`}
      </MathBlock>

      <Prose>
        <p>
          Same guest, three different pitches. A guest&apos;s{' '}
          <KeyTerm>query</KeyTerm> is &ldquo;what am I interested in right
          now?&rdquo; — the topic they&apos;re trying to follow. Their{' '}
          <KeyTerm>key</KeyTerm> is &ldquo;here&apos;s what I have to
          offer&rdquo; — the label they&apos;re broadcasting across the room,
          the nametag that tells everyone else what they&apos;re about. Their{' '}
          <KeyTerm>value</KeyTerm> is &ldquo;here&apos;s what I&apos;ll
          actually say if you call on me&rdquo; — the contribution they make
          once they&apos;re chosen. The weights <code>W_Q</code>,{' '}
          <code>W_K</code>, <code>W_V</code> are the only learnable parameters
          in a single-head attention block, and they&apos;re what distinguish
          a great party host from a random one — they&apos;ve learned which
          questions to ask, which nametags to print, and which things to say
          when called on.
        </p>
        <p>
          Why three roles and not one? Because the question a guest is asking
          (&ldquo;who here is talking about music?&rdquo;) is genuinely
          different from the nametag they&apos;re wearing (&ldquo;I play
          jazz&rdquo;) which is different from the story they&apos;ll tell
          when called on (a fifteen-minute anecdote about a bass player in
          1974). Collapse any two of those roles into one and the mechanism
          stops working — the room loses the ability to look for something
          different from what it&apos;s offering.
        </p>
      </Prose>

      {/* ── Widget 1: QKV Computation ──────────────────────────── */}
      <QKVComputation />

      <Prose>
        <p>
          Four guests at the table, fully projected. Highlight any query row
          and the widget draws a{' '}
          <NeedsBackground slug="single-neuron">dot product</NeedsBackground>{' '}
          with every key — that&apos;s one guest turning to the room and
          sizing up every nametag. The values don&apos;t enter the picture
          yet; they&apos;re waiting patiently on the right, ready to be
          weighted and summed once softmax has decided who gets the floor.
          Notice the shapes: Q and K are the same width (<code>d_k</code>, so
          their dot product is scalar), V can be a different width{' '}
          (<code>d_v</code>) — in practice they&apos;re usually equal, but
          nothing in the math requires it.
        </p>
      </Prose>

      {/* ── The scaled dot-product attention formula ─────────────── */}
      <MathBlock caption="scaled dot-product attention — the entire formula">
{`                       ┌─ Q · Kᵀ ─┐
Attention(Q, K, V)  =  softmax │ ─────── │  · V
                       └─  √d_k  ┘`}
      </MathBlock>

      <Prose>
        <p>
          One line. Four operations. Let&apos;s unpack it piece by piece,
          because every factor in there is doing work.
        </p>
      </Prose>

      <Personify speaker="Query">
        I&apos;m the question each token asks on every forward pass. I don&apos;t carry the
        answer — I just score how relevant every key is to what I&apos;m looking for. Dot me
        against every key, normalize with softmax, and the resulting distribution tells the
        layer above which values to weight and which to ignore. I am replaced at every layer;
        the network gets a fresh set of questions at every depth.
      </Personify>

      <MathBlock caption="step-by-step, one factor at a time">
{`(1)  scores   =  Q · Kᵀ                    (N × N)   — raw similarity
(2)  scaled   =  scores / √d_k                       — variance fix
(3)  weights  =  softmax(scaled, axis=keys)          — row-wise distribution
(4)  output   =  weights · V               (N × d_v) — context-mixed values`}
      </MathBlock>

      <Prose>
        <p>
          <strong>Step 1 — raw scores.</strong> <code>Q · Kᵀ</code> is an{' '}
          <code>N × N</code> matrix. Entry <code>(i, j)</code> is guest{' '}
          <code>i</code>&apos;s query dotted with guest <code>j</code>&apos;s
          key — how interested <em>this</em> guest is in what <em>that</em>{' '}
          guest is offering. Every pair in the room scored at once, in a
          single matmul. This is the payoff of the matrix formulation: the GPU
          does the whole thing with no Python loops in sight.
        </p>
        <p>
          <strong>Step 2 — the <code>√d_k</code> scaling.</strong> This looks
          arbitrary. It isn&apos;t. Suppose <code>q</code> and <code>k</code>{' '}
          are vectors of dimension <code>d_k</code> with unit-variance
          entries. Their dot product is a sum of <code>d_k</code> zero-mean
          unit-variance products, so by the central limit theorem the dot
          product has variance <code>d_k</code> and standard deviation{' '}
          <code>√d_k</code>. For <code>d_k = 64</code>, scores routinely land
          at <code>±8</code>. Feed that into{' '}
          <NeedsBackground slug="softmax">softmax</NeedsBackground> and one
          entry dominates so hard that the gradient on every other entry goes
          to ~0 — the softmax has <em>saturated</em>. A single guest is
          shouting so loudly that nobody else in the room can be heard.
          Dividing by <code>√d_k</code> pulls the variance back to{' '}
          <code>1</code>, keeping the distribution soft and gradients alive.
        </p>
        <p>
          <strong>Step 3 — softmax.</strong> Applied <em>along the keys
          axis</em> (rows of the score matrix). Row <code>i</code> becomes a
          probability distribution over keys: exactly how guest <code>i</code>{' '}
          splits their attention budget across the room. They have one unit
          of listening to spend, and softmax tells them how to spend it. The
          &ldquo;wrong axis&rdquo; mistake — softmaxing along queries instead
          of keys — is one of the most common bugs in from-scratch attention.
          Read Step 3 twice and tape it to your monitor.
        </p>
        <p>
          <strong>Step 4 — weighted sum of values.</strong> Multiply the{' '}
          <code>N × N</code> attention weights by the <code>N × d_v</code>{' '}
          value matrix. Output row <code>i</code> is a convex combination of
          every guest&apos;s value vector, weighted by how much guest{' '}
          <code>i</code> was listening to each of them. Each output is a{' '}
          <em>context-aware</em> impression of that position — a single
          guest&apos;s final takeaway from the room, assembled out of what
          everyone else said, pre-weighted by how much they mattered.
        </p>
      </Prose>

      {/* ── Widget 2: Attention Heatmap ──────────────────────────── */}
      <AttentionHeatmap />

      <Prose>
        <p>
          That&apos;s an actual attention matrix from a small model run on a
          real sentence. Scrub the query selector and watch one row at a time
          light up — that row is a single guest showing you who they ended up
          listening to. The bright diagonal is a hint that guests attend to
          themselves a lot (sensible — a token&apos;s own embedding is usually
          the best single source of information about it, in the same way
          that knowing what <em>you</em> came to the party to talk about is a
          decent prior on what you&apos;ll say next). The off-diagonal heat
          is where the interesting learning lives: a pronoun listening to its
          antecedent, a verb listening to its subject, a closing bracket
          listening to its opener. Every pattern you see here is emergent —
          no one programmed &ldquo;link pronoun to antecedent,&rdquo; the
          network learned to eavesdrop that way because doing so lowered the
          loss.
        </p>
      </Prose>

      <Personify speaker="Softmax(QK^T / √d_k)">
        I&apos;m the attention distribution — the <em>who listens to whom</em> of this layer.
        Every row of me is a probability distribution; I sum to 1 across keys by
        construction. When I&apos;m sharp, I pick one source. When I&apos;m flat, I average
        everything. The network tunes me by moving the Q/K weights that produced me; I
        don&apos;t have parameters of my own, I&apos;m just the shape their interaction
        takes.
      </Personify>

      <Callout variant="insight" title="why attention is permutation-equivariant">
        If you shuffle the rows of <code>X</code> — rearrange the seating
        chart — the output shuffles the same way. The attention weights are
        identical up to the same permutation. Nothing in{' '}
        <code>softmax(QKᵀ/√d_k) · V</code> depends on <em>which</em> seat a
        guest happens to be in. That&apos;s a feature (everyone can be
        processed in parallel, no head of the table) and a bug (the model
        can&apos;t tell &ldquo;the cat saw the dog&rdquo; from &ldquo;the dog
        saw the cat&rdquo; — same guests, same conversations, different
        story). The fix is{' '}
        <NeedsBackground slug="positional-encoding">positional encoding</NeedsBackground>:
        stamp a seat number onto each guest before attention sees them, so the
        room knows who came in first. We cover positional encodings in their
        own lesson — keep this &ldquo;attention is blind to order&rdquo; fact
        in mind until then.
      </Callout>

      {/* ── Three-layer code ─────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations of the same block. Pure Python on a tiny
          4-guest party so you can see every index. NumPy with{' '}
          <code>einsum</code> to do the whole thing in one expression.
          PyTorch&apos;s <code>F.scaled_dot_product_attention</code> — the
          call you actually use in production, which dispatches to
          FlashAttention kernels under the hood.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · attention_scratch.py"
        output={`attention weights (row-stochastic):
 [0.37, 0.28, 0.19, 0.16]
 [0.22, 0.41, 0.24, 0.13]
 [0.18, 0.27, 0.38, 0.17]
 [0.15, 0.20, 0.28, 0.37]
output[0] = [0.41, -0.12, 0.33, 0.07]`}
      >{`import math
import random

def matmul(A, B):
    m, k = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    assert k == k2
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)] for i in range(m)]

def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def softmax_row(row):
    m = max(row)                              # stability — subtract max before exp
    ex = [math.exp(v - m) for v in row]
    s = sum(ex)
    return [v / s for v in ex]

def attention(X, Wq, Wk, Wv):
    Q = matmul(X, Wq)
    K = matmul(X, Wk)
    V = matmul(X, Wv)
    d_k = len(Q[0])
    scores = matmul(Q, transpose(K))          # (N × N) raw similarities
    scale = math.sqrt(d_k)
    scaled = [[s / scale for s in row] for row in scores]
    weights = [softmax_row(row) for row in scaled]   # softmax along keys
    return matmul(weights, V), weights

# 4 tokens, embedding dim 4, projection dim 4
random.seed(0)
X  = [[random.gauss(0, 1) for _ in range(4)] for _ in range(4)]
Wq = [[random.gauss(0, 0.5) for _ in range(4)] for _ in range(4)]
Wk = [[random.gauss(0, 0.5) for _ in range(4)] for _ in range(4)]
Wv = [[random.gauss(0, 0.5) for _ in range(4)] for _ in range(4)]

out, A = attention(X, Wq, Wk, Wv)
for row in A:
    print([round(v, 2) for v in row])
print("output[0] =", [round(v, 2) for v in out[0]])`}</CodeBlock>

      <Prose>
        <p>
          Now with NumPy. The whole block collapses to five lines.{' '}
          <code>einsum</code> makes the axis contractions explicit —{' '}
          <code>&quot;nd,md-&gt;nm&quot;</code> is literally &ldquo;for each
          (n, m) pair, sum over the <code>d</code> axis,&rdquo; which{' '}
          <em>is</em> the definition of <code>Q · Kᵀ</code>. Every guest
          scoring every other guest, one index at a time, but all at once.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — numpy · attention_numpy.py">{`import numpy as np

def scaled_dot_product_attention(X, Wq, Wk, Wv, mask=None):
    Q = X @ Wq                                              # (N, d_k)
    K = X @ Wk                                              # (N, d_k)
    V = X @ Wv                                              # (N, d_v)
    d_k = Q.shape[-1]

    scores = np.einsum("nd,md->nm", Q, K) / np.sqrt(d_k)    # (N, N) scaled
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)            # mask BEFORE softmax

    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)          # softmax along keys

    return weights @ V, weights

rng = np.random.default_rng(0)
X  = rng.standard_normal((4, 4))
Wq, Wk, Wv = [rng.standard_normal((4, 4)) * 0.5 for _ in range(3)]

out, A = scaled_dot_product_attention(X, Wq, Wk, Wv)
print("rows of A sum to 1:", np.allclose(A.sum(axis=-1), 1.0))
print("A shape:", A.shape, " out shape:", out.shape)`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'triple-nested loop for Q · Kᵀ',
            right: 'np.einsum("nd,md->nm", Q, K)',
            note: 'axis labels spell out the contraction — no indexing off-by-ones',
          },
          {
            left: 'per-row softmax with manual max subtract',
            right: 'exp(scores - scores.max(-1, keepdims=True))',
            note: 'same numerical-stability trick, broadcast over all rows at once',
          },
          {
            left: 'math.sqrt(d_k) at each scale',
            right: 'np.sqrt(d_k)',
            note: 'identical scalar, applied to the whole score matrix by broadcast',
          },
        ]}
      />

      <Prose>
        <p>
          And now PyTorch. In real code you never handwrite attention — you
          call <code>F.scaled_dot_product_attention</code>, which dispatches
          to FlashAttention or memory-efficient kernels depending on your
          hardware. Or for a full multi-head block,{' '}
          <code>nn.MultiheadAttention</code>. What used to be 20 lines of
          NumPy is a single call with a causal flag.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · attention_pytorch.py"
        output={`manual == sdpa: True
out shape: torch.Size([2, 4, 4])`}
      >{`import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, N, d_k = 2, 4, 4                                # batch, sequence, head-dim

X = torch.randn(B, N, d_k)
Wq = torch.randn(d_k, d_k) * 0.5
Wk = torch.randn(d_k, d_k) * 0.5
Wv = torch.randn(d_k, d_k) * 0.5

Q, K, V = X @ Wq, X @ Wk, X @ Wv

# (1) Handwritten, for comparison
scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
A = F.softmax(scores, dim=-1)                      # along keys
manual_out = A @ V

# (2) Built-in — dispatches to FlashAttention when available
sdpa_out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)

print("manual == sdpa:", torch.allclose(manual_out, sdpa_out, atol=1e-6))
print("out shape:", sdpa_out.shape)`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'Q = X @ Wq; K = X @ Wk; V = X @ Wv',
            right: 'same, tracked for autograd, runs on GPU',
            note: '@ is identical; tensors carry grad',
          },
          {
            left: 'np.einsum + manual softmax + mask',
            right: 'F.scaled_dot_product_attention(Q, K, V, is_causal=True)',
            note: 'one call — dispatches to FlashAttention, memory-efficient kernels, or math impl',
          },
          {
            left: 'custom block for each head, concatenate',
            right: 'nn.MultiheadAttention(embed_dim, num_heads)',
            note: 'production-ready multi-head with all the bookkeeping done for you',
          },
        ]}
      />

      {/* ── Memory cost callout ─────────────────────────────────── */}
      <Callout variant="warn" title="the O(N²) wall">
        The score matrix is <code>N × N</code>. A dinner party with four
        guests needs sixteen glances. A banquet of 4k guests needs{' '}
        <code>16M</code> floats per head per layer — big but workable. For
        128k guests it&apos;s <code>16B</code> floats per head per layer —
        and you have dozens of heads across dozens of layers. Attention&apos;s
        memory and compute both scale as <code>O(N² · d)</code>, and{' '}
        <em>that</em> is why long-context transformers are hard. The entire
        research programs of <strong>FlashAttention</strong> (tile the
        matmul, never materialize the full <code>N × N</code>),{' '}
        <strong>linear attention</strong> (approximate softmax with a kernel
        to get <code>O(N · d²)</code>), and <strong>sparse attention</strong>{' '}
        (only attend to a subset of keys) exist to punch through this wall.
        You&apos;ll meet them in later lessons.
      </Callout>

      {/* ── Causal masking ──────────────────────────────────────── */}
      <Prose>
        <p>
          One more piece — <KeyTerm>causal masking</KeyTerm>. A language model
          generating text can&apos;t be allowed to peek at future tokens while
          predicting the next one. At our dinner table: guest <code>t</code>{' '}
          is only allowed to listen to guests who&apos;ve already spoken.
          Letting them eavesdrop on guests who haven&apos;t arrived yet would
          be training-time cheating of the most embarrassing kind. The fix is
          a mask applied to the score matrix <em>before</em> softmax: set
          every entry above the diagonal to <code>-∞</code>, so after softmax
          those weights are exactly <code>0</code> and never leak into the
          output.
        </p>
      </Prose>

      <MathBlock caption="causal mask — prevent queries from seeing future keys">
{`         keys →
         k0   k1   k2   k3
q0  [   s00  -∞   -∞   -∞  ]
q1  [   s10  s11  -∞   -∞  ]     ← token t can only attend to tokens ≤ t
q2  [   s20  s21  s22  -∞  ]
q3  [   s30  s31  s32  s33 ]

after softmax along keys:
         k0   k1   k2   k3
q0  [  1.00  0    0    0   ]
q1  [  0.43  0.57 0    0   ]
q2  [  0.29  0.35 0.36 0   ]
q3  [  0.21  0.26 0.27 0.26]`}
      </MathBlock>

      <Prose>
        <p>
          This is what <code>is_causal=True</code> does inside PyTorch&apos;s
          SDPA — it fills the upper triangle with <code>-∞</code> before the
          softmax runs. In a decoder-only model like GPT, <em>every</em>{' '}
          attention layer is causal. In an encoder like BERT, none of them
          are — everyone at the table speaks at once and hears everyone else.
          In an encoder-decoder like T5, the encoder is bidirectional and the
          decoder is causal with an extra cross-attention step.
        </p>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Softmax on the wrong axis:</strong> the
          attention softmax goes over <em>keys</em> (the last dim of a{' '}
          <code className="text-dark-text-primary">Q Kᵀ</code> matrix). Softmax over queries
          instead, and rows of A no longer sum to 1 — the output is silently nonsense and
          the loss still decreases a little, so you won&apos;t notice until validation
          tanks. Always <code className="text-dark-text-primary">dim=-1</code>, always
          triple-checked.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting <code>√d_k</code>:</strong> the
          network will still train, slowly, and with a softmax so peaked that gradients flow
          through one key at a time. Everything converges worse. This bug is invisible on
          toy <code>d_k = 4</code> sequences and devastating at <code>d_k = 64</code>+.
        </p>
        <p>
          <strong className="text-term-amber">Masking after softmax:</strong> if you zero
          out masked positions after the softmax, the remaining weights no longer sum to 1
          and you&apos;ve re-introduced a tiny leak from the masked side through the
          normalization. The mask must be applied by setting scores to{' '}
          <code className="text-dark-text-primary">-∞</code>{' '}
          <em>before</em> softmax. Every time. No exceptions.
        </p>
        <p>
          <strong className="text-term-amber">Mask shape mismatch:</strong> SDPA expects the
          mask to broadcast against{' '}
          <code className="text-dark-text-primary">(B, H, N, N)</code>. A{' '}
          <code className="text-dark-text-primary">(N, N)</code> bool mask works; a{' '}
          <code className="text-dark-text-primary">(B, N)</code> padding mask does not
          (that&apos;s <code>key_padding_mask</code> on{' '}
          <code className="text-dark-text-primary">nn.MultiheadAttention</code>). Read the
          docstring every time — the conventions shift between APIs.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Build causal self-attention from scratch, verify against PyTorch">
        <p>
          Implement a single-head causal self-attention block in PyTorch using only{' '}
          <code>@</code>, <code>F.softmax</code>, and <code>torch.triu</code> — no calls to{' '}
          <code>F.scaled_dot_product_attention</code> or <code>nn.MultiheadAttention</code>.
          Your signature:{' '}
          <code>attention(x, Wq, Wk, Wv) -&gt; (out, weights)</code> for input{' '}
          <code>x</code> of shape <code>(B, N, d)</code>.
        </p>
        <p className="mt-2">
          Build a causal mask with <code>torch.triu(torch.ones(N, N), diagonal=1).bool()</code>
          . Apply it with <code>scores.masked_fill_(mask, float(&apos;-inf&apos;))</code>{' '}
          before the softmax. Divide by <code>√d_k</code>.
        </p>
        <p className="mt-2">
          Now verify. Run the same <code>(x, Wq, Wk, Wv)</code> through{' '}
          <code>F.scaled_dot_product_attention(Q, K, V, is_causal=True)</code> and assert
          that your output matches to <code>1e-6</code>. If it doesn&apos;t, the culprit is
          almost always (a) wrong softmax axis, (b) mask applied after softmax, or (c)
          missing the <code>√d_k</code>.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: print the attention weights for a small <code>N = 6</code> sequence.
          Confirm the upper triangle is exactly 0 and every row sums to exactly 1.
        </p>
      </Challenge>

      {/* ── Closing + teaser ────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Attention is a dinner party
          rendered in linear algebra: every guest turns to every other guest
          with a query, a key, and a value — three learned projections, one
          scaled dot product, one softmax, one weighted sum of values. The{' '}
          <code>√d_k</code> keeps the softmax from saturating at reasonable{' '}
          <code>d_k</code>. Softmax goes along keys. Masks go in before
          softmax. The whole room is permutation-equivariant, which is why
          we&apos;ll staple positional encodings to the input. And the{' '}
          <code>O(N²)</code> memory cost — every guest scoring every other
          guest — is the single biggest bottleneck in modern transformer
          scaling, which is why your next six papers will all be about
          sidestepping it.
        </p>
        <p>
          <strong>Next up — Multi-Headed Self Attention.</strong> One
          conversation at one table is fine — but the room would be richer if
          every guest could have several conversations at once, each
          specializing in a different angle. A pronoun probably wants to find
          its antecedent <em>and</em> its syntactic role <em>and</em> its
          verb, simultaneously. The answer is <KeyTerm>multi-headed self
          attention</KeyTerm>: run several attention heads in parallel with
          different <code>W_Q / W_K / W_V</code>, concatenate their outputs,
          and project down. We&apos;ll derive why <code>n_heads</code>{' '}
          matters, what each head tends to specialize in, and how a 12-head
          attention block is still a single matmul if you squint at it right.
        </p>
      </Prose>

      <WhatNext currentSlug="self-attention" />

      <Quiz
        question={
          <>
            Why does scaled dot-product attention divide by{' '}
            <code>√d_k</code> before the softmax?
          </>
        }
        options={[
          {
            text: 'To normalize the logits so the softmax doesn\u2019t saturate into one-hot vectors when d_k is large.',
            correct: true,
            explain:
              'At high d_k the dot product Q·Kᵀ has variance ∝ d_k, so a few logits dominate and softmax collapses to (nearly) one-hot — which zeroes the gradients everywhere else. Dividing by √d_k rescales the variance back to ~1.',
          },
          {
            text: 'It cancels the bias term from the W_Q and W_K projections.',
            explain:
              'The scaling factor is about variance, not bias — and the projections may not even have bias terms. Remove the scale and the layer still runs; it just trains badly.',
          },
          {
            text: 'Because softmax requires inputs to sum to 1, and dividing by √d_k enforces that.',
            explain:
              'Softmax normalizes its outputs, not its inputs — it handles any real-valued logits. The scale is about keeping those logits in a range where gradients survive.',
          },
          {
            text: 'For numerical stability — otherwise exp() overflows on GPU.',
            explain:
              'Production softmax already shifts by max(logit) for overflow safety. The √d_k factor is motivated by learning dynamics, not floating-point range.',
          },
        ]}
      />

      <References
        items={[
          {
            title: 'Attention Is All You Need',
            author: 'Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin',
            venue: 'NeurIPS 2017 — the paper that introduced the transformer',
            url: 'https://arxiv.org/abs/1706.03762',
            tags: ['paper'],
          },
          {
            title:
              'Neural Machine Translation by Jointly Learning to Align and Translate',
            author: 'Bahdanau, Cho, Bengio',
            venue: 'ICLR 2015 — attention in NMT, the seed of the whole idea',
            url: 'https://arxiv.org/abs/1409.0473',
            tags: ['paper'],
          },
          {
            title: 'Let\u2019s build GPT: from scratch, in code, spelled out',
            author: 'Andrej Karpathy',
            venue: 'YouTube / nanoGPT — the clearest code walkthrough of causal self-attention',
            url: 'https://github.com/karpathy/nanoGPT',
            tags: ['code'],
          },
          {
            title: 'Dive into Deep Learning — Chapter 11: Attention Mechanisms and Transformers',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai §11.1–11.3 — textbook derivation with runnable code',
            url: 'https://d2l.ai/chapter_attention-mechanisms-and-transformers/',
            tags: ['book'],
          },
          {
            title:
              'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
            author: 'Dao, Fu, Ermon, Rudra, Ré',
            venue: 'NeurIPS 2022 — how production engines dodge the O(N²) memory wall',
            url: 'https://arxiv.org/abs/2205.14135',
            tags: ['paper'],
          },
        ]}
      />
    </div>
  )
}
