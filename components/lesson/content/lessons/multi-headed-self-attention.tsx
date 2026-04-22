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
} from '../primitives'
import MultiHeadSplit from '../widgets/MultiHeadSplit'
import HeadSpecialization from '../widgets/HeadSpecialization'

export default function MultiHeadedSelfAttentionLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="multi-headed-self-attention" />

      {/* ── Opening: return to the dinner party ──────────────────── */}
      <Prose>
        <p>
          Last lesson we sat every token at one long table. That&apos;s{' '}
          <NeedsBackground slug="self-attention">self-attention</NeedsBackground>
          {' '}— one big dinner party, every guest talking to every other guest, one
          conversation for the whole room. It&apos;s a beautiful mechanism.
          It&apos;s also, on any real sentence, one table too few.
        </p>
        <p>
          Think about what&apos;s actually happening at a single table. Every
          guest asks one question. <em>One.</em> The query vector for each token
          runs through a single <code>W_Q</code>, every other token offers
          itself through a single <code>W_K</code>, and the score{' '}
          <code>q · k</code> captures exactly one notion of &ldquo;relevant.&rdquo;
          Whatever flavor of relevance that table happens to land on, every
          token is stuck with it.
        </p>
        <p>
          Take the sentence <em>&ldquo;the black cat sat on the mat.&rdquo;</em>{' '}
          The token <code>cat</code> wants to talk to <code>sat</code> because
          that&apos;s its verb. It also wants to talk to <code>black</code>{' '}
          because that&apos;s its adjective. It also wants to talk to{' '}
          <code>mat</code> because that&apos;s where the action lands. Three
          different relationships — syntactic, modifier, argument — and one
          table can only host one conversation at a time. Force a single pair
          of Q/K weights to capture all three and you get a compromise that
          captures none of them cleanly.
        </p>
        <p>
          So: throw a bigger party. Set up several tables in parallel, each
          with its own rules for who finds whom interesting. Table 1 cares
          about subject-verb links. Table 2 cares about coreference — who
          does &ldquo;it&rdquo; refer to? Table 3 cares about positional
          patterns — the token two places to my left. Every guest sits briefly
          at every table, whispers to the others present, and collects what
          they heard. At the end, each guest concatenates their eight whispers
          into one coherent update and walks back to the main floor. That is{' '}
          <KeyTerm>multi-head attention</KeyTerm>, and this lesson is the
          mechanics of how many tables, what each table listens for, and how
          the whispers get sewn back together.
        </p>
      </Prose>

      {/* ── Core formula ─────────────────────────────────────────── */}
      <MathBlock caption="multi-head attention — the whole formula">
{`MultiHead(X)   =   Concat(head₁, head₂, …, head_H) · W_O

where  head_i  =  Attention(X·W_Q^i,  X·W_K^i,  X·W_V^i)

and    Attention(Q, K, V)  =  softmax( Q·Kᵀ / √d_head ) · V`}
      </MathBlock>

      <Prose>
        <p>
          Here&apos;s the part most tutorials skip. You do <em>not</em> give
          each table its own full-width <code>d_model</code> subspace to play
          in — that would multiply the parameter count by <em>H</em> and nobody
          wants that. You <em>split</em> the existing <code>d_model</code>{' '}
          into <em>H</em> slices of width <code>d_head = d_model / H</code>,
          and each head gets one slice as its private subspace. A head sees a
          narrower view; there are just <em>H</em> of them working in parallel,
          so the total dimensionality is preserved. Same parameter budget as a
          single-head model of the same width. The whole cost of having
          multiple tables is bookkeeping.
        </p>
      </Prose>

      {/* ── Widget 1: Multi-Head Split ──────────────────────────── */}
      <MultiHeadSplit />

      <Prose>
        <p>
          The widget is the head-split reveal in pictures. Start with{' '}
          <code>d_model = 64</code> and spin it up into <em>H = 8</em> heads of{' '}
          <code>d_head = 8</code>. Input of shape <code>(B, N, 64)</code> runs
          through the Q/K/V projections — still <code>(B, N, 64)</code> coming
          out — then reshapes and transposes into{' '}
          <code>(B, H, N, d_head) = (B, 8, N, 8)</code>. That reshape{' '}
          <em>is</em> the split: each head now owns its own 8-dimensional
          subspace, carved out of the original 64. From there, eight tables
          run attention in parallel — eight separate <code>N × N</code>{' '}
          score matrices, each computed only over its head&apos;s slice of the
          representation.
        </p>
        <p>
          That <code>(B, H, N, d_head)</code> layout is the whole game. The{' '}
          <code>H</code> axis is a <em>batch</em> dimension from
          attention&apos;s point of view: the op has no idea it&apos;s doing
          eight things at once, it just sees <code>B·H</code> independent
          attention problems and dispatches them to the GPU. Multi-head is
          embarrassingly parallel by design — which is why you can crank{' '}
          <em>H</em> up without the clock time tracking it linearly.
        </p>
      </Prose>

      <Personify speaker="Head">
        I am one of eight at my own table. I see 8 dimensions out of 64, and I
        only have to be good at one kind of relationship — maybe subject-verb,
        maybe coreference, maybe &ldquo;the token two positions to my left.&rdquo;
        I don&apos;t negotiate with my siblings. We all specialize on the same
        input in our own subspaces and hand our whispers to the integrator. I
        am a specialist, on purpose.
      </Personify>

      {/* ── Per-head math ───────────────────────────────────────── */}
      <MathBlock caption="per-head mechanics — all heads in parallel">
{`For i = 1 … H:

    Qᵢ  =  X · W_Qⁱ          shape  (B, N, d_head)
    Kᵢ  =  X · W_Kⁱ          shape  (B, N, d_head)
    Vᵢ  =  X · W_Vⁱ          shape  (B, N, d_head)

    Aᵢ  =  softmax( Qᵢ · Kᵢᵀ  /  √d_head )      shape  (B, N, N)
    Hᵢ  =  Aᵢ · Vᵢ                               shape  (B, N, d_head)

Concat along the last axis:

    H   =  [H₁ | H₂ | … | H_H]                  shape  (B, N, d_model)

Final projection:

    out =  H · W_O                               shape  (B, N, d_model)`}
      </MathBlock>

      <Prose>
        <p>
          The <code>√d_head</code> inside the{' '}
          <NeedsBackground slug="softmax">softmax</NeedsBackground> is the same
          scaled-dot-product trick from last lesson — but notice the number
          under the square root. It&apos;s <code>d_head</code>, not{' '}
          <code>d_model</code>. Each head lives in its own small subspace, so
          its dot products concentrate around a variance of <code>d_head</code>,
          not <code>d_model</code>. Scaling by <code>√d_model</code> here would
          over-shrink them and collapse the softmax into a near-uniform mush;{' '}
          <code>√d_head</code> is the right whistle.
        </p>
      </Prose>

      {/* ── Widget 2: Head Specialization ───────────────────────── */}
      <HeadSpecialization />

      <Prose>
        <p>
          Now the payoff. Four tables at the same party, same sentence on the
          menu, and each table&apos;s attention map lights up a completely
          different conversation. One head locks onto syntactic dependency —
          nouns leaning toward their verbs. Another tracks position — every
          token whispering to the token immediately before it. A third runs
          coreference — pronouns reaching back to their antecedents. The
          fourth is doing something harder to name but clearly structured.
          Each head specializes in its own relational primitive.
        </p>
        <p>
          Nobody told head 3 to handle coreference. Nobody hand-programmed
          table 1 to follow subject-verb. The only training signal is
          &ldquo;predict the next token,&rdquo; and the heads settle into a
          division of labor because that&apos;s the cheapest way to drive the
          loss down. Eight small experts, each focused on one slice of
          relational structure, turn out to be strictly easier for gradient
          descent to sculpt than one giant generalist. Specialization is what
          gradient descent <em>wants</em> to do, given the chance.
        </p>
        <p>
          This is also why a single head of width <code>d_model</code> is{' '}
          <em>computationally equivalent</em> to multi-head of the same total
          width but <em>optimizationally worse</em>. Same parameter budget on
          both sides of the ledger — but only the multi-head version gives
          gradient descent clean separate subspaces to carve specialists into.
          Cram everything into one table and the shared <code>W_Q</code>,{' '}
          <code>W_K</code>, <code>W_V</code> has to encode every pattern at
          once, which it can, just badly.
        </p>
      </Prose>

      <Personify speaker="Concatenation + W_O">
        I am the integrator at the end of the night. Eight specialists walk
        back from their tables and hand me their whispers — syntax from table
        1, coreference from table 2, position from table 3, five more. I
        concatenate them end-to-end into a single long vector and project the
        whole thing back into model space with <code>W_O</code>. My job is to
        fuse eight views of each token into one coherent representation the
        next layer can consume. Without me you have eight experts shouting
        past each other instead of a conversation.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations, same mechanism, three layers of abstraction.
          First NumPy with every reshape written out so you can see the split
          happen. Then PyTorch&apos;s <code>nn.MultiheadAttention</code>{' '}
          one-liner with a causal mask — the prototype-speed option. Then a
          custom module — the kind you actually ship when you need fine
          control over shapes and projections.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy, manual reshape · mha_numpy.py"
        output={`X      : (2, 5, 64)
Q_h    : (2, 8, 5, 8)     # (B, H, N, d_head)
scores : (2, 8, 5, 5)     # one N×N attention matrix per head
output : (2, 5, 64)       # back to (B, N, d_model)`}
      >{`import numpy as np

B, N, d_model, H = 2, 5, 64, 8
d_head = d_model // H              # 64 / 8 = 8

rng = np.random.default_rng(0)
X   = rng.standard_normal((B, N, d_model))
W_Q = rng.standard_normal((d_model, d_model)) * 0.1
W_K = rng.standard_normal((d_model, d_model)) * 0.1
W_V = rng.standard_normal((d_model, d_model)) * 0.1
W_O = rng.standard_normal((d_model, d_model)) * 0.1

# 1. Linear projections — still (B, N, d_model)
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# 2. Reshape + transpose into (B, H, N, d_head) — the critical step
def split_heads(x):
    x = x.reshape(B, N, H, d_head)        # (B, N, H, d_head)
    return x.transpose(0, 2, 1, 3)        # (B, H, N, d_head)

Q_h, K_h, V_h = split_heads(Q), split_heads(K), split_heads(V)

# 3. Scaled dot-product attention, per head, in parallel via broadcasting
scores = Q_h @ K_h.transpose(0, 1, 3, 2) / np.sqrt(d_head)   # (B, H, N, N)
attn   = np.exp(scores - scores.max(-1, keepdims=True))
attn  /= attn.sum(-1, keepdims=True)                          # softmax
head_out = attn @ V_h                                         # (B, H, N, d_head)

# 4. Merge heads back — transpose and reshape to (B, N, d_model)
merged = head_out.transpose(0, 2, 1, 3).reshape(B, N, d_model)

# 5. Final output projection
output = merged @ W_O

print(f"X      : {X.shape}")
print(f"Q_h    : {Q_h.shape}     # (B, H, N, d_head)")
print(f"scores : {scores.shape}     # one N×N attention matrix per head")
print(f"output : {output.shape}       # back to (B, N, d_model)")`}</CodeBlock>

      <Prose>
        <p>
          The <code>split_heads</code> function — reshape, then transpose — is
          the load-bearing trick. You can&apos;t just reshape{' '}
          <code>(B, N, d_model)</code> straight into{' '}
          <code>(B, H, N, d_head)</code>, because the memory layout would
          interleave heads across tokens in the wrong order and you&apos;d
          silently compute attention on the wrong subspace slices. The correct
          order is reshape to <code>(B, N, H, d_head)</code>, <em>then</em>{' '}
          transpose <code>H</code> with <code>N</code>. Flip those two and
          every table at the party is suddenly talking to the wrong guests.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch nn.MultiheadAttention with causal mask · mha_torch.py"
        output={`out shape : torch.Size([2, 5, 64])
attn shape: torch.Size([2, 5, 5])   # averaged over heads by default
causal check — row 0 attends only to col 0: tensor(True)`}
      >{`import torch
import torch.nn as nn

B, N, d_model, H = 2, 5, 64, 8
x = torch.randn(B, N, d_model)

# batch_first=True so shapes match everything else we've been writing.
mha = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=H,
    batch_first=True,      # IMPORTANT — default is (N, B, d_model), which is insane
)

# Causal mask: upper-triangular of -inf means "don't look forward".
causal = torch.triu(torch.full((N, N), float("-inf")), diagonal=1)

out, attn = mha(x, x, x, attn_mask=causal, need_weights=True)

print(f"out shape : {out.shape}")
print(f"attn shape: {attn.shape}   # averaged over heads by default")
print(f"causal check — row 0 attends only to col 0: {(attn[0, 0, 1:] == 0).all()}")`}</CodeBlock>

      <Prose>
        <p>
          PyTorch&apos;s built-in compresses everything above into one call.
          Watch the flags though. <code>batch_first=True</code> is
          non-negotiable — the default is <code>(N, B, d_model)</code>, which
          no sane code path actually wants, and every transformer student has
          been bitten by this at least once. The mask is additive:{' '}
          <code>-inf</code> positions become exactly zero after softmax.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — custom MHA module, the real-codebase version · mha_custom.py"
        output={`custom   : torch.Size([2, 5, 64])
reference: torch.Size([2, 5, 64])
max abs diff (after weight sync): 3.2e-07`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """The version you'd actually write — one big QKV projection, explicit split."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model  = d_model
        self.H        = num_heads
        self.d_head   = d_model // num_heads

        # Fused projection — one matmul is faster than three on GPU.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, _ = x.shape

        # (B, N, 3*d_model) → split into 3 × (B, N, d_model)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, N, d_model) → (B, H, N, d_head)
        def split(t):
            return t.view(B, N, self.H, self.d_head).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)

        # Scaled dot-product attention — F.scaled_dot_product_attention is fastest
        # in PyTorch 2.x (uses Flash-Attention kernels where available).
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0
        )                                             # (B, H, N, d_head)

        # Merge heads → (B, N, d_model)
        out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(self.dropout(out))


# Sanity check: does our MHA match nn.MultiheadAttention on the same weights?
torch.manual_seed(0)
mha = MultiHeadAttention(d_model=64, num_heads=8)
ref = nn.MultiheadAttention(64, 8, batch_first=True, bias=False)

# Copy weights across so the two should produce identical output.
with torch.no_grad():
    ref.in_proj_weight.copy_(mha.qkv_proj.weight)
    ref.out_proj.weight.copy_(mha.out_proj.weight)

x = torch.randn(2, 5, 64)
a = mha(x)
b, _ = ref(x, x, x, need_weights=False)

print(f"custom   : {a.shape}")
print(f"reference: {b.shape}")
print(f"max abs diff (after weight sync): {(a - b).abs().max().item():.1e}")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch built-in → custom module"
        rows={[
          {
            left: 'Q = X @ W_Q; K = X @ W_K; V = X @ W_V',
            right: 'qkv = self.qkv_proj(x); q,k,v = qkv.chunk(3, -1)',
            note: 'one fused matmul replaces three — faster on GPU, same math',
          },
          {
            left: 'x.reshape(B, N, H, d_head).transpose(0, 2, 1, 3)',
            right: 't.view(B, N, H, d_head).transpose(1, 2)',
            note: 'same reshape-then-transpose dance, different API',
          },
          {
            left: 'manual softmax(QKᵀ/√d_head) @ V',
            right: 'F.scaled_dot_product_attention(q, k, v, mask)',
            note: "PyTorch 2's fused kernel — Flash-Attention when hardware allows",
          },
          {
            left: 'output = merged @ W_O',
            right: 'self.out_proj(out)',
            note: 'final projection — concatenate-and-project-back in one line',
          },
        ]}
      />

      {/* ── Real-world configs ──────────────────────────────────── */}
      <Callout variant="insight" title="head counts in the wild">
        <p>
          The trade-off between how many tables you set up and how wide each
          one is is a real hyperparameter. Production models have converged
          onto a short list of stable recipes:
        </p>
        <ul>
          <li>
            <strong>GPT-2 small</strong>: <code>H = 12</code>, <code>d_head = 64</code>,{' '}
            <code>d_model = 768</code>.
          </li>
          <li>
            <strong>GPT-3 175B</strong>: <code>H = 96</code>, <code>d_head = 128</code>,{' '}
            <code>d_model = 12288</code>.
          </li>
          <li>
            <strong>LLaMA 7B</strong>: <code>H = 32</code>, <code>d_head = 128</code>,{' '}
            <code>d_model = 4096</code>.
          </li>
          <li>
            <strong>BERT-base</strong>: <code>H = 12</code>, <code>d_head = 64</code>,{' '}
            <code>d_model = 768</code>.
          </li>
        </ul>
        <p>
          <code>d_head = 64</code> or <code>128</code> is the sweet spot.
          Skinnier and the per-head subspace gets too narrow to carry anything
          a head could usefully specialize on. Fatter and you drift back
          toward single-head behavior — fewer tables, fewer specialists, more
          compromise.
        </p>
      </Callout>

      <Callout variant="note" title="parameter equivalence — a small proof">
        Single-head attention with <code>d_model = 64</code> has three
        projection matrices of shape <code>(64, 64)</code>, for{' '}
        <code>3 × 64 × 64 = 12,288</code> parameters (plus <code>W_O</code>).
        Eight-head attention with <code>d_head = 8</code> has eight sets of
        three <code>(64, 8)</code> matrices, for{' '}
        <code>8 × 3 × 64 × 8 = 12,288</code>. Identical parameter count. The
        only thing that changed is how those parameters are organized — eight
        narrow tables with their own subspaces instead of one wide generalist.
        But the loss landscape is dramatically friendlier when gradients can
        sculpt each head independently.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">d_model must be divisible by num_heads.</strong>{' '}
          <code>d_model=64, H=7</code> gives <code>d_head=9.14…</code>, which
          is not a tensor shape. PyTorch and most frameworks throw an
          assertion. Pick <code>H</code> ∈ <code>{'{1, 2, 4, 8, 16, 32, …}'}</code>{' '}
          and move on.
        </p>
        <p>
          <strong className="text-term-amber">(B, N, d_model) vs (B, H, N, d_head).</strong>{' '}
          Every shape bug in transformer code is really a confusion between
          these two layouts. Keep a mental note: anything before the reshape
          or after the merge is <code>(B, N, d_model)</code>; everything in
          between is <code>(B, H, N, d_head)</code>. Print <code>.shape</code>{' '}
          liberally while debugging.
        </p>
        <p>
          <strong className="text-term-amber">Mask shape broadcasting.</strong>{' '}
          A causal mask is shape <code>(N, N)</code> but attention scores are{' '}
          <code>(B, H, N, N)</code>. The mask broadcasts over the leading{' '}
          <code>(B, H)</code> dimensions automatically — <em>if</em> the
          dimensions are in the right order. Passing <code>(H, N, N)</code> or{' '}
          <code>(B, N, N)</code> instead of <code>(N, N)</code> (or explicit{' '}
          <code>(B, H, N, N)</code>) broadcasts wrong and silently masks the
          wrong positions.
        </p>
        <p>
          <strong className="text-term-amber">nn.MultiheadAttention is NOT batch-first by default.</strong>{' '}
          It expects <code>(N, B, d_model)</code>. This is a PyTorch
          historical artifact that has broken more transformers than any other
          single API choice. Always pass <code>batch_first=True</code>.
        </p>
        <p>
          <strong className="text-term-amber">out_proj is not optional.</strong>{' '}
          Dropping <code>W_O</code> looks harmless — you&apos;re already back
          at <code>d_model</code> after concat — but without it the head
          outputs are just <em>concatenated</em>, never <em>mixed</em>.{' '}
          <code>W_O</code> is how information from table 3 ends up influencing
          the component table 7 produced. It&apos;s what turns eight parallel
          monologues into one integrated representation.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Reimplement MHA and verify against PyTorch">
        <p>
          Write a <code>MultiHeadAttention</code> module from scratch with{' '}
          <code>d_model=32, num_heads=4</code>. Use <em>separate</em>{' '}
          <code>W_Q</code>, <code>W_K</code>, <code>W_V</code>,{' '}
          <code>W_O</code> linear layers (no fused QKV). Initialize all four
          with <code>torch.manual_seed(42)</code>.
        </p>
        <p className="mt-2">
          Then instantiate{' '}
          <code>nn.MultiheadAttention(32, 4, batch_first=True, bias=False)</code>,
          copy your weights into its <code>in_proj_weight</code> (stacked
          Q/K/V) and <code>out_proj.weight</code>, and run both modules on the
          same random input of shape <code>(4, 10, 32)</code>.
        </p>
        <p className="mt-2">
          <strong>Assert</strong>{' '}
          <code>torch.allclose(your_output, reference_output, atol=1e-5)</code>.
          If it fails, the bug is almost always in your reshape order or the
          way you stacked the weights.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: add a causal mask and re-verify. Bonus 2: swap your manual
          softmax for <code>F.scaled_dot_product_attention</code> and confirm
          the output is still bit-identical to{' '}
          <code>nn.MultiheadAttention</code>.
        </p>
      </Challenge>

      {/* ── Closing + teaser ────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Multi-head attention
          isn&apos;t more parameters than single-head of the same width —
          it&apos;s the <em>same</em> parameters reorganized so gradient
          descent can carve out specialists on separate tables. Each head
          projects Q/K/V into its own narrow subspace, runs attention there,
          and the integrator concatenates every head&apos;s output and
          projects it back into model space with <code>W_O</code>. The shape
          dance — <code>(B, N, d_model) → (B, H, N, d_head) → (B, N, d_model)</code>{' '}
          — is the single most load-bearing piece of bookkeeping in the entire
          transformer stack. Once you&apos;ve written it yourself, transformer
          code stops being intimidating.
        </p>
        <p>
          <strong>Next up — the Transformer Block.</strong> Multi-head
          attention gets information moving between positions — that&apos;s
          one half of a transformer layer. But a layer that only mixes
          information without also <em>processing</em> each position is half a
          machine. Attention tells each token <em>who</em> to listen to; it
          doesn&apos;t give it space to think about what it heard. The{' '}
          <code>transformer-block</code> lesson wraps our multi-head
          attention in the other half — a per-token feedforward MLP — plus
          residual connections and{' '}
          <NeedsBackground slug="word-embeddings">embeddings</NeedsBackground>
          -scale normalization that keep the stack trainable at depth. Few new
          ideas, one very famous diagram, and a whole lot of load-bearing{' '}
          <code>+ x</code>.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Attention Is All You Need',
            author: 'Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin',
            venue: 'NeurIPS 2017 — the original multi-head attention paper',
            url: 'https://arxiv.org/abs/1706.03762',
          },
          {
            title: 'What Does BERT Look At? An Analysis of BERT\'s Attention',
            author: 'Clark, Khandelwal, Levy, Manning',
            venue: 'BlackboxNLP 2019 — empirical study of head specialisation',
            url: 'https://arxiv.org/abs/1906.04341',
          },
          {
            title: 'Are Sixteen Heads Really Better Than One?',
            author: 'Michel, Levy, Neubig',
            venue: 'NeurIPS 2019 — when and why you can prune heads',
            url: 'https://arxiv.org/abs/1905.10650',
          },
          {
            title: 'Dive into Deep Learning — §11.5 Multi-Head Attention',
            author: 'Zhang, Lipton, Li, Smola',
            url: 'https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html',
          },
          {
            title: 'A Mathematical Framework for Transformer Circuits',
            author: 'Elhage et al. (Anthropic)',
            year: 2021,
            venue: 'interpretability — heads as composable circuits',
            url: 'https://transformer-circuits.pub/2021/framework/index.html',
          },
        ]}
      />
    </div>
  )
}
