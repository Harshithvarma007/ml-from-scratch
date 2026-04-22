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
import GQAGrouping from '../widgets/GQAGrouping'              // visualize 32 query heads grouped into 8 key/value heads; each group of 4 queries shares one K/V pair
import GQAvsFullMHA from '../widgets/GQAvsFullMHA'          // memory comparison chart: MHA (one K/V per head) vs GQA (one K/V per group) vs MQA (one K/V total) — bar chart of cache size per context length

// Signature anchor: the carpool. Multi-head attention gave every query its
// own K/V car — expensive at scale. Multi-query goes the other extreme:
// all queries share one K/V (cheap but too cramped). GQA is carpooling —
// four queries share one K/V, then the next four share the next one. Same
// ride, fewer cars on the road. Returned to at the opening (traffic jam
// of per-head K/V), the group-size-dial reveal (MQA = one group; MHA =
// all groups), and the "what you give up" section.
export default function GroupedQueryAttentionLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="grouped-query-attention" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Picture rush hour. Every single car on the road has exactly one
          person in it — commuter, steering wheel, four empty seats. The
          freeway grinds. Nobody is going anywhere fast, and none of it is a
          compute problem. It&apos;s a <em>road capacity</em> problem. There
          are too many cars.
        </p>
        <p>
          That&apos;s your LLM at inference time. Every token you generate
          reads the entire past, and that past lives in the{' '}
          <NeedsBackground slug="kv-cache">KV cache</NeedsBackground> — the
          stack of key and value tensors the model has built up one token at
          a time. In{' '}
          <NeedsBackground slug="multi-headed-self-attention">
            multi-head attention
          </NeedsBackground>{' '}
          with 32 heads, the cache holds 32 separate K tensors and 32
          separate V tensors per layer. One car per query head. Scale that
          to a 70B model with 80 layers and a 32k context, and you are
          staring at tens of gigabytes of VRAM reserved for a single
          conversation — a traffic jam of per-head K/V blocking every lane.
          The math doesn&apos;t care that the user only wanted a haiku.
        </p>
        <p>
          The bottleneck in modern LLM inference is not matrix multiplies.
          It&apos;s <em>moving the KV cache from HBM to the compute units</em>,
          over and over, for every new token. Tokens per second is a
          memory-bandwidth problem. The road is full, and every car you keep
          on it shows up on the GPU bill.
        </p>
        <p>
          So: carpool. Four queries share one K and one V, the next four
          share the next one, and so on. Same ride, a quarter of the cars.
          That&apos;s the whole idea of{' '}
          <KeyTerm>grouped-query attention</KeyTerm>. It&apos;s the
          architectural compromise behind LLaMA-2 70B, Mistral, Mixtral, and
          basically every serious open model trained since 2023. A tiny code
          change. Cache shrinks 8×. The quality loss is in the noise.
        </p>
      </Prose>

      {/* ── The parameter math ──────────────────────────────────── */}
      <Prose>
        <p>
          Start with the three flavors side by side. Let <code>d</code> be
          the model dimension, <code>h</code> the number of query heads,{' '}
          <code>d_h = d/h</code> the per-head dimension, and <code>g</code>{' '}
          the number of KV heads — which is to say, the number of carpool
          groups.
        </p>
      </Prose>

      <MathBlock caption="projection parameters — MHA vs GQA vs MQA">
{`MHA  (Multi-Head):     W_Q ∈ ℝ^(d × d)     W_K ∈ ℝ^(d × d)        W_V ∈ ℝ^(d × d)
                       → 3 · d²  parameters in Q, K, V

GQA  (Grouped, g<h):   W_Q ∈ ℝ^(d × d)     W_K ∈ ℝ^(d × g·d_h)    W_V ∈ ℝ^(d × g·d_h)
                       → d² · (1 + 2g/h)  parameters

MQA  (Multi-Query):    W_Q ∈ ℝ^(d × d)     W_K ∈ ℝ^(d × d_h)      W_V ∈ ℝ^(d × d_h)
                       → d² · (1 + 2/h)   parameters

special cases:   g = h → MHA      g = 1 → MQA      1 < g < h → GQA`}
      </MathBlock>

      <Prose>
        <p>
          GQA is a dial, and <code>g</code> is the dial&apos;s position. Turn
          it all the way up to <code>g = h</code> and every query gets its
          own K/V — that&apos;s MHA, one car per commuter, maximum
          expressivity, maximum road. Turn it all the way down to{' '}
          <code>g = 1</code> and every query rides the same K/V — that&apos;s{' '}
          <KeyTerm>multi-query attention</KeyTerm> (Shazeer, 2019), a single
          minivan for 32 people, minimum cache, noticeable quality hit
          because you&apos;ve crammed everyone into one vehicle. GQA lives
          between. Pick <code>g</code>, and you pick your carpool size.
        </p>
      </Prose>

      {/* ── Widget 1: GQA Grouping ──────────────────────────────── */}
      <Prose>
        <p>
          Here is what that dial looks like visually. 32 query heads on the
          left, lined up like commuters. On the right, the KV heads they
          share — the actual cars on the road. Slide <code>g</code> down
          from 32 and watch queries merge into groups, four commuters
          climbing into the same ride at the LLaMA-2 70B setting.
        </p>
      </Prose>

      <GQAGrouping />

      <Callout variant="note" title="why queries can share K/V but not Q">
        Attention heads learn different <em>query patterns</em> — what to
        look for — more than different <em>key/value representations</em> —
        what to return. Empirically, the variance across heads is
        concentrated in Q. K and V end up pointing at roughly the same
        content from different angles, which is why four queries can share
        a single K/V without much loss. GQA exploits that asymmetry: keep
        every Q, pool the redundant K/V into a shared lane.
      </Callout>

      <Personify speaker="GQA">
        I am the compromise nobody asked for and everybody ships. Full MHA
        is too greedy with the road; MQA gives up too much quality
        squeezing everyone into one car. I give you 90% of MHA&apos;s
        quality at an eighth of its cache. You will not notice me in your
        benchmarks. Your GPU bill will.
      </Personify>

      {/* ── Cache arithmetic for a real model ───────────────────── */}
      <Prose>
        <p>
          Put numbers on it. LLaMA-2 70B has <code>d = 8192</code>,{' '}
          <code>h = 64</code> query heads, <code>d_h = 128</code>, and{' '}
          <code>L = 80</code> layers. A full-MHA version — one car per
          query, 64 cars per layer — would cache, per token, across all
          layers:
        </p>
      </Prose>

      <MathBlock caption="KV cache per token — MHA vs GQA for LLaMA-2 70B">
{`bytes_per_token (MHA)   =  2 · L · h · d_h · sizeof(fp16)
                        =  2 · 80 · 64 · 128 · 2 bytes
                        =  2,621,440 bytes      ≈  2.5 MB / token

bytes_per_token (GQA, g=8) =  2 · L · g · d_h · sizeof(fp16)
                           =  2 · 80 · 8  · 128 · 2 bytes
                           =    327,680 bytes   ≈  0.31 MB / token

ratio:  MHA / GQA  =  h / g  =  64 / 8  =  8×  smaller cache`}
      </MathBlock>

      <Prose>
        <p>
          At a 32k context that&apos;s the difference between <code>80 GB</code>{' '}
          of cache (MHA — doesn&apos;t fit on an H100 at all, 64 cars per
          commuter is a non-starter) and <code>10 GB</code> (GQA — fits
          comfortably alongside the 140 GB of weights on 2× H100s, eight
          cars doing the work of sixty-four). GQA is not a nice-to-have. It
          is the reason 70B models run at long context on hardware you can
          actually rent.
        </p>
      </Prose>

      {/* ── Widget 2: GQAvsFullMHA ─────────────────────────────── */}
      <Prose>
        <p>
          Graph it. Sweep the context length and see the three curves
          diverge. MHA&apos;s line goes through the roof long before
          GQA&apos;s does, because each extra token drags another full
          fleet of cars onto the road.
        </p>
      </Prose>

      <GQAvsFullMHA />

      <Prose>
        <p>
          Notice the shape: cache size is <em>linear</em> in context length,
          and the slope is proportional to the number of KV heads — the
          number of unique cars per token. Cut the slope by 8× and you can
          run 8× the context with the same memory budget, or serve 8× more
          concurrent users with the same cache pool. That second one is
          what production inference engines actually care about: more riders
          per lane.
        </p>
      </Prose>

      <Personify speaker="KV head">
        I am the part of{' '}
        <NeedsBackground slug="self-attention">attention</NeedsBackground>{' '}
        that gets cached — the car itself. Every query that walks past me
        dots against my key and reads my value. Under MHA each query had
        its own copy of me idling in its own lane. Under GQA I am shared —
        four queries, one ride. I do less unique work and take up less
        room. Your inference throughput is my productivity review.
      </Personify>

      {/* ── Uptraining callout ──────────────────────────────────── */}
      <Callout variant="insight" title="uptraining: convert MHA to GQA on the cheap">
        You don&apos;t have to retrain from scratch to start carpooling.
        Take an existing MHA checkpoint, group its K and V heads into{' '}
        <code>g</code> buckets, and <em>mean-pool</em> the weights within
        each bucket — average the four cars in each carpool group into one
        shared ride. The result is a GQA model that starts off slightly
        worse than the original, then fine-tunes back to parity in under 5%
        of the original training compute. Ainslie et al. (2023) showed this
        works for T5; everyone since has copied it.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Into code. The operation that matters is <code>repeat_kv</code> —
          expand the grouped K and V tensors so each query head sees a K/V
          of matching shape. Conceptually: every passenger in the carpool
          sees the same car from their own seat. That one function is the
          entire delta between MHA and GQA in practice.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · gqa_repeat_kv.py"
        output={`Q shape: (1, 8, 4, 16)
K shape: (1, 2, 4, 16)       ← grouped: 2 KV heads, not 8
K_expanded: (1, 8, 4, 16)    ← each KV head repeated to match 4 queries`}
      >{`import numpy as np

# toy dims: batch=1, n_heads=8, n_kv_heads=2, seq=4, d_head=16
B, H, G, T, D = 1, 8, 2, 4, 16
n_rep = H // G                           # each KV head is shared by 4 queries

Q = np.random.randn(B, H, T, D)          # full 8 query heads
K = np.random.randn(B, G, T, D)          # only 2 key heads — the "cache"
V = np.random.randn(B, G, T, D)          # only 2 value heads

def repeat_kv(x, n_rep):
    """Broadcast each KV head n_rep times along the head axis."""
    B, G, T, D = x.shape
    if n_rep == 1:
        return x                         # no expansion — this is MHA
    # insert a length-1 axis, tile it, then collapse back
    x = x[:, :, None, :, :]              # (B, G, 1, T, D)
    x = np.broadcast_to(x, (B, G, n_rep, T, D))
    return x.reshape(B, G * n_rep, T, D) # (B, H, T, D)

K_exp = repeat_kv(K, n_rep)
V_exp = repeat_kv(V, n_rep)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("K_expanded:", K_exp.shape)`}</CodeBlock>

      <Prose>
        <p>
          That <code>repeat_kv</code> helper is what every real GQA
          implementation calls before the attention dot-product. Under the
          hood, a smart kernel (FlashAttention-2 and up) never actually
          materializes the expanded tensor — it reads the grouped K/V from
          the cache and does the broadcast inside the attention
          computation, like four passengers looking out the same
          windshield from different seats. Conceptually, the model sees a
          Q, K, V of matching head count, and the math is identical to MHA
          from that point forward.
        </p>
        <p>
          Now lift it into PyTorch as a self-contained module.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — pytorch · gqa_attention.py">{`import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads     = n_heads
        self.n_kv_heads  = n_kv_heads
        self.n_rep       = n_heads // n_kv_heads
        self.d_head      = d_model // n_heads

        # Q gets full width, K and V get compressed width (n_kv_heads * d_head)
        self.q_proj = nn.Linear(d_model, n_heads    * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def repeat_kv(x, n_rep):
        if n_rep == 1: return x
        B, G, T, D = x.shape
        return x[:, :, None, :, :].expand(B, G, n_rep, T, D).reshape(B, G * n_rep, T, D)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads,    self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # the one line that makes it GQA — expand shared K/V to match every Q head
        k = self.repeat_kv(k, self.n_rep)
        v = self.repeat_kv(v, self.n_rep)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))`}</CodeBlock>

      <Prose>
        <p>
          Strip away the boilerplate and there are exactly three changes
          versus a vanilla MHA module: <code>k_proj</code> and{' '}
          <code>v_proj</code> are smaller, a divisibility assertion (carpool
          groups have to divide evenly), and two calls to{' '}
          <code>repeat_kv</code>. That is the entire architectural novelty.
          Everything else — the causal mask, the softmax, the output
          projection — is unchanged. This is why GQA swept the field so
          fast. It drops into existing codebases with almost no friction.
        </p>
        <p>
          Now the production version — the LLaMA-style block with KV cache,
          RoPE, and the divisibility check inlined.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 3 — llama-style · llama_gqa_block.py">{`import torch
import torch.nn as nn
import torch.nn.functional as F

class LlamaAttention(nn.Module):
    """
    LLaMA-2-style attention. GQA when n_kv_heads < n_heads, MHA when equal, MQA when n_kv_heads=1.
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_heads    = cfg.n_heads              # e.g. 64 for 70B
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads   # e.g.  8 for 70B
        self.n_rep      = self.n_heads // self.n_kv_heads  # = 8 for 70B
        self.d_head     = cfg.d_model // cfg.n_heads

        self.wq = nn.Linear(cfg.d_model, self.n_heads    * self.d_head, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # KV cache — shape uses n_kv_heads, not n_heads. This is the whole memory win.
        self.register_buffer("cache_k",
            torch.zeros(cfg.max_batch, cfg.max_seq, self.n_kv_heads, self.d_head))
        self.register_buffer("cache_v",
            torch.zeros(cfg.max_batch, cfg.max_seq, self.n_kv_heads, self.d_head))

    def forward(self, x, start_pos, freqs_cis, mask):
        B, T, _ = x.shape
        xq = self.wq(x).view(B, T, self.n_heads,    self.d_head)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.d_head)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.d_head)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)   # RoPE — covered in its own lesson

        # write the new K/V into the cache slot for this position
        self.cache_k[:B, start_pos : start_pos + T] = xk
        self.cache_v[:B, start_pos : start_pos + T] = xv
        keys   = self.cache_k[:B, : start_pos + T]     # all K up to now
        values = self.cache_v[:B, : start_pos + T]

        # (B, T, n_kv_heads, d_head) → (B, T, n_heads, d_head) via repeat
        keys   = repeat_kv(keys,   self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq, keys, values = (t.transpose(1, 2) for t in (xq, keys, values))
        out = F.scaled_dot_product_attention(xq, keys, values, attn_mask=mask)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))`}</CodeBlock>

      <Bridge
        label="MHA → GQA — what actually changes"
        rows={[
          {
            left: 'W_K ∈ ℝ^(d × d)',
            right: 'W_K ∈ ℝ^(d × n_kv_heads · d_head)',
            note: 'key projection shrinks by h/g — same for W_V',
          },
          {
            left: 'cache_k shape = (B, T, n_heads, d_head)',
            right: 'cache_k shape = (B, T, n_kv_heads, d_head)',
            note: 'this is where the memory savings actually live',
          },
          {
            left: 'K, V used directly in attention',
            right: 'K, V → repeat_kv(·, n_rep) → attention',
            note: 'the one operational change — broadcast grouped K/V',
          },
          {
            left: 'n_heads = n_kv_heads (implicit)',
            right: 'assert n_heads % n_kv_heads == 0',
            note: 'GQA requires even grouping — check it at init',
          },
        ]}
      />

      <Callout variant="insight" title="who uses GQA, and what they picked">
        LLaMA-2 70B: 64 query heads, 8 KV heads (8× reduction — eight cars
        for sixty-four riders). Mistral 7B: 32 / 8 (4×). LLaMA-3 8B: 32 /
        8. LLaMA-3 70B: 64 / 8. Gemma-7B: 16 / 16 — full MHA, but Gemma-2
        moved to GQA. The pattern is clear: at scale, <em>everybody</em>{' '}
        picks <code>g = 8</code> or thereabouts, because inference
        economics demand it.
      </Callout>

      {/* ── What you give up ────────────────────────────────────── */}
      <Prose>
        <p>
          Now the honest part: what you give up when you carpool. Each K/V
          lane now serves four queries instead of one. The four queries in
          a group can&apos;t ask for radically different content from their
          shared ride — whatever that one K/V returns has to be useful to
          all of them. In full MHA, head 3 could specialize on syntax and
          head 4 on long-range coreference with totally different keys and
          values. In GQA with <code>g = 8</code>, heads 3 and 4 share a K/V
          car and have to agree on what to carry. The queries still differ
          — that&apos;s why we keep every Q — but the retrieval surface
          they share is coarser.
        </p>
        <p>
          In practice the quality hit measured in perplexity is under 1%
          and often indistinguishable. The reason is the asymmetry the
          callout flagged: heads diverge in Q much more than in K/V, so
          pooling K/V into shared rides loses less than pooling Q would.
          MQA (one car, everybody in it) loses more. MHA (a car each, road
          saturated) loses nothing but costs everything. GQA is the
          Goldilocks group size — small enough to cut the cache, large
          enough to still differentiate.
        </p>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Divisibility:</strong>{' '}
          <code>n_heads % n_kv_heads != 0</code> breaks the grouping — you
          can&apos;t have 32 commuters split evenly into 6 cars. If you are
          hand-picking <code>g</code>, verify this at config load. 32 / 6
          will run through your shape assertions and then explode inside{' '}
          <code>repeat_kv</code>.
        </p>
        <p>
          <strong className="text-term-amber">Cache shape:</strong> the KV
          cache uses <code>n_kv_heads</code>, not <code>n_heads</code>. If
          you copy a cache-allocation line from an MHA codebase and forget
          to swap the head count, you will waste memory (best case) or
          silently corrupt reads by striding into the wrong rows (worst
          case).
        </p>
        <p>
          <strong className="text-term-amber">repeat_kv placement:</strong>{' '}
          expand K and V <em>after</em> reading from the cache and{' '}
          <em>before</em> the attention dot product. Expanding before the
          cache throws the memory savings away — you&apos;ve stored a full
          fleet when one carpool would do. Expanding never throws a shape
          error.
        </p>
        <p>
          <strong className="text-term-amber">Loading MHA weights into a GQA model:</strong>{' '}
          the K and V projection matrices have different shapes. You must
          either mean-pool the K/V heads from the MHA checkpoint into{' '}
          <code>g</code> groups (the uptraining recipe) or explicitly
          retrain those projections from scratch. Naively loading will
          ValueError at the <code>state_dict</code> level, which is
          merciful.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Uptrain nanoGPT from MHA to GQA">
        <p>
          Take nanoGPT with <code>n_head = 12</code>. Modify{' '}
          <code>CausalSelfAttention</code> to support a separate{' '}
          <code>n_kv_heads</code> parameter. Set <code>n_kv_heads = 2</code>{' '}
          — a 6× reduction, six queries per shared K/V car — and add the{' '}
          <code>repeat_kv</code> call before the attention dot-product.
        </p>
        <p className="mt-2">
          Load the MHA-trained weights: mean-pool the K and V projections
          into 2 groups of 6, leave Q untouched. Measure perplexity on
          wikitext-103 before fine-tuning — it will be a bit worse.
          Fine-tune for 2% of the original training budget. Perplexity
          should recover to within 1% of the MHA baseline.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: benchmark decoding throughput (tokens/sec) at context
          length 2048, 4096, 8192. You should see the GQA version pull
          ahead more dramatically as context grows — the memory-bandwidth
          bottleneck you just loosened by taking cars off the road.
        </p>
      </Challenge>

      {/* ── Closing ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> KV cache size is the
          bottleneck in modern LLM inference, and it scales linearly with
          the number of KV heads — the number of cars in the fleet. GQA
          shrinks the fleet without shrinking the number of queries:
          queries carpool into groups that share a ride. A single{' '}
          <code>repeat_kv</code> operation and a smaller projection, and
          every 7B+ open model you run this year uses it. MHA is GQA with{' '}
          <code>g = h</code> — everybody drives alone. MQA is GQA with{' '}
          <code>g = 1</code> — everybody piles into one minivan. Pick your{' '}
          <code>g</code> per your memory budget.
        </p>
        <p>
          <strong>End of Build GPT section.</strong> We&apos;ve gone from
          the bare attention equation, through positional embeddings,
          through the full transformer block, through the optimizations
          that make it actually run — LayerNorm, residuals, KV caching,
          and now GQA. If you understand this section end-to-end, you
          could write a decoder-only transformer from scratch. More
          importantly, you could read any modern LLM paper and identify
          exactly which of these building blocks they kept, swapped, or
          improved.
        </p>
        <p>
          <strong>Next section — Fine-Tuning &amp; RLHF.</strong> We have a
          model that speaks. Pretraining got us there: predict the next
          token, a trillion times, until the weights know English. What we
          do <em>not</em> have is a model that listens — one that follows
          instructions, refuses bad requests, or sounds like an assistant
          instead of a very fluent auto-complete. That&apos;s the next
          problem. Supervised fine-tuning first, then reward modeling, PPO,
          DPO — the post-training stack that turns &ldquo;predicts next
          token&rdquo; into &ldquo;is ChatGPT.&rdquo; Most of the craft
          lives there.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Fast Transformer Decoding: One Write-Head is All You Need',
            author: 'Noam Shazeer',
            year: 2019,
            venue: 'the original Multi-Query Attention paper',
            url: 'https://arxiv.org/abs/1911.02150',
          },
          {
            title: 'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints',
            author: 'Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai',
            year: 2023,
            venue: 'EMNLP — the paper that named GQA and introduced uptraining',
            url: 'https://arxiv.org/abs/2305.13245',
          },
          {
            title: 'Llama 2: Open Foundation and Fine-Tuned Chat Models',
            author: 'Touvron et al.',
            year: 2023,
            venue: 'Meta — GQA at scale, 64 query heads / 8 KV heads at 70B',
            url: 'https://arxiv.org/abs/2307.09288',
          },
          {
            title: 'Mistral 7B',
            author: 'Jiang et al.',
            year: 2023,
            venue: 'GQA + sliding-window attention in a 7B model',
            url: 'https://arxiv.org/abs/2310.06825',
          },
        ]}
      />
    </div>
  )
}
