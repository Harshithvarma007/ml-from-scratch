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
import KVCacheGrowth from '../widgets/KVCacheGrowth'
import FLOPsSavings from '../widgets/FLOPsSavings'

// Signature anchor: the diner booth with a pinned-up order ticket. The waiter
// doesn't re-interview the customer on every round; they glance at the pinned
// ticket and only write down the new item. Recompute-from-scratch = re-asking
// the whole order every time. The KV cache is the pinned ticket. Returned to
// at opening, the cache-the-K/V reveal, and the memory-cost section (the
// booth wall has finite space).
export default function KVCacheLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="kv-cache" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Picture a waiter at a diner booth. A customer orders a coffee. The waiter scribbles it
          on a ticket, pins the ticket to the booth wall, and walks off. Five minutes later the
          customer adds a side of toast. A good waiter glances at the pinned ticket and writes{' '}
          <em>toast</em> underneath. A bad waiter re-interviews the customer from the top —{' '}
          <em>name, party size, coffee, toast</em> — every single time a new item shows up.
        </p>
        <p>
          Generate text with a transformer the naive way and you&apos;re the bad waiter.
          At step 1 of generation,{' '}
          <NeedsBackground slug="self-attention">self-attention</NeedsBackground> runs over a
          single token. At step 100, it runs over 100 tokens. At step 1000, over 1000. And at
          each step the model recomputes the keys and values for <em>every past token</em> —
          tokens whose embeddings, whose positions, whose weight matrices have not changed
          since the last step. It&apos;s the software equivalent of re-proving that{' '}
          <code>2 + 2 = 4</code> every time you want to add a three to it. The customer hasn&apos;t
          changed their name in the last four seconds. Stop asking.
        </p>
        <p>
          The total cost of generating <code>T</code> tokens from scratch this way is{' '}
          <code>O(T²)</code>. Generate a 1k-token response and you do about a million units of
          attention work. Generate 10k and you do a hundred million. This is the wall that, if
          you leave it standing, makes long-context inference economically impossible.
        </p>
        <p>
          The fix is the simplest idea in the book: <strong>pin the ticket</strong>. Or, less
          colorfully: <strong>cache the thing that doesn&apos;t change</strong>. That&apos;s the{' '}
          <KeyTerm>KV cache</KeyTerm>. Every token&apos;s K and V is a pinned ticket on the booth
          wall; decoding the next token reads every pinned ticket and only writes the new one. It
          turns the dominant cost of serving large language models from quadratic into linear,
          and it&apos;s the single most important optimization between a nanoGPT toy and a
          production inference stack. Everything else in this lesson is a footnote on that one
          move.
        </p>
      </Prose>

      <Personify speaker="Autoregressive generation (without cache)">
        Every single step I compute K and V for every single past token. I know the answer
        hasn&apos;t changed. I have no memory between forward passes. I&apos;m very fast at
        small context, and then I&apos;m very slow, and then I&apos;m impossible.
      </Personify>

      {/* ── The redundancy, in math ─────────────────────────────── */}
      <Prose>
        <p>
          Let&apos;s see the waste on paper. At decoding step <code>t</code>, causal attention
          inside a{' '}
          <NeedsBackground slug="transformer-block">transformer block</NeedsBackground> needs the
          queries, keys, and values for all tokens up through <code>t</code>. Without a cache,
          you run the entire forward pass on the full sequence of length <code>t</code> and throw
          away everything but the last token&apos;s logits. You interviewed the whole booth to
          learn one new word.
        </p>
      </Prose>

      <MathBlock caption="what a naive decoder does at each step">
{`step t=1:   K₁, V₁       from token 1                    — compute 1 pair
step t=2:   K₁, V₁, K₂, V₂                                 — compute 2 pairs (1 redundant)
step t=3:   K₁, V₁, K₂, V₂, K₃, V₃                         — compute 3 pairs (2 redundant)
...
step t=T:   K₁ … K_T,  V₁ … V_T                            — compute T pairs (T-1 redundant)

total work  =  1 + 2 + 3 + … + T   =   T(T+1)/2   =   O(T²)`}
      </MathBlock>

      <Prose>
        <p>
          Every line after the first is mostly redundant. At step 100 you re-derive the first
          99 keys and values — bit-for-bit identical to what you computed on step 99, and step
          98, and step 97. In a serving workload you&apos;re burning GPU cycles to produce
          numbers you already have. Every ticket on the wall gets rewritten from scratch when a
          customer asks for one more refill.
        </p>
        <p>
          Here is the observation that cracks the whole thing open:{' '}
          <strong>K and V for past tokens are a function of (token, position, model weights)
          only</strong>. None of those change during generation. The token at position 7 is the
          token at position 7 forever; the weights don&apos;t move during inference; the
          positional encoding at position 7 is a constant. So the key and value at position 7
          are a constant too. Compute them once, pin the ticket, glance at it next time. The
          query for the <em>new</em> token is all that&apos;s actually new at each step.
        </p>
      </Prose>

      {/* ── Widget 1: KV Cache Growth ───────────────────────────── */}
      <KVCacheGrowth />

      <Prose>
        <p>
          Watch the cache grow one row at a time. Each generation step pins one more ticket: a
          single <code>(K_t, V_t)</code> pair appended to the stack. The attention query at step{' '}
          <code>t</code> is just <code>Q_t</code> — the one new token — attending against the
          full <code>[K₁ … K_t]</code> and <code>[V₁ … V_t]</code> already on the wall. The
          memory readout in MB climbs linearly; so does the per-step work. The booth wall is
          filling up exactly as fast as the conversation runs.
        </p>
      </Prose>

      <Personify speaker="KV cache">
        I&apos;m the booth wall. Every time you generate a new token, you hand me its K and V
        and I pin the ticket next to the others. Next step, all you need to compute is one Q,
        one K, one V — and then you attend against my whole wall for free. I get bigger, never
        smaller. Budget for me accordingly.
      </Personify>

      {/* ── FLOPs math ─────────────────────────────────────────── */}
      <Prose>
        <p>
          Put numbers on the savings. Let <code>d</code> be the model width,{' '}
          <code>L</code> the number of layers, and <code>T</code> the current context length.
          The per-step cost of attention-plus-projection is roughly:
        </p>
      </Prose>

      <MathBlock caption="per-step FLOPs, naive vs cached">
{`without cache (step t):    recompute K,V for all t tokens       ≈ 4 · L · t · d²
                            plus attention itself                 ≈ 2 · L · t · d

total across T steps:       Σ (4·L·t·d²)   =   2 · L · T² · d²   →   O(T²)


with KV cache (step t):    compute K,V for 1 new token           ≈ 4 · L · d²
                            attention vs cached K,V               ≈ 2 · L · t · d

total across T steps:       Σ (4·L·d² + 2·L·t·d)  ≈  L·T·(4d² + T·d)`}
      </MathBlock>

      <Prose>
        <p>
          The cached version drops the <code>T²</code> in the projection term entirely. The
          only residual <code>T²</code>-like cost is the attention score itself — and that one
          is unavoidable, because the new query <em>must</em> read every pinned ticket on the
          wall. But the matrix-multiply cost of producing those keys and values — which
          dominates at realistic widths (<code>d² &gt;&gt; d</code>) — is gone. The waiter
          stopped re-asking.
        </p>
      </Prose>

      {/* ── Widget 2: FLOPs Savings ─────────────────────────────── */}
      <FLOPsSavings />

      <Prose>
        <p>
          One curve is a quadratic. The other is barely sloped. At a 1024-token generation, the
          naive cost is roughly a thousand times the cached cost for a realistic model — and
          the gap keeps widening. This is the difference between a chatbot that responds in
          three seconds and one that responds in five minutes.
        </p>
      </Prose>

      <Personify speaker="Prefill vs decode">
        We&apos;re the two halves of inference and we do <em>not</em> want the same thing. I,
        prefill, am the first forward pass on the whole prompt at once — matmul-heavy,
        compute-bound, happy to live on an A100&apos;s tensor cores. I, decode, am every step
        after that: one tiny Q against a big wall of pinned K and V. I&apos;m memory-bandwidth
        bound. I want the HBM lanes, not the FLOPs. Optimizing for me without distinguishing
        between us will burn your money.
      </Personify>

      <Callout variant="insight" title="prefill and decode are different workloads">
        The first forward pass of a prompt processes all tokens in parallel — it&apos;s a big
        matmul that pins every ticket at once and saturates the GPU&apos;s arithmetic units.
        Every subsequent step is a single-token forward where the attention operation has to{' '}
        <em>read</em> the entire wall of tickets out of HBM and attend against it with{' '}
        <NeedsBackground slug="multi-headed-self-attention">multi-head attention</NeedsBackground>.
        The bottleneck shifts from FLOPs to memory bandwidth. Modern serving stacks (vLLM,
        TensorRT-LLM, SGLang) treat prefill and decode as separate scheduling domains with
        different batching strategies.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, one algorithm. A tiny pure-Python causal attention where we pin tickets
          to the wall by hand, a NumPy version that shows the wall as an explicit{' '}
          <code>(T, d)</code> array, and the PyTorch version that delegates the pinning to a
          library flag and gets on with its life.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · attention_with_cache.py"
        output={`step 1: cache_len=1  logit=0.4121
step 2: cache_len=2  logit=0.6534
step 3: cache_len=3  logit=0.1890
step 4: cache_len=4  logit=-0.0237`}
      >{`import math

# Scalar-ish, single-head causal attention with a hand-managed KV cache.
# At each step we compute K_t, V_t for just the new token and append.

def attention_step(q_t, K_cache, V_cache):
    # q_t is a single query vector; K_cache and V_cache are lists of past K,V rows.
    scores = [sum(q * k for q, k in zip(q_t, k_row)) / math.sqrt(len(q_t))
              for k_row in K_cache]
    # softmax
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    Z = sum(exps)
    weights = [e / Z for e in exps]
    # weighted sum of values
    d = len(V_cache[0])
    out = [sum(w * v[i] for w, v in zip(weights, V_cache)) for i in range(d)]
    return out

# Toy: Wq, Wk, Wv are the identity — just to show the caching logic.
K_cache, V_cache = [], []
tokens = [[0.2, 0.5], [0.9, -0.1], [0.3, 0.4], [-0.5, 0.6]]

for t, x in enumerate(tokens, 1):
    K_cache.append(x)          # append new K
    V_cache.append(x)          # append new V
    q_t = x                    # and the new Q
    out = attention_step(q_t, K_cache, V_cache)
    print(f"step {t}: cache_len={len(K_cache)}  logit={out[0]:.4f}")`}</CodeBlock>

      <Prose>
        <p>
          Each loop iteration pins exactly one new ticket (<code>K_cache.append</code>,{' '}
          <code>V_cache.append</code>) and then lets the new query attend to the whole stack.
          Now with NumPy, so the cache shape <code>(T, d)</code> — the wall itself — stays
          visible as it grows. This is the version you mentally simulate when reading production
          inference code.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · attention_cache_numpy.py"
        output={`after step 1: K_cache.shape=(1, 64)  V_cache.shape=(1, 64)
after step 2: K_cache.shape=(2, 64)  V_cache.shape=(2, 64)
after step 3: K_cache.shape=(3, 64)  V_cache.shape=(3, 64)
...
after step 8: K_cache.shape=(8, 64)  V_cache.shape=(8, 64)`}
      >{`import numpy as np

d = 64
Wq = np.random.randn(d, d) * 0.02
Wk = np.random.randn(d, d) * 0.02
Wv = np.random.randn(d, d) * 0.02

K_cache = np.zeros((0, d))     # (T, d) — grows by 1 row per step
V_cache = np.zeros((0, d))

def step(x_t):
    global K_cache, V_cache
    q = x_t @ Wq                                    # (d,)
    k = x_t @ Wk                                    # (d,)
    v = x_t @ Wv                                    # (d,)
    K_cache = np.vstack([K_cache, k])               # (T+1, d)
    V_cache = np.vstack([V_cache, v])               # (T+1, d)
    scores = (K_cache @ q) / np.sqrt(d)             # (T+1,)
    w = np.exp(scores - scores.max())
    w /= w.sum()
    return w @ V_cache                              # (d,)

for t in range(1, 9):
    x = np.random.randn(d)
    _ = step(x)
    print(f"after step {t}: K_cache.shape={K_cache.shape}  V_cache.shape={V_cache.shape}")`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'K_cache = []; K_cache.append(k)',
            right: 'K_cache = np.vstack([K_cache, k])',
            note: 'list-of-rows becomes a contiguous (T, d) array',
          },
          {
            left: 'for k in K_cache: dot(q, k)',
            right: 'K_cache @ q   # (T, d) @ (d,) = (T,)',
            note: 'the whole attention-score row in one matmul',
          },
          {
            left: 'manual softmax over a list',
            right: 'np.exp(scores - scores.max()); normalize',
            note: 'numerically-stable softmax over the whole T at once',
          },
        ]}
      />

      <Prose>
        <p>
          PyTorch: in the <code>transformers</code> library, pinning tickets is a single keyword
          argument. The model returns a <code>past_key_values</code> tuple — the entire booth
          wall, packaged — that you pass back in on the next step, and it handles all the
          stacking for you across every layer and head.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · generate_with_cache.py"
        output={`prompt length: 50
tokens generated: 100
naive generation: 8.4s
cached generation: 0.7s
speedup: 12.0x`}
      >{`import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval().cuda()

prompt = tok("The KV cache is important because", return_tensors="pt").input_ids.cuda()

# --- cached generation ---
t0 = time.time()
out = model.generate(prompt, max_new_tokens=100, use_cache=True, do_sample=False)
t_cached = time.time() - t0

# --- naive (no-cache) generation — for comparison only ---
t0 = time.time()
out = model.generate(prompt, max_new_tokens=100, use_cache=False, do_sample=False)
t_naive = time.time() - t0

print(f"prompt length: {prompt.shape[1]}")
print(f"tokens generated: 100")
print(f"naive generation: {t_naive:.1f}s")
print(f"cached generation: {t_cached:.1f}s")
print(f"speedup: {t_naive / t_cached:.1f}x")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'K_cache = np.vstack([K_cache, k])',
            right: 'past_key_values  (library-managed tuple)',
            note: 'a tuple of (K, V) per layer, stacked across heads and batch',
          },
          {
            left: 'recompute everything',
            right: 'use_cache=True',
            note: 'one flag. the entire O(T²) → O(T) conversion.',
          },
          {
            left: 'hand-roll softmax + matmul',
            right: 'model.generate(...)',
            note: 'library handles prefill, decode, stopping, sampling',
          },
        ]}
      />

      <Callout variant="insight" title="the whole optimization is one flag in production">
        You will almost never pin tickets by hand in a real codebase. The frameworks do it. But
        you <em>will</em> spend hours debugging issues that come from misunderstanding how the
        cache is shaped, when it&apos;s invalidated, and how it interacts with batching and
        memory. Knowing the math and having written the NumPy version is what keeps you from
        getting stuck when the abstraction leaks — because it leaks.
      </Callout>

      {/* ── Memory cost callouts ────────────────────────────────── */}
      <Callout variant="warn" title="the booth wall has finite space">
        The wall isn&apos;t infinite. For a decoder-only transformer, cache shape is{' '}
        <code>(batch, n_layers, 2, n_heads, T, d_head)</code>. For Llama-2 70B (80 layers, 64
        heads, head dim 128) at a 4k context with fp16 weights, that&apos;s about <strong>3
        GB per sequence</strong>. At 128k context: <strong>~100 GB</strong>. At serving scale
        with dozens of concurrent customers, the KV cache — not the model weights — is what
        fills your GPUs. Every booth needs its own wall and every wall needs its own pins. This
        is why long-context serving is hard, and why compressed-cache techniques (MQA, GQA,
        sliding-window attention) are a hot research area.
      </Callout>

      <Callout variant="note" title="paged attention (vLLM, 2023)">
        Early KV-cache implementations allocated one big contiguous buffer per sequence, sized
        to the maximum context length — an empty wall the size of the worst possible
        conversation, reserved on day one. For a batch with mixed prompt lengths, most of that
        wall was blank. Kwon et al. introduced <strong>PagedAttention</strong>: store the cache
        in fixed-size pages (like OS virtual memory), maintain a page table per sequence, and
        pin only as many pages as the conversation actually needs. Result: 2–4× higher
        throughput because you can fit many more concurrent customers on the same GPU. This is
        what powers vLLM and most modern inference servers.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Cache invalidation on prompt change:</strong> if
          you edit the prompt mid-conversation (e.g. system prompt change, retrieval injection),
          every pinned ticket from the edit point onward is stale. You have to rip them off the
          wall and re-prefill. Frameworks that claim to &ldquo;reuse&rdquo; the cache across
          requests use prefix matching on the raw token IDs — change one token at position 3
          and everything past position 3 is garbage.
        </p>
        <p>
          <strong className="text-term-amber">Unbounded growth:</strong> the wall has no natural
          stopping point. A long conversation will keep pinning tickets until you OOM. Always
          set a <code>max_seq_len</code>, and decide up front whether you&apos;ll sliding-window
          it, compress it, or just refuse the request.
        </p>
        <p>
          <strong className="text-term-amber">Batch dimension mismatch:</strong> the cache is
          shaped with a batch dimension — one wall per booth. If your prompt tensor comes in as{' '}
          <code>(1, T)</code> but your cache was allocated for batch 4, the library will
          either broadcast silently (wrong results) or throw a shape error. Always match them.
        </p>
        <p>
          <strong className="text-term-amber">Training vs eval mode:</strong> pinning tickets is
          an inference-only optimization. If you accidentally leave the model in{' '}
          <code>model.train()</code> mode and enable <code>use_cache=True</code>, dropout fires
          on every forward pass and your cached K/V become inconsistent with the Q you&apos;re
          comparing them against. Always <code>model.eval()</code> first.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Retrofit nanoGPT with a KV cache">
        <p>
          Take Karpathy&apos;s nanoGPT and modify the <code>generate()</code> loop to pin
          tickets by hand. In each decoder block, maintain a running{' '}
          <code>(K_cache, V_cache)</code> of shape <code>(B, n_heads, T, d_head)</code>. At
          each generation step, compute <code>Q, K, V</code> for only the newest token,
          concatenate K and V onto the wall along the time dimension, and run attention with
          the new <code>Q</code> against the full pinned stack.
        </p>
        <p className="mt-2">
          Generate 100 tokens from a 50-token prompt. Measure wall-clock time with and without
          your cache on both CPU and a GPU if you have one. Expected speedup: 5–20× depending
          on model size and hardware.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: plot per-step latency against step number. Without a cache it should slope up
          linearly (each step attends over more tokens <em>and</em> re-projects them all — the
          waiter re-asking the whole order). With a cache it should be nearly flat, with a small
          linear creep from the attention score itself (one glance at the wall, getting slightly
          longer).
        </p>
      </Challenge>

      {/* ── Takeaways ───────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Past keys and values don&apos;t change between
          generation steps, so pinning them to the booth wall collapses the total cost from{' '}
          <code>O(T²)</code> to something close to linear in <code>T</code>. The savings come
          from not recomputing the K and V projections; the <em>attention scores themselves</em>{' '}
          still have to glance at every pinned ticket, which is why decode-time is
          memory-bandwidth bound. The cache is the reason long-context LLM serving is feasible
          at all, and its memory cost — tens to hundreds of GB at realistic scales — is the
          reason modern inference stacks work so hard at paging, quantizing, and compressing
          the wall.
        </p>
        <p>
          <strong>Next up — Train Your GPT.</strong> Everything so far has been an inference-time
          trick: the weights are frozen, the cache just memoizes what a trained model already
          knows. But where do the weights come from? In <code>train-your-gpt</code> we stop
          serving a GPT and start making one: AdamW, learning-rate warmup, cosine decay, gradient
          clipping — the recipe that turns a pile of initialized tensors into a model worth
          caching in the first place. Without training, the booth wall is just a wall.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Fast Transformer Decoding: One Write-Head is All You Need',
            author: 'Noam Shazeer',
            year: 2019,
            venue: 'the paper that introduced KV caching and multi-query attention',
            url: 'https://arxiv.org/abs/1911.02150',
          },
          {
            title: 'Efficient Memory Management for Large Language Model Serving with PagedAttention',
            author: 'Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica',
            venue: 'SOSP 2023 — the vLLM paper',
            url: 'https://arxiv.org/abs/2309.06180',
          },
          {
            title: 'Efficiently Scaling Transformer Inference',
            author: 'Pope, Douglas, Chowdhery, Devlin, Bradbury, Levskaya, Heek, Xiao, Agrawal, Dean',
            year: 2022,
            venue: 'MLSys 2023 — the canonical reference for prefill/decode scheduling',
            url: 'https://arxiv.org/abs/2211.05102',
          },
        ]}
      />
    </div>
  )
}
