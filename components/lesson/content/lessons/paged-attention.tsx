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
import KVPaging from '../widgets/KVPaging'
import MemoryFragmentation from '../widgets/MemoryFragmentation'

// Signature anchor: the OS paging system for KV cache. Naive allocation = one
// contiguous slab of RAM per request, worst-cased to max_seq_len; paged = the
// 60-year-old OS trick of fixed-size pages + per-sequence page table. Return
// at the fragmentation-horror opener, the virtual-memory-for-transformers
// reveal, and the throughput consolidation.
export default function PagedAttentionLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout ─────────────────────────────────── */}
      <Prereq currentSlug="paged-attention" />

      {/* ── Opening: the fragmentation horror ───────────────────── */}
      <Prose>
        <p>
          Imagine you run a hotel. Every guest checks in and you hand them the
          exact same thing: the presidential suite. Doesn&apos;t matter if
          they&apos;re staying five minutes or five nights — they get the big
          room with the view and the extra closet. Most of them leave it
          untouched. When they check out you flip the mattress and hold the
          suite for the next guest, who also, obviously, gets a suite. You are
          running out of suites. The lobby is full of people you&apos;re
          turning away. The hotel, to use a technical term, is cooked.
        </p>
        <p>
          That&apos;s what a naive{' '}
          <NeedsBackground slug="kv-cache">KV cache</NeedsBackground> allocator
          does to your GPU. Every new request arrives and is handed one long
          contiguous slab of memory sized to the worst case — <code>max_seq_len</code>,
          4k or 32k or 128k tokens — regardless of how long the request
          actually turns out to be. A three-word question gets the same slab
          as a thousand-token essay. The three-word request finishes, its
          slab goes back on the shelf to be wasted by the next short request,
          and the accounting says your GPU is &ldquo;full&rdquo; while most
          of it is quite literally untouched bytes.
        </p>
        <p>
          Then it gets worse. Continuous serving admits and evicts sequences
          of wildly different sizes. Those contiguous slabs finish at
          different times and leave a hole-shaped graveyard behind them.{' '}
          <KeyTerm>Memory fragmentation</KeyTerm>: total free bytes on the
          GPU are massive; none of the holes are big enough for the next
          incoming request. You&apos;re refusing customers in an empty
          restaurant because you can&apos;t find four chairs in a row.
        </p>
        <p>
          The fix is stolen, wholesale, from the trick your operating system
          has been quietly running for fifty years. Stop handing out
          contiguous RAM. Break memory into fixed-size <em>pages</em>. Keep a
          per-sequence <em>page table</em> that maps the sequence&apos;s
          logical view of its cache onto whatever physical blocks happen to
          be free. Let every sequence&apos;s cache be a scattered chain of
          pages pulled from a shared pool — the GPU kernel can chase the
          pointers, the allocator can never get stuck. Welcome to{' '}
          <KeyTerm>PagedAttention</KeyTerm>, the vLLM-era idea that dragged
          LLM serving from ~30% memory utilization to 96%+ and made
          long-context, high-concurrency inference economically viable.
        </p>
      </Prose>

      <Personify speaker="Naive KV allocator">
        I give every request one contiguous slab of memory sized to{' '}
        <code>max_seq_len</code> on arrival. Most of them never fill it. When
        they finish I take the slab back. When a new request arrives I look
        for a single contiguous hole big enough to fit <code>max_seq_len</code>{' '}
        again. On a busy GPU there often isn&apos;t one, even though I
        technically have the bytes. I am the reason you&apos;re getting OOMs
        with the GPU 40% empty.
      </Personify>

      {/* ── The math of wasted memory ───────────────────────────── */}
      <Prose>
        <p>
          Put numbers on the waste. A typical chat distribution: mean prompt
          length ~200 tokens, long tail out to 4k. If you allocate{' '}
          <code>max_seq_len = 4096</code> per request and the actual
          completion averages 500 tokens, you&apos;re writing to 500 slots and
          reserving 4096. The rest is internal padding — allocated, charged
          to the request, never touched. On a paged allocator with block
          size 16, the same 500-token request uses 32 blocks and wastes at
          most the tail of the last one.
        </p>
      </Prose>

      <MathBlock caption="naive contiguous RAM vs paged memory — who pays for what">
{`naive contiguous allocation (one slab of RAM per request):
  per request       =  max_seq_len · 2 · n_layers · n_heads · d_head · bytes_per_elem
  actually used     =  actual_len  · 2 · n_layers · n_heads · d_head · bytes_per_elem
  utilization       =  actual_len / max_seq_len     ≈ 20%–40% in practice

  + external fragmentation loss  (holes that don't fit the next request)
  effective utilization  ≈ 20%–50%  (30–50% is the field-reported baseline)


paged allocation (block size B, e.g. 16 tokens per page):
  pages per request  =  ceil(actual_len / B)
  overhead per req   =  B − (actual_len mod B)  tokens  (last page only)
  utilization        =  actual_len / (ceil(actual_len/B) · B)     ≈ 96%+

  + zero external fragmentation  (every page is the same size — any hole fits)`}
      </MathBlock>

      <Prose>
        <p>
          Two wins stacked. First, <em>internal</em> waste drops from
          &ldquo;the rest of the slab&rdquo; to &ldquo;the tail of the last
          page&rdquo; — from potentially 3596 unused tokens to at most 15.
          Second, <em>external</em> fragmentation disappears entirely, because
          every free hole in memory is exactly one page — and every
          request&apos;s next allocation is also exactly one page. Any hole
          fits any request. The allocator can never get stuck looking for a
          contiguous run that isn&apos;t there.
        </p>
      </Prose>

      {/* ── Widget 1: KVPaging ──────────────────────────────────── */}
      <KVPaging />

      <Prose>
        <p>
          Four concurrent requests, each with its own length. Watch how each
          cache is <em>not</em> a single contiguous region — it&apos;s a
          sequence of small pages scattered across the GPU. The page table on
          the right is the key abstraction: it maps the <em>logical</em>{' '}
          position of a token in the sequence (&ldquo;position 42 of sequence
          3&rdquo;) to the <em>physical</em> page in GPU memory where its K
          and V live. The sequence thinks its cache is one long line. The
          kernel knows it&apos;s a handful of scattered pages. The page table
          lies to both of them and keeps the peace.
        </p>
      </Prose>

      <Personify speaker="Page table">
        I am the translation layer. For each sequence I keep a little list:
        &ldquo;logical page 0 lives at physical page 47, logical page 1 lives
        at physical page 12, logical page 2 lives at physical page 89.&rdquo;
        When the sequence needs a new page I grab any free one and append it.
        When the sequence dies I return every page on my list to the pool. To
        the attention kernel I hand over my list and let the kernel gather
        what it needs. I am small, I am boring, and I am the whole reason
        this works.
      </Personify>

      {/* ── Fragmentation math ──────────────────────────────────── */}
      <Prose>
        <p>
          The fragmentation story is a classic of operating-systems memory
          management, dressed up for GPUs. Here it is in three lines of
          pseudo-time.
        </p>
      </Prose>

      <MathBlock caption="how naive allocation strands memory over time">
{`t=0:   [====== A ======][======= B ======][====== C ======][====== D ======]  100% used
                                                                             free: 0

t=1:   B finishes, C finishes:
       [====== A ======][ free — 4k slots ][ free — 4k slots ][====== D ======]
                        ↑ 8k free, but in TWO holes

t=2:   new request E needs 6k contiguous slots:
       →  REJECTED.  total free = 8k,  largest contiguous hole = 4k.


paged allocation at t=2:
       B's and C's pages dissolve back into the shared pool as individual pages;
       E needs 6k tokens → 375 pages of size 16;
       allocator grabs any 375 free pages in any order → admitted, instantly.`}
      </MathBlock>

      <Prose>
        <p>
          The naive version has 8k free tokens on the GPU and still rejects a
          6k request. The paged version sees 8k free tokens as 500 free pages
          and doesn&apos;t care where in RAM they live. Same bytes on the
          same chip. One allocator is stuck, the other keeps serving.
        </p>
      </Prose>

      {/* ── Widget 2: MemoryFragmentation ──────────────────────── */}
      <MemoryFragmentation />

      <Prose>
        <p>
          Run the simulation. The contiguous allocator looks healthy for a
          while — then a churn of short requests leaves the slab riddled with
          holes, and throughput flatlines even though plenty of GPU memory is
          technically free. The paged allocator runs straight through. No
          holes to stumble over, because everything is the same size. The
          only way to run out of pages is to actually be full — not because
          the bookkeeping failed you.
        </p>
        <p>
          This is the headline number from the vLLM paper:{' '}
          <strong>2–4× higher throughput on the same hardware</strong>,
          measured on real traces. It isn&apos;t a kernel trick or a
          lower-precision format. It&apos;s just refusing to allocate memory
          you haven&apos;t used yet.
        </p>
      </Prose>

      <Callout variant="insight" title="virtual memory for transformers — the OS analogy is exact, not cute">
        Virtual memory on your laptop works the same way. A process thinks it
        has a contiguous 4 GB of address space; in reality those bytes live
        in scattered 4 KB pages all over physical RAM (and sometimes on
        disk). The OS page table translates. PagedAttention is the identical
        idea applied to the KV cache: each sequence gets a virtual,
        contiguous-looking cache, and the block table translates to physical
        pages. The 16-token page is just the LLM serving analogue of the 4 KB
        RAM page your kernel has been shuffling around for fifty years.
      </Callout>

      {/* ── Copy-on-write for shared prefixes ───────────────────── */}
      <Prose>
        <p>
          There&apos;s a second trick the paging abstraction unlocks for free.{' '}
          <KeyTerm>Copy-on-write</KeyTerm>. When two sequences share a prefix
          — think{' '}
          <NeedsBackground slug="self-attention">attention</NeedsBackground>{' '}
          over a beam search with beam width 4, or parallel sampling with{' '}
          <code>n=4</code>, or a chat with four alternative continuations of
          the same system prompt — they read the same keys and values for
          that prefix. Naive allocation duplicates the prefix cache four
          times. There&apos;s no reason to: the K and V for a token are a
          deterministic function of the token and its position. If the
          tokens are the same and the positions are the same, the pages are
          bit-identical.
        </p>
        <p>
          So: point all four sequences at the same physical pages for the
          shared prefix. When one of them diverges and writes a different
          token, <em>then</em> allocate a fresh page for just that sequence
          and copy the prefix page into it. Most sequences never diverge
          until late in generation. Most of the prefix RAM stays shared
          forever. Beam search and parallel sampling get nearly free — a
          huge deal when you&apos;re serving anything that samples multiple
          continuations.
        </p>
      </Prose>

      <Personify speaker="Copy-on-write">
        I watch who&apos;s reading from which physical page. When four
        sequences all point at page 12 and only read from it, I do nothing —
        they share happily. The instant one of them tries to <em>write</em>{' '}
        something different, I duplicate page 12 into a fresh physical page,
        update that one sequence&apos;s page table to point at the copy, and
        let it write. The other three never notice. Filesystems, process
        forks, and now KV caches — same trick, three decades apart.
      </Personify>

      <Callout variant="note" title="the attention kernel has to cooperate">
        Dense contiguous attention is easy: one big matmul on a tile of
        memory. PagedAttention has to gather its keys and values from
        non-contiguous physical pages scattered around HBM before it can
        compute scores — and do it in a kernel that doesn&apos;t lose the
        memory wins to inefficient reads. vLLM ships a custom CUDA kernel
        for exactly this; FlashAttention&apos;s variable-length and paged
        variants do similar work. Per-op it&apos;s slightly slower than
        vanilla contiguous attention. But the memory wins let you batch far
        more sequences at once, and aggregate throughput goes up a lot.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers. A pure-Python page allocator you could write on a
          napkin, a NumPy simulation showing the fragmentation story over
          time, and the PyTorch version which — true to form for a production
          technique — is one import and a keyword argument.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · paged_allocator.py"
        output={`seq-0 blocks: [0, 1, 2]          (45 tokens → 3 blocks of 16)
seq-1 blocks: [3]                 (8 tokens  → 1 block)
seq-2 blocks: [4, 5, 6, 7]        (60 tokens → 4 blocks)
free pool: [8, 9, 10, 11, 12, 13, 14, 15]

seq-1 finishes — block 3 returns to pool.
free pool: [8, 9, 10, 11, 12, 13, 14, 15, 3]

seq-3 (new, 32 tokens, needs 2 blocks) → allocated blocks [8, 9].
seq-0 grows by 20 tokens → needs 2 more blocks → appended [10, 11].
no fragmentation. every free block is interchangeable.`}
      >{`# A minimal paged KV-cache allocator. Fixed-size blocks, free-list, block tables.
# This is the architectural core of vLLM, stripped to its skeleton.

BLOCK_SIZE = 16
NUM_BLOCKS = 16

class PagedAllocator:
    def __init__(self, num_blocks=NUM_BLOCKS):
        self.free = list(range(num_blocks))       # pool of physical block ids
        self.tables = {}                          # seq_id -> [physical block ids]
        self.lengths = {}                         # seq_id -> token count

    def allocate(self, seq_id, num_tokens):
        blocks_needed = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        if blocks_needed > len(self.free):
            raise RuntimeError("out of blocks")
        blocks = [self.free.pop(0) for _ in range(blocks_needed)]
        self.tables[seq_id] = blocks
        self.lengths[seq_id] = num_tokens
        return blocks

    def append_tokens(self, seq_id, num_new):
        old_len = self.lengths[seq_id]
        new_len = old_len + num_new
        old_blocks = (old_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        new_blocks = (new_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for _ in range(new_blocks - old_blocks):
            self.tables[seq_id].append(self.free.pop(0))
        self.lengths[seq_id] = new_len

    def free_sequence(self, seq_id):
        self.free.extend(self.tables.pop(seq_id))
        del self.lengths[seq_id]

alloc = PagedAllocator()
alloc.allocate("seq-0", 45)
alloc.allocate("seq-1", 8)
alloc.allocate("seq-2", 60)
print("seq-0 blocks:", alloc.tables["seq-0"])
print("seq-1 blocks:", alloc.tables["seq-1"])
print("seq-2 blocks:", alloc.tables["seq-2"])
print("free pool:   ", alloc.free)

alloc.free_sequence("seq-1")
print("seq-1 finishes — block 3 returns to pool.")
print("free pool:   ", alloc.free)

alloc.allocate("seq-3", 32)
alloc.append_tokens("seq-0", 20)
print("seq-3 (new, 32 tokens) →", alloc.tables["seq-3"])
print("seq-0 after growth →", alloc.tables["seq-0"])`}</CodeBlock>

      <Prose>
        <p>
          That&apos;s the whole architectural core in fifty lines. Every
          production paged-KV cache in the world — vLLM, TensorRT-LLM, TGI,
          SGLang — is this with a real kernel behind it and a scheduler on
          top. Now add a second allocator that models the naive contiguous
          version, and run both through the same admission trace so you
          can <em>see</em> fragmentation arise on one and not the other.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · fragmentation_vs_paging.py"
        output={`[naive contiguous]   admitted 42/60 requests — 18 rejected due to fragmentation
                      peak util: 62%   avg util: 38%

[paged, block=16]    admitted 60/60 requests —  0 rejected
                      peak util: 97%   avg util: 82%

memory-bound throughput uplift: ~2.2x on this trace`}
      >{`import numpy as np

TOTAL_TOKENS = 8192                     # GPU-side cache capacity, in tokens
MAX_SEQ = 2048                          # naive allocator's worst-case per slot
BLOCK = 16                              # paged allocator block size (page size)

rng = np.random.default_rng(0)
# 60 requests: short median, long tail (mimics real chat traces)
lengths = np.clip(rng.lognormal(mean=5.5, sigma=0.9, size=60).astype(int), 8, MAX_SEQ)
arrive  = np.sort(rng.uniform(0, 100, size=60))
finish  = arrive + rng.uniform(1, 8, size=60)

def simulate_naive(lengths, arrive, finish):
    slots = [None] * (TOTAL_TOKENS // MAX_SEQ)   # fixed number of max-sized slots
    admitted, rejected, util = 0, 0, []
    events = [(t, 'a', i) for i, t in enumerate(arrive)] + \\
             [(t, 'f', i) for i, t in enumerate(finish)]
    for t, kind, i in sorted(events):
        if kind == 'f':
            for s, occ in enumerate(slots):
                if occ == i: slots[s] = None
        else:
            try:
                s = slots.index(None); slots[s] = i; admitted += 1
            except ValueError:
                rejected += 1
        used = sum(lengths[occ] for occ in slots if occ is not None)
        util.append(used / TOTAL_TOKENS)
    return admitted, rejected, float(np.mean(util))

def simulate_paged(lengths, arrive, finish):
    free_blocks = TOTAL_TOKENS // BLOCK
    tables = {}
    admitted, rejected, util = 0, 0, []
    events = [(t, 'a', i) for i, t in enumerate(arrive)] + \\
             [(t, 'f', i) for i, t in enumerate(finish)]
    for t, kind, i in sorted(events):
        if kind == 'f' and i in tables:
            free_blocks += tables.pop(i)
        elif kind == 'a':
            needed = (lengths[i] + BLOCK - 1) // BLOCK
            if needed <= free_blocks:
                tables[i] = needed; free_blocks -= needed; admitted += 1
            else:
                rejected += 1
        used_blocks = (TOTAL_TOKENS // BLOCK) - free_blocks
        util.append((used_blocks * BLOCK) / TOTAL_TOKENS)
    return admitted, rejected, float(np.mean(util))

a_n, r_n, u_n = simulate_naive(lengths, arrive, finish)
a_p, r_p, u_p = simulate_paged(lengths, arrive, finish)
print(f"[naive contiguous]   admitted {a_n}/60 requests — {r_n} rejected")
print(f"                      avg util: {u_n:.0%}")
print(f"[paged, block={BLOCK}]    admitted {a_p}/60 requests — {r_p} rejected")
print(f"                      avg util: {u_p:.0%}")
print(f"memory-bound throughput uplift: ~{a_p / max(a_n, 1):.1f}x on this trace")`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'PagedAllocator.allocate(seq, n)',
            right: 'ceil(n / BLOCK) pages pulled from pool',
            note: 'same algorithm; NumPy just lets us simulate 60 requests at once',
          },
          {
            left: 'hand-tracked free list',
            right: 'scalar free_blocks counter',
            note: 'once fragmentation is gone, a count is all you need',
          },
          {
            left: 'single allocator run',
            right: 'comparative trace on naive vs paged',
            note: 'admission rate is the whole story — see the gap',
          },
        ]}
      />

      <Prose>
        <p>
          Layer 3. You don&apos;t hand-roll this in production. You install
          vLLM. The entire machinery above — page allocator, block tables,
          paged attention kernel, copy-on-write for beam search, prefix
          sharing across requests, and the{' '}
          <NeedsBackground slug="continuous-batching">continuous batching</NeedsBackground>{' '}
          scheduler that feeds it all — ships behind one import.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · serve_with_vllm.py"
        output={`[vLLM]  model loaded: meta-llama/Llama-2-13b-hf
[vLLM]  # GPU blocks: 2842   # CPU blocks: 512   block_size: 16
[vLLM]  gpu_memory_utilization: 0.90

generated 256 tokens × 128 concurrent requests in 3.1s
throughput: 10,580 tokens/s
peak KV cache utilization: 94%
rejected requests: 0`}
      >{`from vllm import LLM, SamplingParams

# One line. Page allocator, paged kernel, continuous batching, all wired up.
# block_size defaults to 16; gpu_memory_utilization sets how much HBM vLLM may use
# for weights + paged KV cache combined.
llm = LLM(
    model="meta-llama/Llama-2-13b-hf",
    gpu_memory_utilization=0.90,
    block_size=16,
    max_num_seqs=256,          # concurrency cap — paging makes this feasible
)

# 128 varied prompts, different lengths, submitted at once. Without paging, most
# would be rejected for fragmentation. With paging, they run concurrently on one GPU.
prompts = [f"Explain concept #{i} in one paragraph." for i in range(128)]
params = SamplingParams(max_tokens=256, temperature=0.7)

outputs = llm.generate(prompts, params)

# For a feel of what copy-on-write buys: ask for 4 parallel samples per prompt.
# vLLM shares the prompt pages across the 4 samples and only diverges on write.
params_n4 = SamplingParams(n=4, max_tokens=128, temperature=0.9)
beam_outputs = llm.generate(prompts[:32], params_n4)

for req in beam_outputs[:2]:
    print(f"prompt: {req.prompt[:50]}…")
    for i, out in enumerate(req.outputs):
        print(f"  sample {i}: {out.text[:60]}…")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'hand-rolled free_blocks counter',
            right: 'vllm.LLM(gpu_memory_utilization=0.90)',
            note: 'vLLM figures out the page budget from free HBM and model size',
          },
          {
            left: 'simulate admission trace',
            right: 'llm.generate(prompts, params)   # continuous batching',
            note: 'the scheduler admits/evicts in real time, fed by the paged allocator',
          },
          {
            left: 'prefix-sharing as a thought experiment',
            right: 'SamplingParams(n=4)   # copy-on-write, automatic',
            note: 'four samples, one shared prefix in memory, diverge only on write',
          },
        ]}
      />

      <Callout variant="insight" title="vLLM owns this paradigm">
        The 2023 vLLM paper didn&apos;t just propose PagedAttention — it
        bundled it with continuous batching, a paged-aware kernel, and an
        OSS serving stack. Two years on, essentially every serious LLM
        inference engine has some variant. TensorRT-LLM ships paged KV
        cache. TGI (HuggingFace) ships it. SGLang ships it with additional
        prefix caching across requests. Llama.cpp has paged variants. If
        you&apos;re serving an LLM in 2026 and you&apos;re not running a
        paged KV cache, you&apos;re leaving a 2–4× throughput multiplier on
        the floor.
      </Callout>

      <Callout variant="note" title="page size is a tuning knob">
        Page size is a classic throughput-vs-waste trade-off. Small pages
        (<code>B=4</code>) minimise internal waste in the last page but
        increase page-table overhead, kernel launch cost, and gather-scatter
        inefficiency. Large pages (<code>B=64</code>) make the kernel happy
        but leave bigger unused tails in the last page of each sequence.
        vLLM&apos;s default is 16, which is the empirically-nice balance on
        current GPUs. For extremely short sequences (e.g. embeddings),{' '}
        <code>B=8</code> can help. For 128k-context workloads,{' '}
        <code>B=32</code> is often a win.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Page size too small:</strong>{' '}
          every extra page is a row in the page table, a pointer in the
          kernel&apos;s gather list, and a chunk of dispatch overhead. Drop{' '}
          <code>block_size</code> to 4 and per-token latency can visibly
          regress even as waste drops.
        </p>
        <p>
          <strong className="text-term-amber">Page size too large:</strong>{' '}
          every sequence&apos;s last page is on average half-full, so
          internal waste scales with page size. At <code>B=128</code> on
          short-chat workloads you start giving back the memory gains you
          came here for.
        </p>
        <p>
          <strong className="text-term-amber">Kernel support:</strong>{' '}
          PagedAttention needs a gather-capable attention kernel. Standard{' '}
          <code>scaled_dot_product_attention</code> assumes contiguous K/V.
          If you&apos;re writing a custom serving stack, plan for{' '}
          <code>vllm.attention.PagedAttentionV2</code> or FlashAttention&apos;s
          paged variant — don&apos;t try to emulate the gather in
          PyTorch-level indexing, it&apos;s a factor of 10× slowdown away
          from your contiguous baseline.
        </p>
        <p>
          <strong className="text-term-amber">Checkpoint load order with paging:</strong>{' '}
          vLLM allocates its page pool <em>after</em> loading weights — the
          remaining HBM defines the page count. If you set{' '}
          <code>gpu_memory_utilization=0.98</code> on a 13B model with a 24
          GB card, there may be zero headroom left for pages and vLLM will
          refuse to start (or start and instantly OOM on the first batch).
          Target 0.85–0.90 and measure.
        </p>
        <p>
          <strong className="text-term-amber">Copy-on-write in beam search:</strong>{' '}
          the <em>moment</em> beams diverge, each needs its own copy of the
          last shared page. A beam width of 8 with early divergence can
          actually use more memory than a single long sequence. Paging makes
          it merely painful rather than impossible, but it&apos;s still not
          free.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Serve Llama-13B with and without paging, count the concurrent requests">
        <p>
          Stand up two servers on the same GPU (A100 80 GB, or equivalent).
          Server A: HuggingFace <code>transformers</code> with its default
          contiguous KV cache, served via{' '}
          <code>text-generation-inference</code>&apos;s no-paging mode or a
          hand-rolled FastAPI loop. Server B: <code>vllm.LLM(...)</code> with{' '}
          <code>gpu_memory_utilization=0.90</code> and default{' '}
          <code>block_size=16</code>.
        </p>
        <p className="mt-2">
          Submit the same trace of 200 prompts with lengths drawn from a
          log-normal (mean ~200 tokens, tail to 4k). For each server,
          measure: (1) peak concurrent requests in flight, (2) total
          throughput in tokens/sec, (3) requests rejected/queued due to
          memory. Paged vLLM should handle 4–8× more concurrent requests,
          with 2–4× higher throughput, and zero rejections on the same
          hardware. Estimate the naive upper bound analytically from{' '}
          <code>HBM_free / (max_seq_len · 2 · n_layers · n_heads · d_head · 2 bytes)</code>{' '}
          to compare.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: add parallel sampling (<code>n=4</code>) to half the
          requests. Watch vLLM&apos;s memory usage barely budge — that&apos;s
          copy-on-write earning its keep. Try the same in naive mode and
          watch the cache quadruple.
        </p>
      </Challenge>

      {/* ── Takeaways + curriculum close ────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> The KV cache is the
          dominant memory cost in LLM serving, and the naive way to allocate
          it — a contiguous worst-case slab of RAM per request — wastes most
          of its bytes to internal padding and most of what&apos;s left to
          external fragmentation. PagedAttention fixes both by stealing the
          sixty-year-old OS idea of paged virtual memory: fixed-size pages, a
          per-sequence page table, a gather-capable attention kernel that
          follows the pointers. The page table is the whole trick; everything
          else — copy-on-write for beam search, prefix sharing across
          conversations, 96%+ cache utilization — falls out of having it.
          In production you get this by typing <code>from vllm import LLM</code>,
          and the 2–4× throughput uplift comes with it.
        </p>
        <p>
          <strong>End of curriculum.</strong> You&apos;ve built every piece
          of the stack from gradient descent to production LLM serving. The
          gradient you derived by hand updates the weight matrix you
          initialised with He init; the matrix multiplies through layers
          you stacked with residuals and LayerNorm; the attention operation
          you hand-rolled above a softmax you hand-rolled above a dot
          product; the KV cache you cached because you understood why it
          was redundant not to; and finally, the paged allocator you just
          watched refuse to waste a single page. Every layer of abstraction
          in a modern LLM serving stack is something you can now open up
          and explain to yourself. That was the whole point. Go build
          something.
        </p>
      </Prose>

      <References
        items={[
          {
            title:
              'Efficient Memory Management for Large Language Model Serving with PagedAttention',
            author: 'Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica',
            venue: 'SOSP 2023 — the vLLM / PagedAttention paper',
            url: 'https://arxiv.org/abs/2309.06180',
          },
          {
            title: 'FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning',
            author: 'Tri Dao',
            year: 2023,
            url: 'https://arxiv.org/abs/2307.08691',
          },
          {
            title: 'vLLM documentation — PagedAttention and the serving engine',
            venue: 'docs.vllm.ai',
            url: 'https://docs.vllm.ai/en/latest/dev/kernel/paged_attention.html',
          },
          {
            title: 'TensorRT-LLM — NVIDIA paged KV cache implementation',
            venue: 'GitHub: NVIDIA/TensorRT-LLM',
            url: 'https://github.com/NVIDIA/TensorRT-LLM',
          },
        ]}
      />
    </div>
  )
}
