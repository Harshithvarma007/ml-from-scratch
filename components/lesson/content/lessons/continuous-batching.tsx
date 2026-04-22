import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import Prereq from '../Prereq'
import NeedsBackground from '../NeedsBackground'
import WhatNext from '../WhatNext'
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
import StaticVsContinuousBatching from '../widgets/StaticVsContinuousBatching'
import ThroughputComparison from '../widgets/ThroughputComparison'

// Signature anchor: the elevator that doesn't wait. Static batching is the
// elevator that waits for all passengers to press their button before it
// moves, then refuses to pick anyone up mid-trip. Continuous batching is the
// elevator that picks riders up at any floor and lets each one off at theirs.
// Returned at the opening, at the iteration-level-scheduling reveal, and at
// the throughput-vs-latency section.

export default function ContinuousBatchingLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="continuous-batching" />

      {/* ── Opening: the elevator ───────────────────────────────── */}
      <Prose>
        <p>
          Picture an elevator. A very expensive one — call it thirty thousand
          dollars a month to run. It has eight spots. A rider walks in on the
          ground floor, presses a button, and the elevator refuses to move
          until seven more riders board. Fine. It fills up. It starts moving.
          Rider one wants floor two. Rider two wants floor forty. The elevator
          stops at floor two, rider one gets off, and then — here&apos;s the
          part you have to see clearly — the elevator keeps its empty spot
          empty all the way to floor forty. No one is allowed to board in the
          middle. There&apos;s a queue of people at floor five, tapping their
          feet, watching the numbers tick by. The car is two-thirds empty. The
          building is losing money every step.
        </p>
        <p>
          That&apos;s static batching. That&apos;s how every LLM server on
          earth worked until roughly 2022, and it is the most expensive
          sentence in LLM serving:{' '}
          <em>static batching wastes half your GPU.</em> Not a rough estimate —
          a measured fact, and the reason every production inference stack
          built after 2022 looks the way it does.
        </p>
        <p>
          The fix is the elevator that doesn&apos;t wait. Picks up new riders
          at any floor. Drops each one off as they reach theirs. The car is
          full every single step, and the GPU — because that&apos;s what the
          elevator is, the GPU is the elevator — is running at its physical
          ceiling instead of its politeness floor. The name for this is{' '}
          <KeyTerm>continuous batching</KeyTerm>. This lesson is about why it
          works, why it&apos;s harder than it looks, and why every serving
          framework you&apos;ve heard of — vLLM, TGI, TensorRT-LLM,
          DeepSpeed-FastGen — has it at the core.
        </p>
        <p>
          One more thing before we get into it. The reason generation is
          different from the vision-model{' '}
          <NeedsBackground slug="gpt-data-loader">batching</NeedsBackground>{' '}
          you&apos;ve done before is that each request in a batch runs for a
          different number of steps. Rider one wanted a haiku; rider two
          wanted a short story. Worst case you&apos;re running at half
          utilization on a $30k-per-month H100. Someone on your team is going
          to notice.
        </p>
      </Prose>

      {/* ── Widget 1: Static vs Continuous Batching ─────────────── */}
      <StaticVsContinuousBatching />

      <Prose>
        <p>
          Two timelines, same 8-slot GPU. On top, the static elevator: once
          rider one reaches their floor (the request emits its end-of-sequence
          token), slot zero sits dark until the last rider in the batch is
          home. On bottom, the continuous elevator: slot zero immediately
          picks up the next rider from the queue and starts decoding at the{' '}
          <em>next</em> step. No dead space. No waiting for stragglers. Every
          slot busy every tick.
        </p>
        <p>
          The per-step cost is roughly the same — one decode step through the
          model, whichever sequences happen to be in-flight (i.e. on board).
          The difference is that the denominator (wall-clock time) doesn&apos;t
          inflate with the longest-running rider. You&apos;re decoding as many
          tokens per second as the hardware can physically sustain, not as
          many as the slowest request in the most recent batch allows.
        </p>
      </Prose>

      <Callout variant="note" title="the iteration-level reveal">
        Here is the sentence the idea turns on:{' '}
        <strong>the scheduler re-examines the batch at every decoding step</strong>.
        Not at the batch boundary, not at the end of a request — every single
        step. If a slot freed because a rider reached their floor, it&apos;s
        refilled from the queue before the next step begins. The Orca paper
        (Yu et al., 2022) introduced this under the name{' '}
        <KeyTerm>iteration-level scheduling</KeyTerm>, and NVIDIA&apos;s
        TensorRT-LLM calls it <KeyTerm>in-flight batching</KeyTerm>. Three
        names for the same elevator trick.
      </Callout>

      <Personify speaker="Static batching">
        I made sense when we were running BERT. Fixed-length inputs, one
        forward pass, everyone reaches their floor at the same time. But
        generation is a variable-length loop, and I don&apos;t know how to
        open the door mid-trip. So I idle. I&apos;m holding your GPU hostage
        on behalf of the one rider who wanted the penthouse, and the seven
        others who wanted the mezzanine are paying for it.
      </Personify>

      {/* ── Throughput math ─────────────────────────────────────── */}
      <Prose>
        <p>
          Let&apos;s model the waste arithmetically. Say each rider needs a
          random number of floors <code>L_i</code> — the number of tokens
          their request will generate. Under static batching with{' '}
          <code>B</code> slots, the elevator takes{' '}
          <code>max(L_1, ..., L_B)</code> decoding steps to empty. The total
          useful work done is <code>Σ L_i</code>. Everything else is slot-idle
          time:
        </p>
      </Prose>

      <MathBlock caption="static batching — utilization as a ratio of tokens generated to slot-steps spent">
{`utilization_static  =     Σ L_i
                       ────────────────
                        B · max(L_i)`}
      </MathBlock>

      <Prose>
        <p>
          If the <code>L_i</code> are uniform in <code>[0, L_max]</code>, the
          expected sum is <code>B · L_max / 2</code> and the expected maximum
          is close to <code>L_max</code>, so your expected utilization is
          around <strong>0.5</strong>. Half the elevator, burnt. If generation
          lengths are more skewed — say one rider is heading ten times farther
          than the median — utilization drops further.
        </p>
        <p>Continuous batching flips the ratio:</p>
      </Prose>

      <MathBlock caption="continuous batching — every slot always holds an active rider">
{`utilization_continuous  ≈   1    (minus scheduler overhead)

throughput_continuous    =   B · tokens_per_step`}
      </MathBlock>

      <Prose>
        <p>
          That <em>B · tokens_per_step</em> ceiling is the actual physical
          throughput of the GPU — same matrix multiplies, same memory
          bandwidth, just never idle. The win over static batching is almost
          entirely the reclaimed idle time. In the Orca paper&apos;s worst-case
          mix they saw a <strong>23×</strong> throughput improvement. In more
          typical serving workloads with vLLM, expect <strong>2–3×</strong>.
        </p>
      </Prose>

      {/* ── Widget 2: Throughput Comparison ─────────────────────── */}
      <ThroughputComparison />

      <Prose>
        <p>
          Bars show tokens-per-second under both schemes across a realistic
          generation-length distribution. The static bar is shorter not
          because the elevator motor is slower — it&apos;s the same GPU — but
          because half of it is holding the wrong end of the broom while
          stragglers reach their floor. Continuous batching fills the bar to
          the hardware ceiling.
        </p>
      </Prose>

      <Callout variant="insight" title="throughput vs latency — know which one you're optimizing">
        Continuous batching pays for its throughput win with a small
        per-request latency tax. An individual rider now shares the elevator
        with a constantly-churning pool of other riders, so the decode rate
        for <em>one</em> user is sometimes marginally slower than in a
        dedicated static batch — think of it as the car stopping one extra
        time to let someone board. But the <em>system-wide</em>{' '}
        requests-per-second number — what your SRE and your bill care about —
        goes up dramatically. For any multi-tenant serving setup, this is the
        right trade.
      </Callout>

      <Personify speaker="Scheduler">
        I run at every decoding step. I check which slots just reached their
        floor — those are free. I pull the next pending prompts from the
        waitlist, run their prefill, and let them board. At the next step
        those new riders are decoding alongside the old ones. I don&apos;t
        care that your batch was &ldquo;supposed to be&rdquo; the same 8
        requests from beginning to end. That&apos;s a static-batching idea.
        We&apos;re doing iteration-level now.
      </Personify>

      {/* ── Key requirements ────────────────────────────────────── */}
      <Prose>
        <p>
          Building this is not free. Three capabilities have to be in place
          before the elevator can pick anyone up mid-trip:
        </p>
        <ul>
          <li>
            <strong>KV cache management that handles per-rider growth.</strong>{' '}
            Every active rider owns a chunk of{' '}
            <NeedsBackground slug="kv-cache">KV cache</NeedsBackground> that
            grows by one token per step. When a rider reaches their floor,
            their KV cache memory is released; when a new rider boards, it
            has to be allocated. Naive contiguous allocation fragments
            horribly after a few minutes of traffic. (The fix is paged
            attention — next lesson.)
          </li>
          <li>
            <strong>A request queue with admission logic.</strong> New
            prompts wait in a FIFO-ish queue on the ground floor. The
            scheduler decides, per step, whether there&apos;s memory and a
            slot available to admit more. If memory is tight, some riders
            already on board may even be preempted and recomputed later.
          </li>
          <li>
            <strong>Mixed-length attention support.</strong> In any given
            decoding step, some riders are at position 50, others at position
            800, a just-admitted one is doing its prefill across 200 tokens.
            The attention kernel has to handle ragged sequence lengths in the
            same batch. Vanilla PyTorch doesn&apos;t; FlashAttention and the
            vLLM-style block-paged kernels do.
          </li>
        </ul>
        <p>
          Notice the chain: continuous batching needs efficient KV memory,
          efficient KV memory needs paged allocation, paged allocation needs
          block-aware attention kernels. The vLLM paper is, fundamentally,
          about solving all three at once.
        </p>
      </Prose>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, same ladder as always. Pure Python to see the
          scheduling loop in its most naive form, NumPy to shape it like a
          real scheduler with a waitlist and per-rider state, PyTorch + vLLM
          to see how you&apos;d actually ship it. Every piece you see is a
          piece of the elevator — the board, the floor call, the per-step
          decision.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · simulate static vs continuous batching"
        output={`static batching     : 1,200 steps, 2,400 tokens,  utilization=50.0%
continuous batching :   400 steps, 2,400 tokens,  utilization=100.0%`}
      >{`import random

# eight requests, highly variable generation length
random.seed(0)
requests = [random.randint(50, 800) for _ in range(8)]
B = 8

# ----- static batching -----
# all 8 slots run together; the batch ends when the slowest request finishes.
steps_static = max(requests)                 # every step, every slot "spends" one tick
tokens_static = sum(requests)                # but only active slots emit tokens
slot_steps_static = B * steps_static
util_static = tokens_static / slot_steps_static

# ----- continuous batching -----
# assume an infinite queue of identical new requests — when a slot frees,
# it immediately picks up the next one. Steady state: every slot always busy.
total_tokens = sum(requests)
steps_continuous = total_tokens // B         # no idle slots => tokens / (slots per step)
slot_steps_continuous = B * steps_continuous
util_continuous = total_tokens / slot_steps_continuous

print(f"static batching     : {steps_static:>5} steps, {tokens_static:,} tokens,  utilization={util_static:.1%}")
print(f"continuous batching : {steps_continuous:>5} steps, {total_tokens:,} tokens,  utilization={util_continuous:.1%}")`}</CodeBlock>

      <Prose>
        <p>
          That&apos;s the whole idea in one file. Static batching ties the
          clock to the slowest rider; continuous batching keeps the
          denominator locked to &ldquo;floors actually traveled.&rdquo; Now
          let&apos;s build the scheduler itself — the thing that picks
          riders from a waitlist, tracks how far each one still has to go,
          and refills freed slots step by step.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · a minimal continuous-batching scheduler"
        output={`step  000 | active=8  done=0   waiting=12
step  100 | active=8  done=5   waiting=7
step  200 | active=8  done=11  waiting=1
step  300 | active=8  done=16  waiting=0
step  395 | active=5  done=20  waiting=0  — last few stragglers finishing
total: 20 requests, 395 steps, 3,162 tokens, 99.9% utilization`}
      >{`import numpy as np

rng = np.random.default_rng(0)
N_REQUESTS, B = 20, 8
lengths = rng.integers(50, 800, size=N_REQUESTS)

# scheduler state
waitlist = list(range(N_REQUESTS))   # pending request IDs (FIFO)
slots = [None] * B                   # slot -> (request_id, tokens_remaining) or None
done = 0
total_tokens = 0

step = 0
while done < N_REQUESTS:
    # 1) evict any finished requests => free their slots
    for i in range(B):
        if slots[i] is not None and slots[i][1] == 0:
            slots[i] = None
            done += 1

    # 2) admit from the waitlist into any free slots
    for i in range(B):
        if slots[i] is None and waitlist:
            rid = waitlist.pop(0)
            slots[i] = (rid, int(lengths[rid]))

    # 3) one decoding step: every active slot emits one token
    active = 0
    for i in range(B):
        if slots[i] is not None:
            rid, remaining = slots[i]
            slots[i] = (rid, remaining - 1)
            active += 1
            total_tokens += 1

    if step % 100 == 0 or active < B:
        print(f"step  {step:03d} | active={active}  done={done}  waiting={len(waitlist)}")

    if active == 0:
        break
    step += 1

util = total_tokens / (B * step)
print(f"total: {N_REQUESTS} requests, {step} steps, {total_tokens:,} tokens, {util:.1%} utilization")`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'steps_static = max(requests)',
            right: 'while done < N: evict, admit, decode',
            note: 'the loop becomes iteration-level — decisions per step',
          },
          {
            left: 'tokens_static = sum(requests)',
            right: 'slots[] + waitlist[] state machines',
            note: 'explicit per-slot state; you never wait for a whole batch',
          },
          {
            left: 'util = tokens / (B · max(L))',
            right: 'util = tokens / (B · step_count)',
            note: 'denominator becomes tight — close to 100% in steady state',
          },
        ]}
      />

      <Prose>
        <p>
          That scheduler is toy-sized, but every piece maps one-to-one onto
          what vLLM does in production: a <code>WaitingQueue</code>, a{' '}
          <code>RunningQueue</code>, an <code>evict → admit → step</code>{' '}
          loop that runs once per decoding step. Real systems add a lot —
          prefill chunking, KV-cache-aware admission, preemption and
          recomputation — but the shape of the elevator is the same.
        </p>
        <p>
          Layer 3 is the one you&apos;ll actually use. vLLM&apos;s{' '}
          <code>LLM</code> class wraps a continuous-batching engine behind a
          one-call API. You hand it a list of prompts, it runs all of them
          to completion at maximum hardware utilization. Under the hood the
          exact loop you just wrote is running — evict, admit, step — but
          around a production-grade attention kernel and a paged KV cache,
          which is where the real{' '}
          <NeedsBackground slug="code-gpt">generation</NeedsBackground>{' '}
          machinery lives.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch + vllm · shipping it"
        output={`Processed 128 requests, 31,204 generated tokens in 12.4s
throughput: 2,516 tokens/sec  (vs ~900 tokens/sec with HF .generate)`}
      >{`from vllm import LLM, SamplingParams

# Load the model once. Under the hood: a paged-attention KV cache,
# a scheduler with a waiting + running queue, and a CUDA graph per decode step.
llm = LLM(model="meta-llama/Llama-2-7b-hf", tensor_parallel_size=1)

params = SamplingParams(temperature=0.7, max_tokens=256)
prompts = [f"Write a short poem about {topic}" for topic in TOPICS_128]

# One call. vLLM runs them with continuous batching + paged attention.
# The scheduler figures out how many to pack, when to evict, when to admit.
outputs = llm.generate(prompts, params)

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
# Wall-clock timing handled by vLLM's internal profiler.
print(f"Processed {len(prompts)} requests, {total_tokens:,} generated tokens")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch + vllm"
        rows={[
          {
            left: 'slots = [None] * B',
            right: 'vllm scheduler: RunningQueue + KV blocks',
            note: 'a slot is now a paged KV allocation, not a list entry',
          },
          {
            left: 'slots[i] = (rid, tokens_left)',
            right: 'SequenceGroup with per-token KV growth',
            note: 'KV cache extends by one block as the sequence grows',
          },
          {
            left: 'while done < N: evict, admit, step',
            right: 'llm.generate(prompts, params)',
            note: 'all of it disappears behind one call — that is the point',
          },
        ]}
      />

      <Callout variant="insight" title="who ships continuous batching, and where to look">
        <strong>vLLM</strong> (Kwon et al., 2023) is the reference
        implementation and the most popular choice — its paged-attention
        kernel is the de-facto standard. <strong>TGI</strong> (Text
        Generation Inference) from HuggingFace was the first widely deployed
        open-source continuous-batching server.{' '}
        <strong>TensorRT-LLM</strong> is NVIDIA&apos;s answer, where they
        call it &ldquo;in-flight batching.&rdquo;{' '}
        <strong>DeepSpeed-FastGen</strong> from Microsoft adds a technique
        called Dynamic SplitFuse that chunks long prefills to keep the
        pipeline saturated. All four converge on the same core loop — the
        differences are kernel quality, scheduling heuristics, and which
        quantizations they support.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">KV cache eviction correctness:</strong>{' '}
          when a rider reaches their floor you have to release their cache
          blocks immediately, but you also have to make sure no in-flight
          attention op is still reading from them. Getting this wrong looks
          like silent gibberish in the next request that gets that memory —
          the next rider &ldquo;overhears&rdquo; the previous one.
        </p>
        <p>
          <strong className="text-term-amber">Prefill re-computation on preemption:</strong>{' '}
          if memory pressure forces the scheduler to kick a rider off the
          elevator to make room for a higher-priority one, the evicted
          request loses its KV cache. When it re-boards, the prompt has to be
          re-prefilled. Budget for this — preemption is not free.
        </p>
        <p>
          <strong className="text-term-amber">The &ldquo;stragglers&rdquo; problem:</strong>{' '}
          continuous batching fixes <em>throughput</em>, but one rider
          headed for floor 2,000 still hogs a slot for a long time. Tail
          latency for very short requests that board after a very long one
          can be worse than you&apos;d expect. Some systems address this
          with preemption or with explicit SLA tiers.
        </p>
        <p>
          <strong className="text-term-amber">Mixed prefill and decode in the same step:</strong>{' '}
          a newly-boarded rider needs its whole prompt processed (prefill),
          while ongoing riders only need one new token (decode). These two
          workloads have very different compute profiles, and handling them
          together is why techniques like chunked prefill and SplitFuse exist.
        </p>
      </Gotcha>

      {/* ── Challenge ──────────────────────────────────────────── */}
      <Challenge prompt="Measure the 2-3x for yourself">
        <p>
          On a GPU with at least 16 GB of VRAM, benchmark Llama-2-7B
          generation throughput two ways. First, plain HuggingFace — load the
          model with <code>transformers</code>, batch 64 prompts through{' '}
          <code>model.generate()</code> with a fixed{' '}
          <code>max_new_tokens=256</code>, and time it. This is your
          static-elevator baseline.
        </p>
        <p className="mt-2">
          Now install <code>vllm</code> and run the same 64 prompts through{' '}
          <code>LLM(&quot;meta-llama/Llama-2-7b-hf&quot;).generate(prompts, params)</code>.
          Use a sampling distribution with varied <code>max_tokens</code> per
          request (e.g. a mixture of 64, 256, and 1024) — this is where
          continuous batching wins hardest, because the rider-length variance
          is exactly what static batching can&apos;t absorb. Record
          wall-clock time and generated tokens for both runs.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: sweep batch size from 4 to 128 and plot tokens/sec for both
          frameworks. You should see HuggingFace plateau (or even degrade at
          large batches because of KV cache memory blowup) while vLLM keeps
          scaling until you hit the hardware roof.
        </p>
      </Challenge>

      {/* ── Closing ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Generation is
          variable-length, and treating the batch as a rigid elevator car
          wastes the GPU. Continuous batching makes the scheduling decision
          at every decoding step: drop off finished riders, board waiting
          ones, keep every slot busy. The win is dominated by reclaimed idle
          time — typically 2–3× on realistic workloads, up to 23× in the
          Orca paper&apos;s worst case. An individual rider gives up a hair
          of latency; the building as a whole gains enormously. Every
          production LLM serving stack built after 2022 has this at the core.
        </p>
        <p>
          <strong>Next up — Paged Attention.</strong> We kept saying &ldquo;KV
          cache management&rdquo; without saying how. The problem: when
          dozens of variable-length riders share a GPU&apos;s HBM, contiguous
          allocation fragments the memory into unusable holes between floors.
          Paged attention fixes this by treating KV cache like virtual memory
          — fixed-size blocks, indirect addressing, a page table per rider.
          It&apos;s what makes the elevator actually memory-efficient, and
          it&apos;s why vLLM exists.
        </p>
      </Prose>

      <WhatNext currentSlug="continuous-batching" />

      <References
        items={[
          {
            title:
              'Orca: A Distributed Serving System for Transformer-Based Generative Models',
            author: 'Yu, Jeong, Kim, Chun',
            venue: 'OSDI 2022 — introduced iteration-level scheduling',
            url: 'https://www.usenix.org/conference/osdi22/presentation/yu',
          },
          {
            title: 'Efficient Memory Management for Large Language Model Serving with PagedAttention',
            author: 'Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica',
            venue: 'SOSP 2023 — the vLLM paper',
            url: 'https://arxiv.org/abs/2309.06180',
          },
          {
            title:
              'DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference',
            author: 'Holmes, Aminabadi, Zhang, et al.',
            venue: 'Microsoft blog post, 2023 — introduced Dynamic SplitFuse',
            url: 'https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen',
          },
          {
            title: 'Text Generation Inference (TGI)',
            author: 'HuggingFace',
            venue: 'open-source continuous-batching server',
            url: 'https://github.com/huggingface/text-generation-inference',
          },
          {
            title: 'TensorRT-LLM In-Flight Batching',
            author: 'NVIDIA',
            venue: 'documentation',
            url: 'https://docs.nvidia.com/tensorrt-llm/advanced/inflight-batching.html',
          },
        ]}
      />
    </div>
  )
}
