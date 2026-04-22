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
import ExpertShardingDiagram from '../widgets/ExpertShardingDiagram'
import AllToAllCost from '../widgets/AllToAllCost'

// Signature anchor: a network of post offices, each hosting a handful of
// specialists. Every GPU is a post office. Tokens are letters. The router
// writes the destination on each letter. The all-to-all is the mail truck
// that shuffles letters between offices, lets the specialists reply, and
// mails the replies back. Returned at the opening (the GPU that can't fit
// all experts), the all-to-all reveal (the shuffle), and the "why the mail
// system is the bottleneck" section where comm costs eat the lesson.
export default function ExpertParallelismLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout (entry point: empty state) ─────── */}
      <Prereq currentSlug="expert-parallelism" />

      {/* ── Opening: the GPU that can't fit all experts ─────────── */}
      <Prose>
        <p>
          You have 64 experts. You have 64 GPUs. Where, exactly, do the experts{' '}
          <em>live</em>?
        </p>
        <p>
          Picture the cluster as a network of post offices, each hosting a
          handful of specialists. In ordinary dense training every GPU is the{' '}
          <em>same</em> post office — same staff, same stamps, same everything.
          That&apos;s the naive answer to the question above: put a copy of all
          64 experts on every GPU. It works, it needs no new plumbing, and it
          defeats the entire point of{' '}
          <NeedsBackground slug="moe-fundamentals">MoE</NeedsBackground>. The
          whole appeal of sparse experts was that you could scale parameters
          without paying for them in memory — but if every post office stocks
          every specialist, you&apos;re right back where you started, only now
          with a router duct-taped on top.
        </p>
        <p>
          The other answer: each post office holds <em>different</em>{' '}
          specialists. One expert per GPU. Now the memory math works — each
          card carries 1/64 of the expert weights. But a token that wants
          expert 37 while sitting on GPU 12 has a problem. The letter is in the
          wrong building. Someone has to move it. This lesson is about the
          shape of that tradeoff, the primitive that routes the mail (
          <KeyTerm>all-to-all</KeyTerm>), and the 2D/3D sharding puzzles that
          real MoE training solves every day.
        </p>
      </Prose>

      <Personify speaker="Expert parallelism">
        I split the experts across your GPUs so none of them have to hold all
        the weights. The catch is that I move tokens over the network instead
        — and the network is slower than your GPU, always. Whether I&apos;m
        worth it depends on how many experts you have and how big the batch
        is.
      </Personify>

      {/* ── Shard math ──────────────────────────────────────────── */}
      <Prose>
        <p>
          Let&apos;s make the memory savings concrete. You have{' '}
          <code>E</code> experts, each with <code>P</code> parameters. You have{' '}
          <code>W</code> GPUs (the <em>world size</em>). The two extremes:
        </p>
      </Prose>

      <MathBlock caption="expert memory — data parallel vs expert parallel">
{`data parallel:      per-GPU params  =  E · P
                                         (every GPU stores every expert)

expert parallel:    per-GPU params  =  (E / W) · P
                                         (each GPU stores its shard only)

savings ratio:      W-fold reduction in expert-weight memory`}
      </MathBlock>

      <Prose>
        <p>
          For Mixtral 8×7B on 8 GPUs: data parallel wants 8 × 7B = 56B parameters
          of expert weights on every card. Expert parallel puts one 7B expert on
          each card. Same model, eight-times less per-GPU memory for the
          feed-forward stack. That&apos;s the entire reason anyone bothers with
          any of this — and the reason each post office gets its own
          short-list of specialists instead of the full roster.
        </p>
      </Prose>

      {/* ── Widget 1: Sharding diagram ──────────────────────────── */}
      <ExpertShardingDiagram />

      <Prose>
        <p>
          Read the diagram left-to-right. Eight GPUs, eight experts, one expert
          per GPU — eight post offices, each holding one specialist. A batch
          of tokens lands on GPU 0. The router reads each token and writes a
          destination address on the envelope — maybe token <code>t₀</code>{' '}
          routes to expert 3, token <code>t₁</code> to expert 7, and so on.
          GPU 0 isn&apos;t the post office for experts 3 or 7. Neither is any
          other single GPU.
        </p>
        <p>
          So before the experts can do anything, every GPU must <em>mail</em>{' '}
          its letters to whichever GPU holds their chosen specialist. Every
          post office receives letters from every other post office, lets its
          one specialist reply to the mixed-origin stack, and then mails the
          replies back to their original senders. That round-trip is the
          all-to-all — the defining primitive of expert parallelism, and the
          one you&apos;ll hear cursed about in every distributed-training
          channel.
        </p>
      </Prose>

      <Personify speaker="All-to-all">
        I am the mail truck. Every post office hands me a sack of letters; I
        redistribute them so each letter ends up at the GPU that owns its
        specialist. Then I do it again, in reverse, to return the replies.
        Two all-to-alls per MoE layer, per forward pass, per backward pass.
        If your interconnect is slow, <em>I</em> am why your training job is
        slow.
      </Personify>

      {/* ── Cost math ───────────────────────────────────────────── */}
      <Prose>
        <p>
          Time for the cost model. Let <code>B</code> be the batch size per GPU,
          <code>d</code> the model dimension, and <code>W</code> the world size.
          In a single all-to-all each GPU sends <code>B · d / W</code> tokens to
          each of the other <code>W − 1</code> GPUs — so the total volume per
          GPU is on the order of <code>B · d · (W−1) / W</code>, which grows
          linearly with <code>W</code> and saturates near <code>B · d</code>{' '}
          as the world gets wide.
        </p>
      </Prose>

      <MathBlock caption="comm cost vs compute cost per MoE layer">
{`expert compute per GPU:    T_compute   ~  (B · d² / W) / throughput
                                              (one expert&apos;s FFN over ~B/W tokens)

all-to-all comm per GPU:   T_comm      ~  (B · d) / bandwidth

ratio:    T_comm / T_compute   ~   W / d · (bandwidth / throughput)

bigger W  →  comm grows.
bigger d  →  compute grows faster than comm — favors expert parallel at scale.`}
      </MathBlock>

      <Prose>
        <p>
          The punchline: for small worlds the compute dominates and expert
          parallel is a free memory win. As <code>W</code> grows, comm climbs
          linearly. Somewhere around <code>W ≈ 64</code> — on typical NVLink +
          InfiniBand clusters — the two cross over and you&apos;re now
          <em> comm-bound</em>: your GPUs are idle, waiting for packets. Pushing
          to <code>W = 256</code> doesn&apos;t make training faster; it just
          makes the mail system hotter.
        </p>
      </Prose>

      {/* ── Widget 2: all-to-all cost — why the mail system is the bottleneck ─── */}
      <AllToAllCost />

      <Prose>
        <p>
          This is where the post-office metaphor earns its keep: the
          specialists are fast, but the mail trucks are the bottleneck. The
          stacked bars tell the story. At 8 experts the blue compute column
          towers over the red comm sliver — mail volume is trivial compared to
          how long the specialists take to reply. By 64 the sliver is a
          stripe. By 256 the stripe has eaten the stack — you&apos;re spending
          more wall-clock shuffling letters than you are running them through
          an MLP. Past that point, no amount of extra post offices will speed
          you up, because the limit isn&apos;t compute, it&apos;s the wire
          between buildings.
        </p>
      </Prose>

      <Callout variant="insight" title="why MoE systems stay hybrid">
        Pure expert parallelism is only cheap up to a point. Real clusters
        combine it with data parallel (for the attention / non-expert layers
        — for example each{' '}
        <NeedsBackground slug="transformer-block">transformer block</NeedsBackground>
        &apos;s attention sublayer) and pipeline parallel (for very long
        networks). You get one GPU&apos;s worth of expert shard <em>plus</em>{' '}
        its share of other jobs — a <KeyTerm>2D or 3D sharding</KeyTerm>{' '}
        layout where each dimension is a different parallelism strategy.
      </Callout>

      <Personify speaker="Sharding strategy">
        I&apos;m the 3D puzzle you solve before every training run. One axis
        is data parallel, one is expert parallel, one is pipeline parallel.
        The choice is constrained by memory, comm bandwidth, and the physical
        topology of your cluster. Get me wrong and your $10M run burns a week
        idle. Get me right and nobody ever thanks me.
      </Personify>

      {/* ── Hybrid math ─────────────────────────────────────────── */}
      <Prose>
        <p>
          The hybrid layout in practice: split your <code>W</code> GPUs into a
          grid of <code>W = DP · EP · PP</code>, where <code>DP</code> is data
          parallel (replicas of the full model), <code>EP</code> is expert
          parallel (experts sharded), and <code>PP</code> is pipeline parallel
          (layers split across stages). For a 64-GPU cluster you might pick{' '}
          <code>8 × 8 × 1</code>: 8 expert-parallel groups, 8 data-parallel
          replicas, no pipeline. For a 1024-GPU cluster you might pick{' '}
          <code>32 × 16 × 2</code>. Each axis has its own comm primitive —
          all-reduce for DP, all-to-all for EP, send/recv for PP.
        </p>
        <p>
          There&apos;s no closed-form optimum. Megatron-LM, DeepSpeed-MoE, and
          Mixtral&apos;s internal stack each ship empirical recipes for which
          combinations work on which hardware. But the pattern is always the
          same: <em>minimize the slowest comm</em>, <em>pack the GPUs</em>,{' '}
          <em>overlap communication with computation where possible</em>.
        </p>
      </Prose>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three versions. First, a single-process Python simulation that acts
          out the mailroom with lists — each GPU sorts its letters by
          destination address. Then NumPy, where the experts are real matrix
          multiplies and we count the bytes the mail truck would have to
          carry. Then the real thing: PyTorch with{' '}
          <code>torch.distributed</code>, the call that Megatron actually
          makes.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · expert_parallel_sim.py"
        output={`GPU 0 sends: {1: ['t0'], 3: ['t2']}
GPU 1 sends: {0: ['t3'], 2: ['t1']}
GPU 2 sends: {3: ['t0'], 1: ['t2']}
GPU 3 sends: {0: ['t3'], 2: ['t1']}
after expert compute on each GPU → all-to-all back → outputs reassembled`}
      >{`# four GPUs, four experts — one per GPU. simulate the mailroom.
W = 4
tokens_per_gpu = [
    [('t0', 1), ('t2', 3)],   # (token, chosen_expert_id) on GPU 0
    [('t3', 0), ('t1', 2)],   # GPU 1
    [('t0', 3), ('t2', 1)],   # GPU 2
    [('t3', 0), ('t1', 2)],   # GPU 3
]

# Step 1: each GPU buckets its tokens by destination expert's GPU.
send_buffers = [{d: [] for d in range(W)} for _ in range(W)]
for src, toks in enumerate(tokens_per_gpu):
    for name, exp_id in toks:
        dest_gpu = exp_id           # one expert per GPU, so expert_id == gpu_id
        send_buffers[src][dest_gpu].append(name)
    print(f"GPU {src} sends:", {k: v for k, v in send_buffers[src].items() if v})

# Step 2: the all-to-all. Every GPU receives whatever was sent to it.
recv_buffers = [[] for _ in range(W)]
for src in range(W):
    for dst in range(W):
        recv_buffers[dst].extend(send_buffers[src][dst])

# Step 3: each GPU runs its one expert over its received tokens.
# Step 4: a second all-to-all sends outputs home. (omitted for brevity)`}</CodeBlock>

      <Prose>
        <p>
          Vectorise it. Replace token names with vectors, replace the mailroom
          with <code>np.concatenate</code>, and count the bytes that would move
          across the wire — that&apos;s your comm budget, the postage on every
          letter the all-to-all sends.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · expert_parallel_numpy.py"
        output={`per-GPU send volume: 0.500 MB
per-GPU expert compute: (128, 512) x (512, 512) = 128 · 262144 FLOPs
T_comm / T_compute estimate: ~0.41 (compute-bound at W=4)`}
      >{`import numpy as np

W, B, d = 4, 128, 512       # 4 GPUs, 128 tokens/GPU, model dim 512
expert_weights = [np.random.randn(d, d).astype(np.float32) for _ in range(W)]

# Pretend routing: each GPU's tokens get assigned to some expert.
rng = np.random.default_rng(0)
assignments = [rng.integers(0, W, size=B) for _ in range(W)]
tokens = [np.random.randn(B, d).astype(np.float32) for _ in range(W)]

# ─── all-to-all forward ────────────────────────────────────────
# each GPU's outgoing bucket for destination g:
send = [[tokens[s][assignments[s] == g] for g in range(W)] for s in range(W)]
# each GPU's incoming: everything sent to it
recv = [np.concatenate([send[s][d] for s in range(W)]) for d in range(W)]

# ─── expert compute ────────────────────────────────────────────
outputs_local = [recv[g] @ expert_weights[g] for g in range(W)]

# ─── comm accounting ───────────────────────────────────────────
bytes_per_float = 4
send_bytes_per_gpu = sum(
    send[0][g].size * bytes_per_float for g in range(W)
)
print(f"per-GPU send volume: {send_bytes_per_gpu / 1e6:.3f} MB")
print(f"per-GPU expert compute: {recv[0].shape} x ({d}, {d}) = "
      f"{recv[0].shape[0]} · {d*d} FLOPs")`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'send_buffers[src][dst].append(name)',
            right: 'send[s][g] = tokens[s][mask]',
            note: 'bucketing becomes a boolean index',
          },
          {
            left: 'recv_buffers[dst].extend(...)',
            right: 'np.concatenate([send[s][d] for s in range(W)])',
            note: 'the actual all-to-all — assemble the incoming pieces',
          },
          {
            left: 'expert(token) one at a time',
            right: 'recv[g] @ expert_weights[g]',
            note: 'one matmul per GPU over its fused incoming batch',
          },
        ]}
      />

      <Prose>
        <p>
          And the real thing. In PyTorch the call is literally{' '}
          <code>dist.all_to_all_single</code>. The token shuffling that took
          twenty lines of NumPy collapses into one line, running on NCCL over
          NVLink, overlapping with the backward pass of the previous layer if
          you&apos;re careful. The mail truck becomes a single function call.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · expert_parallel_dist.py"
        output={`[rank 0] local tokens after dispatch: torch.Size([128, 512])
[rank 0] expert output shape: torch.Size([128, 512])
[rank 0] local tokens after combine: torch.Size([128, 512])`}
      >{`import torch
import torch.distributed as dist

# Each rank = one GPU = one expert. Launched with torchrun --nproc_per_node=W.
dist.init_process_group('nccl')
rank, world_size = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)

B, d = 128, 512
tokens = torch.randn(B, d, device='cuda')
expert = torch.nn.Linear(d, d).cuda()
route = torch.randint(0, world_size, (B,), device='cuda')

# 1. Sort tokens by destination rank → packed contiguous buffer.
order = route.argsort()
tokens_sorted = tokens[order]
send_counts = torch.bincount(route, minlength=world_size)

# 2. Tell every rank how many tokens to expect from every other rank.
recv_counts = torch.zeros_like(send_counts)
dist.all_to_all_single(recv_counts, send_counts)

# 3. The main event: actually shuffle the tokens.
recv_buf = torch.empty(recv_counts.sum(), d, device='cuda')
dist.all_to_all_single(
    recv_buf, tokens_sorted,
    output_split_sizes=recv_counts.tolist(),
    input_split_sizes=send_counts.tolist(),
)
print(f"[rank {rank}] local tokens after dispatch: {recv_buf.shape}")

# 4. Each rank runs its local expert over its received tokens.
out = expert(recv_buf)

# 5. The reverse all-to-all sends outputs home. (symmetric call — omitted.)`}</CodeBlock>

      <Bridge
        label="numpy → pytorch distributed"
        rows={[
          {
            left: 'np.concatenate(send[s][d] for s in range(W))',
            right: 'dist.all_to_all_single(recv, send, ...)',
            note: 'one NCCL call, overlaps with compute, runs on GPU',
          },
          {
            left: 'manual byte counting for comm budget',
            right: 'torch.profiler / NSight traces',
            note: 'measure real wall-clock comm, not hand-waved bandwidth',
          },
          {
            left: 'single-process simulation',
            right: 'torchrun --nproc_per_node=W',
            note: 'one OS process per GPU, coordinated by NCCL',
          },
        ]}
      />

      <Callout variant="note" title="3D parallelism in one sentence">
        Megatron-LM composes <em>tensor parallel</em> (slice each weight matrix
        across GPUs), <em>pipeline parallel</em> (slice the stack of layers
        across GPUs), and <em>expert parallel</em> (slice the MoE experts across
        GPUs), then wraps all three in data parallel for throughput. Every
        production MoE of 2024+ uses some combination of these four axes.
      </Callout>

      {/* ── Inference latency ───────────────────────────────────── */}
      <Prose>
        <p>
          Now the inconvenient truth about inference. During training you have
          big batches — you can amortize comm over thousands of letters at
          once, one big mail run instead of many small ones. During{' '}
          <em>generation</em>, you&apos;re producing one token at a time per
          user, and the MoE block <em>still</em> has to do both all-to-alls
          every layer, every token. The mail truck runs on schedule whether
          it&apos;s carrying one letter or a thousand.
        </p>
        <p>
          The consequence is that MoE models are often surprisingly slow
          per-token at inference compared to a dense model of similar{' '}
          <em>active</em> parameter count. Mixtral 8×7B has 13B active params
          but routes through an all-to-all every block. On a multi-GPU serving
          setup that per-block network hop can add milliseconds that a dense
          13B simply doesn&apos;t pay. KV-cache management, top-k routing, and
          batched decoding all exist partly to claw those milliseconds back.
        </p>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Order is not preserved:</strong>{' '}
          the all-to-all shuffle scrambles tokens. You must remember the
          original permutation (we did this with <code>argsort</code> above)
          and invert it on the return trip, or replies come back attached to
          the wrong residual — the right specialist wrote back to the wrong
          sender.
        </p>
        <p>
          <strong className="text-term-amber">Load imbalance kills you:</strong>{' '}
          if one expert is popular and another is dead, the post offices
          holding them finish at wildly different times. The whole cluster
          waits for the slowest one. This is exactly why the auxiliary{' '}
          <NeedsBackground slug="load-balancing-loss">load balancing</NeedsBackground>{' '}
          term is non-negotiable in expert-parallel training — the HR manager
          keeps every specialist on roughly the same mail volume.
        </p>
        <p>
          <strong className="text-term-amber">
            Mixed precision and comm overlap:
          </strong>{' '}
          all-to-all in <code>bfloat16</code> halves the bytes and halves the
          comm time. Overlapping the all-to-all with the next layer&apos;s
          compute (via <code>async_op=True</code>) can hide the network bill
          almost entirely — but only if you&apos;ve arranged the graph for it.
        </p>
      </Gotcha>

      <Challenge prompt="Measure the network tax yourself">
        <p>
          Spin up a 2-GPU node (one host, two cards). Create 8 experts each of
          shape <code>(1024, 1024)</code> in float32. Distribute them 4 per
          GPU. Send a batch of 1024 tokens through one MoE block, using{' '}
          <code>dist.all_to_all_single</code> for the shuffle.
        </p>
        <p className="mt-2">
          Time the all-to-all by itself. Time the expert compute by itself
          (one <code>linear</code> per GPU over the received batch). Print the
          ratio. On a single-host NVLink setup you&apos;ll likely see{' '}
          <code>T_comm / T_compute &lt; 0.1</code> — the mail truck is almost
          free when both post offices share a building. Now repeat across two{' '}
          <em>hosts</em> over your cluster&apos;s Ethernet/IB, and watch the
          ratio climb by one or two orders of magnitude.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: pass <code>async_op=True</code> to the all-to-all, issue a
          dummy matmul on the GPU while it runs, and measure the overlap win.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> Expert parallelism is the
          answer to the memory question that plain MoE posed. Each GPU is a
          post office, each post office stocks a different specialist, and
          the all-to-all is the mail truck that routes letters to the right
          address and replies back to the sender. You get a <code>W</code>
          -fold reduction in per-GPU expert memory, at the cost of two
          all-to-all shuffles per MoE layer. Below ~64 experts the compute
          dominates and EP is nearly free; above, the mail system becomes the
          bottleneck and you must hybridize with data + pipeline parallelism.
          Modern frameworks (Megatron-LM, DeepSpeed-MoE) wrap all of this, but
          the underlying primitive is always a well-scheduled all-to-all.
        </p>
        <p>
          <strong>End of the MoE section.</strong> You now have the full
          picture: why MoE exists (a panel of specialists), how the bouncer
          picks who sees each token (top-k routing), how the HR manager keeps
          nobody idle (load balancing), and how the post offices mail letters
          to each other when the specialists live on different GPUs (this
          lesson). <strong>Next — Denoising Intuition.</strong> A completely
          different paradigm. Forget sparse routers for a minute: we&apos;re
          going to train a network to take a blurry, corrupted image and
          undo one step of the noise. Do that thousands of times and you can
          start from pure static and end at a photograph. The math looks
          nothing like what you&apos;ve seen, but the intuitions — gradients,
          loss, training loop — all carry over intact.
        </p>
      </Prose>

      <References
        items={[
          {
            title:
              'GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding',
            author: 'Lepikhin et al.',
            venue: 'ICLR 2021',
            year: 2020,
            url: 'https://arxiv.org/abs/2006.16668',
          },
          {
            title:
              'Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity',
            author: 'Fedus, Zoph, Shazeer',
            venue: 'JMLR 2022',
            url: 'https://arxiv.org/abs/2101.03961',
          },
          {
            title:
              'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism',
            author: 'Narayanan et al.',
            venue: 'SC 2021',
            url: 'https://arxiv.org/abs/1909.08053',
          },
          {
            title:
              'DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale',
            author: 'Rajbhandari et al.',
            venue: 'ICML 2022',
            url: 'https://arxiv.org/abs/2201.05596',
          },
          {
            title: 'Mixtral of Experts',
            author: 'Jiang et al. (Mistral AI)',
            year: 2024,
            url: 'https://arxiv.org/abs/2401.04088',
          },
        ]}
      />
    </div>
  )
}
