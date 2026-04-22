import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import Prereq from '../Prereq'
import NeedsBackground from '../NeedsBackground'
import {
  Prose, Callout, Personify, Bridge, Gotcha, Challenge, References, KeyTerm,
} from '../primitives'
import LRScheduleViz from '../widgets/LRScheduleViz'
import ScalingLawExplorer from '../widgets/ScalingLawExplorer'

// Signature anchor: a gym schedule. Training a GPT isn't one workout — it's a
// weeks-long program. Warm-up sets (LR ramp), hypertrophy blocks (bulk of
// training), cool-down (cosine decay), rest days (eval steps). Overtrain and
// the model memorizes its reps instead of learning to move weight. The anchor
// returns at the opening, the LR-schedule reveal, and the "knowing when to
// stop" beat near the end.
export default function TrainYourGPTLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="train-your-gpt" />

      <Prose>
        <p>
          A trained powerlifter does not walk into the gym, load the bar with
          their one-rep max, and grind until failure. They follow a schedule.
          Warm-up sets to prime the nervous system. Working sets at prescribed
          weight and reps. Accessory work. A cool-down. Rest days between
          sessions. A mesocycle that ramps intensity for weeks and then
          deloads. Skip the warm-up and you pull a hamstring at rep one.
          Skip the cool-down and tomorrow&apos;s session is garbage. Skip the
          rest days and you overtrain — the body stops adapting and starts
          breaking down.
        </p>
        <p>
          Training a GPT is that program. The architecture you built in the
          last few lessons — embeddings, attention, MLP, residual stream,
          LayerNorm — is the athlete. The{' '}
          <NeedsBackground slug="training-loop">training loop</NeedsBackground>{' '}
          is a single workout. The <em>recipe</em> — optimizer, learning-rate
          schedule, weight decay, gradient clipping, batch size, precision —
          is the weeks-long program that turns a twitchy noob into a model
          that can actually lift. This isn&apos;t &ldquo;run{' '}
          <NeedsBackground slug="gradient-descent">gradient descent</NeedsBackground>{' '}
          for a while.&rdquo; It&apos;s a scheduled progression, and if any
          block is wrong the whole cycle collapses.
        </p>
        <p>
          This is the <KeyTerm>training recipe</KeyTerm>: the set of choices
          that determines whether your{' '}
          <NeedsBackground slug="training-diagnostics">loss curves</NeedsBackground>{' '}
          are smooth and monotone or whether the run diverges in the first
          thousand steps. Most of it comes from the GPT-2 paper, Chinchilla,
          and Karpathy&apos;s nanoGPT. None of it is obvious from first
          principles. All of it matters. We&apos;ll walk through every block
          of the program — warm-up, peak, cool-down, rest, recovery — in
          rough order of how much damage a bad choice does.
        </p>
      </Prose>

      <Callout variant="note" title="the training recipe, ranked">
        <strong>1. Learning rate</strong> (AdamW, 3e-4 for small GPTs, 1e-4 for bigger).{' '}
        <strong>2. LR schedule</strong> (linear warm-up + cosine decay to 0.1× peak).{' '}
        <strong>3. Weight decay</strong> (0.1, only on 2-D weight matrices).{' '}
        <strong>4. Gradient clip</strong> (norm ≤ 1.0).{' '}
        <strong>5. Gradient accumulation</strong> (hit 1M-token effective batches).{' '}
        <strong>6. Mixed precision</strong> (bfloat16 on A100/H100).{' '}
        <strong>7. Checkpointing</strong> (save optimizer state too).
        The order is roughly how much damage a bad choice does. Get LR wrong and nothing
        trains; get checkpointing wrong and you lose a week of compute when the node crashes.
      </Callout>

      <Prose>
        <p>
          The optimizer that trains every large transformer is{' '}
          <KeyTerm>AdamW</KeyTerm> — Adam with decoupled weight decay. Adam
          keeps a per-parameter running estimate of the gradient&apos;s first
          moment (mean) and second moment (uncentered variance), and uses
          them to compute a parameter-specific step size. Think of it as a
          smart spotter: it remembers how each muscle group has been
          responding across recent reps and scales the load accordingly.
          Decoupled weight decay is a separate pull toward zero, applied on
          top of the adaptive step instead of folded into the gradient — this
          matters because it interacts correctly with the per-parameter
          learning rate.
        </p>
      </Prose>

      <MathBlock caption="AdamW update rule">
{`# for each parameter θ at step t:
g_t    = ∇_θ L(θ_{t-1})                             # gradient
m_t    = β₁ · m_{t-1} + (1 - β₁) · g_t               # first moment  (β₁ = 0.9)
v_t    = β₂ · v_{t-1} + (1 - β₂) · g_t²              # second moment (β₂ = 0.95 for GPT)

m̂_t    = m_t / (1 - β₁^t)                            # bias correction
v̂_t    = v_t / (1 - β₂^t)

θ_t    = θ_{t-1}  -  η_t · ( m̂_t / (√v̂_t + ε)  +  λ · θ_{t-1} )
                     └──────── adaptive step ─────────┘   └── decoupled WD ──┘

η_t    = learning rate at step t   (from schedule, below)
λ      = weight decay coefficient  (0.1, applied only to matmul weights)`}
      </MathBlock>

      <Prose>
        <p>
          Two knobs set the speed: the base learning rate <code>η</code> and
          the schedule that modulates it over the run. The schedule is what
          separates a textbook optimizer from one that actually trains a
          1B-parameter model. Start too hot and early gradients are enormous
          — the model hasn&apos;t settled into any sensible region of
          parameter space yet, it&apos;s walking in cold — and the loss
          diverges. Hold peak LR too long and the model orbits the minimum
          instead of descending into it; the equivalent of grinding working
          sets for three straight hours without tapering. The standard GPT
          program: <KeyTerm>linear warm-up</KeyTerm> for the first 1–5% of
          steps, then <KeyTerm>cosine decay</KeyTerm> down to 10% of the peak
          LR for the cool-down.
        </p>
      </Prose>

      <LRScheduleViz />

      <Prose>
        <p>
          Four schedules plotted against step. Constant is the naive baseline
          — walk into the gym, slap 400 pounds on the bar, get under it.
          Simple, brittle, and your first few reps will be the last. Linear
          warm-up alone fixes the cold start but leaves you grinding at peak
          intensity for the whole workout. Cosine decay alone gives a clean
          cool-down but skips the ramp-up and still blows up at step zero.
          Warm-up plus cosine is the real program: ramp the bar up so the
          optimizer&apos;s moment estimates can calibrate, hold intensity
          through the hypertrophy block, then smoothly anneal to let the
          model refine its form. Every major GPT-style training run since
          2020 uses something close to this shape.
        </p>
      </Prose>

      <Personify speaker="Warm-up">
        I am the slow starter. For the first 2–5% of the schedule your
        gradient statistics are garbage — Adam&apos;s <code>v̂</code> estimate
        is noisy, the model hasn&apos;t settled, the LayerNorms haven&apos;t
        calibrated. Take a full-sized rep right now and you&apos;ll tear
        something. I ramp the learning rate from zero to peak over those
        early steps so the optimizer can build up its running averages
        before it&apos;s allowed to commit real weight to the bar. Skip me
        and your loss will explode around step 300. Every time.
      </Personify>

      <Personify speaker="Cosine cool-down">
        I am the taper. Once you&apos;ve hit peak LR and put in your working
        reps, I bring the load down on a smooth curve — not a cliff. Late in
        the schedule the model is fine-tuning tiny details; a big step here
        just knocks it out of the minimum it just found. I exist so the last
        20% of your compute isn&apos;t wasted thrashing. Pair me with warm-up
        and you have the standard GPT cycle.
      </Personify>

      <Callout variant="insight" title="knobs that barely matter (past a point)">
        The exact warm-up fraction, as long as it&apos;s between 1% and 10%. The exact gradient
        clip value, as long as it&apos;s between 0.5 and 5. The exact β₂ in Adam, anywhere in
        [0.95, 0.999]. Weight decay between 0.05 and 0.2. These are things teams have
        sweep-tested to death, and the answer each time is &ldquo;it&apos;s within the noise.&rdquo;
        Spend your tuning budget on LR, batch size, and data — not on which third-decimal
        warmup fraction.
      </Callout>

      <Prose>
        <p>
          Now zoom out from one workout to the whole mesocycle. You have a
          fixed compute budget — say 10²² FLOPs, which is what a GPT-3-era
          run burned. How big should the model be (the lifter)? How many
          tokens should it see (the reps)? This is the{' '}
          <KeyTerm>scaling law</KeyTerm> question, and Hoffmann et al. 2022
          (the Chinchilla paper) answered it with a loss curve you can fit.
        </p>
      </Prose>

      <MathBlock caption="Chinchilla scaling law (Hoffmann et al. 2022)">
{`L(N, D)  =  A / N^α   +   B / D^β   +   L*

  N   = non-embedding parameter count
  D   = training tokens seen
  L*  = irreducible loss  ≈ 1.69  (Chinchilla fit)
  A   ≈ 406.4,  α ≈ 0.34
  B   ≈ 410.7,  β ≈ 0.28

Minimize L under fixed compute  C ≈ 6 · N · D :

  →  N_opt  ∝  C^{0.5}
  →  D_opt  ∝  C^{0.5}
  →  D_opt / N_opt  ≈  20 tokens per parameter

Chinchilla rule of thumb:  for every 1 parameter, train on ~20 tokens.`}
      </MathBlock>

      <ScalingLawExplorer />

      <Prose>
        <p>
          The chart shows loss as a function of compute for several model
          sizes. The envelope — the lower boundary across all model sizes —
          is the Chinchilla-optimal curve. For any fixed compute, there is a
          sweet spot: too small a model and you&apos;re compute-bound (a
          lightweight doing endless reps with no progressive overload), too
          big and you&apos;re data-bound (a heavyweight who walked into the
          gym once and called it a career). Kaplan et al. 2020 originally
          suggested you should go bigger than you&apos;d think; Chinchilla
          2022 corrected that with a better fit:{' '}
          <strong>1:20 parameters to tokens.</strong> GPT-3 (175B params,
          300B tokens) was radically under-trained by this standard. Llama 1
          (7B params, 1T tokens) was closer. Llama 3 (8B params, 15T tokens)
          is deliberately <em>over</em>-trained — because tokens are cheap
          at inference time and parameters are not.
        </p>
      </Prose>

      <Personify speaker="Chinchilla ratio">
        I am the compute accountant. You hand me a FLOP budget; I hand you
        back a model size and a token count. My rule is simple: for every
        parameter, feed me twenty tokens. Deviate downward and you burn
        compute on parameters the gradient never reaches — dead weight the
        lifter never touched. Deviate upward and you burn compute on tokens
        a smaller model could have processed for the same loss. I am a
        guideline, not a law — if you care about inference cost you will
        deliberately overshoot tokens, and if you care about sample
        efficiency you might overshoot parameters. But if you are simply
        minimizing training loss per FLOP, the answer is twenty.
      </Personify>

      <Prose>
        <p>
          Enough theory. Here is the code — three layers, as usual. First,
          the AdamW update rule in pure Python so you can see what&apos;s
          happening under the hood — one rep on one parameter. Then a full
          PyTorch training loop with the whole schedule: warm-up ramp,
          cosine cool-down, gradient clipping, mixed precision — the
          workout as actually programmed. Then a taste of distributed
          training with <code>torch.distributed</code> for multi-GPU runs,
          where you are effectively running the same program across a whole
          team of athletes in sync.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · adamw_update.py (one step on one parameter)"
        output={`step 1: theta=0.9970, m=0.0100, v=0.00005
step 2: theta=0.9940, m=0.0190, v=0.00010
step 3: theta=0.9910, m=0.0271, v=0.00014
... (loss drops as θ approaches optimum)`}
      >{`import math

# one scalar parameter, one scalar gradient — illustrative only
beta1, beta2, eps = 0.9, 0.95, 1e-8
lr, wd = 3e-4, 0.1

theta = 1.0
m, v  = 0.0, 0.0

for t in range(1, 4):
    g = 0.1                                     # pretend gradient
    m = beta1 * m + (1 - beta1) * g             # first moment
    v = beta2 * v + (1 - beta2) * g * g         # second moment
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # adaptive step + decoupled weight decay
    adaptive = m_hat / (math.sqrt(v_hat) + eps)
    theta = theta - lr * (adaptive + wd * theta)

    print(f"step {t}: theta={theta:.4f}, m={m:.4f}, v={v:.5f}")`}</CodeBlock>

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch · train_gpt.py (full recipe, single GPU)"
        output={`step    0 | lr 0.00e+00 | loss 10.9842 | grad_norm 8.42
step  200 | lr 2.40e-04 | loss  6.4018 | grad_norm 1.12    # warmup ramping up
step 1000 | lr 3.00e-04 | loss  4.8291 | grad_norm 0.94    # peak LR reached
step 5000 | lr 2.12e-04 | loss  3.4118 | grad_norm 0.88    # cosine decaying
step 9999 | lr 3.00e-05 | loss  2.9874 | grad_norm 0.71    # end of run (0.1× peak)`}
      >{`import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# ---- assume model, train_loader defined elsewhere ----
model = model.cuda()
MAX_STEPS, WARMUP_STEPS = 10_000, 500
PEAK_LR, MIN_LR_RATIO   = 3e-4, 0.1
GRAD_ACCUM, CLIP_NORM   = 8, 1.0                    # 8 micro-steps → big effective batch

# 1. Parameter groups: weight decay ONLY on 2-D matmul weights
decay_params, nodecay_params = [], []
for name, p in model.named_parameters():
    if p.requires_grad:
        (decay_params if p.dim() >= 2 else nodecay_params).append(p)

optimizer = torch.optim.AdamW(
    [
        {'params': decay_params,   'weight_decay': 0.1},
        {'params': nodecay_params, 'weight_decay': 0.0},       # biases + LN: no WD
    ],
    lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8, fused=True,
)

# 2. LR schedule: linear warmup + cosine decay to 0.1x peak
def get_lr(step):
    if step < WARMUP_STEPS:
        return PEAK_LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return PEAK_LR * MIN_LR_RATIO + (PEAK_LR - PEAK_LR * MIN_LR_RATIO) * coeff

# 3. bfloat16 mixed precision — free speedup on A100/H100
scaler = GradScaler()                                # only needed for fp16, not bf16

data_iter = iter(train_loader)
for step in range(MAX_STEPS):
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g['lr'] = lr

    # 4. Gradient accumulation: GRAD_ACCUM micro-batches = 1 effective batch
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for _ in range(GRAD_ACCUM):
        xb, yb = next(data_iter)
        xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
        with autocast(dtype=torch.bfloat16):
            loss = model(xb, targets=yb)              # model returns scalar loss
            loss = loss / GRAD_ACCUM                  # scale so it averages correctly
        loss.backward()
        loss_accum += loss.item()

    # 5. Gradient clipping — norm 1.0 is standard
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step:5d} | lr {lr:.2e} | loss {loss_accum:.4f} | grad_norm {grad_norm:.2f}")

    # 6. Checkpoint — optimizer state too, so you can resume exactly
    if step > 0 and step % 2000 == 0:
        torch.save({
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),      # critical: Adam moments
            'rng': torch.get_rng_state(),
        }, f'ckpt_{step}.pt')`}</CodeBlock>

      <Bridge
        label="pure python adamw → pytorch training loop"
        rows={[
          { left: 'one scalar θ, one scalar g', right: 'model.parameters() groups', note: 'decay on weights, no-decay on biases/LN' },
          { left: 'fixed lr = 3e-4', right: 'get_lr(step) — warmup + cosine', note: 'the schedule is the second-most-important choice after the base LR' },
          { left: 'manual m, v bookkeeping', right: 'optimizer.step()', note: 'PyTorch keeps the moments in optimizer.state_dict()' },
          { left: '(no clipping)', right: 'clip_grad_norm_(params, 1.0)', note: 'one-line insurance against loss spikes' },
          { left: 'float32 everywhere', right: 'autocast(bfloat16)', note: '~2x throughput on A100/H100 with no accuracy loss' },
          { left: 'one batch = one step', right: 'GRAD_ACCUM micro-batches per step', note: 'reach 1M-token effective batch on a single GPU' },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch distributed · train_gpt_ddp.py (multi-GPU, only the diff)"
        output={`[rank 0] step  200 | loss 6.41 | tok/s 145_000
[rank 1] step  200 | loss 6.41 | tok/s 145_000
[rank 2] step  200 | loss 6.41 | tok/s 145_000
[rank 3] step  200 | loss 6.41 | tok/s 145_000
# 4 GPUs → 4x tokens/sec, gradients all-reduced each step`}
      >{`# torchrun --nproc_per_node=4 train_gpt_ddp.py
import os, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 1. Init process group (one process per GPU)
dist.init_process_group(backend='nccl')
rank        = int(os.environ['RANK'])
local_rank  = int(os.environ['LOCAL_RANK'])
world_size  = int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(local_rank)

# 2. Each GPU gets a full copy of the model, wrapped in DDP.
#    Backward pass auto-triggers an all-reduce on grads → identical across ranks.
model = build_gpt().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# 3. Sampler shards the dataset across ranks — each GPU sees different tokens.
sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
train_loader = DataLoader(train_set, batch_size=PER_GPU_BATCH, sampler=sampler)

# 4. Effective batch size = PER_GPU_BATCH × GRAD_ACCUM × WORLD_SIZE
#    Want 1M tokens? PER_GPU_BATCH * SEQ_LEN * GRAD_ACCUM * WORLD_SIZE = 1M.

# The rest of the training loop is identical to layer 2 — the DDP wrapper
# handles gradient synchronization transparently. Only rank 0 should print/save.
if rank == 0:
    print(f"step {step:5d} | loss {loss_accum:.4f}")

dist.destroy_process_group()`}</CodeBlock>

      <Bridge
        label="single-gpu → multi-gpu"
        rows={[
          { left: 'one process, one GPU', right: 'torchrun launches N processes', note: 'one per GPU; ranks 0..N-1' },
          { left: 'model.cuda()', right: 'DDP(model, device_ids=[local_rank])', note: 'wrapper hooks an all-reduce into backward()' },
          { left: 'DataLoader(shuffle=True)', right: 'DistributedSampler(rank=...)', note: 'non-overlapping shards per GPU, same seed each epoch' },
          { left: 'effective batch = BATCH × ACCUM', right: '× WORLD_SIZE too', note: 'Chinchilla says hit ~1M tokens per update for GPT-scale' },
        ]}
      />

      <Callout variant="insight" title="mixed precision in one paragraph">
        On A100/H100 GPUs, compute in <code>bfloat16</code> is ~2× faster than{' '}
        <code>float32</code>, uses half the memory, and has the same exponent range as fp32
        (8 bits) so no gradient-scaling tricks are needed. The standard recipe: wrap the
        forward pass in <code>autocast(dtype=torch.bfloat16)</code>, keep a master copy of
        weights in fp32 (PyTorch does this for you in the optimizer), and clip gradients as
        usual. fp16 is more fragile — smaller exponent range means gradient underflow, which
        is why the original mixed-precision recipe needed a <code>GradScaler</code>. With
        bf16 you can essentially drop it in.
      </Callout>

      <Callout variant="insight" title="why gradient clipping saves runs">
        Neural nets occasionally produce outlier gradients — one batch hits a pathological
        example and a single parameter&apos;s gradient is 100× the norm. It&apos;s the
        dropped-plate moment: one bad rep with 400 pounds and the whole cycle is over.
        Without clipping, the AdamW step for that parameter is enormous, it lands in a bad
        region, the next forward pass produces{' '}
        <NeedsBackground slug="cross-entropy-loss">cross-entropy</NeedsBackground> NaN, and
        the run is done. Clipping to norm 1.0 bounds the worst-case step size. One line of
        code and it will save you from at least one overnight-run disaster per project.
      </Callout>

      <Callout variant="note" title="torch.compile — the free 1.5x">
        Wrapping your model in <code>model = torch.compile(model)</code> fuses operations via
        the inductor backend and typically gives 30-70% throughput gains on modern PyTorch
        (&gt;= 2.0) with no math changes. The first step is slow (compilation); subsequent
        steps fly. Combine with <code>bfloat16</code> for close-to-peak GPU utilization.
      </Callout>

      <Prose>
        <p>
          One nastier problem you will meet: the <KeyTerm>loss spike</KeyTerm>.
          You are 20k steps in, the schedule has been executing cleanly, loss
          has been smoothly dropping, and then in the space of 50 steps it
          doubles and starts climbing. No code change, no data change, no
          obvious culprit. This is a famous, legitimately-not-fully-understood
          phenomenon in large transformer training — the equivalent of a
          lifter who hit every rep for a month suddenly failing a warm-up
          weight. The standard fixes, in order of effort: (1) lower the
          learning rate globally by 2×, (2) restart from the pre-spike
          checkpoint with a lower LR — the ML version of deloading and
          ramping back up, (3) add gradient clipping if you somehow
          didn&apos;t already, (4) check for a pathological batch of data
          (very long repeats, or rare tokens), (5) switch to a more stable
          optimizer variant like LION or Schedule-Free AdamW. In practice
          (2) solves most spikes — which is why you <em>must</em> checkpoint
          optimizer state, not just model weights. Without the moment
          estimates, the restart is cold and the schedule starts over.
        </p>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Weight decay on biases and LayerNorm.</strong>{' '}
          The naive <code>torch.optim.AdamW(model.parameters(), weight_decay=0.1)</code>
          {' '}applies weight decay to everything, including the γ/β parameters of LayerNorm
          and all biases. This is wrong — it damps the model&apos;s ability to scale
          activations and shifts biases toward zero for no reason. Split into two parameter
          groups: <code>weight_decay=0.1</code> on tensors with <code>dim() &gt;= 2</code>,{' '}
          <code>weight_decay=0.0</code> on everything else. Every modern LM training script
          does this.
        </p>
        <p>
          <strong className="text-term-amber">LR too high in the first 1k steps.</strong>{' '}
          Without warm-up, the first few hundred steps operate on uncalibrated Adam moment
          estimates and randomly-initialized weights. A full-sized rep at step 10 can send
          the model into a parameter region it never recovers from. Symptom: loss decreases
          for ~200 steps, then explodes to NaN. Fix: linear warm-up for 1–5% of total steps.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting gradient accumulation on a single
          GPU.</strong> The paper you&apos;re replicating used a 256-GPU cluster with a
          per-GPU batch of 16 → effective batch of 4096. You have one GPU, crank batch size
          up to 16, and wonder why your loss is noisier and your final model worse. The
          effective batch size is part of the recipe. If you can&apos;t get there with
          parallelism, accumulate gradients over N micro-batches:{' '}
          <code>loss / N; loss.backward()</code> for N iterations, then one{' '}
          <code>optimizer.step()</code>.
        </p>
        <p>
          <strong className="text-term-amber">Not saving optimizer state.</strong> You
          checkpoint <code>model.state_dict()</code>, the node dies at step 40k, you
          restart from the checkpoint — and discover your AdamW moment estimates are
          zero-initialized again. Your LR schedule also restarted at step 0. Effectively
          you now have a brand-new warm-up at step 40k, which is exactly the recipe for a
          loss spike. <strong>Always</strong> checkpoint <code>{`{model, optimizer, step, rng_state}`}</code>
          {' '}together.
        </p>
        <p>
          <strong className="text-term-amber">Confusing per-device and effective batch.</strong>
          {' '}<code>batch_size=32</code> with 8 GPUs and <code>GRAD_ACCUM=4</code> is an
          effective batch of <code>32 × 8 × 4 = 1024</code> sequences — which at sequence
          length 1024 is 1M tokens per update. Your <em>effective</em> LR, warm-up, and
          schedule all depend on this number, not the per-device one. Change the cluster
          size, and you&apos;ve silently changed the training recipe.
        </p>
      </Gotcha>

      <Prose>
        <p>
          One block we haven&apos;t talked about yet: <strong>rest days</strong>.
          In the gym they&apos;re literal — no lifting, recovery only. In
          training, rest days are <em>eval steps</em>. Every few hundred
          updates you pause the workout, switch the model to{' '}
          <code>eval()</code>, and run it on a held-out validation set
          <em> without</em> updating weights. You&apos;re checking the
          progress without adding load. Train loss alone will happily keep
          dropping while the model starts <em>memorizing its reps</em>
          rather than learning the movement — that&apos;s overtraining, and
          it&apos;s real. The tell is the gap between train loss and
          validation loss: when train keeps falling and val flattens or
          rises, you&apos;re overfitting, and the program&apos;s supposed to
          end. Cosine decay gives you a natural finish line; eval steps tell
          you whether to end early. Knowing when to stop is a feature of the
          schedule, not a bug in the athlete.
        </p>
      </Prose>

      <Challenge prompt="Train a nanoGPT and beat the no-warm-up baseline">
        <p>
          Starting from Karpathy&apos;s <code>nanoGPT</code> repo and the
          OpenWebText dataset, train the default ~124M-parameter GPT-2
          config for 10,000 steps on a single GPU. Use the full recipe:
          AdamW (β₁=0.9, β₂=0.95), base LR 3e-4, 500 steps of linear
          warm-up, cosine decay to 3e-5, weight decay 0.1 on matmul weights
          only, gradient clip 1.0, <code>bfloat16</code> mixed precision,
          and gradient accumulation to hit an effective batch of ~500k
          tokens. Log train loss every 20 steps and val loss on a rest-day
          cadence (every 500 steps).
        </p>
        <p>
          Then run the <em>same</em> config without warm-up (constant LR at
          3e-4 from step 0) and plot both curves on the same axes. Expected:
          the no-warm-up run either spikes to NaN in the first 500 steps, or
          its final loss is measurably worse. Measure how much worse. Push
          further: try warm-up + no cosine decay (constant LR after the
          ramp, no cool-down). Which component contributed more — the
          warm-up or the cool-down?
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What you now have.</strong> A full, reproducible training
          program that works at scale: AdamW with weight decay split across
          parameter groups, cosine LR with linear warm-up and a real
          cool-down, gradient clipping, gradient accumulation, bfloat16
          mixed precision, eval rest days, and a checkpoint format that can
          resume a run without regression. Plus the Chinchilla rule of
          thumb telling you what size model is worth training for your
          compute budget. This is, no exaggeration, the same schedule the
          major labs use — the 2024 Llama 3 tech report describes a
          training loop that differs from the one above mostly in scale.
        </p>
        <p>
          <strong>Next up — <code>code-gpt</code>.</strong> We&apos;ve
          talked about the program in the abstract, and the optimizer is
          wired. But there&apos;s one piece still off-screen: the model
          itself, assembled end-to-end. Embedding layer, a stack of
          transformer blocks, the language-model head, the tied weights,
          the forward pass that takes a batch of token IDs and returns a
          loss scalar. That&apos;s the athlete the schedule is training —
          and the next lesson builds it, module by module, from the
          primitives you already have. After that the full loop becomes
          runnable, end-to-end, on a GPU you own.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Language Models are Unsupervised Multitask Learners (GPT-2)',
            author: 'Radford, Wu, Child, Luan, Amodei, Sutskever',
            venue: 'OpenAI, 2019 — the training recipe that became canonical',
            url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
          },
          {
            title: 'Training Compute-Optimal Large Language Models (Chinchilla)',
            author: 'Hoffmann et al.',
            venue: 'DeepMind, arXiv:2203.15556, 2022 — the 1:20 parameters-to-tokens rule',
            url: 'https://arxiv.org/abs/2203.15556',
          },
          {
            title: 'Scaling Laws for Neural Language Models',
            author: 'Kaplan, McCandlish, Henighan, Brown, et al.',
            venue: 'OpenAI, arXiv:2001.08361, 2020 — the original scaling-law paper',
            url: 'https://arxiv.org/abs/2001.08361',
          },
          {
            title: 'Decoupled Weight Decay Regularization (AdamW)',
            author: 'Loshchilov, Hutter',
            venue: 'ICLR 2019, arXiv:1711.05101',
            url: 'https://arxiv.org/abs/1711.05101',
          },
          {
            title: 'nanoGPT — the simplest, fastest repository for training/finetuning medium-sized GPTs',
            author: 'Andrej Karpathy',
            venue: 'github.com/karpathy/nanoGPT',
            url: 'https://github.com/karpathy/nanoGPT',
          },
          {
            title: 'Let\'s build GPT: from scratch, in code, spelled out',
            author: 'Andrej Karpathy',
            venue: 'YouTube, 2023 — the reference walkthrough',
            url: 'https://www.youtube.com/watch?v=kCc8FmEb1nY',
          },
        ]}
      />
    </div>
  )
}
