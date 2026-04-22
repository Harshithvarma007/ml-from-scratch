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
import ExpertCollapse from '../widgets/ExpertCollapse'
import LBLossComputation from '../widgets/LBLossComputation'

// Signature anchor: the overworked star vs the idle bench. Every MoE trained
// without a penalty collapses into a favorite — expert #1 drowning while
// experts #2-8 play solitaire. The load-balancing loss is the HR manager
// that fines the router whenever the workload gets lopsided. The anchor
// returns at the opening, at the aux-loss reveal, and at the "coefficient
// too high" failure-mode section.

export default function LoadBalancingLossLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="load-balancing-loss" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Picture your shiny new <NeedsBackground slug="moe-fundamentals">MoE</NeedsBackground> layer
          on day one. Eight experts, a top-2 router, clean forward pass, coffee in hand. You kick
          off training. One step. Ten. A thousand. Then you peek at which expert the router is
          actually sending tokens to — and the dashboard is a horror show.{' '}
          <strong>Expert #1 is seeing 80% of the tokens.</strong> Two neighbors are splitting the
          scraps. The other five are sitting on the bench, randomly initialized, never trained,
          effectively dead weight. You built a mixture of experts and ended up with{' '}
          <em>one overworked star</em> and seven paperweights playing solitaire.
        </p>
        <p>
          This is the single most famous failure mode of sparse MoE, and it&apos;s the subject of
          this lesson. We&apos;ll see how the collapse happens, why it&apos;s a natural
          equilibrium and not a bug, and the auxiliary loss term — the{' '}
          <KeyTerm>load balancing loss</KeyTerm> — that keeps the router honest. Small term, tiny
          coefficient, huge difference. Think of it as the HR manager who walks the floor and
          fines the router every time the workload gets lopsided. By the end you&apos;ll
          understand why every production MoE stack (Switch, GShard, Mixtral) pays the same ~0.01
          HR tax on every gradient step.
        </p>
      </Prose>

      {/* ── Widget 1: Expert Collapse ───────────────────────────── */}
      <ExpertCollapse />

      <Prose>
        <p>
          Watch the <em>without</em> trace. At step 0 tokens are evenly distributed — the router
          is random, so it routes randomly. Within a couple hundred steps one expert starts
          pulling ahead. Within a thousand it&apos;s a runaway. The curve never recovers.
          That&apos;s <KeyTerm>rich-get-richer</KeyTerm> dynamics: the expert that saw slightly
          more tokens early gets slightly better at its job, the router&apos;s quality signal
          routes slightly more tokens <em>its</em> way, it gets better still, and so on. The
          feedback loop is exponential. The star gets overworked, the bench stays idle, and the
          router&apos;s favorite compounds.
        </p>
        <p>
          The cruel part: the router is doing nothing wrong. Gradient descent is doing exactly
          what it&apos;s supposed to — picking the expert that produces the lowest task loss for
          each token. The trouble is that loss is a function of how well-trained the expert is,
          and how well-trained the expert is, is a function of how often the router picks it. So
          the system collapses into the first attractor it finds. If your init is
          <em>anything</em> other than perfectly uniform — and it never is — one expert wins the
          popularity contest and the other seven are benched forever.
        </p>
      </Prose>

      <Personify speaker="Load balancer">
        I&apos;m the HR manager bolted onto your task loss. I don&apos;t care about accuracy. I
        care that every expert on the payroll is pulling their weight — no favorites, no idle
        bench, no one drowning. Push me too hard and I flatten your router into noise. Push me
        too soft and your router picks a favorite, the workload goes lopsided, and the whole
        point of sparsity evaporates. Tune me with care.
      </Personify>

      {/* ── The LB loss formula ─────────────────────────────────── */}
      <Prose>
        <p>
          The fix is an <KeyTerm>auxiliary loss</KeyTerm> — a small extra term added to the task
          loss whose entire job is to fine the router whenever the workload goes lopsided. Switch
          Transformer (Fedus et al., 2021) wrote down the canonical version, and every modern MoE
          has inherited it with minor tweaks. It is, in one line of math, the HR policy:
        </p>
      </Prose>

      <MathBlock caption="load balancing auxiliary loss — Switch Transformer">
{`              N
L_aux   =  N · Σ  f_i · P_i
             i=1

where for each expert i of N total experts:

   f_i  =  fraction of tokens dispatched to expert i   (hard, discrete)
   P_i  =  average router probability for expert i     (soft, differentiable)

and the total loss is:
   L_total  =  L_task  +  α · L_aux       (α = 0.01 in Switch)`}
      </MathBlock>

      <Prose>
        <p>
          Two pieces, and the cleverness is in the pairing. <code>f_i</code> measures the actual
          routing decision — of this batch&apos;s tokens, what fraction got shoved onto expert{' '}
          <code>i</code>? This is a hard count: it uses the <code>argmax</code> (or top-k) of the
          router, so it&apos;s not differentiable. You can&apos;t backprop through{' '}
          <code>f_i</code>. <code>P_i</code>, on the other hand, is the average of the raw router
          probabilities coming out of the{' '}
          <NeedsBackground slug="softmax">softmax</NeedsBackground> — the pre-argmax soft score.
          That <em>is</em> differentiable. It&apos;s what the router would have sent if routing
          were soft.
        </p>
        <p>
          Multiplying them is the whole trick. The gradient flows through <code>P_i</code>, and{' '}
          <code>f_i</code> rides along as a constant weight that tells the router where the
          lopsidedness is. If expert 3 is the overworked star right now, <code>f_3</code> is huge;
          the gradient on <code>P_3</code> gets a big coefficient, and the router learns to turn
          down the dial on expert 3. If expert 5 is the idle bench, <code>f_5</code> is tiny and{' '}
          <code>P_5</code> barely moves — but the balance force still quietly favors spreading
          work <em>its</em> way. Equilibrium lies at <code>f_i = P_i = 1/N</code> for every
          expert. That is the HR manager&apos;s definition of a balanced staff roster.
        </p>
      </Prose>

      {/* ── Widget 2: LB Loss Computation ───────────────────────── */}
      <LBLossComputation />

      <Prose>
        <p>
          Walk through the widget column by column. You have a batch of tokens, a router that
          produces a <code>softmax</code> over experts, and a top-1 assignment. Column{' '}
          <code>f_i</code> counts assignments and divides by batch size. Column <code>P_i</code>{' '}
          averages raw probabilities down the batch. The product column is what gets summed and
          scaled by <code>N</code>. Drag the skew slider: as one expert&apos;s share climbs,{' '}
          <code>f_i</code> and <code>P_i</code> both climb for that expert, the workload tips
          lopsided, and the loss explodes. Drag it back to uniform and the loss bottoms out at{' '}
          <code>1.0</code> — the minimum for a perfectly balanced router, the HR manager&apos;s
          dream shift.
        </p>
      </Prose>

      <Callout variant="insight" title="why the minimum is exactly 1">
        By Cauchy–Schwarz (or just AM-GM on N equal terms), for non-negative vectors{' '}
        <code>f</code> and <code>P</code> that each sum to 1, the sum{' '}
        <code>Σ f_i · P_i</code> is minimized when both vectors are uniform:{' '}
        <code>f_i = P_i = 1/N</code>. Plug that in: <code>N · N · (1/N)(1/N) = 1</code>. So a
        well-balanced MoE pays a constant aux loss of ~1; a collapsed one pays <code>N</code>{' '}
        times that. The gradient of <code>L_aux</code> is therefore a force pushing the router
        toward uniformity. Always.
      </Callout>

      <MathBlock caption="minimum at the uniform distribution">
{`Constraint:    Σ f_i = 1,   Σ P_i = 1,    f_i, P_i ≥ 0

Claim:         Σ f_i · P_i   ≥   1/N      with equality iff f_i = P_i = 1/N.

Intuition:     sum is a dot product ⟨f, P⟩. Under the simplex
               constraint it's minimized by the "most spread out"
               vectors — both uniform. Then N · 1/N  =  1.`}
      </MathBlock>

      <Personify speaker="Expert capacity">
        I&apos;m the hard cap. The aux loss is the HR manager with a clipboard; I&apos;m the
        bouncer at the door. If expert 3 has already accepted <code>c</code> tokens this batch,
        the 4th one gets turned away — routed to its second choice or skipped entirely. No
        overflow, no OOM, no single worker drowning while the bench idles. Pick me too tight,
        most tokens skip and the layer barely computes anything. Too loose, I do nothing and
        you&apos;re back to praying the aux loss holds.
      </Personify>

      <Prose>
        <p>
          <KeyTerm>Expert capacity</KeyTerm> is the hard, non-differentiable complement to the
          soft aux loss. You compute it as{' '}
          <code>capacity = (tokens_per_batch / N) · capacity_factor</code>, where the{' '}
          <code>capacity_factor</code> is usually between <code>1.0</code> (strict) and{' '}
          <code>1.25</code> (lenient). A token that picks an expert already at capacity gets
          either (a) routed to its next-choice expert, or (b) dropped — passed through the
          residual stream unchanged, no expert computation at all. This is why well-trained MoEs
          tolerate some capacity pressure: the residual path carries enough signal that
          occasional skips don&apos;t destroy the forward pass.
        </p>
      </Prose>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three phases, same formula. First the pure-NumPy walk, term by term, so the math is
          legible. Then a PyTorch version using <code>scatter_add</code> — the vectorised trick
          every real MoE uses on GPU. Then the HF Transformers one-liner you&apos;ll actually
          call in production.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · lb_loss_scratch.py"
        output={`f = [0.75  0.125 0.0625 0.0625]
P = [0.68  0.15  0.095 0.075]
f · P = [0.51   0.01875 0.005938 0.004688]
L_aux = 4 * sum(f · P) = 2.151`}
      >{`import numpy as np

# Pretend we just ran a router on a batch of 16 tokens over N=4 experts.
N = 4
# Router probabilities BEFORE argmax — shape (batch, N). These are soft; gradient flows here.
router_probs = np.array([
    [0.7, 0.2, 0.05, 0.05],    # token 0 strongly prefers expert 0
    [0.8, 0.1, 0.05, 0.05],    # token 1 prefers expert 0
    [0.6, 0.3, 0.05, 0.05],
    [0.75, 0.15, 0.05, 0.05],
    # ...collapsed router — expert 0 is winning everything
] + [[0.7, 0.15, 0.1, 0.05]] * 12)

# Hard assignment — this is f_i (non-differentiable).
assignments = np.argmax(router_probs, axis=1)                 # shape (batch,)
f = np.bincount(assignments, minlength=N) / len(assignments)  # fraction per expert

# Soft average — this is P_i (differentiable).
P = router_probs.mean(axis=0)                                 # shape (N,)

# The aux loss.
L_aux = N * np.sum(f * P)

print("f =", np.round(f, 4))
print("P =", np.round(P, 4))
print("f · P =", np.round(f * P, 6))
print(f"L_aux = {N} * sum(f · P) = {L_aux:.3f}")`}</CodeBlock>

      <Prose>
        <p>
          Now the same thing on GPU. The idiom is <code>scatter_add</code>: given a tensor of
          expert IDs per token, bump a counter at each expert index in parallel. You see this
          pattern in every serious MoE implementation because a Python loop over tokens would be
          catastrophic at scale.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch · lb_loss_torch.py"
        output={`L_aux = 1.073   (close to ideal 1.0 because init is roughly uniform)`}
      >{`import torch
import torch.nn.functional as F

def load_balancing_loss(router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    router_logits: (batch, num_experts) — raw scores from the gate
    returns: scalar aux loss (no coefficient applied)
    """
    # Soft probabilities — gradient flows through these.
    probs = F.softmax(router_logits, dim=-1)                # (B, N)

    # Hard top-1 assignment per token.
    expert_idx = probs.argmax(dim=-1)                       # (B,) long

    # f_i: fraction of tokens routed to each expert. scatter_add builds the histogram.
    ones = torch.ones_like(expert_idx, dtype=probs.dtype)
    tokens_per_expert = torch.zeros(num_experts, device=probs.device, dtype=probs.dtype)
    tokens_per_expert.scatter_add_(0, expert_idx, ones)     # bump per-expert counter
    f = tokens_per_expert / expert_idx.numel()              # (N,)

    # P_i: mean router probability per expert.
    P = probs.mean(dim=0)                                    # (N,)

    # Aux loss.
    return num_experts * (f * P).sum()

torch.manual_seed(0)
logits = torch.randn(64, 8)                                  # 64 tokens, 8 experts
print(f"L_aux = {load_balancing_loss(logits, num_experts=8).item():.3f}")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'np.bincount(assignments, minlength=N)',
            right: 'torch.zeros(N).scatter_add_(0, idx, ones)',
            note: 'same histogram, built on-device with no host sync',
          },
          {
            left: 'router_probs.mean(axis=0)',
            right: 'probs.mean(dim=0)',
            note: 'identical — named dim instead of axis',
          },
          {
            left: 'L_aux = N * (f * P).sum()',
            right: 'num_experts * (f * P).sum()',
            note: 'scalar tensor — gradient flows through P automatically',
          },
        ]}
      />

      <Prose>
        <p>
          In real code you don&apos;t write any of this. <code>transformers</code> ships it as
          part of its Mixtral / Switch implementations. You pass <code>output_router_logits=True</code>{' '}
          on the forward pass and the aux loss is added to your outputs, ready to scale and
          combine with the task loss.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — huggingface · lb_loss_mixtral.py"
        output={`task loss:     2.413
aux loss:      1.082
combined:      2.4238  (task + 0.01 * aux)`}
      >{`from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func
import torch

name = "mistralai/Mixtral-8x7B-v0.1"           # 8 experts, top-2 routing
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)

ids = tok("Sparse activation is the whole point.", return_tensors="pt").input_ids
out = model(ids, labels=ids, output_router_logits=True)

# The Mixtral model exposes all-layer router logits; the helper walks them.
aux = load_balancing_loss_func(
    out.router_logits,
    num_experts=8,
    top_k=2,
    attention_mask=None,
)

total = out.loss + 0.01 * aux                  # 0.01 is the Switch-style coefficient
print(f"task loss:     {out.loss.item():.3f}")
print(f"aux loss:      {aux.item():.3f}")
print(f"combined:      {total.item():.4f}  (task + 0.01 * aux)")`}</CodeBlock>

      <Bridge
        label="pytorch → huggingface"
        rows={[
          {
            left: 'load_balancing_loss(logits, N)',
            right: 'load_balancing_loss_func(router_logits, N, top_k, mask)',
            note: 'same algorithm, generalised over layers, top-k, and padding',
          },
          {
            left: 'loss = task_loss + 0.01 * aux',
            right: 'out.loss + 0.01 * aux',
            note: 'you add the coefficient yourself — HF returns them separately',
          },
          {
            left: 'you compute f and P by hand',
            right: 'output_router_logits=True',
            note: 'opt-in flag; adds one more tensor per layer to the forward pass',
          },
        ]}
      />

      <Callout variant="insight" title="expert choice — the sneaky alternative">
        Zhou et al. (2022) noticed that the whole circus — aux loss, capacity factor, overflow
        handling — exists because <em>tokens pick experts</em>. Flip it. Let each{' '}
        <em>expert</em> pick its top-<code>c</code> tokens instead. Every expert gets exactly{' '}
        <code>c</code> tokens by construction, so load is balanced by construction and nobody
        gets benched. The catch: a token might not be picked by <em>any</em> expert, in which
        case it&apos;s skipped. Tradeoff, not free lunch — and it&apos;s incompatible with
        autoregressive decoding, because expert-choice routing needs to see the whole batch at
        once. Fine for encoders, awkward for decoders, so most LLM MoEs still pay the HR manager
        and live with token-choice + aux loss.
      </Callout>

      <Callout variant="note" title="hybrid approaches in the wild">
        DeepSeekMoE (2024) splits experts into &ldquo;shared&rdquo; (always active) and
        &ldquo;routed&rdquo; (load-balanced), plus adds a finer-grained{' '}
        <NeedsBackground slug="cross-entropy-loss">auxiliary loss</NeedsBackground> per device for
        communication efficiency. Grok-1 uses top-2 with explicit capacity limits. Mixtral is the
        vanilla Switch recipe scaled up. The common thread: everyone pays some form of balance
        tax. No one ships a production MoE with token-choice routing and no regulation — the
        favorite emerges within hours and the bench never gets off the bench.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Aux loss coefficient too high:</strong> bump{' '}
          <code>α</code> from <code>0.01</code> to <code>0.1</code> and the HR manager turns into
          a micromanager. The balance penalty drowns out the task signal, the router gets fined
          so aggressively it gives up and routes uniformly to everything, and every expert sees
          every kind of token. No favorites, sure — but also no specialisation. You&apos;ve
          flattened the staff into eight interchangeable generalists, which is just a slow dense
          layer wearing a MoE costume. The lopsidedness was the symptom; killing it with fire
          kills the whole point of sparsity.
        </p>
        <p>
          <strong className="text-term-amber">Capacity factor too tight:</strong> set{' '}
          <code>capacity_factor = 1.0</code> on a noisy router and ~20% of your tokens will
          overflow and get skipped every batch. Those tokens contribute nothing to the forward
          pass. Your loss curve flatlines early and you wonder why. Bump it to{' '}
          <code>1.25</code>.
        </p>
        <p>
          <strong className="text-term-amber">Not logging per-expert utilisation:</strong> the
          aux loss value alone doesn&apos;t tell you if you&apos;ve actually converged to
          balance. You need to plot <code>f_i</code> over training — is every expert getting
          hits, or is the aux loss low only because <code>P_i</code> is flat while{' '}
          <code>f_i</code> is still concentrated on one favorite and the rest still idle? Log
          both.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting the <code>N</code> scale factor:</strong>{' '}
          the leading <code>N</code> in <code>L_aux</code> is what makes the loss scale-invariant
          across expert counts. Drop it and your 64-expert model has a much smaller aux loss than
          your 8-expert model at the same degree of imbalance, and your coefficient sweep stops
          generalising.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Watch the collapse, then prevent it">
        <p>
          Build a 4-expert MoE in PyTorch: one linear router into a softmax, four tiny expert
          MLPs (just <code>nn.Linear(d, d)</code> each), a top-1 dispatch, and a synthetic task
          — predict a random label from a 32-dim input, 10k steps, batch 128.
        </p>
        <p className="mt-2">
          <strong>Run A (no aux loss):</strong> train to convergence. Every few hundred steps,
          log the histogram of expert assignments and compute the entropy of the empirical
          distribution: <code>H = -Σ f_i log f_i</code>. Uniform would give{' '}
          <code>log(4) ≈ 1.386</code>. You&apos;ll watch it drop toward zero as one expert takes
          over and the other three end up on the idle bench.
        </p>
        <p className="mt-2">
          <strong>Run B (with aux loss):</strong> add the LB loss term with coefficient{' '}
          <code>0.01</code>. Same task, same seed. The entropy should stay near{' '}
          <code>log(4)</code> for the whole run — the HR manager is doing its job and nobody is
          getting overworked.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: sweep <code>α ∈ {'{'}0, 0.001, 0.01, 0.1, 1.0{'}'}</code> and plot final task
          accuracy against final routing entropy. You&apos;ll see the characteristic U —
          extremes both hurt (no fine = collapse, huge fine = forced uniformity), the sweet
          spot is narrow.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> A vanilla token-choice MoE collapses because
          the training dynamics favor whichever expert got a head start — one overworked star
          emerges, the rest of the bench stays idle, and the router never looks back. The load
          balancing auxiliary loss — <code>N · Σ f_i · P_i</code> — is the small, cheap,
          differentiable HR regulator that prevents that collapse by pairing a hard assignment
          fraction with a soft router probability, funneling gradient toward uniform routing
          without killing the router&apos;s ability to specialise. Pair it with a hard expert
          capacity limit for the extreme cases. Every serious MoE stack pays some version of
          this balance tax.
        </p>
        <p>
          <strong>Next up — Expert Parallelism.</strong> So far we&apos;ve been computing
          everything on one device, and everyone on the payroll has a desk in the same office.
          In reality, if you have 64 experts at 7B parameters each, they don&apos;t fit on a
          single GPU — they don&apos;t even fit on a single <em>node</em>. The next lesson is
          about splitting experts across devices: all-to-all communication, the cost of token
          shuffling between nodes, why MoE throughput is bounded by interconnect bandwidth, and
          the tricks (ZeRO, tensor parallelism interleaving) that let Mixtral actually run in
          the wild.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity',
            author: 'Fedus, Zoph, Shazeer',
            venue: 'JMLR 2022 — originally arXiv 2021',
            url: 'https://arxiv.org/abs/2101.03961',
          },
          {
            title: 'Mixture-of-Experts with Expert Choice Routing',
            author: 'Zhou et al.',
            venue: 'NeurIPS 2022',
            url: 'https://arxiv.org/abs/2202.09368',
          },
          {
            title: 'GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding',
            author: 'Lepikhin et al.',
            venue: 'ICLR 2021 — originally arXiv 2020',
            url: 'https://arxiv.org/abs/2006.16668',
          },
          {
            title: 'Mixtral of Experts',
            author: 'Jiang et al. (Mistral AI)',
            year: 2024,
            url: 'https://arxiv.org/abs/2401.04088',
          },
          {
            title: 'DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models',
            author: 'Dai et al.',
            year: 2024,
            url: 'https://arxiv.org/abs/2401.06066',
          },
        ]}
      />
    </div>
  )
}
