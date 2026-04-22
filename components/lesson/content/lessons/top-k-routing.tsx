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
import RouterGating from '../widgets/RouterGating'
import KSweep from '../widgets/KSweep'

// Signature anchor: the bouncer at the door, picking exactly k experts per
// token from the guest list. Introduced in the opening (crowded vs exclusive
// room), returned to at the argmax vs top-k non-differentiability reveal,
// and at the "how k changes the tradeoff" section. Distinct from the MoE
// fundamentals anchor (panel of specialists) and load-balancing (HR manager).
export default function TopKRoutingLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout ────────────────────────────────── */}
      <Prereq currentSlug="top-k-routing" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Picture a room with <code>N</code> specialist experts inside and a single
          door. Every token in your batch is a guest approaching that door one at a
          time. Who gets in? If you throw the door wide open and let everyone enter
          every room, you have <code>N</code> experts doing work per token — a dense
          model wearing a sparse costume, and you just paid <code>N×</code> the
          compute bill for nothing. If you slam the door shut, nobody runs and your
          model outputs zeros. Neither extreme is what{' '}
          <NeedsBackground slug="moe-fundamentals">MoE</NeedsBackground> promised
          you.
        </p>
        <p>
          You need a bouncer. The bouncer stands at the door with a clipboard, reads
          each token&apos;s ID, scores every expert on the guest list against that
          ID, and waves exactly <code>k</code> names through. Everyone else stays
          outside for this token. Next token walks up, gets a fresh scan, different
          <code>k</code> names get waved through. The room is never crowded and
          never empty — it&apos;s always running at exactly <code>k</code> experts
          per guest, which is the one setting where sparse MoE actually pays off.
        </p>
        <p>
          That bouncer is the <KeyTerm>router</KeyTerm>: a tiny linear layer that
          scores every expert for every token, plus a{' '}
          <KeyTerm>top-k selection</KeyTerm> step that keeps only the best few
          names on the list. It&apos;s three lines of code and it&apos;s the reason
          sparse models exist. This lesson covers what the bouncer actually is, why
          we pick <code>k</code> experts instead of <code>1</code> or <code>N</code>,
          the noisy top-k trick that keeps the whole thing from collapsing in
          training, and how to write it in pure Python, NumPy, and PyTorch.
        </p>
      </Prose>

      <Personify speaker="Router">
        I am a <code>Linear(d_model, n_experts)</code>. That is all. A{' '}
        <NeedsBackground slug="softmax">softmax</NeedsBackground> on top of me
        turns my logits into gate probabilities, and <em>top-k</em> picks the
        winners I wave through the door. I&apos;m three hundred parameters guarding
        a billion. Treat me carefully.
      </Personify>

      {/* ── Router math ─────────────────────────────────────────── */}
      <Prose>
        <p>
          Start with one token, vector <code>x ∈ ℝ^d</code> — the guest&apos;s ID
          the bouncer reads. The router is a single matrix{' '}
          <code>W_r ∈ ℝ^(d × N)</code>. Multiply to get a score per expert, softmax
          to turn raw scores into a probability distribution over the{' '}
          <code>N</code> names on the list:
        </p>
      </Prose>

      <MathBlock caption="router — logits and gate probabilities">
{`logits   =   x · W_r               ∈ ℝᴺ

g(x)ᵢ    =   exp(logitsᵢ)
             ────────────────────    softmax over experts
             Σⱼ exp(logitsⱼ)`}
      </MathBlock>

      <Prose>
        <p>
          Now <code>g(x)</code> is a probability vector over experts — it sums to 1
          and every entry is positive. In a <em>dense</em> MoE, the output is just{' '}
          <code>Σᵢ g(x)ᵢ · Eᵢ(x)</code>: run <em>every</em> expert, weight each
          output by its gate probability, sum. That&apos;s the &ldquo;open door&rdquo;
          version — correct, differentiable, and financially ruinous. If you have
          128 experts you just did 128× the work of a dense model with one FFN.
        </p>
        <p>
          Sparse MoE makes a different bet: most of <code>g(x)</code> is
          approximately zero anyway. Only the top few entries carry real weight.
          Keep those, drop the rest — the bouncer only waves through the names that
          actually matter. That&apos;s top-k.
        </p>
      </Prose>

      <MathBlock caption="top-k selection and renormalization">
{`TopK(g, k) = { indices of the k largest entries of g }

g̃(x)ᵢ  =  {  g(x)ᵢ / Σⱼ∈TopK g(x)ⱼ     if i ∈ TopK
          {  0                           otherwise

y      =   Σᵢ∈TopK  g̃(x)ᵢ · Eᵢ(x)`}
      </MathBlock>

      <Prose>
        <p>
          Three things just happened at the door. The bouncer picked the{' '}
          <code>k</code> best-scoring names from the list. Their gates got
          renormalized so they still sum to 1 (otherwise the total gate mass drops
          below 1 and every residual connection starts to drift). And{' '}
          <em>only those</em> <code>k</code> experts actually ran — not{' '}
          <code>N</code>. The FLOPs per token are now <code>k · C_expert</code>{' '}
          instead of <code>N · C_expert</code>, independent of how many experts
          you have. Adding the 128th expert costs parameters but not compute.
          That&apos;s the whole game.
        </p>
      </Prose>

      {/* ── Widget 1: Router Gating ─────────────────────────────── */}
      <Prose>
        <p>
          Below: a batch of token embeddings flowing through a router with 8
          experts. Watch the logits, the softmax, the top-k cut, and the
          renormalization. The chosen names on the list highlight; the rest fall
          dark at the door. Change <code>k</code> and see the mass redistribute.
        </p>
      </Prose>

      <RouterGating />

      <Callout variant="note" title="argmax vs top-k — the non-differentiability gotcha">
        Return to the bouncer. When <code>k = 1</code> this is literally{' '}
        <em>argmax</em>: pick the single highest score, everyone else goes home.
        Argmax is a step function in logit space — nudge a logit by a hair and
        nothing changes, nudge it past a boundary and the whole choice flips. No
        gradient flows through &ldquo;which name was on the list.&rdquo; Top-k for
        <code>k &gt; 1</code> is the same story, just with <code>k</code> winners
        instead of one. Discrete. Non-differentiable. Full stop. But the gate
        weights <code>g̃(x)ᵢ</code> <em>are</em> differentiable (they&apos;re a
        softmax divided by a softmax), and those are what the chosen experts&apos;
        outputs get multiplied by. So the router learns by re-weighing the names
        it waved through, not by reconsidering whose name was on the list. This is
        sufficient because top-k is stable: small logit changes don&apos;t flip
        which expert wins, so gradients through the weights match gradients
        through the choice almost everywhere.
      </Callout>

      <Personify speaker="Router">
        I don&apos;t get gradient through <em>who</em> I pick at the door. I get
        gradient through <em>how much</em> I weight them once they&apos;re inside.
        Turns out that&apos;s plenty — if my top name keeps giving good outputs,
        the softmax will keep voting for it, and its share of the gate will keep
        growing.
      </Personify>

      {/* ── Hard vs soft ─────────────────────────────────────────── */}
      <Prose>
        <p>
          Worth being explicit about the trade-off. <KeyTerm>Soft routing</KeyTerm>{' '}
          is the open-door policy: every name on the list gets the token, weighted
          by <code>g(x)</code>. Fully differentiable, slightly more accurate, and
          it nukes the entire reason to use MoE — you&apos;re doing{' '}
          <code>N×</code> the FLOPs of a single FFN.{' '}
          <KeyTerm>Hard routing</KeyTerm> (top-k with <code>k &lt; N</code>) is the
          bouncer: only the chosen <code>k</code> experts run. The sparsity is the
          payoff; the non-differentiable selection is the cost. Every production
          MoE pays that cost gladly.
        </p>
      </Prose>

      {/* ── FLOPs math ──────────────────────────────────────────── */}
      <MathBlock caption="flops per token: dense vs sparse MoE">
{`Dense FFN              =   C_expert                    (one FFN)

Soft MoE (k = N)       =   N · C_expert                 (every expert runs)

Sparse MoE (top-k)     =   k · C_expert  +  d · N        (k experts + router)
                              ▲                ▲
                              dominant         rounding-error cost

Typical:   d = 4096,  N = 64,  k = 2
           router  ≈  4096 · 64       ≈  0.26M flops
           experts ≈  2 · 50M         ≈  100M flops
           → k dominates; router is free`}
      </MathBlock>

      <Prose>
        <p>
          So the slider that matters is <code>k</code>: it scales FLOPs linearly
          while quality gains plateau fast. Play with it.
        </p>
      </Prose>

      {/* ── Widget 2: k Sweep ───────────────────────────────────── */}
      <KSweep />

      <Prose>
        <p>
          Drag <code>k</code> from 1 to <code>N</code>. The FLOPs line is a ruler —
          perfectly linear in <code>k</code>. The quality line is not: it rises
          sharply from <code>k=1</code> to <code>k=2</code>, creeps up a bit to{' '}
          <code>k=4</code>, and flattens out. Return to the bouncer — this is how{' '}
          <code>k</code> changes the tradeoff at the door, and the published
          choices follow the curve exactly.
        </p>
        <ul>
          <li>
            <strong>k=1 — Switch Transformer.</strong> The simplest, fastest, and —
            thanks to Fedus et al. 2021 — the one that showed you can scale MoE to
            trillions of parameters without the training instability everyone
            feared. A hard choice: the bouncer reads the ID, picks one name, done.{' '}
            <NeedsBackground slug="load-balancing-loss">load balancing</NeedsBackground>{' '}
            is doing a lot of heavy lifting to keep it stable.
          </li>
          <li>
            <strong>k=2 — Mixtral, GLaM, ST-MoE.</strong> Twice the FLOPs of{' '}
            <code>k=1</code>, a meaningful quality bump, and enough redundancy that
            a single bad pick at the door doesn&apos;t tank a token&apos;s forward
            pass. Top expert plus a backup — a tie-break built in. This is the
            current default.
          </li>
          <li>
            <strong>k=4+ — diminishing returns, free lunch over.</strong>{' '}
            You&apos;re buying noise. The room gets crowded, the top two names
            already carry the vast majority of the probability mass on most tokens,
            and adding a third or fourth barely changes the weighted sum. The
            sparsity discount shrinks while the quality line has already flattened.
          </li>
        </ul>
      </Prose>

      <Personify speaker="Top-k">
        I&apos;m the capacity planner at the door. Pick me too small and one lucky
        expert eats the whole batch. Pick me too big and the room is crowded and
        you&apos;ve paid for a dense model in expert&apos;s clothing. Two. The
        answer is usually two.
      </Personify>

      {/* ── Noisy top-k ─────────────────────────────────────────── */}
      <Callout variant="insight" title="noisy top-k — the Shazeer 2017 trick">
        Early in training, the bouncer is basically flipping coins. The expert
        that happens to win the first few tokens gets a gradient signal, its
        logits grow, it wins more tokens, and so on. Within a few hundred steps
        you can find yourself with one name on the list handling 80% of tokens
        and the other 63 experts standing idle outside. Classic rich-get-richer
        collapse. The fix: add Gaussian noise to the router logits <em>before</em>{' '}
        top-k selection:{' '}
        <code>noisy_logits = logits + N(0, σ) · softplus(W_noise · x)</code>. The
        noise breaks ties and lets other names on the list occasionally win,
        giving them gradient and a chance to specialize. Turn σ down toward zero
        over training. This is the single most important trick keeping MoE
        training from eating itself.
      </Callout>

      <Prose>
        <p>
          There&apos;s also a train / inference split worth knowing about. Some
          recipes use dense (soft) routing during training so every expert gets
          gradient every step, and switch to sparse top-k only at inference for
          speed. Others use top-k throughout and rely on noisy top-k plus a load
          balancing loss to keep experts healthy. Reinforce-style gradients —
          treating expert selection as a policy and getting gradient through the
          discrete choice via score-function estimators — have also been explored
          but aren&apos;t the standard today. Straight noisy top-k with a balance
          loss is the current best practice.
        </p>
      </Prose>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations. The pure-Python version shows the skeleton — what
          top-k <em>means</em> when you spell out the bouncer line by line. NumPy
          introduces <code>argpartition</code>, the best-of-both-worlds trick
          (partial sort for free). PyTorch hands you <code>torch.topk</code> and{' '}
          <code>scatter</code> and you use them like any other op.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · top_k_scratch.py"
        output={`logits       = [0.8, -0.2, 1.5, 0.3, -1.1, 2.1, 0.0, 0.9]
gates        = [0.14, 0.05, 0.29, 0.09, 0.02, 0.53, 0.07, 0.16]
top-2 idxs   = [5, 2]
top-2 gates  = [0.64, 0.36]   # renormalized, sums to 1.0`}
      >{`import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]                   # subtract max for stability
    z = sum(exps)
    return [e / z for e in exps]

def top_k_gate(logits, k):
    gates = softmax(logits)                                # probability per expert

    # argsort descending, take first k indices
    ranked = sorted(range(len(gates)), key=lambda i: gates[i], reverse=True)
    top_idx = ranked[:k]

    # renormalize: gates / sum(top-k gates)
    top_mass = sum(gates[i] for i in top_idx)
    top_gates = [gates[i] / top_mass for i in top_idx]

    return top_idx, top_gates

logits = [0.8, -0.2, 1.5, 0.3, -1.1, 2.1, 0.0, 0.9]
idx, gates = top_k_gate(logits, k=2)
print("top-2 idxs  =", idx)
print("top-2 gates =", [round(g, 2) for g in gates])`}</CodeBlock>

      <Prose>
        <p>
          Vectorise. In NumPy we process an entire batch at once — the bouncer
          checks every guest&apos;s ID in parallel — and we use{' '}
          <code>argpartition</code> to avoid paying for a full sort just to find
          the top-k. Partitioning is <code>O(N)</code> instead of{' '}
          <code>O(N log N)</code>.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — numpy · top_k_numpy.py">{`import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)                # log-sum-exp trick
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def top_k_router(x, W_r, k):
    # x: (batch, d_model)        token embeddings
    # W_r: (d_model, n_experts)  router weights
    logits = x @ W_r                                       # (batch, n_experts)
    gates  = softmax(logits, axis=-1)                      # probabilities

    # argpartition: puts the top-k in the last k slots, unsorted. O(N).
    top_idx = np.argpartition(-gates, k, axis=-1)[:, :k]   # (batch, k)

    # gather the top-k gate values
    top_gates = np.take_along_axis(gates, top_idx, axis=-1)

    # renormalize across the k chosen experts
    top_gates = top_gates / top_gates.sum(axis=-1, keepdims=True)

    return top_idx, top_gates

np.random.seed(0)
x   = np.random.randn(4, 16)                               # batch of 4 tokens
W_r = np.random.randn(16, 8) * 0.1                         # 8 experts
idx, gates = top_k_router(x, W_r, k=2)
print("chosen experts\\n", idx)
print("normalized gates\\n", np.round(gates, 3))`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'sorted(range(n), key=...)[:k]',
            right: 'np.argpartition(-gates, k)[:, :k]',
            note: 'partial sort — O(N) instead of O(N log N)',
          },
          {
            left: 'loop over tokens, build gates list',
            right: 'x @ W_r, one matmul for the whole batch',
            note: 'routing is one matrix multiply',
          },
          {
            left: 'sum(gates[i] for i in idx)',
            right: 'take_along_axis(gates, idx, axis=-1).sum(-1, keepdims=True)',
            note: 'batched gather + sum',
          },
        ]}
      />

      <Prose>
        <p>
          PyTorch. <code>torch.topk</code> does both the selection and the gather
          in one call and returns the values sorted. <code>scatter</code> (or a
          simpler masked softmax) pushes the renormalized gates back into a
          full-size vector if you need it.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · top_k_router.py"
        output={`top-k idx
 tensor([[5, 2],
         [3, 7],
         [5, 1],
         [2, 4]])
top-k gates (renormalized)
 tensor([[0.64, 0.36],
         [0.57, 0.43],
         [0.52, 0.48],
         [0.55, 0.45]])`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    def __init__(self, d_model, n_experts, k, noise_std=1.0):
        super().__init__()
        self.w_r     = nn.Linear(d_model, n_experts, bias=False)
        self.w_noise = nn.Linear(d_model, n_experts, bias=False)    # for noisy top-k
        self.k       = k
        self.noise_std = noise_std

    def forward(self, x, train=True):
        # x: (batch, d_model)
        logits = self.w_r(x)                                        # (B, N)

        if train and self.noise_std > 0:
            noise = torch.randn_like(logits) * F.softplus(self.w_noise(x))
            logits = logits + self.noise_std * noise                 # noisy top-k

        # torch.topk: values and indices of the k largest, in one call
        top_logits, top_idx = logits.topk(self.k, dim=-1)            # (B, k) each

        # softmax over just the k chosen logits — equivalent to full softmax
        # followed by renormalization, but numerically nicer and cheaper
        top_gates = F.softmax(top_logits, dim=-1)                    # (B, k)
        return top_idx, top_gates

torch.manual_seed(0)
router = TopKRouter(d_model=16, n_experts=8, k=2, noise_std=0.0)
x = torch.randn(4, 16)
idx, gates = router(x, train=False)
print("top-k idx\\n", idx)
print("top-k gates (renormalized)\\n", torch.round(gates, decimals=2))`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'np.argpartition(-gates, k)[:, :k]',
            right: 'logits.topk(k, dim=-1)',
            note: 'returns values + indices; GPU-native',
          },
          {
            left: 'gates/gates.sum(..., keepdims=True)',
            right: 'F.softmax(top_logits, dim=-1)',
            note: 'softmax over the k chosen logits ≡ renormalized gates',
          },
          {
            left: 'W_r = np.random.randn(d, N)',
            right: 'nn.Linear(d, N, bias=False)',
            note: 'tracked for autograd, weights live on the right device',
          },
        ]}
      />

      <Callout variant="insight" title="softmax-on-top-k, not top-k-of-softmax">
        A small but important optimization in the PyTorch code: we run{' '}
        <code>softmax</code> over the <em>k selected logits</em>, not over all{' '}
        <code>N</code> logits followed by a renormalization. Those are
        mathematically equivalent (softmax is invariant to which logits you
        restrict it to, as long as you normalize over the chosen set), but the
        former is cheaper and numerically tidier. Fedus et al. call this out
        explicitly in the Switch Transformer paper.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Forgetting to renormalize:</strong>{' '}
          if you mask out the non-top-k gates but don&apos;t renormalize, your
          total gate mass drops below 1 and the MoE layer scales its output down
          by a random factor for every token. The bouncer waved the right names
          through but forgot to add up to 1. The residual stream starts drifting
          and you blame the optimizer.
        </p>
        <p>
          <strong className="text-term-amber">Dense routing at inference:</strong>{' '}
          a classic copy-paste bug. You train with top-k, evaluate with the whole
          softmax (because you switched to <code>model.eval()</code> and forgot a
          branch). Suddenly inference is <code>N/k</code>× slower and the metrics
          look great because every expert is helping. Check the FLOPs at eval time.
        </p>
        <p>
          <strong className="text-term-amber">Dropping noisy top-k too early:</strong>{' '}
          without noise the bouncer locks onto a single name within a few hundred
          steps. Keep <code>σ &gt; 0</code> for at least the first epoch, then
          decay toward zero. If you see a wildly unbalanced expert-usage histogram
          in the first 1k steps, the noise is almost always the culprit.
        </p>
        <p>
          <strong className="text-term-amber">k = N:</strong> the soft-routing
          trap. Someone on the team &ldquo;just wanted to try dense&rdquo; and set{' '}
          <code>k = n_experts</code>. Congratulations, you removed the bouncer —
          you have a dense model that costs <code>N×</code> the compute and a
          router you&apos;re not using.
        </p>
      </Gotcha>

      <Challenge prompt="Watch an expert eat the batch">
        <p>
          Build a Switch-style MoE (<code>k=1</code>) with 8 experts and 64-dim
          tokens. Use the <code>TopKRouter</code> above but set{' '}
          <code>noise_std=0.0</code> to disable noisy top-k. Train it on a toy
          sequence-modeling task (or even random targets) for 1000 steps.
        </p>
        <p className="mt-2">
          At every step, compute the fraction of tokens in the batch routed to
          each of the 8 experts — call it <code>load[i]</code>. Plot{' '}
          <code>load</code> over training. Without noise and without a
          load-balancing loss, you will almost certainly watch one name on the
          list see its load rise to near 1.0 while the other seven starve.
          That&apos;s <KeyTerm>router collapse</KeyTerm>, and it&apos;s what the
          next lesson exists to fix.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: re-run with <code>noise_std=1.0</code>. The collapse softens.
          Re-run with noise plus a load-balancing penalty (pre-view of the next
          lesson) and the loads stay roughly uniform across experts for the whole
          run.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> The router is the bouncer at the
          door: a single <code>Linear(d_model, n_experts)</code> plus a softmax.
          Top-k picks the <code>k</code> names it waves through, renormalizes
          their gates, and routes each token to only those experts.{' '}
          <code>k=1</code> is Switch (a hard pick), <code>k=2</code> is the current
          default (top expert plus a backup), <code>k&gt;2</code> and the room gets
          crowded while the &ldquo;free lunch&rdquo; disappears. Noisy top-k is
          non-optional early in training — without it the bouncer collapses onto a
          single name. The two most common bugs are forgetting to renormalize the
          chosen gates, and accidentally running dense routing at inference
          because you forgot the top-k branch.
        </p>
        <p>
          <strong>Next up — Load Balancing Loss.</strong> Noisy top-k buys you
          exploration, but it doesn&apos;t guarantee the names on the list end up
          with similar workloads. The load-balancing auxiliary loss does: a small
          penalty that pushes the bouncer toward a roughly uniform distribution
          over experts. Without it a few experts carry the whole model and the
          rest are dead weight. With it you get what MoE actually promises —{' '}
          <code>N</code> experts, each learning its own specialty. After that,
          one more lesson on expert parallelism — how to shard the room itself
          across GPUs — and this section closes. Then: a whole new room. Diffusion
          Models, starting with denoising intuition.
        </p>
      </Prose>

      <References
        items={[
          {
            title:
              'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer',
            author: 'Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean',
            venue: 'ICLR 2017 — the noisy top-k paper',
            url: 'https://arxiv.org/abs/1701.06538',
          },
          {
            title:
              'Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity',
            author: 'Fedus, Zoph, Shazeer',
            venue: 'JMLR 2022 (arXiv 2021) — the k=1 paper',
            url: 'https://arxiv.org/abs/2101.03961',
          },
          {
            title:
              'GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding',
            author: 'Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer, Chen',
            venue: 'ICLR 2021',
            url: 'https://arxiv.org/abs/2006.16668',
          },
          {
            title: 'ST-MoE: Designing Stable and Transferable Sparse Expert Models',
            author: 'Zoph, Bello, Kumar, Du, Huang, Dean, Shazeer, Fedus',
            year: 2022,
            url: 'https://arxiv.org/abs/2202.08906',
          },
        ]}
      />
    </div>
  )
}
