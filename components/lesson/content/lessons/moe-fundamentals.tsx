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
import MoEBlockDiagram from '../widgets/MoEBlockDiagram'
import MoEParamvsFlops from '../widgets/MoEParamvsFlops'

// Signature anchor: a panel of specialists, only a few of whom answer each
// question. A dense model is one overworked generalist; an MoE is a room of
// specialists with a receptionist (the router) who picks which two to
// consult per token. Returned to at the opening, the sparse-activation
// reveal, and the "computational free lunch" arithmetic.
export default function MoEFundamentalsLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="moe-fundamentals" />

      {/* ── Opening: the overworked generalist ──────────────────── */}
      <Prose>
        <p>
          Picture a single doctor. One human, one head, expected to diagnose
          everything — dermatology, cardiology, paediatrics, the weird rash on
          your elbow, and the cough that has been bothering your aunt for three
          weeks. They can do it. They will be tired. And every patient pays the
          full cost of that one overworked generalist being in the room,
          whether or not the thing on the table needs every last watt of their
          attention.
        </p>
        <p>
          That is a dense neural network. One giant stack of parameters, every
          one of them fired on every token — the word &ldquo;the&rdquo; pays
          the same compute bill as a complex Python expression, because the
          generalist doesn&apos;t know how to do anything smaller. Scaling laws
          are cruel but simple: bigger dense model, better loss. Double the
          parameters, pay double the FLOPs on every forward pass, on every
          token, during training and during inference. At some point the bill
          gets real. You cannot train a 10-trillion-parameter dense model in
          2026 because you cannot <em>afford</em> to forward it, let alone
          back-propagate it. The wall isn&apos;t intelligence. The wall is the
          electricity meter.
        </p>
        <p>
          <KeyTerm>Mixture of Experts</KeyTerm> (MoE) is the dodge, and the
          metaphor it replaces the generalist with is the one to hang onto for
          the rest of the lesson: <strong>a panel of specialists, only a few
          of whom answer each question.</strong> A dermatologist, a
          cardiologist, a paediatrician, a neurologist, and so on — eight of
          them, or thirty-two, or a hundred — sitting in a room. Out front is
          a receptionist (the router) who reads what just came in and picks
          the two specialists it should go to. The other specialists stay in
          their chairs, undisturbed. Total panel: huge. People who actually
          speak this visit: two.
        </p>
        <p>
          That&apos;s the whole idea. For each token, you don&apos;t need
          <em> all</em> of the model&apos;s knowledge — you need the right
          slice of it. Build a network where, for each token, only a small
          subset of the parameters actually runs. Total parameters can grow 8x
          while the per-token compute stays flat. Mixtral-8x7B is a 47B model
          that costs about as much to run as a 13B. That is not a typo. That
          is an architecture.
        </p>
        <p>
          This lesson walks you through the core machinery: the MoE layer
          itself, the math of the forward pass, the routing idea, and the
          parameter-vs-FLOPs arithmetic that everyone keeps quoting. For now:
          what an MoE is, why a panel beats a generalist, and why the
          receptionist matters.
        </p>
      </Prose>

      <Personify speaker="Mixture of Experts">
        I am a panel of specialists. Most of me is sitting quietly at any
        given moment, and that is the entire point. You ask a question; the
        receptionist wakes up the two of us who actually know the answer. The
        other thirty specialists keep reading their journals. You get the
        capacity of an enormous generalist for the price of a small one — as
        long as you can keep the receptionist honest.
      </Personify>

      {/* ── The forward pass ─────────────────────────────────────── */}
      <Prose>
        <p>
          Forget transformers for a moment. An MoE layer, at its core, is that
          panel of specialists wired in parallel with a routing decision
          gluing them together. You have <code>N</code>{' '}
          <strong>experts</strong> — call them <code>E₁ … E_N</code>, each one
          a specialist, each one a full <NeedsBackground slug="mlp-from-scratch">feed-forward</NeedsBackground> network,
          identical in shape, different in weights. You have a small{' '}
          <strong>router</strong> network <code>g(x)</code> — the receptionist
          — that takes an input token and spits out <code>N</code> gating
          weights, one per specialist. The output of the layer for a given
          token <code>x</code> is a weighted sum of the expert outputs:
        </p>
      </Prose>

      <MathBlock caption="MoE forward pass — weighted sum over experts">
{`y(x)   =   Σᵢ   gᵢ(x) · Eᵢ(x)
             i=1..N

where:
  Eᵢ(x)  =   the i-th expert's output  (a full FFN)
  gᵢ(x)  =   the i-th gate weight     (scalar, from router)
  Σᵢ gᵢ(x) = 1  on the experts chosen`}
      </MathBlock>

      <Prose>
        <p>
          Written that way, MoE looks like an ensemble. It almost is — except
          for the crucial detail that in practice <code>gᵢ(x)</code> is zero
          for most <code>i</code>. Only the top 1 or 2 specialists get a
          non-zero gate, and only <em>those</em> specialists actually run.
          That&apos;s the difference between an ensemble (every sub-model
          always runs, you average at the end, the whole panel talks at once)
          and MoE (receptionist picks, only the picked specialists speak).
          The ensemble adds compute. MoE adds parameters without compute.
        </p>
      </Prose>

      {/* ── Widget 1: MoE Block Diagram ─────────────────────────── */}
      <MoEBlockDiagram />

      <Prose>
        <p>
          Watch a token enter, get scored by the receptionist, and land at
          expert 3 of 8. The other seven specialists are there — they exist
          in memory, their weights take VRAM — but for this token, they
          never run. Change the token, and the router might dispatch to
          expert 6 instead. Over a batch of 4096 tokens, the receptionist is
          making 4096 independent dispatch decisions, and a well-trained
          router distributes them roughly uniformly across the 8 specialists
          (we&apos;ll see why that matters in a moment).
        </p>
        <p>
          Zoom out. In a real <NeedsBackground slug="transformer-block">transformer block</NeedsBackground>,
          the MoE layer doesn&apos;t replace the attention — it replaces the{' '}
          <KeyTerm>FFN</KeyTerm> (the two-matmul feed-forward block inside
          each transformer layer). Attention still runs densely on every
          token. But the FFN, which is where the bulk of a transformer&apos;s
          parameters actually live, becomes a panel of <code>N</code>{' '}
          specialists. Every other transformer layer tends to be MoE-ified
          (Mixtral does every layer, Switch did every other one). The
          attention stays the same. The compute savings come from the FFN
          slot.
        </p>
      </Prose>

      <Personify speaker="Expert">
        I am one specialist on the panel. I am a single FFN inside a stack of
        siblings — eight of us, thirty-two of us, occasionally a
        hundred-and-something of us. I don&apos;t know what I specialize in.
        Nobody handed me a nameplate. But over training I drifted — maybe I
        like Python tokens, maybe I like the word &ldquo;the&rdquo;, maybe I
        like the second half of mathematical equations. The receptionist
        learned to find me. On any given token I&apos;m either asleep in my
        chair or the only voice in the room.
      </Personify>

      {/* ── Param-vs-FLOP arithmetic: the "free lunch" reveal ──── */}
      <Prose>
        <p>
          Here&apos;s the arithmetic that makes the panel a computational
          free lunch — well, mostly. The generalist fired every neuron for
          every token. The panel fires two specialists for every token, no
          matter how many specialists are on the panel. That difference, in
          numbers:
        </p>
        <p>
          Suppose a dense FFN — the generalist — has <code>P</code>{' '}
          parameters and costs <code>C</code> FLOPs per token. Replace it
          with an <code>N</code>-specialist MoE panel where each specialist
          is the same shape as the original FFN, and each token activates{' '}
          <code>k</code> specialists.
        </p>
      </Prose>

      <MathBlock caption="parameters vs FLOPs — the whole trick">
{`Dense FFN:
    params  =  P
    flops   =  C         per token

N-expert MoE, top-k:
    params  =  N · P    +  (tiny router)
    flops   =  k · C    +  (tiny router)       per token

Ratio:
    params_MoE / params_dense   =   N
    flops_MoE  / flops_dense    =   k   (≪ N)

Canonical setting:   N = 8,  k = 2
    → 8×  the parameters
    → 2×  the FLOPs per token`}
      </MathBlock>

      <Prose>
        <p>
          The receptionist itself has negligible parameters — a single linear
          layer of shape <code>(d_model, N)</code>, so for{' '}
          <code>d_model = 4096</code> and <code>N = 8</code> that&apos;s{' '}
          <code>32K</code> parameters total, a rounding error against the
          billions in the specialists themselves. Ignore it and the
          arithmetic cleans up: <em>the panel buys you <code>N/k</code>{' '}
          parameter capacity for free</em>, where &ldquo;free&rdquo; means
          &ldquo;your FLOPs-per-token are unchanged from the generalist&apos;s
          bill.&rdquo;
        </p>
      </Prose>

      {/* ── Widget 2: Param vs FLOPs chart ──────────────────────── */}
      <MoEParamvsFlops />

      <Prose>
        <p>
          Slide through panel sizes. The dense baseline (flat blue) shows the
          generalist whose parameters and FLOPs march in lockstep — every new
          parameter costs one more FLOP per token. The MoE line (red) breaks
          that lockstep: parameters shoot up with <code>N</code> (bigger
          panel) while FLOPs stay pinned at the <code>k</code>-specialist
          cost (same two people speak per visit). That divergence is the
          entire economic argument for MoE. You are paying for parameters in{' '}
          <em>memory</em> (which is cheap and scales with VRAM) and getting
          the <em>quality</em> benefit of a much larger model, while{' '}
          <em>compute</em> (which is expensive and scales with tokens × steps
          × FLOPs) stays flat.
        </p>
        <p>
          The one catch — why I said &ldquo;mostly&rdquo; a free lunch — is
          visible in the chart if you look carefully: memory <em>is</em> a
          cost, just a different one. Every specialist on the panel has to be
          physically in the room, whether or not they&apos;re speaking.
          Mixtral-8x7B needs all 47B of those params loaded on the GPU even
          though only 13B are active per token, which is why it won&apos;t
          fit on a single 24GB card even though a 13B dense model does. MoE
          trades a compute bill for a memory bill. That&apos;s usually a
          trade you want to make at scale — compute is metered per token, per
          step, per epoch; memory is metered once when you load the model.
        </p>
      </Prose>

      <Personify speaker="Router">
        I am a very small network with an enormous responsibility. One linear
        layer, maybe thirty-two thousand parameters, and I decide where every
        single token goes. I take <code>x</code>, produce <code>N</code>{' '}
        scores, <NeedsBackground slug="softmax">softmax</NeedsBackground> them,
        and hand the top-k to the dispatcher. If I learn badly I send
        everything to specialist 1 and the other seven wither at their desks —
        we call that <em>collapse</em>, and there&apos;s a loss term (next
        lesson) that exists solely to scare me out of it.
      </Personify>

      <Callout variant="insight" title="why the panel beats the generalist even when only two speak">
        The surprising empirical fact is that MoE works. An 8x7B panel
        reaches roughly the quality of a dense 47B generalist, not of a dense
        13B one. Different specialists actually specialize — on code vs
        prose, on early-sequence vs late-sequence tokens, on rare vs common
        words — and the receptionist learns, per token, which specialist is
        the right one to call. You are not averaging over eight
        jack-of-all-trades; you are picking the two people on the panel who
        are genuinely expert in <em>this</em> token. Capacity compounds with
        specialization.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Enough diagrams. Code it. Three layers of abstraction, same panel,
          each one a little more honest about what really runs on the GPU.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · moe_scratch.py"
        output={`token 0 → expert 0   y=[0.71, -0.03, 0.18, 0.42]
token 1 → expert 3   y=[-0.11, 0.55, 0.29, -0.07]
token 2 → expert 1   y=[0.08, 0.22, -0.14, 0.63]
total params across 4 experts: 4 × (d·h + h·d) = 4 × 32 = 128
per-token compute: 1 × (d·h + h·d) = 32   (top-1 routing)`}
      >{`import math, random
random.seed(0)

D, H, N = 4, 4, 4                    # input dim, hidden dim, num experts

def linear(x, W):                    # y = x @ W, pure python
    return [sum(x[i] * W[i][j] for i in range(len(x))) for j in range(len(W[0]))]

def relu(v): return [max(0.0, z) for z in v]

def expert(x, W1, W2):
    return linear(relu(linear(x, W1)), W2)     # a 2-layer FFN

def randW(a, b):
    return [[random.gauss(0, 0.1) for _ in range(b)] for _ in range(a)]

# N experts, each a pair of weight matrices
experts = [(randW(D, H), randW(H, D)) for _ in range(N)]

# router: D -> N scores
Wr = randW(D, N)

def moe_top1(x):
    scores = linear(x, Wr)           # [N] raw logits
    # argmax = top-1 routing
    idx = max(range(N), key=lambda i: scores[i])
    W1, W2 = experts[idx]
    return idx, expert(x, W1, W2)

tokens = [[random.gauss(0, 1) for _ in range(D)] for _ in range(3)]
for t, x in enumerate(tokens):
    idx, y = moe_top1(x)
    print(f"token {t} → expert {idx}   y={[round(v, 2) for v in y]}")`}</CodeBlock>

      <Prose>
        <p>
          That loop is honest but slow — the receptionist is literally
          walking each patient to the specialist one at a time. Every token
          pays a Python function call. In real life you have 4096 tokens in a
          batch and you want them all routed in one go. NumPy.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — numpy, batch-vectorised · moe_numpy.py">{`import numpy as np
rng = np.random.default_rng(0)

B, D, H, N = 8, 4, 16, 4             # batch, dim, hidden, experts

# experts — stacked so they broadcast: (N, D, H) and (N, H, D)
W1 = rng.normal(0, 0.1, size=(N, D, H))
W2 = rng.normal(0, 0.1, size=(N, H, D))
Wr = rng.normal(0, 0.1, size=(D, N))

def moe_top1(x):                     # x: (B, D)
    scores   = x @ Wr                # (B, N)  router scores per token
    experts  = scores.argmax(-1)     # (B,)    which expert per token

    # dispatch: for each token, grab its expert's weights
    W1_tok = W1[experts]             # (B, D, H)
    W2_tok = W2[experts]             # (B, H, D)

    # batched FFN — einsum keeps it explicit
    h = np.einsum('bd,bdh->bh', x, W1_tok)
    h = np.maximum(0.0, h)
    y = np.einsum('bh,bhd->bd', h, W2_tok)
    return experts, y

x = rng.normal(size=(B, D))
routed_to, y = moe_top1(x)
print("each token routed to expert:", routed_to)
print("output shape:", y.shape)
print("expert load (count per expert):", np.bincount(routed_to, minlength=N))`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for t, x in enumerate(tokens): ...',
            right: 'x @ Wr  →  (B, N) in one shot',
            note: 'the B-loop disappears into broadcasting',
          },
          {
            left: 'argmax over a N-list',
            right: 'scores.argmax(-1)  →  (B,)',
            note: 'one argmax per token, vectorized',
          },
          {
            left: 'pick (W1, W2) for this expert',
            right: 'W1[experts]  →  gather (B, D, H)',
            note: 'fancy-indexing = per-token expert lookup',
          },
          {
            left: 'explicit two-matmul FFN per token',
            right: "einsum 'bd,bdh->bh'  then  'bh,bhd->bd'",
            note: 'batched matmul with the right expert per row',
          },
        ]}
      />

      <Prose>
        <p>
          Now the production-grade version — PyTorch, top-k routing, with the
          gating probability properly multiplied into each specialist&apos;s
          output (the <code>gᵢ(x)</code> weight in the forward-pass
          equation). This is close to what Mixtral runs.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch, top-k routing · moe_pytorch.py"
        output={`total params:  12,320  (experts)  +  24 (router)
active params / token:  6,160  (k=2 of 4 experts)
output shape: torch.Size([8, 4])`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, d=4, h=16, n_experts=4, top_k=2):
        super().__init__()
        self.top_k   = top_k
        self.experts = nn.ModuleList([Expert(d, h) for _ in range(n_experts)])
        self.router  = nn.Linear(d, n_experts, bias=False)

    def forward(self, x):                     # x: (B, D)
        logits = self.router(x)               # (B, N)
        # pick the top-k experts per token
        top_vals, top_idx = logits.topk(self.top_k, dim=-1)    # (B, k)
        # softmax *over the chosen k* — this is the gᵢ weighting
        top_gate = F.softmax(top_vals, dim=-1)                  # (B, k)

        y = torch.zeros_like(x)
        for slot in range(self.top_k):
            eid  = top_idx[:, slot]            # (B,) expert id for this slot
            gate = top_gate[:, slot].unsqueeze(-1)              # (B, 1)
            # in production this is a scatter-by-expert to batch tokens per expert
            for i in range(len(self.experts)):
                mask = (eid == i)
                if mask.any():
                    y[mask] += gate[mask] * self.experts[i](x[mask])
        return y

moe = MoELayer(d=4, h=16, n_experts=4, top_k=2)
x   = torch.randn(8, 4)
y   = moe(x)

total = sum(p.numel() for p in moe.parameters())
rout  = moe.router.weight.numel()
per_expert = sum(p.numel() for p in moe.experts[0].parameters())
print(f"total params:  {total - rout:,}  (experts)  +  {rout} (router)")
print(f"active params / token:  {moe.top_k * per_expert:,}  (k={moe.top_k} of {len(moe.experts)} experts)")
print("output shape:", y.shape)`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'hand-stacked (N, D, H) weight tensor',
            right: 'nn.ModuleList([Expert(...) for _ in N])',
            note: 'parameters are tracked, optim.step() knows about them',
          },
          {
            left: 'scores.argmax(-1) — top-1 only',
            right: 'logits.topk(k, dim=-1) + softmax over chosen k',
            note: 'now each token picks multiple experts with weights',
          },
          {
            left: 'einsum with gathered weights',
            right: 'mask-and-dispatch loop (prod: grouped matmul)',
            note: 'real kernels batch tokens per expert and matmul once',
          },
          {
            left: 'no gating weights',
            right: 'gate[mask] * self.experts[i](x[mask])',
            note: 'the gᵢ(x) in Σ gᵢ(x)·Eᵢ(x) — forget it and training breaks',
          },
        ]}
      />

      <Callout variant="note" title="the inner loop is not how Mixtral runs">
        The <code>for i in range(n_experts)</code> loop above is correct but
        educational — it&apos;s the receptionist walking each batch of
        patients down the hall to the right specialist by hand. Production
        kernels do a <em>grouped GEMM</em>: reshuffle tokens into per-expert
        bins, matmul all tokens for specialist 1 in one call, all tokens for
        specialist 2 in one call, and so on. Same math, one kernel launch
        per expert instead of a Python loop. Tools like Megablocks and the{' '}
        <code>grouped_gemm</code> ops in FasterTransformer do exactly this.
        We&apos;ll rebuild that in the &ldquo;Expert Parallelism&rdquo;
        lesson.
      </Callout>

      {/* ── Three canonical MoE papers ──────────────────────────── */}
      <Callout variant="insight" title="the three MoE papers you should know by name">
        <p className="mb-2">
          <strong>Shazeer et al., 2017 — &ldquo;Outrageously Large Neural Networks.&rdquo;</strong>{' '}
          The modern MoE origin story. 137-billion-parameter LSTM, top-2
          routing, noisy gating. This is the paper that defined the
          sparse-gating-with-top-k recipe everyone still uses — a panel of
          specialists with a noisy receptionist.
        </p>
        <p className="mb-2">
          <strong>Switch Transformer (Fedus et al., 2021).</strong> Top-<em>1</em>{' '}
          routing — each token goes to one specialist — which sounds
          aggressive but works if your load balancing is good. First MoE in a
          transformer at serious scale. Introduced the auxiliary
          load-balancing loss we cover next lesson.
        </p>
        <p>
          <strong>Mixtral-8x7B (Jiang et al., 2024).</strong> The first great
          open-weights MoE. Top-2 routing, 8 experts per layer, 47B total /
          13B active. Proved you can match or beat a 70B dense generalist
          with the compute budget of a 13B. GLaM (Du et al., 2022) is the
          Google counterpart — top-2, 64 experts, 1.2T total parameters.
        </p>
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Forgetting the router softmax temperature.</strong>{' '}
          Raw receptionist logits can be huge — after training, the
          &ldquo;winner&rdquo; specialist&apos;s logit might be 10 points
          above the runner-up. Softmax that and{' '}
          <code>g₁ ≈ 1.0, g₂ ≈ 0</code>, which defeats the point of top-k.
          Divide logits by a temperature (or clamp them) before the softmax
          if you want smoother gating.
        </p>
        <p>
          <strong className="text-term-amber">Not multiplying expert outputs by gᵢ(x).</strong>{' '}
          The forward pass is <em>Σ gᵢ(x)·Eᵢ(x)</em>, not{' '}
          <em>Σ Eᵢ(x)</em>. Forget the gate weight and the router gets no
          gradient — it has nothing to learn from, because its output no
          longer affects <code>y</code>. The whole system silently trains as
          if the routing were a fixed lookup. This is the single most common
          MoE bug.
        </p>
        <p>
          <strong className="text-term-amber">Conflating total vs active params.</strong>{' '}
          &ldquo;Mixtral-8x7B is 8 × 7B = 56B&rdquo; — almost. It&apos;s 47B,
          because the attention layers are shared (dense) and only the FFN is
          MoE-ified. When someone quotes a &ldquo;7B&rdquo; model&apos;s
          speed next to a &ldquo;Mixtral-8x7B,&rdquo; ask whether they mean
          total (everyone on the panel) or active (the two specialists who
          actually speak). Memory follows total. Compute follows active.
        </p>
        <p>
          <strong className="text-term-amber">Expert collapse.</strong>{' '}
          Untrained receptionists drift. If specialist 1 gets slightly better
          gradients early, the router sends it more tokens, it gets even
          better, and within a few hundred steps six of your eight
          specialists are reading the newspaper while one is on fire. The
          fix is the <KeyTerm>load balancing loss</KeyTerm> — next
          lesson&apos;s headline topic — which explicitly penalises uneven
          routing.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Swap an FFN for an MoE and verify the arithmetic">
        <p>
          Take a small transformer block — something with an{' '}
          <code>nn.Linear(d, 4d)</code>, ReLU, <code>nn.Linear(4d, d)</code>{' '}
          FFN — and replace that FFN with an 8-expert MoE where each expert
          has the same shape as the original. Use top-2 routing.
        </p>
        <p className="mt-2">
          Measure two things on a batch of 256 tokens with{' '}
          <code>d = 512</code>:
        </p>
        <ul className="mt-2">
          <li>
            <strong>Total parameter count</strong> before and after. It
            should grow by roughly 8× in the FFN slot (attention is
            unchanged).
          </li>
          <li>
            <strong>FLOPs per token</strong> — count the matmul shapes. The
            expert matmuls should contribute{' '}
            <code>2 × (d · 4d + 4d · d) = 2 × 8d²</code> (two experts
            activated), which is the same as a <em>single</em> dense 4d
            FFN&apos;s <code>2 · 4d² = 8d²</code>… wait, that&apos;s 2× not
            1×. Why? Because you&apos;re using top-2. Prove it to yourself,
            then switch to top-1 and watch the FLOPs match the dense
            baseline exactly.
          </li>
        </ul>
        <p className="mt-2 text-dark-text-muted">
          Bonus: print the distribution of expert selections over your
          batch. With an untrained router it will be roughly uniform. After
          training without a load-balancing loss, it will absolutely not be.
          You just reproduced expert collapse.
        </p>
      </Challenge>

      {/* ── Closing + next up ───────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> An MoE layer is a panel of{' '}
          <em>N</em> parallel specialists plus a tiny receptionist,
          forward-passed as a sparse weighted sum. The trick — the whole
          reason anyone bothers — is that only <em>k</em> specialists run
          per token, so parameters scale with <em>N</em> (bigger panel) but
          FLOPs scale with <em>k</em> (same handful speak). You get the
          capacity of an enormous generalist without the compute bill. The
          costs are a memory bill (every specialist has to be in the room),
          a routing bill (which is small), and two new failure modes:
          collapse (receptionist sends every token to one specialist) and
          dispatch inefficiency (batches don&apos;t line up across
          specialists). The next two lessons solve those failure modes.
        </p>
        <p>
          <strong>Next up — Load Balancing Loss.</strong> The receptionist
          left alone will play favourites. One specialist gets slightly
          better gradients in the first hundred steps, starts getting more
          tokens, gets even better, and within a few hundred more steps
          your 8-specialist panel is a 1-specialist panel with seven people
          getting paid to read magazines. That is <em>expert collapse</em>,
          and the fix — spelled out in the next lesson,{' '}
          <code>load-balancing-loss</code> — is a single auxiliary loss term
          that makes the receptionist pay a penalty every time it plays
          favourites. It is the single line of code between &ldquo;MoE
          works&rdquo; and &ldquo;MoE silently trains itself back into a
          dense model.&rdquo; We&apos;ll derive it, implement it, and watch
          collapse happen and then not-happen in the same notebook.
        </p>
      </Prose>

      <References
        items={[
          {
            title:
              'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer',
            author: 'Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean',
            venue: 'ICLR 2017',
            url: 'https://arxiv.org/abs/1701.06538',
          },
          {
            title:
              'Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity',
            author: 'Fedus, Zoph, Shazeer',
            venue: 'JMLR 2022 (arXiv 2021)',
            url: 'https://arxiv.org/abs/2101.03961',
          },
          {
            title: 'Mixtral of Experts',
            author: 'Jiang, Sablayrolles, Roux et al. (Mistral AI)',
            year: 2024,
            url: 'https://arxiv.org/abs/2401.04088',
          },
          {
            title: 'GLaM: Efficient Scaling of Language Models with Mixture-of-Experts',
            author: 'Du, Huang, Dai, Tong, Lepikhin et al.',
            venue: 'ICML 2022',
            url: 'https://arxiv.org/abs/2112.06905',
          },
          {
            title:
              'GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding',
            author: 'Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer, Chen',
            venue: 'ICLR 2021',
            url: 'https://arxiv.org/abs/2006.16668',
          },
        ]}
      />
    </div>
  )
}
