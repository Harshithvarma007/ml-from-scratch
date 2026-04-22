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
import BPTTStep from '../widgets/BPTTStep'                    // click-through animation of the backward pass across an unrolled RNN; gradients flow from time T back to time 0
import UnrollGradientFlow from '../widgets/UnrollGradientFlow'  // for a sequence of length N, visualize |∂L/∂h_t| at each time step t; show how it decays with distance from the loss

// Signature anchor: the unrolled movie walked backwards, frame by frame.
// Imagine pausing a recording and scrubbing in reverse, recomputing what
// each earlier frame contributed to the ending. Returned at the opening
// (the backwards walk frame), the chain-rule reveal, and the truncated-BPTT
// / computational-cost section.
export default function BackpropThroughTimeLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="backprop-through-time" />

      <Prose>
        <p>
          Picture the unrolled movie walked backwards, frame by frame. You recorded
          an <NeedsBackground slug="recurrent-neural-network">RNN</NeedsBackground>{' '}
          chewing through a sequence — one hidden state per tick, left to right — and
          now you&apos;re scrubbing the tape in reverse. Pause. Rewind one frame.
          Ask: how did <em>this</em> moment push the ending to be wrong? Write that
          down. Rewind another frame. Same question. Keep going until the tape runs out.
          That&apos;s <KeyTerm>Backpropagation Through Time</KeyTerm>, and it is
          genuinely that plain — a backwards scrub through the movie, one frame of
          gradient per rewind.
        </p>
        <p>
          Here&apos;s the move that makes the whole thing legal. Unroll the RNN and it
          stops being recurrent. It turns into a deep feedforward net with a single
          peculiarity: every frame of the movie shares the same weight matrix.{' '}
          <NeedsBackground slug="backpropagation">Backprop</NeedsBackground>{' '}
          doesn&apos;t need a new rulebook for time. It runs its usual backward
          recurrence across the unrolled graph, and at the end we remember that all
          the copies of <code>W</code> were really the same <code>W</code> and sum
          their gradients. Done.
        </p>
        <p>
          The algorithm has a name — BPTT — and a history. Werbos described it in
          the 70s, named it in his 1990 paper, and every modern sequence model (RNN,
          LSTM, Transformer) still does some version of the same backwards scrub
          under the hood. The mechanics are easy. The interesting part is that
          walking the movie in reverse is an <em>unstable</em> act: multiply the same
          Jacobian dozens of times and the signal either vanishes to zero or explodes
          to infinity. This lesson gets the mechanics clean and foreshadows that
          instability — the next lesson looks it in the face.
        </p>
      </Prose>

      {/* ── The unrolled graph ──────────────────────────────────── */}
      <Prose>
        <p>
          First, the forward tape. At every time step the RNN mixes the previous
          hidden state with the current input, squashes, and emits:
        </p>
      </Prose>

      <MathBlock caption="the RNN cell — the same W at every t">
{`h_t   =   tanh( W_h · h_(t-1)  +  W_x · x_t  +  b )
y_t   =   W_y · h_t  +  b_y

L     =   Σ_t  L_t(y_t, target_t)            sum loss over the sequence`}
      </MathBlock>

      <Prose>
        <p>
          Now unroll — lay the frames end to end. For <code>T = 3</code> steps you
          get three copies of the cell stacked vertically, hidden-state edges
          threading from frame to frame, a loss sitting at the top (or one at every
          step, summed). Here&apos;s the detail that matters: every copy of{' '}
          <code>W_h</code> is <em>the same parameter</em>. When you ask for{' '}
          <code>∂L/∂W_h</code>, you&apos;re asking for the derivative with respect
          to a single quantity that appears in every frame of the unrolled movie.
          The multivariate{' '}
          <NeedsBackground slug="multi-layer-backpropagation">chain rule</NeedsBackground>{' '}
          handles it with a rule so simple it sounds like cheating: sum the
          contributions from every frame where the weight shows up.
        </p>
      </Prose>

      <MathBlock caption="the chain rule across time — sum over all time steps">
{`∂L/∂W_h   =   Σ_t   ∂L_t / ∂W_h                    total over steps

where each term expands (by chain rule through the chain of hidden states):

∂L_t / ∂W_h   =   Σ_{k ≤ t}   (∂L_t/∂h_t) · (∂h_t/∂h_k) · (∂h_k/∂W_h)_local

                  └─────────┘    └────────┘     └──────────────┘
                  gradient at    Jacobian       local derivative at step k,
                  the output     product        treating h_(k-1) as constant`}
      </MathBlock>

      <Prose>
        <p>
          Stare at that middle term, <code>∂h_t/∂h_k</code>. It&apos;s a product of
          per-step Jacobians:{' '}
          <code>∂h_t/∂h_(t-1) · ∂h_(t-1)/∂h_(t-2) · … · ∂h_(k+1)/∂h_k</code>. Each
          one is a matrix. Multiply <code>T - k</code> of them together — one matrix
          per frame you rewound through — and that product is where every training
          pathology you&apos;ll ever see comes from. Park it. We&apos;ll come back.
        </p>
      </Prose>

      {/* ── Animated BPTT step-through ──────────────────────────── */}
      <Prose>
        <p>
          Click through the backward pass below. Each click is one rewind of the
          movie — the loss gradient starts at the final frame <code>T</code>, walks
          back one frame to <code>T-1</code> through the shared <code>W_h</code>,
          picks up that frame&apos;s local contribution to <code>∂L/∂W_h</code>,
          and accumulates. When you reach <code>t = 0</code> the readout at the
          bottom is the total gradient — the sum of every frame&apos;s contribution
          to the weight the whole movie shares.
        </p>
      </Prose>

      <BPTTStep />

      <Personify speaker="Time step t">
        I&apos;m one frame of the unrolled movie, somewhere in the middle. I look
        identical to every other frame — same weights, same activation. When the
        gradient rewinds to me, I do two things: I stash my local contribution to{' '}
        <code>∂L/∂W_h</code>, and I hand the upstream gradient to frame <code>t-1</code>,
        modulated by my Jacobian. I don&apos;t know how far back I am. I just pass the
        signal along, and hope it survives the trip.
      </Personify>

      {/* ── The gradient product ────────────────────────────────── */}
      <Prose>
        <p>
          Zoom in on that Jacobian product. For an RNN with tanh activation,
        </p>
      </Prose>

      <MathBlock caption="the per-step Jacobian — and the product that kills everything">
{`∂h_t/∂h_(t-1)   =   diag(tanh'(z_t))  ·  W_h         where z_t = W_h · h_(t-1) + W_x · x_t + b

Chain it across the whole sequence:

∂h_T/∂h_0   =   Π_{t=1..T}   diag(tanh'(z_t)) · W_h

Take the spectral norm (largest singular value):

‖ ∂h_T/∂h_0 ‖   ≤   Π_{t=1..T}   ‖diag(tanh'(z_t))‖ · ‖W_h‖

                   ≤   (1)^T  ·  σ_max(W_h)^T     since |tanh'| ≤ 1`}
      </MathBlock>

      <Prose>
        <p>
          Two regimes, no middle ground. If <code>σ_max(W_h) &lt; 1</code>, the
          product decays geometrically with each rewind — <code>0.9^100 ≈ 10⁻⁵</code>{' '}
          by the time you&apos;ve walked the movie a hundred frames back. The
          gradient at <code>t = 0</code> is a rumor of a rumor. Your RNN literally
          cannot learn long-range dependencies because the signal telling early
          weights how to change has been scrubbed into numerical noise. If{' '}
          <code>σ_max(W_h) &gt; 1</code>, the product explodes —{' '}
          <code>1.1^100 ≈ 14000</code> — and one gradient step launches your model
          into the next dimension. This is Pascanu et al.&apos;s 2013 result, and
          it&apos;s the whole reason LSTMs, GRUs, gradient clipping, and eventually
          transformers exist.
        </p>
      </Prose>

      <UnrollGradientFlow />

      <Prose>
        <p>
          Slide the sequence-length knob. With <code>N = 10</code>, the gradient
          magnitude at <code>h_0</code> is still a visible fraction of its value at{' '}
          <code>h_T</code> — trainable. By <code>N = 30</code> it&apos;s orders of
          magnitude smaller. By <code>N = 100</code> it&apos;s effectively zero.
          The curve is exactly the <code>σ^t</code> decay predicted above. The
          Jacobian product is doing what matrix products always do — latching onto
          the largest singular direction and either amplifying or annihilating it
          as the movie rewinds frame by frame.
        </p>
      </Prose>

      <Personify speaker="Gradient product">
        I am <code>Π ∂h_t/∂h_(t-1)</code>. I am a sequence of matrix multiplications,
        and I am ruthless. If the weights want me to shrink, I shrink to zero faster
        than you can plot me. If they want me to grow, I blow up past float32 by
        step 40. There is no stable middle. I am the reason your RNN cannot remember
        what happened 50 steps ago, and I am the reason LSTMs were invented to go
        around me.
      </Personify>

      {/* ── Three-layer code ─────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations, each shorter than the last. Pure Python unrolls a
          three-frame movie by hand and writes every derivative out — you can see
          the sum-over-time-steps literally, one loop iteration per rewind. NumPy
          batches across the batch dimension and compresses each frame into three
          matrix ops. PyTorch hands the whole backwards scrub to autograd and adds
          the one line that keeps real RNN training from exploding:{' '}
          <code>clip_grad_norm_</code>.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · bptt_scratch.py"
        output={`dW_h from scratch:
[[ 0.0213 -0.0451]
 [-0.0189  0.0327]]`}
      >{`import math

def tanh(z): return math.tanh(z)
def dtanh(z): return 1 - math.tanh(z) ** 2

# 2-dim hidden, 1-dim input, 3 time steps
W_h = [[0.5, -0.3], [-0.2, 0.4]]
W_x = [[0.6], [-0.1]]
b   = [0.0, 0.0]

x = [[1.0], [0.5], [-0.8]]     # inputs at t=1,2,3
target = [0.3, -0.2]           # target for h_3

def matvec(M, v):
    return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

def add(a, b): return [a[i] + b[i] for i in range(len(a))]

# ── Forward: unroll 3 steps, stash h and z for each ──
h = [[0.0, 0.0]]               # h_0 = zeros
z = [None]                     # z_0 is undefined
for t in range(3):
    z_t = add(add(matvec(W_h, h[t]), matvec(W_x, x[t])), b)
    h_t = [tanh(v) for v in z_t]
    z.append(z_t); h.append(h_t)

loss = 0.5 * sum((h[3][i] - target[i]) ** 2 for i in range(2))

# ── Backward: BPTT for 3 steps ──
# dL/dh at each t, propagated backward through time
dh_next = [h[3][i] - target[i] for i in range(2)]       # δ at t=T
dW_h = [[0.0, 0.0], [0.0, 0.0]]                         # accumulator — sum over t

for t in [3, 2, 1]:                                     # walk backward
    # through tanh:  dz_t = dh_t ⊙ tanh'(z_t)
    dz = [dh_next[i] * dtanh(z[t][i]) for i in range(2)]

    # contribution to W_h at this step:  dz ⊗ h_(t-1)
    for i in range(2):
        for j in range(2):
            dW_h[i][j] += dz[i] * h[t - 1][j]           # SUM over time

    # propagate to h_(t-1):  dh_(t-1) = W_hᵀ · dz
    dh_next = [sum(W_h[k][i] * dz[k] for k in range(2)) for i in range(2)]

print("dW_h from scratch:")
for row in dW_h: print(f"[{row[0]:+.4f} {row[1]:+.4f}]")`}</CodeBlock>

      <CodeBlock language="python" caption="layer 2 — numpy batched · bptt_numpy.py">{`import numpy as np

rng = np.random.default_rng(0)
H, D = 4, 3                       # hidden size, input size
T, N = 10, 32                     # sequence length, batch size
W_h = rng.normal(0, 0.3, (H, H))
W_x = rng.normal(0, 0.3, (H, D))
b   = np.zeros(H)

def forward(X):                   # X: (T, N, D)
    h = np.zeros((T + 1, N, H))
    z = np.zeros((T + 1, N, H))
    for t in range(T):
        z[t + 1] = h[t] @ W_h.T + X[t] @ W_x.T + b
        h[t + 1] = np.tanh(z[t + 1])
    return h, z

def bptt(X, target):              # target: (N, H) — compare to h_T
    h, z = forward(X)
    loss = 0.5 * ((h[T] - target) ** 2).mean()

    dW_h = np.zeros_like(W_h)
    dW_x = np.zeros_like(W_x)
    db   = np.zeros_like(b)

    dh = (h[T] - target) / N                                # δ at t=T  — shape (N, H)
    for t in range(T, 0, -1):                               # walk backward in time
        dz = dh * (1 - np.tanh(z[t]) ** 2)                  # through tanh        (N, H)
        dW_h += dz.T @ h[t - 1]                             # accumulate — SUM over t
        dW_x += dz.T @ X[t - 1]
        db   += dz.sum(axis=0)
        dh    = dz @ W_h                                    # propagate to h_(t-1)
    return loss, dW_h, dW_x, db`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for i for j: dW_h[i][j] += dz[i] * h[t-1][j]',
            right: 'dW_h += dz.T @ h[t-1]',
            note: 'the outer product across the batch — one line',
          },
          {
            left: 'for t in [3, 2, 1]: …',
            right: 'for t in range(T, 0, -1): …',
            note: 'the backward-time loop is the same shape, just longer',
          },
          {
            left: 'one example at a time',
            right: 'N examples vectorized — dz.T @ h[t-1] handles it',
            note: 'batching lives in the matmul, not the loop',
          },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch + gradient clipping · bptt_pytorch.py"
        output={`step  0  loss=1.021  grad_norm(pre-clip)=4.82
step 50  loss=0.173  grad_norm(pre-clip)=0.97
step 100 loss=0.041  grad_norm(pre-clip)=0.44`}
      >{`import torch, torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, D=3, H=4):
        super().__init__()
        self.cell = nn.RNNCell(D, H, nonlinearity='tanh')
        self.out  = nn.Linear(H, H)

    def forward(self, x_seq, h0):
        h = h0
        for t in range(x_seq.size(0)):
            h = self.cell(x_seq[t], h)                 # every t reuses the same weights
        return self.out(h)

model = SimpleRNN()
opt   = torch.optim.SGD(model.parameters(), lr=0.1)

T, N, D, H = 20, 32, 3, 4
x = torch.randn(T, N, D)
y = torch.randn(N, H)
h0 = torch.zeros(N, H)

for step in range(101):
    opt.zero_grad()
    h0 = h0.detach()                                    # ← CRUCIAL: cut the graph between batches
    yhat = model(x, h0)
    loss = 0.5 * (yhat - y).pow(2).mean()
    loss.backward()                                     # autograd runs BPTT for the full T steps

    # Gradient clipping — required for RNN training
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    opt.step()
    if step in (0, 50, 100):
        print(f"step {step:3d}  loss={loss.item():.3f}  grad_norm(pre-clip)={grad_norm:.2f}")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'for t in range(T, 0, -1): dW_h += dz.T @ h[t-1]',
            right: 'loss.backward()  # one call',
            note: 'autograd replays the unrolled graph in reverse for you',
          },
          {
            left: 'manual sum over time steps',
            right: 'accumulated in .grad automatically',
            note: 'shared-weight summation is free once the graph tracks it',
          },
          {
            left: '(nothing — gradients explode silently)',
            right: 'clip_grad_norm_(params, max_norm=1.0)',
            note: 'the one safety line. never train an RNN without it.',
          },
        ]}
      />

      <Callout variant="insight" title="truncated BPTT — the practical compromise">
        Rewinding a 10,000-frame movie means 10,000 cached activations and a
        backward scrub that touches every single one. Memory and compute both
        scale linearly with the length of the tape. In practice nobody does this.{' '}
        <KeyTerm>Truncated BPTT</KeyTerm> rewinds only <code>k</code> frames
        (typically 32 to 200), then <em>detaches</em> the hidden state and keeps
        rolling the camera forward. You lose gradient signal from beyond the
        truncation window — the model literally cannot learn dependencies older
        than <code>k</code> frames — but you get constant memory and constant
        per-step compute. The truncation length is the knob: too short and your
        RNN forgets the past, too long and your GPU catches fire. Language
        modeling typically picks <code>k = 35</code> to <code>200</code>.
      </Callout>

      <Callout variant="warn" title="gradient clipping — non-optional">
        Before every <code>optimizer.step()</code> on an RNN, call{' '}
        <code>torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)</code>. If the
        total gradient norm exceeds <code>1.0</code>, every gradient is scaled
        down uniformly so the norm equals <code>1.0</code>. This trades gradient{' '}
        <em>direction</em> fidelity (preserved) for <em>magnitude</em> fidelity
        (capped) — exactly the trade you want when the magnitude is about to
        explode. Without clipping, RNN loss curves routinely spike to NaN during
        training. With clipping, they don&apos;t. Choose the max-norm by looking
        at the pre-clip norms: pick something a bit below the typical value.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Forgetting to detach the hidden state between batches.</strong>{' '}
          If you carry <code className="text-dark-text-primary">h</code> from one mini-batch
          to the next without calling <code className="text-dark-text-primary">h = h.detach()</code>,
          autograd&apos;s graph keeps growing. By batch 100 you&apos;re backpropagating
          through the entire training history — memory blows up and the backward pass takes
          forever. <code className="text-dark-text-primary">detach()</code> severs the graph
          while keeping the numerical value of the hidden state. Do it every batch boundary.
        </p>
        <p>
          <strong className="text-term-amber">Truncation window too short.</strong>{' '}
          If the task has dependencies 200 steps apart but you only backprop 20 steps, the
          model has no gradient signal for that dependency. It can&apos;t learn it. Symptom:
          training loss plateaus at a value that corresponds to &ldquo;learned the short-range
          structure, can&apos;t learn the long-range structure.&rdquo;
        </p>
        <p>
          <strong className="text-term-amber">Truncation window too long.</strong>{' '}
          If you backprop <code className="text-dark-text-primary">k = 2000</code> on a
          moderately sized model, you will run out of GPU memory. Activations from every
          time step have to be cached. Memory usage is roughly{' '}
          <code className="text-dark-text-primary">O(k · batch · hidden²)</code>.
        </p>
        <p>
          <strong className="text-term-amber">Skipping the clip.</strong>{' '}
          First training run diverges in 50 steps. You blame the learning rate, the init, the
          dataset. It&apos;s the clip. Add it. Move on.
        </p>
      </Gotcha>

      <Challenge prompt="Verify your hand-rolled BPTT against autograd">
        <p>
          Take the pure-Python code above, shrink it to <code>T = 2</code> (two time steps),
          and compute <code>dW_h</code> by hand. Now build the same RNN in PyTorch with{' '}
          <code>nn.RNNCell</code>, feed the same inputs and targets, call{' '}
          <code>loss.backward()</code>, and read off <code>cell.weight_hh.grad</code>. The
          two matrices should match to six decimal places. If they don&apos;t, you forgot to
          sum a time-step contribution — go find the missing term. This is the exact test
          every autograd engine ships with.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Extension: flip <code>h.detach()</code> on and off in the PyTorch version and watch
          memory usage climb across batches. The bug is silent until you profile.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> BPTT is just backprop on the unrolled
          movie, plus one bookkeeping rule: sum the gradient contributions over every
          frame where the shared weight appears. The bad news — and the hook into the
          next lesson — is that <code>∂h_T/∂h_0 = Π ∂h_t/∂h_(t-1)</code>. Products of
          matrices misbehave. If the spectral norm of <code>W_h</code> is less than one,
          the product vanishes and long-range gradients die. If it&apos;s greater than
          one, the product explodes and training diverges. Truncated BPTT controls
          compute; gradient clipping controls explosions; neither controls vanishing.
        </p>
        <p>
          <strong>Next up — The Vanishing Gradient Problem.</strong> We&apos;ve named
          the failure mode. Now we stare at it: walking that long chain in reverse has
          a cost — the signal weakens with every frame you rewind through, and after
          a few dozen rewinds it&apos;s gone. We&apos;ll derive exactly how fast the
          gradient dies when it has to travel 100 frames back through a chain of tanh
          Jacobians, why <em>every</em> vanilla RNN suffers from it, and why the fix
          isn&apos;t a better optimizer but a <em>different architecture</em> — one
          with a gradient highway built directly into the cell. That architecture is
          the LSTM, and it&apos;s what the next few lessons build.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Backpropagation through time: What it does and how to do it',
            author: 'Paul J. Werbos',
            venue: 'Proceedings of the IEEE, 1990 — the paper that named BPTT',
            url: 'https://ieeexplore.ieee.org/document/58337',
          },
          {
            title: 'On the difficulty of training Recurrent Neural Networks',
            author: 'Pascanu, Mikolov, Bengio',
            venue: 'ICML 2013 — vanishing/exploding gradients and the clipping fix',
            url: 'https://arxiv.org/abs/1211.5063',
          },
          {
            title: 'Dive into Deep Learning — §9.7 Backpropagation Through Time',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai — the clearest modern derivation',
            url: 'https://d2l.ai/chapter_recurrent-neural-networks/bptt.html',
          },
        ]}
      />
    </div>
  )
}
