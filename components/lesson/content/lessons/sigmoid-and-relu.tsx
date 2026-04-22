import CodeBlock from '../CodeBlock'
import LayeredCode from '../LayeredCode'
import MathBlock from '../MathBlock'
import NeedsBackground from '../NeedsBackground'
import Prereq from '../Prereq'
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
import ActivationPlayground from '../widgets/ActivationPlayground'
import SaturationChain from '../widgets/SaturationChain'
import DeadReluField from '../widgets/DeadReluField'

// Signature anchor: dimmer switch vs light switch. Sigmoid squashes smoothly
// into (0, 1) — dimmer that never hits the rails. ReLU is binary — off for
// negatives, fully on for positives. The "dead ReLU" failure mode is a light
// switch jammed in the always-off position. Anchor returns at the opening,
// the mechanism reveal for each activation, and the dead-ReLU gotcha.
//
// GD closed with a cliffhanger: "one of them has a bad habit of murdering
// gradients entirely." This lesson pays it off. Sigmoid dies softly
// (gradient → 0 at saturation), ReLU dies harder (gradient is literally 0
// forever for a stuck neuron).
export default function SigmoidAndReluLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout ────────────────────────────────── */}
      <Prereq currentSlug="sigmoid-and-relu" />

      {/* ── Opening: problem-first reframe ──────────────────────── */}
      <Prose>
        <p>
          A neural network is a stack of matrix multiplies. Multiply two
          matrices, you get a matrix. Stack a thousand of them, you get — a
          matrix. All that depth, all that compute, and mathematically you
          have built <code>y = Wx + b</code>. The model, to use a technical
          term, is cooked.
        </p>
        <p>
          Something has to bend the line between the layers. A tiny scalar
          function applied elementwise — nothing fancy, just a kink — and
          suddenly the network can carve out shapes no single layer could.
          Those kinks are <KeyTerm>activation functions</KeyTerm>, and the
          whole discipline rests on them.
        </p>
        <p>
          Two of them run the world. <strong>Sigmoid</strong> is a{' '}
          <em>dimmer</em> — input goes in, output gets smoothly squashed
          between 0 and 1, never the rails. <strong>ReLU</strong> is a{' '}
          <em>light switch</em> — off for anything negative, fully on and
          proportional for anything positive. One is elegant. The other is
          crude. Guess which one ate the other&apos;s lunch.
        </p>
      </Prose>

      <Personify speaker="Activation function">
        I&apos;m the bend in the plane. Without me, your thousand-layer
        network is a hundred-line Excel formula. With me, it can learn
        anything. I ask very little: just that my derivative stays alive
        where it matters.
      </Personify>

      {/* ── Widget 1: Activation Playground ─────────────────────── */}
      <Prose>
        <p>
          Five activations under one roof. Drag <code>x</code>, read off the
          output and the derivative. Watch the derivative curves — they are
          not decoration. They are the single number that decides whether
          your network trains or sits there doing nothing.
        </p>
      </Prose>

      <ActivationPlayground />

      <Callout variant="note" title="why the derivative is the whole story">
        During backprop the{' '}
        <NeedsBackground slug="gradient-descent">gradient</NeedsBackground>{' '}
        at each layer gets multiplied by the activation&apos;s derivative.
        If that number is small — sigmoid is never above <code>0.25</code> —
        you are shrinking the gradient at every step. Stack enough layers
        and it vanishes. If it is 0 — ReLU on the negative side — the
        gradient doesn&apos;t just shrink, it dies. Everything below is an
        unpacking of that sentence.
      </Callout>

      {/* ── Sigmoid math ─────────────────────────────────────────── */}
      <Prose>
        <p>
          Start with sigmoid — the dimmer. For a decade it was the default
          activation, and it is still the right choice at the output of a
          binary classifier, because its codomain is exactly{' '}
          <code>(0, 1)</code>. Which is convenient when you want a
          probability.
        </p>
      </Prose>

      <MathBlock caption="sigmoid — the classic squash">
{`σ(x)   =    1
         ───────────
          1 + e⁻ˣ`}
      </MathBlock>

      <Prose>
        <p>
          Picture what the formula does. Feed it <code>+10</code> — the{' '}
          <code>e⁻ˣ</code> collapses to almost zero, output snaps to{' '}
          <code>1</code>. Feed it <code>−10</code> — the <code>e⁻ˣ</code>{' '}
          explodes, output snaps to <code>0</code>. Feed it anything in the
          middle and you get a smooth ride between the two. That&apos;s the
          dimmer — the knob never quite reaches the rails, but it leans
          hard toward them in the tails.
        </p>
        <p>
          The derivative has an unusually pretty form. You can write it as
          a function of <em>itself</em>:
        </p>
      </Prose>

      <MathBlock caption="derivative, in closed form">
{`σ'(x)  =  σ(x) · (1 − σ(x))`}
      </MathBlock>

      <Prose>
        <p>
          That one-liner is why sigmoid is cheap: if you already computed{' '}
          <code>σ(x)</code> on the forward pass, the derivative is one
          multiply away. Here is the proof — one line of calculus, because
          if we don&apos;t derive it ourselves we&apos;ll keep being
          surprised by it.
        </p>
      </Prose>

      <MathBlock caption="a very short proof">
{`σ(x)   =  (1 + e⁻ˣ)⁻¹

σ'(x)  =  −(1 + e⁻ˣ)⁻² · (−e⁻ˣ)                    chain rule

       =   e⁻ˣ / (1 + e⁻ˣ)²

       =   1/(1 + e⁻ˣ)  ·  e⁻ˣ/(1 + e⁻ˣ)

       =   σ(x)   ·   (1 − σ(x))`}
      </MathBlock>

      <Prose>
        <p>
          Plug <code>x = 0</code>: <code>σ(0) = 0.5</code>, so{' '}
          <code>σ&apos;(0) = 0.25</code>. That is the{' '}
          <em>maximum</em> of the derivative. Every other <code>x</code>{' '}
          gives something smaller. At <code>x = ±5</code> you&apos;re down
          to <code>0.0066</code>. At <code>x = ±10</code>, <code>4.5 × 10⁻⁵</code>.
          The dimmer has bottomed out — turn it further and nothing moves.
        </p>
        <p>
          Now stack. In a deep network the effective gradient at layer{' '}
          <code>k</code> is the product of derivatives through layers 1 to{' '}
          <code>k</code>. Every layer contributes a factor of at most{' '}
          <code>0.25</code>. Twenty layers deep, you are multiplying twenty
          numbers each ≤ 0.25 together. That is the{' '}
          <KeyTerm>vanishing gradient problem</KeyTerm>, and it is not
          hand-wavy worry — it is arithmetic.
        </p>
      </Prose>

      {/* ── Widget 2: Saturation Chain ──────────────────────────── */}
      <SaturationChain />

      <Prose>
        <p>
          Move the depth slider. By layer 15 the sigmoid curve has dropped
          below <code>10⁻⁹</code>. By layer 25, below <code>10⁻¹⁴</code> —
          under the precision of a 32-bit float. The deeper layers
          don&apos;t <NeedsBackground slug="gradient-descent">minimize</NeedsBackground>{' '}
          anything because no signal reaches them to minimize against.
          ReLU&apos;s curve stays flat. That is the whole chart. That is
          why sigmoid-as-hidden-activation fell out of fashion circa 2011.
        </p>
      </Prose>

      <Personify speaker="Sigmoid">
        I was the dominant activation for twenty years. Then Krizhevsky,
        Sutskever, and Hinton trained an 8-layer ReLU network on ImageNet
        in 2012, crushed the entire field, and I was effectively retired
        from hidden layers within 18 months. I am still a fine output
        layer for binary classification. Everywhere else, please
        don&apos;t call me.
      </Personify>

      {/* ── ReLU math ────────────────────────────────────────────── */}
      <Prose>
        <p>
          Meet the successor. Mathematically, it is a joke — and that is
          the point. No <code>e</code>. No chain rule. A light switch.
        </p>
      </Prose>

      <MathBlock caption="ReLU — rectified linear unit">
{`ReLU(x)   =   max(0, x)

ReLU'(x)  =   { 1  if x > 0
               { 0  if x ≤ 0`}
      </MathBlock>

      <Prose>
        <p>
          That is the spec. For <code>x &gt; 0</code> the switch is on —
          signal passes through unchanged, gradient passes through
          unchanged. For <code>x ≤ 0</code> the switch is off — signal is
          zero, gradient is zero. No in-between. The dimmer had a smooth
          ride between 0 and 1; the switch has two states.
        </p>
        <p>
          Three things fall out of the spec.
        </p>
        <ul>
          <li>
            <strong>Speed.</strong> <code>max(0, x)</code> is a comparison
            and maybe a zero — one instruction. Compare to{' '}
            <code>1 / (1 + exp(−x))</code> — a divide and an exp. On a
            GPU shoving billions of activations per step, the difference
            pays for the whole paper.
          </li>
          <li>
            <strong>No saturation on the positive side.</strong> When the
            switch is on, the derivative is a flat <code>1</code>.
            Gradients propagate backward through active neurons without
            any arithmetic decay. Stack ReLU layers arbitrarily deep — the
            sigmoid death spiral doesn&apos;t happen.
          </li>
          <li>
            <strong>Sparsity.</strong> Roughly half of ReLU&apos;s outputs
            are zero on typical input. Those neurons contribute nothing to
            that particular forward pass. Think of it as the network
            deciding, per-input, which neurons get to participate.
          </li>
        </ul>
      </Prose>

      <Callout variant="insight" title="one empirical paper, one civilization pivot">
        The shift to ReLU wasn&apos;t a theoretical revolution — it was a
        pragmatic one. AlexNet (2012) trained faster and scored higher
        than anything before it, and it happened to use ReLU. Within two
        years every serious vision network did. When you read a
        transformer paper today and see GELU, it is a direct descendant —
        a smooth variant of the same on/off idea.
      </Callout>

      {/* ── Dead ReLU problem ───────────────────────────────────── */}
      <Prose>
        <p>
          Here is the catch, and here is the payoff for the cliffhanger at
          the end of the last lesson. The question was: one of the two
          activations has a bad habit of murdering{' '}
          <NeedsBackground slug="gradient-descent">gradients</NeedsBackground>{' '}
          entirely. Sigmoid saturates — its gradient shrinks toward zero
          in the tails, the dimmer bottoms out. That is bad. ReLU does
          something worse.
        </p>
        <p>
          If a ReLU neuron&apos;s pre-activation is <em>always</em>{' '}
          negative across the whole training set, it outputs zero on every
          example, and its derivative is zero on every example. Which
          means the gradient on its incoming{' '}
          <NeedsBackground slug="gradient-descent">parameters</NeedsBackground>{' '}
          is zero on every example. Which means those parameters never
          update. Which means it will be stuck in the off position
          forever. That is a <KeyTerm>dead ReLU</KeyTerm> — a light switch
          broken in the always-off position, unrecoverable, dead weight in
          the network.
        </p>
        <p>
          How often does this happen? Depends on initialization. A small,
          correctly-scaled init barely produces any. A cold,
          overly-negative bias can kill a quarter of your neurons on the
          first step and leave them dead for the rest of training.
          Play with it.
        </p>
      </Prose>

      <DeadReluField />

      <Prose>
        <p>
          Drag <code>bias μ</code> toward <code>−3</code>. The grid goes
          dark — most neurons never fire on any example in the batch,
          their gradient is zero, they will sit at zero forever. Now click{' '}
          <code>Leaky ReLU</code>. Nothing is fully dead anymore — the dim
          cells still have a small gradient (the <code>0.1</code> slope on
          the negative side), so an unlucky neuron can still claw its way
          back into usefulness. That is the fix. It is a one-character
          change in code.
        </p>
        <p>
          In practice modern networks use careful initialization (He init,
          coming in a later lesson) to keep most ReLUs alive, and
          sometimes swap in Leaky ReLU or GELU when dying is a real
          concern. But the failure mode is real, and the mitigation is
          worth knowing.
        </p>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Sigmoid in hidden layers:</strong>{' '}
          almost always wrong today. Use ReLU / Leaky ReLU / GELU. Sigmoid
          is fine at the output of a binary classifier; it is not fine
          between layers 3 and 4 of a ResNet.
        </p>
        <p>
          <strong className="text-term-amber">ReLU at the output:</strong>{' '}
          only makes sense if the target is non-negative (e.g. predicting
          a count). Otherwise the network can never predict a negative
          value, which is usually not what you want.
        </p>
        <p>
          <strong className="text-term-amber">Zero-centered inputs:</strong>{' '}
          sigmoid outputs live in{' '}
          <code className="text-dark-text-primary">(0, 1)</code> — the{' '}
          <em>mean</em> of activations is positive. This pushes gradients
          in one direction and slows training. Tanh fixes that (outputs
          are mean-zero) but still saturates in the tails. ReLU
          doesn&apos;t care.
        </p>
      </Gotcha>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          You&apos;ve seen the switch and the dimmer. You&apos;ve seen the
          math. Now write both three times — each shorter than the last,
          and the third one ships with autograd. Pure Python, NumPy,
          PyTorch. Nobody implements these by hand in production. Knowing
          what&apos;s underneath is the point.
        </p>
      </Prose>

      <LayeredCode
        layers={[
          {
            label: 'pure python',
            caption: 'activations_scratch.py',
            runnable: true,
            code: `import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1.0 - s)             # f'(x) = f(x)(1 - f(x))

def relu(x):
    return x if x > 0 else 0.0       # max(0, x)

def relu_deriv(x):
    return 1.0 if x > 0 else 0.0

print(f"σ(0.5)={sigmoid(0.5):.4f}  σ'(0.5)={sigmoid_deriv(0.5):.4f}")
print(f"ReLU(-1.2)={relu(-1.2)}  ReLU'(-1.2)={relu_deriv(-1.2)}")`,
            output: `σ(0.5)=0.6225  σ'(0.5)=0.2350
ReLU(-1.2)=0.0  ReLU'(-1.2)=0`,
          },
          {
            label: 'numpy',
            caption: 'activations_numpy.py',
            runnable: true,
            code: `import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))                # elementwise on any array

def sigmoid_deriv_from_output(s):
    return s * (1.0 - s)                           # pass in the already-computed σ(x)

def relu(x):
    return np.maximum(0.0, x)                      # branchless — the trick on CPUs + GPUs

def relu_deriv(x):
    return (x > 0).astype(x.dtype)                 # 1 where x>0, 0 elsewhere

x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
print("σ(x)    =", np.round(sigmoid(x), 4))
print("ReLU(x) =", relu(x))`,
          },
          {
            label: 'pytorch',
            caption: 'activations_pytorch.py',
            code: `import torch
import torch.nn.functional as F

x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)

sig = torch.sigmoid(x)              # same as F.sigmoid — elementwise σ(x)
rel = F.relu(x)                     # same as x.clamp_min(0)

print("after sigmoid:", torch.round(sig.detach(), decimals=4))
print("after relu:   ", rel.detach())

# Autograd handles the derivatives — no hand-rolled backward required.
loss = rel.sum()
loss.backward()                     # gradient of sum(relu(x)) w.r.t. x is relu_deriv(x)
print("grad on relu input:", x.grad)`,
            output: `after sigmoid: tensor([0.1192, 0.3775, 0.5000, 0.6225, 0.8808])
after relu:    tensor([0.0000, 0.0000, 0.0000, 0.5000, 2.0000])
grad on relu input: tensor([0., 0., 0., 1., 1.])`,
          },
        ]}
      />

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for each scalar x: math.exp(-x)',
            right: 'np.exp(-x)  # vector-in, vector-out',
            note: 'one call replaces the Python loop',
          },
          {
            left: '1.0 if x > 0 else 0.0',
            right: '(x > 0).astype(float)',
            note: 'elementwise comparison — yields a boolean mask',
          },
          {
            left: 'max(0, x)',
            right: 'np.maximum(0, x)',
            note: 'broadcasted max, no branches',
          },
        ]}
      />

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'sigmoid(x) = 1/(1+np.exp(-x))',
            right: 'torch.sigmoid(x)',
            note: 'same math, tracked for autograd, runs on GPU',
          },
          {
            left: 'np.maximum(0, x)',
            right: 'F.relu(x) or x.clamp_min(0)',
            note: 'canonical PyTorch call — both compile identically',
          },
          {
            left: 'relu_deriv(x) = (x > 0).astype(float)',
            right: 'loss.backward()',
            note: 'you never write this — autograd traces it from F.relu',
          },
        ]}
      />

      <Callout variant="insight" title="the point of the three layers">
        Pure Python shows the definitions with zero ceremony. NumPy
        teaches you to think element-wise — the habit every ML engineer
        builds until they stop writing loops. PyTorch collapses both to a
        one-liner and hands you autograd for free. Same two functions,
        three phases of your learning life.
      </Callout>

      {/* ── Challenge + Takeaways ───────────────────────────────── */}
      <Challenge prompt="Kill every ReLU on purpose">
        <p>
          Build a tiny network in PyTorch with a single hidden layer of
          128 ReLU neurons. Set the bias init to <code>-3.0</code> (very
          negative) and the input standard deviation to <code>0.1</code>{' '}
          (tiny). Run one forward pass on a batch of 64.
        </p>
        <p className="mt-2">
          Count how many of the 128 neurons produced a non-zero output on
          any example. That is your &ldquo;alive&rdquo; count. With these
          settings most will be dead — every one of those switches jammed
          in the off position for the rest of training. Now swap in{' '}
          <code>nn.LeakyReLU(0.1)</code> and re-run. Every neuron will
          have at least a small gradient.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: put the network into a training loop for 200 steps. Plot
          the alive-count against step. Watch the ReLU neurons stay dead
          and the Leaky ReLU neurons recover.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> An activation&apos;s
          derivative is a multiplier that backprop applies at every
          layer — if it&apos;s too small or zero, gradients die. Sigmoid
          is the dimmer: it saturates in the tails, its derivative caps
          at 0.25, and multiplying twenty numbers ≤ 0.25 together is
          lethal. ReLU is the light switch: its derivative is a flat 1
          when on, which is why it scaled. Its failure mode — the
          switch stuck off — is real but mostly solved by careful init
          and occasional Leaky / GELU swaps.
        </p>
        <p>
          <strong>Next up — Softmax.</strong> Activations inside the
          network, done. The last thing a classifier does is take a
          vector of raw scores (&ldquo;logits&rdquo;) and turn them into
          a probability distribution over classes. That is softmax — the
          sigmoid&apos;s older sibling, generalised to <code>k</code>{' '}
          outputs. It is also the function cross-entropy loss is defined
          against, which makes it the most load-bearing piece of math
          you&apos;ll meet this section. And it hides a numerical trap
          that blows up models in production. We&apos;ll find it.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'ImageNet Classification with Deep Convolutional Neural Networks',
            author: 'Krizhevsky, Sutskever, Hinton',
            venue: 'NeurIPS 2012 — the AlexNet paper',
            url: 'https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html',
          },
          {
            title: 'Rectified Linear Units Improve Restricted Boltzmann Machines',
            author: 'Nair, Hinton',
            venue: 'ICML 2010',
            url: 'https://icml.cc/Conferences/2010/papers/432.pdf',
          },
          {
            title: 'Gaussian Error Linear Units (GELUs)',
            author: 'Hendrycks, Gimpel',
            year: 2016,
            url: 'https://arxiv.org/abs/1606.08415',
          },
          {
            title: 'Understanding the difficulty of training deep feedforward neural networks',
            author: 'Glorot, Bengio',
            venue: 'AISTATS 2010 — the Xavier init paper',
            url: 'https://proceedings.mlr.press/v9/glorot10a.html',
          },
        ]}
      />
    </div>
  )
}
