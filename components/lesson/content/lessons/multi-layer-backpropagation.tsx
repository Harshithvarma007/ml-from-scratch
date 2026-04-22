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
import LayeredBackprop from '../widgets/LayeredBackprop'
import GradientFlowDepth from '../widgets/GradientFlowDepth'

// Anchor: the blame chain up a corporate hierarchy. The loss is a customer
// complaint at the CEO's desk. She blames her VPs. They blame their
// managers. All the way down to the intern who read the raw input. Every
// weight gets a blame note proportional to how much it contributed. The
// math is the chain rule; the metaphor makes the flow obvious — and sets up
// weight-init by showing how blame collapses or explodes down a long chain.
export default function MultiLayerBackpropagationLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="multi-layer-backpropagation" />

      <Prose>
        <p>
          One-neuron <NeedsBackground slug="backpropagation">backprop</NeedsBackground>{' '}
          was a single-manager operation. One weight misbehaved, one gradient
          told it how to behave better, you updated and moved on. Easy org
          chart. Short meeting.
        </p>
        <p>
          A multi-layer network is a corporate hierarchy. The loss is a
          customer complaint that lands on the CEO&apos;s desk — the output
          layer, which is the only one that ever sees the right answer. The
          CEO was wrong. Rather than fire herself, she walks down the org
          chart handing out blame notes: <em>this much of the mistake was
          because of your numbers</em>. Her VPs take their share, break it
          down further for their managers, and so on. All the way down to the
          first hidden layer — the intern who read the raw input. Every
          weight in the company gets a blame note proportional to how much it
          contributed to the miss. Then everyone adjusts.
        </p>
        <p>
          That&apos;s backprop through depth. The math is still the chain
          rule, just now with a subscript per layer. This lesson is where the
          recursion shows up — the pattern that lets you compute gradients
          through a network of arbitrary depth in the same asymptotic time as
          the forward pass. It&apos;s the reason deep learning is
          computationally feasible. It&apos;s also the reason training deep
          nets was impossible for twenty years between Minsky&apos;s 1969
          book and Rumelhart-Hinton-Williams&apos; 1986 rediscovery. Once you
          know the recursion it feels obvious. Before, it genuinely
          isn&apos;t.
        </p>
      </Prose>

      {/* ── The recursion ───────────────────────────────────────── */}
      <Prose>
        <p>
          Set up the company. Number the layers <code>ℓ = 1, 2, …, L</code>.
          The intern is layer 1, the CEO is layer <code>L</code>. Every
          manager does the same job — take what the layer below reported,
          multiply by your own weights, add a bias, squash through a{' '}
          <NeedsBackground slug="single-neuron">nonlinearity</NeedsBackground>,
          pass it up:
        </p>
      </Prose>

      <MathBlock caption="a generic dense layer">
{`z_ℓ   =   W_ℓ · a_(ℓ-1)  +  b_ℓ                 linear combo
a_ℓ   =   f(z_ℓ)                                activation

Inputs to the first layer:    a_0 = x
Outputs of the last layer:    ŷ = a_L`}
      </MathBlock>

      <Prose>
        <p>
          Now follow one complaint backwards. Define{' '}
          <code>δ_ℓ  ≡  ∂L / ∂z_ℓ</code> — the blame note for layer{' '}
          <code>ℓ</code>&apos;s pre-activation. It&apos;s the same quantity
          you computed for a single neuron last lesson, just with a
          subscript. Once a manager holds their δ, the gradients they need
          to actually fix anything are trivial — weights and biases fall out
          immediately:
        </p>
      </Prose>

      <MathBlock caption="parameter gradients from δ — identical at every layer">
{`∂L/∂W_ℓ   =   δ_ℓ · a_(ℓ-1)ᵀ            outer product: (out, in) shape
∂L/∂b_ℓ   =   δ_ℓ                        same shape as b_ℓ`}
      </MathBlock>

      <Prose>
        <p>
          So the only real question is: how does a middle manager get their
          δ in the first place? Answer: from the manager directly above, who
          already has theirs. That&apos;s the punchline — the <em>one
          equation</em> that makes deep learning possible, and the rule that
          runs the whole corporate blame chain:
        </p>
      </Prose>

      <MathBlock caption="the backprop recurrence — memorize this">
{`δ_ℓ   =   (W_(ℓ+1)ᵀ · δ_(ℓ+1))  ⊙  f'(z_ℓ)

Base case (at the output layer):
δ_L   =   ∂L / ∂a_L  ⊙  f'(z_L)

where  ⊙  is elementwise multiplication.`}
      </MathBlock>

      <Prose>
        <p>
          Read the recurrence as a conversation. The manager above hands you
          their blame note <code>δ_(ℓ+1)</code>. You pull it back through the
          weights that connected you to them (<code>W_(ℓ+1)ᵀ</code> — the
          transpose is how the blame flows against the direction of the
          forward pass). Then you mask by your own slope — <code>f&apos;(z_ℓ)</code> —
          because a neuron that was sitting in a flat part of its activation
          didn&apos;t move the needle, so it doesn&apos;t deserve much
          blame. That&apos;s your δ. Hand it to the manager below. Repeat
          until you reach the intern. Every step is cheap and local. Total
          cost: one matmul per layer — the same asymptotic cost as the
          forward pass.
        </p>
      </Prose>

      <Personify speaker="δ (delta)">
        I am the blame note. If you tell me how wrong the CEO was, I can
        walk myself down every floor of the building and deliver the right
        share to every manager on the way. I&apos;m the only reason any
        weight in the network knows which way to move. Without me, nobody
        downstairs even knows there was a complaint.
      </Personify>

      {/* ── Animated layered backprop ────────────────────────────── */}
      <Prose>
        <p>
          Click through the steps below. Each click either advances the
          forward pass one layer or walks the blame one more floor down the
          org chart. Watch the forward values fill in left to right (cyan) —
          that&apos;s the company doing its actual job. Then the pink δ
          bubbles appear right to left — that&apos;s the walk of shame. The
          gradient sitting on each neuron is the blame note the recurrence
          just handed it.
        </p>
      </Prose>

      <LayeredBackprop />

      <Callout variant="insight" title="why the algorithm is O(network size)">
        Naively, computing <code>∂L/∂w</code> for every weight by running a
        separate forward pass per weight (finite differences) costs{' '}
        <code>O(P · forward_cost)</code> where <code>P</code> is parameter
        count — hundreds of billions of forward passes for a modern LLM.
        Backprop is <code>O(forward_cost)</code> total. That&apos;s an
        eight-order-of-magnitude improvement. Or: instead of auditing every
        employee individually, the CEO issues one complaint and the org
        chart distributes it in a single sweep. If you ever feel confused
        about why training works at all, that ratio is the answer.
      </Callout>

      {/* ── Vanishing gradients ─────────────────────────────────── */}
      <Prose>
        <p>
          There&apos;s a catch hiding in the recurrence, and it&apos;s the
          thing that drives the next two lessons. At every layer, the blame
          note gets multiplied by <code>f&apos;(z)</code>. For sigmoid and
          tanh, that derivative is at most <code>0.25</code>, and usually
          smaller. Every floor the note passes through, it shrinks. After
          twenty floors — <code>(0.25)²⁰ ≈ 10⁻¹²</code> — the intern gets a
          blame note that reads &ldquo;you were wrong by zero point zero
          zero zero zero zero zero zero zero zero zero zero one.&rdquo; The
          intern shrugs and does nothing. Your early layers stop training.
        </p>
        <p>
          Run the arithmetic the other way and the same mechanism blows up.
          If the weights are large, the chain-rule product of Jacobians
          compounds: the intern gets a blame note the size of a small
          country and updates themselves into orbit. The model is cooked.
        </p>
        <p>
          This is the <KeyTerm>vanishing gradient problem</KeyTerm> and its
          evil twin, the exploding gradient problem — and it was why
          sigmoid + deep networks just didn&apos;t work in the 1990s. The
          fix is a stack of architectural and initialisation choices
          we&apos;ll unpack over the next few lessons:{' '}
          <NeedsBackground slug="backpropagation">ReLU</NeedsBackground>{' '}
          (derivative is 1 where active, so the note survives), careful
          weight initialisation (Xavier/He — two lessons from now, and the
          whole reason it&apos;s in the curriculum), and residual
          connections (a standalone equation:{' '}
          <code>a_ℓ = f(W · a_(ℓ-1)) + a_(ℓ-1)</code>, which adds an
          identity path the blame note can use when the multiplicative one
          collapses).
        </p>
      </Prose>

      <GradientFlowDepth />

      <Prose>
        <p>
          Move the depth slider. With <code>L = 10</code>, all three
          activations still have usable gradients — the org chart is short
          enough that the note arrives with signal. By <code>L = 20</code>,
          sigmoid + naive init has collapsed to <code>10⁻⁹</code>. By{' '}
          <code>L = 30</code> it&apos;s under float precision — the early
          layers literally cannot train, blame note arrives as a rounding
          error. Now tick &ldquo;residual connections.&rdquo; Every curve
          flattens or recovers. Residual paths add an identity gradient
          term <code>∂a_ℓ/∂a_(ℓ-1) = I + (chain stuff)</code>, so depth
          stops compounding the vanishing factor. That&apos;s the entire
          mathematical justification for ResNet, and it&apos;s also why
          every modern transformer has one of those{' '}
          <code>x + attention(x)</code> shortcuts wired around each sublayer.
        </p>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Gradient clipping fixes the other failure mode.</strong>{' '}
          When the chain-rule product explodes instead of collapses — common
          with bad init or recurrent nets — each step sends the model off
          to infinity. Clip the gradient norm to some ceiling —{' '}
          <code className="text-dark-text-primary">torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)</code>{' '}
          — before calling <code className="text-dark-text-primary">optimizer.step()</code>.
          Cheap, almost always correct, done.
        </p>
        <p>
          <strong className="text-term-amber">Don&apos;t confuse δ with ∂L/∂a.</strong>{' '}
          δ<sub>ℓ</sub> is the blame at the <em>pre-activation</em>. The
          gradient at the post-activation is{' '}
          <code className="text-dark-text-primary">W_(ℓ+1)ᵀ · δ_(ℓ+1)</code>{' '}
          (which is δ at the <em>next</em> layer, before the activation).
          This distinction matters when you implement backprop by hand —
          off by one layer and nothing works.
        </p>
        <p>
          <strong className="text-term-amber">Activation derivatives need the pre-activation.</strong>{' '}
          <code className="text-dark-text-primary">f&apos;(z_ℓ)</code>, not{' '}
          <code className="text-dark-text-primary">f&apos;(a_ℓ)</code>. For
          sigmoid there&apos;s a shortcut (
          <code className="text-dark-text-primary">σ&apos;(z) = σ(z)(1-σ(z)) = a(1-a)</code>
          ) because the derivative can be written in terms of the output.
          For most other activations you have to stash the pre-activation
          on the forward pass. Forget, and your backward pass quietly
          lies — the worst kind of bug.
        </p>
      </Gotcha>

      {/* ── Three-layer code ─────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations, same company, three different management
          styles. Pure Python runs the recurrence with explicit nested loops
          — every blame note computed by hand, painful but unambiguous.
          NumPy compresses the per-layer step into three matrix operations.
          PyTorch lets you write only the forward pass and throws away the
          whole lesson — you declare what the company does, it figures out
          the blame chain for you.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · mlp_backprop_scratch.py"
        output={`final loss = 0.0183`}
      >{`import math, random

def sigmoid(z): return 1/(1+math.exp(-z))
def relu(z): return z if z > 0 else 0.0

random.seed(0)
# 3 → 4 → 1 net, sigmoid output, MSE loss
def init_matrix(inD, outD): return [[random.gauss(0, 0.5) for _ in range(inD)] for _ in range(outD)]
W1 = init_matrix(3, 4); b1 = [random.gauss(0, 0.1) for _ in range(4)]
W2 = init_matrix(4, 1); b2 = [0.0]

def train_step(x, y, lr=0.3):
    global W1, b1, W2, b2
    # Forward
    z1 = [sum(W1[i][j]*x[j] for j in range(3)) + b1[i] for i in range(4)]
    a1 = [relu(v) for v in z1]
    z2 = sum(W2[0][j]*a1[j] for j in range(4)) + b2[0]
    yhat = sigmoid(z2)
    loss = 0.5 * (yhat - y) ** 2

    # Backward
    d2 = (yhat - y) * yhat * (1 - yhat)              # δ at output
    dW2 = [[d2 * a1[j] for j in range(4)]]           # ∂L/∂W2 = δ · a1ᵀ
    db2 = [d2]

    d1 = [sum(W2[0][j] * d2 for _ in [0]) * (1 if z1[j] > 0 else 0)
          for j in range(4)]                         # δ at hidden (ReLU')
    dW1 = [[d1[i] * x[j] for j in range(3)] for i in range(4)]
    db1 = list(d1)

    # Update
    for i in range(4):
        for j in range(3):
            W1[i][j] -= lr * dW1[i][j]
        b1[i] -= lr * db1[i]
    for j in range(4):
        W2[0][j] -= lr * dW2[0][j]
    b2[0] -= lr * db2[0]
    return loss

for step in range(1000):
    loss = train_step([0.6, -0.4, 0.9], 1.0)
print(f"final loss = {loss:.4f}")`}</CodeBlock>

      <CodeBlock language="python" caption="layer 2 — numpy · mlp_backprop_numpy.py">{`import numpy as np

rng = np.random.default_rng(0)
W1 = rng.normal(0, 0.5, size=(4, 3))   # layer 1: 3 → 4
b1 = rng.normal(0, 0.1, size=(4,))
W2 = rng.normal(0, 0.5, size=(1, 4))   # layer 2: 4 → 1
b2 = np.zeros(1)

def sigmoid(z): return 1 / (1 + np.exp(-z))

def step(X, y, lr=0.3):
    global W1, b1, W2, b2
    # Forward (batched)
    z1 = X @ W1.T + b1                                 # (N, 4)
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2.T + b2                                # (N, 1)
    yhat = sigmoid(z2).ravel()
    loss = 0.5 * np.mean((yhat - y) ** 2)

    # Backward — apply the recurrence in matrix form
    delta2 = ((yhat - y) * yhat * (1 - yhat)).reshape(-1, 1) / len(X)  # (N, 1)
    dW2 = delta2.T @ a1                                # (1, 4)
    db2 = delta2.sum(axis=0)                           # (1,)

    delta1 = (delta2 @ W2) * (z1 > 0)                  # (N, 4) · ReLU'
    dW1 = delta1.T @ X                                 # (4, 3)
    db1 = delta1.sum(axis=0)                           # (4,)

    # Update
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1
    return loss`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for i for j: dW1[i][j] = d1[i] * x[j]',
            right: 'dW1 = delta1.T @ X',
            note: 'nested loops collapse into one transpose-matmul',
          },
          {
            left: 'z1[j] > 0',
            right: '(z1 > 0).astype(float)',
            note: 'the ReLU derivative — elementwise mask',
          },
          {
            left: 'per-example loop',
            right: 'batch · broadcast · done',
            note: 'NumPy handles the N dimension for free',
          },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · mlp_backprop_pytorch.py"
        output={`step 0: loss = 0.1762
step 200: loss = 0.0193
step 600: loss = 0.0024`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

model = MLP()
optimizer = torch.optim.SGD(model.parameters(), lr=0.3)

x = torch.tensor([[0.6, -0.4, 0.9]])
y = torch.tensor([[1.0]])

for step in range(1001):
    optimizer.zero_grad()
    yhat = model(x)
    loss = 0.5 * (yhat - y).pow(2).mean()
    loss.backward()                      # autograd runs the full backward recurrence
    optimizer.step()
    if step in (0, 200, 600):
        print(f"step {step}: loss = {loss.item():.4f}")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'delta2 = (yhat - y) * yhat * (1-yhat)',
            right: 'loss.backward()  # one call',
            note: 'autograd walks the recurrence for every parameter',
          },
          {
            left: 'dW1 = delta1.T @ X  ; dW2 = delta2.T @ a1',
            right: 'stored in .grad automatically',
            note: 'you never write the outer product',
          },
          {
            left: 'transpose + matmul dance',
            right: 'model.parameters() — just weights to update',
            note: 'the actual gradient flow is invisible',
          },
        ]}
      />

      <Callout variant="insight" title="what the framework is actually doing">
        Every time you call <code>loss.backward()</code>, PyTorch walks the
        computation graph in reverse, calling a <code>backward</code>{' '}
        function registered for every op in the forward.{' '}
        <code>nn.Linear.backward</code> does exactly the{' '}
        <code>delta.T @ X / W.T @ delta</code> dance you just wrote.
        ReLU&apos;s backward is <code>grad_out * (z &gt; 0)</code>.
        Sigmoid&apos;s is <code>grad_out * a * (1-a)</code>. None of it is
        magic; it&apos;s a library of hand-written local derivatives,
        stitched together by the graph — a professional HR department that
        knows every blame formula by heart. You could build a miniature
        version of autograd in about 300 lines — the <em>micrograd</em>{' '}
        repo does exactly that.
      </Callout>

      <Challenge prompt="Build a 2-layer MLP that solves XOR">
        <p>
          Take the pure-Python two-layer code above. Set the architecture
          to 2 → 2 → 1 (two inputs, two ReLU hidden units, one sigmoid
          output). Train on XOR —{' '}
          <code>[(0,0,0), (0,1,1), (1,0,1), (1,1,0)]</code> — for 5000
          steps at <code>lr = 0.5</code>. It should hit loss below{' '}
          <code>0.01</code>. Print the hidden-layer weights at the end —
          interpret them. Do the two hidden neurons look like they learned
          to detect &ldquo;exactly one input is 1&rdquo;?
        </p>
        <p className="mt-2 text-dark-text-muted">
          This is the smallest network that solves XOR. You&apos;re
          repeating Rumelhart-Hinton-Williams&apos; 1986 result on a laptop
          in twelve lines of NumPy. That paper changed the course of AI.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> Multi-layer backprop is
          one equation repeated:{' '}
          <code>δ_ℓ = (W_(ℓ+1)ᵀ · δ_(ℓ+1)) ⊙ f&apos;(z_ℓ)</code>. A CEO
          with a complaint, an org chart walking blame downward, every
          manager&apos;s share proportional to how much they moved things.
          Parameter gradients fall out as outer products. Total cost equals
          the forward pass. And the dirty secret: the same product of
          Jacobians that makes the algorithm cheap is also what makes deep
          networks fragile — blame collapses to zero or explodes to
          infinity as the chain grows. The widget curves are that story
          made visible, and they&apos;re the whole reason{' '}
          <NeedsBackground slug="gradient-descent">gradient descent</NeedsBackground>{' '}
          in a deep net isn&apos;t automatically easy.
        </p>
        <p>
          <strong>Next up — Backprop Ninja.</strong> The theory holds. Now
          do the derivation yourself, on paper, for a two-layer MLP — every
          δ, every outer product, every elementwise mask — and we&apos;ll
          check your work with finite differences. If you got off by one
          transpose anywhere in this lesson, ninja will catch it. Bring a
          pencil.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Learning representations by back-propagating errors',
            author: 'Rumelhart, Hinton, Williams',
            venue: 'Nature, 1986',
            url: 'https://www.nature.com/articles/323533a0',
          },
          {
            title: 'Understanding the difficulty of training deep feedforward neural networks',
            author: 'Glorot, Bengio',
            venue: 'AISTATS 2010 — the paper that named the vanishing-gradient issue',
            url: 'https://proceedings.mlr.press/v9/glorot10a.html',
          },
          {
            title: 'Deep Residual Learning for Image Recognition',
            author: 'He, Zhang, Ren, Sun',
            venue: 'CVPR 2016 — ResNet, the skip-connection paper',
            url: 'https://arxiv.org/abs/1512.03385',
          },
          {
            title: 'micrograd',
            author: 'Andrej Karpathy',
            venue: 'github.com/karpathy/micrograd',
            url: 'https://github.com/karpathy/micrograd',
          },
        ]}
      />
    </div>
  )
}
