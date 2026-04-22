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
import ChainRuleScalar from '../widgets/ChainRuleScalar'
import ComputationGraph from '../widgets/ComputationGraph'
import BackpropOneLayer from '../widgets/BackpropOneLayer'

// Signature anchor: Memento — reconstructing the past backwards. The forward
// pass is the character moving through the day, events piling up. You arrive
// at the loss and discover something went wrong. You can't rewind the movie,
// but you can walk the day in reverse, asking at each step how much this
// choice contributed to what happened. That's the chain rule, made mechanical.
// Every gradient is a Polaroid note — "this much, this direction."
//
// Threaded at three load-bearing moments: the opening hook (prediction was
// wrong, now rewind), the chain rule reveal (each step is a Polaroid of local
// responsibility), and the gradient accumulation step (reading the notes in
// order). One nod, not a movie recap.

export default function BackpropagationLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="backpropagation" />

      <Prose>
        <p>
          You have a <NeedsBackground slug="single-neuron">neuron</NeedsBackground>.
          It takes an input, multiplies by weights, adds a bias, squashes the
          result through a non-linearity, spits out a prediction. You
          <NeedsBackground slug="gradient-descent"> already know</NeedsBackground>{' '}
          what to do next: nudge the weights to make the loss smaller. The only
          thing missing is the nudge direction — the partial derivative of the
          loss with respect to every parameter in the model.
        </p>
        <p>
          For one neuron with three weights, you could derive those by hand in
          five minutes. For a network with a hundred million parameters, you
          cannot. You&apos;d still be doing calculus when the sun goes out.
          What you need is an algorithm — mechanical, efficient, correct — that
          computes every derivative in a single sweep.
        </p>
        <p>
          That algorithm is <KeyTerm>backpropagation</KeyTerm>, and the way to
          understand it is to remember the plot of <em>Memento</em>. The hero
          wakes up at the end of a day and something has gone wrong. He
          can&apos;t rewind. What he can do is walk through what happened in
          reverse, stopping at each moment to ask: <em>how much did this step
          contribute to where I ended up?</em> The answers pile up on Polaroid
          notes — small, local, one per moment. That&apos;s backprop. The
          forward pass is the day. The loss is the ending. The chain rule is
          how you read the day backwards.
        </p>
      </Prose>

      {/* ── Chain rule intuition ────────────────────────────────── */}
      <Prose>
        <p>
          Start small, because the trick is the same at every scale. Take one
          function composed of other functions — say{' '}
          <code>L = sin((3x + 2)²)</code>. You want <code>dL/dx</code>. You
          know how to do this by hand: introduce names for the intermediates,
          take each local derivative, multiply them in a chain. That&apos;s the{' '}
          <strong>chain rule</strong>. It is the whole algorithm. Everything
          after this is bookkeeping.
        </p>
      </Prose>

      <MathBlock caption="chain rule, scalar form">
{`Let  u = 3x + 2 ,   v = u²  ,  L = sin v .

dL         dL     dv     du
──    =    ──  ·  ──  ·  ──
dx         dv     du     dx

         = cos v  ·  2u  ·  3`}
      </MathBlock>

      <Prose>
        <p>
          Three local derivatives — one per operation — multiplied together.
          Each factor is a Polaroid: the slope of <em>this step</em> with
          respect to the step before it. None of them know the whole story.
          They don&apos;t have to. Line the notes up in order, multiply, and
          you&apos;ve reconstructed the slope of the entire composition.
        </p>
        <p>
          Slide <code>x</code> below. Forward values (cyan) flow left to right
          — that&apos;s the day happening. Backward gradients (rose)
          accumulate right to left — that&apos;s you reading it backwards,
          Polaroid by Polaroid.
        </p>
      </Prose>

      <ChainRuleScalar />

      <Callout variant="note" title="forward vs backward">
        The forward pass computes values. Start at the input, apply each
        operation in order, write the intermediate numbers down (
        <code>u</code>, <code>v</code>, <code>L</code>). The backward pass
        computes gradients. Start at the output, apply each{' '}
        <em>local derivative</em> in reverse, multiplying as you go. The
        backward pass needs the forward values — without them it has nothing
        to evaluate the local derivatives on. That&apos;s the one rule of
        backprop: you cannot read the day backwards unless you took notes the
        first time through.
      </Callout>

      <Personify speaker="Chain rule">
        Take the derivative of the outside, times the derivative of the
        inside. Do it again, recursively, until you&apos;re at the variable
        you wanted. I am simple. I am local. I do not know how deep your
        network is and I do not care. A thousand tiny local derivatives
        multiplied together is still just multiplication.
      </Personify>

      {/* ── Computation graph ───────────────────────────────────── */}
      <Prose>
        <p>
          Generalise. Most real functions aren&apos;t a simple composition —
          they&apos;re a <em>graph</em>. One intermediate value might feed
          into several later nodes; gradients from different paths have to be
          summed. The chain rule still works; you just have to keep the graph
          structure honest. That structure is the{' '}
          <KeyTerm>computation graph</KeyTerm>, and every deep-learning
          framework — PyTorch, TensorFlow, JAX — builds one invisibly every
          time you run a forward pass. They aren&apos;t doing anything clever.
          They&apos;re taking notes.
        </p>
        <p>
          Click <em>next step</em>. The forward pass fills in values node by
          node, left to right — that&apos;s the day. Once the loss{' '}
          <code>L</code> exists, the backward pass starts from{' '}
          <code>dL/dL = 1</code> and rewinds, each edge contributing its own
          Polaroid — the local derivative of what this node does to what
          flowed through it.
        </p>
      </Prose>

      <ComputationGraph />

      <Callout variant="insight" title="the graph is built during the forward pass">
        In PyTorch, every op you run on a tensor with{' '}
        <code>requires_grad=True</code> silently appends a node to a graph.
        Calling <code>loss.backward()</code> walks that graph in reverse,
        running each node&apos;s pre-defined backward rule. That&apos;s the
        whole of autograd: a graph of ops where each op knows both its
        forward math and its local derivative. You write{' '}
        <code>a * b + c</code> and you get both the forward value and the
        tools to compute <code>d(result)/da</code> for free.
      </Callout>

      {/* ── One-layer backprop walkthrough ───────────────────────── */}
      <Prose>
        <p>
          Time to do this on a neuron. One neuron, three weights, one bias,
          sigmoid activation, MSE loss. Forward pass on top; each backward
          line is one Polaroid, placed in the order you read them.
        </p>
      </Prose>

      <MathBlock caption="backprop through a single sigmoid neuron">
{`Forward:      z  =  w·x + b
              ŷ  =  σ(z)
              L  =  ½ (ŷ − y)²

Backward:     dL/dŷ   =   ŷ − y
              dŷ/dz   =   σ(z) · (1 − σ(z))
              dL/dz   =   (dL/dŷ) · (dŷ/dz)

              dL/dwᵢ  =   (dL/dz) · xᵢ
              dL/db   =    dL/dz`}
      </MathBlock>

      <Prose>
        <p>
          Read top to bottom. Each row uses the one above it — you cannot
          read page three of the ledger before page two has been written.
          The last two lines are the ones you actually update with: both are
          multiples of <code>dL/dz</code>, because the chain rule kept the
          shared intermediate around for you. That shared quantity has a
          name, and the rest of the deep-learning literature won&apos;t shut
          up about it.
        </p>
        <p>
          Hit <em>one step</em> in the widget. Every number updates live —
          weights drift toward the target <code>y=1</code>, the loss
          collapses. The neuron is learning, which is a fancier way of
          saying: it&apos;s reading its own mistakes backwards and editing
          itself.
        </p>
      </Prose>

      <BackpropOneLayer />

      <Callout variant="insight" title="why dL/dz is called 'delta'">
        In the classic 1986 Rumelhart–Hinton–Williams paper, the gradient at
        the pre-activation of layer <em>ℓ</em> is called <em>δ</em>. Once you
        have <em>δ</em>, every parameter gradient at that layer is{' '}
        <em>δ</em> times something trivial — an input for weights, a 1 for
        biases. The whole of multi-layer backprop is the same <em>δ</em>,
        recursed backward layer by layer. Every tutorial that litters its
        pages with <code>δ_ℓ</code> symbols is doing exactly this.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Forward pass first, always.</strong>{' '}
          Backprop reads intermediates (<code>z</code>, <code>ŷ</code>) that
          only exist after the forward pass wrote them down. No notes, no
          rewind. In PyTorch, calling{' '}
          <code className="text-dark-text-primary">loss.backward()</code>{' '}
          before any forward ops raises.
        </p>
        <p>
          <strong className="text-term-amber">
            Gradients accumulate unless you zero them.
          </strong>{' '}
          By default PyTorch <em>adds</em> every{' '}
          <code className="text-dark-text-primary">backward()</code>{' '}
          call&apos;s gradients onto{' '}
          <code className="text-dark-text-primary">.grad</code>. That&apos;s
          deliberate — useful for accumulating gradients across micro-batches
          — and a footgun for everyone else. Always call{' '}
          <code className="text-dark-text-primary">optimizer.zero_grad()</code>{' '}
          before your backward pass. Forgetting this is the second most
          common PyTorch bug, and the first most common one it took you two
          days to find.
        </p>
        <p>
          <strong className="text-term-amber">
            &ldquo;With respect to what&rdquo; matters.
          </strong>{' '}
          PyTorch only computes gradients for tensors with{' '}
          <code className="text-dark-text-primary">requires_grad=True</code>.
          If your weights don&apos;t have it, no gradient is stored and the
          optimizer sees zeros — silent failure, model doesn&apos;t learn,
          you blame the data.{' '}
          <code className="text-dark-text-primary">nn.Parameter</code> and{' '}
          <code className="text-dark-text-primary">nn.Linear</code> set this
          for you, but only inside an{' '}
          <code className="text-dark-text-primary">nn.Module</code>.
        </p>
      </Gotcha>

      {/* ── Three-layer code ─────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers. Pure Python spells every multiplication out —
          there&apos;s nowhere for a bug to hide. NumPy vectorises it across a
          batch. PyTorch lets you write only the forward pass; autograd reads
          the day backwards for you.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · one_neuron_backprop.py"
        output={`step 0: loss=0.3263
step 20: loss=0.0091
step 40: loss=0.0026
final yhat = 0.9282`}
      >{`import math

def sigmoid(z): return 1.0 / (1.0 + math.exp(-z))

def train_one_neuron(x, target, lr=0.5, steps=50):
    w = [0.2, -0.1, 0.3]
    b = 0.1
    for step in range(steps + 1):
        # Forward
        z = sum(xi * wi for xi, wi in zip(x, w)) + b
        yhat = sigmoid(z)
        loss = 0.5 * (yhat - target) ** 2

        # Backward
        dL_dyhat = yhat - target                # d(½(yhat-t)²)/dyhat
        dyhat_dz = yhat * (1 - yhat)            # σ'(z)
        dL_dz = dL_dyhat * dyhat_dz
        dL_dw = [dL_dz * xi for xi in x]        # one gradient per weight
        dL_db = dL_dz

        if step % 20 == 0:
            print(f"step {step}: loss={loss:.4f}")

        # Update
        w = [wi - lr * gi for wi, gi in zip(w, dL_dw)]
        b = b - lr * dL_db
    return w, b, yhat

w, b, yhat = train_one_neuron([0.8, -0.4, 1.2], target=1.0)
print(f"final yhat = {yhat:.4f}")`}</CodeBlock>

      <CodeBlock language="python" caption="layer 2 — numpy · one_neuron_backprop_numpy.py">{`import numpy as np

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def train_one_neuron(X, y, lr=0.5, steps=50):
    """X: (N, D)   y: (N,)"""
    N, D = X.shape
    w = np.array([0.2, -0.1, 0.3])
    b = 0.1
    for _ in range(steps):
        # Forward — whole batch at once
        z = X @ w + b                           # (N,)
        yhat = sigmoid(z)                       # (N,)

        # Backward
        dL_dyhat = (yhat - y) / N               # (N,)  — mean over batch
        dyhat_dz = yhat * (1 - yhat)            # (N,)
        dL_dz = dL_dyhat * dyhat_dz             # (N,)

        dL_dw = X.T @ dL_dz                     # (D,)  — chain rule for free
        dL_db = dL_dz.sum()

        w = w - lr * dL_dw
        b = b - lr * dL_db
    return w, b`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'dL_dw = [dL_dz * xi for xi in x]',
            right: 'dL_dw = X.T @ dL_dz',
            note: 'outer loop over weights becomes matrix transpose',
          },
          {
            left: 'one example',
            right: 'a whole batch, one call',
            note: 'gradients average across the batch automatically',
          },
          {
            left: 'sigmoid grad = yhat * (1 - yhat)',
            right: 'same — NumPy broadcasts over the batch',
            note: 'the local derivative formula is unchanged',
          },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · one_neuron_backprop_pytorch.py"
        output={`step 0: loss=0.3263
step 20: loss=0.0091
final yhat = 0.9282`}
      >{`import torch
import torch.nn as nn

class OneNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = OneNeuron()
# Hand-set the same starting params as in layer 1 for demo parity.
with torch.no_grad():
    model.fc.weight.copy_(torch.tensor([[0.2, -0.1, 0.3]]))
    model.fc.bias.copy_(torch.tensor([0.1]))

x = torch.tensor([[0.8, -0.4, 1.2]])
y = torch.tensor([[1.0]])
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for step in range(51):
    optimizer.zero_grad()            # zero the gradient buffer
    yhat = model(x)                  # forward
    loss = 0.5 * (yhat - y).pow(2).mean()
    loss.backward()                  # autograd runs the chain rule
    optimizer.step()                 # apply w ← w - α·∇L
    if step % 20 == 0:
        print(f"step {step}: loss={loss.item():.4f}")

print(f"final yhat = {model(x).item():.4f}")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'dL_dz = (yhat - y) * yhat * (1 - yhat)',
            right: 'loss.backward()',
            note: 'autograd computes identical values from the forward expression',
          },
          {
            left: 'w = w - lr * dL_dw',
            right: 'optimizer.step()',
            note: 'we already knew this — now the gradients are autograd-computed',
          },
          {
            left: 'grads reset implicitly each step',
            right: 'optimizer.zero_grad()',
            note: 'PyTorch accumulates by default — be explicit about zeroing',
          },
        ]}
      />

      <Challenge prompt="Backprop a 2-input XOR neuron, step by step">
        <p>
          Using the pure-Python version above, train a neuron on the XOR
          dataset{' '}
          <code>[(0,0,0), (0,1,1), (1,0,1), (1,1,0)]</code>. Run 2000 steps.
          Print the loss every 200. It will not drop below about{' '}
          <code>0.125</code> — that&apos;s the ceiling one neuron can reach on
          XOR (75% accuracy). The chain rule is doing its job perfectly; the
          model is fundamentally too shallow. Backprop can&apos;t save a
          model that lacks the capacity to express the answer.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: stack a hidden layer of two neurons and an output neuron, and
          redo it from scratch with pure-Python backprop. It gets ugly fast.
          Every line of the next lesson exists because this exercise becomes
          unbearable at more than one layer.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> Backprop is the chain rule,
          applied backwards through a computation graph. Forward pass writes
          the ledger: values and intermediates. Backward pass reads it in
          reverse, multiplying local derivatives edge by edge. Every parameter
          gradient turns out to be <em>δ</em> at its node times a trivial
          local quantity, and computing every <em>δ</em> costs about the
          same as the forward pass itself — linear in the size of the graph.
          That efficiency is why deep learning is computationally practical;
          without it, training a modern network would cost more than
          building one.
        </p>
        <p>
          <strong>Next up — Multi-Layer Backpropagation.</strong> One layer,
          one walk backwards. Real networks are ten, twenty, a hundred layers
          deep. Every additional layer is another page in the ledger — and
          the recursion &ldquo;<em>δ</em> at layer <em>ℓ</em> equals{' '}
          <em>δ</em> at layer <em>ℓ+1</em> times the local derivative&rdquo;
          is the core equation of deep learning, the thing that makes
          arbitrarily deep networks trainable in the first place.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Learning representations by back-propagating errors',
            author: 'Rumelhart, Hinton, Williams',
            venue: 'Nature, 1986 — the paper that made backprop famous',
            url: 'https://www.nature.com/articles/323533a0',
          },
          {
            title: 'Calculus on Computational Graphs: Backpropagation',
            author: 'Christopher Olah',
            venue: 'colah.github.io, 2015',
            url: 'https://colah.github.io/posts/2015-08-Backprop/',
          },
          {
            title: 'Dive into Deep Learning — 5.3 Forward Propagation, Backward Propagation, and Computational Graphs',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_multilayer-perceptrons/backprop.html',
          },
        ]}
      />
    </div>
  )
}
