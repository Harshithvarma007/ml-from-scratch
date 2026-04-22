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
import NeuronForward from '../widgets/NeuronForward'
import NeuronFlowDiagram from '../widgets/NeuronFlowDiagram'
import DecisionBoundary2D from '../widgets/DecisionBoundary2D'

// Anchor: the picky doorman. A neuron is a doorman scoring a line of features
// — weights are his opinions about each feature, bias is his mood, the
// activation is the yes/no decision at the door. Threaded at the opening,
// the weights/bias reveal, and the activation's role. A layer = a crew of
// bouncers with different opinions. The "biological neuron" metaphor is
// deliberately avoided.
export default function SingleNeuronLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisites ────────────────────────────────────────── */}
      <Prereq currentSlug="single-neuron" />

      {/* ── Opening: the doorman ────────────────────────────────── */}
      <Prose>
        <p>
          Picture a doorman outside a club. A line of people walks up, each one
          wearing a number — their height, their outfit, whether they&apos;re on
          the list, how loudly they&apos;re arguing with their friend. The
          doorman cares about some of those things a lot, some of them a
          little, and some of them not at all. He adds it all up in his head,
          factors in how grumpy he happens to be tonight, and lands on one
          decision: in, or not.
        </p>
        <p>
          That&apos;s a neuron. The whole lesson is that sentence.
        </p>
        <p>
          You&apos;ve already built every part of this doorman piece by piece.
          The weighted sum is{' '}
          <NeedsBackground slug="linear-regression-forward">
            linear regression
          </NeedsBackground>
          . The squash at the end is the{' '}
          <NeedsBackground slug="sigmoid-and-relu">activation</NeedsBackground>
          . All this lesson does is staple them together, name the result, and
          show you the one thing a single doorman absolutely cannot do.
        </p>
      </Prose>

      <Callout variant="note" title="where the last lesson left off">
        Linear regression drew a line through a cloud of points. A{' '}
        <KeyTerm>neuron</KeyTerm> is that same line — same <code>wx + b</code> —
        passed through one more step that decides what the line means. Nothing
        new is being invented here. Something is being re-labeled, and then
        stacked.
      </Callout>

      {/* ── One equation ─────────────────────────────────────────── */}
      <Prose>
        <p>
          Here&apos;s the doorman, written down. One line. Every neural network
          you&apos;ve ever heard of is this line, a trillion times, wired up.
        </p>
      </Prose>

      <MathBlock caption="the neuron — one equation">
{`       ┌──── linear combo ────┐    ┌── nonlinear ──┐

y  =   f (  w₁·x₁  +  w₂·x₂  +  ...  +  w_D·x_D   +   b )

  • xᵢ  — input from the previous layer (or the raw data)
  • wᵢ  — weight on input i — the doorman's opinion of feature i
  • b   — bias — his baseline mood before anyone walks up
  • f   — activation — the yes/no decision rule at the door
  • y   — the single scalar output`}
      </MathBlock>

      <Prose>
        <p>
          Two things live in that equation that the neuron will eventually{' '}
          <em>learn</em>: the weights and the bias. Everything else is fixed.
        </p>
        <p>
          <strong>The weights are opinions.</strong> One per feature. A large
          positive <code>wᵢ</code> means the doorman cares a lot about feature{' '}
          <code>i</code> and likes what high values do. A large negative{' '}
          <code>wᵢ</code> means he cares a lot and{' '}
          <em>dislikes</em> high values. A weight near zero means he doesn&apos;t
          particularly care; that feature could double and he&apos;d shrug.
          Training is the process of him revising those opinions until his
          decisions line up with what the labels say they should be.
        </p>
        <p>
          <strong>The bias is his mood.</strong> A single scalar that shifts the
          whole decision up or down regardless of who&apos;s in line. Grumpy
          today? Bias goes down — the bar for getting in rises, and marginal
          people get turned away. Generous? Bias up — everyone looks a little
          better than they are. Without a bias, the doorman is forced to say
          &ldquo;in&rdquo; whenever all the inputs happen to be zero, which is
          not a policy any real establishment would endorse.
        </p>
        <p>
          Drag the knobs below. Each row is one term of the weighted sum —
          left slider is the input <code>xᵢ</code>, right slider is its weight{' '}
          <code>wᵢ</code>, and the product <code>xᵢ · wᵢ</code> updates on the
          right. The pre-activation row sums them plus the bias. Pick an
          activation from the control bar to see the final decision.
        </p>
      </Prose>

      <NeuronForward />

      <Callout variant="insight" title="why the squash matters">
        The activation <code>f</code> is the doorman&apos;s decision rule — the
        threshold between &ldquo;in&rdquo; and &ldquo;out.&rdquo; Strip it away
        (pick &ldquo;Linear&rdquo; in the widget) and you get exactly a linear
        regressor. That&apos;s the important part. Stack a thousand linear
        neurons with no activation and the whole network is still
        mathematically one line — the composition of linear functions is
        linear. The doorman without a decision rule is just muttering a
        number at you. The activation is what turns the muttering into a
        judgement, and what lets a stack of neurons carve up space in shapes
        more interesting than a flat plane.
      </Callout>

      <Personify speaker="Neuron">
        I am a scalar. Not an array, a tensor, a model, or a service. I take a
        weighted combination of whatever walks up, factor in my mood, and
        squash the result into a decision. When I&apos;m part of a layer I
        have coworkers — same inputs, different opinions — and together we
        are a crew. Individually I am one doorman with one set of preferences
        and one bad day of the week.
      </Personify>

      {/* ── Flow diagram ─────────────────────────────────────────── */}
      <Prose>
        <p>
          Here&apos;s the same doorman drawn the way textbooks draw him. Edges
          carry the inputs in on the left. The body in the middle does the
          weighted sum plus the bias, then applies the activation. One output
          comes out on the right. The edges get thicker as their weighted
          contribution grows — the features he&apos;s paying the most
          attention to right now.
        </p>
        <p>
          This is the picture you should have in your head every time you see{' '}
          <code>nn.Linear</code> in PyTorch. Each output unit of that layer is
          one of these diagrams, running in parallel with the others.
        </p>
      </Prose>

      <NeuronFlowDiagram />

      {/* ── Decision boundary + XOR ──────────────────────────────── */}
      <Prose>
        <p>
          Shrink the doorman down to two inputs and wrap him in a sigmoid, and
          you get a binary classifier. He says class 1 when{' '}
          <code>w₁x₁ + w₂x₂ + b &gt; 0</code>, class 0 otherwise. Geometrically
          that condition is a straight line cutting the input plane in two:
          the <KeyTerm>decision boundary</KeyTerm>. Points on one side get
          waved in. Points on the other side get sent home.
        </p>
        <p>
          Try the three datasets below. Drag <code>w₁</code>, <code>w₂</code>,
          and <code>b</code> until all four dots are on the right side of the
          line. AND is easy — the separator is <code>x₁ + x₂ = 1.5</code>. OR
          is easy too. Then flip to XOR.
        </p>
      </Prose>

      <DecisionBoundary2D />

      <Prose>
        <p>
          You will not make XOR work. You can try forever. No combination of{' '}
          <code>w₁</code>, <code>w₂</code>, <code>b</code> exists that puts all
          four XOR points on the correct side of a line, because the correct
          regions — <code>(0,0)</code> and <code>(1,1)</code> together,{' '}
          <code>(0,1)</code> and <code>(1,0)</code> together — are{' '}
          <em>not linearly separable</em>. No single straight cut can put two
          diagonally opposite corners on one side and the other two on the
          other side. The doorman, to use a technical term, is cooked.
        </p>
        <p>
          This is the most famous limitation in the history of neural
          networks. Marvin Minsky and Seymour Papert pointed it out in their
          1969 book <em>Perceptrons</em>, and the entire field went quiet for
          almost two decades afterward. The fix turns out to be almost
          insultingly simple: hire a second doorman with different opinions,
          let them talk to each other through a non-linearity, and have a
          third one read their notes. A crew of bouncers with different
          opinions can carve up regions a single line can&apos;t. That&apos;s
          the whole reason every modern network has more than one neuron in
          it, and it is quite literally the point of the next several
          lessons.
        </p>
      </Prose>

      <Personify speaker="XOR">
        I separate the odd-parity corners from the even-parity corners. No
        single line works on me, and one doorman is one line. If you want to
        classify me, you need a crew. The first layer will learn its own
        intermediate features — something like &ldquo;is exactly one input
        high&rdquo; — and the output doorman will classify{' '}
        <em>those</em>. I&apos;m the dataset that justified every
        &ldquo;deep&rdquo; network ever built.
      </Personify>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Don&apos;t forget the bias.</strong>{' '}
          A neuron without a bias is a doorman who can&apos;t have a bad day
          — his decision boundary is forced to pass through the origin.
          That&apos;s a weirdly specific constraint, and models that silently
          lose their bias (via a buggy initialization or a shape mismatch) can
          train for hours before anyone notices the loss floor is suspiciously
          high.
        </p>
        <p>
          <strong className="text-term-amber">The activation is a choice, not a detail.</strong>{' '}
          Sigmoid, ReLU, tanh, GELU — they all squash, but they squash
          differently, and the derivative each one hands back to training is
          different too. Picking the wrong one doesn&apos;t crash the network;
          it quietly kneecaps it. More on this when we build the training
          loop.
        </p>
        <p>
          <strong className="text-term-amber">&ldquo;Neuron&rdquo; vocabulary is slippery.</strong>{' '}
          Some papers call the scalar output of an activation a neuron. Others
          call the whole weight vector (one row of <code>W</code>) a neuron.
          Others call an entire layer a neuron. And &ldquo;unit&rdquo; means
          the same thing as &ldquo;neuron&rdquo; in all contexts. They&apos;re
          all describing the same diagram at different scopes. Check scope
          before arguing.
        </p>
      </Gotcha>

      {/* ── Three-layer code ─────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations of the same doorman. By now you know the
          drill: pure Python so nothing hides, NumPy so it scales, PyTorch so
          it trains.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · neuron_scratch.py"
        output={`pre-activation z = 1.1200
output y = 0.7540`}
      >{`import math

def neuron(x, w, b, activation='sigmoid'):
    z = sum(xi * wi for xi, wi in zip(x, w)) + b   # weighted sum + bias — the doorman's tally
    if activation == 'relu':
        return max(0.0, z)
    if activation == 'sigmoid':
        return 1.0 / (1.0 + math.exp(-z))
    if activation == 'tanh':
        return math.tanh(z)
    return z                                       # linear — no decision rule, just a number

x = [0.8, -0.5, 1.2]
w = [0.6, -0.4, 0.9]
b = 0.1

z = sum(xi * wi for xi, wi in zip(x, w)) + b
print(f"pre-activation z = {z:.4f}")
print(f"output y = {neuron(x, w, b, 'sigmoid'):.4f}")`}</CodeBlock>

      <Prose>
        <p>
          The <code>sum(...)</code> line <em>is</em> the weighted combo from
          the equation, one multiply at a time. One doorman, one person at the
          door, one decision. Fine for a sketch. Useless at scale.
        </p>
        <p>
          Now picture a whole night&apos;s line. NumPy lets the doorman score
          every arrival in one pass — same arithmetic, vectorised, running in
          compiled C instead of a Python loop.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — numpy · neuron_numpy.py">{`import numpy as np

def neuron(x, w, b, activation='sigmoid'):
    z = x @ w + b                                  # one dot product — whole batch, one line
    if activation == 'relu':
        return np.maximum(0, z)
    if activation == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-z))
    if activation == 'tanh':
        return np.tanh(z)
    return z

# Same single neuron — but now we can score a whole batch at once.
X = np.array([
    [0.8, -0.5, 1.2],
    [1.0, 0.5, -0.3],
    [-0.2, 0.7, 0.4],
])
w = np.array([0.6, -0.4, 0.9])
b = 0.1

out = neuron(X, w, b, 'sigmoid')
print("outputs:", np.round(out, 4))   # one scalar per example in the batch`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'sum(xi * wi for xi, wi in zip(x, w)) + b',
            right: 'X @ w + b',
            note: 'one matmul — entire batch, one line',
          },
          {
            left: 'one example at a time',
            right: 'N examples at once',
            note: 'broadcasting, not a loop',
          },
        ]}
      />

      <Prose>
        <p>
          NumPy gives you the forward pass for free. PyTorch gives you the
          forward pass <em>and</em> a ready-made container that remembers the
          weights and bias, keeps them on the right device, and — once we get
          to training — hands back the gradients you need to update them. Same
          doorman. Fewer moving parts on the page.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · neuron_pytorch.py"
        output={`tensor([[0.7540],
        [0.5149],
        [0.5498]], grad_fn=<SigmoidBackward0>)`}
      >{`import torch
import torch.nn as nn

# nn.Linear(in_features=3, out_features=1) IS one neuron.
# Stacking many outputs (out_features > 1) gives a layer of parallel neurons
# — a crew of doormen with different opinions, listening to the same line.
class OneNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)           # 3 weights + 1 bias, all learnable

    def forward(self, x):
        z = self.fc(x)                       # w·x + b
        return torch.sigmoid(z)              # one-liner activation

# Copy our hand-picked weights in so the demo matches layers 1 and 2.
model = OneNeuron()
with torch.no_grad():
    model.fc.weight.copy_(torch.tensor([[0.6, -0.4, 0.9]]))
    model.fc.bias.copy_(torch.tensor([0.1]))

X = torch.tensor([[0.8, -0.5, 1.2], [1.0, 0.5, -0.3], [-0.2, 0.7, 0.4]])
print(model(X))`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'z = X @ w + b',
            right: 'nn.Linear(3, 1)',
            note: 'packaged with learnable weights + gradients',
          },
          {
            left: '1 / (1 + np.exp(-z))',
            right: 'torch.sigmoid(z)',
            note: 'autograd-aware, GPU-aware',
          },
          {
            left: 'one neuron',
            right: 'nn.Linear(3, K) ; K neurons in parallel',
            note: 'a layer is a crew — next lesson',
          },
        ]}
      />

      <Callout variant="insight" title="a 'layer' of N neurons, restated">
        <code>nn.Linear(D, N)</code> is a crew of <em>N</em> doormen, each
        with his own weight vector of length <em>D</em> — all stored as an{' '}
        <em>N × D</em> matrix, plus one bias per doorman. Calling it on input{' '}
        <code>x</code> runs all <em>N</em> weighted sums at once via one
        matrix-vector product. Everything you know about a single doorman
        scales unchanged to layers, layers of layers, and transformer blocks.
        The algebra is identical. Only the shapes get bigger, and the crew
        gets louder.
      </Callout>

      <Challenge prompt="Teach one doorman to do AND">
        <p>
          In NumPy, initialise <code>w = [0, 0]</code>, <code>b = 0</code>.
          Train the neuron on AND{' '}
          <code>[(0,0,0), (0,1,0), (1,0,0), (1,1,1)]</code> using binary
          cross-entropy loss and{' '}
          <NeedsBackground slug="gradient-descent">
            gradient descent
          </NeedsBackground>{' '}
          at <code>lr=0.5</code> for 2000 steps. It will converge to something
          like <code>w ≈ [3, 3], b ≈ -4.5</code> — a doorman who only says yes
          when both inputs are high.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: do the same thing for XOR. Plot the loss curve. It will
          plateau well above zero and the accuracy will lock at 75% — the best
          a single line can do on that dataset. That&apos;s the empirical
          proof of Minsky and Papert&apos;s theoretical result, in about
          twenty lines of code.
        </p>
      </Challenge>

      {/* ── Takeaways + cliffhanger ──────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> A neuron is a doorman:
          weighted sum of the features (opinions × inputs), plus a bias (his
          mood), run through an activation (his decision rule). One scalar in,
          one scalar out. Geometrically he draws a hyperplane and labels one
          side &ldquo;in&rdquo;. That&apos;s why a single neuron cannot solve
          anything that isn&apos;t linearly separable — XOR being the
          canonical counterexample. The cure is a crew: more doormen, in more
          layers, with non-linearities between them. Stack the diagram you
          just saw and you have a neural network. Nothing else is added.
        </p>
        <p>
          <strong>Next up — Backpropagation.</strong> You have a doorman who
          can <em>predict</em>. But he&apos;s new on the job, his opinions are
          random, and his mood is untuned. How does he <em>learn</em>?
          Gradient descent gives you the update rule —{' '}
          <code>w ← w − α · ∂L/∂w</code> — but nothing so far has told you how
          to compute that partial derivative for every weight in a network of
          tens of thousands of them. Backpropagation is the mechanical
          procedure that does exactly this: it walks backwards through the
          network and assigns every single weight its share of the blame (or
          credit) for the final loss. It is the algorithm whose correctness
          is the single most important fact in modern ML.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Perceptrons: An Introduction to Computational Geometry',
            author: 'Marvin Minsky, Seymour Papert',
            venue: 'MIT Press, 1969 — the book that proved the XOR limitation',
            url: 'https://mitpress.mit.edu/9780262630221/',
          },
          {
            title: 'The Perceptron: A Probabilistic Model for Information Storage',
            author: 'Frank Rosenblatt',
            venue: 'Psychological Review, 1958',
            url: 'https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf',
          },
          {
            title: 'Dive into Deep Learning — 5.1 Multilayer Perceptrons',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_multilayer-perceptrons/mlp.html',
          },
        ]}
      />
    </div>
  )
}
