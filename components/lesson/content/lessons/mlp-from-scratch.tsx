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
import MLPDecisionBoundary from '../widgets/MLPDecisionBoundary'
import FeatureWarp from '../widgets/FeatureWarp'
import MLPArchitecture from '../widgets/MLPArchitecture'

// Anchor: the kit car build. You've spent six lessons buying parts —
// a neuron, activations, a loss, backprop, multi-layer backprop — and
// today you bolt them together and try to turn the engine over. The
// anchor lands at three load-bearing moments: the opening ("you have
// the kit, today you build it"), the full-loop assembly ("forward →
// loss → backward → update, four strokes per revolution"), and the
// first successful training run ("the car starts"). Cliffhanger points
// to weight-initialization ("it won't turn over if you start the
// weights wrong").

export default function MLPFromScratchLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="mlp-from-scratch" />

      <Prose>
        <p>
          You&apos;ve been buying parts. A{' '}
          <NeedsBackground slug="single-neuron">neuron</NeedsBackground> —
          weighted sum plus activation, one dial at a time. A{' '}
          <NeedsBackground slug="cross-entropy-loss">loss</NeedsBackground>{' '}
          that squashes a whole prediction down to one number that says{' '}
          <em>how wrong</em>. A{' '}
          <NeedsBackground slug="softmax">softmax output</NeedsBackground>{' '}
          that turns raw scores into probabilities that actually sum to one. A{' '}
          <NeedsBackground slug="backpropagation">backward pass</NeedsBackground>{' '}
          that reads the chain rule in reverse, plus the{' '}
          <NeedsBackground slug="multi-layer-backpropagation">
            chain rule across layers
          </NeedsBackground>{' '}
          version that doesn&apos;t flinch at arbitrary depth. And an{' '}
          <NeedsBackground slug="gradient-descent">
            update rule
          </NeedsBackground>{' '}
          that nudges the dials the tiny opposite-of-the-slope amount.
        </p>
        <p>
          Six lessons of parts in boxes. Today you bolt them together into a
          kit car. Forward pass is the fuel intake — data goes in, the engine
          squeezes it through the cylinders, a prediction comes out. Loss is
          combustion — the measurement of how wrong. Backward pass is the
          exhaust, pushed back through the system to tune the next firing.
          Four strokes per revolution. One epoch per drive around the block.
          This is the lesson where the kit becomes a car.
        </p>
        <p>
          By the end of this page you&apos;ll have (a) a working 2-input
          classifier training live in your browser, (b) a visceral feel for
          why hidden layers are coordinate systems the network invents on the
          fly, and (c) forty lines of NumPy that generalize to any width and
          depth you hand them. No new math. Just the assembly.
        </p>
      </Prose>

      {/* ── Live training widget ─────────────────────────────────── */}
      <Prose>
        <p>
          Don&apos;t read anything else yet. Hit <em>train</em> below on the{' '}
          <em>moons</em> dataset with 8 hidden neurons. Within a second or
          two the decision boundary morphs from a random scribble into a
          clean, S-curved wall that separates the two classes. That&apos;s
          the kit car starting. It sputters, it catches, it runs. Pure
          JavaScript gradient descent, no library doing the heavy lifting —
          the exact algorithm you just derived, four strokes per cycle,
          running in real time on the page.
        </p>
      </Prose>

      <MLPDecisionBoundary />

      <Prose>
        <p>
          Play with the knobs. Switch to <em>circles</em> — classic
          concentric rings — and try to train with <code>H = 1</code>. It
          cannot work: one hidden neuron is one line, and no single line
          separates a ring from its middle. Try <code>H = 2</code>; barely
          better. By <code>H = 4</code> it finds a closed region. By{' '}
          <code>H = 8</code> it&apos;s clean. By <code>H = 32</code>{' '}
          it&apos;s overkill — faster to converge but no better at the end.
          That progression is what &ldquo;make the network wider&rdquo;
          means in one plot: capacity scales with width until the problem
          becomes solvable, then keeps scaling for nothing but speed.
        </p>
      </Prose>

      <Callout variant="insight" title="the universal approximation theorem, loosely">
        An MLP with <em>one</em> hidden layer, enough neurons, and a
        reasonable activation can approximate any continuous function on a
        compact set to any desired accuracy. So why bother with depth?
        Because &ldquo;enough neurons&rdquo; grows exponentially with the
        complexity of the function — a single wide layer is theoretically
        universal and practically useless. Deep networks hit the same
        targets with exponentially fewer parameters. Depth is efficient;
        width is expensive. Nobody builds a race car out of one enormous
        piston.
      </Callout>

      {/* ── Hidden layers as warping ────────────────────────────── */}
      <Prose>
        <p>
          Zoom in on what the hidden layer actually <em>does</em>. Take
          XOR — the four corners of a unit square, labelled in a pinwheel
          — which a single neuron cannot solve because no straight line
          separates the classes. One hidden layer with two ReLU neurons is
          enough. But <em>why</em>? The hidden layer isn&apos;t
          classifying. It&apos;s remapping. It bends the plane until a
          straight line finally works.
        </p>
      </Prose>

      <FeatureWarp />

      <Prose>
        <p>
          The default weights tilt and fold the XOR corners so that the two
          labelled-1 points end up at <code>(0, 1)</code> and{' '}
          <code>(1, 0)</code> in hidden space — now linearly separable.
          Drag any weight; watch the hidden-space points drift; watch the
          green dashed line lose its grip. This is the clearest statement
          of what &ldquo;deep learning&rdquo; actually is. The output
          layer is a linear classifier, same as lesson one. What changed
          is that it&apos;s classifying a <em>learned</em> representation
          of the data, not the raw data. The hidden layer is bodywork —
          reshape the frame, and suddenly a plain straight axle fits.
        </p>
      </Prose>

      <Personify speaker="Hidden layer">
        I don&apos;t classify anything. I transform. I take inputs and move
        them to new positions — rotating, shearing, folding, clipping — so
        the final output layer can draw a single line and call it a day.
        Every fancy architecture — convnets, transformers — is a specific
        flavour of me with extra chrome. I am learned feature extraction.
      </Personify>

      {/* ── Architecture widget ─────────────────────────────────── */}
      <Prose>
        <p>
          One last thing before the code. Drag the architecture sliders and
          watch the parameter count move. A few intuition checks:
        </p>
        <ul>
          <li>
            Depth 1, width 64, input 10, output 1:{' '}
            <code>10·64 + 64 + 64·1 + 1 ≈ 769</code> parameters. A go-kart.
          </li>
          <li>
            Depth 3, width 128, input 256, output 10: jumps to{' '}
            <code>~116K</code> — still small. A hatchback.
          </li>
          <li>
            Depth 8, width 512, input 256: millions. This is where
            &ldquo;big model&rdquo; starts. You are no longer in the kit
            car section of the catalog.
          </li>
        </ul>
      </Prose>

      <MLPArchitecture />

      <Callout variant="note" title="where the parameters live">
        Middle layers dominate for any non-trivial depth. A single
        hidden-to-hidden connection contributes <code>width²</code>{' '}
        parameters — for <code>width = 1024</code> that&apos;s a million
        per layer. Most of a transformer&apos;s parameters live in
        exactly this kind of &ldquo;feed-forward&rdquo; block. The engine
        block, in other words, is where the weight sits.
      </Callout>

      {/* ── Three-layer code ─────────────────────────────────────── */}
      <Prose>
        <p>
          Time to bolt the kit together. This is the most substantial code
          block in the lesson — read it carefully. Every line is either
          setting up the chassis, running the forward pass (intake →
          compression → ignition → exhaust prediction), or running the
          backward pass (exhaust fed back to tune the next firing). Six
          lessons of parts, all of them showing up here at once.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — numpy · mlp.py (full, general-depth)">{`import numpy as np

class MLP:
    """A multi-layer perceptron: inputs → [hidden-ReLU] × L → output.
       Binary output (sigmoid head + BCE loss) for simplicity; easy to swap."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.Ws, self.bs = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            scale = np.sqrt(2 / fan_in)                  # He init
            self.Ws.append(rng.normal(0, scale, size=(sizes[i + 1], sizes[i])))
            self.bs.append(np.zeros(sizes[i + 1]))

    def forward(self, X):
        acts = [X]
        pres = [X]
        for l, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = acts[-1] @ W.T + b
            pres.append(z)
            if l < len(self.Ws) - 1:
                acts.append(np.maximum(0, z))            # ReLU hidden
            else:
                acts.append(1 / (1 + np.exp(-z)))         # sigmoid output
        return acts, pres

    def train_step(self, X, y, lr=0.1):
        acts, pres = self.forward(X)
        L = len(self.Ws)
        N = len(X)

        # Loss
        yhat = acts[-1].ravel()
        loss = -np.mean(y * np.log(np.clip(yhat, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - yhat, 1e-9, 1)))

        # Backward — the recurrence, written in matrix form.
        # δ at the output (sigmoid + BCE collapses to p - y)
        delta = (yhat - y).reshape(-1, 1) / N           # (N, out_dim)
        grads_W = [None] * L
        grads_b = [None] * L
        for l in range(L - 1, -1, -1):
            grads_W[l] = delta.T @ acts[l]               # (out, in)
            grads_b[l] = delta.sum(axis=0)
            if l > 0:
                # Push δ back through the weights, then mask by ReLU'
                delta = (delta @ self.Ws[l]) * (pres[l] > 0)

        # Update
        for l in range(L):
            self.Ws[l] -= lr * grads_W[l]
            self.bs[l] -= lr * grads_b[l]
        return loss

# Smoke test on XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)

mlp = MLP([2, 8, 1])
for step in range(3000):
    loss = mlp.train_step(X, y, lr=0.5)
    if step % 500 == 0:
        print(f"step {step}: loss={loss:.4f}")

probs = mlp.forward(X)[0][-1].ravel()
print("predictions:", np.round(probs, 2))   # -> close to [0, 1, 1, 0]`}</CodeBlock>

      <Prose>
        <p>
          Read it once. Read it again. And listen — this is the moment the
          kit car starts. Four strokes, every time, in the order that
          matters. Forward pulls data through the layers. The loss line
          measures how wrong. The backward loop walks the chain rule in
          reverse, one matmul per layer, pushing blame back toward the
          weights that caused it. The update loop nudges every dial a
          little opposite to its blame. Four strokes per revolution, three
          thousand revolutions, and XOR is solved. Rumelhart, Hinton, and
          Williams published this in 1986 and it fits in forty lines of
          NumPy. The compute has changed. The algorithm hasn&apos;t.
        </p>
      </Prose>

      <Bridge
        label="the two loops that matter"
        rows={[
          {
            left: 'forward: for l: z = a @ W.T + b ; a = f(z)',
            right: 'apply each layer in order',
            note: 'collect pre-activations and post-activations for backward',
          },
          {
            left: 'backward: for l in reverse: δ = (δ @ W) ⊙ f\'(z)',
            right: 'chain rule, one matmul per layer',
            note: 'param grads are δᵀ @ a — outer product',
          },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · mlp_pytorch.py"
        output={`step 0: loss=0.7148
step 500: loss=0.0086
predictions: [0.01, 0.99, 0.99, 0.01]`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F

# The entire MLP class from the NumPy layer — now it's four lines.
class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        return torch.sigmoid(self.layers[-1](x))

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

model = MLP([2, 8, 1])
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for step in range(3000):
    optimizer.zero_grad()
    yhat = model(X)
    loss = F.binary_cross_entropy(yhat, y)
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"step {step}: loss={loss.item():.4f}")

print("predictions:", torch.round(model(X), decimals=2).ravel())`}</CodeBlock>

      <Bridge
        label="numpy → pytorch (what disappeared)"
        rows={[
          {
            left: 'manual He-init in __init__',
            right: 'nn.Linear defaults to Kaiming uniform',
            note: 'same idea — PyTorch picks a reasonable default',
          },
          {
            left: 'forward: hand-coded activation branching',
            right: 'for layer in self.layers[:-1]: x = F.relu(layer(x))',
            note: 'cleaner loop, identical semantics',
          },
          {
            left: 'backward: 20 lines of matmuls + deltas',
            right: 'loss.backward() — one line',
            note: 'autograd runs the recurrence for you',
          },
        ]}
      />

      <Gotcha>
        <p>
          <strong className="text-term-amber">Shape bugs are the tax.</strong>{' '}
          More training runs die on a transposed matrix than on any real
          modeling problem. If <code className="text-dark-text-primary">X</code>{' '}
          is <code className="text-dark-text-primary">(N, D)</code> and{' '}
          <code className="text-dark-text-primary">W</code> is{' '}
          <code className="text-dark-text-primary">(out, in)</code>, then{' '}
          <code className="text-dark-text-primary">X @ W.T + b</code>{' '}
          gives you <code className="text-dark-text-primary">(N, out)</code>{' '}
          — flip a transpose and NumPy will happily broadcast nonsense.
          Print shapes at every layer while you&apos;re debugging. It&apos;s
          not elegant, but neither is pulling a head gasket on the driveway.
        </p>
        <p>
          <strong className="text-term-amber">
            Output activation matches the loss.
          </strong>{' '}
          Binary classification → sigmoid + BCE. Multi-class → softmax +
          cross-entropy. Regression → no output activation + MSE. Mixing
          them (sigmoid + MSE, softmax + BCE) sometimes &ldquo;trains&rdquo;
          but gives bad gradients. The clean{' '}
          <code className="text-dark-text-primary">p − y</code> structure in
          backprop only falls out when the pairing is right.
        </p>
        <p>
          <strong className="text-term-amber">
            MLP output dim = number of classes, not <code>1</code>.
          </strong>{' '}
          For 10-way MNIST classification,{' '}
          <code className="text-dark-text-primary">sizes = [784, 128, 10]</code>,
          not <code className="text-dark-text-primary">[784, 128, 1]</code>.
          One output per class; softmax combines them into a distribution.
        </p>
        <p>
          <strong className="text-term-amber">Biases start at zero.</strong>{' '}
          Not random. If you randomly initialize biases the way you do
          weights, each neuron begins with a baked-in preference toward one
          class before it&apos;s seen a single input — a carburetor that
          runs rich from the factory. Zero lets the gradient pick the
          offset the data actually wants.
        </p>
        <p>
          <strong className="text-term-amber">Learning-rate scale is the
          throttle.</strong>{' '}
          The <code className="text-dark-text-primary">lr=0.5</code> on XOR
          works because XOR is a clean bowl. On real datasets{' '}
          <code className="text-dark-text-primary">0.5</code> will blow the
          cylinder head off. Start at <code className="text-dark-text-primary">1e-3</code>{' '}
          and climb until loss starts diverging, then back off by half.
          Same convergence math as the gradient-descent lesson —
          just with sharper teeth now that the loss surface has curvature.
        </p>
      </Gotcha>

      <Challenge prompt="Beat the moons dataset with as few parameters as possible">
        <p>
          Using the NumPy MLP above, train on{' '}
          <code>sklearn.datasets.make_moons(n_samples=500, noise=0.15)</code>.
          Your goal is <code>&gt; 97%</code> accuracy on a held-out test set
          with the smallest total parameter count. Try{' '}
          <code>[2, 4, 1]</code> (21 params). Then <code>[2, 8, 1]</code>{' '}
          (41). Then <code>[2, 4, 4, 1]</code> (33 params). Which wins?
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: plot the decision boundary. The deepest-narrowest network
          carves the most &ldquo;creative&rdquo; boundary; the wide-shallow
          network draws something more predictable. This is the
          depth-vs-width tradeoff showing up in a toy problem — and the
          reason every architecture diagram in the last decade has been a
          debate between tall and fat.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> An MLP is a stack of{' '}
          <code>linear → nonlinearity → linear → …</code> with one output
          head. Every hidden layer is a learned coordinate system the output
          head classifies against. Implementation is two loops: a forward
          loop that caches intermediates, a backward loop that applies the
          δ recurrence. Forty lines of NumPy. Four lines of PyTorch. The
          algorithm hasn&apos;t changed since 1986; what changed is the
          compute that runs it. You now own a kit car. It runs. The parts
          lesson is over.
        </p>
        <p>
          <strong>Next up — Weight Initialization.</strong> Everything here
          quietly assumed a sensible starting point for the weights. Look
          at the <code>scale = np.sqrt(2 / fan_in)</code> line in{' '}
          <code>__init__</code> — that&apos;s a load-bearing detail
          masquerading as a throwaway. It trains. But only if you start the
          weights right. Start them wrong and the car won&apos;t turn over
          no matter how many times you twist the key — activations
          collapse to zero, gradients vanish, the loss stays flat forever.
          Xavier and He init are the answers, and they&apos;re the last
          lesson before you leave this section.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Dive into Deep Learning — 5.1 Multilayer Perceptrons',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_multilayer-perceptrons/mlp.html',
          },
          {
            title: 'Approximation by superpositions of a sigmoidal function',
            author: 'George Cybenko',
            venue: 'Mathematics of Control, Signals, and Systems, 1989 — the universal approximation theorem',
            url: 'https://link.springer.com/article/10.1007/BF02551274',
          },
          {
            title: 'Neural Networks and Deep Learning',
            author: 'Michael Nielsen',
            venue: 'free online book — chapter 1 covers the MLP in depth',
            url: 'http://neuralnetworksanddeeplearning.com/chap1.html',
          },
        ]}
      />
    </div>
  )
}
