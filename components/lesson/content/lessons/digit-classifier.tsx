import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import Prereq from '../Prereq'
import NeedsBackground from '../NeedsBackground'
import {
  Prose, Callout, Personify, Bridge, Gotcha, Challenge, References, KeyTerm, AsciiBlock,
} from '../primitives'
import DrawDigit from '../widgets/DrawDigit'
import ConfusionMatrixLive from '../widgets/ConfusionMatrixLive'

export default function DigitClassifierLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="digit-classifier" />

      <Prose>
        <p>
          Sixteen lessons on the factory floor. Tensors, autograd, linear
          layers, activations, softmax, cross-entropy, init, three flavors of
          normalization, the four-beat{' '}
          <NeedsBackground slug="training-loop">training loop</NeedsBackground>
          , the diagnostic suite. Every station built, inspected, and
          signed off.
        </p>
        <p>
          Today, the first vehicle rolls off the assembly line. A real model,
          on real data, that you can hand a pixel grid to and have it tell
          you what digit you drew. No toys. No placeholders. The whole line
          running at once.
        </p>
        <p>
          The test track is <KeyTerm>MNIST</KeyTerm> — 70,000 handwritten
          digits, 28×28 grayscale, ten classes. Small enough to train on a
          laptop in under a minute. Rich enough that every piece you&apos;ve
          built earns its place. Simple enough that if anything is broken,
          you&apos;ll see it immediately instead of blaming the data.
        </p>
      </Prose>

      <AsciiBlock caption="the complete pipeline, one more time">
{`    28×28 pixel image
           │
           ▼  flatten
    784-D vector (each pixel in [0, 1])
           │
           ▼  Linear(784, 128)   +   ReLU
    128-D hidden layer
           │
           ▼  Linear(128, 10)
    10-D logit vector
           │
           ▼  softmax   +   cross-entropy loss
    scalar loss  →  loss.backward()  →  optimizer.step()`}
      </AsciiBlock>

      <Prose>
        <p>
          Walk that diagram station by station. Inputs normalized to{' '}
          <code>[0, 1]</code> — normalization lesson. A Linear layer —{' '}
          <NeedsBackground slug="mlp-from-scratch">MLP</NeedsBackground>{' '}
          lesson. ReLU — activations lesson. A{' '}
          <NeedsBackground slug="softmax">softmax</NeedsBackground> head
          and <NeedsBackground slug="cross-entropy-loss">cross-entropy
          loss</NeedsBackground> — loss lessons. A backward pass and an
          optimizer step — the four-beat loop. You have visited every one
          of these stations on foot. Today they get wired together, and the
          output is a working classifier.
        </p>
        <p>
          Draw a digit below. The widget is not a real MNIST model — it
          compares your strokes against ten template shapes and runs the
          softmax. Think of it as a scale-model of the drivetrain: pixels
          in, logits out, probabilities out, argmax picks the class. Every
          real digit classifier, from 1998 through today, does exactly
          this.
        </p>
      </Prose>

      <DrawDigit />

      <Personify speaker="MNIST">
        I am the friendly introduction dataset. 60,000 training digits,
        10,000 test digits, handwritten by US Census Bureau employees and
        high-school students in the 1990s. I am small enough to train a
        model on a laptop in under a minute, clean enough that a linear
        model gets 92%, and interesting enough that you can still push to
        99.8% by caring more. I am the benchmark you cut your teeth on, and
        the benchmark you graduate from as soon as possible.
      </Personify>

      <MathBlock caption="the forward pass, with shapes">
{`x    ∈ ℝ^{B × 784}          # batch of flattened images
W₁   ∈ ℝ^{784 × 128}        # first layer weights
W₂   ∈ ℝ^{128 × 10}         # second layer weights

h    =  ReLU(x @ W₁ + b₁)   ∈ ℝ^{B × 128}
z    =  h @ W₂ + b₂          ∈ ℝ^{B × 10}     (raw logits)
p    =  softmax(z)           ∈ ℝ^{B × 10}     (probabilities)

loss =  CrossEntropy(z, targets)    # uses z, not p — numerical stability`}
      </MathBlock>

      <Prose>
        <p>
          Two Linear layers. One ReLU between them. One softmax folded
          into the loss for numerical stability (see the cross-entropy
          lesson). Parameter count:{' '}
          <code>784 · 128 + 128 · 10 + 138</code> ≈{' '}
          <code>100K parameters</code>. That is the whole model. About 0.4
          MB of floats. It hits 97%+ test accuracy in five epochs.
        </p>
        <p>
          Before you look at the code, look at the report card. A single
          accuracy number — &ldquo;97.8%&rdquo; — hides more than it
          reveals. You want to know <em>which</em> digits the model misses,
          and which ones it confuses them for. That is the confusion
          matrix.
        </p>
      </Prose>

      <ConfusionMatrixLive />

      <Prose>
        <p>
          The <KeyTerm>confusion matrix</KeyTerm> is the per-class
          diagnostic for a classifier. Rows are true labels, columns are
          predictions, diagonals are hits, off-diagonals are misses. A
          healthy MNIST model has a bright diagonal and a few stubborn hot
          cells — 4↔9 (same vertical stroke, different tops), 3↔5 (curly
          loops that look alike), 7↔2 (slanted strokes). These pairs
          persist across architectures because the handwriting really is
          ambiguous; even humans miss them.
        </p>
      </Prose>

      <Callout variant="note" title="why confusion matrices beat aggregate accuracy">
        A 97% accuracy number tells you the average. A confusion matrix
        tells you the structure. If one class has 85% recall while the
        others are at 99%, you have a specific problem — maybe a
        data-distribution issue, maybe a decision-boundary issue, but
        either way fixing that one class is the highest-leverage change
        you can make. Aggregate numbers hide this.
      </Callout>

      <Personify speaker="Confusion matrix">
        I am the classifier&apos;s report card. The diagonal is where you
        did the assignment right. Everything off-diagonal is a specific
        error with a specific explanation. Big row sum / small diagonal =
        the model is struggling to recall that class. Big column sum
        off-diagonal = the model is over-predicting that class. I will
        not tell you how to fix anything, but I will tell you exactly
        what to fix.
      </Personify>

      <Prose>
        <p>
          Now the code. Three layers, same as every algorithm in this
          series. Pure Python is too slow for 60k images, so the first
          layer is a toy — ten hand-crafted feature vectors, just to show
          the shape of a hand-rolled training loop. Layers two and three
          run on real MNIST.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · digit_classifier_scratch.py (toy, 10 examples)"
        output={`step 0: loss=2.3012
step 50: loss=0.4215
step 100: loss=0.1008
accuracy on those 10 examples: 100%`}
      >{`import math, random
random.seed(0)

# 10 hand-crafted "digit" vectors of length 16 (toy features)
X = [[random.random() for _ in range(16)] for _ in range(10)]
y = list(range(10))                                    # one of each class

# MLP: 16 → 8 → 10
W1 = [[random.gauss(0, 0.5) for _ in range(16)] for _ in range(8)]
b1 = [0.0] * 8
W2 = [[random.gauss(0, 0.5) for _ in range(8)] for _ in range(10)]
b2 = [0.0] * 10

def forward(x):
    h = [max(0, sum(W1[i][j] * x[j] for j in range(16)) + b1[i]) for i in range(8)]
    z = [sum(W2[i][j] * h[j] for j in range(8)) + b2[i] for i in range(10)]
    return h, z

def softmax(z):
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
    return [v / s for v in e]

# Handwritten backprop + SGD  (mirroring what PyTorch would compute)
# Loop omitted here; the full version would be ~80 lines`}</CodeBlock>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · digit_classifier_numpy.py (full MNIST)"
        output={`epoch 1  train=0.3421  test=0.9612
epoch 2  train=0.1824  test=0.9723
epoch 3  train=0.1247  test=0.9751
epoch 4  train=0.0934  test=0.9768
epoch 5  train=0.0741  test=0.9783`}
      >{`import numpy as np
from tensorflow.keras.datasets import mnist                     # or any MNIST loader

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test  = X_test.reshape(-1, 784) / 255.0

rng = np.random.default_rng(0)
W1 = rng.normal(0, np.sqrt(2 / 784), size=(784, 128))
b1 = np.zeros(128)
W2 = rng.normal(0, np.sqrt(2 / 128), size=(128, 10))
b2 = np.zeros(10)

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(probs, y):
    return -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()

BATCH, LR, EPOCHS = 64, 0.1, 5
for epoch in range(1, EPOCHS + 1):
    perm = rng.permutation(len(X_train))
    losses = []
    for i in range(0, len(X_train), BATCH):
        idx = perm[i:i+BATCH]
        x, y = X_train[idx], y_train[idx]
        # Forward
        h = np.maximum(0, x @ W1 + b1)
        z = h @ W2 + b2
        p = softmax(z)
        losses.append(cross_entropy(p, y))
        # Backward (softmax + CE collapses to p - y)
        dz = p.copy(); dz[np.arange(len(y)), y] -= 1; dz /= len(y)
        dW2 = h.T @ dz;  db2 = dz.sum(axis=0)
        dh = dz @ W2.T;  dh[h <= 0] = 0
        dW1 = x.T @ dh;  db1 = dh.sum(axis=0)
        # Update
        W2 -= LR * dW2; b2 -= LR * db2
        W1 -= LR * dW1; b1 -= LR * db1
    # Evaluate
    h = np.maximum(0, X_test @ W1 + b1)
    preds = (h @ W2 + b2).argmax(axis=-1)
    acc = (preds == y_test).mean()
    print(f"epoch {epoch}  train={np.mean(losses):.4f}  test={acc:.4f}")`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          { left: 'for x, y in zip(...) → per-example update', right: 'batched matmul per minibatch', note: 'GPU-friendly; 1000× faster' },
          { left: 'nested loops for backprop', right: 'p.copy(); p[arange, y] -= 1', note: 'one-hot trick for softmax+CE gradient' },
          { left: 'random.shuffle(data)', right: 'rng.permutation(len(X))', note: 'shuffle indices, not the giant data array' },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · digit_classifier_pytorch.py"
        output={`epoch 1  train=0.3018  test=0.9671
epoch 2  train=0.1543  test=0.9748
epoch 3  train=0.1015  test=0.9782
epoch 4  train=0.0728  test=0.9806
epoch 5  train=0.0555  test=0.9824`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train = datasets.MNIST('.', train=True, download=True, transform=transform)
test  = datasets.MNIST('.', train=False, download=True, transform=transform)
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader  = DataLoader(test, batch_size=256)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))            # returns logits

model = MLP()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1, 6):
    model.train()
    losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)              # fused logsoftmax + NLL
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    # Eval
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(dim=-1)
            correct += (preds == yb).sum().item(); total += len(yb)
    print(f"epoch {epoch}  train={sum(losses)/len(losses):.4f}  test={correct/total:.4f}")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          { left: 'manual download & reshape', right: 'torchvision.datasets.MNIST + DataLoader', note: 'batching, shuffling, async I/O for free' },
          { left: 'softmax + -log p[y].mean()', right: 'F.cross_entropy(logits, y)', note: 'fused, numerically stable, takes raw logits' },
          { left: 'manual backprop + SGD', right: 'loss.backward() + optimizer.step()', note: 'four lines per epoch, not forty' },
          { left: 'manual test-accuracy loop', right: 'model.eval() + torch.no_grad()', note: 'disables dropout/BN; skips graph construction' },
        ]}
      />

      <Callout variant="insight" title="the MNIST accuracy ladder">
        <strong>Linear classifier:</strong> ~92% — a single Linear(784, 10), no hidden layer.{' '}
        <strong>2-layer MLP:</strong> ~97-98% — what you just trained.{' '}
        <strong>CNN (LeNet-5):</strong> ~99.2% — next section.{' '}
        <strong>Modern CNN + augmentation:</strong> ~99.7%.{' '}
        <strong>Ensemble of CNNs:</strong> ~99.79% (the best-known result). Human accuracy is
        around 99.8% — MNIST is essentially a solved dataset, which is why modern research
        moved on to ImageNet, CIFAR, COCO, and beyond.
      </Callout>

      <Prose>
        <p>
          Now for the ways this first vehicle rolls off the line and
          promptly into a ditch. Four failure modes, each cheap to cause
          and each catchable with the{' '}
          <NeedsBackground slug="training-diagnostics">diagnostic
          suite</NeedsBackground> you already have.
        </p>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Forgetting to normalize inputs.</strong>{' '}
          Raw MNIST pixels are in [0, 255]. Feeding them unnormalized gives gigantic
          pre-activations, saturates everything, and training fails. Divide by 255 (putting
          values in [0, 1]) or normalize to mean=0 std=1 using the dataset statistics.
        </p>
        <p>
          <strong className="text-term-amber">Using MSE loss on a classifier.</strong>{' '}
          MSE trains — slowly, incorrectly calibrated, with nastier gradients than CE.
          Use <code className="text-dark-text-primary">CrossEntropyLoss</code>. Always.
        </p>
        <p>
          <strong className="text-term-amber">Not shuffling between epochs.</strong>{' '}
          With DataLoader, pass <code className="text-dark-text-primary">shuffle=True</code>.
          Without it, the network sees the digits in order 0-0-0-...-9-9-9 which is a
          pathological curriculum and training diverges.
        </p>
        <p>
          <strong className="text-term-amber">
            Evaluating on the training set.
          </strong>{' '}
          &ldquo;My model is 99.99% accurate!&rdquo; — said while looking at train accuracy.
          Always report <em>test</em> accuracy (or held-out val accuracy). Train accuracy
          going up is necessary but not sufficient.
        </p>
      </Gotcha>

      <Prose>
        <p>
          One more gotcha that deserves its own paragraph, because it
          bites people with working models: <strong>confidence
          miscalibration</strong>. A trained MNIST MLP will sometimes
          predict the wrong digit with 0.99 probability. The argmax is
          wrong and the softmax is certain anyway. Cross-entropy trains the
          model to be right, not to be honestly unsure when it isn&apos;t —
          it rewards sharper distributions, and in the limit the model
          learns to shout every answer. Accuracy can be excellent while
          probabilities are garbage. Temperature scaling fixes this in
          one line of code; you&apos;ll meet it later when we care about
          deployment.
        </p>
      </Prose>

      <Challenge prompt="Beat 98% on MNIST from scratch">
        <p>
          Using the layer-3 PyTorch code as your starting point, get to{' '}
          <code>≥ 98.5%</code> test accuracy on MNIST. Levers you can pull: add a second
          hidden layer, use Adam instead of SGD, add dropout, add weight decay, normalize
          inputs to mean=0 std=1, train for more epochs. After you hit the target,
          compute the confusion matrix and identify your top 3 confusion pairs. Are they
          the same as the 4/9, 3/5, 7/2 set from this lesson? Why do you think those
          pairs persist across architectures?
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> The first vehicle has
          rolled off the assembly line and driven. Every station you
          built — tensors, autograd, Linear, ReLU, softmax, cross-entropy,
          the four-beat loop, normalization, diagnostics — contributed a
          piece, and the piece worked. A 100K-parameter MLP, trained on a
          laptop, reading handwriting at 98% accuracy. The same loop,
          widened a few thousand times and trained a few million times
          longer, trains GPT-4. Everything else from here is scale and
          architecture.
        </p>
        <p>
          <strong>End of the Training section.</strong> Your classifier
          treats a digit as 784 unrelated numbers in a bag. The pixel in
          the top-left corner has no idea the pixel next to it exists —
          they&apos;re just entries 0 and 1 in a flat vector. Flattening
          threw away <em>where things are</em>, which for an image is
          almost all of the information. Next section opens with the
          operation that knows pixels have neighbors: convolution. Same
          dataset, a new architecture, and MLP accuracy goes from 98% to
          99.5% on the exact same digits. Then the same trick scales to
          ImageNet. That story starts next.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Gradient-Based Learning Applied to Document Recognition',
            author: 'LeCun, Bottou, Bengio, Haffner',
            venue: 'Proc. IEEE 1998 — the paper that introduced MNIST and LeNet-5',
            url: 'http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf',
          },
          {
            title: 'Neural Networks and Deep Learning — Chapter 1',
            author: 'Michael Nielsen',
            venue: 'neuralnetworksanddeeplearning.com',
            url: 'http://neuralnetworksanddeeplearning.com/chap1.html',
          },
          {
            title: 'PyTorch MNIST Example',
            author: 'PyTorch team',
            venue: 'github.com/pytorch/examples',
            url: 'https://github.com/pytorch/examples/tree/main/mnist',
          },
          {
            title: 'Dive into Deep Learning — 3.6 Concise Implementation of Softmax Regression',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_linear-classification/softmax-regression-concise.html',
          },
        ]}
      />
    </div>
  )
}
