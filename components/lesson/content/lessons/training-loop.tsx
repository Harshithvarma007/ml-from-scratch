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
  AsciiBlock,
} from '../primitives'
import FullTrainingRun from '../widgets/FullTrainingRun'
import LossCurveSmoothness from '../widgets/LossCurveSmoothness'

export default function TrainingLoopLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="training-loop" />

      <Prose>
        <p>
          Every model that has ever been trained, anywhere, runs on a
          four-beat metronome. Forward. Loss. Backward. Step. One, two, three,
          four, repeat. A tiny MLP learning to fit a line. A 70B-parameter LLM
          burning through a GPU pod for six weeks. Same four beats. Same
          order. Different scale.
        </p>
        <p>
          You&apos;ve already built every one of those beats by hand. The{' '}
          <NeedsBackground slug="mlp-from-scratch">forward pass</NeedsBackground>{' '}
          is the model computing a prediction. The loss is one scalar saying
          how wrong that prediction is. The backward pass is{' '}
          <NeedsBackground slug="backpropagation">backpropagation</NeedsBackground>{' '}
          walking the chain rule to fill <code>.grad</code>. The step is{' '}
          <NeedsBackground slug="gradient-descent">gradient descent</NeedsBackground>{' '}
          nudging the parameters against their gradients. This lesson is the
          ritual that wires them together into something you can run a million
          times without thinking about it.
        </p>
        <p>
          The <KeyTerm>training loop</KeyTerm> is not incidentally the
          heartbeat of modern ML. Structurally, it is the only computation
          that matters. Schedulers, gradient clipping, mixed precision,
          distributed backends, checkpointing — all of it is ornamentation on
          the metronome&apos;s four beats. The rhythm is identical across every
          architecture, every framework, every scale.
        </p>
      </Prose>

      <AsciiBlock caption="the training loop, in context">
{`                          ┌────── inner loop: one batch ──────┐

for epoch in range(EPOCHS):             # 1. outer loop — passes over the dataset
    for batch in dataloader:            # 2. inner loop — one batch at a time
        optimizer.zero_grad()           #      clear .grad buffers
        yhat = model(batch.x)           #      forward pass (build graph)
        loss = criterion(yhat, batch.y) #      scalar loss
        loss.backward()                 #      walk the graph, fill .grad
        optimizer.step()                #      θ ← θ − α·∇L

One epoch  =  one full pass over the training set.
One step   =  one batch processed  =  one parameter update.
If dataset has N examples and batch size is B, one epoch is ⌈N/B⌉ steps.`}
      </AsciiBlock>

      <Prose>
        <p>
          Press play below. A tiny linear regression trains itself in your
          browser, one batch at a time, following the four-beat pattern above
          exactly. The sparkline shows per-batch loss; the dashed line is the
          epoch-averaged value. Shift the batch size and the curve
          smoothness changes — which is the topic of the next widget.
        </p>
      </Prose>

      <FullTrainingRun />

      <Personify speaker="Training loop">
        I do not care what your model is. I do not care whether it has six
        parameters or seventy billion. I do one thing: draw a batch, run
        forward, compute loss, run backward, step the optimizer. I will do
        this a million times without complaint. Give me good data, a sensible
        learning rate, and a working forward pass, and I will produce a
        trained model. Get any of those wrong and I will produce garbage,
        with equal enthusiasm.
      </Personify>

      <Prose>
        <p>
          Notice what&apos;s moving inside each beat of the metronome. The
          model never sees the whole dataset at once; it sees a{' '}
          <em>batch</em>. Processing batches — rather than computing the exact
          gradient over the entire training set — comes down to three things,
          each one damning on its own:
        </p>
        <ul>
          <li>
            <strong>Memory.</strong> A modern LLM&apos;s full dataset is
            trillions of tokens; the activations for even a thousand examples
            won&apos;t fit on a GPU. Batches are what&apos;s physically
            possible.
          </li>
          <li>
            <strong>Compute throughput.</strong> GPUs are matrix processors.
            Processing 64 examples simultaneously is maybe 0.5% slower than
            processing one, because the matmul machinery is saturated either
            way. Full-batch training leaves hardware idle.
          </li>
          <li>
            <strong>Generalization.</strong> The noise introduced by sampling
            a different batch each step is, empirically, a regulariser.
            Stochastic gradients help networks escape sharp local minima that
            generalise poorly.
          </li>
        </ul>
      </Prose>

      <MathBlock caption="stochastic gradients — unbiased estimators of the full gradient">
{`Full-batch gradient (what we want, can't afford):
  ∇L(θ)   =   (1/N) · Σ_{i=1..N}   ∇ℓ(θ; xᵢ, yᵢ)

Minibatch gradient (what we actually compute):
  ∇L_B(θ) =   (1/|B|) · Σ_{i ∈ B}   ∇ℓ(θ; xᵢ, yᵢ)

Key property: E[∇L_B] = ∇L for B drawn uniformly at random.
So each minibatch step is, on expectation, a step in the true descent direction —
plus mean-zero noise whose variance shrinks as batch size grows.`}
      </MathBlock>

      <Prose>
        <p>
          Variance of the minibatch gradient is proportional to{' '}
          <code>1 / |B|</code>. Double the batch size and the noise halves.
          The curves below make it visceral: three training runs on the same
          loss surface, different batch sizes.
        </p>
      </Prose>

      <LossCurveSmoothness />

      <Callout variant="insight" title="the three curves, decoded">
        Full-batch is a laminar flow — deterministic, smooth, converges
        cleanly. SGD (batch=1) is turbulent — the noise is enormous, and
        you&apos;d never use this for real training, but it can tunnel through
        sharp minima that full-batch gets stuck in. Minibatch (32–512 for
        most tasks) is the compromise every practitioner uses: most of
        full-batch&apos;s stability, most of SGD&apos;s implicit
        regularisation, and the GPU happy to chew on a batch-sized matmul.
      </Callout>

      <Personify speaker="Batch size">
        I am the single hyperparameter that changes everything underneath and
        nothing on top. Pick me too small and you are noise-limited. Pick me
        too large and you burn memory and lose the regularisation from
        stochasticity. Pick me a power of two between 32 and 512 for vision,
        between 1M and 4M tokens for LLMs, and you will be fine.
      </Personify>

      <Prose>
        <p>
          Now write the metronome from scratch. Three layers, same four beats
          at each level of the stack you&apos;ll meet in practice. The pure
          Python version does every beat by hand; the NumPy version
          vectorises the arithmetic; the{' '}
          <NeedsBackground slug="pytorch-basics">PyTorch</NeedsBackground>{' '}
          version cedes the bookkeeping to the library and leaves you with
          the four-line core you&apos;ll write from memory for the rest of
          your career.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · training_loop_scratch.py"
        output={`epoch 0  batch_avg_loss=1.3245
epoch 1  batch_avg_loss=0.7123
epoch 2  batch_avg_loss=0.3881
epoch 3  batch_avg_loss=0.2094
epoch 4  batch_avg_loss=0.1138`}
      >{`import random
random.seed(0)

# Toy dataset: y = 1.7x - 0.4 + noise
data = [(x := random.gauss(0, 1), 1.7 * x - 0.4 + random.gauss(0, 0.3)) for _ in range(128)]

w, b = 0.0, 0.0
lr = 0.05
BATCH = 16

for epoch in range(5):
    random.shuffle(data)                                         # ← critical
    epoch_loss = 0.0
    for start in range(0, len(data), BATCH):
        batch = data[start:start + BATCH]
        # zero_grad + forward + loss + backward + step, all by hand
        gw = 0.0; gb = 0.0; loss = 0.0
        for x, y in batch:
            yhat = w * x + b                                      # forward
            err = yhat - y                                        # loss pre-sum
            loss += err * err                                     # MSE
            gw += 2 * err * x                                     # backward
            gb += 2 * err
        loss /= len(batch); gw /= len(batch); gb /= len(batch)
        w -= lr * gw                                              # step
        b -= lr * gb
        epoch_loss += loss
    epoch_loss /= (len(data) / BATCH)
    print(f"epoch {epoch}  batch_avg_loss={epoch_loss:.4f}")`}</CodeBlock>

      <CodeBlock language="python" caption="layer 2 — numpy · training_loop_numpy.py">{`import numpy as np
rng = np.random.default_rng(0)

N, BATCH, EPOCHS, LR = 128, 16, 5, 0.05
x = rng.normal(size=(N,))
y = 1.7 * x - 0.4 + rng.normal(scale=0.3, size=(N,))
w, b = np.array(0.0), np.array(0.0)

for epoch in range(EPOCHS):
    perm = rng.permutation(N)                                    # shuffle each epoch
    losses = []
    for start in range(0, N, BATCH):
        idx = perm[start:start + BATCH]
        xb, yb = x[idx], y[idx]
        yhat = w * xb + b                                        # forward (vectorised)
        err = yhat - yb
        loss = (err * err).mean()                                # MSE
        gw = (2 * err * xb).mean()                               # backward
        gb = (2 * err).mean()
        w -= LR * gw                                             # step
        b -= LR * gb
        losses.append(loss)
    print(f"epoch {epoch}  batch_avg_loss={np.mean(losses):.4f}")`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'random.shuffle(data)',
            right: 'perm = rng.permutation(N)',
            note: 'shuffle indices, not the dataset — cache-friendly',
          },
          {
            left: 'for x, y in batch: ...',
            right: 'xb = x[idx]   # fancy indexing',
            note: 'whole batch slice, no Python loop',
          },
          {
            left: 'gw += 2*err*x',
            right: '(2 * err * xb).mean()',
            note: 'vectorised gradient, averaged over the batch',
          },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · training_loop_pytorch.py"
        output={`epoch 0  avg_loss=1.2833
epoch 1  avg_loss=0.6942
epoch 2  avg_loss=0.3747
epoch 3  avg_loss=0.2021
epoch 4  avg_loss=0.1096`}
      >{`import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
x = torch.randn(128, 1)
y = 1.7 * x - 0.4 + 0.3 * torch.randn_like(x)

ds = TensorDataset(x, y)
loader = DataLoader(ds, batch_size=16, shuffle=True)              # DataLoader = shuffling + batching

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(5):
    losses = []
    for xb, yb in loader:                                         # one batch per iter
        optimizer.zero_grad()                                     # 1
        yhat = model(xb)                                          # 2
        loss = criterion(yhat, yb)                                # 3
        loss.backward()                                           # 4
        optimizer.step()                                          # 5
        losses.append(loss.item())
    print(f"epoch {epoch}  avg_loss={sum(losses)/len(losses):.4f}")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'for start in range(0, N, BATCH): ... idx = perm[...]',
            right: 'for xb, yb in loader:',
            note: 'DataLoader handles batching + shuffling + worker-parallel loading',
          },
          {
            left: 'explicit forward + grad + step',
            right: 'the five one-liners',
            note: 'autograd + nn.Module + optim own the bookkeeping',
          },
          {
            left: 'manual RNG for reproducibility',
            right: 'torch.manual_seed(0) plus DataLoader generator',
            note: 'covers dataset shuffling + layer init + dropout + augmentations',
          },
        ]}
      />

      <Prose>
        <p>
          Look at the inner loop of layer 3. Five lines — well, four beats
          plus a log. <code>zero_grad</code>, <code>forward</code>,{' '}
          <code>loss</code>, <code>backward</code>, <code>step</code>. That
          is the metronome. You will type those exact five lines thousands of
          times. You will write them in your sleep. Every &ldquo;production
          training script&rdquo; on GitHub is those five lines with a
          thousand more wrapped around them for logging, checkpointing, and
          distributed coordination.
        </p>
      </Prose>

      <Callout variant="insight" title="the loop scales unchanged">
        Turning this into a real training script is 100% additive. Move
        tensors to GPU (<code>model.cuda()</code>, <code>xb.cuda()</code>).
        Add a learning-rate scheduler (<code>scheduler.step()</code> after{' '}
        <code>optimizer.step()</code>). Add gradient clipping (
        <code>nn.utils.clip_grad_norm_</code> before{' '}
        <code>optimizer.step</code>). Add mixed precision (
        <code>autocast</code> wrapping forward/loss). Add checkpointing every
        N steps. The <em>core loop</em> — those four beats — never changes.
        Every ML engineer&apos;s career is about bolting bigger things onto
        this exact scaffold.
      </Callout>

      <Prose>
        <p>
          Now the ways the metronome breaks. Most training bugs are not
          subtle errors deep in a model; they&apos;re one of the four beats
          played in the wrong order, or skipped entirely. The first beat has
          to be <code>zero_grad</code>. Backward has to come before step.
          Step has to come before the next forward. Miss any of that and the
          loop still runs — it just doesn&apos;t train.
        </p>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Forgetting <code className="text-dark-text-primary">zero_grad</code> — the dropped downbeat.</strong>{' '}
          PyTorch accumulates gradients by default. Skip the zero and step N
          does gradient descent on the sum of gradients from step N plus
          every previous step. By step 10 the effective learning rate is 10×
          what you asked for. Training blows up immediately, and it looks
          like an exploding-gradient problem when it&apos;s actually a
          forgotten beat.
        </p>
        <p>
          <strong className="text-term-amber">Stepping before backward.</strong>{' '}
          Calling <code className="text-dark-text-primary">optimizer.step()</code>{' '}
          before <code className="text-dark-text-primary">loss.backward()</code>{' '}
          applies whatever gradients are currently in <code>.grad</code> — on
          iteration one, that&apos;s zero, so nothing moves. On later
          iterations it&apos;s last step&apos;s gradient, so you train on
          stale signal. Silent, and wrong.
        </p>
        <p>
          <strong className="text-term-amber">Not shuffling between epochs.</strong>{' '}
          Without <code className="text-dark-text-primary">shuffle=True</code>{' '}
          on the DataLoader, every epoch sees batches in the same order.
          Gradients become correlated and training collapses to a
          meaningful-looking but biased solution. Always shuffle the training
          set; never shuffle the validation set.
        </p>
        <p>
          <strong className="text-term-amber">Reusing a drained iterator.</strong>{' '}
          A DataLoader is exhausted after one full pass. Wrapping the outer
          loop in{' '}
          <code className="text-dark-text-primary">for batch in iter(loader)</code>{' '}
          inside an epoch loop is wrong — the inner call gets a new iterator
          each epoch, which is what you want. Don&apos;t cache the iterator.
        </p>
        <p>
          <strong className="text-term-amber">Calling <code className="text-dark-text-primary">.item()</code> inside the loss accumulator.</strong>{' '}
          <code className="text-dark-text-primary">loss.item()</code> forces a
          CPU sync, which kills throughput on GPU. Either accumulate the
          tensor (keep it on GPU) or call{' '}
          <code className="text-dark-text-primary">.item()</code> only when
          logging.
        </p>
      </Gotcha>

      <Challenge prompt="Write the loop from memory, no Googling">
        <p>
          Close this page. Open a new notebook. Without referring back, write
          a complete training loop for MNIST —{' '}
          <code>nn.Linear(784, 10)</code>, CE loss, SGD at lr=0.1, 5 epochs,
          batch size 64. Hit <code>&gt; 90%</code> test accuracy.
        </p>
        <p className="mt-2 text-dark-text-muted">
          If you can do this, you&apos;ve internalised the metronome and
          everything in the rest of this curriculum is a variation on it. If
          you can&apos;t, come back here and read the layer-3 code one more
          time. It should feel familiar, not copied.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> The training loop is four
          beats — forward, loss, backward, step — preceded by{' '}
          <code>zero_grad</code> and wrapped in two loops (epochs over
          batches). Minibatching is the default because of memory,
          throughput, and generalisation, all at once. <code>DataLoader</code>{' '}
          handles shuffling, batching, and async I/O for you. Scale
          everything else around the metronome and leave the metronome
          itself alone.
        </p>
        <p>
          <strong>Next up — Training Diagnostics.</strong> The metronome is
          running. The loss is dropping — or it isn&apos;t, or it&apos;s
          dropping too slowly, or it went to NaN on epoch three. How do you
          know the model is actually learning? The next lesson is about
          reading the vital signs — loss curves, gradient norms, parameter
          statistics — well enough to distinguish &ldquo;converging
          slowly&rdquo; from &ldquo;silently diverging&rdquo;, which is a
          skill that will save you more time than any single algorithmic
          improvement.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Deep Learning — Chapter 8: Optimization for Training Deep Models',
            author: 'Goodfellow, Bengio, Courville',
            venue: 'MIT Press, 2016',
            url: 'https://www.deeplearningbook.org/contents/optimization.html',
          },
          {
            title: 'Large-Scale Machine Learning with Stochastic Gradient Descent',
            author: 'Léon Bottou',
            venue: 'COMPSTAT 2010',
            url: 'https://leon.bottou.org/publications/pdf/compstat-2010.pdf',
          },
          {
            title: 'Dive into Deep Learning — 12.5 Minibatch Stochastic Gradient Descent',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_optimization/minibatch-sgd.html',
          },
          {
            title: 'A Recipe for Training Neural Networks',
            author: 'Andrej Karpathy',
            venue: 'karpathy.github.io, 2019',
            url: 'https://karpathy.github.io/2019/04/25/recipe/',
          },
        ]}
      />
    </div>
  )
}
