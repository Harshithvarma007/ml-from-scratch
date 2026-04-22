import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import Prereq from '../Prereq'
import NeedsBackground from '../NeedsBackground'
import {
  Prose, Callout, Personify, Bridge, Gotcha, Challenge, References, KeyTerm,
} from '../primitives'
import LossCurveGallery from '../widgets/LossCurveGallery'
import GradNormMonitor from '../widgets/GradNormMonitor'
import OverfitDetector from '../widgets/OverfitDetector'

export default function TrainingDiagnosticsLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="training-diagnostics" />

      <Prose>
        <p>
          Loss is going down. That&apos;s the first thing everyone checks, and
          it&apos;s the first thing that lies to you. Your{' '}
          <NeedsBackground slug="training-loop">training loop</NeedsBackground>{' '}
          is turning, the curve is slipping to the right and down, and somewhere
          inside the model a layer has been clinically dead for three hundred
          steps. You wouldn&apos;t know. Not from loss alone.
        </p>
        <p>
          This is the stethoscope lesson. Training a neural net looks fine from
          the outside — one number, trending in the right direction — but
          inside, things quietly rot. A layer dies. Gradients vanish. One
          weight dominates. Updates shrink to nothing. A bad{' '}
          <NeedsBackground slug="weight-initialization">initialization</NeedsBackground>{' '}
          smuggles in a disaster on step zero that doesn&apos;t surface until
          step eight thousand. You can&apos;t fix what you can&apos;t see, and
          &ldquo;the loss is decreasing&rdquo; sees almost nothing.
        </p>
        <p>
          So we do what doctors do. We don&apos;t ask one question. We run the
          exam. <KeyTerm>Training diagnostics</KeyTerm> is the full physical:
          the loss curve as your EKG, the gradient norm as your pulse, the
          parameter/update ratio as your blood pressure, activation histograms
          as your skin tone, the learning-rate schedule as your sleep hygiene.
          A healthy model has a characteristic rhythm on each instrument. When
          the rhythm changes, you diagnose — you don&apos;t just keep training
          and hope the patient walks it off.
        </p>
      </Prose>

      <Callout variant="note" title="the five instruments">
        <div className="space-y-1">
          <p><strong>Loss curve</strong> — the EKG. Shape, monotonicity, gap.</p>
          <p><strong>Gradient norm</strong> — the pulse. Per layer, not just global.</p>
          <p><strong>Update-to-param ratio</strong> — the blood pressure. How hard each step is hitting each weight.</p>
          <p><strong>Activation histograms</strong> — the skin tone. Are neurons firing, saturated, or dead?</p>
          <p><strong>LR schedule</strong> — the sleep hygiene. When weird things happen at step 8000, the scheduler is usually holding the knife.</p>
        </div>
      </Callout>

      <Prose>
        <p>
          Start with the EKG. Below is a gallery of the loss curves you will
          actually see in the wild. Read them like a doctor reads a rhythm
          strip: each shape is a diagnosis, each diagnosis has a standard
          intervention. The skill — and it is a real skill, one you will use on
          every project for the rest of your career — is recognizing the shape
          inside thirty seconds and naming the cause.
        </p>
      </Prose>

      <LossCurveGallery />

      <Prose>
        <p>
          Most of these are fixed with one knob: a smaller learning rate, a
          regularizer, more data. One is different. The <em>diverging</em>{' '}
          curve is the only one where every additional step makes the patient
          worse. If the loss turns upward after step 100 and you keep training,
          you are paying GPU time to actively ruin your model. Stop the run.
        </p>
      </Prose>

      <Callout variant="note" title="a healthy curve, described">
        Log-scale y, decreasing monotonically in the rough shape of{' '}
        <code>t⁻α</code>. Train loss slightly below val loss. No sudden spikes.
        Gradient norms within one order of magnitude of their init value.
        That&apos;s the target pulse. Anything else — diagnose.
      </Callout>

      <Personify speaker="Loss curve">
        I am the first thing a competent engineer looks at, and I am also the
        most reliable liar in the room. I will not tell you <em>why</em> your
        model is failing — but I will tell you, within seconds, <em>that</em>{' '}
        it is, and what kind of failure it is. Learn my shapes. I have six of
        them. That&apos;s enough to diagnose nine problems in ten. For the
        tenth, you&apos;re going to need the other instruments.
      </Personify>

      <MathBlock caption="the per-batch loss variance — why curves are noisy at all">
{`L_batch   =   (1/|B|) · Σ_{i ∈ B}   ℓ(θ; xᵢ, yᵢ)

Var(L_batch)   =   Var(ℓ) / |B|

Halve the batch size → double the curve's wiggle.
Apply an exponential moving average to the loss log → smoother but lagged.
If the curve wiggles past what the EMA predicts, your LR is too high.`}
      </MathBlock>

      <Prose>
        <p>
          Here is the lie you need to know about, because it will cost you a
          week the first time it happens. A loss curve can slope gently down
          while a chunk of the model is dead. Gradients upstream don&apos;t
          reach the dead layer, the live parameters keep fitting the residual,
          and the overall loss improves — just slower than it should, with a
          ceiling you cannot explain. The EKG looks fine. The patient is
          quietly missing a lung. This is where the stethoscope moves to the
          next instrument.
        </p>
        <p>
          Loss tells you <em>what</em>. Gradient norms tell you <em>where</em>.
          If loss is going nowhere, or going somewhere too slowly, the next
          listen is per-layer gradient magnitude. Vanishing in the deep
          layers? You have an activation-or-init problem, and the fix lives in
          sigmoid/ReLU choice or Xavier/He scaling. Exploding? You need
          gradient clipping or a smaller step. Cycle through the four regimes
          in the widget and watch the per-layer bars while the loss looks
          roughly identical on top.
        </p>
      </Prose>

      <GradNormMonitor />

      <Callout variant="insight" title="the ratio to watch">
        Total gradient norm (summed across all parameters) is useful but hides
        problems that show up per-layer. A healthy run has a roughly uniform
        grad-norm profile across layers — maybe a factor of 3 between shallow
        and deep, not 10,000×. When the shallow-to-deep ratio exceeds 10⁴,
        training for the quiet layer has effectively stopped. The pulse is
        there, but only in one arm.
      </Callout>

      <Personify speaker="Gradient norm">
        I tell you whether signal is reaching every layer. Averaged across all
        parameters I&apos;m a blunt instrument — one number for the whole
        body. Logged per-layer I&apos;m surgical. If I&apos;m <code>10¹</code>{' '}
        at layer 1 and <code>10⁻⁹</code> at layer 6, your loss can look
        perfectly healthy while forty percent of your model has stopped
        learning. Read me layer by layer or don&apos;t read me at all.
      </Personify>

      <Prose>
        <p>
          Two instruments down. The third asks a different question — not{' '}
          <em>is the model training</em>, but <em>is the model learning the
          right thing</em>. A model that memorizes your training set will keep
          getting better on train loss while val loss climbs. Train goes to
          zero; the model is an honor student on the homework and a disaster on
          the exam. You can produce this failure at will by shrinking the
          dataset. You can cure it with regularization.
        </p>
      </Prose>

      <OverfitDetector />

      <Callout variant="insight" title="the three regularisers, ranked">
        In modern practice the order is: (1) <strong>more data</strong> —
        always the best fix, always the hardest. (2) <strong>data
        augmentation</strong> — for vision and audio, free labeled data from
        existing labeled data. (3) <strong>weight decay</strong> — a clean L2
        penalty, near-free, always on in production LLMs. (4){' '}
        <strong>dropout</strong> — still useful for CNNs and classic MLPs,
        mostly skipped in transformers. (5) <strong>early stopping</strong> —
        the cheapest regulariser: halt when val loss stops improving. Use
        multiple in combination for best effect.
      </Callout>

      <Prose>
        <p>
          Now write the instruments. Same three-layer progression as every
          other lesson in the course. Pure Python: print every loss and grad
          norm by hand — good for understanding, useless at scale. NumPy:
          aggregate the numbers so you can plot them later. PyTorch: hook into
          the real machinery with TensorBoard and per-layer backward hooks.
          The third layer is what production training scripts actually do.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 1 — pure python · minimal_diagnostics.py">{`import random, math
random.seed(0)

w, b = 0.1, 0.0
lr = 0.05
data = [(random.gauss(0,1), random.gauss(0,1)) for _ in range(100)]
targets = [2 * x - 1 for x, _ in data]

for step in range(50):
    # Forward + loss + grad
    total_loss = 0.0
    gw = 0.0; gb = 0.0
    for (x, _), y in zip(data, targets):
        yhat = w * x + b
        err = yhat - y
        total_loss += err * err
        gw += 2 * err * x
        gb += 2 * err
    total_loss /= len(data); gw /= len(data); gb /= len(data)

    # Grad norm (per "parameter", here a scalar each)
    grad_norm = math.sqrt(gw * gw + gb * gb)

    # The tracing you need: loss, grad_norm, param_norm, LR
    if step % 10 == 0:
        print(f"step {step:3d}  loss={total_loss:.4f}  |grad|={grad_norm:.4f}  "
              f"|param|={math.sqrt(w*w + b*b):.4f}")

    w -= lr * gw; b -= lr * gb`}</CodeBlock>

      <CodeBlock language="python" caption="layer 2 — numpy · aggregated diagnostics">{`import numpy as np

def run_with_diagnostics(X, y, lr=0.05, steps=100):
    w = np.zeros(X.shape[1])
    b = 0.0
    logs = {"loss": [], "grad_norm": [], "param_norm": []}
    for _ in range(steps):
        yhat = X @ w + b
        err = yhat - y
        loss = (err * err).mean()
        gw = (2 * X.T @ err) / len(X)
        gb = (2 * err).mean()
        grad_norm = np.sqrt((gw ** 2).sum() + gb ** 2)
        logs["loss"].append(loss)
        logs["grad_norm"].append(grad_norm)
        logs["param_norm"].append(np.sqrt((w ** 2).sum() + b ** 2))
        w -= lr * gw; b -= lr * gb
    return logs`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          { left: 'print every 10 steps', right: 'append every step to a dict of lists', note: 'can plot / analyse later' },
          { left: 'manual sum of squared grads', right: '(gw ** 2).sum() via numpy', note: 'scales to million-parameter models' },
        ]}
      />

      <CodeBlock language="python" caption="layer 3 — pytorch · production diagnostics">{`import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
writer = SummaryWriter()

# Register a hook on each Linear to record per-layer grad norm
per_layer_norms = {}
def make_hook(name):
    def hook(module, grad_input, grad_output):
        per_layer_norms[name] = grad_output[0].norm().item()
    return hook

for name, m in model.named_modules():
    if isinstance(m, nn.Linear):
        m.register_full_backward_hook(make_hook(name))

for step in range(1000):
    optimizer.zero_grad()
    x = torch.randn(64, 10); y = torch.randn(64, 1)
    yhat = model(x)
    loss = nn.functional.mse_loss(yhat, y)
    loss.backward()

    # The diagnostics you want on every run
    writer.add_scalar("loss", loss.item(), step)
    total_grad_norm = sum((p.grad ** 2).sum().item() for p in model.parameters() if p.grad is not None) ** 0.5
    writer.add_scalar("grad_norm/total", total_grad_norm, step)
    for name, n in per_layer_norms.items():
        writer.add_scalar(f"grad_norm/{name}", n, step)

    optimizer.step()`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          { left: 'manual dict of lists', right: 'tensorboard SummaryWriter', note: 'live plots in the browser as training runs' },
          { left: 'aggregate grad norm', right: 'per-layer grad norm via backward hooks', note: 'where the diagnostic signal actually is' },
          { left: 'loss only', right: 'loss + grad_norm + param_norm + lr', note: 'the four quantities every training script should log' },
        ]}
      />

      <Callout variant="insight" title="the four-line debugging contract">
        Every production training script should log, at minimum:{' '}
        <strong>loss</strong>, <strong>per-layer grad norm</strong>,{' '}
        <strong>per-layer param norm</strong>, and{' '}
        <strong>current learning rate</strong>. Ninety percent of debugging
        happens on those four plots. Fancy dashboards are nice, but they
        don&apos;t replace the basics — any more than an MRI replaces the
        stethoscope.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Looking only at train loss.</strong>{' '}
          Train loss going to zero means nothing — it means the model
          memorised the training set. Always log val loss. Always. Even on
          tiny runs.
        </p>
        <p>
          <strong className="text-term-amber">Smoothing too aggressively.</strong>{' '}
          TensorBoard&apos;s default EMA can hide loss spikes for 50+ steps.
          When investigating a divergence, look at the raw unsmoothed curve.
          The EMA is the patient&apos;s calm voice in the waiting room; the
          raw curve is the actual vitals.
        </p>
        <p>
          <strong className="text-term-amber">Not logging LR.</strong>{' '}
          When something weird happens at step 8000, the answer is often
          &ldquo;the scheduler just dropped LR by 10×&rdquo; or &ldquo;warmup
          just finished.&rdquo; Log LR on the same timestep axis as loss.
        </p>
        <p>
          <strong className="text-term-amber">Confusing EMA loss with raw loss.</strong>{' '}
          The number you print can be an EMA (smooth but lagged) or the raw
          per-batch value (current but noisy). Two different numbers;
          don&apos;t conflate them when comparing runs.
        </p>
      </Gotcha>

      <Challenge prompt="Reproduce 4 failure modes on purpose">
        <p>
          Take a small MLP on MNIST. Reproduce, in order: (1) diverging loss
          by setting LR to 10. (2) A plateau by setting LR to{' '}
          <code>1e-6</code>. (3) Overfit by shrinking the dataset to 100
          examples. (4) Vanishing gradients by using sigmoid in every hidden
          layer of a{' '}
          <NeedsBackground slug="multi-layer-backpropagation">10-layer network</NeedsBackground>.
          Log train loss, val loss, and total gradient norm for each. Save
          the plots. Assemble a one-page visual guide of &ldquo;these are
          the four most common problems, and this is what they look
          like.&rdquo; Tape it above your desk. You will use it.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> Training curves tell you
          what kind of failure you have. Gradient norms per layer tell you
          where the failure lives. The generalisation gap tells you whether
          you&apos;re learning or memorising. Log all three on every training
          run, from the smallest experiment to the largest pre-train. Loss
          alone is a rhythm strip, not a diagnosis. The time you save
          debugging with the full exam is enormous and the added cost is
          basically zero.
        </p>
        <p>
          <strong>Next up — Dead ReLU Detector.</strong> Of all the things
          that can silently rot inside a healthy-looking curve, one failure
          mode is so common, so invisible to aggregate stats, and so lethal
          to capacity that it earns its own lesson. Gradient norms can look
          fine on average while half your neurons are clinically dead — no
          signal in, no signal out, no gradient, no recovery. The next
          lesson builds the dedicated instrument for that hunt.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Deep Learning — Chapter 11: Practical Methodology',
            author: 'Goodfellow, Bengio, Courville',
            venue: 'MIT Press, 2016',
            url: 'https://www.deeplearningbook.org/contents/guidelines.html',
          },
          {
            title: 'A Recipe for Training Neural Networks',
            author: 'Andrej Karpathy',
            venue: 'karpathy.github.io, 2019',
            url: 'https://karpathy.github.io/2019/04/25/recipe/',
          },
          {
            title: 'Practical Recommendations for Gradient-Based Training of Deep Architectures',
            author: 'Yoshua Bengio',
            venue: 'arXiv 2012',
            url: 'https://arxiv.org/abs/1206.5533',
          },
        ]}
      />
    </div>
  )
}
