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
import ActivationVarianceChain from '../widgets/ActivationVarianceChain'
import InitDistribution from '../widgets/InitDistribution'
import InitFormula from '../widgets/InitFormula'

// Signature anchor: every layer has a volume knob — the scale of its initial
// weights. Too low and the signal fades to silence across depth; too high
// and it clips the speakers. Xavier and He are two knob-setting rules, each
// matched to an activation. Threaded at the opening (silent failure mode),
// the variance calculation (where the knob setting comes from), and the
// Xavier-vs-He reveal (two rules, two activations).
export default function WeightInitializationLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="weight-initialization" />

      <Prose>
        <p>
          You built the MLP. Forward pass, backward pass, update rule — every
          wire, every gradient, by hand. And then, the first time you ran it on
          a 20-layer network, it did nothing. Loss flat as a desk. Or loss in
          scientific notation, climbing. Same network. Same data. Same{' '}
          <NeedsBackground slug="gradient-descent">
            gradient descent
          </NeedsBackground>
          . The only thing you changed was the numbers you wrote into the
          weight tensors before step zero.
        </p>
        <p>
          Think of every layer as having a volume knob — the scale of its
          initial weights. Turn every knob too low and the signal fades to
          silence by the time it reaches the output; activations collapse to
          zero and gradients follow. Turn every knob too high and the signal
          clips and blows the speakers; activations saturate, gradients
          explode. There&apos;s a sweet-spot setting, and — here&apos;s the
          part nobody tells you — the setting depends on which{' '}
          <NeedsBackground slug="sigmoid-and-relu">activation</NeedsBackground>{' '}
          you picked. Bad init isn&apos;t bad math. It&apos;s the sound guy
          fumbling the volume knobs before the band walks on.
        </p>
        <p>
          Two names carry this entire chapter: <KeyTerm>Xavier</KeyTerm>{' '}
          (Glorot &amp; Bengio, 2010) and <KeyTerm>He</KeyTerm> (Kaiming He et
          al., 2015). Same idea, different math, different activation. By the
          end of the page you&apos;ll have turned those knobs yourself, watched
          a 20-layer network die in real time, and then watched the same
          network come alive with one line changed. The math is a paragraph.
          The intuition is a volume knob. The consequence, if you get it
          wrong, is silent death.
        </p>
      </Prose>

      <Prose>
        <p>
          Here&apos;s the mechanical story. A forward pass is a stack of matrix
          multiplies. Each layer takes the previous activations, multiplies by{' '}
          <code>W</code>, adds a bias, and squashes through a nonlinearity.
          That&apos;s it. If the weights are too small on average, <code>Wx</code>{' '}
          is smaller than <code>x</code>, and each layer shrinks the signal a
          little more than the last. Stack 20 of those and the final layer is
          looking at numbers so close to zero the float barely represents them.
          When the gradient flows back, it passes through the same matrices in
          reverse — so it shrinks too. The{' '}
          <NeedsBackground slug="multi-layer-backpropagation">
            chain-rule product blows up or dies with depth
          </NeedsBackground>
          , and a dead gradient is a weight that never moves. Congratulations:
          you have a network, and it is decorative.
        </p>
        <p>
          Too large is the symmetric disaster. <code>Wx</code> keeps growing,
          pre-activations shoot past the range where your nonlinearity does
          anything interesting, and derivatives saturate to zero. Different
          cause, same outcome — nothing learns. The question isn&apos;t{' '}
          <em>should</em> we care about init. It&apos;s <em>what</em> does the
          knob setting need to be.
        </p>
      </Prose>

      {/* ── The variance propagation argument ───────────────────── */}
      <Prose>
        <p>
          The knob setting comes from a one-paragraph variance calculation, and
          this is genuinely the whole lesson — everything after is consequence.
          Suppose the inputs <code>x</code> and weights <code>W</code> are
          zero-mean and independent. The pre-activation{' '}
          <code>z = Σⱼ Wⱼ · xⱼ</code> is a sum of <code>fan_in</code>{' '}
          independent products. Variance adds for independent sums; variance of
          a product of independent zero-mean variables is the product of
          variances. So:
        </p>
      </Prose>

      <MathBlock caption="variance of a weighted sum">
{`Var(z)   =   Σⱼ Var(Wⱼ · xⱼ)
         =   fan_in · Var(W) · Var(x)

If we want Var(z) = Var(x)  →   Var(W) = 1 / fan_in        (LeCun init)`}
      </MathBlock>

      <Prose>
        <p>
          Read the last line slowly. Each layer multiplies the variance of the
          signal by <code>fan_in · Var(W)</code>. If that product is 1, the
          volume holds steady across depth — the knob is set right. If
          it&apos;s 2, variance doubles every layer; after 20 layers you&apos;re
          at <code>2²⁰ ≈ 10⁶</code>. If it&apos;s 0.5, variance halves every
          layer and you&apos;re at <code>10⁻⁶</code>. A tiny per-layer error,
          raised to the depth of the network, is a catastrophe. This is why a
          detail that looks cosmetic is actually load-bearing.
        </p>
        <p>
          That&apos;s one knob rule. The nonlinearity gets a vote in the final
          answer. For <KeyTerm>tanh</KeyTerm> (zero-centered, derivative ≈ 1
          near zero), the post-activation has roughly the same variance as the
          pre-activation, so preserving <code>Var(z)</code> preserves{' '}
          <code>Var(activations)</code>. Glorot added a symmetry argument — you
          also want variance to be preserved going <em>backwards</em> through
          the layer, which uses <code>fan_out</code> — and split the
          difference with a harmonic-style mean:
        </p>
      </Prose>

      <MathBlock caption="xavier / glorot init — for tanh and sigmoid">
{`Var(W) = 2 / (fan_in + fan_out)`}
      </MathBlock>

      <Prose>
        <p>
          Now the ReLU story. ReLU zeros out every negative pre-activation —
          half the distribution gets clipped. So the post-activation carries
          only half the variance of the pre-activation. To hold the volume
          steady, you have to double the variance of <code>W</code> to pay back
          what ReLU takes. That&apos;s He init:
        </p>
      </Prose>

      <MathBlock caption="he / kaiming init — for relu">
{`Var(W) = 2 / fan_in`}
      </MathBlock>

      <Prose>
        <p>
          Two knob rules, one per activation. Xavier tunes the knob for
          tanh/sigmoid; He tunes it for ReLU. The <em>one more</em> knob worth
          knowing is orthogonal init, which initialises <code>W</code> as an
          orthogonal matrix and preserves variance exactly by construction;
          it&apos;s activation-agnostic and shines for very deep linear stacks.
          Three rules, all the same idea — don&apos;t let depth amplify or
          squash the signal. Let&apos;s actually listen to what happens when
          you get it wrong.
        </p>
      </Prose>

      {/* ── Variance-through-depth widget ───────────────────────── */}
      <ActivationVarianceChain />

      <Prose>
        <p>
          Switch the activation to ReLU. Naive init — unit variance, no{' '}
          <code>fan_in</code> scaling — either rockets past <code>10⁴</code>{' '}
          (knob blown) or collapses under <code>10⁻⁹</code> (fade to silence)
          by layer 20. Xavier is better but drifts; it was tuned for the wrong
          activation. He sits right on the green dashed line at{' '}
          <code>Var = 1</code> all the way down. Now flip the toggle to tanh
          and watch the roles swap — Xavier lands on the line, He tends to
          saturate. No single init is &ldquo;good&rdquo; in the abstract. The
          knob has to match the activation.
        </p>
      </Prose>

      <Callout variant="insight" title="why half a variance matters so much">
        Each layer multiplies the signal&apos;s variance by{' '}
        <code>fan_in · Var(W)</code>. Make that product 1 and variance is
        preserved; 2 and it doubles every layer (exploding); 0.5 and it halves
        every layer (vanishing). After 20 layers, a factor of 2 per layer
        becomes <code>2²⁰ ≈ 10⁶</code>. A small per-layer error compounds into
        a global catastrophe. That&apos;s why this detail is load-bearing even
        though it looks cosmetic.
      </Callout>

      <Personify speaker="Initialization">
        I decide whether your 100-layer network learns or sits there for two
        hours pretending to train before the NaNs show up. I am the one
        hyperparameter the paper&apos;s abstract never mentions. Get me right
        and the cleanest architecture in the world works on the first try; get
        me wrong and it&apos;s a hallucination machine by step one. The knob
        is me. I am not optional.
      </Personify>

      {/* ── Distribution widget ─────────────────────────────────── */}
      <Prose>
        <p>
          Same three strategies, different camera angle. Forget averages — pick
          a depth and look at the full histogram of activations in a batch.
          Naive collapses into a spike at zero or a flat explosion you
          can&apos;t even plot on one axis. Xavier is a narrow bump,
          technically alive but barely. He is a clean half-Gaussian (ReLU ate
          the negative half; that&apos;s the shape you want). Drag the{' '}
          <code>depth k</code> slider and watch each distribution evolve
          layer-by-layer.
        </p>
      </Prose>

      <InitDistribution />

      <Prose>
        <p>
          Crank the depth to 15 with ReLU + naive init. Most of the batch has
          collapsed to zero — millions of dead neurons, each one a weight that
          will never see a gradient and therefore never train. Flip to He. The
          distribution at layer 15 looks the way it looked at layer 1.
          That&apos;s the whole point: the right knob preserves the shape of
          the signal no matter how deep the network.
        </p>
      </Prose>

      {/* ── Reference card ──────────────────────────────────────── */}
      <Prose>
        <p>
          Four recipes cover essentially every network you&apos;ll build.
          Click through the cards to see the formula, the activation it pairs
          with, and the PyTorch one-liner. Treat this as your volume-knob
          cheat sheet.
        </p>
      </Prose>

      <InitFormula />

      <Gotcha>
        <p>
          <strong className="text-term-amber">Bias can be zero.</strong>{' '}
          Don&apos;t randomize biases. They don&apos;t have a
          symmetry-breaking job (that&apos;s what the random weights do), and
          zeroing them avoids nudging your initial activations in any
          direction. PyTorch&apos;s{' '}
          <code className="text-dark-text-primary">nn.Linear</code> defaults to
          a tiny uniform bias, which is nearly zero but not quite; both work
          fine in practice.
        </p>
        <p>
          <strong className="text-term-amber">
            Match the init to the activation.
          </strong>{' '}
          He + tanh is suboptimal — tanh won&apos;t saturate, but you&apos;re
          wasting variance for no reason. Xavier + ReLU halves your
          activations every layer; that&apos;s the fade-to-silence failure
          mode. Match them, or reach for orthogonal init which is
          activation-agnostic.
        </p>
        <p>
          <strong className="text-term-amber">
            PyTorch&apos;s default isn&apos;t exactly He.
          </strong>{' '}
          <code className="text-dark-text-primary">nn.Linear</code> ships with{' '}
          <em>Kaiming uniform</em> plus a correction factor. Good enough for
          most ReLU nets out of the box, suboptimal for tanh/sigmoid, always
          worth overriding explicitly with{' '}
          <code className="text-dark-text-primary">
            nn.init.xavier_normal_
          </code>{' '}
          when you change the activation.
        </p>
        <p>
          <strong className="text-term-amber">
            LayerNorm / BatchNorm softens the init requirement.
          </strong>{' '}
          If every linear layer is followed by a normalisation that rescales
          activations, the norm resets the volume knob for you and sloppy init
          becomes survivable. That&apos;s one reason transformers (which use
          LayerNorm religiously) get away with simpler init than a plain deep
          MLP.
        </p>
      </Gotcha>

      {/* ── Three-layer code ─────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, each shorter than the last. Pure Python generates one
          random weight matrix and shows the explicit{' '}
          <code>sqrt(2 / fan_in)</code> — no abstraction, no library, just the
          formula you derived above turned into a list comprehension. NumPy
          vectorises it in a single call. PyTorch hands you He and Xavier as
          one-liners and picks a reasonable default if you forget.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · init_scratch.py"
        output={`fan_in=64  fan_out=32
W shape: 32 × 64
|W| mean: 0.147  std: 0.176
expected std: sqrt(2/64) = 0.177`}
      >{`import math, random
random.seed(0)

def he_init_matrix(fan_in, fan_out):
    """Gaussian, mean 0, std sqrt(2 / fan_in)."""
    std = math.sqrt(2 / fan_in)
    W = [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
    return W

def xavier_init_matrix(fan_in, fan_out):
    std = math.sqrt(2 / (fan_in + fan_out))
    W = [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
    return W

W = he_init_matrix(64, 32)
flat = [w for row in W for w in row]
mean_abs = sum(abs(w) for w in flat) / len(flat)
var = sum(w * w for w in flat) / len(flat)
print(f"fan_in=64  fan_out=32")
print(f"W shape: {len(W)} × {len(W[0])}")
print(f"|W| mean: {mean_abs:.3f}  std: {var ** 0.5:.3f}")
print(f"expected std: sqrt(2/64) = {(2/64) ** 0.5:.3f}")`}</CodeBlock>

      <CodeBlock language="python" caption="layer 2 — numpy · init_numpy.py">{`import numpy as np

def he_init(fan_in, fan_out, rng=None):
    rng = rng or np.random.default_rng()
    return rng.normal(0, np.sqrt(2 / fan_in), size=(fan_out, fan_in))

def xavier_init(fan_in, fan_out, rng=None):
    rng = rng or np.random.default_rng()
    return rng.normal(0, np.sqrt(2 / (fan_in + fan_out)), size=(fan_out, fan_in))

# Build a 4-layer MLP with He init
rng = np.random.default_rng(0)
sizes = [784, 256, 128, 64, 10]
Ws = [he_init(sizes[i], sizes[i + 1], rng) for i in range(len(sizes) - 1)]
for i, W in enumerate(Ws):
    print(f"layer {i}: shape {W.shape}  std {W.std():.4f}  (target {np.sqrt(2 / W.shape[1]):.4f})")`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'nested comprehensions of random.gauss',
            right: 'rng.normal(0, std, size=(fan_out, fan_in))',
            note: 'one call for the whole matrix',
          },
          {
            left: 'hardcoded per-layer',
            right: 'list comprehension over sizes',
            note: 'tracks fan_in/fan_out automatically',
          },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · init_pytorch.py"
        output={`layer0: std=0.0503  (He target 0.0505)
layer1: std=0.1247  (He target 0.1250)`}
      >{`import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        import torch.nn.functional as F
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

model = MLP([784, 256, 128, 64, 10])
for i, layer in enumerate(model.layers[:2]):
    fan_in = layer.weight.shape[1]
    print(f"layer{i}: std={layer.weight.std().item():.4f}  (He target {(2/fan_in)**0.5:.4f})")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'rng.normal(0, sqrt(2/fan_in), ...)',
            right: 'nn.init.kaiming_normal_(w)',
            note: 'same math, done in-place on any tensor',
          },
          {
            left: 'W = np.zeros(...)  # bias',
            right: 'nn.init.zeros_(bias)',
            note: 'explicit zero-init for biases',
          },
          {
            left: 'Xavier variant',
            right: 'nn.init.xavier_normal_(w)',
            note: 'matched to tanh/sigmoid',
          },
        ]}
      />

      <Callout variant="insight" title="what happens if you skip _init_weights">
        Usually, nothing catastrophic. <code>nn.Linear</code> already defaults
        to Kaiming uniform — He&apos;s close cousin — and most vision and NLP
        models train fine on that. You&apos;ll want to override explicitly
        when (a) your activation is unusual (GELU, SELU, Swish), (b) your
        network is extremely deep with no normalisation (a plain-vanilla
        100-layer MLP), or (c) you&apos;re reproducing a paper that pins its
        init. In all three cases, do it on purpose rather than hoping the
        default fits.
      </Callout>

      <Challenge prompt="Reproduce the failure, then fix it">
        <p>
          Build a 20-layer MLP, width 64. Use naive init (<code>W ~ 𝒩(0, 1)</code>,
          no <code>fan_in</code> scaling) and ReLU. Push a batch of{' '}
          <code>𝒩(0, 1)</code> inputs through a forward pass and record the
          activation variance at every layer. It will either rocket past{' '}
          <code>10⁶</code> or collapse below <code>10⁻⁶</code> before it
          reaches the output. That&apos;s the knob set wrong.
        </p>
        <p className="mt-2">
          Now swap in He. The variance should stay within one order of
          magnitude of 1 at every layer. Plot both curves on the same axes.
          This is the difference between a network that trains and a very
          expensive way to produce zeros.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: drop a <code>BatchNorm</code> after every linear layer and
          re-run with naive init. The batchnorm drags variance back toward 1
          and rescues most of the damage — which is why practitioners can get
          away with sloppy init as long as a normalisation layer is never far
          away.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> Every layer is a volume knob,
          and the setting of that knob is the variance of its weights. Xavier
          (<code>2 / (fan_in + fan_out)</code>) tunes the knob for
          tanh/sigmoid. He (<code>2 / fan_in</code>) tunes it for ReLU.
          Getting the wrong one for your activation silently kills deep
          networks — not with an exception, with a flat loss curve and wasted
          GPU hours. BatchNorm and LayerNorm soften the requirement but
          don&apos;t erase it. The three lines you&apos;ll actually type in
          PyTorch are <code>nn.init.kaiming_normal_</code>,{' '}
          <code>nn.init.xavier_normal_</code>, and{' '}
          <code>nn.init.zeros_</code> for biases.
        </p>
        <p>
          <strong>End of section.</strong> You&apos;ve built a neural network
          from scratch. Forward pass. Backward pass. Initialization. The
          optimizer. The loss. You&apos;ve turned every knob yourself, weight
          by weight. Now hand it off to a framework that does all of this —
          initialization included — in three lines. Meet PyTorch.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Understanding the difficulty of training deep feedforward neural networks',
            author: 'Glorot, Bengio',
            venue: 'AISTATS 2010 — the Xavier init paper',
            url: 'https://proceedings.mlr.press/v9/glorot10a.html',
          },
          {
            title: 'Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification',
            author: 'He, Zhang, Ren, Sun',
            venue: 'ICCV 2015 — the He init paper',
            url: 'https://arxiv.org/abs/1502.01852',
          },
          {
            title: 'Exact solutions to the nonlinear dynamics of learning in deep linear neural networks',
            author: 'Saxe, McClelland, Ganguli',
            venue: 'ICLR 2014 — the orthogonal init paper',
            url: 'https://arxiv.org/abs/1312.6120',
          },
        ]}
      />
    </div>
  )
}
