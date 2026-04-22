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
import LayerNormVisualization from '../widgets/LayerNormVisualization'
import NormAxisSelector from '../widgets/NormAxisSelector'

// Signature anchor: a per-track sound mixer. Each input sample is a track;
// LayerNorm pulls each one to the same reference loudness and tames peaks
// within that one track, never glancing across tracks. Returned to at the
// opening (why mixing is needed), the formula reveal (per-sample, across
// features), and the γ/β reveal (the EQ that puts color back).
export default function LayerNormalizationLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="layer-normalization" />

      <Prose>
        <p>
          You spent the last section hand-tuning{' '}
          <NeedsBackground slug="weight-initialization">
            weight initialization
          </NeedsBackground>{' '}
          so activations wouldn&apos;t blow up or collapse as signals propagated
          through your{' '}
          <NeedsBackground slug="mlp-from-scratch">
            multi-layer perceptron
          </NeedsBackground>
          . Good init buys you a clean starting distribution. The problem: the
          moment{' '}
          <NeedsBackground slug="gradient-descent">
            gradient descent
          </NeedsBackground>{' '}
          starts moving the weights, the statistics start drifting again. Layer
          by layer, the signal gets louder on some channels, quieter on others,
          and by layer forty it sounds like every instrument is fighting for
          the same three dB.
        </p>
        <p>
          Think of the post-MLP signal as a mixing console. Every input sample
          is its own <em>track</em>. The {' '}
          <KeyTerm>per-track sound mixer</KeyTerm> sits after each layer, takes
          one track at a time, and evens it out: pulls it to the same
          reference loudness (mean zero), tames the peaks (unit variance), and
          hands it back. That mixer is <KeyTerm>LayerNorm</KeyTerm>.
        </p>
        <p>
          LayerNorm sits inside every transformer block ever built. GPT uses
          it. BERT uses it. Llama uses a close cousin (RMSNorm, two lessons
          out) that strips one of the knobs off. Before we get to attention
          you need to understand what LayerNorm does, which axis it averages
          over, and why it dethroned its older sibling BatchNorm in sequence
          models.
        </p>
      </Prose>

      {/* ── The operation ───────────────────────────────────────── */}
      <Prose>
        <p>
          The whole operation is three lines of arithmetic, then a fourth
          line that puts the color back. For a single track — a vector of
          features <code>x ∈ ℝᴰ</code> belonging to one example:
        </p>
      </Prose>

      <MathBlock caption="LayerNorm — the full operation">
{`μ    =   (1/D) · Σⱼ xⱼ                             # mean over features
σ²   =   (1/D) · Σⱼ (xⱼ − μ)²                       # variance over features

x̂ⱼ  =   (xⱼ − μ) / √(σ² + ε)                      # normalize to mean 0, var 1
yⱼ   =   γⱼ · x̂ⱼ  +  βⱼ                            # learned per-feature scale + shift`}
      </MathBlock>

      <Prose>
        <p>
          Read it left to right and the mixer does exactly what the metaphor
          promised. Lines one and two compute this track&apos;s mean and
          variance — the loudness and the peak energy — using only its own
          features. Line three pulls the loudness to zero and divides out the
          peak. Nothing in any of that has glanced at another track in the
          batch. Your batch size could be ten thousand, or one; the numbers
          would be identical.
        </p>
        <p>
          Line four is where the engineer gets to EQ. <code>γ</code> (gain)
          and <code>β</code> (bias) are two learned parameters per feature
          that let the network scale and shift each normalized channel back
          up. The normalization itself is parameter-free — the per-feature
          EQ is the only part training touches.
        </p>
        <p>
          The widget below starts with a batch of six tracks, each recorded
          at a different loudness and dynamic range — exactly the mess that
          drifts through a deep network. Toggle <em>LayerNorm</em> on. Each
          row snaps to mean 0, std 1, one row at a time. Click a row to see
          its stats before and after.
        </p>
      </Prose>

      <LayerNormVisualization />

      <Callout variant="note" title="why the learned γ, β matter">
        Pure <code>(x − μ) / σ</code> would force every layer&apos;s output to mean 0 variance
        1 — but sometimes the downstream task actually wants a biased distribution (an output
        logit should be concentrated near a specific value). <code>γ</code> and <code>β</code>{' '}
        let the network recover any affine transform of the normalized result, <em>after</em>{' '}
        the variance has been stabilized. If the optimum really is &ldquo;do nothing,&rdquo;
        training can set <code>γ = σ, β = μ</code> and get the raw input back. The network
        chooses.
      </Callout>

      <Personify speaker="LayerNorm">
        I work one sample at a time. I do not care how many examples you gave me, or what
        other examples in the batch look like. I take your feature vector, compute its mean
        and variance across the feature axis, subtract and divide, scale and shift with my
        learned <em>γ</em> and <em>β</em>, and hand you a feature vector whose statistics
        are stable layer after layer. A batch of one works. A batch of ten thousand works.
        I do not discriminate.
      </Personify>

      {/* ── Which axis ─────────────────────────────────────────── */}
      <Prose>
        <p>
          &ldquo;Which axis is being averaged&rdquo; is the single thing most
          people get wrong about LayerNorm, BatchNorm, and their cousins.
          Keep the mixer picture in mind: LayerNorm runs down one track at a
          time, averaging across that track&apos;s own features. BatchNorm
          does the opposite — it picks one channel and averages across every
          track in the studio. Completely different sets of numbers.
        </p>
      </Prose>

      <NormAxisSelector />

      <Prose>
        <p>
          Flip between the four options. The highlighted cells are the ones
          pooled into a single mean and variance. LayerNorm&apos;s pool is
          always one example&apos;s feature vector. BatchNorm&apos;s pool is
          every example in the batch (plus the sequence dimension in 3D
          tensors) for a single feature. Same-feature-across-examples versus
          same-example-across-features. The whole argument comes down to
          that one line.
        </p>
      </Prose>

      <Callout variant="insight" title="why transformers picked LayerNorm">
        Three reasons. (1) Sequence models have variable-length inputs; BatchNorm&apos;s
        batch statistics don&apos;t mix well with padding and packed sequences. (2)
        Transformers are often trained with effectively tiny per-device batches (or even
        batch size 1 during inference), and BatchNorm&apos;s statistics degrade as the batch
        shrinks. (3) LayerNorm is <em>per-sample</em> — which means inference behaves
        identically to training, with no running-stats bookkeeping. The modern default is
        LayerNorm, not BatchNorm, for precisely these reasons.
      </Callout>

      {/* ── Where to put it ─────────────────────────────────────── */}
      <Prose>
        <p>
          Two spots in a transformer block where the mixer can sit, and the
          choice is not cosmetic. The original 2017 paper used{' '}
          <strong>post-norm</strong> — normalize after the residual add.
          Modern implementations (GPT-2 onwards) use <strong>pre-norm</strong>{' '}
          — normalize before the sublayer. Pre-norm is more stable, trains
          more reliably, and is the default in essentially every transformer
          implementation you&apos;ll read today.
        </p>
      </Prose>

      <MathBlock caption="post-norm vs pre-norm — one indent matters">
{`# original transformer (Vaswani 2017) — post-norm
x = LayerNorm( x + Sublayer(x) )          # normalise after the residual add

# modern implementation — pre-norm
x = x + Sublayer( LayerNorm(x) )          # normalise first, then add residual`}
      </MathBlock>

      <Gotcha>
        <p>
          <strong className="text-term-amber">
            <code className="text-dark-text-primary">nn.LayerNorm(shape)</code> takes the shape to normalize <em>over</em>,
            not the batch size.
          </strong>{' '}
          For a tensor of shape <code className="text-dark-text-primary">(batch, seq, features)</code>,
          you want <code className="text-dark-text-primary">nn.LayerNorm(features)</code>. The
          axis is the last N dimensions matching that shape.
        </p>
        <p>
          <strong className="text-term-amber">
            Do not add dropout between linear + LayerNorm + activation
          </strong>{' '}
          unless you mean to. It changes the variance LayerNorm sees and can destabilize
          training. Standard transformer block is{' '}
          <code className="text-dark-text-primary">LN → attention → residual → LN → mlp → residual</code>,
          dropout applied to the residual output, not between layers.
        </p>
        <p>
          <strong className="text-term-amber">
            LayerNorm is implemented in higher precision
          </strong>{' '}
          than the rest of the model. Under fp16, the reciprocal-sqrt operation can
          catastrophically underflow. PyTorch and every serious framework computes the
          mean, variance, and division in fp32 and casts back down. Don&apos;t try to &ldquo;optimize&rdquo;
          by forcing LayerNorm into fp16 — it&apos;s one of the few ops where precision
          really matters.
        </p>
      </Gotcha>

      {/* ── Three-layer code ─────────────────────────────────────── */}
      <Prose>
        <p>
          From scratch in four lines of NumPy — the mixer, with a soldering
          iron. PyTorch ships it as a one-liner with the EQ knobs already
          wired to autograd. Same operation either way; only one runs on a
          GPU.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure numpy · layer_norm_numpy.py"
        output={`before: mean per row = [ 1.38 -0.97  2.41]  std per row = [0.87 0.95 1.01]
 after: mean per row = [-0.00 -0.00  0.00]  std per row = [1.00 1.00 1.00]`}
      >{`import numpy as np

def layer_norm(x, eps=1e-5, gamma=None, beta=None):
    """x: (..., D). Normalize over the last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    if gamma is not None: x_hat = x_hat * gamma
    if beta is not None:  x_hat = x_hat + beta
    return x_hat

rng = np.random.default_rng(0)
x = rng.normal(loc=[1, -1, 2], scale=[1, 1, 1], size=(8, 3)).T  # rows have different means
print("before: mean per row =", np.round(x.mean(-1), 2), " std per row =", np.round(x.std(-1), 2))
y = layer_norm(x)
print(" after: mean per row =", np.round(y.mean(-1), 2), " std per row =", np.round(y.std(-1), 2))`}</CodeBlock>

      <Bridge
        label="the four lines of math ↔ the four lines of code"
        rows={[
          { left: 'μ = (1/D) · Σⱼ xⱼ', right: 'x.mean(axis=-1, keepdims=True)', note: 'keepdims so broadcasting works' },
          { left: 'σ² = (1/D) · Σⱼ (xⱼ − μ)²', right: 'x.var(axis=-1, keepdims=True)', note: '' },
          { left: 'x̂ = (x − μ) / √(σ² + ε)', right: '(x - mean) / np.sqrt(var + eps)', note: 'the ε avoids div-by-zero' },
          { left: 'y = γ · x̂ + β', right: 'x_hat * gamma + beta', note: 'learned affine restoration' },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch · layer_norm_pytorch.py"
        output={`torch.Size([32, 12, 768])
mean per (b, s) = ~0.00,  std per (b, s) = ~1.00`}
      >{`import torch
import torch.nn as nn

# Shape conventions for a transformer tensor:
# batch × sequence × features.   LayerNorm is applied per (batch, seq).
x = torch.randn(32, 12, 768)

norm = nn.LayerNorm(normalized_shape=768)   # γ, β have shape (768,)
y = norm(x)

print(y.shape)
print(f"mean per (b, s) = ~{y.mean(dim=-1).abs().mean():.2f},  "
      f"std per (b, s) = ~{y.std(dim=-1).mean():.2f}")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'hand-rolled 4 lines',
            right: 'nn.LayerNorm(features)',
            note: 'fused kernel, mixed-precision aware, gradient-ready',
          },
          {
            left: 'manual gamma, beta',
            right: 'automatically learnable via autograd',
            note: 'they register as nn.Parameter',
          },
          {
            left: 'always fp32',
            right: 'internally fp32 even in fp16/bfloat16 mode',
            note: 'PyTorch handles the precision hack for you',
          },
        ]}
      />

      <Callout variant="insight" title="the backward pass is non-trivial">
        Deriving the gradient of LayerNorm by hand is an exercise in keeping track of which
        terms depend on the mean and which on the std, since every element contributes to
        both. The result is a three-term expression where half the gradient flows through the
        centered-and-scaled pathway and the other half flows through the mean/variance
        corrections. PyTorch&apos;s implementation does this with a fused CUDA kernel — if
        you ever try to write a custom backward for LayerNorm, make sure you pull the full
        derivation from a textbook; small mistakes compound.
      </Callout>

      <Challenge prompt="Compare training with and without LayerNorm">
        <p>
          Take the MLP from the previous section. Build two versions: one plain, one with{' '}
          <code>nn.LayerNorm</code> after every linear layer (before the activation). Train
          both on MNIST for 5 epochs. Plot validation accuracy vs epoch for both.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: swap in <code>nn.BatchNorm1d</code> instead of <code>nn.LayerNorm</code>{' '}
          for a third configuration. At standard MLP sizes all three converge; notice that
          BatchNorm converges fastest for medium-sized batches and LayerNorm is the most
          robust across batch sizes. The real differences show up in transformers — which
          is the whole point of the upcoming attention section.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> LayerNorm is the per-track
          mixer. It standardizes each example&apos;s feature vector to mean
          0, std 1 — independently of the batch — then lets the network
          EQ the result back with a learned <code>γ</code> and <code>β</code>.
          Two parameters per feature; four lines of arithmetic plus a
          learnable rescale. Because it never glances across tracks, batch
          size and sequence length don&apos;t enter the picture, which is
          precisely why every transformer you&apos;ve ever heard of reaches
          for it.
        </p>
        <p>
          <strong>Next up — Batch Normalization.</strong> LayerNorm&apos;s
          older cousin from 2015, which works the opposite way: one feature
          at a time, averaged <em>across</em> the batch. It dominated CNN
          training for years and invented a whole new category of &ldquo;why
          did the loss spike on batch-size-two&rdquo; debugging stories.
          When is averaging across the batch the better call, and when does
          it fall apart? That&apos;s the next lesson.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Layer Normalization',
            author: 'Ba, Kiros, Hinton',
            venue: 'arXiv 2016',
            url: 'https://arxiv.org/abs/1607.06450',
          },
          {
            title: 'On Layer Normalization in the Transformer Architecture',
            author: 'Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang, Liu',
            venue: 'ICML 2020 — the pre-norm vs post-norm analysis',
            url: 'https://arxiv.org/abs/2002.04745',
          },
          {
            title: 'Understanding and Improving Layer Normalization',
            author: 'Xu, Sun, Zhang, Zhao, Lin',
            venue: 'NeurIPS 2019',
            url: 'https://arxiv.org/abs/1911.07013',
          },
        ]}
      />
    </div>
  )
}
