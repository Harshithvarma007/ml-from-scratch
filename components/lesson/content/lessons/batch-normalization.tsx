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
import BatchNormLive from '../widgets/BatchNormLive'
import BatchSizeFailure from '../widgets/BatchSizeFailure'

// Signature anchor: a teacher who grades on the curve. BatchNorm looks at how
// the whole class (batch) scored on each exam question (feature), centers
// everyone around the class mean, and rescales to a consistent spread.
// Returns at three load-bearing moments: the opening hook (older cousin of
// LayerNorm), the per-feature-across-batch formula reveal, and the
// train/eval mode-switch gotcha.
export default function BatchNormalizationLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="batch-normalization" />

      <Prose>
        <p>
          You just met{' '}
          <NeedsBackground slug="layer-normalization">LayerNorm</NeedsBackground>
          , which normalizes each example in isolation. Meet its older, rowdier
          cousin. BatchNorm looks at a whole batch at once and asks a different
          question: not &ldquo;how does this example&apos;s features compare to
          each other,&rdquo; but &ldquo;how does this feature compare to the
          same feature across everyone else in the batch?&rdquo;
        </p>
        <p>
          Picture a teacher grading on the curve. You and thirty-one classmates
          take an exam. For each question (call that a <em>feature</em>), she
          computes the class mean and the class spread, then rescales every
          student&apos;s score: subtract the class mean, divide by the class
          spread. A 74 on a question the class bombed is now great. A 74 on a
          question everyone aced is now mediocre. That is BatchNorm. The batch
          is the class. The feature is the exam question. The statistics come
          from <em>across the class</em>, not from within one student.
        </p>
        <p>
          <KeyTerm>Batch Normalization</KeyTerm> landed in 2015 (Ioffe &amp;
          Szegedy) and quietly rewrote deep learning. Vision networks doubled
          in depth the year after. Learning rates went up by an order of
          magnitude. The whole &ldquo;go very deep&rdquo; CNN era is hard to
          picture without it. Sequence models eventually defected to LayerNorm
          because BatchNorm has a flaw we&apos;ll get to — but every convnet
          you will ever load still has BN sprinkled through it like salt.
        </p>
      </Prose>

      {/* ── Operation ───────────────────────────────────────────── */}
      <Prose>
        <p>
          Mechanically it looks like its cousin: subtract mean, divide by
          standard deviation, restore with a learned affine. The only thing
          that changes is <em>which axis</em> the mean and variance are
          computed over. For BatchNorm on a dense layer with feature dim{' '}
          <code>D</code> and a batch of <code>N</code> examples:
        </p>
      </Prose>

      <MathBlock caption="BatchNorm — per-feature statistics across the batch">
{`For each feature j ∈ {1 … D}, over the batch:

μⱼ     =   (1/N) · Σᵢ xᵢⱼ                           # mean across the batch
σⱼ²    =   (1/N) · Σᵢ (xᵢⱼ − μⱼ)²                   # variance across the batch

x̂ᵢⱼ   =   (xᵢⱼ − μⱼ) / √(σⱼ² + ε)                 # normalize
yᵢⱼ   =   γⱼ · x̂ᵢⱼ  +  βⱼ                          # learned per-feature affine`}
      </MathBlock>

      <Prose>
        <p>
          Read the indices carefully, because this is where BatchNorm and
          LayerNorm part ways. LayerNorm fixes the example <code>i</code> and
          sweeps the mean across the feature index <code>j</code>. BatchNorm
          fixes the feature <code>j</code> and sweeps across the example
          index <code>i</code>. Same operation, perpendicular axis. Both
          output a tensor shaped like the input. Both ship with two learned
          parameters per feature. The axis is the whole fight.
        </p>
        <p>
          Back to the classroom. Each student is one row <code>i</code>. Each
          exam question is one column <code>j</code>. BatchNorm reads
          column-wise — grade question 1 on the class curve, grade question 2
          on the class curve, and so on. LayerNorm reads row-wise — grade each
          student against their own average. Different teachers, different
          philosophies.
        </p>
      </Prose>

      {/* ── Train/eval mode + running stats ─────────────────────── */}
      <Prose>
        <p>
          Now the infamous catch. What happens on exam day when only one
          student shows up? The curve is meaningless — a single score has no
          mean to subtract and no spread to divide by. BatchNorm has the same
          problem: at inference time you might feed it a single example, and a
          sample of one has no variance. So it keeps a second set of books.
        </p>
        <p>
          During training, the mean and variance come from the current batch —
          live class statistics. On the side, BatchNorm maintains an{' '}
          <em>exponential moving average</em> of those per-feature stats
          across every batch it has seen. Think of it as the historical class
          curve: what has this feature looked like, on average, across the
          whole semester? At inference time the live batch is ignored and the
          historical curve takes over. Grades stay sensible even when one
          student walks in alone.
        </p>
      </Prose>

      <MathBlock caption="running statistics — the update rule during training">
{`running_mean   ←   (1 − momentum) · running_mean   +   momentum · batch_mean
running_var    ←   (1 − momentum) · running_var    +   momentum · batch_var

# momentum is typically 0.1 (i.e. 10% new info per step)
# at eval time: use running_mean and running_var, do not update them`}
      </MathBlock>

      <Prose>
        <p>
          Watch it happen. Below, a simulated training run samples batches
          from a slowly drifting distribution — imagine the upstream layers
          (which, by the way, are being tuned by{' '}
          <NeedsBackground slug="gradient-descent">
            gradient descent
          </NeedsBackground>{' '}
          and{' '}
          <NeedsBackground slug="weight-initialization">
            whatever weights you started from
          </NeedsBackground>
          ) steadily shifting what each feature looks like. The dashed curves
          are the per-batch stats. The solid curves are the exponential
          average. Flip to <em>eval</em> mode and the solid curves freeze.
          That freeze is <code>model.eval()</code> in PyTorch.
        </p>
      </Prose>

      <BatchNormLive />

      <Callout variant="note" title="model.train() vs model.eval()">
        These two calls are the only way to change BatchNorm&apos;s behavior,
        and the single most common reason a PyTorch model gives different
        answers in a script than it did in your notebook. Always{' '}
        <code>model.eval()</code> before inference. Always{' '}
        <code>model.train()</code> before resuming training. Dropout reads the
        same flag — set it once, both layers do the right thing. Forget it and
        your test-time BatchNorm silently grades on a one-student curve, or
        your dropout zeros half your activations at serving time. Either way,
        the numbers get strange and you lose an afternoon.
      </Callout>

      <Personify speaker="BatchNorm">
        In training I trust the class in the room. I take the mean and
        variance of the thirty-two exams on my desk, per question, and hand
        back curved scores. While I do that I&apos;m also updating a running
        record of the semester. At eval time the room is empty — maybe one
        student walks in — so I ignore them and use the record. Shrink the
        class to four students and my curve gets noisy. Shrink it to one and I
        have nothing to grade on.
      </Personify>

      {/* ── Batch size matters ──────────────────────────────────── */}
      <Prose>
        <p>
          That noise problem is the whole reason BatchNorm is not a universal
          answer. The standard error of a sample mean scales like{' '}
          <code>σ / √N</code>. Halve the batch and your error goes up by{' '}
          <code>√2</code>. Run BatchNorm with a batch of one and the sample
          mean is literally the sample — zero variance, nothing to divide by
          except <code>ε</code>. Our teacher with a class of one just turns in
          a blank gradebook.
        </p>
      </Prose>

      <BatchSizeFailure />

      <Prose>
        <p>
          At <code>N ≥ 32</code> the estimate is trustworthy. Below that,
          BatchNorm starts injecting batch-specific noise into the forward
          pass, and the running averages lag whatever distribution the layers
          above are actually producing. This is why transformer people
          eventually gave up on BN: per-device batch sizes of 1-4 are routine
          in that world (sequences are long, memory is short), and BatchNorm
          simply does not cope. LayerNorm, normalizing within a single
          example, never cared how many students were in the room.
        </p>
      </Prose>

      <Callout variant="insight" title="why BatchNorm actually helps training">
        Ioffe and Szegedy&apos;s original pitch was &ldquo;internal covariate
        shift&rdquo; — BN stabilizes the distribution each layer sees as the
        ones below it learn. Later work (Santurkar et al., 2018) showed that
        story is mostly wrong. BatchNorm helps because it <em>smooths the
        loss landscape</em>, making gradients more predictable and larger
        learning rates safe. Whatever the mechanism, the effect is real: 10×
        bigger learning rates, networks that used to diverge now converge.
        The explanation moved; the empirical win didn&apos;t.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">
            BatchNorm1d vs BatchNorm2d vs BatchNorm3d.
          </strong>{' '}
          PyTorch ships three variants.{' '}
          <code className="text-dark-text-primary">BatchNorm1d</code> takes
          feature vectors (<code className="text-dark-text-primary">(N, D)</code>{' '}
          or <code className="text-dark-text-primary">(N, D, L)</code>).{' '}
          <code className="text-dark-text-primary">BatchNorm2d</code> takes
          images (<code className="text-dark-text-primary">(N, C, H, W)</code>)
          and averages over{' '}
          <code className="text-dark-text-primary">(N, H, W)</code> per
          channel — every pixel of every image in the batch counts as a
          sample for that channel&apos;s curve. Pick the variant that matches
          your tensor layout; the error message when you don&apos;t is
          unhelpful.
        </p>
        <p>
          <strong className="text-term-amber">
            Don&apos;t add a bias to the Linear right before BatchNorm.
          </strong>{' '}
          BatchNorm subtracts the mean, which cancels any upstream constant.
          The <code className="text-dark-text-primary">β</code> parameter
          inside BN already plays the bias role. Pass{' '}
          <code className="text-dark-text-primary">
            nn.Linear(.., .., bias=False)
          </code>{' '}
          when you&apos;re feeding into a BN. Same for Conv2d. Otherwise
          you&apos;re training a parameter that gets zeroed on every forward
          pass, which is a fun thing to explain in code review.
        </p>
        <p>
          <strong className="text-term-amber">
            SyncBatchNorm for multi-GPU training.
          </strong>{' '}
          A plain BN on 8 GPUs with batch 8 each computes eight separate
          class curves of size 8 — somehow worse than a single curve of 64.
          Wrap with{' '}
          <code className="text-dark-text-primary">nn.SyncBatchNorm</code> to
          pool stats across devices. Not optional for anything serious on
          multi-GPU.
        </p>
        <p>
          <strong className="text-term-amber">
            Don&apos;t weight-decay{' '}
            <code className="text-dark-text-primary">γ, β</code>.
          </strong>{' '}
          BN&apos;s scale and bias don&apos;t control model capacity the way
          conv weights do. Decaying them toward zero is mild sabotage. Put
          them in a separate parameter group with zero weight decay — every
          modern optimizer config does this, and most beginner bugs come from
          forgetting it.
        </p>
        <p>
          <strong className="text-term-amber">
            Fusion at inference.
          </strong>{' '}
          At eval time BN is a fixed affine transform (the stats are frozen,
          the γ/β are frozen). Every production inference stack folds that
          affine into the preceding Conv or Linear weights, deleting the BN
          op entirely. Your model shrinks, latency drops, and the output
          doesn&apos;t change. That fusion only works because of the
          train/eval distinction — another reason the mode switch matters.
        </p>
      </Gotcha>

      {/* ── Three-layer code ─────────────────────────────────────── */}
      <Prose>
        <p>
          From scratch, then PyTorch. The NumPy version has to manage the
          running stats by hand, which is exactly what PyTorch hides inside
          <code> nn.BatchNorm1d</code>. Watch the bookkeeping; then watch it
          disappear.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure numpy · batch_norm.py"
        output={`training step 0: batch_mean=[0.15 -0.03]  running_mean=[0.015 -0.003]
training step 99: batch_mean=[0.42 0.18]   running_mean=[0.28  0.14]
eval output mean=[0.28 0.14]   (= running_mean, frozen)`}
      >{`import numpy as np

class BatchNorm1d:
    def __init__(self, features, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        self.running_mean = np.zeros(features)
        self.running_var = np.ones(features)
        self.momentum, self.eps = momentum, eps
        self.training = True

    def __call__(self, x):
        if self.training:
            mu = x.mean(axis=0)                            # per-feature batch mean
            var = x.var(axis=0)                            # per-feature batch var
            # Exponential running average — running_mean drifts toward batch_mean
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mu, var = self.running_mean, self.running_var  # frozen at eval
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

# Demo
rng = np.random.default_rng(0)
bn = BatchNorm1d(features=2)
for step in range(100):
    x = rng.normal(loc=[0.4, 0.2], scale=[1, 1], size=(32, 2))
    bn(x)

bn.training = False
test = rng.normal(loc=[0.4, 0.2], scale=[1, 1], size=(32, 2))
out = bn(test)
print("eval output mean=", np.round(out.mean(axis=0), 2),
      "  (= running_mean, frozen)")`}</CodeBlock>

      <Bridge
        label="the pattern to memorize"
        rows={[
          {
            left: 'self.training = True   # compute from batch',
            right: 'model.train()',
            note: 'batch stats in, update running stats',
          },
          {
            left: 'self.training = False  # use running',
            right: 'model.eval()',
            note: 'frozen stats, no updates',
          },
          {
            left: 'manual running_mean bookkeeping',
            right: 'registered as a "buffer" — saves with the model',
            note: 'saved to checkpoint, restored on load',
          },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch · batch_norm_pytorch.py"
        output={`torch.Size([32, 64])
conv layout: torch.Size([16, 3, 28, 28])`}
      >{`import torch
import torch.nn as nn

# For a dense feed-forward tensor (batch, features)
bn_1d = nn.BatchNorm1d(num_features=64)
x = torch.randn(32, 64)
print(bn_1d(x).shape)

# For a conv tensor (batch, channels, height, width)
bn_2d = nn.BatchNorm2d(num_features=3)         # num_features = channel count
img = torch.randn(16, 3, 28, 28)
print("conv layout:", bn_2d(img).shape)

# In a real model you'd combine them:
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, bias=False)   # bias=False → BN takes over
        self.bn   = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))`}</CodeBlock>

      <Callout variant="insight" title="a little tour of related norms">
        <strong>InstanceNorm</strong> is BN without the batch axis — each
        sample and channel is its own class of one. Used in style transfer.{' '}
        <strong>GroupNorm</strong> splits channels into groups and curves
        within each group. A middle ground between LayerNorm (one group) and
        InstanceNorm (C groups) that doesn&apos;t collapse at small batch
        sizes. <strong>RMSNorm</strong> — up next — is LayerNorm with the
        mean subtraction deleted, because someone looked at the math and
        asked whether that step was earning its keep. The takeaway:
        normalization is a design space, and LayerNorm / BatchNorm are two
        corners of it.
      </Callout>

      <Challenge prompt="Break BatchNorm with a batch of 1">
        <p>
          Take any classifier with BatchNorm. Put it in <code>.train()</code>{' '}
          mode and feed it one example at a time. The first thing that
          breaks: the per-feature variance of a one-sample batch is exactly
          zero, so the division by <code>√(σ² + ε)</code> is entirely
          determined by <code>ε</code>. Training goes sideways immediately.
        </p>
        <p className="mt-2">
          Now switch to <code>.eval()</code> and run the same inputs. The
          output is fine — BN uses the running stats, which don&apos;t care
          about batch size. Write a small script that asserts the two modes
          give different outputs at batch size 1, and explain why in one
          sentence.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: swap the BatchNorm for LayerNorm and rerun. The batch-of-one
          case suddenly works in training mode too. That is not magic; it is
          the whole reason transformers moved to per-sample normalization.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> BatchNorm grades on the
          curve across the batch, per feature — the perpendicular axis to
          LayerNorm. Two modes: training reads the live class, updates the
          running record; eval reads the running record and freezes. Switch
          with <code>model.train()</code> / <code>model.eval()</code>, or
          expect confusion. It breaks at small batch sizes because a class
          of four is a lousy curve and a class of one is no curve at all.
          It dominated vision from 2015 to 2018, still lives inside every
          convnet you&apos;ll load, and lost the sequence-modeling world
          specifically because of that small-batch weakness.
        </p>
        <p>
          <strong>Next up — RMS Normalization.</strong> One more
          normalization variant, and this one took a hard look at LayerNorm
          and decided it could skip a step. No mean subtraction. Just divide
          by root-mean-square and move on. Does it cost accuracy? In
          transformers, barely. Does it save compute? Yes — and that is why
          it is now the default inside Llama, PaLM, and most 2023+ large
          language models. We&apos;ll find out why dropping the mean subtract
          turned out to be free.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift',
            author: 'Ioffe, Szegedy',
            venue: 'ICML 2015 — the original paper',
            url: 'https://arxiv.org/abs/1502.03167',
          },
          {
            title: 'How Does Batch Normalization Help Optimization?',
            author: 'Santurkar, Tsipras, Ilyas, Madry',
            venue: 'NeurIPS 2018 — shows the internal-covariate-shift story is mostly wrong',
            url: 'https://arxiv.org/abs/1805.11604',
          },
          {
            title: 'Group Normalization',
            author: 'Yuxin Wu, Kaiming He',
            venue: 'ECCV 2018 — the batch-size-robust alternative',
            url: 'https://arxiv.org/abs/1803.08494',
          },
        ]}
      />
    </div>
  )
}
