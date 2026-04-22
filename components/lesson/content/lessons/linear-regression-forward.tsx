import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import Prereq from '../Prereq'
import NeedsBackground from '../NeedsBackground'
import {
  Prose,
  Callout,
  Personify,
  Bridge,
  Challenge,
  References,
  KeyTerm,
} from '../primitives'
import LineFit2D from '../widgets/LineFit2D'
import MatmulForward from '../widgets/MatmulForward'

// One flowing narrative. Anchor: the ruler through a point cloud. The
// reader is picking up a ruler, laying it across a scatter of data, and
// reading y for each x. The forward pass is that read. The vectorized
// form is reading every y at once instead of point by point. The anchor
// is threaded at the opening, at the y = wx + b reveal, and at the
// batched/matrix section.
export default function LinearRegressionForwardLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout (none declared; renders as "start here") ── */}
      <Prereq currentSlug="linear-regression-forward" />

      {/* ── Opening: anchor establishment ───────────────────────── */}
      <Prose>
        <p>
          You have a scatter of points on a plane. A constellation, if you
          squint. Each point is a house — an (x, y) pair — where x is the
          square footage and y is the price someone paid for it. There is no
          formula. No rule. Just twenty dots where twenty houses happened.
        </p>
        <p>
          Now pick up a ruler. Lay it across the cloud. Wiggle it until it
          sits more-or-less through the middle of the points. You&apos;ve
          just done linear regression in your hands — a model whose entire
          being is a straight edge, with a slope and a vertical offset, and
          whose prediction for any new x is simply: look at the ruler above
          that x and read off the y.
        </p>
        <p>
          That&apos;s the whole thing. The <KeyTerm>forward pass</KeyTerm> of
          a linear regression is laying the ruler down and reading the y
          values. Training — which comes next lesson — is figuring out which
          ruler to pick in the first place. This lesson stays inside the
          read. We&apos;ll go from one point at a time to all the points at
          once, and by the end the same gesture will scale to every dense
          layer in every neural network ever shipped.
        </p>
      </Prose>

      {/* ── y = wx + b: the ruler's two knobs ───────────────────── */}
      <Prose>
        <p>
          Two numbers fully describe a ruler on a plane. The slope — how
          steeply it tilts — and where it crosses the y-axis. Those two
          numbers are the entire model.
        </p>
      </Prose>

      <MathBlock caption="univariate linear regression">
{`ŷ   =   w · x   +   b

  • x  — the input (one feature)
  • w  — the weight: the ruler's slope
  • b  — the bias: where the ruler crosses the y-axis
  • ŷ  — the prediction: the y your ruler reads off at x`}
      </MathBlock>

      <Prose>
        <p>
          w and b are the <NeedsBackground slug="gradient-descent">parameters</NeedsBackground>{' '}
          — the two dials from the previous lesson, now wearing geometry
          costumes. Twist w and the ruler tilts. Slide b and it lifts or
          drops. Every straight ruler you could ever lay across this plane
          corresponds to exactly one (w, b) pair. Picking a model is picking
          a point in this two-dimensional space of rulers.
        </p>
        <p>
          Below is a scatter of twenty noisy points from an underlying
          linear relationship. Drag <code>w</code> and <code>b</code> — watch
          the ruler pivot and slide — and try to shrink the MSE readout by
          hand. The residual sticks are how much each point disagrees with
          the ruler&apos;s reading.
        </p>
      </Prose>

      <LineFit2D />

      <Callout variant="note" title="what the MSE is telling you">
        The mean squared error (MSE) is the average of <code>(y − ŷ)²</code>{' '}
        across the dataset — the length of each residual stick, squared, then
        averaged. Longer sticks cost more than short ones, and they cost
        disproportionately more because of the square, which is why one wild
        outlier can drag the whole ruler toward it. We&apos;ll treat MSE as a{' '}
        <NeedsBackground slug="gradient-descent">loss</NeedsBackground> next
        lesson and find the exact (w, b) that minimises it. For now, just
        notice: a minimum exists. You can feel it when you stop being able to
        make the number any smaller.
      </Callout>

      <Personify speaker="Linear model">
        I&apos;m a ruler. I can&apos;t curve. You cannot make me fit a
        parabola, a sine wave, or a smile. But I am <em>very good</em> at
        being wrong in predictable, diagnosable ways — and my two knobs mean
        anything about me. w<sub>sqft</sub> is literally &ldquo;dollars per
        square foot.&rdquo; I am what every intro stats class teaches and
        what every ML engineer still reaches for when they need a
        sanity-check baseline before anything fancier.
      </Personify>

      {/* ── The naive forward: one point at a time ──────────────── */}
      <Prose>
        <p>
          Fine — we have a ruler. How do we actually read it? The obvious
          way: walk along the x-axis, point by point, and for each one
          compute <code>w · x + b</code>. Five houses, five multiplications,
          five additions. Written as a loop, it&apos;s the thing you already
          know how to write.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="naive forward — one point at a time · one_at_a_time.py"
        output="ŷ = [140.0, 230.0, 320.0, 410.0, 500.0]"
      >{`# A single feature: square footage (in units of 1000). Five houses.
X = [1.2, 2.1, 3.0, 3.9, 4.8]
w = 100.0        # dollars per (1000 sqft)
b = 20.0         # base price

preds = []
for x in X:
    yhat = w * x + b         # read the ruler above x
    preds.append(yhat)

print("ŷ =", preds)`}</CodeBlock>

      <Prose>
        <p>
          No magic. Python walks the list, does the arithmetic, yields five
          numbers. This works for five houses. It also works for a million
          houses, if you&apos;re patient enough to wait out a Python loop.
          You will not be.
        </p>
      </Prose>

      {/* ── Multi-feature: one weight per feature ───────────────── */}
      <Prose>
        <p>
          Real problems have more than one feature. A house has square
          footage, bedrooms, age, neighbourhood, lot size. The ruler
          generalises: one slope per feature, still one offset. The
          arithmetic is the same, just with more terms.
        </p>
      </Prose>

      <MathBlock caption="multivariate linear regression">
{`ŷ   =   w₁·x₁  +  w₂·x₂  +  ...  +  w_D·x_D   +   b

     =   Σᵢ wᵢ·xᵢ   +   b          D features total`}
      </MathBlock>

      <Prose>
        <p>
          With D features the ruler doesn&apos;t fit on a whiteboard anymore
          — it&apos;s a flat surface (a hyperplane) sitting inside a
          D+1-dimensional space. You can&apos;t draw it and you wouldn&apos;t
          want to. But the operation is mechanically identical: multiply
          each feature by its weight, add them up, add the bias.
        </p>
      </Prose>

      {/* ── The reveal: dot product, then batched matrix multiply ─ */}
      <Prose>
        <p>
          Here&apos;s the reframing that earns the lesson. Stack the inputs
          into a vector <code>x</code> and the weights into a vector{' '}
          <code>w</code>, and that whole sum{' '}
          <code>w₁x₁ + w₂x₂ + … + w_Dx_D</code> is just a <em>dot product</em>{' '}
          — one of the most common operations in all of numerical computing,
          and one that modern hardware will murder in its sleep.
        </p>
        <p>
          Now do it for every house at once. Stack all N examples into a
          matrix <code>X</code> of shape <code>(N × D)</code> — each row is a
          house, each column a feature — and one matrix-vector product
          reads the whole constellation through the ruler in a single
          stroke:
        </p>
      </Prose>

      <MathBlock caption="batched linear regression — the matrix form">
{`                 X         ·      w       +    b    =      ŷ
              (N × D)         (D,)         scalar      (N,)

ŷ₀   =   w₁·X₀₁  +  w₂·X₀₂  +  ...  +  w_D·X₀_D   +   b
ŷ₁   =   w₁·X₁₁  +  w₂·X₁₂  +  ...  +  w_D·X₁_D   +   b
...
ŷ_N  =   w₁·X_N1 +  w₂·X_N2 +  ...  +  w_D·X_N_D  +   b`}
      </MathBlock>

      <Prose>
        <p>
          This is what the rest of the field means by &ldquo;vectorizing the
          forward pass.&rdquo; Instead of laying the ruler down and reading
          each point one at a time, you press the ruler through the entire
          scatter in one motion — every prediction falls out simultaneously.
          The Python loop dies. The arithmetic happens in compiled code on
          contiguous memory, and on a GPU that matrix multiply becomes the
          single most optimized operation on the machine.
        </p>
        <p>
          Hover any row of X in the widget below. The bottom panel shows{' '}
          <em>exactly</em> the dot product that produces that row&apos;s
          prediction. This is a toy housing dataset — three features (size,
          bedrooms, age), five houses, weights hand-picked so the numbers
          land somewhere believable.
        </p>
      </Prose>

      <MatmulForward />

      <Callout variant="insight" title="this is also what one layer of a neural net does">
        Every <code>nn.Linear(in_features, out_features)</code> in PyTorch is
        doing this operation — just with <code>out_features</code> parallel
        outputs instead of one. A dense hidden layer with 128 neurons? It is
        128 linear regressors in parallel, each with its own weight vector,
        stacked into a weight <em>matrix</em> <code>W</code> of shape{' '}
        <code>D × 128</code>. Then a non-linearity goes on top. Pop the
        non-linearity off and you have 128 linear regressions reading the
        same constellation through 128 different rulers. The forward pass
        you just saw <em>is</em> the forward pass of a 1-neuron linear
        layer — every dense layer in a deep net is this operation at
        industrial scale.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations, same prediction, each shorter than the last.
          The abstractions keep changing; the operation never does.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · linear_forward_scratch.py"
        output="ŷ = [ 210.0, 335.0, 516.0, 190.0, 315.0 ]"
      >{`def linear_forward(X, w, b):
    preds = []
    for row in X:
        # The dot product, written out with no abstractions.
        yhat = sum(row[i] * w[i] for i in range(len(w))) + b
        preds.append(yhat)
    return preds

# 5 houses · 3 features (sqft/1000, bedrooms, age)
X = [
    [1.4, 2, 10],
    [2.1, 3, 5],
    [3.2, 4, 2],
    [1.8, 2, 20],
    [2.6, 3, 15],
]
w = [100, 25, -2]      # dollars-per-sqft, per-bedroom, per-year-old
b = 20                 # base price

preds = linear_forward(X, w, b)
print("ŷ =", [round(p, 1) for p in preds])`}</CodeBlock>

      <Prose>
        <p>
          This is the definition made executable. One Python loop over
          houses, one dot product inside, one addition for the bias. If you
          squint, you can see the ruler sliding across the x-axis, pausing
          at each house, reading the y.
        </p>
        <p>
          Now the upgrade. NumPy lets you write the dot-product-over-batch as
          a single operator, and the inner arithmetic runs in compiled C
          over whole arrays. Same computation. No Python loop.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · linear_forward_numpy.py"
        output="[210. 335. 516. 190. 315.]"
      >{`import numpy as np

def linear_forward(X, w, b):
    return X @ w + b               # one line, arbitrary batch, arbitrary feature count

X = np.array([
    [1.4, 2, 10],
    [2.1, 3, 5],
    [3.2, 4, 2],
    [1.8, 2, 20],
    [2.6, 3, 15],
])
w = np.array([100.0, 25.0, -2.0])
b = 20.0

print(linear_forward(X, w, b))`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'sum(row[i] * w[i] for i in range(D))',
            right: 'X @ w       # or np.dot(X, w)',
            note: 'one operator for N×D @ D → N predictions',
          },
          {
            left: 'for row in X: ...',
            right: 'batch is implicit',
            note: 'numpy does the loop over N in compiled C',
          },
        ]}
      />

      <Prose>
        <p>
          The <code>+ b</code> at the end is secretly doing something: the
          left-hand side is an N-vector and <code>b</code> is a scalar, so
          NumPy <em>broadcasts</em> b across every element. That&apos;s
          fine, expected, and also a rich source of bugs — more on that in
          the Gotcha section.
        </p>
        <p>
          Real neural networks chain thousands of operations, and writing
          out the gradient of each one by hand is a job for a person with
          very few deadlines. PyTorch replaces the manual bookkeeping with
          autograd. The forward pass looks the same — the framework just
          also remembers what it did, so it can differentiate later.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · linear_forward_pytorch.py"
        output={`model's prediction: tensor([210.0000, 335.0000, 516.0000, 190.0000, 315.0000],
       grad_fn=<AddBackward0>)`}
      >{`import torch
import torch.nn as nn

# nn.Linear(in_features=3, out_features=1) IS linear regression.
# It owns a weight matrix (1, 3) and a bias vector (1,), both learnable.
model = nn.Linear(in_features=3, out_features=1, bias=True)

# Load our hand-picked weights directly for the demo.
with torch.no_grad():
    model.weight.copy_(torch.tensor([[100.0, 25.0, -2.0]]))
    model.bias.copy_(torch.tensor([20.0]))

X = torch.tensor([
    [1.4, 2, 10],
    [2.1, 3, 5],
    [3.2, 4, 2],
    [1.8, 2, 20],
    [2.6, 3, 15],
])

preds = model(X).squeeze(-1)
print("model's prediction:", preds)`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'X @ w + b',
            right: 'nn.Linear(in_features=D, out_features=1)',
            note: 'packaged with learnable params + autograd',
          },
          {
            left: 'w = np.array([...])  # you own the array',
            right: 'model.weight  # learned during training',
            note: 'the Module keeps its parameters for you',
          },
          {
            left: 'one regressor',
            right: 'nn.Linear(D, K)  # K regressors in parallel',
            note: 'scale up by asking for more output features',
          },
        ]}
      />

      <Callout variant="insight" title="scaling up to 128 or 4096 outputs">
        When we get to MLPs we&apos;ll stack <code>nn.Linear(3, 64)</code>{' '}
        then <code>nn.ReLU()</code> then <code>nn.Linear(64, 1)</code>. The
        first layer is 64 linear regressions on the same input. The second
        is 1 linear regression on the outputs of those 64 (after ReLU has
        bent them). The whole &ldquo;deep learning&rdquo; thing is: stacks
        of linear regressions, alternated with non-linearities. That&apos;s
        it. You now know the forward pass of every layer of every
        feed-forward network.
      </Callout>

      {/* ── Gotchas: the two real traps ─────────────────────────── */}
      <Prose>
        <p>
          Two traps lie in wait for everyone writing this for the first
          time. Both are avoidable; both will eat an afternoon if they
          aren&apos;t.
        </p>
      </Prose>

      <Callout variant="warn" title="shape discipline and feature scaling">
        <p>
          <strong>Broadcasting is convenient until it isn&apos;t.</strong>{' '}
          If <code>X</code> is <code>(N, D)</code> and <code>w</code> is{' '}
          <code>(D,)</code>, then <code>X @ w</code> gives you{' '}
          <code>(N,)</code>. If you accidentally make <code>w</code> shape{' '}
          <code>(D, 1)</code>, you get <code>(N, 1)</code> — same numbers,
          different shape, and every downstream operation that assumed a
          1-D vector now does something subtly wrong. Losses stop matching.
          Losses stop being numbers. Print <code>.shape</code> at every
          boundary the first time you wire a model.
        </p>
        <p>
          <strong>Features at different scales are a problem.</strong>{' '}
          Square footage lives in the thousands; bedroom count lives in 2-4;
          house age lives in 0-100. Any training algorithm that uses
          gradients (and that&apos;s all of them) will effectively see
          square footage a thousand times more than bedroom count — not
          because it&apos;s more important, but because it&apos;s bigger.
          Standardise your features (subtract mean, divide by std) before
          training. The forward pass doesn&apos;t care; training will care a
          lot, and we&apos;ll come back to this next lesson.
        </p>
      </Callout>

      {/* ── Consolidation challenge ─────────────────────────────── */}
      <Challenge prompt="Manual regression, one feature">
        <p>
          Grab a tiny dataset by hand — say,{' '}
          <code>[(0, 1.1), (1, 2.9), (2, 5.1), (3, 7.0), (4, 9.1)]</code>.
          Eyeball it: the relationship is roughly <code>y = 2x + 1</code>.
          Write the pure-Python <code>linear_forward</code>, try{' '}
          <code>w = 2.0, b = 1.0</code>, compute the predictions, and print
          the residuals. Can you tweak <code>w</code> and <code>b</code> by
          hand to drive the MSE below <code>0.02</code>?
        </p>
        <p className="mt-2 text-dark-text-muted">
          You&apos;re doing by hand what next lesson&apos;s algorithm does
          mechanically — nudging the ruler until the sticks shrink.
        </p>
      </Challenge>

      {/* ── Takeaways + cliffhanger ─────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> A linear regression is a
          ruler through a point cloud; the forward pass is the ruler&apos;s
          reading at each x. Numerically, one prediction is a dot product of
          the feature vector with a weight vector, plus a bias. Batched, the
          whole dataset becomes one matrix multiply: <code>X @ w + b</code>,
          shapes <code>(N × D) · (D,) = (N,)</code>. Every dense layer in
          every neural network is this operation at larger scale, with a
          non-linearity on top.
        </p>
        <p>
          <strong>Next up — Linear Regression (Training).</strong> You can
          lay a ruler down. But <em>which</em> ruler is the best one? How
          do you find it? There are two answers. One is a closed-form
          formula from pure linear algebra that solves the problem exactly
          in a single matrix inversion — elegant, but it gives up the moment
          your dataset stops fitting in memory. The other is gradient
          descent on the MSE, the same algorithm you already know, now
          wearing a new loss. One works everywhere the other doesn&apos;t,
          and the contrast is the entire reason the rest of deep learning
          looks the way it does.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Dive into Deep Learning — 3.1 Linear Regression',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_linear-regression/linear-regression.html',
          },
          {
            title: 'The Elements of Statistical Learning — 3.2 Linear Regression Models',
            author: 'Hastie, Tibshirani, Friedman',
            venue: 'Springer, 2009',
            url: 'https://hastie.su.domains/ElemStatLearn/',
          },
          {
            title: 'Pattern Recognition and Machine Learning — 3.1 Linear Basis Function Models',
            author: 'Christopher Bishop',
            venue: 'Springer, 2006',
            url: 'https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf',
          },
        ]}
      />
    </div>
  )
}
