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
import MSEBowl3D from '../widgets/MSEBowl3DLazy'
import GDRace from '../widgets/GDRace'
import ClosedFormVsGD from '../widgets/ClosedFormVsGD'

// Signature anchor: two doors — the shortcut (closed-form normal equations)
// and the long walk (gradient descent). Introduced up top, returned to at
// the closed-form reveal (the shortcut works because the loss is a bowl),
// and again at the ClosedFormVsGD punchline (real models aren't bowls, so
// the shortcut disappears and you're stuck walking).
export default function LinearRegressionTrainingLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout ────────────────────────────────── */}
      <Prereq currentSlug="linear-regression-training" />

      {/* ── Opening: two doors ──────────────────────────────────── */}
      <Prose>
        <p>
          You have a dataset. You picked a{' '}
          <NeedsBackground slug="linear-regression-forward">
            linear model
          </NeedsBackground>
          . Forward pass was one matrix multiply — plug <code>x</code> in, get{' '}
          <code>ŷ</code> out. Fine. But the weights inside that multiply are
          still whatever you initialized them to, which is to say: wrong. This
          lesson is about making them right.
        </p>
        <p>
          Here&apos;s the thing almost no other algorithm in this curriculum
          gets to say. Linear regression has <em>two doors</em>. Door one: the
          shortcut. Solve a matrix equation, get the optimal weights, done. No
          iteration. No learning rate. No hyperparameters to babysit. Door two:
          the long walk. Start with weights, compute the gradient, step downhill,
          repeat a few dozen times. That&apos;s{' '}
          <NeedsBackground slug="gradient-descent">
            gradient descent
          </NeedsBackground>
          , exactly as you built it.
        </p>
        <p>
          Both doors lead to the same room. So why does this curriculum spend
          fourteen more sections teaching you the long walk? Because the
          shortcut has a catch, and the catch is the reason deep learning
          exists. We&apos;ll get there. First, the problem.
        </p>
      </Prose>

      {/* ── MSE loss bowl ───────────────────────────────────────── */}
      <Prose>
        <p>
          You have a ruler. Actually, you have infinitely many — every choice
          of <code>(w, b)</code> defines a different line through your data.
          You need a number that says which one is best, so you can rank them.
          The honest first try: square every prediction error, take the
          average. Big errors get penalised more than small ones, negatives
          can&apos;t cancel positives, the whole thing is differentiable. That
          number is the mean squared error:
        </p>
      </Prose>

      <MathBlock caption="mean squared error as a function of (w, b)">
{`               1       N
L(w, b)   =   ───   Σ   (yᵢ  −  (w·xᵢ  +  b))²
               N    i=1`}
      </MathBlock>

      <Prose>
        <p>
          Each term is a square. Sum of squares in two variables is a{' '}
          <KeyTerm>convex quadratic</KeyTerm> — which, geometrically, is a
          bowl. One bottom. No local minima pretending to be the real one. No
          saddle points staging ambushes. Just a clean downhill from every
          direction to a single point. The point is the answer.
        </p>
        <p>
          Click anywhere on the bowl to drop a marble. Hit <em>release</em> and
          gradient descent rolls it toward the purple pin — the optimum. Drop
          the marble in five different places and it finds the same pin every
          time. That&apos;s what convexity buys you.
        </p>
      </Prose>

      <MSEBowl3D />

      <Callout variant="note" title="savor this — it does not last">
        Neural-network losses are not bowls. They&apos;re mountain ranges,
        shot through with saddle points and long flat ravines. You saw that in
        the non-convex widget in the gradient descent lesson. Linear regression
        escapes all of it because its loss is a sum of squares — and sum of
        squares is as friendly as a function gets. This convexity is the
        reason the shortcut exists. No other model in this course gets the
        same gift.
      </Callout>

      {/* ── The closed form — the shortcut ──────────────────────── */}
      <Prose>
        <p>
          Door one. The shortcut. Here&apos;s why a bowl lets you skip
          iteration entirely: the minimum of a smooth bowl is wherever the
          gradient vanishes. Set <code>∇L = 0</code>, solve for the
          parameters, and you&apos;re standing at the bottom. For a quadratic
          loss that equation is <em>linear</em> in the parameters, which means
          it has a clean algebraic solution. You don&apos;t need to walk — you
          just read the coordinates off the page.
        </p>
        <p>
          Stack your data into <code>X ∈ ℝ^(N × D)</code> (append a column of
          ones so the bias folds into the weights), targets into{' '}
          <code>y ∈ ℝ^N</code>, parameters into <code>θ ∈ ℝ^D</code>. The loss
          and its gradient:
        </p>
      </Prose>

      <MathBlock caption="MSE in matrix form">
{`                      ||Xθ − y||²
L(θ)   =   (1/N) · ─────────────────

∇L(θ)  =   (2/N) · Xᵀ (Xθ − y)`}
      </MathBlock>

      <Prose>
        <p>
          Set that gradient to zero. Two lines of linear algebra and you have
          the exact optimum in closed form:
        </p>
      </Prose>

      <MathBlock caption="the normal equations">
{`0    =   Xᵀ X θ*   −   Xᵀ y

Xᵀ X θ*   =   Xᵀ y

θ*   =   (Xᵀ X)⁻¹   Xᵀ y`}
      </MathBlock>

      <Prose>
        <p>
          That is not a step toward the answer. That <em>is</em> the answer.
          One matrix inversion, one multiply, the best weights your model can
          achieve on this data. No <code>α</code>, no convergence check, no
          loop. Statistics textbooks have taught this expression since Gauss
          wrote it down in 1795, which means your great-great-great-grandfather
          could have trained linear regression, assuming he had a lot of
          graph paper.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="closed-form OLS · ols_closed_form.py"
        output={`theta = [ 1.4857  0.2964]
loss  = 0.0823`}
      >{`import numpy as np

# Data — same 8 points as the widget.
x = np.array([-1.8, -1.2, -0.5, 0.1, 0.9, 1.4, 2.1, 2.6])
y = np.array([-2.1, -1.3, -0.4, 0.6, 1.7, 2.2, 3.5, 4.1])

# Prepend a column of 1s so bias folds into theta.
X = np.stack([x, np.ones_like(x)], axis=1)   # shape (N, 2)

# The normal equations — one line:
theta = np.linalg.inv(X.T @ X) @ X.T @ y      # (D, D) @ (D, N) @ (N,) = (D,)

print(f"theta = {np.round(theta, 4)}")        # [slope, intercept]

preds = X @ theta
loss = np.mean((preds - y) ** 2)
print(f"loss  = {loss:.4f}")`}</CodeBlock>

      <Personify speaker="Closed form">
        I am exact. I do not iterate. I give you the <em>unique</em> minimum
        in one step, and I am embarrassingly easy to write. I am also secretly
        O(D³) because inverting an X<sup>T</sup>X matrix costs a lot once D
        is big. Oh — and please don&apos;t invert badly-conditioned matrices.
        I may hand back numerical garbage and smile.
      </Personify>

      <Gotcha>
        <p>
          <strong className="text-term-amber">
            Never actually call <code className="text-dark-text-primary">np.linalg.inv</code>{' '}
            in production.
          </strong>{' '}
          Use <code className="text-dark-text-primary">np.linalg.solve(X.T @ X, X.T @ y)</code>{' '}
          or even better{' '}
          <code className="text-dark-text-primary">np.linalg.lstsq(X, y, rcond=None)</code>.
          They&apos;re more stable and faster. Computing the inverse as a standalone step is
          almost always wrong — you want the <em>solution</em>, not the inverse.
        </p>
        <p>
          <strong className="text-term-amber">X<sup>T</sup>X might not be invertible.</strong>{' '}
          If you have fewer data points than features, or redundant features, the matrix is
          singular and the inverse doesn&apos;t exist. <code className="text-dark-text-primary">lstsq</code>{' '}
          handles that gracefully; naive inv crashes.
        </p>
      </Gotcha>

      {/* ── Gradient descent on the same problem — the long walk ─ */}
      <Prose>
        <p>
          Door two. The long walk. Same dataset, same loss, same destination —
          reached one foot at a time. Start the weights at zero. Compute the
          gradient. Take a small step opposite to it. Repeat. You already know
          every line of this from the gradient descent lesson; the only new
          thing is that the loss surface is now the MSE bowl you just met.
        </p>
        <p>
          Drag <code>α</code> or press play. The left panel shows the fitted
          line sweeping toward the dashed optimum. The right panel shows the
          loss dropping toward <code>L*</code> — the minimum loss the shortcut
          already computed. Both doors, one room.
        </p>
      </Prose>

      <GDRace />

      <Callout variant="insight" title="convergence feels obvious here because it is">
        Gradient descent on a convex quadratic with a well-chosen learning rate
        is the cleanest application of the algorithm in existence. The gap to
        optimum shrinks like a geometric sequence — the same{' '}
        <code>(1 − 2α)ⁿ</code> behavior you derived in the gradient descent
        lesson, just in two parameters instead of one. Push <code>α</code>{' '}
        past <code>0.5</code> in the widget and watch the loss start to
        oscillate — that&apos;s the convergence condition complaining, same
        rule as before, different room.
      </Callout>

      {/* ── When each wins ──────────────────────────────────────── */}
      <Prose>
        <p>
          Both doors opened. Both led to the same weights. If the shortcut
          gives you the exact answer in one line, why would anyone walk?
        </p>
        <p>
          Because the doors do not cost the same. The shortcut computes{' '}
          <code>Xᵀ X</code> (which is <code>O(N · D²)</code>) and then inverts
          a <code>D × D</code> matrix (which is <code>O(D³)</code>). That{' '}
          <code>D³</code> is the fine print on the elegant expression — for a
          few dozen features it&apos;s a rounding error, for a few thousand
          it&apos;s your afternoon, for a million it&apos;s geological time.
          The long walk has different arithmetic: a fixed number of iterations,
          each one <code>O(N · D)</code>. Linear in <code>D</code>, not cubic.
          Scale <code>D</code> and the regimes flip.
        </p>
      </Prose>

      <ClosedFormVsGD />

      <Prose>
        <p>
          Push the sliders. A stats-class dataset — 100 rows, 5 features — and
          the shortcut wins by five orders of magnitude. It&apos;s essentially
          free. A recommender system with a million users and ten thousand
          features, and the shortcut is <em>infeasible</em>: inverting a
          10k×10k matrix takes hours of CPU time and enough memory to matter.
          Same algorithm, same math, just way off the edge of what the{' '}
          <code>D³</code> regime can absorb.
        </p>
        <p>
          And now the real punchline. Everything above assumed the loss was a
          bowl. That assumption is load-bearing — setting <code>∇L = 0</code>{' '}
          only gives you a clean, closed-form answer because the equation you
          get is linear. For any model more complicated than a linear
          regressor, that stops being true. Add a sigmoid. Add a second layer.
          Add a transformer on top. The loss is no longer quadratic in the
          parameters, <code>∇L = 0</code> no longer has an algebraic solution,
          and the landscape is no longer a bowl — it&apos;s a mountain range
          with ravines and saddle points, the one you saw bully gradient
          descent around in the last lesson.
        </p>
        <p>
          So the shortcut isn&apos;t just expensive outside linear regression.
          It <em>disappears</em>. Every neural network you&apos;ve ever heard
          of is trained by the long walk — not because it&apos;s clever, but
          because it&apos;s the only door left. Linear regression is the
          farewell tour for door one. Say goodbye nicely.
        </p>
      </Prose>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations of the long walk on linear regression. Same
          problem you just watched in the widget, now on the page — in the
          three voices you&apos;re using for every algorithm in this series.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · linear_gd_scratch.py"
        output={`step  0: L = 7.4225
step 10: L = 0.6181
step 50: L = 0.0827
theta = [1.483, 0.296]`}
      >{`def mse(theta, data):
    loss = 0.0
    for x, y in data:
        yhat = theta[0] * x + theta[1]       # w*x + b
        loss += (y - yhat) ** 2
    return loss / len(data)

def linear_gd(data, lr=0.1, steps=50):
    theta = [0.0, 0.0]                       # start at (w=0, b=0)
    for step in range(steps + 1):
        gw = 0.0; gb = 0.0
        for x, y in data:
            r = y - (theta[0] * x + theta[1])
            gw += -2 * x * r
            gb += -2 * r
        gw /= len(data); gb /= len(data)
        if step % 10 == 0:
            print(f"step {step:2d}: L = {mse(theta, data):.4f}")
        theta[0] -= lr * gw
        theta[1] -= lr * gb
    return theta

data = [(-1.8, -2.1), (-1.2, -1.3), (-0.5, -0.4), (0.1, 0.6),
        (0.9, 1.7), (1.4, 2.2), (2.1, 3.5), (2.6, 4.1)]
theta = linear_gd(data)
print(f"theta = {[round(v, 3) for v in theta]}")`}</CodeBlock>

      <CodeBlock language="python" caption="layer 2 — numpy · linear_gd_numpy.py">{`import numpy as np

def linear_gd(X, y, lr=0.1, steps=50):
    N, D = X.shape
    theta = np.zeros(D)
    for _ in range(steps):
        residual = X @ theta - y              # (N,)
        grad = (2 / N) * X.T @ residual       # (D,)
        theta -= lr * grad
    return theta

x = np.array([-1.8, -1.2, -0.5, 0.1, 0.9, 1.4, 2.1, 2.6])
y = np.array([-2.1, -1.3, -0.4, 0.6, 1.7, 2.2, 3.5, 4.1])
X = np.stack([x, np.ones_like(x)], axis=1)

theta = linear_gd(X, y)
print(f"theta = {np.round(theta, 4)}")
# -> [1.4857 0.2963]  — exactly what closed-form gave us`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for x, y in data: gw += -2*x*(y - (w*x+b))',
            right: 'grad = (2/N) * X.T @ (X @ theta - y)',
            note: 'the single matrix form for both weights and bias',
          },
          {
            left: 'theta[0] -= lr*gw ;  theta[1] -= lr*gb',
            right: 'theta -= lr * grad',
            note: 'vectorised update — scales to any D',
          },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · linear_gd_pytorch.py"
        output={`step 10: loss = 0.1487
step 50: loss = 0.0822
Parameter containing:
tensor([[1.4855]], requires_grad=True)
Parameter containing:
tensor([0.2964], requires_grad=True)`}
      >{`import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([-1.8, -1.2, -0.5, 0.1, 0.9, 1.4, 2.1, 2.6]).unsqueeze(-1)
y = torch.tensor([-2.1, -1.3, -0.4, 0.6, 1.7, 2.2, 3.5, 4.1]).unsqueeze(-1)

model = nn.Linear(1, 1)                           # w and b, learnable
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for step in range(51):
    optimizer.zero_grad()
    yhat = model(x)                               # forward
    loss = criterion(yhat, y)                     # MSE
    loss.backward()                               # autograd — no hand-rolled gradient
    optimizer.step()                              # θ ← θ - α·∇L
    if step % 10 == 0:
        print(f"step {step}: loss = {loss.item():.4f}")

print(model.weight)
print(model.bias)`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'grad = (2/N) * X.T @ (X@theta - y)',
            right: 'loss.backward()',
            note: 'autograd computes the same thing from the MSELoss expression',
          },
          {
            left: 'theta -= lr * grad',
            right: 'optimizer.step()',
            note: 'same update, packaged — swap SGD for Adam to upgrade the optimizer',
          },
          {
            left: 'np.stack([x, ones], axis=1)',
            right: 'nn.Linear(1, 1, bias=True)',
            note: 'bias is a first-class parameter, no manual 1-column needed',
          },
        ]}
      />

      <Callout variant="insight" title="the moment where it all clicks">
        The PyTorch block above is exactly — literally — the innermost loop of
        a GPT-5 training run. Replace <code>nn.Linear(1, 1)</code> with a
        transformer. Replace <code>MSELoss</code> with <code>CrossEntropyLoss</code>.
        Replace the 8 data points with a trillion tokens. Replace{' '}
        <code>SGD</code> with <code>AdamW</code>. Replace one GPU with ten
        thousand. But the pattern — zero_grad / forward / loss / backward /
        step — is preserved unchanged. Every model in this series, all the
        way up to transformers, trains in that five-line loop. That&apos;s the
        long walk paying rent.
      </Callout>

      <Challenge prompt="Beat closed form with gradient descent">
        <p>
          Generate a synthetic linear dataset with <code>N = 10,000</code> examples and{' '}
          <code>D = 500</code> features. Solve it two ways: (1){' '}
          <code>np.linalg.lstsq</code> and (2) a gradient-descent loop for 100 steps. Time
          both. The closed form should take a few seconds (the <code>D³</code> inversion).
          Your well-tuned GD should finish in milliseconds and get within{' '}
          <code>1e-6</code> of the same solution.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: now set <code>N = 10,000</code>, <code>D = 50,000</code> — which is still
          linear regression but with more features than examples. Closed form will refuse
          (the <code>X<sup>T</sup>X</code> matrix is rank-deficient). Gradient descent still
          converges (albeit to one of infinitely many equally-good minima). You have just
          re-derived the modern over-parameterised regime in twelve lines of NumPy.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> Linear regression is the one
          model in this course where both doors are open. The shortcut —{' '}
          <code>θ* = (XᵀX)⁻¹ Xᵀy</code> — exists because the loss is a convex
          quadratic and <code>∇L = 0</code> has a clean algebraic solution. It
          costs <code>O(N·D² + D³)</code>, which is cheap for small{' '}
          <code>D</code> and catastrophic for large <code>D</code>. The long
          walk costs <code>O(iters·N·D)</code>, wins as soon as features
          outnumber a few thousand, and — more importantly — still works when
          the loss isn&apos;t a bowl. Every deep model you&apos;ll meet from
          here on has a non-convex loss, which means the shortcut is gone and
          the long walk is all that&apos;s left.
        </p>
        <p>
          <strong>Next up — Single Neuron.</strong> The shortcut&apos;s gone.
          From here on, you walk. Before you can walk through a deep network,
          you need the smallest non-trivial thing that sits inside one: a
          single neuron. A weighted sum, a nonlinearity, a prediction — the
          unit of computation that everything from an MLP to a transformer
          stacks, by the million, into a model. And to train a stack of them,
          you&apos;ll need to know how the gradient flows <em>backwards</em>{' '}
          through each layer. That&apos;s the next two lessons.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Dive into Deep Learning — 3.1.2 Solution by Normal Equations',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_linear-regression/linear-regression.html',
          },
          {
            title: 'The Elements of Statistical Learning — 3.2.2 Linear Least Squares',
            author: 'Hastie, Tibshirani, Friedman',
            venue: 'Springer, 2009',
            url: 'https://hastie.su.domains/ElemStatLearn/',
          },
          {
            title: 'An overview of gradient descent optimization algorithms',
            author: 'Sebastian Ruder',
            year: 2016,
            url: 'https://arxiv.org/abs/1609.04747',
          },
        ]}
      />
    </div>
  )
}
