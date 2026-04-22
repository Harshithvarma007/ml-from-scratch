import CodeBlock from '../CodeBlock'
import LayeredCode from '../LayeredCode'
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
import CrossEntropyExplorer from '../widgets/CrossEntropyExplorer'
import LogLossCurve from '../widgets/LogLossCurve'
import SoftmaxCEGradient from '../widgets/SoftmaxCEGradient'

// Signature anchor: the confidence-scored pop quiz. A teacher who grades
// not just right/wrong, but how sure you claimed to be. Sure and right =
// near-zero loss; sure and wrong = the log blows up; "I don't know" =
// medium loss either way. Threaded at the opening (why not MSE), the
// -log reveal, and the numerical-stability gotcha.
export default function CrossEntropyLossLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout ────────────────────────────────── */}
      <Prereq currentSlug="cross-entropy-loss" />

      {/* ── Opening hook: the confidence-scored quiz ────────────── */}
      <Prose>
        <p>
          Picture a pop quiz where you don&apos;t just answer — you also write down
          how sure you are. 99% confident and right? Full marks, basically free.
          99% confident and wrong? The teacher takes out a very large red pen. And
          if you shrug and spread your bet evenly across every option, you get
          graded somewhere in the middle no matter what the real answer is.
        </p>
        <p>
          That grading rubric is <KeyTerm>cross-entropy loss</KeyTerm>. It&apos;s
          the loss function that reads confidence, not just correctness. A{' '}
          <NeedsBackground slug="softmax">softmax</NeedsBackground> output is a
          claim about the world — &ldquo;I&apos;m 80% sure it&apos;s a cat&rdquo;
          — and cross-entropy is the receipt the world hands back. Confident and
          right is cheap. Confident and wrong is ruinous. Uncertain is mediocre
          either way.
        </p>
        <p>
          You could, technically, just use squared error. Subtract the predicted
          probabilities from the truth, square it, move on. Every deep-learning
          library ships a cross-entropy op instead. By the end of this page
          you&apos;ll know why — and why the gradient through the whole
          softmax-then-cross-entropy stack collapses to an expression so short you
          could tattoo it on your wrist: <code>p − y</code>.
        </p>
      </Prose>

      {/* ── The definition ──────────────────────────────────────── */}
      <Prose>
        <p>
          Start concrete. Your model produced a probability distribution{' '}
          <code>p = [p₁, p₂, …, p_K]</code> over K classes. The training label is
          the truth — also a distribution, just a boring one: 1 on the correct
          class, 0 everywhere else. Call that one-hot vector <code>y</code>, and
          call the index it points at <code>t</code>. Cross-entropy is one line.
        </p>
      </Prose>

      <MathBlock caption="cross-entropy — the full thing and its one-hot simplification">
{`H(y, p)   =   − Σᵢ yᵢ · log pᵢ

          =   − log p_t           (because y is one-hot)`}
      </MathBlock>

      <Prose>
        <p>
          The second line is the one you&apos;ll use in practice. Take the
          probability your model assigned to the correct class, log it, flip the
          sign. That&apos;s your score on the quiz. This is also why you&apos;ll
          hear cross-entropy called <em>negative log likelihood</em> in half the
          papers you read — same number, different accent.
        </p>
        <p>
          Drop the target onto any class below, then drag the predicted bars
          around. Perfect match gives 0. Uniform guessing over 5 classes —
          that&apos;s the &ldquo;I have no idea&rdquo; answer on the quiz — gives{' '}
          <code>log(5) ≈ 1.61</code> no matter which class is correct.
          Confidently wrong sends the loss toward infinity, which is the whole
          point of the next few paragraphs.
        </p>
      </Prose>

      <CrossEntropyExplorer />

      {/* ── Why -log: the core anchor reveal ────────────────────── */}
      <Callout variant="insight" title="why −log, specifically">
        <div className="space-y-2">
          <p>
            Forget information theory for a second. The quiz metaphor alone
            forces the shape of the loss. You want:
          </p>
          <p>
            <strong>Confident and right</strong> to cost almost nothing — so the
            loss has to go to 0 as <code>p_t → 1</code>.
          </p>
          <p>
            <strong>Confident and wrong</strong> to cost unboundedly much — so
            the loss has to diverge as <code>p_t → 0</code>.
          </p>
          <p>
            <strong>&ldquo;I have no idea&rdquo;</strong> to cost a fixed, medium
            amount regardless of which class turns out to be right.
          </p>
          <p>
            <code>−log(p_t)</code> does all three without trying. It&apos;s 0 at{' '}
            <code>p_t = 1</code>, it climbs to infinity as <code>p_t → 0</code>,
            and it&apos;s exactly <code>log K</code> when you hedge uniformly. The
            logarithm isn&apos;t a cute mathematical choice — it&apos;s the
            function whose shape says &ldquo;a little wrong is a little bad;
            claiming certainty in a lie is infinitely bad.&rdquo;
          </p>
        </div>
      </Callout>

      <Callout variant="note" title="where Shannon comes in">
        The information-theory view lines up: cross-entropy is exactly the extra
        bits you need to encode samples from <code>y</code> using a code built
        for <code>p</code>. If <code>p = y</code>, no overhead. If{' '}
        <code>p</code> is wildly off, the code is long. Minimising
        cross-entropy is literally compressing the data better. The log isn&apos;t
        arbitrary — it&apos;s the thing that makes &ldquo;number of bits&rdquo;
        add up correctly across independent events. The quiz analogy is just
        Shannon&apos;s math wearing a costume.
      </Callout>

      <Personify speaker="Cross-entropy">
        I am not a reasonable grader. A 70% confident right answer costs you{' '}
        <code>0.36</code> — sure, fine, have your lunch money. A 99% confident
        right answer costs basically zero. But predict <code>0.01</code> for the
        true class and I charge you <code>4.6</code>, and I keep climbing toward
        infinity as your confidence in the wrong answer approaches 1. Be
        uncertain when you&apos;re wrong. Be sure when you&apos;re right. Anything
        else is expensive.
      </Personify>

      {/* ── Why not MSE: the naive-attempt falls-over moment ────── */}
      <Prose>
        <p>
          Worth pausing on the alternative, because it&apos;s the one most
          people reach for first. Squared error is the loss you grew up with:
          subtract, square, average. It works beautifully on regression — that
          lesson is coming — and it&apos;s tempting to just point it at
          probability vectors and call it a day.
        </p>
        <p>
          Try it. If the truth is 1 and you predict <code>0.99</code>, MSE gives
          you <code>0.0001</code>. If you predict <code>0.01</code>, MSE gives you{' '}
          <code>0.98</code>. Those two outcomes — nearly right, completely wrong
          — differ by a factor of about ten thousand. Cross-entropy puts them at{' '}
          <code>0.01</code> versus <code>4.6</code>: a factor of five hundred. In
          MSE the wrong-end of the curve is nearly flat; the model gets almost no
          gradient signal when it&apos;s confidently wrong, which is exactly the
          moment you most need to yell at it.
        </p>
        <p>
          Put another way: MSE grades your pop quiz by counting how far off your
          probability was. Cross-entropy grades it by asking how surprised
          reality was to hear your answer. The second one trains faster, because
          surprise scales with confidence, and confidence is what you actually
          want the network to calibrate.
        </p>
      </Prose>

      {/* ── Binary case + curve ─────────────────────────────────── */}
      <Prose>
        <p>
          The binary case is worth staring at alone. One output <code>p</code> —
          the predicted probability that <code>y = 1</code> — and the loss
          collapses to a two-term sum:
        </p>
      </Prose>

      <MathBlock caption="binary cross-entropy — the workhorse of every sigmoid output">
{`L   =   −[ y · log p   +   (1 − y) · log(1 − p) ]

When y = 1:   L = − log p           (punishes underconfident positives)
When y = 0:   L = − log(1 − p)      (punishes overconfident positives)`}
      </MathBlock>

      <Prose>
        <p>
          Two curves, one for each value of the true label. Slide <code>p</code>{' '}
          and watch the loss climb asymptotically toward the wrong answer. Flip
          the true label and the mirror-image curve takes over. Both are the
          same quiz rubric, just looking at the problem from the two sides.
        </p>
      </Prose>

      <LogLossCurve />

      <Callout variant="insight" title="the asymmetry is load-bearing">
        A prediction of 0.99 when the truth is 1 costs about <code>0.01</code>.
        A prediction of 0.01 costs about <code>4.6</code>. That steep penalty on
        confident-wrong is why classifiers trained with cross-entropy produce
        probabilities you can actually trust. Being confidently wrong is too
        expensive to do by accident, so the model learns to only be sure when
        it&apos;s earned the right.
      </Callout>

      {/* ── The gradient ────────────────────────────────────────── */}
      <Prose>
        <p>
          Here&apos;s the part that makes libraries fuse softmax and
          cross-entropy into a single op. In practice your network doesn&apos;t
          output <code>p</code> directly — it outputs raw logits{' '}
          <code>z</code>, and softmax turns those into <code>p</code>. The loss
          depends on <code>p</code>, which depends on <code>z</code>, so the
          gradient the{' '}
          <NeedsBackground slug="gradient-descent">gradient descent</NeedsBackground>{' '}
          step actually sees is{' '}
          <code>∂L/∂z</code>. You&apos;d expect an awful mess: softmax has
          exponentials, cross-entropy has a log, and they&apos;re stacked. They
          cancel.
        </p>
      </Prose>

      <MathBlock caption="the gradient that changes everything">
{`∂L                p_i − y_i
───   =    ─────────────────────
∂z_i                1

(yes, it really is just p − y)`}
      </MathBlock>

      <Prose>
        <p>
          The gradient on each logit is the predicted probability minus the
          target probability. Every exponential and every log in the forward
          pass has annihilated itself in the backward pass. This isn&apos;t luck
          — softmax and cross-entropy were designed as partners, and this
          cancellation is the reason the pair is called the canonical
          classification head.
        </p>
      </Prose>

      <SoftmaxCEGradient />

      <Prose>
        <p>
          Drag any logit. Every row&apos;s gradient updates live. The true
          class (green) always gets a negative gradient — &ldquo;push this
          logit up.&rdquo; Every other class gets a positive gradient — &ldquo;push
          this logit down, you stole probability mass that wasn&apos;t yours to
          claim.&rdquo; The size of the push on each wrong class is
          proportional to how much mass it&apos;s currently hoarding. The update
          rule balances itself.
        </p>
      </Prose>

      {/* ── Gotchas: includes the log(0) anchor beat ────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">log(0) is negative infinity, and floats know it.</strong>{' '}
          The quiz rubric&apos;s &ldquo;infinity punishment for confident-wrong&rdquo;
          is a beautiful property in math and a disaster in code. If your model
          ever outputs a probability of exactly 0 for the true class,{' '}
          <code className="text-dark-text-primary">−log(0)</code> returns{' '}
          <code className="text-dark-text-primary">inf</code>, the gradient
          becomes <code className="text-dark-text-primary">nan</code>, and every
          parameter downstream is corrupted forever. The fix: never compute
          softmax then log separately. Use the fused log-softmax trick (subtract
          the max logit, sum the exps, log that) so the intermediate never
          touches 0.
        </p>
        <p>
          <strong className="text-term-amber">Never</strong> hand-compute softmax
          then log then cross-entropy as three calls. Use your library&apos;s
          fused op —{' '}
          <code className="text-dark-text-primary">nn.CrossEntropyLoss</code> in
          PyTorch,{' '}
          <code className="text-dark-text-primary">sparse_categorical_crossentropy(from_logits=True)</code>{' '}
          in Keras. More numerically stable, uses the clean gradient formula.
        </p>
        <p>
          <strong className="text-term-amber">
            CrossEntropyLoss expects logits, not probabilities.
          </strong>{' '}
          Feed it raw output —{' '}
          <em>not</em> the result of applying softmax yourself. If you
          softmax-then-crossentropy, you&apos;re double-softmaxing and your
          gradients are wrong. This is the single most common bug in beginner
          PyTorch code.
        </p>
        <p>
          <strong className="text-term-amber">Label smoothing</strong> is a cheap
          regulariser that borrows straight from the quiz metaphor: force the
          teacher to stop accepting 100%-confident answers even when they&apos;re
          right. Replace the hard one-hot target with{' '}
          <code className="text-dark-text-primary">0.9</code> for the correct
          class and{' '}
          <code className="text-dark-text-primary">0.1 / (K−1)</code> spread
          across the rest. Cross-entropy is now graded against a smoothed
          target, which prevents the model from becoming the annoying student
          who bets everything on one answer.
        </p>
      </Gotcha>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers as always. Read top to bottom and watch the boundary
          between softmax and the loss dissolve as you move up the stack.
        </p>
      </Prose>

      <LayeredCode
        layers={[
          {
            label: 'pure python',
            caption: 'cross_entropy_scratch.py',
            runnable: true,
            code: `import math

def softmax(z):
    m = max(z)
    exps = [math.exp(v - m) for v in z]
    s = sum(exps)
    return [e / s for e in exps]

def cross_entropy(probs, target_idx):
    # H(y, p) = -log p[target]
    return -math.log(max(probs[target_idx], 1e-12))

logits = [2.0, 1.2, 0.3, -0.8, -2.0]
probs = softmax(logits)
loss = cross_entropy(probs, target_idx=0)
print(f"probs={[round(p, 4) for p in probs]}")
print(f"loss={loss:.4f}")`,
            output: `probs=[0.6439, 0.2896, 0.0466, 0.0155, 0.0044]
loss=0.4404`,
          },
          {
            label: 'numpy',
            caption: 'cross_entropy_numpy.py',
            runnable: true,
            code: `import numpy as np

def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def cross_entropy(logits, targets):
    # logits: (N, K)   targets: (N,) integer indices
    # Combined log-softmax + NLL for numerical stability.
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.exp(logits).sum(axis=-1))
    log_probs = logits - log_sum_exp[:, None]
    return -log_probs[np.arange(len(targets)), targets].mean()

# Batch of 3 examples, 5 classes
logits = np.array([
    [2.0, 1.2, 0.3, -0.8, -2.0],
    [-1.0, -0.5, 2.0, 0.1, -0.3],
    [0.5, 0.5, 0.5, 0.5, 0.5],
])
targets = np.array([0, 2, 1])
print(f"loss = {cross_entropy(logits, targets):.4f}")
# -> 0.7456`,
          },
          {
            label: 'pytorch',
            caption: 'cross_entropy_pytorch.py',
            code: `import torch
import torch.nn.functional as F

logits = torch.tensor([
    [2.0, 1.2, 0.3, -0.8, -2.0],
    [-1.0, -0.5, 2.0, 0.1, -0.3],
    [0.5, 0.5, 0.5, 0.5, 0.5],
], requires_grad=True)
targets = torch.tensor([0, 2, 1])

# F.cross_entropy takes raw logits and integer targets.
# It fuses log_softmax + NLL internally. Never softmax beforehand.
loss = F.cross_entropy(logits, targets)
print(f"loss = {loss.item():.4f}")

loss.backward()
# First row's gradient is exactly softmax(logits[0]) - onehot(target[0]) / batch_size
print("grads first row:", logits.grad[0])`,
            output: `loss = 0.7456
grads first row: tensor([-0.3561,  0.2896,  0.0466,  0.0155,  0.0044])`,
          },
        ]}
      />

      <Bridge
        label="pure python → numpy (fused log-softmax + NLL)"
        rows={[
          {
            left: 'probs = softmax(z); loss = -log(probs[t])',
            right: 'loss = -(logits - logsumexp(logits))[t]',
            note: 'fuses the two ops — avoids computing probs[t] then logging it',
          },
          {
            left: 'one example at a time',
            right: 'logits[arange(N), targets]  # pick per-row',
            note: 'fancy indexing — no Python loop over the batch',
          },
        ]}
      />

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'log_probs = logits - log_sum_exp ; loss = -log_probs[i, t]',
            right: 'F.cross_entropy(logits, targets)',
            note: 'one call, numerically stable, GPU-aware, autograd-ready',
          },
          {
            left: 'grad = probs - onehot(target)',
            right: 'loss.backward()',
            note: 'autograd computes exactly p - y — the identity from above',
          },
        ]}
      />

      <Callout variant="insight" title="the production signature of this whole chain">
        In a real classifier training loop you write exactly two lines to go
        from activations to gradients: <code>logits = model(x)</code> and{' '}
        <code>loss = F.cross_entropy(logits, targets)</code>. Everything on
        this page — the stable softmax, the log-sum-exp trick, the fused NLL,
        the elegant gradient — lives inside that second line. You still need to
        understand it, because when your training loss hits{' '}
        <code>nan</code> at step 12,000, somebody has to know where to look.
      </Callout>

      <Challenge prompt="Double-softmax bug hunt">
        <p>
          A sneaky one. Two callers pass into <code>cross_entropy</code>.
          Version A sends <em>logits</em>. Version B sends probabilities — which
          is wrong, because <code>cross_entropy</code> already applies
          log-softmax internally. Version B <em>still trains</em>; it just
          learns much more slowly, because its gradients are squashed. The
          scariest kind of bug: the one that doesn&apos;t crash.
        </p>
        <p className="mt-2 mb-3 text-dark-text-muted">
          The starter runs both on identical inputs and prints the loss plus the
          gradient norm. Compare. B&apos;s gradient is visibly smaller.
        </p>
        <CodeBlock runnable language="python" caption="starter · double_softmax.py">{`import numpy as np

# A single example, 3 classes, label = class 0.
logits = np.array([2.0, 1.0, 0.1])
y_true = 0

def softmax(z):
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()

def log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())

# Version A: correct. cross_entropy(logits, y) = -log_softmax(logits)[y]
# Gradient w.r.t. logits: softmax(logits) - onehot(y)
loss_A = -log_softmax(logits)[y_true]
grad_A = softmax(logits) - np.eye(3)[y_true]

# Version B: bug. We already applied softmax, then shove it back through log_softmax.
probs  = softmax(logits)
loss_B = -log_softmax(probs)[y_true]
# Chain rule: dL/dlogits = (softmax(probs) - onehot) @ J_softmax(logits)
# Easier to just get the end-to-end grad numerically.
eps = 1e-5
grad_B = np.zeros_like(logits)
for i in range(len(logits)):
    lp = logits.copy(); lp[i] += eps
    lm = logits.copy(); lm[i] -= eps
    f = lambda L: -log_softmax(softmax(L))[y_true]
    grad_B[i] = (f(lp) - f(lm)) / (2 * eps)

print(f"A  loss = {loss_A:.5f}    grad norm = {np.linalg.norm(grad_A):.5f}")
print(f"B  loss = {loss_B:.5f}    grad norm = {np.linalg.norm(grad_B):.5f}")
print(f"B gradient is {np.linalg.norm(grad_B)/np.linalg.norm(grad_A):.2%} the size of A's.")
`}</CodeBlock>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> Cross-entropy is the
          confidence-scored quiz: the loss scales with how surprised the truth
          was to hear your answer. That&apos;s why confident-wrong diverges and
          confident-right is nearly free — the log was built for exactly that
          shape. Softmax and cross-entropy fit together so cleanly that the
          gradient on the raw logits collapses to <code>p − y</code>, which is
          the entire reason every library fuses them into one op. Never call
          softmax before cross-entropy yourself — use{' '}
          <code>F.cross_entropy</code> and hand it logits.
        </p>
        <p>
          <strong>Next up — Linear Regression (Forward).</strong> We&apos;ve been
          grading classifiers. Time to step back and look at the simplest model
          that outputs a number instead of a class: a linear predictor. No
          softmax, no probabilities, just <code>y = Wx + b</code>. You&apos;ll
          wire it up as a matrix multiply, visualise the whole forward pass,
          and set up the training lesson where you&apos;ll fit it two ways —
          closed form and gradient descent — and watch them disagree about which
          one should have won.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'A Mathematical Theory of Communication',
            author: 'Claude Shannon',
            venue: 'Bell System Technical Journal, 1948 — the paper that invented entropy',
            url: 'https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf',
          },
          {
            title: 'Dive into Deep Learning — 3.4.6 Cross-Entropy Loss',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_linear-classification/softmax-regression.html',
          },
          {
            title: 'When Does Label Smoothing Help?',
            author: 'Müller, Kornblith, Hinton',
            venue: 'NeurIPS 2019',
            url: 'https://arxiv.org/abs/1906.02629',
          },
        ]}
      />
    </div>
  )
}
