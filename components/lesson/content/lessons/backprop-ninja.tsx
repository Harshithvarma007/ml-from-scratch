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
  Eq,
  EqRef,
} from '../primitives'
import BackpropDiff from '../widgets/BackpropDiff'

// "Becoming a backprop ninja" — a homage to Karpathy's exercise, done
// in-browser with a live gradient checker. The reader fills in four tensors'
// worth of gradients and gets per-tensor pass/fail via central-difference
// comparison. Four green checks = ninja achieved.
//
// Signature anchor: the lab assistant standing behind you with a finite-
// difference ruler. Every analytic derivation is immediately checked
// numerically; if your math matches the assistant's measurement to ~1e-6,
// you're right. If not, the eyebrow goes up and you redo the line. Anchor
// threads at the opening (prove you understand backprop), the op-by-op check
// mid-lesson, and the broadcasting gotcha where the assistant catches the
// subtle bug.
export default function BackpropNinjaLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout ────────────────────────────────── */}
      <Prereq currentSlug="backprop-ninja" />

      {/* ── Opening: you think you understand backprop. prove it. ─ */}
      <Prose>
        <p>
          You&apos;ve seen{' '}
          <NeedsBackground slug="backpropagation">backpropagation</NeedsBackground>{' '}
          derived in the abstract. You&apos;ve watched the chain rule march
          through a{' '}
          <NeedsBackground slug="multi-layer-backpropagation">
            multi-layer network
          </NeedsBackground>{' '}
          on paper. Comfortable? Good. Now prove it.
        </p>
        <p>
          Close the tab with the autograd library. Open a blank editor.
          Derive the gradients of a two-layer MLP by hand — every op, every
          shape, every sum. No <code>loss.backward()</code>. No{' '}
          <code>autograd.grad</code>. Just <em>you</em>, the chain rule, and
          a lab assistant standing behind you with a ruler.
        </p>
        <p>
          The lab assistant is finite differences. You write an analytic
          gradient — your <em>hypothesis</em> about how the loss changes when
          you nudge one value. The assistant nudges that value by a tiny
          epsilon, runs the forward pass twice, subtracts, divides. If your
          formula matches their measurement to about <code>1e-6</code>,
          you&apos;re right. If not, they raise an eyebrow and you redo the
          math. No hand-waving possible. The numbers either tie out or they
          don&apos;t.
        </p>
        <p>
          Andrej Karpathy calls this exercise{' '}
          <KeyTerm>becoming a backprop ninja</KeyTerm>, and he&apos;s right
          to make a big deal of it. The first time every gradient passes,
          you stop being afraid of the backward pass forever. Ten minutes if
          you&apos;re sharp, two hours if you&apos;re learning it honestly.
          Both are fine. What matters is that every line you write gets
          checked.
        </p>
      </Prose>

      <Personify speaker="Chain rule">
        I am one sentence: <code>dL/dx = (dL/dy)·(dy/dx)</code>. That&apos;s
        it. If you know what comes out of a box and you know what the box
        does, I tell you what went in. Compose me n times and I still take
        exactly one line.
      </Personify>

      <Personify speaker="Lab assistant">
        I don&apos;t care about your derivation. I nudge{' '}
        <code>x</code> by <code>ε</code>, I run forward twice, I subtract,
        I divide. My answer is close enough to the truth that if yours
        doesn&apos;t match mine, yours is wrong. I&apos;m watching.
      </Personify>

      {/* ── The setup ──────────────────────────────────────────── */}
      <Prose>
        <p>
          Here is the exact network the checker below runs against. Four
          examples, three inputs, a hidden layer of five, three output
          classes, cross-entropy loss. Tiny on purpose — the assistant has
          to finite-difference every parameter, which costs{' '}
          <code>O(P · 2 · forward)</code>, and we want the audit to finish
          in under a second.
        </p>
      </Prose>

      <MathBlock caption="forward pass">
{`z1  =  X @ W1 + b1         # (N, H)
h   =  tanh(z1)             # (N, H)
z2  =  h @ W2 + b2          # (N, K)
p   =  softmax(z2)          # (N, K)
L   =  −(1/N) · Σᵢ log p[i, yᵢ]`}
      </MathBlock>

      <Prose>
        <p>
          Four parameters to get gradients for: <code>W1 (3, 5)</code>,{' '}
          <code>b1 (5,)</code>, <code>W2 (5, 3)</code>,{' '}
          <code>b2 (3,)</code>. The forward cache hands you everything you
          need on the way down: <code>X</code>, <code>z1</code>,{' '}
          <code>h</code>, <code>z2</code>, <code>logp</code>. No peeking at
          gradients — those are <em>your</em> job. The assistant&apos;s job
          is to catch you if you botch one.
        </p>
      </Prose>

      {/* ── The one equation that earns this lesson ────────────── */}
      <Prose>
        <p>
          One derivation you absolutely must have memorised: softmax
          followed by negative-log-likelihood. If you try to chain through
          softmax with a generic Jacobian you will suffer — and the
          assistant will watch you suffer. The fused rule is three symbols:
        </p>
      </Prose>

      <Eq id="softmax-nll" number="1.1" caption="softmax + NLL — combined gradient">
{`∂L/∂z2  =  (p − Y) / N`}
      </Eq>

      <Prose>
        <p>
          where <code>p</code> is the softmax probabilities and{' '}
          <code>Y</code> is the one-hot of the labels. Three terms. That&apos;s
          the whole thing. Every other gradient in this lesson is mechanical
          after <EqRef id="softmax-nll" number="1.1" /> — composition of
          building blocks you already have, each one verified numerically
          before you move on.
        </p>
      </Prose>

      <Bridge
        label="building blocks you already have"
        rows={[
          {
            left: 'y = x @ W',
            right: 'dL/dx = dL/dy @ W.T;  dL/dW = x.T @ dL/dy',
            note: 'matrix-multiply backward — transpose on the other side',
          },
          {
            left: 'y = x + b  (broadcast)',
            right: 'dL/db = dL/dy.sum(axis=0)',
            note: 'bias backward — sum across whatever axis broadcast',
          },
          {
            left: 'y = tanh(x)',
            right: 'dL/dx = dL/dy * (1 − y**2)',
            note: 'tanh derivative — reuse the forward output',
          },
          {
            left: 'y = log(softmax(z))[i, yᵢ] / N',
            right: 'dL/dz = (p − onehot(y)) / N',
            note: 'memorise this one (1.1) — chaining softmax + log is a trap',
          },
        ]}
      />

      <Callout variant="insight" title="the /N is load-bearing">
        Every textbook derivation assumes a single example. Real loss is an{' '}
        <em>average</em> over the batch — hence the <code>1/N</code> in{' '}
        <EqRef id="softmax-nll" number="1.1" />. Forget it and your
        gradients will be off by exactly a factor of <code>N</code>, which
        means the assistant&apos;s numerical check will report a ratio of
        4:1 across the board (we&apos;re running a batch of four). That&apos;s
        the first bug most readers hit. Measure twice, divide once.
      </Callout>

      {/* ── Hero widget: analytic vs numeric, op by op ───────── */}
      <Prose>
        <p>
          The checker is below. Edit the body of <code>backward()</code>,
          press <em>check</em>, and watch four red X&apos;s turn green one
          at a time. The widget is the assistant made visible: it runs your
          analytic code, then re-runs the forward pass <code>2·P</code>{' '}
          times with each parameter nudged by <code>±ε</code>, assembles a
          central-difference gradient, and compares. Pass = max absolute
          error under <code>1e-4</code>. That threshold is generous. If
          your math is right, you&apos;ll tie out closer to <code>1e-9</code>.
        </p>
      </Prose>

      <BackpropDiff />

      <Prose>
        <p>
          Recommended order: <code>dW2</code> and <code>db2</code> first —
          those are cleanest, and they unblock everything else. Then{' '}
          <code>dh</code> as an intermediate (it doesn&apos;t need to be
          returned, but it&apos;s the bridge from the output layer back to
          the hidden one). Then <code>dz1 = dh * (1 − h**2)</code>, then{' '}
          <code>dW1</code>, then <code>db1</code>. If one tensor passes and
          the next one fails, the bug lives <em>after</em> the one that
          passed — chain-rule errors propagate, they don&apos;t teleport.
          The assistant is effectively giving you a binary search over your
          own derivation.
        </p>
        <p>
          This is the payoff moment the whole lesson is built around. You
          derive one op, the assistant numerically verifies it, you move
          on. Derive, check, tie out. Derive, check, tie out. By the time
          the fourth green check lands, you haven&apos;t just written
          backprop — you&apos;ve <em>audited</em> it.
        </p>
      </Prose>

      {/* ── Gotchas: where the assistant catches you ──────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">
            Shape mismatch (broadcasting bit you).
          </strong>{' '}
          <code className="text-dark-text-primary">dW1</code> must match{' '}
          <code className="text-dark-text-primary">W1</code>. If the
          assistant reports <em>wrong shape</em>, you almost certainly
          forgot a transpose —{' '}
          <code className="text-dark-text-primary">X.T @ dz1</code>, not{' '}
          <code className="text-dark-text-primary">X @ dz1</code>. This is
          the classic place where the assistant catches a subtle bug you
          couldn&apos;t have talked yourself out of: the math looked right
          on paper, the numbers disagree, the transpose was the
          difference.
        </p>
        <p>
          <strong className="text-term-amber">
            Wrong axis on the bias sum.
          </strong>{' '}
          <code className="text-dark-text-primary">b1</code> has shape{' '}
          <code className="text-dark-text-primary">(H,)</code> but it got
          broadcast across the batch dimension during the forward pass, so
          its gradient is{' '}
          <code className="text-dark-text-primary">dz1.sum(axis=0)</code>{' '}
          — sum over the examples, not over the hidden units. Get the axis
          wrong and the assistant reports the right magnitude on the wrong
          shape. This is the second-most-common bug in the whole exercise.
        </p>
        <p>
          <strong className="text-term-amber">
            Off by a factor of N.
          </strong>{' '}
          Every gradient already includes the{' '}
          <code className="text-dark-text-primary">/N</code> because it&apos;s
          baked into <EqRef id="softmax-nll" number="1.1" />. Don&apos;t
          divide by N <em>again</em> inside the matmul gradients; the
          divide only happens once, at the softmax step.
        </p>
        <p>
          <strong className="text-term-amber">
            Numerical error passes.
          </strong>{' '}
          Central difference with{' '}
          <code className="text-dark-text-primary">ε = 1e-5</code> is
          accurate to about{' '}
          <code className="text-dark-text-primary">1e-9</code> on smooth
          functions. A gradient under{' '}
          <code className="text-dark-text-primary">1e-4</code> passes
          comfortably. If yours is at{' '}
          <code className="text-dark-text-primary">1e-3</code>, that&apos;s
          not rounding — that&apos;s a bug the assistant is being polite
          about.
        </p>
        <p>
          <strong className="text-term-amber">
            <code className="text-dark-text-primary">tanh</code>&apos;
            reuses the output.
          </strong>{' '}
          The derivative of{' '}
          <code className="text-dark-text-primary">tanh(x)</code> is{' '}
          <code className="text-dark-text-primary">1 − tanh(x)**2</code>,
          which is <code className="text-dark-text-primary">1 − h**2</code>{' '}
          — you already have{' '}
          <code className="text-dark-text-primary">h</code> in the cache.
          No need to pass{' '}
          <code className="text-dark-text-primary">z1</code> through
          again.
        </p>
      </Gotcha>

      {/* ── Follow-up prose: consolidation ─────────────────────── */}
      <Prose>
        <p>
          Once every tensor passes — and it <em>will</em> pass, because
          the assistant doesn&apos;t grade on vibes — take a second and
          look at what you wrote. Maybe twelve lines of NumPy. A decade ago
          this was the entire backward pass of a production-adjacent
          classifier; today it&apos;s a warm-up. You just implemented, by
          hand, the object that PyTorch&apos;s <code>autograd</code>{' '}
          constructs dynamically for every model you&apos;ve ever trained
          — and you verified it, op by op, against finite differences. Not
          &ldquo;it compiled&rdquo;; not &ldquo;the loss went down&rdquo;;
          actually verified.
        </p>
        <p>
          The reason this is the standard rite of passage is that once you
          can write backprop for a two-layer MLP, the failure modes of
          bigger networks start to make sense. Vanishing gradients? You
          now know exactly which <code>@</code> and which <code>*</code>{' '}
          is shrinking them. Exploding gradients? Same question, other
          direction. Shapes not matching? You can read a stack trace
          without fear, because you know what shape{' '}
          <em>should</em> be there. The ninja badge isn&apos;t that you
          did the math; it&apos;s that the numbers tied out and you know
          they did.
        </p>
      </Prose>

      {/* ── Challenge: ninja, part two ─────────────────────────── */}
      <Challenge prompt="Ninja, part two">
        <p>
          Swap <code>tanh</code> for <code>ReLU</code> in the forward pass.
          Work out the new <code>dz1</code> (hint: the derivative of ReLU
          is a 0/1 mask you can read straight off <code>z1</code>).
          Everything else stays identical. Then run the check — if your
          new derivation is right, the assistant should tie out the same
          way it did before.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: add a third hidden layer and re-derive. You&apos;ll
          notice the pattern — every new hidden layer adds exactly two
          gradient lines to <code>backward()</code>. That pattern is what
          makes <code>autograd</code> possible in the first place.
        </p>
      </Challenge>

      {/* ── Takeaway + cliffhanger → mlp-from-scratch ──────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> You can now write
          backprop for an MLP without looking anything up, and — more
          importantly — you can prove it. The chain rule is one line; the
          building blocks (matmul backward, bias backward, element-wise
          backward) are four. The softmax + NLL fusion is the only
          memorisation tax. Everything else is composition, and finite
          differences is the lab assistant who keeps you honest while
          you&apos;re composing.
        </p>
        <p>
          <strong>Next up — MLP from Scratch.</strong> You&apos;ve verified
          the math. Now bolt it into a training loop and make something
          actually learn. Forward, backward (the one you just wrote),{' '}
          <NeedsBackground slug="gradient-descent">
            gradient descent
          </NeedsBackground>{' '}
          update, repeat. The assistant will quiet down — real training
          runs don&apos;t finite-difference every step — but the
          confidence you just earned is what lets you read your own loss
          curve and know whether your backward pass is the thing at
          fault.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Becoming a Backprop Ninja',
            author: 'Andrej Karpathy',
            venue: 'Neural Networks: Zero to Hero, lecture 4',
            year: 2022,
            url: 'https://www.youtube.com/watch?v=q8SA3rM6ckI',
            tags: ['blog'],
          },
          {
            title: 'CS231n: Backpropagation, Intuitions',
            author: 'Andrej Karpathy',
            venue: 'Stanford CS231n',
            url: 'https://cs231n.github.io/optimization-2/',
            tags: ['blog'],
          },
          {
            title: 'The Matrix Calculus You Need For Deep Learning',
            author: 'Parr & Howard',
            year: 2018,
            url: 'https://arxiv.org/abs/1802.01528',
            tags: ['paper'],
          },
          {
            title: 'Deep Learning — Chapter 6: Deep Feedforward Networks',
            author: 'Goodfellow, Bengio, Courville',
            venue: 'MIT Press, 2016',
            url: 'https://www.deeplearningbook.org/contents/mlp.html',
            tags: ['book'],
          },
        ]}
      />
    </div>
  )
}
