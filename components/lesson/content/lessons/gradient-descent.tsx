import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import Quiz from '../Quiz'
import WhatNext from '../WhatNext'
import Prereq from '../Prereq'
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
  AsciiBlock,
} from '../primitives'
import LossSurface3D from '../widgets/LossSurface3DLazy'
import LearningRateExplorer from '../widgets/LearningRateExplorer'
import TrajectoryScrubber from '../widgets/TrajectoryScrubber'
import NonConvexExplorer from '../widgets/NonConvexExplorerLazy'
import MomentumCompare from '../widgets/MomentumCompare'

// A single flowing narrative. Widgets carry most of the visual weight; prose
// glues the beats together. No section headers — the lesson should read like
// one long scroll, not a tabbed manual.
//
// Signature anchor: a blindfolded night hiker feeling for the downhill
// direction. Introduced in the pre-primer and returned to at the gradient
// reveal, the learning-rate section, and the saddle-point section. The rest
// of the curriculum can link back here via <NeedsBackground slug=
// "gradient-descent"> whenever "minimize the loss" first shows up.
//
// Single cinematic beat: one Indiana Jones / boulder wink at the learning-rate
// section. Not recurring — just a land.
export default function GradientDescentLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout (entry point: empty state) ─────── */}
      <Prereq currentSlug="gradient-descent" />

      {/* ── Pre-primer: zero-knowledge on-ramp ──────────────────── */}
      <Prose>
        <p>
          Imagine you&apos;re blindfolded. Someone drops you on a hillside at
          night and tells you to walk to the bottom of the valley. You
          can&apos;t see the hill. You can&apos;t see the valley. The only
          thing your body has to work with is the slope of the ground under
          your feet.
        </p>
        <p>
          What do you do? You shuffle a little, feel which way the ground
          falls away fastest, step that way, and repeat. When every direction
          feels flat, you&apos;re at the bottom. Or, honestly, you&apos;re at{' '}
          <em>a</em> bottom — but we&apos;ll get to that disappointment later.
        </p>
        <p>
          That&apos;s gradient descent. Whole algorithm. Before any math,
          before any code, before any neural network — that&apos;s the thing
          we&apos;re doing. The rest of this page is what the blindfold, the
          hill, and the shuffle become when the hill is the inside of a model.
        </p>
      </Prose>

      <Callout variant="insight" title="three words before we start">
        <div className="space-y-2">
          <p>Three plain-English words you&apos;ll see over and over.</p>
          <p>
            <strong>Model</strong> — a formula with dials. An input goes in, a
            prediction comes out. Turn the dials, get different predictions. A
            neural network is one of these with millions to billions of dials.
          </p>
          <p>
            <strong>Loss</strong> — one number. Given where the dials are set
            right now, how wrong is the model&apos;s prediction? Bigger =
            worse.
          </p>
          <p>
            <strong>Training</strong> — turn the dials until the loss is small.
            That&apos;s it. That&apos;s the whole objective of the entire
            multi-billion-dollar industry.
          </p>
        </div>
      </Callout>

      <Prose>
        <p>
          The hill is the loss. Each spot on the hillside corresponds to one
          setting of the model&apos;s dials, and the height at that spot is
          how wrong the model is when its dials are set that way. Low ground =
          small loss = model gets answers right. The blindfolded shuffle{' '}
          <em>is</em> training: you can&apos;t see the whole hill — it lives
          in millions of dimensions, one per dial — so you feel your way down
          it.
        </p>
      </Prose>

      <AsciiBlock caption="the whole lesson, in a diagram">
{`       dials (θ)  ──▶  MODEL  ──▶  prediction
                                      │
                       right answer ──┴──▶  compare  ──▶  LOSS (one number)


       training loop:
         1. feel the slope under your feet   ←  gradient
         2. step the opposite way, a little  ←  update rule
         3. repeat until the ground is flat  ←  done`}
      </AsciiBlock>

      <Prose>
        <p>
          From here on: when you read <em>parameter</em>, picture a dial. When
          you read <em>gradient</em>, picture the way the ground slopes under
          your blindfolded foot. When you read <em>learning rate</em>, picture
          the size of the step you&apos;re willing to take before stopping to
          feel the ground again. Everything else is detail.
        </p>
      </Prose>

      {/* ── Opening: problem-first reframe ──────────────────────── */}
      <Prose>
        <p>
          Let&apos;s make the problem concrete. You have a model — say, one
          that looks at an image and predicts whether it&apos;s a cat. Fresh
          out of the box, it guesses &ldquo;cat&rdquo; for everything: dogs,
          trees, your lunch. The loss is enormous. The model, to use a
          technical term, is cooked.
        </p>
        <p>
          You could tweak the dials by hand. There are a hundred million of
          them. You&apos;ll finish sometime in the next thousand years,
          assuming you don&apos;t stop to eat.
        </p>
        <p>
          Or you could do what every neural network you&apos;ve ever heard of
          — GPT-4, DALL·E, AlphaFold, the thing that picks your next YouTube
          video — does, which is: shuffle. One loop, running billions of times
          a second, each step asking the same question:{' '}
          <em>am I going the right direction?</em>
        </p>
        <p>
          That&apos;s <KeyTerm>gradient descent</KeyTerm>. Genuinely, almost
          insultingly simple. Most tutorials draw an arrow pointing downhill,
          write <code>w ← w − α∇L</code>, and walk away. You&apos;ll leave
          this page with something better: a feel for the algorithm in your
          fingers. Drop a marble on a loss surface. Crank the learning rate
          until your model detonates. Scrub through every iteration by hand.
          Then — and only then — write the code.
        </p>
        <p>
          <strong>Start here.</strong> The hero widget below is a 3D loss
          surface. Click anywhere to place a marble, then press{' '}
          <em>release</em>. Tilt your view, drop marbles in different places,
          slide α around. The rest of the lesson is an unpacking of what you
          see here.
        </p>
      </Prose>

      {/* ── Hero widget ─────────────────────────────────────────── */}
      <LossSurface3D />

      <Prose>
        <p>
          A bowl-shaped surface. A marble. The marble feels the slope right
          where it is and rolls downhill. Same hiker as before — just
          well-animated, and lit. The math you&apos;re about to see{' '}
          <em>is</em> a marble on a surface, with the word
          &ldquo;marble&rdquo; replaced by <em>parameter vector</em> and the
          word &ldquo;surface&rdquo; replaced by <em>loss function</em>. No
          metaphor is being smuggled.
        </p>
      </Prose>

      <Callout variant="note" title="where this lives in the loop">
        Gradient descent is an <strong>optimization algorithm</strong>. Its
        one job: given a function, find the input that makes it as small as
        possible. In ML that function is the <em>loss</em> — a number
        measuring how wrong the model is right now — and the inputs are the
        model&apos;s parameters. Every <code>.fit()</code>, every{' '}
        <code>loss.backward()</code>, every training run on every GPU in
        every data center you&apos;ve heard of — gradient descent is the
        innermost loop.
      </Callout>

      {/* ── Two personas ────────────────────────────────────────── */}
      <Prose>
        <p>
          The marble only ever controls two things. First: which direction to
          step. Second: how far. Meet them.
        </p>
      </Prose>

      <Personify speaker="Gradient">
        I point uphill. That&apos;s my whole personality. If you want down,
        step the opposite way — and please don&apos;t ask me to plan more
        than one step ahead, I can only tell you about right here, right now.
      </Personify>

      <Personify speaker="Learning rate">
        Set me too high and I&apos;ll blow up your model. Set me too low and
        your model will still be training when the sun burns out. There is
        no correct value for me in the abstract — only a correct value for{' '}
        <em>your</em> loss surface. Good luck.
      </Personify>

      <Prose>
        <p>
          The gradient tells you which way. The learning rate tells you how
          far. You want the biggest step you can take without eating the
          pavement — think Indy, sprinting down the corridor with the boulder
          on his heels, one bad stride from being part of the floor. Every
          other optimizer on the planet — SGD, Adam, RMSProp, momentum,
          Adagrad — is an elaboration of those two choices. Get the
          intuition right here and the rest will feel like footnotes.
        </p>
        <p>
          Before any math, let&apos;s put that personification under a
          microscope. Strip the surface from 3D down to 1D — same idea, but
          now you can watch the marble step back and forth along a line and{' '}
          <em>see</em> what the learning rate actually does.
        </p>
      </Prose>

      {/* ── Learning rate side-view ─────────────────────────────── */}
      <LearningRateExplorer />

      <Prose>
        <p>
          Drag α upward slowly. Around <code>α = 0.5</code> the marble stops
          settling and starts bouncing — every step overshoots the bottom.
          Push past <code>α = 1</code> and it launches off into the void —
          our blindfolded hiker just tripped over their own feet. There&apos;s
          a hard convergence condition lurking here, and you just found it by
          feel. Now we find it by math.
        </p>
      </Prose>

      {/* ── Math derivation ─────────────────────────────────────── */}
      <Prose>
        <p>
          Pick the simplest possible loss: <code>f(x) = x²</code>. A bowl
          with its bottom at <code>x = 0</code>. Everything we need follows
          from two facts.
        </p>
      </Prose>

      <MathBlock caption="derivative of the loss">
{`f(x)   =  x²
f'(x)  =  2x`}
      </MathBlock>

      <Prose>
        <p>
          At any point <code>x</code>, <code>f&apos;(x)</code> is the slope —
          positive means climbing right, negative means climbing left. This
          is the number the hiker feels under their foot. The update rule
          says <em>move opposite to the slope</em>:
        </p>
      </Prose>

      <Eq id="gd-update" number="1.1" caption="gradient descent update on f(x) = x²">
{`x_new  =  x_old  −  α · f'(x_old)
       =  x_old  −  α · 2 · x_old
       =  x_old · (1 − 2α)`}
      </Eq>

      <Prose>
        <p>
          Look at the last line of <EqRef id="gd-update" number="1.1" />.
          Every step multiplies <code>x</code> by the same constant. So after{' '}
          <code>n</code> steps starting from <code>x₀</code>:
        </p>
      </Prose>

      <Eq id="gd-closed-form" number="1.2" caption="closed form after n steps">
{`x_n  =  x_0 · (1 − 2α)^n`}
      </Eq>

      <Prose>
        <p>
          This is a geometric sequence — the math version of the thing you
          already saw happen. <EqRef id="gd-closed-form" number="1.2" />{' '}
          decays to zero iff <code>|1 − 2α| &lt; 1</code>, i.e.{' '}
          <code>0 &lt; α &lt; 1</code>. Below <code>α = 0.5</code> the
          multiplier is positive — every step is the same sign, the hiker
          walks straight down. Between <code>0.5</code> and <code>1</code>{' '}
          the multiplier flips negative — they overshoot, land on the far
          side, overshoot less, repeat. Zigzag, but converging. At exactly{' '}
          <code>α = 1</code>, <code>|1 − 2α| = 1</code> — they bounce
          between <code>x₀</code> and <code>−x₀</code> forever, a tennis
          match with no umpire. Past <code>α = 1</code>, divergence. The
          hiker is now airborne. Flip back up and try those thresholds in
          the widget — the math says exactly what your eyes saw.
        </p>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">α too high (e.g. 0.6):</strong>{' '}
          <code className="text-dark-text-primary">(1 − 2·0.6) = −0.2</code>,
          so <code className="text-dark-text-primary">x</code> flips sign
          every step. Still converges — just zigzags.
        </p>
        <p>
          <strong className="text-term-amber">α ≥ 1:</strong> the multiplier
          exceeds 1 in absolute value. Each step overshoots by more than it
          corrects. <code className="text-dark-text-primary">x</code> flies
          off to infinity. The model is cooked.
        </p>
        <p>
          <strong className="text-term-amber">Real networks:</strong>{' '}
          <code className="text-dark-text-primary">f(x) = x²</code> is the
          friendliest loss in existence. Real loss surfaces have curvature
          that varies wildly across parameters, which is why a single global{' '}
          <code className="text-dark-text-primary">α</code> almost never
          works and why adaptive optimizers (Adam and friends) exist.
        </p>
      </Gotcha>

      {/* ── Trace by hand ───────────────────────────────────────── */}
      <Prose>
        <p>
          The formula says <code>x₂₅ = 5 · 0.8²⁵ ≈ 0.0189</code>. That&apos;s
          true in the same way &ldquo;Paris is the capital of France&rdquo;
          is true — correct and utterly unconvincing until you&apos;ve been
          there. Scrub through 25 steps below. Watch <code>x</code> shrink;
          watch the loss collapse. The two numbers the marble sees at each
          step are just <code>x</code> and <code>f(x)</code>. Nothing else.
        </p>
      </Prose>

      <TrajectoryScrubber />

      <Callout variant="insight" title="what the numbers show">
        The first five steps crush the loss by a factor of six. The last
        five steps barely move it. This exponential decay is the fingerprint
        of gradient descent on a quadratic — early progress is cheap, late
        progress is expensive. In deep nets you&apos;ll see the same shape
        in every training curve ever plotted.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          You&apos;ve seen it move. You&apos;ve seen the math. Now write it
          three times, each shorter than the last — and the third one trains
          real neural networks. Pure Python, NumPy, PyTorch. Every production
          training script in the world is one of those three with a million
          more lines of bookkeeping wrapped around it.
        </p>
      </Prose>

      <CodeBlock
        runnable
        language="python"
        caption="layer 1 — pure python · gradient_descent_scratch.py"
        output="After 25 steps: 0.01889"
      >{`def gradient_descent_scratch(init, learning_rate, iterations):
    x = init                                   # start at our initial position
    for step in range(iterations):
        gradient = 2 * x                       # f'(x) = 2x — the slope at current x
        x = x - learning_rate * gradient       # x_new = x_old - a * f'(x_old)
    return x

result = gradient_descent_scratch(init=5.0, learning_rate=0.1, iterations=25)
print(f"After 25 steps: {result:.5f}")`}</CodeBlock>

      <Prose>
        <p>
          The line <code>x = x - learning_rate * gradient</code> <em>is</em>{' '}
          the update rule from the math. Nothing is hidden. Matches the
          scrubber&apos;s final value to the digit.
        </p>
        <p>
          One parameter is cute. Real models have billions. Looping in pure
          Python over a billion parameters per step would finish training
          sometime around the heat death of the sun. Enter NumPy — same
          loop, but the inner arithmetic runs in compiled C on whole vectors
          at once.
        </p>
      </Prose>

      <CodeBlock runnable language="python" caption="layer 2 — numpy · gd_multi_numpy.py">{`import numpy as np   # NumPy — Python's numerical backbone. Vectorised arithmetic in C.

def gd_multi_numpy(theta_init, learning_rate, iterations):
    theta = np.array(theta_init, dtype=float)   # wrap the list in a vector
    for _ in range(iterations):
        gradient = 2 * theta                    # operates on ALL elements simultaneously
        theta = theta - learning_rate * gradient
    return theta

result = gd_multi_numpy([5.0, -3.0, 2.0], learning_rate=0.1, iterations=25)
print(np.round(result, 4))
# -> [ 0.0189 -0.0113  0.0076]`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'gradients = [2 * t for t in theta]',
            right: 'gradient = 2 * theta',
            note: 'one operation, all elements — broadcasting',
          },
          {
            left: 'theta = [t - lr * g for t, g in zip(...)]',
            right: 'theta = theta - lr * gradient',
            note: 'vector subtraction, no Python loop',
          },
        ]}
      />

      <Prose>
        <p>
          NumPy knows what <code>2 * theta</code> means because the gradient
          of <code>Σ θᵢ²</code> is closed-form. Real networks chain thousands
          of operations — each layer&apos;s output is the input to the next,
          and each operation contributes its own slope to the final slope.
          That stacking of slopes has a name (the <em>chain rule</em>, coming
          up in the Backpropagation lesson), and it&apos;s why you can&apos;t
          hardcode the gradient of GPT-4 by hand. That&apos;s why PyTorch
          exists: it computes gradients automatically.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · gd_pytorch.py"
        output={`tensor([ 0.0189, -0.0113,  0.0076])`}
      >{`import torch
import torch.optim as optim    # every optimizer lives here — SGD, Adam, RMSProp, all of them

# requires_grad=True tells PyTorch to track this tensor for automatic differentiation.
theta = torch.tensor([5.0, -3.0, 2.0], requires_grad=True)

# optim.SGD is the packaged version of our update rule: theta -= lr * gradient.
optimizer = optim.SGD([theta], lr=0.1)

for step in range(25):
    optimizer.zero_grad()        # PyTorch accumulates gradients — reset each step
    loss = (theta ** 2).sum()    # f(theta) = sum(theta_i ** 2)
    loss.backward()              # compute gradients automatically (autograd)
    optimizer.step()             # apply: theta = theta - lr * gradient

print(torch.round(theta.detach(), decimals=4))`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'loss = (theta ** 2).sum()',
            right: 'f(θ) = Σ θᵢ²',
            note: 'same objective, defined in Python',
          },
          {
            left: 'loss.backward()',
            right: 'gradient = 2 * theta',
            note: 'we hardcoded this — autograd derives it from the loss expression',
          },
          {
            left: 'optimizer.step()',
            right: 'theta = theta - lr * gradient',
            note: 'identical update, packaged as a single call',
          },
        ]}
      />

      <Callout variant="insight" title="the one line that earns the library">
        In layers 1 and 2 we wrote <code>gradient = 2 * theta</code> because
        we knew the function by hand. In a real network — millions of
        parameters, thousands of operations chained together — PyTorch
        computes every gradient <em>automatically</em> by walking the chain
        rule backwards through the computation graph. That&apos;s the
        superpower, and it&apos;s the reason nobody implements{' '}
        <code>backward()</code> by hand for real models anymore.
      </Callout>

      {/* ── Real landscapes aren't bowls ────────────────────────── */}
      <Prose>
        <p>
          Now the hard part. Everything so far has been <code>f(x) = x²</code>{' '}
          — a cooperative bowl with a single minimum and smooth curvature
          everywhere. Your intuition has been trained on the friendliest
          function that exists.
        </p>
        <p>
          Real loss surfaces are not cooperative. They have{' '}
          <strong>local minima</strong> — small bowls nested inside the big
          one. They have <strong>saddle points</strong>, where the ground
          goes up in one direction and down in another — the hiker&apos;s
          foot feels flat on average, even though they&apos;re nowhere near
          a valley floor. They have <strong>plateaus</strong> where progress
          dies for hundreds of steps at a time. Drop the marble here, then
          there, then somewhere else — and watch gradient descent give you a
          different answer every time.
        </p>
      </Prose>

      <NonConvexExplorer />

      <Prose>
        <p>
          Same algorithm. Same α. Different starting points, wildly
          different final answers. That&apos;s not a bug in gradient descent
          — it&apos;s the nature of non-convex optimization, and it&apos;s
          why &ldquo;initialization&rdquo; is a real line item in every
          modern training pipeline. The marble only ever sees the local
          slope, so where you drop it <em>matters</em>.
        </p>
        <p>
          The honest takeaway is slightly unsettling: in a big enough neural
          net, you&apos;re never really finding <em>the</em> minimum.
          You&apos;re finding <em>a</em> minimum. Fortunately, a weird
          empirical fact rescues us — in high dimensions, the local minima
          of realistic loss surfaces tend to have nearly identical loss
          values. The surface is messy, but most of the messy spots are
          about equally good. Hundreds of valleys, each one a fine place to
          stop. More on that when we get to the training chapter.
        </p>
      </Prose>

      {/* ── Momentum teaser ─────────────────────────────────────── */}
      <Prose>
        <p>
          One more thing vanilla gradient descent is terrible at: narrow
          ravines. Long thin valleys where the slope across the valley is
          huge and the slope along it is tiny. The marble burns every step
          ping-ponging across the ravine and barely any traveling forward.
          The cure is inertia — give the marble momentum so it smooths out
          the zigzag and picks up speed along the valley floor. Watch the
          two of them side by side.
        </p>
      </Prose>

      <MomentumCompare />

      <Prose>
        <p>
          Blue is vanilla. Gold is momentum. Same start, same α — one of
          them ends near the minimum, the other is still zigzagging. The
          whole zoo of modern optimizers (Adam, RMSProp, Adafactor) are
          variations on the same idea: carry state across steps so a single
          noisy gradient can&apos;t knock you off course. We&apos;ll build
          momentum from scratch in a later lesson. For now just note: this
          is what people mean when they say &ldquo;SGD with momentum.&rdquo;
        </p>
      </Prose>

      {/* ── Challenge + Takeaways ───────────────────────────────── */}
      <Challenge prompt="Break it on purpose">
        <p>
          Bump <code>lr</code> past <code>0.5</code>. Within a handful of
          steps <code>x</code> starts bouncing across zero. Push it past{' '}
          <code>1.0</code> and it runs away to infinity. That&apos;s the
          convergence condition from{' '}
          <EqRef id="gd-closed-form" number="1.2" /> —{' '}
          <code>|1 − 2α| &lt; 1</code> — enforcing itself in real code.
        </p>
        <p className="mt-2 mb-3 text-dark-text-muted">
          Try it. Change <code>lr</code>, re-run, then graph <code>trail</code>{' '}
          mentally against the rule.
        </p>
        <CodeBlock runnable language="python" caption="starter · break_it.py">{`# Vanilla gradient descent on f(x) = x^2. Starts at x = 5.
# Challenge: find the largest lr for which |x_25| < 1e-3.
lr = 0.1            # try 0.5, 0.9, 1.0, 1.1 — watch the regime flip
steps = 25
x = 5.0
trail = [x]
for _ in range(steps):
    grad = 2 * x
    x = x - lr * grad
    trail.append(x)

print(f"lr = {lr}")
print(f"final x       = {x:+.5f}")
print(f"max |x| seen  = {max(abs(v) for v in trail):+.5f}")
print(f"converged?    = {abs(x) < 1e-3}")`}</CodeBlock>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> Gradient descent is the
          loop inside every training algorithm in ML — not optional,
          load-bearing. The learning rate isn&apos;t just a hyperparameter;
          it&apos;s a <em>convergence condition</em>, and breaking it
          breaks the model completely. The three-layer progression — pure
          Python, NumPy, PyTorch — is the same progression we&apos;ll use
          for every algorithm in this series: see the mechanics, scale them
          up, then cede them to the library. Every layer above is a shortcut
          for a layer below, never magic.
        </p>
        <p>
          <strong>Next up — Sigmoid &amp; ReLU.</strong> Gradient descent
          needs a function to differentiate. Inside a neural net that
          function is a stack of matrix multiplies with a little
          shape-bending non-linearity wedged between each layer — and the
          whole thing collapses into a single straight line without those
          non-linearities. The two most common are sigmoid and ReLU. Small,
          humble, and they decide what &ldquo;firing&rdquo; means for a
          neuron. Their derivatives plug directly into the update rule you
          just learned — and one of them has a bad habit of murdering
          gradients entirely. We&apos;ll find out which, and why.
        </p>
      </Prose>

      <WhatNext currentSlug="gradient-descent" hidePrerequisites />

      <Quiz
        question={
          <>
            You triple the learning rate on a model that was training fine at{' '}
            <code>lr=0.01</code>. The loss shoots up, then to NaN. What&apos;s
            the mechanical story?
          </>
        }
        options={[
          {
            text: 'Each step overshoots the minimum by more than the last, so |w| grows geometrically until it overflows.',
            correct: true,
            explain:
              'Right. The update ‐α∇L(w) takes you past the minimum; the gradient on the far side points back with a bigger magnitude; α triples that, and the next step overshoots farther. That&apos;s exponential divergence — eventually a float64 can&apos;t hold it.',
          },
          {
            text: 'The gradient is zero at the minimum, so tripling α makes it do nothing and training stalls.',
            explain:
              'Zero gradient gives no update regardless of α — that would stall, not explode. The problem here is the opposite: you\u2019re nowhere near the minimum and the update jumps too far.',
          },
          {
            text: 'The model is just undertrained; you need more epochs, not a smaller lr.',
            explain:
              'Epochs and lr solve different problems. If each step is already divergent, adding more of them only speeds up the explosion.',
          },
        ]}
      />

      <References
        items={[
          {
            title: 'Dive into Deep Learning — Chapter 12: Optimization Algorithms',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_optimization/index.html',
            tags: ['book'],
          },
          {
            title: 'Deep Learning — Chapter 4: Numerical Computation',
            author: 'Goodfellow, Bengio, Courville',
            venue: 'MIT Press, 2016',
            url: 'https://www.deeplearningbook.org/contents/numerical.html',
            tags: ['book'],
          },
          {
            title: 'An overview of gradient descent optimization algorithms',
            author: 'Sebastian Ruder',
            year: 2016,
            url: 'https://arxiv.org/abs/1609.04747',
            tags: ['paper', 'blog'],
          },
          {
            title: 'Visualizing the Loss Landscape of Neural Nets',
            author: 'Li, Xu, Taylor, Studer, Goldstein',
            venue: 'NeurIPS 2018',
            url: 'https://arxiv.org/abs/1712.09913',
            tags: ['paper'],
          },
        ]}
      />
    </div>
  )
}
