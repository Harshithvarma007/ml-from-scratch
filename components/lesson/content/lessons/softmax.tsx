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
import SoftmaxExplorer from '../widgets/SoftmaxExplorer'
import TemperatureRegimes from '../widgets/TemperatureRegimes'
import SoftmaxStability from '../widgets/SoftmaxStability'

// Signature anchor: softmax as the pollster that converts raw opinions
// (logits) into a probability distribution (percentages that sum to 1).
// Returned to at the opening hook, the normalization reveal, and the
// temperature section — temperature is the pollster's conviction dial.
export default function SoftmaxLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout ─────────────────────────────────── */}
      <Prereq currentSlug="softmax" />

      {/* ── Opening: problem-first reframe ──────────────────────── */}
      <Prose>
        <p>
          Your network just spat out a vector of numbers — one per class — and
          now you have to answer two questions the user actually cares about:{' '}
          <em>which class, and how sure are you?</em> The raw numbers can be
          anything. Positive, negative, tiny, huge. They don&apos;t sum to
          anything. They aren&apos;t probabilities. They&apos;re opinions with
          no units.
        </p>
        <p>
          Think of it like an exit poll before anyone has normalized the
          counts. One candidate got <code>+2.1</code>, another{' '}
          <code>-0.8</code>, a third <code>+4.3</code>. The numbers rank the
          field but mean nothing as percentages. You need a pollster — someone
          who takes raw opinion scores and turns them into{' '}
          <em>&ldquo;candidate A: 87%, candidate B: 11%, candidate C: 2%&rdquo;</em>.
          A clean distribution that adds up to one, with no negatives and no
          dishonest rounding.
        </p>
        <p>
          <KeyTerm>Softmax</KeyTerm> is that pollster. It takes a vector of
          logits, keeps their ranking, and returns a genuine probability
          distribution — non-negative numbers that sum to exactly one. It is
          the last operation in practically every classifier you&apos;ll ever
          build.
        </p>
        <p>
          The formula is small. The behavior is rich. And the one
          implementation detail that keeps it from detonating in production
          is your first real taste of numerical stability in this series.
        </p>
      </Prose>

      {/* ── The naive attempt ───────────────────────────────────── */}
      <Callout variant="note" title="why not just divide by the sum?">
        Before we write anything fancy, try the obvious thing. To turn a list
        into percentages, divide each entry by the sum. Works for poll counts.
        Try it on logits <code>[2, -1, 3]</code>: the sum is{' '}
        <code>4</code>, so you get <code>[0.5, -0.25, 0.75]</code>. One of
        your &ldquo;probabilities&rdquo; is negative. The probability axioms
        just filed a complaint. We need a step that forces everything positive
        before we normalize.
      </Callout>

      <Prose>
        <p>
          The exponential function <code>e^x</code> does exactly that. It
          sends every real number to a positive number — big positives become
          huge, big negatives become tiny, and zero becomes one. Exponentiate
          first, then divide by the sum. Every output is now positive, and
          the whole vector sums to one. That&apos;s softmax.
        </p>
      </Prose>

      <MathBlock caption="softmax — the whole thing">
{`               e^(zᵢ / T)
softmax(z)ᵢ = ─────────────
              Σⱼ e^(zⱼ / T)`}
      </MathBlock>

      <Prose>
        <p>
          Four moving parts. The exponent makes every output positive (because{' '}
          <code>e^x</code> is always positive). The sum in the denominator
          normalises the whole vector to one — this is the pollster writing
          &ldquo;of 100 simulated voters&rdquo; at the top of the chart. The
          temperature <code>T</code> controls how peaky the result is.
          That&apos;s it.
        </p>
        <p>
          Drag the sliders below. Left column is logits, right column is the
          resulting probabilities. Push one class&apos;s logit above the
          others and watch its bar dominate — but notice the others never{' '}
          <em>quite</em> go to zero. Softmax is smooth, not greedy. It never
          declares a landslide when the poll hasn&apos;t earned one.
        </p>
      </Prose>

      <SoftmaxExplorer />

      <Callout variant="note" title="softmax is not the same as max">
        <code>argmax</code> picks the winner and sets every other class to
        zero. Softmax is a softened, differentiable version: it keeps the
        winner, but also leaves every loser with a little probability mass.
        That softness is the whole point — you can backprop through it, which
        you cannot do through <code>argmax</code> (the derivative is zero
        everywhere it&apos;s defined, undefined at the transitions). Every
        classifier{' '}
        <NeedsBackground slug="gradient-descent">loss</NeedsBackground>{' '}
        you&apos;ll meet was written assuming softmax, not argmax.
      </Callout>

      <Personify speaker="Softmax">
        I turn your opinions into a distribution. I&apos;m smooth,
        differentiable, and order-preserving. Give me a big logit and
        I&apos;ll give that class most of the mass — but I always save a
        crumb for the losers so{' '}
        <NeedsBackground slug="gradient-descent">gradients</NeedsBackground>{' '}
        can flow back to them. It&apos;s an inclusion policy, not a reward
        for being kind.
      </Personify>

      <Prose>
        <p>
          One subtle thing worth pointing out before we move on. Exponentials
          amplify. Small gaps in the logits become big gaps in the
          probabilities. A candidate ahead by two points in raw opinion can
          end up with 90% of the poll after softmax runs. That&apos;s not a
          bug — it&apos;s the whole reason we use{' '}
          <code>exp</code> instead of some gentler positive function. Confidence
          in the logits gets translated into confidence in the distribution,
          loudly.
        </p>
      </Prose>

      {/* ── Temperature ────────────────────────────────────────── */}
      <Prose>
        <p>
          Now the temperature knob. <code>T</code> divides every logit before
          the exponential, so <code>T &lt; 1</code> makes differences bigger
          (sharper distribution) and <code>T &gt; 1</code> makes them smaller
          (flatter). In the limit <code>T → 0</code> softmax becomes argmax —
          all mass on the top class. In the limit <code>T → ∞</code> it
          becomes uniform — every class equal.
        </p>
        <p>
          Back to the pollster: <code>T</code> is the conviction dial. Low
          temperature, the pollster is calling a landslide — 98% for the
          leader, scraps for the rest. High temperature, the pollster is
          hedging — &ldquo;the race is wide open, anyone could win&rdquo; —
          and the percentages spread out toward uniform. Same logits. Same
          ranking. Totally different story about how confident you should be.
        </p>
        <p>
          The plot below tracks entropy as a function of <code>T</code> for a
          fixed set of logits. Entropy is exactly a measure of uncertainty —
          it&apos;s zero when the distribution is one-hot and{' '}
          <code>log₂(K)</code> when the distribution is uniform over{' '}
          <code>K</code> classes. Slide <code>T</code>; watch the bar chart
          reshape and the entropy dot trace the curve.
        </p>
      </Prose>

      <TemperatureRegimes />

      <Callout variant="insight" title="temperature in the wild">
        When you set <code>temperature=0.7</code> in a ChatGPT API call, this
        is exactly what you&apos;re doing: cooling the softmax over the
        next-token logits so sampling becomes more decisive (and therefore
        more &ldquo;focused&rdquo;). Crank it to 1.5 and the model starts
        picking lower-probability words, becoming more creative and more
        unhinged. Temperature isn&apos;t a hack — it&apos;s a fundamental
        knob built into the function.
      </Callout>

      {/* ── Numerical stability ─────────────────────────────────── */}
      <Prose>
        <p>
          Now the implementation catch. The textbook formula above is{' '}
          <strong>numerically unstable</strong>. Here&apos;s why: in a real
          language model the final logit vector can contain values like{' '}
          <code>z = 842.3</code>. Compute <code>exp(842.3)</code> in IEEE-754
          double precision and you get <code>Infinity</code>. The denominator
          becomes <code>Infinity</code>. The division is <code>NaN</code>.
          Your model&apos;s prediction is now… nothing.
        </p>
        <p>
          The fix is a one-line algebraic identity with enormous consequences.
          Subtract the max of the logit vector from every element <em>before</em>{' '}
          exponentiating. Mathematically you&apos;re multiplying top and bottom
          by <code>exp(-max z)</code>, which cancels — the output is
          identical. Numerically, the largest exponent is now exactly{' '}
          <code>0</code>, so <code>exp</code> never blows up.
        </p>
      </Prose>

      <MathBlock caption="the shift-by-max trick">
{`softmax(z)ᵢ   =   e^(zᵢ)       /   Σⱼ e^(zⱼ)

              =   e^(zᵢ − m)  /   Σⱼ e^(zⱼ − m)      where m = max(z)

All exponents are now ≤ 0. Largest is exp(0) = 1. No overflow, ever.`}
      </MathBlock>

      <Prose>
        <p>
          See it fail and then stop failing. The left column below runs the
          naive formula, the right column runs the shift-by-max version.
          Crank the offset slider past 700 and the naive column collapses
          into NaNs; the stable column stays serene.
        </p>
      </Prose>

      <SoftmaxStability />

      <Gotcha>
        <p>
          <strong className="text-term-amber">Never compute softmax by exp-then-divide</strong>{' '}
          in production code. Always subtract the max first. Every ML library already does this
          internally (<code className="text-dark-text-primary">torch.softmax</code>,{' '}
          <code className="text-dark-text-primary">scipy.special.softmax</code>,{' '}
          <code className="text-dark-text-primary">tf.nn.softmax</code> all ship the stable
          version). But if you ever hand-roll it — you&apos;ll write the bug.
        </p>
        <p>
          <strong className="text-term-amber">Softmax + cross-entropy are fused</strong> in
          PyTorch for even better stability. Use{' '}
          <code className="text-dark-text-primary">nn.CrossEntropyLoss</code> which takes raw
          logits, not <code className="text-dark-text-primary">nn.Softmax</code> followed by{' '}
          <code className="text-dark-text-primary">nn.NLLLoss</code>. Next lesson unpacks why.
        </p>
        <p>
          <strong className="text-term-amber">Softmax over one class is identity.</strong>{' '}
          If you find yourself applying softmax to a single-logit output (for regression or
          binary classification), stop — you want{' '}
          <code className="text-dark-text-primary">sigmoid</code> instead. Softmax at K=2 is a
          reparameterization of sigmoid with one redundant parameter.
        </p>
      </Gotcha>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, three implementations of the same function. You&apos;ve
          seen the pollster work by hand; now write it in pure Python, then
          NumPy, then PyTorch. Same function, progressively less of it
          visible.
        </p>
      </Prose>

      <LayeredCode
        layers={[
          {
            label: 'pure python',
            caption: 'softmax_scratch.py',
            runnable: true,
            code: `import math

def softmax(z, temperature=1.0):
    z = [v / temperature for v in z]
    m = max(z)                                  # the stability trick
    exps = [math.exp(v - m) for v in z]         # all exponents ≤ 0
    s = sum(exps)
    return [e / s for e in exps]

probs = softmax([2.0, 1.2, 0.3, -0.8, -2.0])
print("probs=", [round(p, 4) for p in probs])
print("sum=", round(sum(probs), 4))`,
            output: `probs=[0.6439, 0.2896, 0.0466, 0.0155, 0.0044]
sum=1.0`,
          },
          {
            label: 'numpy',
            caption: 'softmax_numpy.py',
            runnable: true,
            code: `import numpy as np

def softmax(z, temperature=1.0, axis=-1):
    z = z / temperature
    z = z - np.max(z, axis=axis, keepdims=True)       # broadcast-safe subtraction
    exps = np.exp(z)
    return exps / np.sum(exps, axis=axis, keepdims=True)

batch = np.array([
    [2.0, 1.2, 0.3, -0.8, -2.0],
    [0.1, 0.1, 0.1, 0.1, 0.1],                        # flat → uniform
])
print(np.round(softmax(batch), 4))
# -> [[0.6439 0.2896 0.0466 0.0155 0.0044]
#     [0.2    0.2    0.2    0.2    0.2   ]]`,
          },
          {
            label: 'pytorch',
            caption: 'softmax_pytorch.py',
            code: `import torch
import torch.nn.functional as F

logits = torch.tensor([
    [2.0, 1.2, 0.3, -0.8, -2.0],
    [0.1, 0.1, 0.1, 0.1, 0.1],
])

# torch.softmax and F.softmax are the same function — pick whichever you like.
probs = F.softmax(logits, dim=-1)           # dim=-1 = classes axis
print(torch.round(probs, decimals=4))

# log-softmax is also its own op — more stable when the next step is a log.
log_probs = F.log_softmax(logits, dim=-1)   # log(softmax(x)) without the intermediate`,
            output: `tensor([[0.6439, 0.2896, 0.0466, 0.0155, 0.0044],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000]])`,
          },
        ]}
      />

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'm = max(z); exps = [math.exp(v - m) for v in z]',
            right: 'np.exp(z - np.max(z, axis=-1, keepdims=True))',
            note: 'broadcasting along the class axis — works for batches for free',
          },
          {
            left: 's = sum(exps); return [e / s for e in exps]',
            right: 'exps / exps.sum(axis=-1, keepdims=True)',
            note: 'vectorised normalisation, no loops',
          },
        ]}
      />

      <Callout variant="insight" title="why log-softmax is its own function">
        If your loss is <code>−log p_target</code> (it almost always is — that&apos;s the next
        lesson), computing <code>log(softmax(x))</code> via two separate calls is wasteful and
        less stable. The combined <code>log-softmax</code> operation simplifies to{' '}
        <code>z_i − max(z) − log Σⱼ exp(z_j − max(z))</code>, which avoids the exponential-then-
        log round trip entirely. That&apos;s why PyTorch ships it separately.
      </Callout>

      <Challenge prompt="Build GPT-style temperature sampling">
        <p>
          Take a vector of logits, apply temperature-softmax, and sample tokens. Tweak{' '}
          <code>T</code> below and re-run: low temperature collapses onto the argmax, high
          temperature flattens the distribution toward uniform. This is the sampling loop at the
          heart of every LLM deployment in existence.
        </p>
        <p className="mt-2 mb-3 text-dark-text-muted">
          Bonus: uncomment the top-k block — zero out every probability except the top{' '}
          <code>k</code>, renormalise, then sample. Watch the diversity collapse.
        </p>
        <CodeBlock runnable language="python" caption="starter · temperature_sampling.py">{`import numpy as np

# A tiny vocabulary so the output is readable.
vocab   = ["the", "a", "cat", "dog", "sat", "ran", "slept", "<eos>"]
logits  = np.array([3.0, 2.5, 1.8, 1.2, 0.9, 0.6, 0.2, -0.4])

def softmax_T(logits, T):
    z = logits / T
    z = z - z.max()                 # shift for numerical stability
    p = np.exp(z)
    return p / p.sum()

def sample(probs, n, rng):
    return rng.choice(len(probs), size=n, p=probs)

rng = np.random.default_rng(0)

for T in (0.3, 1.0, 2.0):
    probs  = softmax_T(logits, T)
    draws  = sample(probs, n=200, rng=rng)
    counts = np.bincount(draws, minlength=len(vocab))
    print(f"T = {T:>3}    {dict(zip(vocab, counts))}")

# Uncomment to try top-k sampling:
# k = 3
# probs = softmax_T(logits, T=1.0)
# top   = np.argsort(probs)[-k:]
# mask  = np.zeros_like(probs); mask[top] = probs[top]
# probs = mask / mask.sum()
# print("top-k:", dict(zip(vocab, sample(probs, 200, rng))))
`}</CodeBlock>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> Softmax is the pollster —
          exponentiate to force positivity, normalise to make the percentages
          sum to one, and you have a real probability distribution. Temperature
          is the pollster&apos;s conviction dial, from landslide to wide-open
          race. Never implement softmax without the shift-by-max trick unless
          you enjoy <code>NaN</code>. And in PyTorch the three names
          you&apos;ll reach for are <code>F.softmax</code> (get probabilities),{' '}
          <code>F.log_softmax</code> (better for losses), and{' '}
          <code>nn.CrossEntropyLoss</code> (the fused, production-safe combo).
        </p>
        <p>
          <strong>Next up — Cross-Entropy Loss.</strong> The pollster hands
          you a distribution. Fine. But how wrong is it? You need a single
          number that&apos;s small when the model puts most of its mass on
          the correct class and large when it confidently picks the wrong
          one — a{' '}
          <NeedsBackground slug="gradient-descent">loss</NeedsBackground>{' '}
          you can actually{' '}
          <NeedsBackground slug="gradient-descent">minimize</NeedsBackground>.
          That&apos;s cross-entropy, and it&apos;s the piece that lets you
          grade the pollster&apos;s work and send{' '}
          <NeedsBackground slug="gradient-descent">gradients</NeedsBackground>{' '}
          back to every{' '}
          <NeedsBackground slug="gradient-descent">parameter</NeedsBackground>{' '}
          in the network. Up next.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Dive into Deep Learning — 3.4 Softmax Regression',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_linear-classification/softmax-regression.html',
          },
          {
            title: 'Deep Learning — 6.2.2 Softmax Units',
            author: 'Goodfellow, Bengio, Courville',
            venue: 'MIT Press, 2016',
            url: 'https://www.deeplearningbook.org/contents/mlp.html',
          },
          {
            title: 'The Curious Case of Neural Text Degeneration',
            author: 'Holtzman, Buys, Du, Forbes, Choi',
            venue: 'ICLR 2020 — where nucleus sampling was introduced',
            url: 'https://arxiv.org/abs/1904.09751',
          },
        ]}
      />
    </div>
  )
}
