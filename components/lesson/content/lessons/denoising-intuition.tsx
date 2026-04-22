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
  AsciiBlock,
} from '../primitives'
import NoiseStaircase from '../widgets/NoiseStaircase'
import DenoisingTaskSim from '../widgets/DenoisingTaskSim'

// Signature anchor: Michelangelo and the sculptor staring at a block of marble.
// "The statue is already in the stone; I just remove what isn't David." Diffusion
// is that, but the marble is a screen of TV static and the statue is the image.
// Anchor returns at the opening (static screen vs emerging statue), the reveal
// that training teaches the chisel what's noise vs what's signal, and the
// "why can't we do this in one step" section.
export default function DenoisingIntuitionLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="denoising-intuition" />

      {/* ── Opening: the sculptor ─────────────────────────────── */}
      <Prose>
        <p>
          Michelangelo, asked how he carved the David out of a block of Carrara
          marble, is supposed to have said: <em>the statue is already in the
          stone; I just remove what isn&apos;t David</em>. Whether he actually
          said it or some biographer wrote it in for him is beside the point —
          the image is perfect. A sculptor stares at a featureless block, and
          chip by chip, removes everything that isn&apos;t the figure. The
          statue doesn&apos;t get built. It gets <em>uncovered</em>.
        </p>
        <p>
          Keep that image. Now swap the marble for a television tuned to a dead
          channel — a full screen of grey, hissing static, the kind of snow
          that used to fill CRTs at 3 a.m. between broadcasts. And swap the
          sculptor&apos;s chisel for a neural network. That&apos;s a diffusion
          model. It stares at a rectangle of pure static, decides what in the
          static isn&apos;t part of the final image, chisels a little of it
          away, and repeats. Do this a thousand times and a photograph emerges
          — a cat, a castle, your cousin Linda — out of what started as
          nothing but noise.
        </p>
        <p>
          The generative modeling problem, stated plainly: somewhere out there
          is a distribution <code>p(x)</code> — the set of all photographs a
          camera could plausibly take, the set of all English sentences a
          person might write, the set of all protein folds that don&apos;t
          collapse on themselves. You have a finite pile of samples from it.
          You want a machine that, on demand, produces a <em>new</em> sample
          that looks like it came from the same distribution but isn&apos;t a
          copy of anything in your pile. A new face, a new paragraph, a new
          molecule.
        </p>
        <p>
          This is a monstrously hard ask. The space of possible images at{' '}
          <code>256 × 256 × 3</code> resolution has{' '}
          <code>256<sup>196608</sup></code> points in it, and the tiny sliver
          that looks like a real photo is, in relative terms, essentially zero
          volume. You are asking a model to find the needle while the haystack
          is most of the observable universe.
        </p>
        <p>
          <KeyTerm>Diffusion models</KeyTerm> solve this with the sculptor&apos;s
          move. Take a clean image. Add a little Gaussian noise. Add a little
          more. Keep adding for a thousand steps until the image is
          indistinguishable from the static on that dead channel. Now train a
          neural network to undo <em>one</em> step of that process — one
          chisel stroke, no more. That&apos;s it. At generation time you start
          with a fresh screen of static and run the chisel a thousand times.
          Out the other end falls a statue.
        </p>
      </Prose>

      <Callout variant="insight" title="what the sculptor knows">
        Michelangelo didn&apos;t need to invent the David. He needed to know,
        for every fleck of marble, whether it was part of the figure or part
        of the waste. That&apos;s the only job. Training a diffusion model is
        teaching the chisel that same distinction: given a noisy patch of
        pixels, which flecks are signal and which are noise? The marble-vs-David
        judgement, done at the pixel level, a thousand times per image. Every
        other complication — the U-Net, the schedules, the guidance tricks — is
        bookkeeping around that single discrimination.
      </Callout>

      <Personify speaker="Diffusion">
        I do not compress. I do not compete. I take a path you can see — clean
        image to pure static, one chisel stroke at a time — and I teach a
        network to walk it backward. Every single stroke is easy. The whole
        marathon is what looks like magic.
      </Personify>

      {/* ── ASCII of forward / reverse ─────────────────────────── */}
      <AsciiBlock caption="the two processes — one buries the statue, one uncovers it">
{`     forward  (fixed, no learning)            reverse  (learned chisel)
     ─────────────────────────────>            <─────────────────────────────
      x₀ ─► x₁ ─► x₂ ─► ··· ─► xₜ               xₜ ◄─ xₜ₋₁ ◄─ ··· ◄─ x₁ ◄─ x₀
     clean    +ε    +ε         pure            static      chisel          clean
     image          noise      static          (marble)                    image
                                                                        (the statue)

                   (entropy ↑)                             (entropy ↓, learned)`}
      </AsciiBlock>

      {/* ── Widget 1: Noise Staircase ──────────────────────────── */}
      <Prose>
        <p>
          Here is the forward process, visually. One image, ten rungs, each
          rung a little noisier than the last — the sculptor&apos;s process in
          reverse, the statue being slowly buried under chips of marble until
          only the block remains. Slide left to right and you watch a cat
          dissolve into static. Slide right to left — the arrow labelled{' '}
          <em>denoising</em> — and you watch the path the chisel is trying to
          learn. Forward buries the statue. Reverse uncovers it.
        </p>
      </Prose>

      <NoiseStaircase />

      <Prose>
        <p>
          The key insight: each rung of this staircase differs from its
          neighbour by only a tiny amount of noise. Predicting{' '}
          <code>x<sub>t-1</sub></code> from <code>x<sub>t</sub></code> is not
          a mystical leap — it&apos;s a near-identity map with a small
          correction. The far ends look nothing alike (clean cat vs hissing
          static), but any two adjacent rungs look nearly identical. You&apos;re
          not asking the chisel to carve the David out of a raw block in one
          swing. You&apos;re asking it to take a David that&apos;s 99.9%
          finished and flick off one last sliver of marble.
        </p>
        <p>
          That&apos;s why this works and VAEs struggled and GANs fought each
          other. The denoising objective splits an impossibly hard problem —
          sample from <code>p(x)</code> — into <em>T</em> tractable
          sub-problems, each one a short-range regression. You walk down an
          easy staircase instead of parachuting out of a helicopter.
        </p>
      </Prose>

      <Personify speaker="Noise schedule">
        I&apos;m the timekeeper. I decide how much marble the sculptor covers
        the statue in at each step — a dusting at first, then bigger handfuls
        toward the end. Linear, cosine, learned, it&apos;s all me. Get me
        wrong and your chisel has nothing to learn (strokes too small) or
        everything to learn at once (strokes too big). I don&apos;t get credit
        in the paper but I run the show.
      </Personify>

      {/* ── Why one step isn't allowed ─────────────────────────── */}
      <Prose>
        <p>
          An obvious question at this point: if the chisel is so good, why
          can&apos;t we just do this in one swing? Point the network at the
          block of static, say &ldquo;give me the cat,&rdquo; and let it rip.
          Why a thousand strokes instead of one?
        </p>
        <p>
          Because the one-stroke version is exactly the problem diffusion was
          invented to avoid. Asking a network to map pure static directly to a
          photograph is asking Michelangelo to look at an uncarved block and
          chisel the David in a single motion, blindfolded, with the power of
          his mind. The mapping from <code>𝒩(0, I)</code> to the
          distribution of real images is wildly non-linear, multi-modal, and
          contested — for any patch of static, there are a trillion equally
          plausible photographs it could become. A one-shot model has to pick
          one of those trillion on the spot, with no context, no intermediate
          goalpost, nothing. GANs tried this. Their training was an absolute
          horror show for exactly this reason.
        </p>
        <p>
          Split the job into a thousand strokes and the math cooperates. Each
          stroke looks at marble that&apos;s already most of the way to being
          a statue and chips away a sliver. The ambiguity is local — &ldquo;is
          this particular fleck signal or waste?&rdquo; — not global. A small
          network with a plain MSE loss can learn that. Chain a thousand of
          those small learned moves together and the composition recovers the
          full distribution. The sculptor does the David one chisel stroke at
          a time because no sculptor, human or mechanical, can do it in one.
        </p>
      </Prose>

      {/* ── Forward process math ───────────────────────────────── */}
      <Prose>
        <p>
          One equation, no derivation. This is the forward process — the rule
          that defines the staircase, the rate at which the sculptor buries
          the statue under chips of marble:
        </p>
      </Prose>

      <MathBlock caption="one step of Gaussian corruption — the whole forward process">
{`q(xₜ | xₜ₋₁)  =  𝒩( xₜ ;  √(1 − βₜ) · xₜ₋₁ ,  βₜ · I )

          ┌──────────────────┐          ┌─────────────┐
          │  scale the old   │          │   add noise │
          │  image down a    │          │   variance  │
          │  little          │          │   βₜ        │
          └──────────────────┘          └─────────────┘

       where  βₜ  is small  (~10⁻⁴ at t=1, ~0.02 at t=T)`}
      </MathBlock>

      <Prose>
        <p>
          Read it as a sentence: <em>to get the image at step t, take the image
          at step t−1, shrink it slightly, and sprinkle in a dash of Gaussian
          noise.</em> That&apos;s the forward process in full. There is no
          neural network here. There is no training. It is a fixed,
          hand-specified recipe for burying a statue under marble dust one
          sprinkle at a time.
        </p>
        <p>
          Because <code>β<sub>t</sub></code> is small, one step is barely
          visible. Because you do it a thousand times, the end state{' '}
          <code>x<sub>T</sub></code> is indistinguishable from standard
          Gaussian noise — <code>𝒩(0, I)</code>. That&apos;s the crucial
          property: the static on the top rung of the staircase is a
          distribution you already know how to sample from. Draw random
          numbers, start chiselling.
        </p>
      </Prose>

      {/* ── Widget 2: Denoising Task Sim ───────────────────────── */}
      <Prose>
        <p>
          Images are hard to reason about in 196,608 dimensions. Let&apos;s
          drop to two. Below is a 2D Gaussian blob — stand-in for any
          structured data distribution, or if you prefer, a flat-projection
          David. Watch noise bury it into a featureless cloud of static. Then
          watch a learned chisel, one stroke at a time, pull the statue back
          out.
        </p>
      </Prose>

      <DenoisingTaskSim />

      <Prose>
        <p>
          Two things to notice. First, during the forward phase the blob
          doesn&apos;t vanish suddenly — it widens, flattens, blurs into the
          background. Each stroke of the sculptor&apos;s hammer covers up a
          sliver. Second, during the reverse phase the chisel isn&apos;t
          drawing the blob from thin air — it&apos;s nudging points, one
          stroke at a time, back toward where the statue lived. The blob
          re-emerges in the right place because the chisel has, during
          training, learned where the data <em>tends</em> to sit. It knows
          what&apos;s signal and what&apos;s marble waste.
        </p>
        <p>
          That&apos;s the geometric picture. The data distribution is a thin
          manifold in a high-dimensional space — the statue, hiding inside the
          block. The forward process smears mass off the manifold into the
          whole space, burying the statue. The reverse process is a learned
          vector field pointing back toward it. Generation is a walk along
          that field starting from a random point in the static.
        </p>
      </Prose>

      <Personify speaker="The chisel">
        Give me a noisy tensor and a timestep, and I&apos;ll tell you what
        marble to flick away. I don&apos;t know what a cat is. I don&apos;t
        know what a sentence is. I know what the statue tends to look like
        and what the block tends to look like at every stage in between, and
        I close the gap. Repeat me a thousand times and you get a David.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          You&apos;ve seen it move. Now write it three times, each layer a bit
          closer to the real thing. First the noise schedule on a single scalar
          — the arithmetic laid bare. Then a 2D <NeedsBackground slug="mlp-from-scratch">neural network</NeedsBackground>{' '}
          toy in NumPy where we actually destroy and reconstruct a Gaussian.
          Finally, the PyTorch skeleton of the real algorithm, stripped to
          bones; the DDPM lesson later fills in the rest. Each layer uses{' '}
          <NeedsBackground slug="gradient-descent">training</NeedsBackground>{' '}
          — the same loop you already know — with a comically simple loss
          function.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · noise_schedule.py"
        output={`t=0   β=0.0001   x=1.0000   noise var so far=0.0000
t=250 β=0.0051   x=0.2873   noise var so far=0.9175
t=500 β=0.0101   x=0.0112   noise var so far=0.9999
t=750 β=0.0151   x=0.0001   noise var so far=1.0000
t=999 β=0.0200   x=0.0000   noise var so far=1.0000`}
      >{`import math, random

T = 1000
beta_start, beta_end = 1e-4, 2e-2
betas = [beta_start + (beta_end - beta_start) * t / (T - 1) for t in range(T)]

# Forward process on a single scalar — start at x=1.0, bury it step by step.
random.seed(0)
x = 1.0
var_accum = 0.0                         # variance of noise accumulated so far
for t in range(T):
    x = math.sqrt(1 - betas[t]) * x + math.sqrt(betas[t]) * random.gauss(0, 1)
    var_accum = (1 - betas[t]) * var_accum + betas[t]
    if t in (0, 250, 500, 750, 999):
        print(f"t={t:<4} β={betas[t]:.4f}   x={x:+.4f}   "
              f"noise var so far={var_accum:.4f}")`}</CodeBlock>

      <Prose>
        <p>
          The original signal decays, the accumulated noise variance saturates
          at 1. By step 1000 the scalar is, for all intents and purposes, a
          draw from <code>𝒩(0, 1)</code>. Whatever it was at the start is
          gone — statue completely buried under marble. That&apos;s the forward
          process finishing its job.
        </p>
        <p>
          Step up a dimension. In NumPy we can actually destroy a 2D
          distribution and train a tiny chisel to put it back. This is the
          real algorithm in miniature — same losses, same schedule, same
          sampling loop, just tractable enough to run in a notebook.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — numpy · diffusion_2d.py">{`import numpy as np

rng = np.random.default_rng(0)
T = 100
betas = np.linspace(1e-4, 2e-2, T)
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)           # α̅ₜ = ∏ (1 - βᵢ)

# True data: a tight 2D Gaussian blob.  Pretend this is an image dataset.
x0 = rng.normal(loc=[2.0, -1.0], scale=0.3, size=(1024, 2))

# Forward in closed form — no looping needed.
# xₜ = √α̅ₜ · x₀ + √(1 - α̅ₜ) · ε       (this identity falls out of the schedule)
def q_sample(x0, t):
    ab = alpha_bar[t]
    eps = rng.standard_normal(x0.shape)
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps, eps

# A trivially-parametric "chisel": a learned mean for each t.
# Stand-in for a neural net — shows the shape of the problem.
mu = np.zeros((T, 2))
for step in range(5_000):
    t = rng.integers(0, T)
    xt, eps = q_sample(x0, t)
    pred = np.mean(xt, axis=0)           # our "model" predicts the batch mean
    mu[t] = 0.99 * mu[t] + 0.01 * pred   # EMA update — cartoon of gradient descent

# Reverse: sample pure static, chisel step by step using learned means.
x = rng.standard_normal((256, 2))
for t in reversed(range(T)):
    x = (x - np.sqrt(1 - alpha_bar[t]) * (x - mu[t])) / np.sqrt(alpha_bar[t])
print("recovered mean:", np.round(x.mean(axis=0), 2), "   target: [2. -1.]")`}</CodeBlock>

      <Prose>
        <p>
          That&apos;s the whole pipeline compressed to 20 lines. A real DDPM
          swaps the <code>mu[t]</code> table for a U-Net that takes{' '}
          <code>(x<sub>t</sub>, t)</code> and returns the noise to subtract —
          the chisel stroke for that timestep — but the scaffolding
          (schedule, forward equation, reverse loop) is exactly this. When you
          read Ho et al. 2020, you&apos;re reading a more careful, conditional,
          parameterised version of the snippet above. For images the chisel
          uses <NeedsBackground slug="convolution-operation">convolutions</NeedsBackground>{' '}
          because they respect the 2D locality of pixels; the same scaffolding
          holds.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 3 — pytorch skeleton · ddpm_sketch.py">{`import torch
import torch.nn as nn
import torch.nn.functional as F

T = 1000
betas = torch.linspace(1e-4, 2e-2, T)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

class Denoiser(nn.Module):
    """Predicts the noise ε added to x₀ to produce xₜ.  A U-Net in real life."""
    def __init__(self):
        super().__init__()
        # ...architecture lives here; filled out in the DDPM lesson...

    def forward(self, x_t, t):
        # Returns ε̂ with shape matching x_t.
        ...

model = Denoiser()
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

# Training step — one of the simplest losses in deep learning.
def train_step(x0):
    t = torch.randint(0, T, (x0.size(0),))
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps     # forward, closed-form
    eps_hat = model(x_t, t)
    loss = F.mse_loss(eps_hat, eps)                  # predict the noise — that's it
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

# Sampling — what the user sees at inference time.
@torch.no_grad()
def sample(shape):
    x = torch.randn(shape)                           # start from pure static
    for t in reversed(range(T)):
        eps_hat = model(x, torch.full((shape[0],), t))
        # ... plug eps_hat into the reverse mean/variance formulas ...
    return x`}</CodeBlock>

      <Bridge
        label="pure python → numpy → pytorch"
        rows={[
          {
            left: 'scalar: x = √(1-β)·x + √β·ε',
            right: 'batched: x_t = √α̅·x₀ + √(1-α̅)·ε',
            note: 'one step per call → one call per timestep (α̅ folds T steps into one)',
          },
          {
            left: 'learn a table mu[t]',
            right: 'learn a network eps_hat = model(x_t, t)',
            note: 'the toy table becomes a U-Net conditioned on the timestep',
          },
          {
            left: 'reverse loop on NumPy array',
            right: '@torch.no_grad() sampling loop on GPU',
            note: 'same logic; autograd off because we aren\'t training during sampling',
          },
        ]}
      />

      <Callout variant="insight" title="one loss function, no games">
        The training loss is <code>F.mse_loss(eps_hat, eps)</code>. That&apos;s
        a regression between two tensors. There is no discriminator to fight
        with, no KL term to balance, no mode-collapse failure mode. You buried
        the statue, you ask the chisel to guess the marble dust it would take
        to unbury it, you take one step of SGD. That&apos;s the entire
        training signal. Stability at this level is why diffusion took over
        image generation.
      </Callout>

      {/* ── Comparison callouts ─────────────────────────────────── */}
      <Callout variant="note" title="versus VAE">
        A variational autoencoder compresses the input to a low-dim latent{' '}
        <code>z</code>, then decodes. Training balances reconstruction against
        a KL term that pulls <code>z</code> toward a Gaussian. The trade-off
        is painful — push too hard on KL and samples blur; push too little
        and the latent isn&apos;t Gaussian and you can&apos;t sample from it
        cleanly. Diffusion sidesteps the whole dilemma: the &ldquo;latent&rdquo;
        is just pure static, guaranteed Gaussian by construction, no KL
        negotiation required.
      </Callout>

      <Callout variant="note" title="versus GAN">
        A generative adversarial network pits a generator against a
        discriminator — one tries to sculpt a statue in a single swing of the
        mallet, the other tries to call it fake. When it works, sample quality
        is stunning. When it doesn&apos;t, you get mode collapse (the generator
        cheats by carving one pose the discriminator can&apos;t classify),
        oscillation, or silent divergence. Hyperparameters matter more than
        they should. Diffusion swaps the two-player game for a thousand small
        chisel strokes and an MSE loss — slower to sample, dramatically more
        stable to train.
      </Callout>

      <Callout variant="insight" title="thermodynamic origin">
        The name <em>diffusion</em> is borrowed directly from physics. The
        forward process is an entropy-increasing random walk — a drop of ink
        spreading in water, or the dust of a pulverised statue scattering in
        the wind. The reverse process is the thing that&apos;s supposed to
        be impossible by the second law: watching the dust reassemble itself
        into the figure. Sohl-Dickstein&apos;s 2015 paper argued you can learn
        it from examples, frame by frame, because each frame-to-frame reversal
        is a tractable Gaussian problem. Ho 2020 showed the engineering to
        make it actually work at image scale. Every Stable Diffusion and
        DALL-E checkpoint you&apos;ve ever used is a direct descendant of
        that pair of papers.
      </Callout>

      {/* ── T matters ──────────────────────────────────────────── */}
      <Prose>
        <p>
          How many chisel strokes should the staircase have? More strokes mean
          each one is smaller, which means the chisel has an easier regression
          target at every single step. Quality goes up. But generation time
          scales linearly with <code>T</code> — 1000 strokes is 1000 forward
          passes of the U-Net, which for a big model is measured in seconds
          or tens of seconds per sample.
        </p>
        <p>
          Original DDPM used <code>T = 1000</code>. Modern accelerated samplers
          — DDIM, DPM-Solver, and their descendants — reframe the reverse
          process as an ODE you can integrate with 50 or 20 or even 4 strokes,
          recovering most of the quality at a fraction of the cost. The
          training schedule stays at T=1000; only inference gets cheap.
          We&apos;ll spell that out in the samplers lesson.
        </p>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">&ldquo;Predict the noise&rdquo;
          vs &ldquo;predict the clean image&rdquo;:</strong> both
          parameterisations work — they&apos;re algebraically equivalent given
          the schedule. Most implementations predict <code>ε</code> (the marble
          dust) because the loss landscape is friendlier; some predict{' '}
          <code>x₀</code> (the finished statue) directly; some predict{' '}
          <code>v = α̅<sup>½</sup>ε − (1−α̅)<sup>½</sup>x₀</code> (the
          v-prediction from Progressive Distillation). Pick one and be
          consistent. Mixing them silently is the kind of bug that burns a
          week.
        </p>
        <p>
          <strong className="text-term-amber">Schedule bugs:</strong>{' '}
          off-by-one indexing on <code>α̅<sub>t</sub></code>, using{' '}
          <code>β</code> where the code expects <code>α = 1 − β</code>,
          computing <code>cumprod</code> in the wrong direction. These produce
          training runs that look like they&apos;re working — loss goes down —
          and then sample pure static at inference. Always validate by
          decoding a known noisy sample end-to-end before touching the model.
        </p>
        <p>
          <strong className="text-term-amber">T too small:</strong> if the
          forward process doesn&apos;t fully bury the statue by step{' '}
          <code>T</code>, <code>x<sub>T</sub></code> is not Gaussian, and
          sampling from <code>𝒩(0, I)</code> at inference means the reverse
          process starts from the wrong block of marble. Quality drops.
          Confirm by pushing a real image all the way through the forward
          schedule and measuring its mean and variance; it should look like
          pure static.
        </p>
      </Gotcha>

      {/* ── Challenge ─────────────────────────────────────────── */}
      <Challenge prompt="Chisel a 2D spiral out of static">
        <p>
          Generate 2048 points along a 2D Archimedean spiral — call this your
          data distribution <code>x₀</code>, the statue to uncover. Build a
          small MLP (3 hidden layers, 128 units, SiLU) that takes{' '}
          <code>(x<sub>t</sub>, t)</code> as input and outputs a noise
          prediction in ℝ². Train it for 10,000 steps with the DDPM loss from
          layer 3.
        </p>
        <p className="mt-2">
          Now sample 512 points from <code>𝒩(0, I)</code> — 512 tiny blocks
          of static — and run the reverse process three times: once with{' '}
          <code>T = 10</code> chisel strokes, once with <code>T = 50</code>,
          once with <code>T = 1000</code>. Scatter-plot each against the
          original spiral.
        </p>
        <p className="mt-2 text-dark-text-muted">
          With T=10 you&apos;ll see a smeary blob near the spiral — the
          sculptor cut too much in each swing. At T=50 the spiral shape starts
          to emerge. At T=1000 it should be crisp and distributed all along
          the curve. That&apos;s the bias-variance trade-off of the step
          count, visualised on something you can plot.
        </p>
      </Challenge>

      {/* ── Closing + teaser ────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Diffusion splits generative
          modeling into a fixed corruption process and a learned inverse —
          the sculptor burying the statue under marble, and the chisel
          uncovering it again. Each stroke of the chisel is a small, local
          denoising job. Training is plain MSE against the noise you added —
          no adversary, no latent-space bargaining. Generation is a walk from{' '}
          <code>𝒩(0, I)</code> down the staircase, with the learned chisel as
          your guide at each rung. Everything else — the U-Net, the attention
          conditioning, the classifier-free guidance, the latent-space
          compression that makes Stable Diffusion fast — is engineering on
          top of this one idea.
        </p>
        <p>
          <strong>Next up — Forward &amp; Reverse Diffusion.</strong>{' '}
          We&apos;ve been loose with the math so far, reading equations like
          sentences. In <em>forward-and-reverse-diffusion</em> we derive them
          properly. The closed-form expression for{' '}
          <code>q(x<sub>t</sub> | x<sub>0</sub>)</code> — why you don&apos;t
          actually loop T times during training. The posterior{' '}
          <code>q(x<sub>t-1</sub> | x<sub>t</sub>, x<sub>0</sub>)</code>{' '}
          that gives the reverse process its mean and variance. The ELBO that
          collapses, miraculously, into the MSE loss we just wrote. A single
          page of algebra that turns &ldquo;chisel away the noise&rdquo; from
          a metaphor into a theorem.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Deep Unsupervised Learning using Nonequilibrium Thermodynamics',
            author: 'Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli',
            venue: 'ICML 2015 — the diffusion-as-generative-model original',
            url: 'https://arxiv.org/abs/1503.03585',
          },
          {
            title: 'Denoising Diffusion Probabilistic Models',
            author: 'Ho, Jain, Abbeel',
            venue: 'NeurIPS 2020 — DDPM, the paper that made it work',
            url: 'https://arxiv.org/abs/2006.11239',
          },
          {
            title: 'Dive into Deep Learning — Chapter 23, Diffusion Models',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_generative-adversarial-networks/index.html',
          },
          {
            title: 'Denoising Diffusion Implicit Models',
            author: 'Song, Meng, Ermon',
            venue: 'ICLR 2021 — DDIM, the accelerated sampler',
            url: 'https://arxiv.org/abs/2010.02502',
          },
          {
            title: 'DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling',
            author: 'Lu, Zhou, Bao, Chen, Li, Zhu',
            venue: 'NeurIPS 2022',
            url: 'https://arxiv.org/abs/2206.00927',
          },
        ]}
      />
    </div>
  )
}
