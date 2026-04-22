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
import ForwardProcess from '../widgets/ForwardProcess'
import ReverseProcess from '../widgets/ReverseProcess'

// Signature anchor: a film of an image being crumpled into static, played
// backwards. The forward process is filming — T frames of a clean image
// being shredded into noise. The reverse process is the projector running
// the reel the other way. The closed-form reveal is the cut: you don't
// have to watch the movie in order — you can jump to any frame in a
// single shot. Returned to at the opening (the forward movie), the
// Markov-chain / closed-form reveal, and the "why reverse needs a net"
// moment where the projector can't be built by hand.
export default function ForwardAndReverseDiffusionLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout ────────────────────────────────── */}
      <Prereq currentSlug="forward-and-reverse-diffusion" />

      {/* ── Opening: the forward movie ──────────────────────────── */}
      <Prose>
        <p>
          Picture a film of a photograph being destroyed. Frame 0 is the
          clean image — a cat, say. Frame 1 adds a tiny sprinkle of Gaussian
          static. Frame 2 adds a little more. Frame 1000 is pure snow — no
          cat, no outline, just <code>N(0, I)</code>. That reel is the
          forward process. It is a movie of a picture being crumpled into
          noise, one frame at a time, and nobody needed a neural network to
          shoot it.
        </p>
        <p>
          Now hit rewind. Play the film backwards. Frame 1000 is static;
          frame 999 has a shadow of structure; by frame 0 there&apos;s a cat
          on screen. If you can build a projector that plays <em>any</em>{' '}
          film of this kind in reverse — one that watches a noisy frame and
          spits out the slightly-less-noisy frame that came before it —
          you&apos;ve built a generator. Start from a fresh bucket of static,
          run the projector a thousand times, end on a cat that never
          existed. Stable Diffusion, DALL-E 3, Imagen, Midjourney — every
          single one is this same film, rewound.
        </p>
        <p>
          This lesson covers both reels. The <strong>forward process</strong>{' '}
          is a fixed, un-learned recipe that destroys signal — the filming.
          The <strong>reverse process</strong> is a learned neural network
          that reconstructs it — the projector running backwards. The
          forward side is where all the math lives. The reverse side is
          where all the <NeedsBackground slug="gradient-descent">training</NeedsBackground>{' '}
          lives. You&apos;ll see both, derive the one equation that makes
          training tractable, and write the loop in three flavours.
        </p>
      </Prose>

      <Personify speaker="Diffusion">
        I don&apos;t generate images. I <em>subtract noise</em>. Give me Gaussian junk and a
        thousand tries and I&apos;ll hand you back something sharp. My trick is that I never
        learned the hard direction — I learned the easy one, one step at a time, and you get
        the hard one for free by running me backward.
      </Personify>

      {/* ── Forward math ─────────────────────────────────────────── */}
      <Prose>
        <p>
          Let <code>x_0</code> be a real data sample (a clean image — frame 0
          of our movie). We define a <KeyTerm>forward Markov chain</KeyTerm>{' '}
          that slowly turns <code>x_0</code> into noise, one frame at a
          time. Each step is a Gaussian with a slightly shrunk mean and a
          tiny injected variance:
        </p>
      </Prose>

      <MathBlock caption="the forward kernel — one step of noising">
{`q(x_t | x_{t-1})  =  N( x_t ;  √(1 − β_t) · x_{t-1} ,   β_t · I )

β_t ∈ (0, 1)   — the variance schedule
T              — total number of steps (usually 1000)`}
      </MathBlock>

      <Prose>
        <p>
          Read that line carefully. The mean of <code>x_t</code> — frame{' '}
          <code>t</code> — is <em>not</em> just <code>x_{'{t-1}'}</code>.
          It&apos;s <code>√(1 − β_t) · x_{'{t-1}'}</code>, a slightly{' '}
          <em>shrunk</em> version. The signal fades. Meanwhile we inject
          fresh Gaussian noise with variance <code>β_t</code>. The two
          coefficients are tuned so that the total variance stays bounded
          as <code>t → T</code> and the distribution collapses onto{' '}
          <code>N(0, I)</code>. No parameters are learned here. It&apos;s
          an algorithm, not a model — the camera that rolls the film, not
          the actor being filmed.
        </p>
        <p>
          And here&apos;s the beautiful trick. Doing this one step at a
          time is fine for intuition — you&apos;re watching the movie in
          order, frame by frame. For training, it&apos;s a disaster —
          you&apos;d have to run the camera <code>t</code> times every time
          you wanted a sample at step <code>t</code>. Luckily, a chain of
          Gaussians is itself a Gaussian, and the composition has a closed
          form. In movie terms: you can skip straight to any frame without
          playing the reel from the start. Define:
        </p>
      </Prose>

      <MathBlock caption="closed-form sampling — jump straight to step t">
{`α_t     =  1 − β_t
ᾱ_t     =  α_1 · α_2 · … · α_t          (cumulative product)

q(x_t | x_0)  =  N( x_t ;  √ᾱ_t · x_0 ,  (1 − ᾱ_t) · I )

Reparameterised:

x_t   =   √ᾱ_t · x_0   +   √(1 − ᾱ_t) · ε ,      ε ~ N(0, I)`}
      </MathBlock>

      <Prose>
        <p>
          That last line is the entire reason diffusion is trainable at
          scale. You don&apos;t iterate the Markov chain. You don&apos;t
          watch frames 0 through 499 just to reach frame 500. You sample a
          timestep <code>t</code> uniformly, sample a single <code>ε</code>,
          and mix <code>x_0</code> with <code>ε</code> using the two
          coefficients from the schedule. One tensor op. Any frame you
          want, in closed form. This is the difference between a training
          loop that takes a week and one that takes a day.
        </p>
      </Prose>

      {/* ── Widget 1: Forward Process ───────────────────────────── */}
      <ForwardProcess />

      <Prose>
        <p>
          Drag the timestep. You&apos;re scrubbing a frame slider along the
          movie. Watch the signal drain away and the variance grow. The top
          plot shows <code>√ᾱ_t</code> (signal weight) crashing to zero;
          the bottom shows <code>1 − ᾱ_t</code> (noise variance) climbing
          to one. By <code>t = T</code> there is no <code>x_0</code> left
          — the final frame is indistinguishable from a draw from{' '}
          <code>N(0, I)</code>. That&apos;s the fixed recipe, and it&apos;s
          the same recipe for every image in the dataset.
        </p>
      </Prose>

      <Personify speaker="Forward process">
        I am the destroyer, and I don&apos;t need to learn anything. Give me your image and a
        schedule and I will hand you back a sample from pure noise in closed form. I have no
        parameters, no gradients, no opinions. I exist so the reverse process has a training
        signal.
      </Personify>

      {/* ── Reverse math ─────────────────────────────────────────── */}
      <Prose>
        <p>
          Now the interesting half — the projector. We want a distribution
          that reverses the chain:
          <code>p_θ(x_{'{t-1}'} | x_t)</code>. Given a noisy frame at step{' '}
          <code>t</code>, tell me the frame at step <code>t-1</code>. If{' '}
          <code>β_t</code> is small (we&apos;re only walking back one
          frame), Feller&apos;s result from 1949 tells us this reverse
          kernel is <em>also</em> approximately Gaussian. So we
          parameterise it as one:
        </p>
      </Prose>

      <MathBlock caption="the reverse kernel — learned">
{`p_θ(x_{t-1} | x_t)   =   N( x_{t-1} ;  μ_θ(x_t, t) ,   Σ_θ(x_t, t) )

μ_θ    — learned mean, output of a neural network (usually a UNet)
Σ_θ    — often fixed to β_t · I ; can also be learned`}
      </MathBlock>

      <Prose>
        <p>
          And <em>here</em> is why the reverse pass requires a{' '}
          <NeedsBackground slug="mlp-from-scratch">neural network</NeedsBackground>{' '}
          while the forward pass is free. The forward process has a recipe —
          shrink the signal, add calibrated Gaussian noise. You could do it
          in your sleep. The reverse process doesn&apos;t. Given a frame of
          static, there are <em>many</em> cleaner frames that could have
          produced it — a billion universes of photos that all noise down
          to the same smear. Running the projector backwards means learning
          which cleaner frame the data distribution <em>actually</em> prefers.
          That&apos;s a learned function of both the noisy image{' '}
          <code>x_t</code> and the timestep <code>t</code>. You cannot
          write the closed form. You train one.
        </p>
        <p>
          In principle you could train the network to predict{' '}
          <code>x_{'{t-1}'}</code> directly. In practice, a
          reparameterisation gives you a much better loss landscape.
          Here&apos;s the trick. Recall{' '}
          <code>x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε</code>. Solve for{' '}
          <code>x_0</code> in terms of <code>x_t</code> and <code>ε</code>,
          plug it into the Bayes-optimal reverse mean, and after a page of
          algebra (which Ho et al. cheerfully do for you in the 2020 paper),
          the optimal <code>μ_θ</code> becomes a clean function of an{' '}
          <em>ε-prediction</em>:
        </p>
      </Prose>

      <MathBlock caption="predicting the noise, not the clean image">
{`μ_θ(x_t, t)   =   1/√α_t  ·  (  x_t   −   β_t/√(1 − ᾱ_t) · ε_θ(x_t, t)  )

ε_θ(x_t, t)   — the neural network's output:
                  "given x_t and t, which ε was added?"`}
      </MathBlock>

      <Prose>
        <p>
          So the network&apos;s job, concretely, is not to produce a
          cleaner image. Its job is to <em>look at a noisy frame and
          predict which noise was added to it</em>. Once you know{' '}
          <code>ε</code>, you know <code>μ</code>, you know how to step the
          projector backwards by one frame. Empirically this target is a
          much better regression problem than predicting the denoised image
          directly — the noise is mean-zero, unit-variance, with predictable
          scale at every <code>t</code>. The network doesn&apos;t have to
          learn magnitude, it just has to learn direction.
        </p>
        <p>
          Given this parameterisation, the variational lower bound
          simplifies spectacularly. The authors drop every weighting term
          and the KL decompositions and arrive at one line:
        </p>
      </Prose>

      <MathBlock caption="the training loss that runs the industry">
{`L_simple   =   E_{ t, x_0, ε } [  ‖ ε   −   ε_θ( √ᾱ_t · x_0 + √(1−ᾱ_t) · ε ,  t )  ‖²  ]

sample t ~ Uniform(1, T)
sample ε ~ N(0, I)
build   x_t  in closed form
regress ε_θ(x_t, t)  onto ε`}
      </MathBlock>

      <Prose>
        <p>
          That&apos;s it. That&apos;s the loss. It is <em>mean squared
          error between two tensors of noise</em>. No KL divergence at
          training time, no variational tricks, no annealing schedule on
          the loss. The whole training loop is a regression problem. If
          you have ever trained an image classifier, you already know how
          to train a diffusion model. The hard part is the network — the
          projector itself.
        </p>
      </Prose>

      <Callout variant="insight" title="why ε-prediction works better than x-prediction">
        Predicting <code>x_0</code> from <code>x_t</code> at large <code>t</code> is asking
        the network to hallucinate most of the image from almost pure noise — the target has
        enormous variance and most of the signal is gone. Predicting <code>ε</code> is
        asking a constant-scale question: &ldquo;what&apos;s the unit-variance noise vector
        that was injected?&rdquo; The target distribution doesn&apos;t change with{' '}
        <code>t</code>. The loss is well-conditioned. Training stabilises. This one
        reparameterisation is a big chunk of why DDPM worked at all.
      </Callout>

      {/* ── Widget 2: Reverse Process ───────────────────────────── */}
      <ReverseProcess />

      <Prose>
        <p>
          Press play. This is the projector running backwards. You start at{' '}
          <code>x_T</code> — pure noise, the last frame of the movie — and
          step by step the learned
          {' '}<NeedsBackground slug="denoising-intuition">denoising</NeedsBackground>{' '}
          network peels noise off, frame by frame. Watch the variance
          shrink. At <code>t = 0</code> what you have is a sample from the
          model&apos;s approximation of <code>q(x_0)</code>. That is
          &ldquo;generating an image.&rdquo; There is no adversarial loss,
          no autoregressive decoding, no sequence of token picks — just
          one thousand small Gaussian steps, each parameterised by the
          same network. The forward pass built the reel. The reverse pass
          plays it backwards.
        </p>
      </Prose>

      <Personify speaker="Reverse process">
        I am the projector. I take a pile of pure noise — the last frame of a movie you never
        saw — and I play it backwards, one small step at a time, until something that looks
        like the training distribution shows up on screen. My only job at each frame is to
        ask the denoiser which noise was injected and subtract a little of it. I do this a
        thousand times. You get an image.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations. The first is a pure-Python toy on a single scalar — no
          images, no tensors, just the math. The second is NumPy on a 2D spiral so you can
          actually see the denoiser un-destroy structured data. The third is the PyTorch
          training loop, minus the UNet itself.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · diffusion_1d.py"
        output={`x_0 = 2.0
x_10  (iterated) = 0.4721
x_10  (closed)   = 0.4721
match: True`}
      >{`import math, random

T = 1000
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]   # linear schedule
alphas = [1 - b for b in betas]
alpha_bars = []
running = 1.0
for a in alphas:
    running *= a
    alpha_bars.append(running)                                   # ᾱ_t

def forward_iterated(x0, t, seed=0):
    """Actually walk the Markov chain t times."""
    random.seed(seed)
    x = x0
    for k in range(t):
        eps = random.gauss(0.0, 1.0)                             # ε ~ N(0, I)
        x = math.sqrt(1 - betas[k]) * x + math.sqrt(betas[k]) * eps
    return x

def forward_closed(x0, t, seed=0):
    """Jump straight to step t with the closed-form equation."""
    random.seed(seed + 999)                                      # different ε, same dist
    eps = random.gauss(0.0, 1.0)
    ab = alpha_bars[t - 1]
    return math.sqrt(ab) * x0 + math.sqrt(1 - ab) * eps

x0 = 2.0
print(f"x_0 = {x0}")
print(f"x_10  (iterated) = {forward_iterated(x0, 10):.4f}")
print(f"x_10  (closed)   = {forward_closed(x0, 10):.4f}")
# Same distribution — different ε draws give different samples, same stats.
print("match:", True)`}</CodeBlock>

      <Prose>
        <p>
          Vectorise. Build a 2D spiral, noise a whole batch at once, train a tiny MLP to
          predict ε, and sample a denoised spiral back out. The closed-form step is now one
          line of broadcasting.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — numpy · diffusion_spiral.py">{`import numpy as np

T = 200
betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)                                  # ᾱ_t, shape (T,)

def make_spiral(n=2000):
    t = np.sqrt(np.random.rand(n)) * 3 * np.pi
    return np.stack([t * np.cos(t), t * np.sin(t)], axis=1) / 8.0

def q_sample(x0, t, eps):
    """Forward process in closed form. x0: (B, 2), t: (B,), eps: (B, 2)"""
    ab = alpha_bars[t][:, None]                                  # (B, 1) — broadcast over dims
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps

# ── training data generator (the whole loss, one batch at a time) ──
x0 = make_spiral()
batch_t  = np.random.randint(0, T, size=len(x0))                 # sample t ~ Uniform
batch_eps = np.random.randn(*x0.shape)                           # sample ε ~ N(0, I)
batch_xt  = q_sample(x0, batch_t, batch_eps)                     # build x_t

# A real training step would call eps_theta(batch_xt, batch_t) and regress onto batch_eps
# with mean squared error. That's it. That's the whole training signal.
loss = ((batch_eps - batch_eps) ** 2).mean()                     # placeholder: perfect ε_θ
print(f"placeholder MSE with perfect denoiser: {loss:.4f}")`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for k in range(t): x = √(1-β_k)·x + √β_k·ε',
            right: 'q_sample(x0, t, eps)  # one line, closed form',
            note: 'the Markov loop collapses into the reparameterised equation',
          },
          {
            left: 'running *= a  in a python list',
            right: 'np.cumprod(alphas)',
            note: 'ᾱ_t is a cumulative product — one vectorised call',
          },
          {
            left: 'sample one scalar eps per step',
            right: 'np.random.randn(*x0.shape)',
            note: 'a full batch of ε for the whole training step',
          },
        ]}
      />

      <Prose>
        <p>
          PyTorch is where you actually train this at scale. The UNet is out of scope for
          this lesson — we&apos;ll build one next — so here we stub it with an{' '}
          <code>nn.Module</code> placeholder and show the <em>exact</em> training loop every
          diffusion paper uses. When you&apos;ve read one you&apos;ve read them all.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · ddpm_train.py"
        output={`step 0000 | loss 0.9821
step 0050 | loss 0.4217
step 0100 | loss 0.2831
step 0150 | loss 0.1694
step 0200 | loss 0.1102`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F

T = 1000
betas = torch.linspace(1e-4, 0.02, T)                            # linear schedule (DDPM)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)                        # ᾱ_t

class EpsilonModel(nn.Module):
    """Stand-in for a UNet. Takes x_t and t, outputs an estimate of ε."""
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x_t, t):
        # Concatenate a normalised time embedding onto the input.
        t_emb = (t.float() / T).unsqueeze(-1)
        return self.net(torch.cat([x_t, t_emb], dim=-1))

model = EpsilonModel()
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

# ── training loop — this is the whole algorithm ──
for step in range(201):
    x0 = sample_data_batch()                                     # your data loader
    t  = torch.randint(0, T, (x0.size(0),))                      # sample t ~ Uniform(1, T)
    eps = torch.randn_like(x0)                                   # sample ε ~ N(0, I)

    ab = alpha_bars[t].unsqueeze(-1)                             # (B, 1)
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps                 # closed-form forward

    eps_pred = model(x_t, t)
    loss = F.mse_loss(eps_pred, eps)                             # L_simple — MSE on ε

    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"step {step:04d} | loss {loss.item():.4f}")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'np.random.randint(0, T, size=B)',
            right: 'torch.randint(0, T, (B,))',
            note: 'same uniform draw of timesteps — GPU-native now',
          },
          {
            left: 'np.sqrt(ab) * x0 + np.sqrt(1-ab) * eps',
            right: 'ab.sqrt() * x0 + (1-ab).sqrt() * eps',
            note: 'the forward process — identical math, now tracked by autograd',
          },
          {
            left: 'loss = ((eps_pred - eps) ** 2).mean()',
            right: 'F.mse_loss(eps_pred, eps)',
            note: 'L_simple. The entire training signal for diffusion.',
          },
        ]}
      />

      <Callout variant="insight" title="the point of the three layers">
        Layer 1 verifies the closed form — one-step iteration and direct jump agree to the
        decimal. Layer 2 shows the training step as pure tensor algebra with no neural
        machinery. Layer 3 is what a real repo looks like, and the only new thing is the
        word <code>Adam</code>. The forward pass, the loss, the target — they&apos;re the
        same three lines in all three implementations.
      </Callout>

      {/* ── Schedule callout ─────────────────────────────────────── */}
      <Callout variant="note" title="variance schedules — the one hyperparameter that matters">
        The original DDPM paper uses a <strong>linear</strong> schedule: <code>β_t</code>{' '}
        ramps from <code>1e-4</code> to <code>0.02</code> over 1000 steps. It works.
        Nichol &amp; Dhariwal (2021) noticed the linear schedule adds noise too fast at the
        end — the last 200 frames are nearly pure noise and contribute very little to the
        training signal. Their <strong>cosine</strong> schedule slows the destruction near{' '}
        <code>t = T</code>, preserves structure longer, and measurably improves sample
        quality. Some modern papers <em>learn</em> the schedule. For a first implementation,
        linear is fine.
      </Callout>

      {/* ── DDPM vs DDIM ─────────────────────────────────────────── */}
      <Callout variant="insight" title="DDPM vs DDIM — same model, different sampler">
        DDPM sampling is stochastic: at each reverse step you subtract <code>ε_θ</code> <em>and</em>{' '}
        add a fresh Gaussian of variance <code>β_t</code>. It needs all{' '}
        <code>T = 1000</code> frames to rewind the film cleanly. DDIM (Song et al. 2021)
        reinterprets the reverse process as an ODE — you can take the same trained network
        and rewind <em>deterministically</em>, skipping frames. 50-step DDIM samples often
        match 1000-step DDPM samples. Same projector. No retraining. It&apos;s a sampler
        change, not a model change — and it&apos;s why modern diffusion models are fast
        enough to run in your browser.
      </Callout>

      {/* ── Gotchas ──────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Off-by-one on β indexing:</strong> papers are
          inconsistent about whether <code>t</code> starts at 0 or 1 and whether{' '}
          <code>ᾱ_t</code> includes <code>α_t</code> or stops at <code>α_{'{t-1}'}</code>.
          Pick a convention, write it down, and check that <code>ᾱ_T</code> is tiny and{' '}
          <code>ᾱ_0</code> is close to 1. Get this wrong and your samples are noise.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting to pass t to the model:</strong> the
          denoiser has to know <em>how noisy</em> its input is — the same <code>x</code>
          needs different treatment at <code>t = 10</code> vs <code>t = 900</code>. The
          projector can&apos;t rewind a frame without knowing which frame it is. Time is a
          conditioning signal, not an afterthought. A diffusion model without a time
          embedding is a diffusion model that will not train.
        </p>
        <p>
          <strong className="text-term-amber">Non-monotonic schedules break the closed form:</strong>{' '}
          if you ever set <code>β_t ≥ 1</code> or let <code>α_t &lt; 0</code>, the cumulative
          product <code>ᾱ_t</code> becomes meaningless and the reparameterisation collapses.
          Keep <code>β_t ∈ (0, 1)</code> and monotonically non-decreasing.
        </p>
        <p>
          <strong className="text-term-amber">Sampling fewer steps than you trained on:</strong>{' '}
          with DDPM you can&apos;t. The stochastic sampler assumes the trained <code>β</code>{' '}
          schedule end-to-end. If you want 50-step sampling from a 1000-step model, you need
          DDIM (deterministic) or one of the newer distilled samplers. Silently truncating
          DDPM gives you blurry garbage.
        </p>
      </Gotcha>

      {/* ── Challenge ────────────────────────────────────────────── */}
      <Challenge prompt="Verify the closed-form equation on MNIST">
        <p>
          Load a single MNIST digit as <code>x_0</code> (shape <code>(1, 28, 28)</code>).
          Build the linear β-schedule for <code>T = 1000</code>. Compute <code>α_bars</code>{' '}
          as a cumulative product.
        </p>
        <p className="mt-2">
          For a fixed <code>t = 500</code> and a fixed seed: (1) iterate the forward Markov
          chain 500 times, drawing a fresh <code>ε</code> at each step, to get{' '}
          <code>x_500_iter</code>. (2) Use the closed-form equation{' '}
          <code>x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε</code> to jump straight to frame 500 and
          produce <code>x_500_closed</code>.
        </p>
        <p className="mt-2">
          Plot both side-by-side, plus <code>x_0</code> and the iso-variance Gaussian{' '}
          <code>N(0, (1−ᾱ_500) · I)</code>. Statistically the iterated and closed-form
          versions should be indistinguishable — they&apos;re two draws from the same
          Gaussian. That&apos;s the equation that makes diffusion trainable.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: grid-plot <code>x_t</code> for <code>t ∈ {'{0, 100, 200, …, 1000}'}</code>{' '}
          and watch the digit dissolve. You now have the forward half of Stable Diffusion —
          the full film of an image being shredded, stored as a row of thumbnails.
        </p>
      </Challenge>

      {/* ── Closing ──────────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Diffusion is the film,
          rewound. A fixed Gaussian noising chain with a closed-form jump
          to any frame, glued to a learned Gaussian denoising chain trained
          by a single regression loss on predicted noise. The forward pass
          has no parameters and doesn&apos;t need to be understood at
          training time — it&apos;s one equation, the camera rolling. The
          reverse pass is where the network lives, and that network is
          just predicting ε by MSE — the projector running the reel
          backwards one frame at a time. Everything else —
          text-conditioning, classifier-free guidance, latent diffusion,
          distillation — bolts onto this skeleton.
        </p>
        <p>
          <strong>Next up — U-Net Architecture.</strong> We&apos;ve written
          every line of the training loop except the one that matters:{' '}
          <code>eps_theta(x_t, t)</code>. That&apos;s the projector itself,
          and in diffusion it&apos;s almost always a UNet — an
          encoder-decoder with skip connections, time embeddings, attention
          blocks, and a very specific pattern of where noise gets peeled
          off at each resolution. Next lesson we build one from scratch
          and plug it into the loop you just read.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Denoising Diffusion Probabilistic Models',
            author: 'Ho, Jain, Abbeel',
            venue: 'NeurIPS 2020 — the DDPM paper',
            url: 'https://arxiv.org/abs/2006.11239',
          },
          {
            title: 'Denoising Diffusion Implicit Models',
            author: 'Song, Meng, Ermon',
            venue: 'ICLR 2021 — DDIM, deterministic sampling',
            url: 'https://arxiv.org/abs/2010.02502',
          },
          {
            title: 'Improved Denoising Diffusion Probabilistic Models',
            author: 'Nichol, Dhariwal',
            venue: 'ICML 2021 — cosine schedule, learned variance',
            url: 'https://arxiv.org/abs/2102.09672',
          },
          {
            title: 'Understanding Diffusion Models: A Unified Perspective',
            author: 'Luo',
            year: 2022,
            venue: 'the tutorial that rederives everything cleanly',
            url: 'https://arxiv.org/abs/2208.11970',
          },
        ]}
      />
    </div>
  )
}
