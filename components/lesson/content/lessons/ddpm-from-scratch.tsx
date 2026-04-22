import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import Prereq from '../Prereq'
import NeedsBackground from '../NeedsBackground'
import {
  Prose, Callout, Personify, Bridge, Gotcha, Challenge, References, KeyTerm, AsciiBlock,
} from '../primitives'
import DDPMTraining from '../widgets/DDPMTraining'                // animated: sample t, add noise, predict noise, MSE loss, update. Epoch counter and loss curve.
import DDPMGeneration from '../widgets/DDPMGeneration'            // start from pure noise, step through reverse process; see MNIST-like digits emerging

// Signature anchor: the full cookbook, every ingredient in one kitchen. Every
// earlier diffusion lesson was a single chapter of the recipe — what noise is,
// what the reverse process looks like, what network to use. This lesson cooks
// the whole thing. Noise schedule is the marinade timing, the U-Net is the
// stove, the MSE loss is the thermometer, the sampling loop is the plating.
// Return at the opening (the pantry with every ingredient), the assembly-order
// reveal (the recipe runs in this exact sequence), and the closing dish
// consolidation. No tabs, no section headers — one long scroll.
export default function DDPMFromScratchLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="ddpm-from-scratch" />

      <Prose>
        <p>
          You have been collecting ingredients. Three lessons of pantry-stocking.{' '}
          <NeedsBackground slug="denoising-intuition">Denoising</NeedsBackground> gave you the
          raw idea — a staircase of Gaussian corruption, run in reverse. The{' '}
          <NeedsBackground slug="forward-and-reverse-diffusion">forward process</NeedsBackground>{' '}
          gave you the closed-form jump — the marinade, timed by a variance schedule, that
          takes a clean image and turns it into a specific flavor of noise. The{' '}
          <NeedsBackground slug="unet-architecture">UNet</NeedsBackground> gave you the stove
          — an hourglass of convolutions that eats a noisy image and a timestep and spits out
          the noise it thinks is hiding inside.
        </p>
        <p>
          This lesson is the whole kitchen, running at once. One recipe. Every ingredient
          lined up on the counter, every tool plugged in, the timer set. A{' '}
          <KeyTerm>DDPM</KeyTerm> — denoising diffusion probabilistic model — is what you
          cook when you stop treating those pieces as separate dishes and finally assemble
          them in the order the recipe demands. Training is a loop. Generation is a loop.
          Both loops are short, and the surprise — the reason every modern image model
          (Stable Diffusion, Imagen, DALL·E 3) is a diffusion model in disguise — is that
          these short loops are enough to turn a bin of pure noise into a recognizable
          handwritten digit. You will cook one here, end to end, and something will come out
          of the oven that you can actually taste.
        </p>
      </Prose>

      <Callout variant="insight" title="the pantry, every ingredient on the counter">
        <div className="space-y-2">
          <p>
            Five ingredients. Each one you have seen before, alone. This lesson is about
            what happens when you finally cook them together.
          </p>
          <p>
            <strong>Noise schedule</strong> — the marinade timing. A sequence of{' '}
            <code>β_t</code> values that says how aggressively to corrupt at each step.
          </p>
          <p>
            <strong>Forward jump</strong> — the one-line shortcut{' '}
            <code>x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε</code> that lets you marinate to any
            timestep in a single matmul instead of simulating the chain.
          </p>
          <p>
            <strong>UNet</strong> — the stove. Eats a noisy image and a timestep, outputs
            a noise estimate.
          </p>
          <p>
            <strong>MSE loss</strong> — the thermometer. One number that says how close the
            UNet&apos;s estimate is to the noise you actually added.
          </p>
          <p>
            <strong>Sampling loop</strong> — the plating. Start from pure noise, run the
            UNet in reverse T times, serve at <code>x_0</code>.
          </p>
        </div>
      </Callout>

      <AsciiBlock caption="the whole recipe, end to end — the order is the dish">
{`  ┌──────────────────────── training ────────────────────────┐
  │                                                          │
  │   x₀  ── sample batch ─┐                                  │
  │                        │                                  │
  │   t   ∼ U(1, T)  ──────┤                                  │
  │   ε   ∼ N(0, I)  ──────┤                                  │
  │                        ▼                                  │
  │        x_t = √ᾱ_t · x₀  +  √(1-ᾱ_t) · ε                    │
  │                        │                                  │
  │                        ▼                                  │
  │              U-Net(x_t, t)  →  ε̂                          │
  │                        │                                  │
  │                        ▼                                  │
  │          loss = ‖ε - ε̂‖²   →  backward → step             │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

  ┌─────────────────────── generation ───────────────────────┐
  │                                                          │
  │     x_T ∼ N(0, I)   ← pure noise, no image yet           │
  │                                                          │
  │     for t = T, T-1, … , 1:                               │
  │         ε̂ = U-Net(x_t, t)                                │
  │         x_{t-1} = mean(x_t, ε̂)  +  σ_t · z               │
  │                                                          │
  │     return x_0   ← a digit                              │
  │                                                          │
  └──────────────────────────────────────────────────────────┘`}
      </AsciiBlock>

      <Prose>
        <p>
          Two assembly tricks make this cookbook actually cook. First, the closed-form jump{' '}
          <code>x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε</code> means we never simulate a chain of{' '}
          <code>T</code> corruption steps during training — we jump to a random timestep in
          one matmul and train on that single pair. It is the recipe&apos;s &ldquo;marinate
          overnight in one shot&rdquo; cheat. Second, the regression target is{' '}
          <em>the noise we added</em>, not the clean image. The UNet learns a{' '}
          <KeyTerm>noise prediction</KeyTerm> function, and everything else — the sampler,
          the variance, the mean — falls out of rearranging that noise estimate through the
          forward-process algebra. One ingredient predicted; everything else derived.
        </p>
      </Prose>

      <MathBlock caption="training objective — the entire loss, one line">
{`β_1, β_2, … , β_T  ∈ (0, 1)                      # variance schedule (linear: 1e-4 → 0.02)
α_t     = 1 - β_t
ᾱ_t     = ∏_{s=1..t} α_s                         # cumulative product, precompute once

x_0     ∼  p_data                                # sample a clean image
t       ∼  U{1, …, T}                            # sample a timestep
ε       ∼  N(0, I)                               # sample target noise

x_t     =  √ᾱ_t · x_0  +  √(1 - ᾱ_t) · ε         # forward jump, no simulation
ε̂       =  u_net(x_t, t; θ)                      # U-Net prediction

L_simple(θ)  =  E_{x_0, t, ε} ‖ ε - ε̂ ‖²          # just MSE on the noise`}
      </MathBlock>

      <Prose>
        <p>
          That is it. One expectation, one MSE. Ho et al. 2020 showed this{' '}
          <em>&ldquo;simple&rdquo;</em> loss is a reweighted variational lower bound and
          that the reweighting is actually <em>better</em> for sample quality than the
          theoretically-correct weights. So the loss every diffusion paper since 2020 uses
          is three lines of PyTorch. The hard work — the ingredient prep — was the UNet and
          the schedule. You already did it. The thermometer is cheap.
        </p>
      </Prose>

      <DDPMTraining />

      <Prose>
        <p>
          Watch what the kitchen is actually doing. Each step: grab a digit from the pantry,
          pick a random <code>t</code>, add exactly the amount of noise the schedule says
          corresponds to that <code>t</code>, ask the UNet to recover the noise pattern,
          take the squared error, backprop. The network never sees a &ldquo;full chain&rdquo;
          of <code>T=1000</code> denoisings during training — it sees thousands of
          independent one-shot noise-prediction problems, and the chain only assembles itself
          at generation time. That decoupling is what makes DDPM training tractable, and it
          is the single most important thing to notice about the recipe: the UNet is not
          learning a sequence, it is learning a one-ingredient skill that the sampler reuses
          a thousand times in a row.
        </p>
        <p>
          The loss curve looks boring. It drops fast in the first few hundred steps (the
          network learns &ldquo;output something that averages to zero&rdquo;) and then
          flattens into a slow, noisy decline for the rest of the run. Do not be fooled:
          sample quality keeps improving long after the loss curve has visually plateaued,
          because what is improving is the network&apos;s accuracy on the <em>hard</em>{' '}
          timesteps — the mid-range <code>t</code> where the image is mostly-but-not-quite
          noise and the signal is subtle.
        </p>
      </Prose>

      <Personify speaker="Noise prediction">
        I am the regression target, and I am deliberately boring. I am not a clean digit.
        I am not even close to one. I am a fresh Gaussian sample the trainer rolled right
        before adding me to <code>x_0</code>. The UNet&apos;s job is to look at the
        contaminated image and guess me back — not the digit underneath, just me. Why? Once
        the UNet can do that, rearranging{' '}
        <code>x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε</code> gives you the digit for free. I am the
        supervision signal that makes everything else closed-form.
      </Personify>

      <MathBlock caption="sampling — the reverse process, step by step">
{`x_T  ∼  N(0, I)                                  # start from pure noise

for t = T, T-1, … , 1:
    ε̂        =  u_net(x_t, t; θ)                  # predict the noise that is "in" x_t

    μ_θ(x_t, t)  =  1/√α_t · ( x_t  -  β_t / √(1-ᾱ_t) · ε̂ )
    #                    ↑ subtract a scaled noise estimate; scale back up by 1/√α_t

    σ_t²         =  β_t                            # DDPM variance (β̃_t also common)

    z  ∼  N(0, I)  if t > 1  else  0
    x_{t-1}  =  μ_θ(x_t, t)  +  σ_t · z           # one denoising step

return x_0                                        # a sample from the model`}
      </MathBlock>

      <Prose>
        <p>
          Plating. The reverse walk is where the meal leaves the kitchen. At each step we
          ask the UNet &ldquo;how much noise is in this?&rdquo;, subtract a scaled estimate,
          rescale, and add a dollop of fresh noise for stochasticity. The mean formula looks
          mysterious but is just algebra: it is the posterior mean of <code>x_{'{t-1}'}</code>{' '}
          given <code>x_t</code> and our best guess of <code>x_0</code>, worked out under the
          Gaussian forward model. You memorize the formula; the paper derives it in two
          pages of completing-the-square. Every ingredient from the pantry earns its seat
          right here — the schedule sets <code>β_t</code> and <code>ᾱ_t</code>, the UNet
          delivers <code>ε̂</code>, the algebra folds them into one mean, and the loop
          repeats until the dish is done.
        </p>
      </Prose>

      <DDPMGeneration />

      <Prose>
        <p>
          The widget starts from a square of Gaussian static — as informative as TV snow —
          and runs <code>T=1000</code> reverse steps. For the first ~700 steps nothing you
          can see happens. Around step 300 a blurry low-frequency blob emerges. By step 100
          you can guess the digit. By step 0 the strokes have sharp edges and a plausible
          thickness. The reverse chain is a coarse-to-fine generator: big shapes resolve
          first, textures last. Diffusion&apos;s multi-scale behavior is why it blows
          single-step generators (GANs, VAEs) out of the water on sample diversity — each
          timestep is a fresh opportunity for the model to commit to a different global
          structure.
        </p>
      </Prose>

      <Personify speaker="Sample trajectory">
        I am the path a generated image takes through noise levels. I start at{' '}
        <code>x_T</code>, pure static, and end at <code>x_0</code>, a clean digit. You can
        plot me as 1000 thumbnails. Early thumbnails are indistinguishable from each other
        — all noise. The middle thumbnails are where the image &ldquo;decides&rdquo; what
        digit to be; this is where different random seeds diverge. Late thumbnails just
        sharpen what the middle already committed to. If you want a sample to follow a
        specific trajectory, you intervene in the middle. That is the doorway to
        classifier-free guidance, img2img, inpainting, and every other controllable
        diffusion trick.
      </Personify>

      <Prose>
        <p>
          Three layers of code, one recipe at three scales. We start with a 2-D toy —
          diffusion on a Swiss roll, where the &ldquo;stove&rdquo; is a tiny MLP and you can
          plot the whole thing in matplotlib — then graduate to a real PyTorch
          implementation that trains on MNIST with a small UNet, and finally we show how the
          same dish takes about fifteen lines in Hugging Face&apos;s <code>diffusers</code>{' '}
          library with a pretrained backbone. Same cookbook, three kitchens.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · ddpm_swissroll.py (2-D toy, MLP noise predictor)"
        output={`step    0  loss=1.0312
step  500  loss=0.2178
step 1000  loss=0.1034
step 1500  loss=0.0721
step 2000  loss=0.0613
generated 2000 samples; visually matches the swiss-roll density.`}
      >{`import numpy as np
from sklearn.datasets import make_swiss_roll

rng = np.random.default_rng(0)

# ---- data: 2-D swiss roll, standardized ----
X, _ = make_swiss_roll(n_samples=10_000, noise=0.5)
X = X[:, [0, 2]] / 10.0                            # drop the y-axis; 2-D points

# ---- schedule ----
T = 200
betas = np.linspace(1e-4, 0.02, T)
alphas = 1 - betas
abar = np.cumprod(alphas)                          # ᾱ_t

# ---- tiny MLP noise predictor:  (x_t, t) -> ε̂ ----
def init(n_in, n_out):
    return rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
W1, W2, W3 = init(2 + 16, 128), init(128, 128), init(128, 2)
b1, b2, b3 = np.zeros(128), np.zeros(128), np.zeros(2)

def time_embed(t):                                 # sinusoidal, 16-D
    freqs = 10 ** np.linspace(0, 2, 8)
    ang = t[:, None] / freqs
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)

def forward(x, t):
    h = np.concatenate([x, time_embed(t)], axis=-1)
    h = np.maximum(0, h @ W1 + b1)
    h = np.maximum(0, h @ W2 + b2)
    return h @ W3 + b3

# ---- training loop (sketch; full backprop omitted for brevity) ----
LR, BATCH = 1e-3, 256
for step in range(2001):
    idx = rng.integers(0, len(X), BATCH)
    x0 = X[idx]
    t  = rng.integers(1, T, BATCH)
    eps = rng.standard_normal(x0.shape)
    xt  = np.sqrt(abar[t])[:, None] * x0 + np.sqrt(1 - abar[t])[:, None] * eps
    eps_hat = forward(xt, t.astype(float))
    loss = ((eps - eps_hat) ** 2).mean()
    # ... manual backprop + SGD on W1..W3, b1..b3 ...
    if step % 500 == 0:
        print(f"step {step:4d}  loss={loss:.4f}")

# ---- generation: reverse walk ----
x = rng.standard_normal((2000, 2))
for t in range(T - 1, -1, -1):
    eps_hat = forward(x, np.full(len(x), t, dtype=float))
    mean = (x - betas[t] / np.sqrt(1 - abar[t]) * eps_hat) / np.sqrt(alphas[t])
    x = mean + (np.sqrt(betas[t]) * rng.standard_normal(x.shape) if t > 0 else 0)
# 'x' now resembles the swiss-roll distribution`}</CodeBlock>

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch · ddpm_mnist.py (tiny U-Net, full DDPM on MNIST)"
        output={`epoch  1  step  937  loss=0.1481
epoch  5  step 4685  loss=0.0654
epoch 10  step 9370  loss=0.0498
epoch 20  step 18740 loss=0.0421
epoch 40  step 37480 loss=0.0388
saved samples/ema_epoch_40.png  (8x8 grid of generated digits)`}
      >{`import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from copy import deepcopy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
T = 1000
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1 - betas
abar   = torch.cumprod(alphas, dim=0)                 # ᾱ_t

def q_sample(x0, t, eps):
    """forward: jump straight to x_t."""
    a = abar[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * eps

class TinyUNet(nn.Module):                            # the one you built last lesson
    # ... conv-down-conv-up with skip connections and sinusoidal time embedding ...
    def forward(self, x, t): ...

model     = TinyUNet().to(device)
ema_model = deepcopy(model).eval()                    # running average of weights
for p in ema_model.parameters(): p.requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform),
                    batch_size=64, shuffle=True)

def ema_update(ema, online, decay=0.999):
    for pe, po in zip(ema.parameters(), online.parameters()):
        pe.data.mul_(decay).add_(po.data, alpha=1 - decay)

for epoch in range(1, 41):
    for xb, _ in loader:
        xb = xb.to(device)                             # [B, 1, 28, 28], in [-1, 1]
        t  = torch.randint(0, T, (xb.size(0),), device=device)
        eps = torch.randn_like(xb)
        x_t = q_sample(xb, t, eps)
        eps_hat = model(x_t, t)
        loss = F.mse_loss(eps_hat, eps)
        opt.zero_grad(); loss.backward(); opt.step()
        ema_update(ema_model, model)

# ---- generation (use EMA weights) ----
@torch.no_grad()
def sample(n=64):
    x = torch.randn(n, 1, 28, 28, device=device)
    for t in reversed(range(T)):
        tb = torch.full((n,), t, device=device, dtype=torch.long)
        eps_hat = ema_model(x, tb)
        a_t, b_t, ab_t = alphas[t], betas[t], abar[t]
        mean = (x - b_t / (1 - ab_t).sqrt() * eps_hat) / a_t.sqrt()
        x = mean + (b_t.sqrt() * torch.randn_like(x) if t > 0 else 0)
    return x.clamp(-1, 1)`}</CodeBlock>

      <Bridge
        label="numpy toy → pytorch mnist"
        rows={[
          { left: 'MLP(2+16 → 128 → 128 → 2)', right: 'U-Net with conv down/up + skip conns', note: 'spatial structure demands convolutions; MLPs do not scale to images' },
          { left: 'manual sinusoidal time embedding', right: 'same embedding, fed into every ResBlock', note: 'the U-Net uses t at every scale, not just the input' },
          { left: 'no EMA, greedy sampling', right: 'EMA of weights + EMA-only sampling', note: 'single biggest sample-quality improvement per line of code' },
          { left: 'abar precomputed in numpy', right: 'abar as a registered GPU tensor', note: 'cumulative product is index-once, use-everywhere' },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 3 — diffusers · ddpm_pretrained.py"
        output={`Downloaded google/ddpm-celebahq-256  (holy smokes, 339 MB)
Running 1000 reverse steps on cuda:0 ...
done.  Saved 4 samples to ./out/  (256x256, unmistakably faces)`}
      >{`from diffusers import DDPMPipeline
import torch

# A pretrained DDPM that someone else spent a week training on A100s.
pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
pipe = pipe.to("cuda")

# One line replaces the entire reverse-process loop above.
images = pipe(num_inference_steps=1000, batch_size=4).images
for i, img in enumerate(images):
    img.save(f"out/sample_{i}.png")`}</CodeBlock>

      <Bridge
        label="pytorch scratch → diffusers"
        rows={[
          { left: 'hand-coded schedule, q_sample, sampler', right: 'DDPMScheduler handles both', note: 'one object, unified API for DDPM / DDIM / PNDM' },
          { left: 'TinyUNet class with conditioning', right: 'UNet2DModel / UNet2DConditionModel', note: 'production-grade blocks, attention, class conditioning, FP16' },
          { left: 'manual EMA update loop', right: 'EMAModel wrapper, one call per step', note: 'also does FP32 shadow, decay warmup' },
          { left: 'hand-rolled training loop', right: 'accelerate + diffusers/examples/unconditional_image_generation', note: 'multi-GPU, mixed precision, checkpointing, wandb — for free' },
        ]}
      />

      <Callout variant="insight" title="EMA is not optional">
        The exponential moving average of the model weights — usually decay ≈ 0.9999 — is
        one of those tricks that looks like a footnote in the paper and turns out to be the
        difference between a model that samples and a model that doesn&apos;t.{' '}
        SGD-and-friends leave the weights <em>oscillating</em> around a good point. At any
        single step, the model is slightly off the center of the good region. EMA averages
        those oscillations away. Generation with online weights: blurry, artifact-ridden.
        Generation with EMA weights: sharp. Cost: one extra buffer of parameters and a{' '}
        <code>mul_add_</code> per step. You do this for every diffusion model from now on.
      </Callout>

      <Callout variant="note" title="evaluating a generator with FID">
        Diffusion models do not have a single validation loss that corresponds to sample
        quality, because the MSE loss is not what we care about — we care about{' '}
        <em>how the samples look</em>. The standard metric is{' '}
        <KeyTerm>Fréchet Inception Distance</KeyTerm>. Run a pretrained Inception-v3
        classifier on your generated samples and on a held-out set of real images, grab the
        2048-D pool features, fit a Gaussian to each, and compute the Fréchet distance
        (squared-mean-difference plus trace-of-covariance-difference) between them. Lower
        is better. DDPM on CIFAR-10 scores ≈ 3.2; a well-tuned modern diffusion model
        scores ≈ 1.8; the real-vs-real noise floor is around 1.0. FID is not perfect (it
        rewards matching Inception&apos;s biases, which is a weird gold standard for
        anything that is not natural images) but it is the number everyone reports.
      </Callout>

      <Callout variant="note" title="compute cost, so you know what you are signing up for">
        A small DDPM on MNIST with the UNet you built: ~10 minutes on one modern GPU to
        reach convincing samples. A DDPM on CIFAR-10 at 32×32: a few hours on a single
        A100. A DDPM on 256×256 faces (CelebA-HQ): a day or two on 4×A100. Stable
        Diffusion at 512×512 on LAION-2B: ~150 000 A100-hours, roughly $600 000. The cost
        scales with image resolution (quadratic in pixel count for the UNet) <em>and</em>{' '}
        dataset size. The good news: the inference cost of a trained DDPM is just{' '}
        <code>T</code> UNet forward passes, which is parallelizable across a batch.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">
            Predicting x₀ instead of ε when the schedule expects ε.
          </strong>{' '}
          Both parameterizations are valid (x₀-prediction, ε-prediction, and Karras&apos;
          v-prediction all work) but the sampler has to match the training target. A
          UNet trained with ε-loss plugged into an x₀-sampler produces elegant noise. The
          number you subtract in the mean formula is scaled differently in each case;
          eyeball the{' '}
          <code>√(1-ᾱ_t)</code> term — that is the ε-prediction coefficient.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting EMA at sample time.</strong>{' '}
          Training with EMA is pointless if you sample from the online model. Keep two
          state dicts around, and only save the EMA one for inference.
        </p>
        <p>
          <strong className="text-term-amber">
            Using a different T at inference than at training.
          </strong>{' '}
          The ᾱ schedule is indexed by integer <code>t ∈ [0, T)</code>. If you trained with{' '}
          <code>T=1000</code> and try to sample with <code>T=100</code> by just dropping
          indices, you get nonsense — the noise levels don&apos;t line up. To sample faster
          you need a principled subsampler like <KeyTerm>DDIM</KeyTerm>, which picks a
          subset of timesteps and reweights the variance so the reverse process still
          ends at <code>x_0</code>.
        </p>
        <p>
          <strong className="text-term-amber">
            Bad initialization on the convolutional weights.
          </strong>{' '}
          The output layer of the UNet predicts noise with mean ~0 and std ~1. If the
          last convolution is initialized with a big gain, early predictions overshoot,
          MSE blows up to 100+, and the optimizer wastes its first few thousand steps just
          pulling everything back to zero. Initialize the last conv with{' '}
          <code>zero_init</code> (weight = 0, bias = 0) — a standard trick from{' '}
          Karras&apos; EDM paper — or use Kaiming init with gain 0.1.
        </p>
        <p>
          <strong className="text-term-amber">Normalizing inputs to [0, 1].</strong>{' '}
          Diffusion expects data centered at 0, so normalize MNIST pixels to{' '}
          <code>[-1, 1]</code> (i.e. <code>(x - 0.5) / 0.5</code>). If you leave the data
          in <code>[0, 1]</code>, the forward process drifts off-center and the noise
          target no longer matches what the UNet expects.
        </p>
      </Gotcha>

      <Challenge prompt="Train a DDPM on Fashion-MNIST, compute FID, vary T">
        <p>
          Swap <code>datasets.MNIST</code> for <code>datasets.FashionMNIST</code> in the
          layer-2 code (same 28×28 grayscale, different label set: shirts, bags, sneakers)
          and retrain the same UNet for 40 epochs. Hold out 10% of the training set to
          compute <KeyTerm>FID</KeyTerm> — grab a pretrained Inception-v3, extract
          2048-D pool features for 5000 generated samples and 5000 held-out real samples,
          fit Gaussians, compute the Fréchet distance.
        </p>
        <p>
          Then, with the exact same trained weights, sample once with{' '}
          <code>T=1000</code> and once with a DDIM sampler using <code>T=100</code>. Record
          FID for each. Questions: how much sample quality did you lose by sampling 10×
          faster? Was it worse for clothes than it would be for digits — why? (Hint:
          think about where fine-scale detail lives in a shirt vs. a &ldquo;3&rdquo;.)
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> A diffusion model is a complete dish —
          noise schedule, UNet, MSE loss, sampling loop — cooked in the one order the
          cookbook allows. Training is three lines. Sampling is five. Every ingredient from
          the earlier lessons finally earned its seat on the counter: the schedule sets the
          marinade, the UNet is the stove, the MSE thermometer calls the finish, the
          sampler handles the plating. Everything else you will meet — guidance, inpainting,
          latent diffusion, video diffusion, flow matching — is a variation on this
          assembly, with a different ingredient swapped in. The kitchen you just used is the
          same one, at the core, that cooks Midjourney and Stable Diffusion outputs. They
          just have bigger UNets, more data, and a text encoder bolted on.
        </p>
        <p>
          <strong>Next up — Classifier-Free Guidance.</strong> Right now your DDPM cooks{' '}
          <em>any</em> digit — a 3, a 7, whatever the random noise rolls. The next lesson
          is how to tell the recipe &ldquo;give me a 7&rdquo; without training a separate
          classifier to taste-test every dish. The trick is to train the UNet jointly on
          conditioned and unconditioned examples, then at sample time extrapolate away from
          the unconditional prediction toward the conditional one. It is the single most
          important post-DDPM paper, and it is the ingredient that turns this cookbook into
          the one behind every text-to-image model you have heard of.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Denoising Diffusion Probabilistic Models',
            author: 'Ho, Jain, Abbeel',
            venue: 'NeurIPS 2020 — the DDPM paper; derives the simple ε-loss and the reverse-process mean',
            url: 'https://arxiv.org/abs/2006.11239',
          },
          {
            title: 'Improved Denoising Diffusion Probabilistic Models',
            author: 'Nichol, Dhariwal',
            venue: 'ICML 2021 — cosine schedule, learned variance, better FID per step',
            url: 'https://arxiv.org/abs/2102.09672',
          },
          {
            title: 'Elucidating the Design Space of Diffusion-Based Generative Models',
            author: 'Karras, Aittala, Aila, Laine',
            venue: 'NeurIPS 2022 — the EDM paper; unifies DDPM/DDIM/score-SDE under one parameterization and fixes a pile of tiny mistakes everyone was making',
            url: 'https://arxiv.org/abs/2206.00364',
          },
          {
            title: 'Hugging Face diffusers library',
            author: 'Hugging Face',
            venue: 'github.com/huggingface/diffusers — schedulers, pipelines, training scripts for every diffusion variant',
            url: 'https://github.com/huggingface/diffusers',
          },
          {
            title: 'The Annotated Diffusion Model',
            author: 'Niels Rogge, Kashif Rasul',
            venue: 'Hugging Face blog — DDPM paper annotated line-by-line with runnable PyTorch',
            url: 'https://huggingface.co/blog/annotated-diffusion',
          },
        ]}
      />
    </div>
  )
}
