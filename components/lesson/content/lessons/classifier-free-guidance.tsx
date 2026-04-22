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
import CFGScaleEffect from '../widgets/CFGScaleEffect'
import ConditionalVsUnconditional from '../widgets/ConditionalVsUnconditional'

// Signature anchor: two radio signals, one dial. One signal is the model
// listening to the caption (conditional); the other is the model tuning into
// its own hum (unconditional). CFG is the dial that blends them — and when
// cranked past 1, it extrapolates away from the unconditional toward the
// conditional. Returns at the opening (the dial), the training reveal (the
// two-models-in-one trick), and the high-w photorealism/diversity section.
export default function ClassifierFreeGuidanceLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout ────────────────────────────────── */}
      <Prereq currentSlug="classifier-free-guidance" />

      {/* ── Opening: the two signals and the dial ───────────────── */}
      <Prose>
        <p>
          Picture an old radio with one dial and two antennae. The first
          antenna picks up a signal that says &ldquo;a corgi wearing
          sunglasses&rdquo; — loud, specific, demanding. The second antenna
          picks up the station the model would play if you hadn&apos;t asked
          for anything at all — generic ambient hum, the sound of{' '}
          <em>some image, any image</em>. Between those two signals is a
          single dial. Turn it one way and the radio ignores your request and
          plays the hum. Turn it the other way and the radio shouts your
          prompt at you in over-saturated technicolor. Finding the sweet spot
          in the middle is what modern image generation is, mostly.
        </p>
        <p>
          That dial has a name: the <KeyTerm>guidance scale</KeyTerm>. The
          trick that makes it work — a single network trained to listen on
          both antennae — has a name too:{' '}
          <KeyTerm>Classifier-Free Guidance</KeyTerm> (Ho &amp; Salimans,
          2022). Every text-to-image model you&apos;ve ever used — Stable
          Diffusion, DALL-E, Imagen, Midjourney, Flux — is one big CFG machine
          under the hood. When a user slides the &ldquo;prompt adherence&rdquo;
          knob in a UI, they are turning one number in the equation
          we&apos;re about to derive.
        </p>
      </Prose>

      <Callout variant="insight" title="three words before we start">
        <div className="space-y-2">
          <p>Three plain-English words you&apos;ll see on repeat.</p>
          <p>
            <strong>Conditional</strong> — the model with the prompt in its
            ear. Given the caption, what noise does it think is mixed into
            this image? Structured, peaked, opinionated.
          </p>
          <p>
            <strong>Unconditional</strong> — the same model with the prompt
            muted. No caption, no target, just the ambient hum of &ldquo;some
            image lives here&rdquo;. Bland, diffuse, low-information.
          </p>
          <p>
            <strong>Guidance</strong> — the dial that blends them. Low and the
            prompt is a whisper. High and the prompt is the only voice in the
            room. Cranked past 1 and the model is extrapolating past the
            conditional signal in the direction it was already going.
          </p>
        </div>
      </Callout>

      <Prose>
        <p>
          A bit of context before we start tuning the dial. A{' '}
          <NeedsBackground slug="forward-and-reverse-diffusion">
            diffusion
          </NeedsBackground>{' '}
          model trained on millions of images will happily generate plausible
          pictures forever — unconditionally, meaning without you asking for
          anything in particular. The hard problem is the next one: you type{' '}
          <em>&ldquo;a corgi wearing sunglasses&rdquo;</em> and you want a
          corgi wearing sunglasses, not a nice landscape. Getting a{' '}
          <NeedsBackground slug="denoising-intuition">denoising</NeedsBackground>{' '}
          model to actually <em>obey</em> a caption turns out to be a
          surprisingly deep research question.
        </p>
        <p>
          The first real answer, from 2021, was{' '}
          <KeyTerm>classifier guidance</KeyTerm>: train a separate image
          classifier on noisy images, then use its gradient at each sampling
          step to nudge the sample toward the target class. It works. But now
          you need two models, and the classifier has to be trained on noisy
          data, and nobody wants two models.
        </p>
        <p>
          CFG&apos;s answer is cleaner: don&apos;t bolt on a second model —
          teach the <em>same</em> model to speak on both channels. Randomly
          drop the caption during training so the network learns the prompted
          signal and the ambient hum in the same set of weights. At generation
          time, run the net twice per step, read both signals, and extrapolate
          along the direction between them. One dial, one knob, one
          hyperparameter. That&apos;s the whole algorithm.
        </p>
      </Prose>

      <Personify speaker="Diffusion model">
        On my own I&apos;ll draw you <em>something</em>. Attach a prompt and
        I&apos;ll try, but my obedience is negotiable. Crank the guidance dial
        up and I&apos;ll drop the hedge — I&apos;ll give you exactly what you
        asked for, for better or worse.
      </Personify>

      {/* ── The CFG update: the dial ─────────────────────────────── */}
      <Prose>
        <p>
          Here&apos;s the move, concretely. At every sampling step, run the
          U-Net twice: once with the prompt (conditional channel), once
          without (unconditional — you pass a null token, typically an empty
          string or a special learned embedding). You now have two noise
          predictions, two radio signals:
        </p>
        <ul>
          <li>
            <code>ε_cond</code> — what the model thinks the noise is,{' '}
            <em>given</em> the prompt. The conditional signal.
          </li>
          <li>
            <code>ε_uncond</code> — what the model thinks the noise is with no
            prompt at all. The unconditional hum.
          </li>
        </ul>
        <p>
          The difference <code>ε_cond − ε_uncond</code> is a direction in
          noise-prediction space. It points the way from &ldquo;generic
          image&rdquo; toward &ldquo;this prompt&rdquo;. The dial takes that
          direction and <em>amplifies</em> it:
        </p>
      </Prose>

      <MathBlock caption="the CFG update rule">
{`ε̂_guided  =  ε_uncond  +  s · (ε_cond − ε_uncond)

              ←──── baseline ────→      ←── amplified direction ──→

where  s  =  guidance scale  (a.k.a. CFG scale, w, or guidance weight)

s = 0       →   pure unconditional — ignores the prompt
s = 1       →   normal conditional generation
s = 7.5     →   Stable Diffusion default — strong adherence
s = 15–20   →   aggressive — saturated colors, prompt dominates`}
      </MathBlock>

      <Prose>
        <p>
          Rewrite the same equation as{' '}
          <code>ε̂ = (1 − s) · ε_uncond + s · ε_cond</code> and the structure
          becomes obvious: it&apos;s a <em>linear extrapolation</em>. When{' '}
          <code>s &gt; 1</code> you&apos;re not interpolating between the two
          signals, you&apos;re shooting <em>past</em> the conditional one in
          the direction it was already heading. That&apos;s the whole trick —
          crank the dial and the radio doesn&apos;t just play the prompted
          station louder, it extrapolates into a realm where the prompted
          signal is even more itself than it was to begin with.
        </p>
      </Prose>

      {/* ── Widget 1: the dial ──────────────────────────────────── */}
      <CFGScaleEffect />

      <Prose>
        <p>
          Slide <code>s</code> from 0 to 20 and watch the dial work. At{' '}
          <code>s = 0</code> the model generates whatever it wants — the
          prompt is invisible, pure unconditional hum. At <code>s = 1</code>{' '}
          you get honest conditional generation; the prompt registers, but
          weakly. Crank past <code>s = 5</code> and the prompt starts
          dominating — colors intensify, the subject centers itself, stylistic
          cues get loud. Around <code>s = 15–20</code> you tip into the
          over-saturated regime: contrast blows out, fine detail collapses,
          and the image starts to look like a caricature of the prompt rather
          than a picture that matches it.
        </p>
        <p>
          There is no &ldquo;correct&rdquo; position on the dial. It&apos;s a
          knob you tune per-prompt, per model, per use-case.{' '}
          <code>s = 7.5</code> is the Stable Diffusion default not because
          it&apos;s theoretically optimal but because it looked good on a lot
          of prompts during development.
        </p>
      </Prose>

      <Callout variant="note" title="what s = 0 vs s = 1 actually does">
        <p>
          <code>s = 0</code> <em>throws away</em> the conditional signal — the
          guided noise is just the unconditional one. The prompt has zero
          effect. <code>s = 1</code> simplifies to{' '}
          <code>ε̂ = ε_cond</code>: the unconditional term cancels out, and
          you&apos;re doing ordinary conditional sampling, no guidance at all.
          Every value in-between or beyond is a convex or extrapolated
          combination of the two.
        </p>
      </Callout>

      <Personify speaker="Null condition">
        I&apos;m the empty prompt — the ambient hum the model plays when no
        one&apos;s asked for anything. Subtract me from the conditional signal
        and whatever&apos;s left is the pure flavor of the prompt. Without me,
        the dial has nothing to amplify away from.
      </Personify>

      {/* ── Two models in one: the training reveal ──────────────── */}
      <Prose>
        <p>
          We skipped the most important part on purpose. How does one network
          produce both signals? It doesn&apos;t own two brains. There&apos;s
          no second unconditional model sitting in a locker. The whole thing
          rests on a single training trick, and it&apos;s the bit the
          &ldquo;classifier-free&rdquo; in the name is really about.
        </p>
        <p>
          During training, with some small probability — usually 10% — you
          <em> replace </em>the real caption with a null embedding before
          feeding the batch to the network. Nine times out of ten the model
          sees &ldquo;a red car on a coastal highway&rdquo; and learns the
          conditional signal for that caption. The tenth time it sees a
          special empty token and learns the ambient hum — what noise, on
          average, looks like without any caption at all. Same weights. Same
          loss. One network quietly moonlighting as two.
        </p>
        <p>
          That&apos;s why it&apos;s called classifier-<em>free</em>. No
          auxiliary classifier, no second model, no extra training pipeline.
          Just one line of code in the training loop that says{' '}
          <em>sometimes the caption is missing</em>, and suddenly you have two
          radio stations broadcasting from one transmitter.
        </p>
      </Prose>

      {/* ── The tradeoff: high w kills diversity ─────────────────── */}
      <Prose>
        <p>
          The dial comes with a price tag. What you buy by cranking the
          guidance scale is <em>prompt adherence</em> — the image looks more
          like what you asked for. What you pay is <em>diversity</em> and{' '}
          <em>naturalness</em>. The intuition: amplifying the same direction
          step after step pushes the sample outside the data distribution the
          model was trained on. Colors saturate because the model is
          exaggerating the &ldquo;this is a red apple&rdquo; signal past what
          real red apples look like.
        </p>
        <p>
          High <code>w</code> buys you photorealism that <em>feels</em>{' '}
          photographic — contrast, specular highlights, unmistakable
          subject-centering — and costs you every sample that doesn&apos;t fit
          that one aesthetic. Run the same prompt a hundred times at{' '}
          <code>s = 2</code> and you get a hundred different corgis in a
          hundred different moods. Run it at <code>s = 15</code> and you get
          the same corgi, same pose, same lens, a hundred times. Crank the
          dial, kill the diversity.
        </p>
      </Prose>

      <MathBlock caption="what the scale is trading off">
{`                 ┌───────────────────────────────────────────────┐
                 │       adherence          ↔        quality      │
                 └───────────────────────────────────────────────┘

low  s  (0–3)   :  faithful to the data distribution
                   diverse samples, but may miss the prompt
                   e.g. "red car" → sometimes returns a blue one

mid  s  (5–9)   :  sweet spot for most text-to-image
                   prompt lands, artifacts stay rare

high s  (12–20) :  prompt dominates, samples look stylized
                   colors saturate, skin smooths, detail thins
                   diversity collapses — same prompt, same image

very high s (30+):  model leaves the manifold of plausible images
                   artifacts, neon blobs, structural nonsense`}
      </MathBlock>

      {/* ── Widget 2: cond vs uncond vs guided ──────────────────── */}
      <ConditionalVsUnconditional />

      <Prose>
        <p>
          Prompt: <em>&ldquo;a cat&rdquo;</em>. The left panel shows{' '}
          <code>ε_uncond</code> — what the model predicts without the prompt,
          the ambient unconditional hum. The middle shows <code>ε_cond</code>{' '}
          — the cat-flavored conditional signal. The right panel shows the
          guided combination at the current dial setting. Turn up the scale
          and watch the guided prediction drift <em>away</em> from the
          unconditional baseline and <em>past</em> the conditional one. At{' '}
          <code>s = 1</code> the guided prediction equals <code>ε_cond</code>{' '}
          exactly. At <code>s = 10</code> it&apos;s somewhere well beyond.
        </p>
        <p>
          This is what&apos;s actually happening at every single sampling
          step — all 50, or 20, or whatever you&apos;re using — throughout the
          denoising process. The U-Net runs twice per step, and you spend
          roughly 2× the compute for classifier-free guidance compared to
          plain conditional sampling. That&apos;s not a rounding error;
          it&apos;s why batched inference for diffusion models always pairs
          conditional and unconditional forward passes.
        </p>
      </Prose>

      <Personify speaker="Guidance scale">
        I&apos;m the adherence dial. Turn me low and the model is tasteful but
        mumbly. Turn me high and the model shouts your prompt in saturated
        technicolor. Somewhere between five and ten I usually sit. Nobody
        knows the right value for me — you&apos;ll find it by trial and
        error.
      </Personify>

      {/* ── Why does this even work? ─────────────────────────────── */}
      <Prose>
        <p>
          The original paper frames CFG as implicit Bayes. If you train a
          single network to predict <code>ε</code> under both{' '}
          <code>p(x|c)</code> (conditional) and <code>p(x)</code>{' '}
          (unconditional), then sampling with guidance is approximately
          sampling from a sharpened distribution{' '}
          <code>
            p(x|c)<sup>s</sup> · p(x)<sup>1−s</sup>
          </code>
          . When <code>s &gt; 1</code> this distribution is peakier than the
          original <code>p(x|c)</code> — high-likelihood modes get amplified,
          low-likelihood ones get crushed. That&apos;s the math behind the
          intuition: cranking the dial concentrates mass on samples that are
          especially prompt-like.
        </p>
        <p>
          You don&apos;t need to carry that derivation in your head, but
          it&apos;s useful to know:{' '}
          <strong>guidance trades diversity for peakiness</strong>. It is,
          literally and mathematically, a temperature-like sharpening of the
          conditional distribution.
        </p>
      </Prose>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, as always. First the update rule on toy signals — just
          to see the linear extrapolation doing its thing. Then the training
          trick (10% unconditional caption dropout) inside a PyTorch loop.
          Then the one-liner in <code>diffusers</code> that every production
          pipeline ultimately boils down to.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · cfg_update.py"
        output={`uncond = [0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10]
cond   = [0.02 0.04 0.08 0.14 0.20 0.20 0.14 0.08 0.06 0.04]
s=0.0  → [0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10]
s=1.0  → [0.02 0.04 0.08 0.14 0.20 0.20 0.14 0.08 0.06 0.04]
s=7.5  → [-0.50 -0.35 -0.05 0.40 0.85 0.85 0.40 -0.05 -0.20 -0.35]`}
      >{`import numpy as np

# Pretend these are noise predictions from a denoiser at some step.
# Uncond = "nothing asked for" → flat-ish, low-information — the ambient hum.
# Cond   = "given the prompt"  → structured, peaked where the prompt wants signal.
uncond = np.full(10, 0.10)
cond   = np.array([0.02, 0.04, 0.08, 0.14, 0.20, 0.20, 0.14, 0.08, 0.06, 0.04])

def cfg(uncond, cond, s):
    return uncond + s * (cond - uncond)        # the entire algorithm, one line

for s in [0.0, 1.0, 7.5]:
    guided = cfg(uncond, cond, s)
    print(f"s={s:<4} →", np.round(guided, 2))`}</CodeBlock>

      <Prose>
        <p>
          That&apos;s the sampling side of the dial. The training side is
          where CFG earns its name — you never train a separate classifier,
          you just randomly drop the caption some of the time so the same
          network learns both the conditional signal and the unconditional
          hum.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — pytorch · cfg_training.py">{`import torch
import torch.nn as nn

UNCOND_DROP_PROB = 0.10        # 10% of batches trained with the null condition

def train_step(unet, x0, cond_emb, null_emb, scheduler, optimizer):
    # 1. Sample timestep and noise — ordinary DDPM.
    t     = torch.randint(0, scheduler.num_train_timesteps, (x0.size(0),), device=x0.device)
    noise = torch.randn_like(x0)
    xt    = scheduler.add_noise(x0, noise, t)

    # 2. CFG's one and only training trick:
    #    with prob p, swap the real conditioning for the null embedding.
    mask  = (torch.rand(x0.size(0), device=x0.device) < UNCOND_DROP_PROB).view(-1, 1, 1, 1)
    c     = torch.where(mask.expand_as(cond_emb), null_emb, cond_emb)

    # 3. Predict noise and regress against the true noise — same MSE as always.
    pred  = unet(xt, t, c)
    loss  = nn.functional.mse_loss(pred, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def sample_with_cfg(unet, scheduler, cond_emb, null_emb, guidance_scale=7.5,
                    shape=(1, 3, 64, 64), num_steps=50):
    x = torch.randn(shape, device=cond_emb.device)
    scheduler.set_timesteps(num_steps)

    for t in scheduler.timesteps:
        # One batched forward pass over [uncond, cond] — twice the memory, one call.
        x_in   = torch.cat([x, x], dim=0)
        c_in   = torch.cat([null_emb, cond_emb], dim=0)
        eps    = unet(x_in, t, c_in)
        eps_u, eps_c = eps.chunk(2)

        # The CFG update rule — identical to layer 1. Crank s to crank adherence.
        eps_hat = eps_u + guidance_scale * (eps_c - eps_u)

        x = scheduler.step(eps_hat, t, x).prev_sample
    return x`}</CodeBlock>

      <Bridge
        label="layer 1 → layer 2"
        rows={[
          {
            left: 'uncond = np.full(10, 0.10)',
            right: 'c = torch.where(mask, null_emb, cond_emb)',
            note: "the unconditional branch isn't a separate model — it's the same model fed the null embedding",
          },
          {
            left: 'guided = uncond + s*(cond - uncond)',
            right: 'eps_hat = eps_u + s*(eps_c - eps_u)',
            note: 'identical formula; the only difference is eps lives on a GPU now',
          },
          {
            left: '(no training step — prediction only)',
            right: 'mask = rand() < 0.10 → swap to null_emb',
            note: 'this one-line randomized dropout is what makes the two-signals-in-one-net trick work',
          },
        ]}
      />

      <Prose>
        <p>
          The last layer is what 99% of production code looks like.{' '}
          <code>diffusers</code> hides all of the above — the two signals, the
          doubled forward pass, the extrapolation — behind a single argument.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 3 — diffusers · cfg_pipeline.py">{`from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

prompt = "a corgi wearing sunglasses, 35mm film, sharp focus"

# The only CFG-relevant argument is guidance_scale — the dial itself.
# Everything you saw in layer 2 — null embedding, batched cond/uncond
# forward passes, the linear extrapolation — is happening inside this call.
for s in [1.0, 5.0, 7.5, 15.0]:
    image = pipe(
        prompt,
        guidance_scale=s,
        num_inference_steps=30,
    ).images[0]
    image.save(f"corgi_s{s}.png")`}</CodeBlock>

      <Bridge
        label="pytorch → diffusers"
        rows={[
          {
            left: 'torch.cat([x, x]); unet(x_in, t, c_in).chunk(2)',
            right: 'pipe(prompt, guidance_scale=s)',
            note: "the double forward pass is still happening — it's just inside the pipeline",
          },
          {
            left: 'UNCOND_DROP_PROB = 0.10 at train time',
            right: '(already trained — baked into the checkpoint)',
            note: 'Stable Diffusion was trained with 10% uncond dropout; you inherit that',
          },
          {
            left: 'x = scheduler.step(eps_hat, t, x).prev_sample',
            right: 'num_inference_steps=30',
            note: 'the denoising loop is a kwarg now',
          },
        ]}
      />

      <Callout variant="insight" title="the 2× compute cost is the whole deal">
        Every sampling step with guidance runs the U-Net twice — once for the
        conditional signal, once for the unconditional. That&apos;s roughly 2×
        the inference latency of unguided sampling. This is why
        &ldquo;turbo&rdquo; and &ldquo;fast&rdquo; variants of diffusion
        models often drop CFG or use distilled shortcuts: the compute budget
        is dominated by that second forward pass. When you see a
        &ldquo;guidance distillation&rdquo; paper, its entire goal is to fold
        the dial into a single-pass network.
      </Callout>

      {/* ── Gotcha ──────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">
            Forgetting the 10% dropout at train time:
          </strong>{' '}
          if the model never saw the null embedding during training, its
          unconditional signal at sampling time is garbage — there&apos;s no
          ambient hum, just noise — and cranking the dial just amplifies that
          garbage. The dropout is not optional; it&apos;s how the
          unconditional channel <em>exists</em>.
        </p>
        <p>
          <strong className="text-term-amber">
            Applying CFG to a model not trained for it:
          </strong>{' '}
          passing a null prompt to a plain conditional diffusion model
          doesn&apos;t give you a sensible <code>ε_uncond</code>. You&apos;ll
          get out-of-distribution noise and weird, broken images. Guidance is
          a train-time and inference-time contract together.
        </p>
        <p>
          <strong className="text-term-amber">
            Guidance only on some steps:
          </strong>{' '}
          some pipelines apply the dial only during part of the denoising
          trajectory (e.g. early steps) to save compute. This can work but
          subtly changes the output; if you compare samples across papers,
          check whether guidance was applied uniformly or scheduled.
        </p>
        <p>
          <strong className="text-term-amber">Negative prompts:</strong> in
          Stable Diffusion UIs the &ldquo;negative prompt&rdquo; is literally
          the unconditional embedding being replaced with a prompt for{' '}
          <em>things you don&apos;t want</em>. The dial then extrapolates
          away from those things. Same equation, different null choice.
        </p>
        <p>
          <strong className="text-term-amber">
            Very high scales (s &gt; 20):
          </strong>{' '}
          the linear extrapolation leaves the data manifold and artifacts
          appear. Some pipelines use dynamic thresholding (Imagen) to clip
          extreme values back in range. Unexplained saturation at high
          settings is usually this.
        </p>
      </Gotcha>

      {/* ── Challenge ────────────────────────────────────────────── */}
      <Challenge prompt="Train CFG on CIFAR-10 and watch the dial do its work">
        <p>
          Take a small class-conditional DDPM on CIFAR-10 (any of the
          reference implementations are fine — the whole model fits on a
          single GPU). Add the two CFG lines: (1) during training, drop the
          class label to a null token with probability 0.1; (2) during
          sampling, run the U-Net on both the class and the null token and
          combine with the guidance scale.
        </p>
        <p className="mt-2">
          Now generate 16 samples each for class <code>&ldquo;cat&rdquo;</code>{' '}
          at <code>s = 1</code>, <code>s = 5</code>, and <code>s = 10</code>.
          Lay them out in a 3×16 grid.
        </p>
        <p className="mt-2">
          What you should see as you crank the dial: at <code>s = 1</code>{' '}
          some samples are clearly cats, others are ambiguous or drifting
          toward neighboring classes. At <code>s = 5</code> almost every
          sample is unambiguously a cat, with more of the stereotypical
          &ldquo;catness&rdquo; the dataset contains. At <code>s = 10</code>{' '}
          they&apos;re all very obviously cats, but diversity collapses and
          the colors start to look chalky or over-saturated. Document it in a
          short write-up.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: compute the FID score at each scale. You will almost always
          see FID degrade as <code>s</code> increases past 2–3 — even though
          human raters judge the high-scale samples as &ldquo;more
          cat-like&rdquo;. That gap is the fidelity-adherence tradeoff in
          numbers.
        </p>
      </Challenge>

      {/* ── Takeaways + cliffhanger ─────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> CFG is one equation —{' '}
          <code>ε̂ = ε_uncond + s·(ε_cond − ε_uncond)</code> — and one
          training trick: randomly drop the caption 10% of the time so the
          same network learns both the conditional signal and the
          unconditional hum. At sampling you pay 2× compute for two forward
          passes, and you get a single dial (<code>s</code>) that trades
          diversity for prompt adherence. It is the standard conditioning
          mechanism in every text-to-image, text-to-audio, and text-to-video
          diffusion model deployed today.
        </p>
        <p>
          The beyond-image extensions all use the same equation with
          different conditioning: AudioLDM guides on text → spectrograms,
          VideoDM guides on text → video frames, and there&apos;s a growing
          body of work applying CFG-style guidance to autoregressive LLMs.
          When you see &ldquo;guidance scale&rdquo; in a paper about anything
          generative, it&apos;s almost certainly this same dial between two
          signals.
        </p>
        <p>
          <strong>Next up — Latent Diffusion.</strong> So far we&apos;ve been
          running diffusion directly in pixel space — which is fine for CIFAR
          at 32×32 but catastrophically expensive at 1024×1024. Latent
          diffusion runs the whole process in a compressed VAE latent space
          instead, making Stable Diffusion-scale models possible on consumer
          hardware. It&apos;s the single architectural trick that made
          text-to-image go from research demo to phone app. We&apos;ll build
          it next.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Classifier-Free Diffusion Guidance',
            author: 'Ho, Salimans',
            venue: 'NeurIPS Workshop 2021 / arXiv 2207.12598',
            year: 2022,
            url: 'https://arxiv.org/abs/2207.12598',
          },
          {
            title: 'Diffusion Models Beat GANs on Image Synthesis',
            author: 'Dhariwal, Nichol',
            venue: 'NeurIPS 2021 — the classifier guidance paper',
            year: 2021,
            url: 'https://arxiv.org/abs/2105.05233',
          },
          {
            title: 'High-Resolution Image Synthesis with Latent Diffusion Models',
            author: 'Rombach, Blattmann, Lorenz, Esser, Ommer',
            venue: 'CVPR 2022 — Stable Diffusion',
            year: 2022,
            url: 'https://arxiv.org/abs/2112.10752',
          },
          {
            title: 'GLIDE: Photorealistic Image Generation with Text-Guided Diffusion',
            author: 'Nichol, Dhariwal, Ramesh, et al.',
            venue: 'ICML 2022 — first large-scale text-to-image CFG',
            year: 2022,
            url: 'https://arxiv.org/abs/2112.10741',
          },
          {
            title: 'Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding',
            author: 'Saharia et al.',
            venue: 'NeurIPS 2022 — Imagen, dynamic thresholding for high CFG',
            year: 2022,
            url: 'https://arxiv.org/abs/2205.11487',
          },
          {
            title: 'On Distillation of Guided Diffusion Models',
            author: 'Meng, Rombach, Gao, et al.',
            venue: 'CVPR 2023 — folding CFG into a single forward pass',
            year: 2023,
            url: 'https://arxiv.org/abs/2210.03142',
          },
        ]}
      />
    </div>
  )
}
