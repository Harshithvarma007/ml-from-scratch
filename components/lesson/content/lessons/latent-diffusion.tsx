import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import WhatNext from '../WhatNext'
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
import PixelVsLatentCost from '../widgets/PixelVsLatentCost'
import VAELatentSpace from '../widgets/VAELatentSpace'

// Signature anchor: the thumbnail-shop upscaler. The lesson opens at the
// pixel-space compute wall (running diffusion on 786,432 numbers per step is
// a compute bill nobody can pay), reveals the autoencoder trick (compress
// every image into a tiny thumbnail, diffuse there, upscale at the end), and
// consolidates with why this one move made Stable Diffusion possible. The
// diffusion model doesn't know it's working on thumbnails — it just sees
// tensors. Peer lessons have their own images (sculptor, film, radio dial,
// hourglass, cookbook); the thumbnail-shop is ours alone.
export default function LatentDiffusionLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="latent-diffusion" />

      {/* ── Opening: the pixel-space compute wall ──────────────── */}
      <Prose>
        <p>
          A 512×512 RGB image has 786,432 numbers in it. A diffusion model has to run its U-Net
          on every one of those numbers, at every one of 1,000 denoising steps, for every
          sample. Do the multiplication: that&apos;s the compute budget of a small country per
          batch of pretty pictures. On an A100 it&apos;s tens of seconds per image. On your
          laptop it&apos;s a coffee break. The <NeedsBackground slug="ddpm-from-scratch">DDPM</NeedsBackground>{' '}
          paper (2020) was a miracle, but it was also a compute wall — diffusion stuck in pixel
          space is a pile of pixel arithmetic the field could not scale.
        </p>
        <p>
          Here&apos;s the move that broke the wall. Before diffusion runs, take every training
          image to a <strong>thumbnail shop</strong>: a small network compresses each 512×512
          photo down to a 64×64 tile — eight times smaller on a side, a quarter of a percent
          of the bits — and stores it there. Run the entire noisy denoising loop on those
          thumbnails, never on the real photos. When you&apos;re done, hand the finished
          thumbnail back to the shop and it upscales it into a full image. That&apos;s latent
          diffusion in one sentence. The <NeedsBackground slug="unet-architecture">UNet</NeedsBackground>{' '}
          doesn&apos;t know it&apos;s working on thumbnails — it just sees tensors.
        </p>
        <p>
          The thumbnail shop is an{' '}
          <NeedsBackground slug="convolution-operation">autoencoder</NeedsBackground> — specifically
          a VAE — trained once and frozen. <KeyTerm>Latent Diffusion Models</KeyTerm> (Rombach
          et al., 2022) took this embarrassingly simple idea and turned it into Stable
          Diffusion. Pixel diffusion wanted to paint every brushstroke on the full canvas. This
          lesson hires the thumbnail shop instead, diffuses on the compressed tile, and upscales
          once at the end. Same final image, a hundredth of the bill.
        </p>
        <p>
          By the end you should be able to sketch the Stable Diffusion architecture on a
          napkin — VAE encoder, latent U-Net, text encoder, VAE decoder — and not lie about
          any of it.
        </p>
      </Prose>

      {/* ── The compute wall, in numbers ────────────────────────── */}
      <Prose>
        <p>
          First the arithmetic that forced the whole field to change course. Here&apos;s what
          &ldquo;diffuse directly on pixels&rdquo; actually asks of a GPU — and what the
          thumbnail shop buys you in return:
        </p>
      </Prose>

      <MathBlock caption="the pixel-space cost — and what compression buys you">
{`pixel space:   512 × 512 × 3         =     786,432  dims
latent space:   64 ×  64 × 4         =      16,384  dims

compression ratio:
    786,432 / 16,384  =  48×    fewer numbers to touch per step

diffusion runs ~1000 steps per sample, so the savings compound:
    pixel:   786,432 × 1000  ≈  7.9 × 10⁸   dim·steps
    latent:   16,384 × 1000  ≈  1.6 × 10⁷   dim·steps

same image, ~48× less compute. (U-Net flops scale super-linearly
in spatial size, so the real wall-clock savings are often larger.)`}
      </MathBlock>

      <Prose>
        <p>
          The widget below puts dollars on it. Slide the resolution and the downsample factor
          and watch the training cost estimate move. Stable Diffusion 1.5 was trained for
          roughly $600k on LAION. The pixel-space equivalent — same U-Net, same steps, no
          thumbnail shop — would have cost well north of $30M, which is why it was never
          built.
        </p>
      </Prose>

      <PixelVsLatentCost />

      <Prose>
        <p>
          Pay attention to what happens at 8× downsample (the Stable Diffusion default). The
          cost drops by almost two orders of magnitude, and — here&apos;s the miracle —
          <em> perceptual quality barely moves</em>. Push the compression further (16×, 32×)
          and the thumbnails get mangled: the upscaler can&apos;t recover fine detail once
          you&apos;ve thrown it away. Pull back to 2× and you&apos;re paying for redundancy.
          4× linear / 8× area downsample is the sweet spot the Rombach paper landed on after
          an ablation grid, and everyone since has basically used the same ratio.
        </p>
      </Prose>

      <Callout variant="note" title="why pixels are so redundant">
        Natural images are not random tensors — they&apos;re drawn from an extremely thin
        manifold inside R⁷⁸⁶⁴³². Neighboring pixels correlate. Edges, textures, and faces
        occupy smooth sub-regions. A competent encoder can throw away most of the pixel-level
        bits and still recover everything you&apos;d actually perceive. That&apos;s the whole
        conceit of image compression (JPEG, H.264) and it&apos;s exactly what the thumbnail
        shop exploits.
      </Callout>

      {/* ── The thumbnail shop opens for business ──────────────── */}
      <Personify speaker="VAE (the thumbnail shop)">
        I am the compressor and the upscaler. I eat a 512×512 image and spit out a 4×64×64
        latent tile — 48× smaller, but still holding everything you need to reconstruct the
        picture. I was trained once, for a long time, with perceptual and adversarial losses,
        and now I&apos;m frozen. Diffusion does its thing in my latent space; I just encode
        photos going in and upscale thumbnails coming out. I&apos;m unglamorous infrastructure,
        and the entire generative image economy runs on me.
      </Personify>

      {/* ── Full pipeline math ──────────────────────────────────── */}
      <Prose>
        <p>
          Zoom out. Stable Diffusion is not a single model — it&apos;s four trained components
          stitched together. Here&apos;s the forward pass for text-to-image, end to end,
          with the thumbnail shop at both ends of the loop:
        </p>
      </Prose>

      <MathBlock caption="the full latent diffusion pipeline — text to pixels">
{`    "a cat wearing a space helmet"
              │
              ▼
    ┌─────────────────────┐
    │  CLIP / T5 encoder  │     frozen text tower
    └─────────────────────┘
              │   c ∈ R^(77 × 768)      conditioning vectors
              ▼
    ┌─────────────────────┐     z_T ~ N(0, I)   ← random latent noise
    │   Latent U-Net      │                      shape: 4 × 64 × 64
    │   (cross-attends c) │     loop T=1000 → 50 steps with DDIM
    └─────────────────────┘
              │   z_0        clean latent, 4 × 64 × 64
              ▼
    ┌─────────────────────┐
    │   VAE decoder       │     frozen, pretrained
    └─────────────────────┘
              │
              ▼
         512 × 512 × 3   RGB image

training objective (only the U-Net is learned here):
    L  =  E_{z_0, t, ε}  ‖ ε  −  ε_θ( z_t, t, c ) ‖²

       with z_0 = E_vae(x),   z_t = √ᾱ_t · z_0 + √(1−ᾱ_t) · ε`}
      </MathBlock>

      <Prose>
        <p>
          Three of the four blocks are frozen. The VAE encoder and decoder — the thumbnail
          shop&apos;s compress and upscale counters — were pretrained separately and never get
          touched during diffusion training. The text encoder (CLIP ViT-L for SD 1.5, T5-XXL
          for SD 3) is also frozen. Only the U-Net learns to denoise, and it learns inside
          the thumbnail shop&apos;s coordinate system. That factorization is the architectural
          insight — each module solves the problem it&apos;s good at, and none of them fight
          each other&apos;s loss.
        </p>
      </Prose>

      {/* ── Widget: what the thumbnail shop's latent space looks like ── */}
      <Prose>
        <p>
          Let&apos;s look at what the autoencoder actually <em>does</em> to an image. Pick a
          pair of source images, watch them get compressed down to their 4×64×64 thumbnails,
          and drag the slider to interpolate between those two latents and decode the result.
          This is the bit that makes the latent a useful substrate — linear mixes of
          thumbnails upscale into smooth, coherent blends of the source images, in a way
          that linear mixes of <em>pixels</em> never would.
        </p>
      </Prose>

      <VAELatentSpace />

      <Prose>
        <p>
          The interpolation is the point. In pixel space, averaging two images gives you a
          literal double-exposure — two faces ghosted on top of each other. In the thumbnail
          shop&apos;s latent, averaging two encodings and upscaling gives you something that
          looks like a plausible face halfway between them. The latent coordinates parameterize
          a smooth manifold of &ldquo;image-like things,&rdquo; and that smooth manifold is
          exactly what diffusion needs in order to trace a path from noise to a realistic
          sample.
        </p>
        <p>
          A quick note on VAE training, because the &ldquo;V&rdquo; matters. A vanilla
          autoencoder would learn to compress without any constraint on the latent
          distribution — the thumbnails could cluster into weird, disconnected islands. The
          VAE adds a KL term that pushes the latent distribution toward an isotropic Gaussian,
          which gives you the smooth, interpolable space you see above. Stable Diffusion&apos;s
          thumbnail shop goes further: it&apos;s trained with a <em>perceptual loss</em>{' '}
          (LPIPS — distance in a pretrained VGG feature space) and an <em>adversarial loss</em>{' '}
          (a discriminator, like in a GAN). Those two losses are what make the upscaler
          produce crisp, detailed reconstructions instead of the blurry porridge a plain
          L2-trained VAE gives you.
        </p>
      </Prose>

      <Callout variant="insight" title="the thumbnail shop does two jobs at once">
        It has to <em>compress</em> (reduce dims by 48×) and <em>regularize</em> (make the
        latent space smooth enough for diffusion to trace a path through it). Either job alone
        is easy; doing both well is the reason training a good image autoencoder takes weeks
        on a cluster. When people fine-tune Stable Diffusion they almost never retrain the
        VAE — they treat it as a fixed substrate, exactly as you&apos;d treat a BPE tokenizer
        in LLM work.
      </Callout>

      {/* ── Personify Latent diffusion ──────────────────────────── */}
      <Personify speaker="Latent diffusion">
        I&apos;m the cheap path. Pixel diffusion wanted to hire a million workers to paint
        every brushstroke on a 512×512 canvas; I send the image to the thumbnail shop, let
        four workers sketch the composition on the tiny tile, and have the shop upscale it
        back at the end. Same final output, a hundredth of the bill. That&apos;s why Stable
        Diffusion runs on your gaming laptop and DALL·E 2 runs on a datacenter.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, same story as every other lesson: a numpy toy that just counts flops,
          a pytorch sketch of the training loop with a real thumbnail shop in the pipeline,
          and the one-line diffusers call that&apos;s how you&apos;d actually use this in
          practice.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · pixel_vs_latent_cost.py"
        output={`pixel-space:  786,432 dims × 1000 steps × 50 flops/dim  =  3.93e+10 flops/sample
latent-space:  16,384 dims × 1000 steps × 50 flops/dim  =  8.19e+08 flops/sample
speedup:                                                 48.0×
for 1B training samples (SD-class run):
  pixel training cost estimate:   $30,720,000
  latent training cost estimate:      $640,000`}
      >{`import numpy as np

# image spec
H, W, C = 512, 512, 3
pixel_dims  = H * W * C                     # 786,432
latent_dims = (H // 8) * (W // 8) * 4       #  16,384   (8× spatial, 4 latent channels)

T           = 1000            # diffusion steps per sample
flops_per_d = 50              # crude per-dim U-Net flop estimate
samples     = 1_000_000_000   # SD-scale training dataset

pixel_cost  = pixel_dims  * T * flops_per_d
latent_cost = latent_dims * T * flops_per_d

print(f"pixel-space:  {pixel_dims:>7,} dims × {T} steps × {flops_per_d} flops/dim  =  {pixel_cost:.2e} flops/sample")
print(f"latent-space: {latent_dims:>7,} dims × {T} steps × {flops_per_d} flops/dim  =  {latent_cost:.2e} flops/sample")
print(f"speedup: {pixel_cost / latent_cost:>39.1f}×")

# assume $X per 1e15 flops on an A100
dollars_per_pflop = 0.04
print(f"for {samples/1e9:.0f}B training samples (SD-class run):")
print(f"  pixel training cost estimate:   \${pixel_cost  * samples / 1e15 * dollars_per_pflop:>12,.0f}")
print(f"  latent training cost estimate:  \${latent_cost * samples / 1e15 * dollars_per_pflop:>12,.0f}")`}</CodeBlock>

      <Prose>
        <p>
          That&apos;s the motivation. Now the thing itself — one training step of an LDM,
          using a real pretrained thumbnail shop to encode pixels, sampling a timestep,
          adding noise, predicting it back. This is what actually runs inside Stable
          Diffusion&apos;s <code>train_step()</code>.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — pytorch · ldm_train_step.py">{`import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

# ─── frozen substrate: VAE + text encoder ────────────────────────
vae        = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval()
tokenizer  = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_enc   = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").eval()

for p in vae.parameters():      p.requires_grad_(False)
for p in text_enc.parameters(): p.requires_grad_(False)

# ─── trainable: the U-Net that denoises in latent space ──────────
unet       = UNet2DConditionModel(sample_size=64, in_channels=4,
                                  out_channels=4, cross_attention_dim=768)
scheduler  = DDPMScheduler(num_train_timesteps=1000)

def train_step(images, captions, optimizer):
    # 1. encode pixels to latents.  shape 3×512×512  →  4×64×64
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215                # SD's magic scaling factor

    # 2. encode text  →  77 × 768  conditioning
    with torch.no_grad():
        tokens = tokenizer(captions, padding="max_length",
                           max_length=77, return_tensors="pt")
        cond   = text_enc(tokens.input_ids).last_hidden_state

    # 3. sample a timestep and a noise vector, noise the latent
    t       = torch.randint(0, 1000, (latents.size(0),))
    noise   = torch.randn_like(latents)
    noisy   = scheduler.add_noise(latents, noise, t)         # z_t

    # 4. predict the noise with the U-Net  (this is ε_θ(z_t, t, c))
    pred    = unet(noisy, t, encoder_hidden_states=cond).sample

    # 5. MSE between true noise and prediction  ← the diffusion loss
    loss    = F.mse_loss(pred, noise)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    return loss.item()`}</CodeBlock>

      <Bridge
        label="pixel-space DDPM → latent-space LDM"
        rows={[
          {
            left: 'noise = randn(3, 512, 512)',
            right: 'noise = randn(4, 64, 64)',
            note: '48× fewer numbers per step — the whole point',
          },
          {
            left: 'ε_θ(x_t, t)  U-Net over pixels',
            right: 'ε_θ(z_t, t, c)  U-Net over latents + cross-attn(text)',
            note: 'same loss, different domain + conditioning on CLIP text',
          },
          {
            left: 'image = sample()',
            right: 'latent = sample(); image = vae.decode(latent)',
            note: 'one VAE decode at the very end — O(1) overhead per sample',
          },
        ]}
      />

      <Prose>
        <p>
          In practice you never write any of that by hand. Hugging Face&apos;s{' '}
          <code>diffusers</code> library wraps the whole stack — thumbnail shop, text encoder,
          U-Net, scheduler,{' '}
          <NeedsBackground slug="classifier-free-guidance">guidance</NeedsBackground>, safety
          checker — behind one pipeline object. Three lines, one image.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — diffusers · stable_diffusion.py"
        output={`Loading pipeline components...
  - VAE              (frozen)
  - text_encoder     (frozen, CLIP-ViT-L)
  - unet             (trained, ~860M params)
  - scheduler        (DDIM, 50 steps)
100%|██████████| 50/50 [00:03<00:00, 16.21it/s]
saved to astronaut_horse.png  (512×512 RGB)`}
      >{`from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

# One call: text → CLIP → 50-step DDIM latent denoise → VAE decode → pixels.
# Everything we built in layer 2 lives inside .__call__(...)
image = pipe(
    prompt="an astronaut riding a horse on Mars, photorealistic, 4k",
    num_inference_steps=50,
    guidance_scale=7.5,           # classifier-free guidance strength
).images[0]

image.save("astronaut_horse.png")`}</CodeBlock>

      <Bridge
        label="pytorch LDM → diffusers pipeline"
        rows={[
          {
            left: 'vae, text_enc, unet, scheduler = …  # assemble by hand',
            right: 'pipe = StableDiffusionPipeline.from_pretrained(...)',
            note: 'one call pulls all four trained components from the hub',
          },
          {
            left: 'for t in timesteps:  z = denoise(z, t, c)',
            right: 'pipe(prompt, num_inference_steps=50)',
            note: 'the denoising loop is hidden inside the pipeline call',
          },
          {
            left: 'image = vae.decode(z) ; normalize ; to_uint8',
            right: '.images[0]',
            note: 'you get a PIL.Image back — no scaling gymnastics',
          },
        ]}
      />

      <Callout variant="insight" title="the point of the three layers">
        Layer 1 is the napkin math that justifies the entire architecture&apos;s existence —
        compression is where the savings are. Layer 2 is what&apos;s actually happening inside
        the training loop: compress pixels to thumbnails, noise the thumbnails, predict noise,
        backprop through the U-Net only. Layer 3 is what 99% of real usage looks like in 2026 —
        a pretrained pipeline call, because nobody is retraining the thumbnail shop or the
        text encoder.
      </Callout>

      {/* ── Democratization callout ─────────────────────────────── */}
      <Callout variant="insight" title="the economics that opened the floodgates">
        Before Stable Diffusion, state-of-the-art image generation (DALL·E 2, Imagen) lived
        behind API paywalls because the models were too big to give away. Latent diffusion
        shrank the compute bill for <em>both</em> training <em>and</em> inference to the point
        where Stability AI could just release the weights. That single decision in August 2022
        is why you have Dreambooth, LoRA, ControlNet, AUTOMATIC1111, ComfyUI, a hundred
        fine-tunes on Civitai, and an entire open-source ecosystem. None of it would exist if
        the models still took 40GB of VRAM to sample from. One thumbnail shop, one upscale
        call — that&apos;s what opened the floodgates.
      </Callout>

      <Callout variant="note" title="latents ≠ embeddings">
        The thumbnail shop&apos;s 4×64×64 latent is not the same species of object as a CLIP
        text embedding or an LLM&apos;s hidden state. It&apos;s a <em>spatial</em> tensor —
        you can still point to the top-left region and know you&apos;re looking at the top-left
        of the image — just with fewer channels than RGB and at 1/8 the resolution. The U-Net
        keeps its convolutional structure because the latent keeps its spatial structure.
        Don&apos;t confuse it with the semantic vector a language encoder produces.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Upscaler reconstruction artifacts:</strong>{' '}
          SD&apos;s thumbnail shop has known failure modes — &ldquo;fish-eye&rdquo; artifacts
          on eyes, wobble on text, grid patterns on large flat areas. These aren&apos;t the
          U-Net&apos;s fault; the upscaler itself cannot reconstruct those regions cleanly.
          Upgrading to <code>sd-vae-ft-mse</code> or the SDXL VAE fixes most of them.
        </p>
        <p>
          <strong className="text-term-amber">Wrong thumbnail shop for the LDM:</strong> every
          diffusion model is trained in <em>its own</em> autoencoder&apos;s coordinate system.
          Swapping in SDXL&apos;s VAE under a SD 1.5 U-Net gives you colorful static — the
          U-Net is denoising in the wrong coordinate frame. Keep the VAE paired with the
          U-Net you trained it under.
        </p>
        <p>
          <strong className="text-term-amber">Not scaling the latent:</strong> SD multiplies
          encoded latents by <code>0.18215</code> before diffusion and divides by the same
          number after. Forget the scaling and your thumbnails are an order of magnitude
          larger than the U-Net was trained to expect — outputs come out washed out or
          saturated.
        </p>
        <p>
          <strong className="text-term-amber">CFG in pixel space:</strong> classifier-free
          guidance (the <code>guidance_scale=7.5</code> knob) is defined on the U-Net&apos;s
          predicted noise, which lives in the thumbnail. Some older code re-implements CFG by
          upscaling conditional and unconditional latents and blending the <em>pixels</em>. It
          doesn&apos;t work — you get ghosting and color bleed. Guide the latents, upscale
          once at the end.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Measure the thumbnail shop's reconstruction error">
        <p>
          Load Stable Diffusion 1.5&apos;s VAE (<code>stabilityai/sd-vae-ft-mse</code>). Pick
          any 512×512 RGB image — a photograph, a screenshot, whatever. Send it through the
          compress side (encode to 4×64×64 thumbnail), then immediately through the upscale
          side (decode back to pixels) <em>without running any diffusion at all</em>. You now
          have a before and an after.
        </p>
        <p className="mt-2">
          Compute three things: the mean-squared error between the original and the
          reconstruction; the fraction of pixels that changed by more than 5/255 (i.e.
          perceptibly); and a side-by-side difference image (amplify the diff by 10× so you
          can see where the upscaler struggled). If you&apos;re on a face, the eyes and teeth
          will light up. If you&apos;re on text, the letterforms will.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: repeat with the SDXL VAE (<code>madebyollin/sdxl-vae-fp16-fix</code>) and
          observe how much cleaner the reconstruction is. That delta is most of the quality
          jump between SD 1.5 and SDXL — more than half of SDXL&apos;s gain came from the
          better thumbnail shop, not the bigger U-Net.
        </p>
      </Challenge>

      {/* ── Closing ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Latent diffusion is the trick that made
          generative images cheap: do the hard work (denoising) on a compressed thumbnail in a
          semantically dense latent, and let a pretrained autoencoder handle the compress/
          upscale round-trip to and from pixels. The thumbnail shop is infrastructure, not a
          model you retrain. The U-Net does the same ε-prediction as pixel-space DDPM, just on
          48× fewer numbers and with cross-attention on text. Stable Diffusion isn&apos;t a
          single architecture — it&apos;s four trained components (VAE encoder, text encoder,
          U-Net, VAE decoder) wired together, and the separation of concerns is why the whole
          stack is tractable.
        </p>
        <p>
          <strong>End of the Diffusion section.</strong> You&apos;ve walked from the forward
          noising process, through the reverse denoising ELBO, through the U-Net and DDIM,
          into classifier-free guidance and text conditioning, and out the other side with the
          thumbnail-shop architecture that actually powers Stable Diffusion. That&apos;s the
          whole story of a major generative modality.
        </p>
        <p>
          <strong>Next section — Reinforcement Learning, starting with Markov Decision
          Processes.</strong> We leave the world where a model is handed a static dataset of
          images and a fixed loss. RL is the regime where an agent takes actions in an
          environment, receives rewards, and has to figure out a policy from scratch. No
          labels, no thumbnails — just states, actions, and a signal that says &ldquo;that was
          better&rdquo; or &ldquo;that was worse.&rdquo; The first lesson builds the contract:
          states <em>s</em>, actions <em>a</em>, transition probabilities{' '}
          <em>P(s&apos; | s, a)</em>, rewards <em>r</em>, and the Markov assumption that
          tomorrow only depends on today. Bellman equations instead of log-likelihoods, policy
          gradients instead of supervised MLE. Bring a clean notebook.
        </p>
      </Prose>

      <WhatNext currentSlug="latent-diffusion" hidePrerequisites />

      <References
        items={[
          {
            title: 'High-Resolution Image Synthesis with Latent Diffusion Models',
            author: 'Rombach, Blattmann, Lorenz, Esser, Ommer',
            venue: 'CVPR 2022 — the Stable Diffusion / LDM paper',
            url: 'https://arxiv.org/abs/2112.10752',
          },
          {
            title: 'Auto-Encoding Variational Bayes',
            author: 'Kingma, Welling',
            venue: 'ICLR 2014 — the original VAE',
            url: 'https://arxiv.org/abs/1312.6114',
          },
          {
            title: 'Taming Transformers for High-Resolution Image Synthesis',
            author: 'Esser, Rombach, Ommer',
            venue: 'CVPR 2021 — the VQGAN paper that seeded the LDM VAE',
            url: 'https://arxiv.org/abs/2012.09841',
          },
          {
            title: 'diffusers — the Hugging Face diffusion library',
            author: 'von Platen et al.',
            venue: 'github.com/huggingface/diffusers',
            url: 'https://github.com/huggingface/diffusers',
          },
        ]}
      />
    </div>
  )
}
