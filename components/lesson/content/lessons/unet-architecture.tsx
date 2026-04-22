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
import UNetShape from '../widgets/UNetShape'                    // U-shaped diagram showing contracting + expanding paths with skip connections; click a layer for shapes
import TimestepConditioning from '../widgets/TimestepConditioning'  // visualize the sinusoidal timestep embedding being injected into each U-Net block

// Signature anchor: the hourglass with memory wires. Shape-first — the U-Net
// funnels the image down through a stack of narrowing encoder levels, pinches
// to a tiny neck in the middle, then un-funnels up through a mirror stack of
// decoder levels. The memory wires are the skip connections that jump from
// the top of the hourglass straight across to the bottom, carrying fine
// detail past the narrow neck where it would otherwise drown. Returned to at
// opening (why naive encoder-decoder loses detail), skip-connection reveal
// (the memory wires save the day), and "why this shape is perfect for
// diffusion" section.
export default function UnetArchitectureLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="unet-architecture" />

      {/* ── Opening: the hourglass problem ──────────────────────── */}
      <Prose>
        <p>
          Most neural nets you have met so far do one job: take a big thing and
          produce a small thing. A classifier eats a <code>3×224×224</code>{' '}
          image and emits a 1000-way probability. A regressor eats an image and
          emits a single number. One{' '}
          <NeedsBackground slug="convolution-operation">convolution</NeedsBackground>{' '}
          after another, one{' '}
          <NeedsBackground slug="pooling">pooling</NeedsBackground> layer after
          the next, the spatial grid gets crushed down until only a vector
          survives. The whole machine is a funnel.
        </p>
        <p>
          <NeedsBackground slug="denoising-intuition">Denoising</NeedsBackground>{' '}
          is weirder. Given a noisy image <code>x_t</code> of shape{' '}
          <code>3×H×W</code>, the network has to predict the noise{' '}
          <code>ε</code> at <em>every single pixel</em> — an output of shape{' '}
          <code>3×H×W</code>. Same resolution in, same resolution out. A plain
          funnel cannot do this. You would have to bolt a second, reversed
          funnel onto the first — compress, then decompress. An encoder taped
          to a decoder. Shape-wise, that works.
        </p>
        <p>
          And then the output looks like soup. Here is why: by the time your
          encoder has crushed <code>H×W</code> down to <code>H/16×W/16</code>,
          it has thrown the crisp edges away. The decoder only sees that tiny,
          compressed summary. It knows roughly where things live; it does not
          know where the exact pixel boundary is. A cell membrane becomes a
          smudge. Whiskers become a suggestion. The output is blurry because
          the information that made it sharp was last seen four downsamples
          ago, and nothing in the decoder ever gets to look at it again. You
          cannot re-invent pixel-accurate detail from a feature map the size
          of a postage stamp.
        </p>
        <p>
          Ronneberger and co-authors hit this wall in 2015 trying to segment
          cells in microscopy images. Their fix was the one fix that
          everything after has basically just reused: <em>don&apos;t make the
          decoder re-invent the detail — hand it across</em>. Staple wires
          from each encoder level directly to its mirror decoder level, so
          the high-resolution features skip the bottleneck entirely. They
          called the thing a <KeyTerm>U-Net</KeyTerm>, because when you draw
          the resolutions on a page, it looks like the letter U. Six years
          later, the diffusion crowd realized the same shape was the perfect
          denoiser, bolted on a timestep, and now it is the backbone of
          Stable Diffusion, DALL·E 2, Imagen, and every image-diffusion
          model you can name. Two careers, one hourglass.
        </p>
      </Prose>

      <Personify speaker="U-Net">
        I am an hourglass with memory wires. I funnel the image down through
        narrowing levels so my neck can see the whole scene at once. Then I
        un-funnel back up through a mirror stack of the same levels. And
        because squeezing loses fine detail, I run wires across the gap —
        every encoder level whispers what it saw directly to its matching
        decoder level. I am old, I am simple, and I am somehow still state of
        the art.
      </Personify>

      {/* ── The U shape, drawn ──────────────────────────────────── */}
      <Prose>
        <p>
          Before any math, stare at the shape. The name is not a marketing
          choice — it is literally what the data flow looks like when you draw
          the feature-map resolutions on a page. A funnel on the left, a
          mirror funnel on the right, a thin neck in the middle, and wires
          running straight across.
        </p>
      </Prose>

      <AsciiBlock caption="The hourglass — encoder funnel down on the left, decoder funnel up on the right, memory wires across">
{`         input  3×H×W
            │
     ┌──────▼──────┐    encoder block 1 — ────────────────────┐ wire
     │  H×W, 64ch  │                                           │
     └──────┬──────┘                                           │
        downsample (↓2)                                        │
     ┌──────▼──────┐    encoder block 2 — ──────────────┐ wire │
     │ H/2×W/2,128 │                                     │     │
     └──────┬──────┘                                     │     │
        downsample                                       │     │
     ┌──────▼──────┐    encoder block 3 — ────────┐ wire│     │
     │ H/4×W/4,256 │                               │     │     │
     └──────┬──────┘                               │     │     │
        downsample                                 │     │     │
     ┌──────▼──────┐    encoder block 4 — ──┐ wire│     │     │
     │ H/8×W/8,512 │                         │     │     │     │
     └──────┬──────┘                         │     │     │     │
        downsample                           │     │     │     │
     ┌──────▼──────┐                         │     │     │     │
     │    NECK     │   H/16×W/16, 512ch      │     │     │     │
     │  (self-attn)│   sees the whole image  │     │     │     │
     └──────┬──────┘                         │     │     │     │
        upsample (↑2)                        │     │     │     │
     ┌──────▼──────┐    decoder block 4  ◀───┘     │     │     │
     │ H/8×W/8,512 │      concat + conv            │     │     │
     └──────┬──────┘                               │     │     │
        upsample                                    │     │     │
     ┌──────▼──────┐    decoder block 3  ◀─────────┘     │     │
     │ H/4×W/4,256 │                                      │     │
     └──────┬──────┘                                      │     │
        upsample                                          │     │
     ┌──────▼──────┐    decoder block 2  ◀────────────────┘     │
     │ H/2×W/2,128 │                                            │
     └──────┬──────┘                                            │
        upsample                                                │
     ┌──────▼──────┐    decoder block 1  ◀──────────────────────┘
     │  H×W, 64ch  │
     └──────┬──────┘
            │
         output  3×H×W    (noise estimate ε̂, same shape as input)`}
      </AsciiBlock>

      <Prose>
        <p>
          Two things to notice. First: the input and output have{' '}
          <em>identical</em> spatial dimensions. H and W at the top, H and W
          at the bottom. That constraint is the whole reason the hourglass
          exists. Second: every encoder level has a wire running across to
          its mirror decoder level. Those are the{' '}
          <KeyTerm>skip connections</KeyTerm>, and they are the only thing
          separating a real U-Net from the naive encoder-taped-to-decoder
          that returns soup.
        </p>
      </Prose>

      {/* ── Skip-connection reveal ──────────────────────────────── */}
      <Prose>
        <p>
          Here is the reveal. The encoder funnel and the decoder funnel are
          not doing the same job. The encoder&apos;s job is to get{' '}
          <em>abstract</em> — to pack more and more global context into
          fewer and fewer spatial positions until the neck holds a single
          view of the whole image. The decoder&apos;s job is the opposite:
          to get <em>sharp</em>, to reinflate that global view into a
          pixel-exact output. Those two jobs need different information.
          Abstraction wants low-frequency summaries; sharpness wants
          high-frequency edges.
        </p>
        <p>
          The bottleneck has the abstraction. It threw the edges away at
          each downsample — that was the point. Ask the decoder to
          reconstruct whiskers from a <code>H/16×W/16</code> feature map
          and it is guessing. The skip wires are the fix: the encoder
          keeps a copy of its high-resolution features before it crushes
          them, and hands that copy directly across the hourglass to the
          decoder at the matching resolution. The decoder concatenates
          the skip onto its upsampled blur, runs a conv to mix them, and
          the whiskers snap back. The wire does nothing in the forward
          pass except <em>exist</em>. And that is exactly why it works —
          no information had to survive the squeeze.
        </p>
      </Prose>

      <Personify speaker="Skip connection">
        I am the memory wire. Down at the neck everything is a blurry
        abstraction — the network can tell you there is a cat in the
        lower-left, but not where its whiskers end. I take a copy of the
        encoder&apos;s high-resolution features and run straight across the
        hourglass to the mirror decoder level, skipping the bottleneck
        entirely. The decoder concatenates me onto its upsampled blur, a
        conv mixes us, and the whiskers snap into focus. I carry detail
        past the pinch where it would drown.
      </Personify>

      {/* ── Shape arithmetic ────────────────────────────────────── */}
      <MathBlock caption="Hourglass shape arithmetic — channels double as resolution halves">
{`Let (H, W, C) denote (height, width, channel count) of a feature map.

Standard four-stage U-Net with base channel count C₀ = 64:

  level 0 (encoder 1):    H        × W        × 64      ─┐
  level 1 (encoder 2):    H/2      × W/2      × 128     ─┤  memory
  level 2 (encoder 3):    H/4      × W/4      × 256     ─┤  wires
  level 3 (encoder 4):    H/8      × W/8      × 512     ─┤
  level 4 (neck):         H/16     × W/16     × 512     (or 1024)
  level 3 (decoder 4):    H/8      × W/8      × 512     ◀┤
  level 2 (decoder 3):    H/4      × W/4      × 256     ◀┤  mirror
  level 1 (decoder 2):    H/2      × W/2      × 128     ◀┤
  level 0 (decoder 1):    H        × W        × 64      ◀┘
  output head:            H        × W        × 3       (or input channels)

Channel count roughly doubles at every downsample, halves at every upsample.
Total pixels at each level:
      H·W,  H·W/4,  H·W/16,  H·W/64,  H·W/256  →  geometric decay.
Most of the compute is at the top of the hourglass — high-res, fewer channels.
Most of the "thinking" is at the bottleneck — low-res, all global context.`}
      </MathBlock>

      <Prose>
        <p>
          The channel-doubling trick is not accidental. As resolution halves,
          you have four times fewer spatial positions, so you can afford
          twice as many channels before compute per block goes up. Roughly
          constant FLOPs per level of the funnel, more abstract features as
          you descend toward the neck. It is the same pattern you saw in{' '}
          <NeedsBackground slug="resnet-and-skip-connections">
            skip connections
          </NeedsBackground>{' '}
          and ResNet stages, for the same reason.
        </p>
      </Prose>

      {/* ── Widget 1: U-Net Shape ──────────────────────────────── */}
      <Prose>
        <p>
          Here is the hourglass drawn interactively. Click any level — encoder
          side or mirror decoder side — to see the exact feature map shape at
          that point for a <code>256×256</code> input. Pay attention to the
          memory wires: they always connect <em>matching resolutions</em>,
          never crossing levels. And notice how each decoder block&apos;s
          channel count stays higher than you&apos;d expect right after the
          upsample — that is because the skip was just concatenated onto the
          upsampled tensor, and the conv inside the block halves it back
          down.
        </p>
      </Prose>

      <UNetShape />

      <Callout variant="note" title="concatenate, not add">
        ResNet skip connections <em>add</em>: <code>y = F(x) + x</code>. U-Net
        skip connections <em>concatenate</em> along the channel axis:{' '}
        <code>y = F(concat[upsample(z), x_skip])</code>. Why the difference?
        ResNets live at constant resolution — addition needs matching shapes.
        U-Net wires splice a high-res encoder tensor onto an upsampled low-res
        decoder tensor, same spatial shape but carrying different information.
        Concatenation lets the next conv decide how to mix them; addition
        would pre-blur that choice into a single sum.
      </Callout>

      {/* ── Why this shape is perfect for diffusion ─────────────── */}
      <Prose>
        <p>
          <strong>Why this shape is perfect for diffusion.</strong> Step back
          and think about what diffusion actually asks of a network. You
          hand it a noisy image and say: tell me, at every pixel, what noise
          you added. The output has to be pixel-exact (<code>3×H×W</code>{' '}
          out, same as in). The answer at any given pixel depends on the{' '}
          <em>whole scene</em> — the model can&apos;t know what noise lives
          on a cat&apos;s ear unless it knows there is a cat. Global context
          in, pixel-precise output. Soup not acceptable.
        </p>
        <p>
          That is the hourglass&apos;s entire design brief. The funnel down
          aggregates global context into the neck, where every position can
          see everything. The funnel up reinflates that decision back to the
          original grid. The memory wires keep the output from going soupy
          by carrying fine detail across the pinch. Input shape preserved,
          global reasoning available at the neck, crisp edges delivered at
          the mirror top — every requirement diffusion has, the U-Net
          already met in 2015 for a completely different problem. The
          architecture did not need to be invented for diffusion. It needed
          to be <em>noticed</em>.
        </p>
        <p>
          One extra wrinkle. The same network has to denoise at{' '}
          <em>every</em> noise level, from faint haze (<code>t=1</code>) to
          pure Gaussian static (<code>t=1000</code>). If it does not know
          which level it is looking at, it cannot know how aggressively to
          clean. So we inject a timestep signal into every block of the
          hourglass — encoder funnel, neck, decoder funnel, every single
          one. That is the one thing the diffusion people added on top of
          the 2015 shape.
        </p>
      </Prose>

      {/* ── Timestep conditioning math ──────────────────────────── */}
      <Prose>
        <p>
          The timestep is a single integer. The network needs a smooth,
          expressive encoding of it that every block can read. The standard
          recipe is the same one the Transformer used for token positions:
          map the scalar <code>t</code> into a high-dimensional sinusoidal
          vector, push it through a tiny MLP, then <em>add</em> it into the
          activations inside every residual block.
        </p>
      </Prose>

      <MathBlock caption="sinusoidal timestep embedding, then per-block injection">
{`1) Sinusoidal embedding — turn scalar t into a d-dim vector.

    for i = 0, 1, …, d/2 − 1:

        emb[t, 2i]     = sin( t / 10000^(2i/d) )
        emb[t, 2i+1]   = cos( t / 10000^(2i/d) )

    Different frequencies for different indices — the network can read
    coarse ("what step bucket?") and fine ("exactly which step?") info
    from different slices of the same vector.

2) Learnable projection — match the block's channel count.

    t_emb   =   MLP(emb[t])         # Linear → SiLU → Linear
                                    # shape: (d_model,)

3) Per-block injection — add as a bias on every channel.

    h   =   conv1(x)                # shape: (B, C, H, W)
    h   =   h   +   proj(t_emb)[None, :, None, None]       # broadcast
    h   =   SiLU(GroupNorm(h))
    h   =   conv2(h)
    out =   h   +   skip(x)         # residual, inside the block

    The "+ proj(t_emb)" line is what makes this a diffusion U-Net. Every
    residual block — encoder, neck, mirror decoder — gets the timestep
    added as a channel-wise shift. Remove it and the network has no idea
    what noise level it is denoising.`}
      </MathBlock>

      <Prose>
        <p>
          The sinusoidal part is recycled wisdom. Same trick the original
          Transformer used for token positions, for the same reason: a
          smooth, periodic, infinitely-distinguishable encoding of a scalar
          that the network does not have to learn from scratch. Slow sines
          give you &ldquo;rough region of <code>t</code>&rdquo;; fast sines
          give you &ldquo;exactly which step&rdquo;. The downstream MLP
          picks whichever granularity it needs.
        </p>
        <p>
          The <em>injection</em> point is the real design choice. Every
          residual block in the hourglass — top of the funnel down, every
          level of the encoder, the neck itself, every mirror level of the
          decoder funnel — adds <code>t_emb</code> as a per-channel bias on
          its hidden activations. Not once at the input. Every block. That
          is how the model gets a strong, repeated reminder of &ldquo;this
          is noise level <code>t</code>, adjust accordingly&rdquo; at every
          scale of processing.
        </p>
      </Prose>

      {/* ── Widget 2: Timestep Conditioning ─────────────────────── */}
      <Prose>
        <p>
          Here is the timestep embedding and its injection, made visible.
          Drag the <code>t</code> slider and watch two things. First, the
          sinusoidal vector changes smoothly — slow components evolve
          slowly, fast components ripple. No hard jumps. Second, the
          colored bias bars alongside each block of the hourglass change
          too: that is the per-channel shift being added to the activations
          inside every residual block, telling that block what noise level
          to denoise.
        </p>
      </Prose>

      <TimestepConditioning />

      <Personify speaker="Timestep embedding">
        I am the clock. At <code>t = 1</code> I whisper to the network
        &ldquo;barely any noise, refine gently&rdquo;. At <code>t = 1000</code>{' '}
        I shout &ldquo;pure static, aim for a plausible image, don&apos;t be
        shy&rdquo;. Every block of the hourglass hears me — I get added as a
        channel-wise bias inside each one. I cost almost nothing and I am
        the reason one network can denoise a thousand different corruption
        levels with the same weights.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations, the usual ladder. Pure Python to build a
          single residual block with timestep injection — arithmetic
          visible, nothing hidden. PyTorch to wire the blocks into an
          actual tiny U-Net for MNIST. Diffusers to show the packaged
          version you would reach for in production.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · unet_block_scratch.py"
        output={`input shape  : (1, 8, 16, 16)
t_emb shape  : (1, 32)
output shape : (1, 16, 16, 16)
t=10   mean bias contribution: 0.0847
t=500  mean bias contribution: -0.0214
t=999  mean bias contribution: 0.1356
# same weights, different t → different output. That is the whole point.`}
      >{`import math
import numpy as np

# ---- 1. Sinusoidal timestep embedding --------------------------------------
def sinusoidal_embedding(t, dim):
    """Map scalar t (or batch of them) to a dim-dimensional vector."""
    half = dim // 2
    freqs = np.exp(-math.log(10000) * np.arange(half) / half)   # (half,)
    args  = np.asarray(t)[:, None] * freqs[None, :]             # (B, half)
    return np.concatenate([np.sin(args), np.cos(args)], axis=1) # (B, dim)

# ---- 2. One U-Net residual block, with t injected --------------------------
def unet_block(x, t_emb, W1, W2, W_t, b_t):
    """
    x      : (B, C_in,  H, W)   — input feature map
    t_emb  : (B, D)              — timestep embedding
    W1, W2 : conv weights        — shape (C_out, C_in, 3, 3), (C_out, C_out, 3, 3)
    W_t, b_t : projection from t_emb to a per-channel bias — (D, C_out), (C_out,)
    """
    # Conv-1 (fake it with a channel-wise transform for brevity)
    h = np.einsum("bchw,oc->bohw", x, W1[..., 0, 0])           # shape (B, C_out, H, W)

    # Timestep bias — project t_emb down to C_out, broadcast over H, W
    t_bias = t_emb @ W_t + b_t                                 # (B, C_out)
    h = h + t_bias[:, :, None, None]                           # ← the injection line

    # Nonlinearity + Conv-2
    h = np.maximum(0, h)                                       # stand-in for SiLU
    h = np.einsum("bchw,oc->bohw", h, W2[..., 0, 0])

    # Residual skip
    return h + x if x.shape[1] == h.shape[1] else h

# ---- 3. Try it out ---------------------------------------------------------
rng = np.random.default_rng(0)
B, C_in, C_out, H, W, D = 1, 8, 16, 16, 16, 32
x = rng.normal(size=(B, C_in, H, W))

W1  = rng.normal(size=(C_out, C_in, 3, 3)) * 0.1
W2  = rng.normal(size=(C_out, C_out, 3, 3)) * 0.1
W_t = rng.normal(size=(D, C_out)) * 0.1
b_t = np.zeros(C_out)

for t in [10, 500, 999]:
    t_emb = sinusoidal_embedding(np.array([t]), D)
    y     = unet_block(x, t_emb, W1, W2, W_t, b_t)
    print(f"t={t}   mean bias contribution: {(t_emb @ W_t + b_t).mean():.4f}")

print("input shape  :", x.shape)
print("t_emb shape  :", t_emb.shape)
print("output shape :", y.shape)`}</CodeBlock>

      <Bridge
        label="math ←→ numpy"
        rows={[
          {
            left: 'emb[t, 2i]   = sin(t / 10000^(2i/d))',
            right: 'freqs = exp(-log(10000) * i / half)',
            note: 'log-space is numerically safer than pow(10000, …)',
          },
          {
            left: 'h + proj(t_emb)[None, :, None, None]',
            right: 'h + t_bias[:, :, None, None]',
            note: 'broadcast the (B, C_out) bias over H and W',
          },
          {
            left: 'per-block injection',
            right: 'every call to unet_block() takes t_emb',
            note: 'not shared across blocks — each block has its own W_t',
          },
        ]}
      />

      <Prose>
        <p>
          Now put the blocks in the hourglass. This is a minimal PyTorch
          U-Net — three encoder levels down the funnel, a neck, three mirror
          decoder levels back up, trained on 28×28 MNIST. Small enough to
          fit on a laptop, complete enough that its skeleton maps directly
          onto Stable Diffusion&apos;s.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — pytorch · tiny_unet_mnist.py">{`import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Timestep embedding -----------------------------------------------
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):                       # t : (B,)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args  = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=1)    # (B, dim)

# ---------- U-Net residual block with time conditioning ----------------------
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)              # timestep → channel bias
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]  # ← inject timestep
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

# ---------- The U-Net itself -------------------------------------------------
class TinyUNet(nn.Module):
    def __init__(self, t_dim=64):
        super().__init__()
        self.t_emb  = nn.Sequential(SinusoidalTimeEmb(t_dim),
                                    nn.Linear(t_dim, t_dim), nn.SiLU(),
                                    nn.Linear(t_dim, t_dim))
        # Encoder funnel (down)
        self.d1 = Block(1,   32, t_dim)
        self.d2 = Block(32,  64, t_dim)
        self.d3 = Block(64, 128, t_dim)
        # Neck
        self.mid = Block(128, 128, t_dim)
        # Decoder funnel (up) — input channels DOUBLED because of the memory wire concat
        self.u3 = Block(128 + 128, 64,  t_dim)
        self.u2 = Block(64  + 64,  32,  t_dim)
        self.u1 = Block(32  + 32,  32,  t_dim)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        te = self.t_emb(t)                                      # (B, t_dim)
        # ---- Encoder funnel down — stash each level for the memory wire -----
        s1 = self.d1(x,                  te);   p1 = F.avg_pool2d(s1, 2)   # 28 → 14
        s2 = self.d2(p1,                 te);   p2 = F.avg_pool2d(s2, 2)   # 14 → 7
        s3 = self.d3(p2,                 te);   p3 = F.avg_pool2d(s3, 2)   #  7 → 3 (roughly)
        # ---- Neck ------------------------------------------------------------
        m  = self.mid(p3,                te)
        # ---- Decoder funnel up: upsample, concat memory wire, block --------
        u3 = F.interpolate(m,  size=s3.shape[-2:], mode="nearest")
        u3 = self.u3(torch.cat([u3, s3], dim=1), te)
        u2 = F.interpolate(u3, size=s2.shape[-2:], mode="nearest")
        u2 = self.u2(torch.cat([u2, s2], dim=1), te)
        u1 = F.interpolate(u2, size=s1.shape[-2:], mode="nearest")
        u1 = self.u1(torch.cat([u1, s1], dim=1), te)
        return self.out(u1)                                     # (B, 1, 28, 28) ← same as input

# ---- Sanity check: input and output spatial dims must match -----------------
net = TinyUNet()
x   = torch.randn(4, 1, 28, 28)
t   = torch.randint(0, 1000, (4,))
y   = net(x, t)
print("input :", x.shape, "→  output :", y.shape)
print("params:", sum(p.numel() for p in net.parameters()) / 1e6, "M")`}</CodeBlock>

      <Bridge
        label="pure python → pytorch"
        rows={[
          {
            left: 'sinusoidal_embedding(t, dim)',
            right: 'SinusoidalTimeEmb(dim) → MLP',
            note: 'raw sinusoids + a two-layer MLP — standard recipe',
          },
          {
            left: 'h + t_bias[:, :, None, None]',
            right: 'h + self.t_proj(F.silu(t_emb))[:, :, None, None]',
            note: 'each Block owns its own projection — unique per level',
          },
          {
            left: 'x = h + x   # addition',
            right: 'torch.cat([u3, s3], dim=1)',
            note: 'memory wires CONCAT, then the block halves channels again',
          },
        ]}
      />

      <Prose>
        <p>
          The decoder blocks take double the channels because the upsampled
          tensor and its incoming memory wire are concatenated before the
          conv. After the block, the channel count is back to what you
          would expect for that level of the mirror. This is where most
          first-time U-Net bugs live — off-by-factor-of-two channel
          mismatches at the wire site.
        </p>
        <p>
          And the production version. Hugging Face&apos;s{' '}
          <code>UNet2DModel</code> is what sits inside Stable Diffusion and
          its cousins — a full hourglass with ResNet blocks, self-attention
          at low resolutions, timestep conditioning, and (for text-to-image)
          cross-attention.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — diffusers · unet2d_stable_diffusion.py"
        output={`UNet2DModel loaded — 273.9M params
sample shape: torch.Size([2, 3, 64, 64]) → prediction shape: torch.Size([2, 3, 64, 64])
block types (encoder): ('DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D')
block types (decoder): ('AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D')
# Notice: self-attention is only at the lower-resolution levels.`}
      >{`import torch
from diffusers import UNet2DModel

# A UNet2D for unconditional 64×64 diffusion, the same topology used by
# many image-diffusion papers before Stable Diffusion's text-conditioned UNet.
unet = UNet2DModel(
    sample_size=64,                           # input spatial dim (H = W)
    in_channels=3,                            # RGB in
    out_channels=3,                           # RGB noise estimate out
    layers_per_block=2,                       # two ResNet blocks per level
    block_out_channels=(128, 256, 512, 512),  # channel count per encoder level
    down_block_types=(
        "DownBlock2D",                        # plain conv block (top of funnel)
        "AttnDownBlock2D",                    # + self-attention (32×32)
        "AttnDownBlock2D",                    # + self-attention (16×16)
        "AttnDownBlock2D",                    # + self-attention (8×8 — near neck)
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
)

print(f"UNet2DModel loaded — {sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M params")

# Forward pass: (noisy_sample, timestep) → predicted noise
x = torch.randn(2, 3, 64, 64)
t = torch.tensor([10, 500])
pred = unet(x, t).sample                      # identical shape to x
print("sample shape:", x.shape, "→ prediction shape:", pred.shape)
print("block types (encoder):", tuple(type(b).__name__ for b in unet.down_blocks))
print("block types (decoder):", tuple(type(b).__name__ for b in unet.up_blocks))`}</CodeBlock>

      <Bridge
        label="tiny pytorch → diffusers"
        rows={[
          {
            left: 'Block(in, out, t_dim)',
            right: 'ResnetBlock2D inside every Down/Up block',
            note: 'diffusers bundles norm + conv + time-proj + residual',
          },
          {
            left: 'self.mid = Block(…)',
            right: 'UNetMidBlock2D with self-attention',
            note: 'the neck gets a full attention block — global context',
          },
          {
            left: 'one forward → predict ε',
            right: 'unet(x, t).sample',
            note: '.sample is just the tensor; diffusers wraps outputs for convenience',
          },
        ]}
      />

      <Callout variant="insight" title="text-to-image = cross-attention on CLIP embeddings">
        Stable Diffusion&apos;s hourglass has one more trick. Each ResNet
        block is followed by a transformer block containing <em>both</em>{' '}
        self-attention (spatial positions attending to each other) and{' '}
        <em>cross-attention</em>, where the queries come from the image
        feature map and the keys/values come from a text embedding —
        typically CLIP&apos;s output for the prompt. That is the entire
        text-conditioning mechanism: the prompt is turned into a sequence
        of embeddings once, and every block of the U attends to it. Swap
        CLIP for T5 and you have Imagen; add a longer prompt budget and you
        have SDXL. The hourglass does not change — the attention
        keys/values just change what they are pointing at.
      </Callout>

      <Callout variant="note" title="why self-attention only near the neck">
        Attention is <code>O(N²)</code> in sequence length, where N is the
        number of spatial positions. At <code>64×64</code> that is 4096
        tokens — 16M attention weights per head, painful. At{' '}
        <code>8×8</code> it is 64 tokens — trivial. So U-Nets put
        self-attention near the bottom of the hourglass, where each
        position can cheaply see the whole downsampled image, and leave the
        top-of-funnel levels as plain convs. Global reasoning and
        pixel-precise output without paying quadratic cost at every scale.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Channel doubling at each downsample.</strong>{' '}
          A U-Net that keeps channel count fixed across levels looks fine in
          theory and performs poorly in practice — the neck does not have
          enough capacity to hold global context. Roughly double channels
          at every downsample, halve at every upsample. (Some modern
          variants cap at e.g. 512 to save params — fine, but halve
          consistently.)
        </p>
        <p>
          <strong className="text-term-amber">Memory wire shape matching.</strong>{' '}
          The wire from encoder level <code>k</code> concatenates with the
          upsampled tensor at mirror decoder level <code>k</code>. Spatial
          dims must match exactly. If your input size is not a clean
          multiple of <code>2^(num_levels)</code>, rounding in the
          downsamples leaves the upsampled tensor off by a pixel. Fix: pad
          inputs to a multiple of <code>2^(levels)</code>, or use{' '}
          <code>F.interpolate(..., size=skip.shape[-2:])</code> instead of a
          fixed scale factor.
        </p>
        <p>
          <strong className="text-term-amber">Conditioning at the wrong layer.</strong>{' '}
          A first-pass mistake is to concatenate <code>t_emb</code> onto the
          input tensor once at the start. Technically works, empirically
          fails. The repeated per-block injection is what lets every scale
          of the hourglass adapt to the noise level. Inject everywhere —
          every encoder block, the neck, every mirror decoder block.
        </p>
        <p>
          <strong className="text-term-amber">Missing timestep dependency.</strong>{' '}
          If you forget to pass <code>t</code> into the blocks entirely, the
          network still trains — it just learns a single mean denoiser that
          does nothing useful. Sanity check: run two forward passes with
          the same <code>x</code> and different <code>t</code>. The outputs
          must be numerically different. Bit-identical means your timestep
          is not wired in.
        </p>
        <p>
          <strong className="text-term-amber">Concat vs add confusion.</strong>{' '}
          ResNet residuals: <code>+</code>. U-Net memory wires:{' '}
          <code>cat</code>. The decoder block&apos;s input channel count is
          the <em>sum of upsampled + wire channels</em>, not just the
          upsampled count. Get this wrong and PyTorch throws a channel
          mismatch at the first wire site.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Build and sanity-check a tiny 28×28 hourglass">
        <p>
          Take the PyTorch <code>TinyUNet</code> above and wire up three
          sanity checks before any training. First: pass a batch of shape{' '}
          <code>(8, 1, 28, 28)</code> with random <code>t</code> and verify
          the output is also <code>(8, 1, 28, 28)</code>. Any shape
          mismatch in the memory wires or channel counts will blow up
          here.
        </p>
        <p className="mt-2">
          Second: check timestep dependency. Call the network twice with
          the same <code>x</code> and two different <code>t</code> values
          (say 10 and 990). The outputs must differ — if they are
          bit-identical, your timestep embedding is not being read.
        </p>
        <p className="mt-2">
          Third: check memory-wire dependency. Monkey-patch the forward
          pass to replace each skip tensor with{' '}
          <code>torch.zeros_like(skip)</code> before concatenation. Run one
          backward and measure the loss. It should be meaningfully worse
          than the un-ablated network — if the loss barely moves, your
          wires are not carrying useful information, which means your
          hourglass is effectively a plain encoder-decoder and will
          produce soup.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: train it for one epoch on MNIST with the DDPM objective
          (random <code>t</code>, add noise, predict it). Sample from the
          trained model after epoch 1. The samples will look like garbage —
          but they should look like <em>structured</em> garbage, not pure
          static. That is your first diffusion model, running.
        </p>
      </Challenge>

      {/* ── Closing ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> A U-Net is the hourglass
          with memory wires: funnel down for global context, neck in the
          middle, mirror funnel up for pixel-exact output, and skip
          connections running from each encoder level straight across to
          its mirror decoder level so fine detail survives the pinch.
          Channels roughly double at every downsample, halve at every
          upsample. For diffusion we add two things: a sinusoidal timestep
          embedding injected as a channel-wise bias inside every block of
          the hourglass, and (for text-to-image) cross-attention to a text
          embedding at the neck. Self-attention lives near the bottleneck
          where it is cheap. The whole skeleton fits in ~200 lines of
          PyTorch and is the backbone of every image diffusion model in
          production.
        </p>
        <p>
          <strong>Next up — DDPM from Scratch.</strong> We have the
          hourglass. Next lesson we build the training objective on top of
          it — the forward noising process, the reverse sampler, and the
          famously simple L2 loss that does all the work. You will write
          the full DDPM loop, train the tiny U-Net from this lesson on
          MNIST, and sample digits out of pure Gaussian noise. That is the
          moment the rest of this section stops being theory.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'U-Net: Convolutional Networks for Biomedical Image Segmentation',
            author: 'Ronneberger, Fischer, Brox',
            venue: 'MICCAI 2015 — the original U-Net paper',
            url: 'https://arxiv.org/abs/1505.04597',
          },
          {
            title: 'Denoising Diffusion Probabilistic Models',
            author: 'Ho, Jain, Abbeel',
            venue: 'NeurIPS 2020 — DDPM, the U-Net applied to diffusion',
            url: 'https://arxiv.org/abs/2006.11239',
          },
          {
            title: 'Diffusion Models Beat GANs on Image Synthesis',
            author: 'Dhariwal, Nichol',
            venue: 'NeurIPS 2021 — architectural scaling of the diffusion U-Net',
            url: 'https://arxiv.org/abs/2105.05233',
          },
          {
            title: 'High-Resolution Image Synthesis with Latent Diffusion Models',
            author: 'Rombach, Blattmann, Lorenz, Esser, Ommer',
            venue: 'CVPR 2022 — the Stable Diffusion paper',
            url: 'https://arxiv.org/abs/2112.10752',
          },
          {
            title: 'diffusers — UNet2DModel reference implementation',
            author: 'Hugging Face',
            venue: 'the one you\'ll actually use',
            url: 'https://huggingface.co/docs/diffusers/api/models/unet2d',
          },
          {
            title: 'Attention Is All You Need',
            author: 'Vaswani et al.',
            venue: 'NeurIPS 2017 — origin of the sinusoidal embedding recycled here',
            url: 'https://arxiv.org/abs/1706.03762',
          },
        ]}
      />
    </div>
  )
}
