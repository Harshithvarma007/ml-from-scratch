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
import FloatToInt from '../widgets/FloatToInt'
import PrecisionLadder from '../widgets/PrecisionLadder'

// Signature anchor: a trained model is a grand symphony score written out in
// fp32 — every rest and accidental precise to the nanosecond. Quantization
// is encoding it to a lo-fi mp3: fewer bits per note, most listeners can't
// hear the difference, and the file is a quarter the size. Scale = tuning
// fork, zero-point = middle C, quantization step = rounding each note to the
// nearest marked key. Returned at: opening (fp32 full score), scale /
// zero-point reveal, "what you can hear vs not hear" section.
export default function QuantizationBasicsLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="quantization-basics" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          A trained model is a symphony score. Every{' '}
          <NeedsBackground slug="mlp-from-scratch">weight</NeedsBackground> is a
          note written out in fp32 — four bytes of painstaking notation, every
          rest and accidental pinned to the nanosecond. It took a small
          orchestra two weeks and a fortune in GPU time to write this score
          down. Now you want to ship it.
        </p>
        <p>
          A 7-billion-parameter full score weighs <code>7 × 10⁹ × 4 = 28 GB</code>{' '}
          before you&apos;ve loaded a single KV cache or activation buffer. It
          doesn&apos;t fit on one GPU. It doesn&apos;t fit on a MacBook. It
          barely fits in a <em>conversation</em> about deployment cost.
        </p>
        <p>
          Here&apos;s the inconvenient question: does the audience actually
          hear the difference between nanosecond-precise notation and
          millisecond-precise notation? The notes you trained so carefully
          almost all sit inside a narrow band — say <code>[-0.5, 0.5]</code> —
          and fp32 is wasting most of its dynamic range on orders of magnitude
          you&apos;ll never touch. What if we snapped each note to the nearest
          of 256 keys on a much smaller keyboard and stored a single{' '}
          <code>float</code> that says how wide each key is? That&apos;s 8 bits
          per note. The file is a quarter the size. Most listeners won&apos;t
          notice.
        </p>
        <p>
          This is <KeyTerm>quantization</KeyTerm>: the JPEG for weights, the
          lo-fi mp3 of neural networks. You start with a pristine fp32 score
          and end with a compressed file that plays almost the same music.
          You&apos;ll learn the quantize/dequantize equations, the precision
          ladder from fp32 down to int4, the difference between symmetric and
          asymmetric, per-tensor and per-channel, and why a few outlier notes
          can single-handedly ruin the recording.
        </p>
      </Prose>

      <Personify speaker="Quantization">
        I am the art of saying &ldquo;close enough.&rdquo; Your 32-bit weights
        pretend to have seven decimal digits of precision — a score marked
        with accidentals no ear on earth can tell apart. I ask: could you get
        by with two? Usually, the answer is yes, and the file just got four
        times smaller.
      </Personify>

      {/* ── Quantize / Dequantize math ──────────────────────────── */}
      <Prose>
        <p>
          Here is the whole idea in two equations, and two new characters. The{' '}
          <KeyTerm>scale</KeyTerm> is the tuning fork: it decides how far apart
          two adjacent keys on our little keyboard are in float space — how
          many floats does one integer step cover? The{' '}
          <KeyTerm>zero-point</KeyTerm> is middle C: the integer value that
          corresponds to the float <code>0.0</code>, the note everything else
          is measured against. Together they define a simple affine map
          between floats and integers — the sheet music for our compressed
          score.
        </p>
      </Prose>

      <MathBlock caption="quantize / dequantize — the affine map">
{`quantize(x)     =   round( x / s )  +  z        ∈  [q_min, q_max]

dequantize(q)   =   s · ( q  −  z )              ∈  ℝ

where    s  =  (x_max − x_min) / (q_max − q_min)        // scale
         z  =  q_min  −  round(x_min / s)               // zero-point`}
      </MathBlock>

      <Prose>
        <p>
          For <code>int8</code> the keyboard has 256 keys, labelled{' '}
          <code>[-128, 127]</code>. If your float range is{' '}
          <code>[-0.5, 0.5]</code>, the scale — the spacing between adjacent
          keys — is about <code>1/255 ≈ 0.00392</code>. Every note gets rounded
          to the nearest marked key. The rounding error is at most{' '}
          <code>s/2</code>: that&apos;s the compression tax, the hiss you pay
          for the smaller file.
        </p>
        <p>
          The widget below lets you feel this. Drag the range, watch the
          continuous float axis get chopped into 256 int8 buckets, and see the{' '}
          <em>quantize then dequantize</em> error curve — a little sawtooth,
          bounded by half a scale step.
        </p>
      </Prose>

      <FloatToInt />

      <Prose>
        <p>
          Two things to notice. First, the error is uniform — no note is more
          than half a key off. Second, the scale determines the fidelity. A
          tight float range gives a tiny scale and low error — a well-tempered
          scale, notes almost indistinguishable from the original. A wide
          float range spreads the same 256 keys over a larger span and the
          error grows. This is why <strong>outliers are so dangerous</strong>:
          one huge weight — a single rogue fortissimo — stretches the range,
          inflates the scale, and now all your <em>normal</em> notes are being
          rounded with too coarse a step.
        </p>
      </Prose>

      <Personify speaker="Scale">
        I am the tuning fork. I pack your floats into a fixed number of
        integer bins. Wider float range, wider bins, coarser rounding. Bring
        me outliers and I&apos;ll ruin every other note to accommodate them.
        Bring me a tight distribution and I&apos;ll compress it almost
        losslessly.
      </Personify>

      <Callout variant="note" title="symmetric vs asymmetric">
        <p>
          <strong>Symmetric</strong> quantization fixes the zero-point at 0 and
          uses a range like <code>[-α, α]</code>. It&apos;s perfect for weights
          — they&apos;re usually zero-centered — and it makes the arithmetic
          cheaper (one multiply, no offset add).
        </p>
        <p>
          <strong>Asymmetric</strong> quantization lets the zero-point be
          nonzero. It&apos;s the right choice for activations after ReLU, which
          are all non-negative and heavily skewed. Forcing a symmetric range
          on <code>[0, 6]</code> would waste half your integer buckets on
          negative values that never occur.
        </p>
      </Callout>

      {/* ── Precision ladder ────────────────────────────────────── */}
      <Prose>
        <p>
          Quantization isn&apos;t a single switch — it&apos;s a ladder of
          audio codecs. At the top, the uncompressed studio master; at the
          bottom, a 4-bit ringtone. Each rung trades precision for memory and
          speed. Here&apos;s the whole ladder for a 7B model, in memory and
          multiply-per-cycle terms:
        </p>
      </Prose>

      <MathBlock caption="the precision ladder — 7B model, weights only">
{`dtype     bits   bytes/weight    7B weights       relative speed (A100)
─────     ────   ────────────    ──────────       ─────────────────────
fp32       32         4             28 GB           1.0 ×   (baseline)
fp16       16         2             14 GB           ≈ 2 ×
bf16       16         2             14 GB           ≈ 2 ×
int8        8         1              7 GB           ≈ 2–4 ×  (tensor cores)
int4        4       0.5            3.5 GB           ≈ 4–8 ×  (with kernels)`}
      </MathBlock>

      <Prose>
        <p>
          <strong>fp16</strong> and <strong>bf16</strong> both use 16 bits but
          split them differently. <code>fp16</code> has a 5-bit exponent and
          10-bit mantissa — good precision, narrow range (overflow at{' '}
          <code>~65504</code>). <code>bf16</code> has an 8-bit exponent and
          7-bit mantissa — same dynamic range as <code>fp32</code>, less
          precision. For training and inference of large models, bf16 is the
          modern default: the extra range prevents the overflow headaches that
          plagued fp16 mixed-precision training circa 2018.
        </p>
        <p>
          <strong>int8</strong> is the workhorse mp3 of deployment. Four
          times smaller than the fp32 master, with &lt; 1% quantization error
          on typical weight distributions. Both NVIDIA tensor cores and modern
          CPU AVX have dedicated int8 math units — you get a speedup on top of
          the memory savings.
        </p>
        <p>
          <strong>int4</strong> is the aggressive end of the compression dial
          — the lo-fi ringtone. Half as big as int8, quantization error in the
          2–5% range. You don&apos;t get here with a naive quantize step — you
          need calibration, clever block-wise scales, or quantization-aware
          training. This is the regime of QLoRA, GGUF <code>Q4_K_M</code>,
          AWQ, and GPTQ.
        </p>
      </Prose>

      <PrecisionLadder />

      <Prose>
        <p>
          Slide through the rungs. This is the &ldquo;what you can hear vs
          what you can&apos;t&rdquo; part of the lesson — the whole economic
          argument for quantization compressed into one slider. At fp16/bf16
          the quality gap vs the fp32 master is barely measurable; the
          audiophile with the $4000 headphones can&apos;t tell. At int8, a
          well-calibrated model loses &lt; 1% accuracy on most benchmarks —
          radio-quality mp3, fine for a commute. At int4 the story depends on
          the codec: naive round-to-nearest sounds like it&apos;s playing
          underwater, but AWQ and GPTQ are the FLAC-of-lossy — they listen to
          which notes matter and spend their bits there. Same file size, far
          cleaner sound.
        </p>
      </Prose>

      <Personify speaker="Zero-point">
        I am middle C — the integer that means &ldquo;zero.&rdquo; Symmetric
        quantization pins me at 0 and calls it a day: the score is centered
        around me and the keys fan out above and below. Asymmetric
        quantization lets me slide — if your data lives in <code>[0, 6]</code>,
        I&apos;ll sit at <code>-128</code> and let the integer range shift to
        match. I am the offset that turns a range into a range-with-direction.
      </Personify>

      <Callout variant="insight" title="per-tensor vs per-channel">
        <p>
          <strong>Per-tensor</strong>: one scale and zero-point for the entire
          weight matrix — one tuning fork for the whole orchestra. Simple,
          fast, and lossy when different output channels have wildly different
          magnitudes.
        </p>
        <p>
          <strong>Per-channel</strong>: a separate scale per output channel
          (one per row of the weight matrix) — each section of the orchestra
          gets its own tuning fork. A tiny fraction more storage, visibly
          better accuracy, and the default in every modern quantization
          library. It&apos;s the reason int8 works in practice at all —
          without it, one loud channel drags down every other.
        </p>
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Time to implement it. We&apos;ll quantize a{' '}
          <NeedsBackground slug="pytorch-basics">tensor</NeedsBackground>{' '}
          three ways — pure Python on a list of floats, NumPy per-channel, and
          PyTorch with the first-class <code>torch.ao.quantization</code>{' '}
          hooks. Same affine map, three progressively more production-shaped
          implementations. Same sheet music, three different recording studios.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · quantize_scratch.py"
        output={`weights        = [-0.42, -0.11, 0.0, 0.23, 0.51]
quantized int8 = [-105, -27, 0, 57, 127]
dequantized    = [-0.4213, -0.1084, 0.0000, 0.2288, 0.5098]
max abs error  = 0.0018`}
      >{`def quantize(x, s, z, q_min=-128, q_max=127):
    q = round(x / s) + z
    return max(q_min, min(q_max, q))          # clamp into the int8 range

def dequantize(q, s, z):
    return s * (q - z)                        # inverse affine map

# Symmetric int8 on a small weight vector.
weights = [-0.42, -0.11, 0.0, 0.23, 0.51]
x_max = max(abs(w) for w in weights)          # 0.51
s = x_max / 127                               # scale
z = 0                                         # symmetric — zero-point pinned

qs = [quantize(w, s, z) for w in weights]
ds = [dequantize(q, s, z) for q in qs]
err = max(abs(w - d) for w, d in zip(weights, ds))

print("weights       =", weights)
print("quantized int8 =", qs)
print("dequantized   =", [round(d, 4) for d in ds])
print("max abs error  =", round(err, 4))`}</CodeBlock>

      <Prose>
        <p>
          Vectorise it. Real models quantize <em>per output channel</em> — one
          scale per row of the weight matrix, one tuning fork per orchestral
          section — which means we need per-channel max, per-channel scale,
          and broadcasting. NumPy makes this five lines.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · quantize_numpy.py"
        output={`W shape            : (4, 8)
per-channel scales : [0.00698 0.00372 0.00516 0.00291]
max abs error      : 0.0035`}
      >{`import numpy as np

def quantize_per_channel(W, bits=8):
    # W: (out_features, in_features) — one scale per row (output channel).
    q_max = 2 ** (bits - 1) - 1                           # 127 for int8
    x_max = np.max(np.abs(W), axis=1, keepdims=True)      # (out, 1)
    s = x_max / q_max                                     # scale per row
    q = np.round(W / s).clip(-q_max - 1, q_max).astype(np.int8)
    return q, s

def dequantize_per_channel(q, s):
    return s * q.astype(np.float32)                       # broadcasts back

rng = np.random.default_rng(0)
W = rng.standard_normal((4, 8)).astype(np.float32) * 0.3   # small, zero-centered

q, s = quantize_per_channel(W)
W_hat = dequantize_per_channel(q, s)

print("W shape            :", W.shape)
print("per-channel scales :", np.round(s.ravel(), 5))
print("max abs error      :", round(np.max(np.abs(W - W_hat)), 4))`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for w in weights: quantize(w, s, z)',
            right: 'np.round(W / s).clip(-128, 127).astype(int8)',
            note: 'vectorised — one scale per row via broadcasting',
          },
          {
            left: 'x_max = max(abs(w) for w in weights)',
            right: 'np.max(np.abs(W), axis=1, keepdims=True)',
            note: 'per-channel max along the in-features axis',
          },
          {
            left: 'single scalar scale',
            right: 's.shape == (out_features, 1)',
            note: 'per-channel: one scale per output row',
          },
        ]}
      />

      <Prose>
        <p>
          PyTorch ships the whole pipeline behind <code>torch.ao.quantization</code>.
          You get <code>QuantStub</code> / <code>DeQuantStub</code> modules, observer
          objects that collect min/max during calibration, and a{' '}
          <code>convert()</code> call that swaps every float module for its int8 twin.
          This is what production code looks like — you rarely roll your own.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · quantize_pytorch.py"
        output={`fp32 params : 101,770
int8 params :  25,498   (weights packed int8, biases fp32)
size ratio  : ~4× smaller
max |Δy|    : 0.0041   (output drift on one test batch)`}
      >{`import torch
import torch.nn as nn
import torch.ao.quantization as tq

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant   = tq.QuantStub()            # float → int8 at the boundary
        self.fc1     = nn.Linear(128, 256)
        self.act     = nn.ReLU()
        self.fc2     = nn.Linear(256, 10)
        self.dequant = tq.DeQuantStub()          # int8 → float on the way out

    def forward(self, x):
        x = self.quant(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.dequant(x)

# 1) Train or load an fp32 model.
model_fp32 = MLP().eval()

# 2) Attach a per-channel int8 config and insert observers.
model_fp32.qconfig = tq.get_default_qconfig("fbgemm")     # per-channel weights, per-tensor acts
tq.prepare(model_fp32, inplace=True)

# 3) Calibrate: run a handful of real batches so observers learn the activation ranges.
with torch.no_grad():
    for _ in range(32):
        model_fp32(torch.randn(64, 128))

# 4) Convert: swap every float module for its int8 equivalent.
model_int8 = tq.convert(model_fp32, inplace=False)

# Compare outputs on a fresh batch.
x = torch.randn(16, 128)
with torch.no_grad():
    y_fp  = MLP().eval()(x)                    # a freshly-init fp32 model for comparison
    y_int = model_int8(x)
print("max |Δy|   :", float((y_fp - y_int).abs().max()))`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'manual np.max(abs(W)) calibration',
            right: 'tq.prepare() inserts observers',
            note: 'PyTorch tracks min/max automatically during a calibration pass',
          },
          {
            left: 'W_q, s = quantize_per_channel(W)',
            right: 'tq.convert() swaps modules in-place',
            note: 'one call replaces every nn.Linear with its int8 twin',
          },
          {
            left: 'dequant = s * q',
            right: 'DeQuantStub at model boundary',
            note: 'internal ops stay int8; only inputs/outputs round-trip to float',
          },
        ]}
      />

      <Callout variant="insight" title="PTQ vs QAT">
        <p>
          <strong>Post-training quantization (PTQ)</strong> is what the PyTorch snippet
          does: take a trained model, calibrate on a few hundred examples, quantize. Fast,
          cheap, usually good enough down to int8. A mastering engineer encoding the
          finished record to mp3 after the fact.
        </p>
        <p>
          <strong>Quantization-aware training (QAT)</strong> simulates the rounding
          during training by inserting <em>fake-quant</em> ops into the forward pass.
          Gradients flow through the rounding (via a straight-through estimator) and the
          weights <em>learn</em> to be quantization-friendly — the orchestra rehearses
          already knowing the final recording will be lo-fi, and phrases the notes to
          survive compression. Strictly more expensive, but the standard approach when
          you need int4 or want to keep every last point of accuracy.
        </p>
      </Callout>

      <Callout variant="note" title="GGUF, AWQ, GPTQ — the int4 zoo">
        <p>
          When you download a &ldquo;<code>Q4_K_M</code>&rdquo; or &ldquo;<code>4bit-awq</code>&rdquo;
          model from Hugging Face, you&apos;re picking an int4 quantization <em>strategy</em>,
          not just a bit width. <strong>GGUF</strong> is llama.cpp&apos;s block-wise format
          for CPU/metal inference — groups of 32 weights share a scale.{' '}
          <strong>AWQ</strong> (activation-aware weight quantization) keeps the 1% of
          weights that get hit by large activations in higher precision.{' '}
          <strong>GPTQ</strong> uses second-order error correction column-by-column to
          minimise reconstruction error. All three preserve accuracy much better than naive
          round-to-nearest at 4 bits.
        </p>
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Outliers dominate the scale.</strong> One
          weight at <code>5.0</code> in a distribution that&apos;s otherwise in{' '}
          <code>[-0.5, 0.5]</code> stretches the scale by 10×. Every other note now has
          10× the quantization error — one fortissimo has forced the whole recording into
          a coarser mp3. Fix: clip outliers before quantizing, or use per-channel scales
          so the outlier only ruins one channel.
        </p>
        <p>
          <strong className="text-term-amber">Clamping range mismatched to the data.</strong>{' '}
          If your observer saw activations in <code>[0, 6]</code> during calibration and
          production hits <code>[0, 20]</code>, everything above 6 saturates. Calibrate on
          data that looks like production.
        </p>
        <p>
          <strong className="text-term-amber">Accumulate in int32, not int8.</strong> A
          matmul is a sum of products. Two int8s multiply into int16, and summing 1024 of
          them overflows int8 catastrophically. Every real int8 kernel accumulates into
          int32, then requantizes the result. If you hand-roll this and forget, your output
          will be nonsense.
        </p>
        <p>
          <strong className="text-term-amber">Symmetric quantization on activations.</strong>{' '}
          ReLU outputs are all ≥ 0. Using a symmetric range wastes half your integer codes
          on negative values that never appear. Use asymmetric for activations, symmetric
          for weights.
        </p>
      </Gotcha>

      {/* ── Challenge ──────────────────────────────────────────── */}
      <Challenge prompt="Quantize an MLP and measure the damage">
        <p>
          Train a small MLP on MNIST in PyTorch — two hidden layers of 256, one fp32 epoch,
          any optimiser you like. Record the test accuracy; call it <code>acc_fp32</code>.
          That&apos;s the studio master.
        </p>
        <p className="mt-2">
          Now quantize the trained model to int8 <em>per-channel</em> using{' '}
          <code>torch.ao.quantization</code> with a calibration pass of 512 training
          examples. Re-measure accuracy on the test set; call it <code>acc_int8</code>.
          That&apos;s the mp3. On MNIST you should see a drop of well under 0.5 percentage
          points — the audience can&apos;t tell.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: swap per-channel for per-tensor and re-run. The accuracy hit is usually
          small on MNIST but noticeably larger — you&apos;ve just felt, numerically, why
          the default in every modern library is per-channel.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Extra bonus: measure model file size with <code>torch.save()</code> on both.
          Expect the int8 state dict to be roughly four times smaller, which is the entire
          economic argument for quantization in one line of <code>ls -lh</code>.
        </p>
      </Challenge>

      {/* ── Close + teaser ─────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Quantization is the JPEG for
          weights — an affine map from floats to integers, governed by a scale
          (the tuning fork) and a zero-point (middle C for the integer
          keyboard). fp16/bf16 is nearly free; int8 with per-channel scales is
          the production sweet spot, the 192-kbps mp3 of neural nets; int4
          needs smarter codecs (AWQ / GPTQ / GGUF) to preserve accuracy.
          Outliers dominate the scale, so real systems clip, per-channel, or
          calibrate away from the tails.
        </p>
        <p>
          <strong>Next up — INT8 &amp; INT4 Quantization.</strong> We sketched
          the ladder; next lesson we climb it, going ever lower. The exact math
          of int8 GEMMs (int32 accumulators, the requantize step), the
          specific tricks AWQ and GPTQ use to survive at 4 bits, and a working
          pipeline you can run on a downloaded Llama-3 checkpoint on your
          laptop. Same affine map, much sharper teeth — the lo-fi mp3 taken
          down to a ringtone that still sounds like music.
        </p>
      </Prose>

      <References
        items={[
          {
            title:
              'Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper',
            author: 'Krishnamoorthi',
            year: 2018,
            url: 'https://arxiv.org/abs/1806.08342',
          },
          {
            title:
              'Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference',
            author: 'Jacob, Kligys, Chen, Zhu, Tang, Howard, Adam, Kalenichenko',
            venue: 'CVPR 2018',
            url: 'https://arxiv.org/abs/1712.05877',
          },
          {
            title: 'LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale',
            author: 'Dettmers, Lewis, Belkada, Zettlemoyer',
            year: 2022,
            venue: 'bitsandbytes — arxiv 2208.07339',
            url: 'https://arxiv.org/abs/2208.07339',
          },
          {
            title: 'PyTorch Quantization — official docs',
            venue: 'torch.ao.quantization',
            url: 'https://pytorch.org/docs/stable/quantization.html',
          },
          {
            title: 'AWQ: Activation-aware Weight Quantization for LLM Compression',
            author: 'Lin, Tang, Tang, Yang, Chen, Wang, Wang, Xiao, Dang, Han',
            year: 2023,
            url: 'https://arxiv.org/abs/2306.00978',
          },
          {
            title: 'GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers',
            author: 'Frantar, Ashkboos, Hoefler, Alistarh',
            year: 2022,
            url: 'https://arxiv.org/abs/2210.17323',
          },
        ]}
      />
    </div>
  )
}
