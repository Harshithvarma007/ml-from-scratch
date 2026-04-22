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
import PoolingViz from '../widgets/PoolingViz'
import DownsamplingChain from '../widgets/DownsamplingChain'

export default function PoolingLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout ────────────────────────────────── */}
      <Prereq currentSlug="pooling" />

      {/* ── Opening: feature maps are bigger than they need to be ─ */}
      <Prose>
        <p>
          A{' '}
          <NeedsBackground slug="convolution-operation">
            convolution
          </NeedsBackground>{' '}
          at full resolution is expensive — and most of what it produces is
          redundant. A 224×224 image through one 64-channel conv is{' '}
          <code>224 · 224 · 64 ≈ 3.2M</code> activations. Stack six of those
          and you&apos;re hauling around twenty million floats per forward
          pass, most of them encoding minor variations of &ldquo;this same
          edge, shifted over by one pixel.&rdquo; The network is doing the
          work of a detail-obsessed intern: faithful, exhaustive, and mostly
          wasted.
        </p>
        <p>
          Here&apos;s the anchor for the whole lesson:{' '}
          <strong>the thumbnail — keep the gist, drop the pixels.</strong>{' '}
          When your phone generates a thumbnail of a photo, it keeps the
          subject (&ldquo;there&apos;s a face, roughly here&rdquo;) and
          throws away the pixel-level particulars (&ldquo;the exact position
          of the left eye&rdquo;). That&apos;s precisely what you want to do
          to a feature map once the conv has detected its features. You
          don&apos;t need the edge at pixel (37, 42). You need to know the
          edge fired <em>in this general neighborhood</em>.
        </p>
        <p>
          That&apos;s <KeyTerm>pooling</KeyTerm>. Slide a 2×2 window with
          stride 2, collapse each window to a single number, and you&apos;ve
          halved both spatial dimensions — quartered the activation count —
          for the cost of a comparison. No weights. Nothing to learn. Just a
          shrink.
        </p>
      </Prose>

      <Personify speaker="Pooling">
        I have no weights. I do not learn. I look at a neighborhood, summarize
        it in one number, and throw the rest away. Your downstream layers
        thank me: they see the same features at a quarter the cost and twice
        the effective reach.
      </Personify>

      {/* ── Shape math ──────────────────────────────────────────── */}
      <Prose>
        <p>
          The output shape follows the same arithmetic you already memorized
          for convolutions. Window size <code>k</code>, stride <code>s</code>,
          no padding:
        </p>
      </Prose>

      <MathBlock caption="output shape — same arithmetic as conv, no padding">
{`H_out  =  ⌊ (H_in − k) / s ⌋ + 1
W_out  =  ⌊ (W_in − k) / s ⌋ + 1

typical:   k = 2,  s = 2
           →  H_out = H_in / 2,  W_out = W_in / 2
           channels unchanged`}
      </MathBlock>

      <Prose>
        <p>
          Two things trip people up. <strong>Channels don&apos;t change.</strong>{' '}
          Pooling runs per-channel — a 32-channel feature map goes in, 32
          channels come out, just smaller in H and W. Each channel gets its
          own independent thumbnail. And <strong>stride equals kernel size</strong>{' '}
          in the textbook 2×2 pool: windows tile the input edge to edge, no
          overlap. Overlapping pools exist (AlexNet used 3×3 kernel with
          stride 2), but the modern default is clean tiling.
        </p>
      </Prose>

      {/* ── Widget 1: PoolingViz — the reveal ───────────────────── */}
      <Prose>
        <p>
          Two flavors of thumbnail. Flip between them in the widget below and
          watch how differently they summarize the same input.
        </p>
      </Prose>

      <PoolingViz />

      <Prose>
        <p>
          Max pool is the ruthless thumbnail. Each 2×2 window forwards exactly
          one value — the loudest — and discards the other three without
          ceremony. If a neuron anywhere in that window was screaming about a
          vertical edge, max pool forwards the scream at full volume. This is
          the right instinct mid-network, where a conv filter&apos;s whole
          personality is &ldquo;I fire hard when I see my feature and stay
          quiet otherwise&rdquo; — you want to preserve the fire.
        </p>
        <p>
          Average pool is the soft thumbnail. It smooths. Three zeros and a
          nine become a 2.25, not a 9. Fine if what you want is a summary;
          actively wrong if the signal you care about is precisely &ldquo;the
          loudest value in this neighborhood.&rdquo; You don&apos;t average a
          smoke alarm with three silent rooms and call that a fire detector.
        </p>
      </Prose>

      <Callout variant="note" title="why mid-network prefers max">
        Early and mid CNN layers are detectors. A filter tuned to a
        horizontal edge fires loudly where the edge is and quietly elsewhere.
        Max pool preserves &ldquo;the edge fired somewhere in this
        region&rdquo; at full strength. Averaging would water it down with
        three near-zero neighbors. That&apos;s why LeNet, VGG, and early
        ResNet all reach for max between conv blocks.
      </Callout>

      <Personify speaker="Max pool">
        Winner-take-all. Four values in, one value out — the loudest. If your
        convolution detected an edge anywhere in my window, I preserve the
        detection at full strength. I lose position information within the
        window, which is exactly the point: downstream layers need to know
        the feature is <em>here-ish</em>, not that it sat at pixel (5, 7).
      </Personify>

      {/* ── The three jobs at once — receptive field / Widget 2 ─── */}
      <Prose>
        <p>
          Here&apos;s why pooling earns its keep. That one shrink does{' '}
          <strong>three jobs at the same time.</strong>
        </p>
        <p>
          <strong>Job one: cost.</strong> Spatial dims halve, activations
          quarter. Every downstream conv runs on a smaller grid. On a deep
          network this compounds into the difference between &ldquo;trainable
          on a single GPU&rdquo; and &ldquo;trainable on a cluster your lab
          doesn&apos;t own.&rdquo;
        </p>
        <p>
          <strong>Job two: translation invariance.</strong> Nudge the input
          over by one pixel. The max over a 2×2 window almost certainly picks
          the same value. The output barely moves. That&apos;s not a
          philosophical claim about invariance in the deep sense — it&apos;s
          a mechanical fact about what happens when you take a max over a
          small region. The thumbnail doesn&apos;t care about subpixel
          jitter, and neither does the next layer up.
        </p>
        <p>
          <strong>Job three: receptive field.</strong> This one is quieter
          and more important. Each pool doubles the effective{' '}
          <KeyTerm>receptive field</KeyTerm> of every downstream neuron. A
          3×3 conv sees a 3×3 patch of its input. Stack two 3×3 convs and
          the top one sees 5×5 of the original. Slip a pool between them and
          the top conv now sees a 6×6 region — because each of its inputs was
          already a summary over a 2×2 patch. Four (conv, pool) blocks and
          the final layer is reasoning over a ~30×30 chunk of the original
          image while computing on a ~14×14 grid.
        </p>
        <p>
          That&apos;s the whole game. You want a neuron deep in the network
          to see enough of the image to recognize a dog&apos;s face. You can
          get there with (a) huge kernels early — expensive and clumsy — or
          (b) small kernels with repeated shrinks. CNN architecture is a
          long, committed bet on option (b).
        </p>
      </Prose>

      <DownsamplingChain />

      <Prose>
        <p>
          Notice the inverted pyramid. Spatial dims halve at every pool.
          Channels grow the other way — <code>32 → 64 → 128 → 256</code> —
          because as each neuron sees more of the input, it takes more
          channels to describe all the things that bigger chunk might contain.
          Shallow layers recognize edges. Deep layers recognize object parts.
          A face-detector genuinely needs more channels than an edge-detector.
          Total activations per layer stay roughly flat (H·W shrinks 4×, C
          doubles) while the semantic density climbs.
        </p>
      </Prose>

      <Callout variant="insight" title="pooling is one of three downsampling moves">
        Three ways to cut spatial resolution: (1) conv + pool, (2) strided
        conv with no pool — let the network <em>learn</em> the downsample,
        (3) pool alone. Modern architectures — ResNet v2, most
        transformers&apos; patch embeddings — often skip pooling entirely and
        use strided convolutions. The cost is a few more learned parameters;
        the benefit is end-to-end learnable downsampling. Pool is the old
        standby, still everywhere, but not the only move.
      </Callout>

      {/* ── Global average pooling ──────────────────────────────── */}
      <Prose>
        <p>
          One more thumbnail, turned up to eleven:{' '}
          <KeyTerm>global average pooling</KeyTerm> (GAP). Instead of a 2×2
          window, the window is the <em>entire feature map</em>. A{' '}
          <code>7×7×512</code> tensor goes in, a <code>1×1×512</code> vector
          comes out — each channel squeezed to a single number, its mean over
          every spatial position. It&apos;s the softest possible summary: one
          number per channel for the whole image.
        </p>
        <p>
          Before 2013, the standard classifier tail was{' '}
          <code>conv → flatten → dense(4096) → dense(num_classes)</code>. That
          flatten-then-dense combo is enormous — a <code>7·7·512 = 25,088</code>
          -dim flatten feeding a dense layer gives you{' '}
          <code>25,088 · 4096 ≈ 100M</code> parameters in a single layer. Most
          of AlexNet&apos;s weights lived there. Worse, it&apos;s brittle:
          flatten assumes the feature map is <em>exactly</em> 7×7. Change
          input size, rebuild the classifier.
        </p>
        <p>
          GAP replaces the whole thing with{' '}
          <code>global_avg_pool → dense(num_classes)</code>. No flatten, no
          25K-dim vector, zero parameters in the pool, and the network
          suddenly works on any input size — because the pool collapses
          whatever H×W you hand it to 1×1. That&apos;s why every serious
          classifier from GoogLeNet onward ends with a GAP before the head.
        </p>
      </Prose>

      <Personify speaker="Global avg pool">
        I am the summary. Give me a feature map and I return one number per
        channel — the channel&apos;s average response over the whole image. I
        erase the 100-million-parameter dense layers your grandparents&apos;
        networks used, and I refuse to overfit because I have no weights to
        overfit with. Every good classification head ends with me.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Time to write it. Max pool with a 2×2 window, stride 2, three ways:
          nested Python loops, a NumPy reshape trick, the one-line PyTorch
          call. All three produce identical numbers on the same input.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · pooling_scratch.py"
        output={`input (4×4):
  [[ 1  3  2  1]
   [ 4  2  1  0]
   [ 5  6  1  2]
   [ 7  8  3  4]]
max-pool 2×2 stride 2 (2×2):
  [[4, 3],
   [8, 4]]`}
      >{`def max_pool_2x2(x):
    """x is a 2D list (H, W). Returns H/2 × W/2."""
    H, W = len(x), len(x[0])
    out = [[0] * (W // 2) for _ in range(H // 2)]
    for i in range(H // 2):
        for j in range(W // 2):
            window = [
                x[2*i    ][2*j], x[2*i    ][2*j + 1],
                x[2*i + 1][2*j], x[2*i + 1][2*j + 1],
            ]
            out[i][j] = max(window)            # the whole operation: one max()
    return out

x = [[1, 3, 2, 1],
     [4, 2, 1, 0],
     [5, 6, 1, 2],
     [7, 8, 3, 4]]

for row in max_pool_2x2(x):
    print(row)`}</CodeBlock>

      <Prose>
        <p>
          Two nested loops, one <code>max()</code> call. That&apos;s the
          entire algorithm — every line you&apos;ve read so far was
          motivation. The pure-Python version is plenty for a 4×4 grid; on a
          real feature map with channels and batch, we vectorize.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · pooling_numpy.py"
        output={`max:  [[4 3]
 [8 4]]
avg:  [[2.5 1. ]
 [6.5 2.5]]`}
      >{`import numpy as np

def pool2d(x, mode="max"):
    """x is (H, W) with H, W divisible by 2. Returns H/2 × W/2."""
    H, W = x.shape
    # Key trick: reshape into (H/2, 2, W/2, 2) — the 2s are the within-window axes.
    tiles = x.reshape(H // 2, 2, W // 2, 2)
    if mode == "max":
        return tiles.max(axis=(1, 3))          # collapse the two window axes
    else:
        return tiles.mean(axis=(1, 3))

x = np.array([[1, 3, 2, 1],
              [4, 2, 1, 0],
              [5, 6, 1, 2],
              [7, 8, 3, 4]])

print("max: ", pool2d(x, mode="max"))
print("avg: ", pool2d(x, mode="avg"))`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for i in H/2: for j in W/2: …',
            right: 'x.reshape(H/2, 2, W/2, 2)',
            note: 'the 2s are the within-window axes — reduce along them',
          },
          {
            left: 'max(window)',
            right: 'tiles.max(axis=(1, 3))',
            note: 'broadcasted max, runs as a single kernel',
          },
          {
            left: 'change max() to sum()/4 for avg-pool',
            right: 'tiles.mean(axis=(1, 3))',
            note: 'one function, mode flag — same shape math',
          },
        ]}
      />

      <Prose>
        <p>
          PyTorch ships all of this. You will not hand-roll a pool in
          production — you <em>will</em> have to pick between{' '}
          <code>MaxPool2d</code>, <code>AvgPool2d</code>, and the adaptive
          (global) variants, which is why it&apos;s worth knowing all three.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · pooling_pytorch.py"
        output={`input shape:          torch.Size([1, 32, 16, 16])
after maxpool 2x2:    torch.Size([1, 32, 8, 8])
after avgpool 2x2:    torch.Size([1, 32, 8, 8])
after global avgpool: torch.Size([1, 32, 1, 1])
GAP output (flat):    torch.Size([1, 32])`}
      >{`import torch
import torch.nn as nn

x = torch.randn(1, 32, 16, 16)                  # (batch, channels, H, W)

max_pool   = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool   = nn.AvgPool2d(kernel_size=2, stride=2)
gap        = nn.AdaptiveAvgPool2d(output_size=1)   # reduces any H×W to 1×1

print("input shape:         ", x.shape)
print("after maxpool 2x2:   ", max_pool(x).shape)
print("after avgpool 2x2:   ", avg_pool(x).shape)

gap_out = gap(x)
print("after global avgpool:", gap_out.shape)
print("GAP output (flat):   ", gap_out.flatten(1).shape)   # ready for nn.Linear

# Note: pooling has no parameters.
print("params in maxpool:", sum(p.numel() for p in max_pool.parameters()))`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'x.reshape(...).max(axis=(1, 3))',
            right: 'nn.MaxPool2d(2, 2)(x)',
            note: 'handles any kernel/stride, runs on GPU, no shape gymnastics',
          },
          {
            left: 'pool2d(x, mode="avg")',
            right: 'nn.AvgPool2d(2, 2)(x)',
            note: 'same call signature — mode becomes the class name',
          },
          {
            left: 'tiles.mean(axis=(1, 2, 3))  # whole map',
            right: 'nn.AdaptiveAvgPool2d(1)(x)',
            note: 'GAP — collapses any H×W to 1×1, size-agnostic',
          },
        ]}
      />

      <Callout variant="insight" title="the weightless layer">
        Pooling has <strong>zero learned parameters</strong>. Sum{' '}
        <code>.parameters()</code> on any pooling module and you get 0.
        That&apos;s why it doesn&apos;t show up in parameter counts, even
        though it dominates the compute budget of a classic CNN. Pool is
        architecture, not learning.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Stride ≠ kernel size:</strong>{' '}
          the familiar 2×2 pool has{' '}
          <code className="text-dark-text-primary">kernel_size = stride = 2</code>
          , but those are two independent knobs. A 3×3 kernel with stride 2
          (AlexNet-style overlapping pool) is a different animal — the
          windows overlap and H_out arithmetic changes. Write it out before
          you ship it.
        </p>
        <p>
          <strong className="text-term-amber">Pool has no parameters:</strong>{' '}
          if an architecture diagram lists pool layers in the param count,
          someone drew it wrong. You can insert or remove pool layers without
          touching the weight file — only the activation shapes move. Handy
          when converting a classifier to a fully-convolutional segmentation
          net.
        </p>
        <p>
          <strong className="text-term-amber">Avg-pool mid-network:</strong>{' '}
          smooths away the sharp responses your conv layers worked hard to
          produce. Unless you have a specific reason (some segmentation nets,
          some low-level image processing), mid-network pools should be max.
          Save avg for the classifier tail (GAP).
        </p>
        <p>
          <strong className="text-term-amber">Odd input dims:</strong> a 2×2
          stride-2 pool on a 7×7 map produces a 3×3 output — the last row
          and column vanish silently. Either pad, use{' '}
          <code className="text-dark-text-primary">AdaptiveAvgPool2d</code>,
          or make your conv strides keep the spatial dims divisible by your
          pool stride.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Build a feature pyramid">
        <p>
          Stack four <code>(Conv2d 3×3, ReLU, MaxPool2d 2×2)</code> blocks.
          Start with an input of shape <code>(1, 3, 64, 64)</code> and channel
          widths <code>3 → 32 → 64 → 128 → 256</code>. Print the output shape
          after each block.
        </p>
        <p className="mt-2">
          You should see spatial dims halve at every block: 64 → 32 → 16 → 8
          → 4, while channels grow: 3 → 32 → 64 → 128 → 256. That&apos;s the
          canonical feature-pyramid geometry.
        </p>
        <p className="mt-2">
          Now compute the <strong>receptive field</strong> at each depth. A
          3×3 conv adds 2. A 2×2 stride-2 pool multiplies by 2 (and adds 1
          for the window). Work it out by hand, then verify: at depth 4, each
          output cell should correspond to a ~30×30 patch of the input.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: replace every <code>MaxPool2d(2, 2)</code> with{' '}
          <code>Conv2d(stride=2)</code> (strided conv). Measure the parameter
          count difference and the output shape. This is the modern move that
          largely replaced pool in ResNet v2 and beyond.
        </p>
      </Challenge>

      {/* ── Closing + cliffhanger → build-a-cnn ─────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Pooling is the thumbnail
          step: keep the gist, drop the pixels. Max pool preserves sharp
          feature responses in the middle of a network; global average pool
          replaces the flatten-plus-dense monster at the classifier head and
          shaves millions of parameters. The same shrink buys you three
          things at once — cheaper compute, a little translation invariance,
          and a doubled receptive field downstream — which is why a
          zero-parameter layer has survived three architectural eras. The
          operation has no gradient of its own to learn: max pool routes the
          upstream gradient to whichever input cell was the winner, avg pool
          splits it evenly across the window. Nothing to train.
        </p>
        <p>
          <strong>Next up — Build a CNN.</strong> You now have every part:
          convolutions that stamp features onto a map, activations that
          keep the network non-linear, and pools that summarize. The next
          lesson bolts them together into a working image classifier — the
          LeNet-to-VGG lineage in roughly forty lines — and ends with a
          fully-connected head that turns the final thumbnail into class
          probabilities. Stamps plus thumbnails plus a fully-connected head,
          trained end-to-end. A real conv net, finally assembled.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Network In Network',
            author: 'Lin, Chen, Yan',
            venue: 'ICLR 2014 — popularized global average pooling',
            url: 'https://arxiv.org/abs/1312.4400',
          },
          {
            title: 'Striving for Simplicity: The All Convolutional Net',
            author: 'Springenberg, Dosovitskiy, Brox, Riedmiller',
            venue: 'ICLR Workshop 2015 — strided conv as pooling replacement',
            url: 'https://arxiv.org/abs/1412.6806',
          },
          {
            title: 'Dive into Deep Learning — 7.5: Pooling',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai — canonical textbook treatment',
            url: 'https://d2l.ai/chapter_convolutional-neural-networks/pooling.html',
          },
          {
            title: 'Gradient-Based Learning Applied to Document Recognition',
            author: 'LeCun, Bottou, Bengio, Haffner',
            venue: 'Proc. IEEE 1998 — the LeNet paper, original subsampling',
            url: 'http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf',
          },
        ]}
      />
    </div>
  )
}
