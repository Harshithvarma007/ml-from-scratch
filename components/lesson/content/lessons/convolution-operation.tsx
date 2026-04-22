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
import ConvKernelSlide from '../widgets/ConvKernelSlide'
import FilterGallery from '../widgets/FilterGallery'

// A single flowing narrative, anchored on one image: the rubber stamp
// sliding across the page. A kernel is a tiny stamp; at every position
// you press it, multiply the pixels under it by the stamp's weights,
// sum, and leave one ink blot on a new page. That anchor returns at
// the mechanism reveal and at the three-knobs section, so the reader
// never loses the picture while the shapes get technical.
export default function ConvolutionOperationLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout ─────────────────────────────────── */}
      <Prereq currentSlug="convolution-operation" />

      {/* ── Opening: MLP has no sense of locality ───────────────── */}
      <Prose>
        <p>
          Your{' '}
          <NeedsBackground slug="digit-classifier">MNIST classifier</NeedsBackground>{' '}
          did something quietly violent on the way in. The first line of the
          forward pass was <code>x.flatten()</code>. Seven hundred eighty-four
          pixels, poured into a vector, every neighbor relationship the image
          had going for it: gone. Pixel <code>(3, 4)</code> and pixel{' '}
          <code>(3, 5)</code> were one step apart on the grid. In the flat
          vector they&apos;re index 88 and index 89 — which is the same
          distance as index 0 and index 1, which is the same distance as index
          0 and index 783.
        </p>
        <p>
          The dense layer has no way to know any of that. It sees 784
          unrelated numbers. If two pixels belong together, the network has to
          re-discover it from supervision signal alone, and it has to
          re-discover it <em>separately for every location in the image</em>{' '}
          because the weights that look at the top-left corner have nothing to
          do with the weights that look at the bottom-right.
        </p>
        <p>
          MNIST forgives you. The digits are tiny, centered, and there are
          60,000 of them to brute-force the problem away. Swap in a 224×224
          RGB ImageNet crop — 150,528 inputs, tens of millions of weights in
          the first layer alone — and the whole setup falls over. An{' '}
          <NeedsBackground slug="mlp-from-scratch">MLP</NeedsBackground> is
          asking supervision signal to teach it what a neighbor is. It&apos;s
          asking too much.
        </p>
        <p>
          The fix is simple and a little insulting. Stop throwing away the
          grid. Look at small patches at a time. Use the same weights for
          every patch. That&apos;s <KeyTerm>convolution</KeyTerm>, and the
          entire reason computer vision works is that we stopped pretending
          an image was a bag of numbers.
        </p>
      </Prose>

      <Personify speaker="Convolution">
        I&apos;m a rubber stamp. A small square of numbers. You press me at
        one corner of your image, ink a single number onto a new page, slide
        me one pixel, press again. Same stamp, every press. By the time
        I&apos;ve walked the whole image, I&apos;ve drawn a new picture — an
        ink map of wherever my pattern fired. That&apos;s the whole job.
      </Personify>

      {/* ── The mechanism: stamp, slide, sum, ink ───────────────── */}
      <Prose>
        <p>
          Let&apos;s make the stamp literal. Given an input of shape{' '}
          <code>(H, W)</code> and a kernel of shape <code>(k, k)</code> — the
          stamp — the output at position <code>(i, j)</code> is the
          element-wise product of the stamp with the <code>k×k</code> patch of
          image directly under it, summed into one scalar. Press, read the
          blot, slide one pixel, press again. Repeat until you&apos;ve covered
          the page.
        </p>
      </Prose>

      <MathBlock caption="2D convolution — one output pixel">
{`y[i, j]  =  Σ   Σ   x[i + u, j + v]  ·  k[u, v]
           u=0 v=0

                with  u, v  ∈  [0, k-1]`}
      </MathBlock>

      <Prose>
        <p>
          Two knobs decide the shape of the page you end up with: the{' '}
          <KeyTerm>stride</KeyTerm> <code>s</code> — how far the stamp jumps
          between presses — and the <KeyTerm>padding</KeyTerm> <code>p</code>{' '}
          — how many rows and columns of zeros you paste around the border so
          the stamp can hang off the edge without falling off. Plug them in
          and the output shape drops out:
        </p>
      </Prose>

      <MathBlock caption="the shape arithmetic you will do a thousand times">
{`H'  =  ⌊ (H + 2p − k) / s ⌋ + 1
W'  =  ⌊ (W + 2p − k) / s ⌋ + 1

e.g.  H=28, k=3, s=1, p=1   →   H' = (28 + 2 − 3)/1 + 1 = 28      "same" padding
      H=28, k=3, s=1, p=0   →   H' = (28 + 0 − 3)/1 + 1 = 26      "valid" padding
      H=28, k=3, s=2, p=1   →   H' = ⌊(28 + 2 − 3)/2⌋ + 1 = 14    stride-2 downsample`}
      </MathBlock>

      <Prose>
        <p>
          Memorize the formula. You&apos;ll type it into a mental calculator
          every time you design a conv net, every time you debug a shape
          mismatch, and every time a PyTorch stack trace complains it was
          expecting <code>(B, 64, 14, 14)</code> and you handed it{' '}
          <code>(B, 64, 13, 13)</code>. Convolutions are unforgiving about
          one pixel.
        </p>
      </Prose>

      {/* ── Widget 1: Kernel slide ──────────────────────────────── */}
      <Prose>
        <p>
          Watch the stamp do its thing. The 3×3 grid on the left is the
          stamp. It parks at one corner of the input, the nine numbers
          underneath it get multiplied by the nine stamp weights, the products
          get summed, and that single number is inked into the output on the
          right. Then the stamp shifts one pixel and presses again.
        </p>
      </Prose>

      <ConvKernelSlide />

      <Prose>
        <p>
          Two things to notice. First, the output page is smaller than the
          input — a 5×5 input with a 3×3 stamp and no padding leaves a 3×3
          output. That&apos;s the shape formula above enforcing itself, and
          it&apos;s why people reach for padding the instant they want to
          stack more than three or four conv layers without their feature map
          shrinking to a single dot. Second, the same nine numbers were used
          at every press. You aren&apos;t training 9 × 9 = 81 weights. You
          are training exactly 9. Forever. That&apos;s the trick the whole
          field runs on.
        </p>
      </Prose>

      <Personify speaker="Kernel">
        I am nine numbers. That&apos;s the whole me. I do not care how big
        your image is — 28 pixels, 1024 pixels, a billboard — I walk across
        all of it with the same weights. A fully-connected layer on a
        megapixel image wants a million weights per output neuron. I want
        nine. That is my entire pitch.
      </Personify>

      {/* ── Why CNN beats dense ─────────────────────────────────── */}
      <Prose>
        <p>
          That last observation is one of three properties that together
          explain why convolutional nets annihilated dense nets on images.
        </p>
        <ul>
          <li>
            <strong>Parameter sharing.</strong> One stamp, pressed at every
            position. A 3×3 stamp on a 224×224 image has 9 weights and gets
            reused 224² ≈ 50,000 times. A dense layer with the same output
            size would need 224² × 224² ≈ 2.5 billion weights for a single
            layer. That&apos;s a ~300 million times saving. Not a
            micro-optimization — the difference between possible and not.
          </li>
          <li>
            <strong>Translation equivariance.</strong> If a feature — an
            edge, a corner, a nose — shows up top-left in one image and
            bottom-right in another, the same stamp fires in both. Train the
            detector once, it works everywhere. A dense net would have to
            re-learn &ldquo;nose&rdquo; at every location independently,
            which is as absurd as it sounds.
          </li>
          <li>
            <strong>Local connectivity.</strong> Each output pixel only reads
            a small <code>k×k</code> window of the input. Early layers see a
            small neighborhood; stacked layers progressively see larger
            regions (the <KeyTerm>receptive field</KeyTerm>). This matches
            how pixel-level structure actually behaves — you need a few
            neighbors to decide &ldquo;edge&rdquo;, not the entire image.
          </li>
        </ul>
      </Prose>

      <Callout variant="insight" title="three properties, one inductive bias">
        What a conv layer is really saying to the network: &ldquo;whatever
        you learn to look for, it&apos;s probably (a) small, (b) repeated,
        and (c) position-independent.&rdquo; That&apos;s a strong prior — it
        would be a terrible prior on, say, a legal document where absolute
        position matters — and it happens to be a <em>fantastic</em> prior
        on natural images. CNNs won vision because this prior is correct.
      </Callout>

      {/* ── Filter gallery ──────────────────────────────────────── */}
      <Prose>
        <p>
          Before neural nets, computer vision was a pile of hand-designed
          stamps. Researchers would sit down and write a 3×3 matrix of
          numbers because they knew that stamp, pressed across an image, would
          ink a useful picture — an edge map, a blur, a sharpened version.
          Flip through a few:
        </p>
      </Prose>

      <FilterGallery />

      <Prose>
        <p>
          The Sobel stamp on the left detects vertical edges because its
          weights are arranged to compute a horizontal gradient: positive on
          the right column, negative on the left. Press it across an image
          and bright blots appear exactly where pixels jump from dark to
          light horizontally. The blur stamp is uniformly positive — every
          neighbor contributes equally to the sum — which is, by definition,
          a local average. The sharpen stamp is subtract-the-blur-from-the-
          original. Each of these was a Ph.D. thesis in the 1970s.
        </p>
        <p>
          Here&apos;s the punchline. A conv <em>network</em> starts with
          random weights in its stamps, and after a few epochs of training on
          images, the stamps it learns look, empirically, a lot like these
          hand-designed ones. The first conv layer of a trained ImageNet
          model contains Gabor-like edge detectors, color blobs, little
          oriented line detectors — the same library of features vision
          scientists reverse-engineered out of the mammalian V1 cortex in
          the 1960s. The network rediscovers them because they&apos;re the
          right answer. We just stopped writing them by hand.
        </p>
      </Prose>

      <Personify speaker="Padding">
        Without me, every conv layer eats pixels off the border. A 3×3 stamp
        chews 2 pixels off each side per layer, so after 14 layers your 32×32
        image is a single pixel and you&apos;re out of signal. I paste zeros
        around the edge so the stamp can hang off the paper and the output
        stays the same size as the input. Nobody thinks about me until
        they&apos;re deep into a ResNet and notice the only reason it works
        is that every block keeps the spatial size constant.
      </Personify>

      {/* ── Multi-channel / multi-filter ────────────────────────── */}
      <Prose>
        <p>
          Two generalizations and we have the whole picture.
        </p>
        <p>
          <strong>Multi-channel input.</strong> An RGB image isn&apos;t{' '}
          <code>(H, W)</code> — it&apos;s <code>(3, H, W)</code>. To press a
          stamp on it, the stamp grows a third dimension too: it becomes{' '}
          <code>(3, k, k)</code>. At each spatial position you sum the
          element-wise products across all three channels at once. Still one
          scalar of ink per press. The channel dimension gets folded into
          the dot product.
        </p>
        <p>
          <strong>Multi-filter banks.</strong> One stamp gives you one
          feature map. Usually you want many — say 64 different edge / color
          / texture detectors running in parallel. Stack 64 stamps into a{' '}
          <KeyTerm>filter bank</KeyTerm>, and the weight tensor becomes{' '}
          <code>(out_ch, in_ch, k, k)</code>. Run the input through all 64
          and the output is <code>(64, H&apos;, W&apos;)</code> — 64 stacked
          feature maps, one per stamp, one tensor with channel depth.
        </p>
        <p>
          Put it together with a batch dimension and the canonical PyTorch
          shape is:
        </p>
      </Prose>

      <MathBlock caption="the shapes that matter in a real Conv2d">
{`input   :   (B,  in_ch,  H,   W )
weight  :   (out_ch,  in_ch,  k_h,  k_w)
bias    :   (out_ch,)
output  :   (B,  out_ch,  H',  W')`}
      </MathBlock>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations of the same operation. Pure Python to show
          the nested sum with nothing hidden. NumPy to vectorize what we
          can. PyTorch to hand the whole thing to a library that will run it
          on a GPU, differentiate it, and give you production-grade speed in
          seven lines. Same progression as{' '}
          <NeedsBackground slug="gradient-descent">gradient descent</NeedsBackground>:
          see the mechanics, scale them, then cede them.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · conv_scratch.py"
        output={`input  (5x5):
 [1 2 3 0 1]
 [0 1 2 3 4]
 [2 1 0 1 2]
 [3 0 1 2 3]
 [1 2 3 4 0]
output (3x3):
 [-3  1  1]
 [ 5  1 -3]
 [-5  1  3]`}
      >{`# 2D convolution, no channels, no batch — the textbook version.
def conv2d_naive(x, k):
    H, W = len(x), len(x[0])
    kh, kw = len(k), len(k[0])
    out_h, out_w = H - kh + 1, W - kw + 1         # no padding, stride 1
    y = [[0.0] * out_w for _ in range(out_h)]

    for i in range(out_h):                        # slide down
        for j in range(out_w):                    # slide across
            s = 0.0
            for u in range(kh):                   # window rows
                for v in range(kw):               # window cols
                    s += x[i + u][j + v] * k[u][v]
            y[i][j] = s
    return y

x = [[1, 2, 3, 0, 1],
     [0, 1, 2, 3, 4],
     [2, 1, 0, 1, 2],
     [3, 0, 1, 2, 3],
     [1, 2, 3, 4, 0]]

# vertical-edge detector (Sobel-x)
k = [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]]

y = conv2d_naive(x, k)
print("input  (5x5):"); [print('', row) for row in x]
print("output (3x3):"); [print('', row) for row in y]`}</CodeBlock>

      <Prose>
        <p>
          Four nested loops. Correct, obvious, unbearably slow. A 224×224
          image through a 64-filter bank running on this code on CPU would
          outlast a coffee break. Vectorize.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · conv_numpy.py"
        output={`output shape: (1, 1, 26, 26)
top-left 3x3 of output:
 [[-3.  1.  1.]
  [ 5.  1. -3.]
  [-5.  1.  3.]]`}
      >{`import numpy as np

def conv2d_numpy(x, w, b=None):
    """
    x : (B, in_ch, H, W)
    w : (out_ch, in_ch, kh, kw)
    b : (out_ch,)  or None
    returns (B, out_ch, H-kh+1, W-kw+1)  — valid padding, stride 1.
    """
    B, in_ch, H, W = x.shape
    out_ch, _, kh, kw = w.shape
    H_out, W_out = H - kh + 1, W - kw + 1

    y = np.zeros((B, out_ch, H_out, W_out), dtype=x.dtype)
    for i in range(H_out):
        for j in range(W_out):
            # window: (B, in_ch, kh, kw)
            window = x[:, :, i:i+kh, j:j+kw]
            # einsum collapses in_ch, kh, kw — leaves (B, out_ch)
            y[:, :, i, j] = np.einsum('bchw,ochw->bo', window, w)
    if b is not None:
        y += b.reshape(1, -1, 1, 1)
    return y

# one image, one channel, one 3x3 filter
x = np.array([[1, 2, 3, 0, 1],
              [0, 1, 2, 3, 4],
              [2, 1, 0, 1, 2],
              [3, 0, 1, 2, 3],
              [1, 2, 3, 4, 0]], dtype=np.float32).reshape(1, 1, 5, 5)

k = np.array([[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]], dtype=np.float32).reshape(1, 1, 3, 3)

# pretend we have a 28x28 input and a bank of 1 filter, valid padding
big = np.random.randn(1, 1, 28, 28).astype(np.float32)
y_big = conv2d_numpy(big, k)
print("output shape:", y_big.shape)

y_small = conv2d_numpy(x, k)
print("top-left 3x3 of output:\\n", y_small[0, 0])`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for u in range(kh): for v in range(kw): s += x[i+u][j+v] * k[u][v]',
            right: "np.einsum('bchw,ochw->bo', window, w)",
            note: 'the inner window × kernel dot product becomes a single einsum',
          },
          {
            left: 'no channels — (H, W) only',
            right: '(B, in_ch, H, W)',
            note: 'real conv layers always have a batch and channel axis',
          },
          {
            left: 'no bias',
            right: 'y += b.reshape(1, -1, 1, 1)',
            note: 'one bias scalar per output channel, broadcast across spatial dims',
          },
        ]}
      />

      <Prose>
        <p>
          Now the PyTorch version. In real code you&apos;ll never write
          either of the above. One line — <code>nn.Conv2d(in_ch, out_ch, k)</code>{' '}
          — and a tuned library (cuDNN on NVIDIA, MPS on Mac) runs the
          operation orders of magnitude faster than anything you can manage
          in Python.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · conv_pytorch.py"
        output={`weight shape: torch.Size([16, 3, 3, 3])
bias shape:   torch.Size([16])
input shape:  torch.Size([8, 3, 32, 32])
output shape: torch.Size([8, 16, 32, 32])    # same spatial size thanks to padding=1`}
      >{`import torch
import torch.nn as nn

# A canonical early-layer conv: RGB input, 16 learned filters, 3x3 kernel,
# stride 1, padding 1 (→ "same" padding → output H,W == input H,W).
conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1,
)

# Inspect the weight tensor shape — this is the thing most people get wrong.
# It is (out_ch, in_ch, kH, kW) — out-channels first.
print("weight shape:", conv.weight.shape)     # (16, 3, 3, 3)
print("bias shape:  ", conv.bias.shape)       # (16,)

# Batch of 8 RGB 32x32 images.
x = torch.randn(8, 3, 32, 32)
y = conv(x)
print("input shape: ", x.shape)               # (8, 3, 32, 32)
print("output shape:", y.shape)               # (8, 16, 32, 32)

# Total learnable params in this one layer:
#   weight: 16 * 3 * 3 * 3 = 432
#   bias:   16
#   total:  448     — and those 448 numbers are reused across all 32*32 = 1024
#                     spatial positions of every image in the batch.`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'conv2d_numpy(x, w, b)',
            right: 'conv = nn.Conv2d(3, 16, 3, padding=1); y = conv(x)',
            note: 'the layer owns its own weight and bias tensors as nn.Parameters',
          },
          {
            left: 'manual padding / stride arithmetic',
            right: 'padding=1, stride=1 arguments',
            note: 'PyTorch handles the shape math for you — but it will not forgive a mismatch',
          },
          {
            left: 'nested for loops over spatial positions',
            right: 'cuDNN kernel on GPU',
            note: 'under the hood, conv becomes a giant matrix multiply (im2col + GEMM)',
          },
        ]}
      />

      <Callout variant="note" title="how Conv2d is really implemented">
        You&apos;d think <code>nn.Conv2d</code> runs those nested loops on a
        GPU. It doesn&apos;t. The standard trick is <KeyTerm>im2col</KeyTerm>{' '}
        — extract every <code>k×k</code> window from the input as a column of
        a big matrix, reshape the kernel as another matrix, run a single GEMM
        (matrix multiply). GPUs are extraordinarily fast at matrix multiplies.
        Turning convolution into one is why a modern GPU can push a ResNet-50
        forward pass through in under 2 milliseconds. The stamp is a metaphor;
        under the hood it&apos;s linear algebra all the way down.
      </Callout>

      {/* ── Why CNNs won ─────────────────────────────────────────── */}
      <Callout variant="insight" title="why CNNs ate computer vision">
        AlexNet (2012) was an 8-layer CNN that beat the next-best ImageNet
        entry by more than 10 percentage points in top-5 error. The next year
        everyone switched. Within four years top-5 error dropped below
        human-level. Before 2012, vision was hand-engineered features (SIFT,
        HOG) fed into SVMs. After 2012, vision was CNNs end-to-end. The
        three properties — parameter sharing, translation equivariance, local
        connectivity — are what let you train a hundred-million-parameter
        network on a million-image dataset without overfitting into oblivion.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Same vs valid padding:</strong>{' '}
          <code>padding=0</code> (&ldquo;valid&rdquo;) means no zeros pasted —
          the stamp never hangs off the page — so the output shrinks by{' '}
          <code>k − 1</code> pixels every layer.{' '}
          <code>padding=(k-1)/2</code> (&ldquo;same&rdquo;, for odd{' '}
          <code>k</code>) keeps the spatial size constant. Stack valid conv
          layers and you&apos;ll hit a 0-pixel feature map sooner than
          you&apos;d guess. ResNets, VGGs, basically every modern conv block
          use same padding for a reason.
        </p>
        <p>
          <strong className="text-term-amber">Weight shape order:</strong>{' '}
          PyTorch stores the weight as <code>(out_ch, in_ch, kH, kW)</code>.
          TensorFlow defaults to <code>(kH, kW, in_ch, out_ch)</code>. Port a
          model between them without permuting and you&apos;ll get silent
          garbage outputs with no error message — the worst kind of bug.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting the channel dimension:</strong>{' '}
          the most common bug in early CNN code is writing a kernel of shape{' '}
          <code>(k, k)</code> for a 3-channel image. PyTorch refuses, but
          refuses with a stack trace three screens long. The kernel must
          always have an <code>in_ch</code> axis, even if{' '}
          <code>in_ch == 1</code>.
        </p>
        <p>
          <strong className="text-term-amber">Off-by-one in the shape formula:</strong>{' '}
          the floor in <code>⌊(H + 2p − k)/s⌋ + 1</code> is not a formality.
          With <code>H=28, k=3, s=2, p=0</code> you get 13, not 13.5. If your
          loss is NaN at step one, check the output shape before anything
          else.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Hand-code a Sobel edge detector as a Conv2d">
        <p>
          Build an <code>nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)</code>{' '}
          — 1 input channel, 2 output channels, 3×3 kernels, no bias.
          Manually set the two 3×3 weight matrices to the horizontal and
          vertical Sobel stamps:
          <code> [[-1,0,1],[-2,0,2],[-1,0,1]] </code> and{' '}
          <code> [[-1,-2,-1],[0,0,0],[1,2,1]] </code>.
        </p>
        <p className="mt-2">
          Load a grayscale image (any <code>(1, 1, H, W)</code> tensor — a
          letter glyph from MNIST is fine), press your Conv2d across it, and
          plot both output channels. You should see a vertical-edge map in
          channel 0 and a horizontal-edge map in channel 1.
        </p>
        <p className="mt-2">
          Verify correctness: run <code>scipy.ndimage.sobel(img, axis=1)</code>{' '}
          on the same image and check that it matches your Conv2d output (up
          to sign convention and a border-handling difference). When it
          matches, you&apos;ve just proved to yourself that the operation
          scipy has spent 20 years optimizing is the same operation PyTorch
          learns from scratch in any CNN.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: set <code>conv.weight.requires_grad = True</code>, run a
          tiny edge-detection training loop, and watch the initial Sobel
          weights drift. What&apos;s the network finding that Sobel missed?
        </p>
      </Challenge>

      {/* ── Takeaway + teaser ───────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> A convolution is a small
          stamp sliding across a grid, pressing one dot product per position
          onto a new page. Same stamp everywhere (parameter sharing), same
          detector works at any location (translation equivariance), each
          press only reads a local neighborhood (local connectivity). Three
          knobs shape the output: kernel size, stride, padding. Output shape
          is <code>⌊(H + 2p − k)/s⌋ + 1</code>, and you will use that formula
          more often than you&apos;d like. PyTorch&apos;s <code>nn.Conv2d</code>{' '}
          stores weights as <code>(out_ch, in_ch, k, k)</code>; miss the
          channel dimension or confuse the order and nothing works.
        </p>
        <p>
          <strong>Next up — Pooling.</strong> A conv layer keeps the spatial
          size (with same padding) — which means your feature maps keep
          growing. Stack 64 filters, then 128, then 256, and the tensor at
          layer 10 is enormous. A second operation earns its keep by
          trimming them down: take a 2×2 window, keep the max or the mean,
          throw the rest away. Pooling keeps the gist, drops the pixels.
          That&apos;s what turns a stack of conv layers into a real CNN.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Gradient-Based Learning Applied to Document Recognition',
            author: 'LeCun, Bottou, Bengio, Haffner',
            venue: 'Proc. IEEE 1998 — the LeNet paper, the origin of modern CNNs',
            url: 'http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf',
          },
          {
            title: 'A guide to convolution arithmetic for deep learning',
            author: 'Dumoulin, Visin',
            year: 2016,
            url: 'https://arxiv.org/abs/1603.07285',
          },
          {
            title: 'Dive into Deep Learning — Chapter 7: Convolutional Neural Networks',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai — the best free reference on the mechanics',
            url: 'https://d2l.ai/chapter_convolutional-neural-networks/index.html',
          },
          {
            title: 'ImageNet Classification with Deep Convolutional Neural Networks',
            author: 'Krizhevsky, Sutskever, Hinton',
            venue: 'NeurIPS 2012 — AlexNet, the paper that settled the argument',
            url: 'https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html',
          },
        ]}
      />
    </div>
  )
}
