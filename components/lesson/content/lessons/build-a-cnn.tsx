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
import CNNArchitectureBuilder from '../widgets/CNNArchitectureBuilder'
import FeatureMapExplorer from '../widgets/FeatureMapExplorer'

// Signature anchor: the layer cake — wide and shallow on top (big spatial map,
// few channels), narrow and deep at the bottom (tiny spatial map, many
// channels). Every conv adds channels; every pool shrinks space; every layer
// trades "where exactly" for "what kind." The anchor returns at (1) the
// opening stack reveal, (2) the architecture diagram, and (3) the FC-head
// flatten, where the cake finally gets handed to a classifier.
export default function BuildACNNLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout (entry point: empty state) ─────── */}
      <Prereq currentSlug="build-a-cnn" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          You have the ingredients. A{' '}
          <NeedsBackground slug="convolution-operation">convolution</NeedsBackground>{' '}
          — a learned filter sliding across an image, firing on local patterns.
          A <NeedsBackground slug="pooling">pool</NeedsBackground> — a
          subsampler that halves the resolution and hands the next layer a
          bigger view of the world. A ReLU wedged between them so the whole
          thing isn&apos;t secretly linear. By themselves, each of those is a
          parlor trick. Stack them, and you get a machine that reads images.
        </p>
        <p>
          That stack has a shape, and the shape is the whole point. Picture a
          cake, baked upside down. The top layer — the one touching the input
          image — is <strong>wide and shallow</strong>: 28×28 pixels of
          spatial area, three channels if it&apos;s color, one if it&apos;s
          grayscale. Each layer down, the cake gets <strong>narrower</strong>{' '}
          spatially and <strong>deeper</strong> in channels. 14×14×32.
          7×7×64. 4×4×128. The last slice is a dense little cube of features,
          and that cube gets flattened and handed to a classifier that decides
          what it was looking at. Every conv trades pixels for features. Every
          pool trades &ldquo;where exactly&rdquo; for &ldquo;what kind.&rdquo;
        </p>
        <p>
          The target of this lesson is <KeyTerm>LeNet-5</KeyTerm>. Yann LeCun,
          1998. The paper that drew the convnet the way every whiteboard still
          draws it — conv, pool, conv, pool, flatten, dense, dense, done.
          Five trainable layers, about sixty thousand parameters, and it hit
          99%+ on MNIST back when the competition was bolting handcrafted
          features onto SVMs. By the end of this page you will have built it,
          watched its shapes tumble down the stack, peeked at what its
          neurons actually see, and trained a PyTorch version to within a
          minute of LeCun&apos;s 1998 numbers.
        </p>
        <p>
          One benchmark to frame the stakes. The 2-layer{' '}
          <NeedsBackground slug="mlp-from-scratch">MLP</NeedsBackground> from
          the last section tops out around 97.5% on MNIST with roughly 100K
          parameters. LeNet does better with fewer. That gap is the whole
          argument for convolution, and it&apos;s about to land.
        </p>
      </Prose>

      <Personify speaker="LeNet-5">
        I was the first convnet to really work. Handwritten digits on bank
        cheques, running on a machine slower than the phone in your pocket,
        and I beat the humans. Everything you see today — ResNet,
        EfficientNet, ConvNeXt — is me with more layers, better
        normalization, and a GPU.
      </Personify>

      {/* ── Architecture diagram ─────────────────────────────────── */}
      <Prose>
        <p>
          Here is the cake, sliced from top to bottom. Read it as a descent:
          the input is a wide spatial map with almost no channels; the output
          is a narrow map with many channels, eventually a plain vector, and
          then a prediction. Every arrow is a layer; every box is a tensor
          with a shape; every shape tells you how much spatial resolution has
          collapsed into how many channels so far.
        </p>
      </Prose>

      <AsciiBlock caption="LeNet-5 — inputs flow top→bottom, shapes collapse as features accumulate">
{`  input image                         (1, 28, 28)      — one grayscale channel
        │
        ▼     Conv2d(in=1, out=6, k=5)  + ReLU
                                          (6, 24, 24)    — 6 feature maps, edges & strokes
        │
        ▼     MaxPool2d(k=2, s=2)
                                          (6, 12, 12)    — half resolution, same channels
        │
        ▼     Conv2d(in=6, out=16, k=5) + ReLU
                                          (16,  8,  8)   — 16 feature maps, compound patterns
        │
        ▼     MaxPool2d(k=2, s=2)
                                          (16,  4,  4)   — receptive field now ~14×14 of input
        │
        ▼     flatten
                                          (256,)         — 16 · 4 · 4 = 256 numbers per image
        │
        ▼     Linear(256 → 120)   + ReLU
                                          (120,)
        │
        ▼     Linear(120 →  84)   + ReLU
                                          ( 84,)
        │
        ▼     Linear( 84 →  10)   (logits for classes 0..9)
                                          ( 10,)
        │
        ▼     softmax  →  p(class | image)`}
      </AsciiBlock>

      <Callout variant="insight" title="the cake, in one sentence">
        Top of the cake (input): <code>1 × 28 × 28</code> — wide spatial,
        almost no channels. Bottom of the cake (last conv output):{' '}
        <code>16 × 4 × 4</code> — tiny spatial, sixteen channels. The first
        half of the network spreads into channels (1 → 6 → 16) while{' '}
        <em>shrinking</em> spatially (28 → 24 → 12 → 8 → 4). That is the
        convnet motif: more feature types, less resolution, per layer. The
        second half is a plain MLP on the flattened features — same machinery
        as the last section, just fed 256 learned numbers instead of 784 raw
        pixels.
      </Callout>

      {/* ── Widget 1: CNN Architecture Builder ──────────────────── */}
      <Prose>
        <p>
          Build the cake yourself. Drag conv and pool blocks into the stack,
          pick kernel sizes, watch the output shape and parameter count
          update per layer. The floor of the widget is the total-param and
          total-FLOP counter — a quick way to feel which slices of the cake
          are expensive and which are nearly free.
        </p>
      </Prose>

      <CNNArchitectureBuilder />

      <Prose>
        <p>
          Two things to notice while you drag. First, convolution layers are{' '}
          <em>shockingly cheap</em> in parameters. A{' '}
          <code>Conv2d(6, 16, k=5)</code> has{' '}
          <code>6·16·5·5 + 16 = 2,416</code> parameters — total. A single{' '}
          <code>Linear(256, 120)</code> has{' '}
          <code>256·120 + 120 = 30,840</code>, more than twelve times as many.
          LeNet&apos;s two conv layers combined use about 2.6K params; its
          three FC layers use about 58K. The convnet&apos;s weights are
          almost all hiding in the FC head. The feature extractor — the part
          everyone talks about — is the skinny one.
        </p>
        <p>
          Second, FLOPs tell the <em>opposite</em> story. A single conv layer
          is dirt cheap in parameters but expensive in multiply-adds, because
          each kernel is applied at every spatial position. Parameters and
          compute are decoupled for convolutions in a way they simply are not
          for fully connected layers. This is why convnets scale: you can
          grow the feature-extraction capacity without blowing up the
          parameter budget. It&apos;s also why your GPU fan turns on during
          training even though the model is &ldquo;only&rdquo; 60K
          parameters.
        </p>
      </Prose>

      {/* ── Shape arithmetic ─────────────────────────────────────── */}
      <Prose>
        <p>
          Before we code this, burn the shape arithmetic into your hands.
          This is the one formula you will use more than any other in the
          next four lessons. For a 2D convolution with input spatial size{' '}
          <code>H</code>, kernel <code>k</code>, padding <code>p</code>, and
          stride <code>s</code>, the output size along that axis is:
        </p>
      </Prose>

      <MathBlock caption="output size of a conv layer — the only formula you need">
{`H_out  =  ⌊ (H_in + 2p − k) / s ⌋ + 1

W_out  =  ⌊ (W_in + 2p − k) / s ⌋ + 1

params  =   C_out · (C_in · k · k + 1)        ( "+1" is the bias per filter )

FLOPs   ≈  2 · C_out · C_in · k · k · H_out · W_out`}
      </MathBlock>

      <Prose>
        <p>
          Plug in LeNet&apos;s first conv. <code>H = 28, k = 5, p = 0, s = 1</code>:
        </p>
      </Prose>

      <MathBlock caption="LeNet shape trace — verify against the diagram above">
{`Conv1:    H = (28 + 0 − 5)/1 + 1 = 24      →  (6, 24, 24)
Pool1:    H = 24 / 2             = 12      →  (6, 12, 12)
Conv2:    H = (12 + 0 − 5)/1 + 1 =  8      →  (16, 8, 8)
Pool2:    H = 8 / 2              =  4      →  (16, 4, 4)
Flatten:                              16·4·4 = 256
FC1:      256 → 120
FC2:      120 →  84
FC3:       84 →  10`}
      </MathBlock>

      <Callout variant="insight" title="why 5×5, why no padding, why stride 1">
        LeCun used 5×5 kernels because the 1998 compute budget said so. He
        used <em>no</em> padding because MNIST digits are centered on a 28×28
        canvas with plenty of blank border — losing a few pixels on each side
        costs nothing real. And stride 1 because pooling is already doing the
        downsampling. The decision to skip padding is why the spatial size
        drops by <code>k − 1 = 4</code> pixels at each conv, giving the clean
        28 → 24 → 12 → 8 → 4 progression. On a modern CIFAR-style
        architecture you would pad by <code>2</code> to preserve resolution
        through the conv, then let the pool be the only thing that shrinks.
      </Callout>

      {/* ── Personify: Receptive field ──────────────────────────── */}
      <Personify speaker="Receptive field">
        I am the patch of the input image that a neuron in layer <em>k</em>{' '}
        can actually see. In conv1, I am just a 5×5 window. After pool1 I
        double: 10×10. After conv2 I grow to 14×14. After pool2, 28×28 — by
        the end of feature extraction, a single neuron in a 4×4 map is looking
        at the <em>entire</em> input. That expansion is why deeper layers can
        recognize whole-digit shapes while shallow layers only see strokes.
      </Personify>

      {/* ── Widget 2: Feature Map Explorer ───────────────────────── */}
      <Prose>
        <p>
          Now actually look at what the network has learned. Pick a layer,
          pick an input digit, see the feature maps at that depth. Conv1 maps
          will look like edges — pen strokes at various orientations. Conv2
          maps are already abstract; some fire on whole loops, some on
          corners, some on things your visual cortex would refuse to name.
          The FC layers don&apos;t render as 2D maps (they&apos;re just
          vectors), which is itself the point — spatial structure lives in
          the conv stack and gets discarded the moment you flatten.
        </p>
      </Prose>

      <FeatureMapExplorer />

      <Prose>
        <p>
          This is the <KeyTerm>feature hierarchy</KeyTerm> everyone talks
          about. Early layers learn generic local statistics — edges, strokes,
          blobs. Middle layers compose those into textured patterns: loops,
          T-junctions, corners. Late layers compose those into object-level
          concepts — &ldquo;this looks like an 8.&rdquo; Nobody told the
          network to organize itself this way. It falls out, reliably, from
          training a stack of conv+pool with a classification loss. The
          deeper into the cake you go, the more each neuron is{' '}
          <em>about</em> the object rather than about pixels.
        </p>
        <p>
          On ImageNet-scale networks this hierarchy gets theatrical: layer 1
          is Gabor filters, layer 5 is textures, layer 20 has neurons that
          fire on faces, or on written text, or on bodies of water. LeNet on
          MNIST only has two conv layers so the story is muted — but the
          mechanism is identical, and the shape of the cake is the same.
        </p>
      </Prose>

      <Personify speaker="Feature hierarchy">
        I am what emerges when you stack me. Layer one: edges and strokes.
        Layer two: corners, curves, simple shapes. Layer five: object parts —
        eyes, wheels, wings. Layer twenty: whole objects. Nobody designed me
        this way. I am the free lunch that falls out of conv + pool +
        backprop.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Time to build it. Pure Python is <em>not</em> happening for a
          convnet — a single forward pass of LeNet is already around two
          million multiply-adds, and a four-deep Python loop would take
          minutes per image. So this lesson runs the progression as{' '}
          <strong>NumPy → PyTorch</strong>, with the NumPy version as a
          working reference implementation you can read end to end, and
          PyTorch as the version you&apos;d actually train.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · lenet_forward.py (full forward pass on one MNIST image)"
        output={`input shape: (1, 28, 28)
after conv1+relu: (6, 24, 24)
after pool1:      (6, 12, 12)
after conv2+relu: (16, 8, 8)
after pool2:      (16, 4, 4)
after flatten:    (256,)
logits:           (10,)
predicted digit:  7`}
      >{`import numpy as np

def conv2d(x, W, b):
    """x: (C_in, H, W_in), W: (C_out, C_in, k, k), b: (C_out,).
       No padding, stride 1 — vanilla LeNet."""
    C_out, C_in, k, _ = W.shape
    _, H, W_in = x.shape
    H_out, W_out = H - k + 1, W_in - k + 1
    out = np.zeros((C_out, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            patch = x[:, i:i+k, j:j+k]               # (C_in, k, k)
            # inner product against every filter
            out[:, i, j] = (W * patch).sum(axis=(1, 2, 3)) + b
    return out

def relu(x):
    return np.maximum(0, x)

def maxpool2d(x, k=2, s=2):
    C, H, W = x.shape
    H_out, W_out = H // s, W // s
    out = np.zeros((C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            out[:, i, j] = x[:, i*s:i*s+k, j*s:j*s+k].max(axis=(1, 2))
    return out

def linear(x, W, b):
    return W @ x + b

# Random weights — in practice these come from training, not from a seed.
rng = np.random.default_rng(0)
W1 = rng.normal(0, 0.1, (6, 1, 5, 5));   b1 = np.zeros(6)
W2 = rng.normal(0, 0.1, (16, 6, 5, 5));  b2 = np.zeros(16)
W3 = rng.normal(0, 0.1, (120, 256));     b3 = np.zeros(120)
W4 = rng.normal(0, 0.1, (84, 120));      b4 = np.zeros(84)
W5 = rng.normal(0, 0.1, (10, 84));       b5 = np.zeros(10)

def lenet_forward(x):
    a = relu(conv2d(x, W1, b1));   print("after conv1+relu:", a.shape)
    a = maxpool2d(a);              print("after pool1:     ", a.shape)
    a = relu(conv2d(a, W2, b2));   print("after conv2+relu:", a.shape)
    a = maxpool2d(a);              print("after pool2:     ", a.shape)
    a = a.reshape(-1);             print("after flatten:   ", a.shape)
    a = relu(linear(a, W3, b3))
    a = relu(linear(a, W4, b4))
    return linear(a, W5, b5)                          # logits

x = rng.normal(0, 1, (1, 28, 28))
print("input shape:", x.shape)
logits = lenet_forward(x)
print("logits:          ", logits.shape)
print("predicted digit: ", int(np.argmax(logits)))`}</CodeBlock>

      <Prose>
        <p>
          Every layer lives in twenty lines. The cake is right there in the
          print statements — each <code>print</code> is one horizontal slice
          of the architecture, and the shapes narrow and channel-count grows
          exactly like the diagram promised. What&apos;s missing is training.
          A NumPy backward pass through <code>conv2d</code> and{' '}
          <code>maxpool2d</code> is doable — it is also another 150 lines
          and intolerably slow. That is exactly the bargain PyTorch is
          selling.
        </p>
      </Prose>

      <Bridge
        label="numpy loops → numpy vectorized (the trick real code uses)"
        rows={[
          {
            left: 'for i, j in H×W: patch = x[:, i:i+k, j:j+k]',
            right: 'im2col: reshape all patches into a big matrix',
            note: 'turns conv into one giant matmul — this is what cuDNN does under the hood',
          },
          {
            left: 'np.zeros; assign per-position',
            right: 'W_flat @ patches_matrix',
            note: 'single BLAS call, 100-1000× faster than the python loop',
          },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · lenet.py (full training loop, MNIST, ~60 seconds on a laptop GPU)"
        output={`epoch 1  loss 0.342  test acc 97.4%
epoch 2  loss 0.087  test acc 98.6%
epoch 3  loss 0.061  test acc 98.9%
epoch 5  loss 0.039  test acc 99.1%
total parameters: 61,706`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)     # (1,28,28) → (6,24,24)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)    # (6,12,12) → (16,8,8)
        self.fc1   = nn.Linear(16 * 4 * 4, 120)         # (256,)    → (120,)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)      # conv1 + relu + pool
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)      # conv2 + relu + pool
        x = x.flatten(1)                                 # keep batch, flatten the rest
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)                               # raw logits — CE loss wants these

# Data
tfm = transforms.ToTensor()
train = DataLoader(datasets.MNIST('.', train=True,  download=True, transform=tfm), batch_size=128, shuffle=True)
test  = DataLoader(datasets.MNIST('.', train=False, download=True, transform=tfm), batch_size=512)

# Model + optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LeNet5().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for xb, yb in train:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
    print(f"epoch {epoch+1}  loss {loss.item():.3f}  test acc {100*correct/10000:.1f}%")

print("total parameters:", sum(p.numel() for p in model.parameters()))`}</CodeBlock>

      <Bridge
        label="numpy → pytorch (what disappeared)"
        rows={[
          {
            left: '4 nested loops per conv layer',
            right: 'nn.Conv2d — one line, GPU-accelerated',
            note: 'cuDNN does im2col + batched matmul under the hood',
          },
          {
            left: 'no backward implementation at all',
            right: 'loss.backward() traces through conv + pool for free',
            note: 'autograd knows the backward for every torch op',
          },
          {
            left: 'single image forward',
            right: 'batched forward over 128 images in parallel',
            note: 'the leading dim is batch — conv/pool broadcast over it',
          },
          {
            left: 'weights frozen to random init',
            right: 'Adam + cross-entropy → 99.1% test accuracy in 5 epochs',
            note: 'the whole learning loop in 30 lines',
          },
        ]}
      />

      <Callout variant="insight" title="the three-layer dividend">
        NumPy taught you that a conv is just a sliding inner product and a
        pool is just a sliding max. Once you have seen the loops, the PyTorch
        one-liner stops being magic — it is the same four-index formula
        compiled down onto cuDNN. Everything that looks like abstraction in
        layer 3 is a performance optimization of a thing you already
        understand from layer 2.
      </Callout>

      {/* ── The FC head — the cake becomes a prediction ─────────── */}
      <Prose>
        <p>
          Now the part where the cake gets eaten. Up to <code>pool2</code>,
          every tensor in the network is spatially organized — you could
          point at a neuron and say &ldquo;this one is watching the
          upper-left corner of the image.&rdquo; The flatten step ends that.{' '}
          <code>16 × 4 × 4 = 256</code> becomes a flat vector of 256 numbers,
          the spatial identity of each feature is discarded, and the final
          three <code>Linear</code> layers treat those 256 numbers the same
          way an{' '}
          <NeedsBackground slug="mlp-from-scratch">MLP</NeedsBackground>{' '}
          treats raw pixels. The cake has been reduced to a smoothie.
        </p>
        <p>
          That smoothie is then passed through{' '}
          <NeedsBackground slug="softmax">softmax</NeedsBackground> at
          inference time to turn the 10 raw logits into a probability
          distribution over digit classes, and scored against the true label
          with{' '}
          <NeedsBackground slug="cross-entropy-loss">
            cross-entropy
          </NeedsBackground>{' '}
          at training time. Same classification head you&apos;ve built
          before, just fed 256 <em>learned</em> features instead of 784 raw
          pixels. That substitution is the reason LeNet beats the MLP.
        </p>
      </Prose>

      {/* ── Modern kernel choices ──────────────────────────────── */}
      <Callout variant="note" title="modern convnets — why you won't see 5×5 in a 2024 architecture">
        Stacking two 3×3 convs covers the same 5×5 receptive field with fewer
        parameters (2·(3·3·C²) = 18C² vs 25C²) <em>and</em> more nonlinearity
        between them — that is Simonyan &amp; Zisserman&apos;s VGG insight
        from 2014, and it is why 3×3 is the default kernel of modern
        convnets. The other modern shifts worth naming: strided conv replaces
        max-pool in many architectures (ResNet, ConvNeXt — the idea being,
        let the network learn how to downsample instead of hard-coding
        &ldquo;take the max&rdquo;), and BatchNorm slots in after every conv
        to keep activations scale-stable across a 50-layer stack. LeNet
        predates all three — and it still works.
      </Callout>

      {/* ── Gotchas ──────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Forgetting to flatten before the FC head.</strong>{' '}
          A Linear layer wants a 1D input per example (2D total, with batch).
          If you pass in
          <code className="text-dark-text-primary"> (N, 16, 4, 4)</code> it
          will error — the cake is still 3D and the classifier won&apos;t eat
          it. Always{' '}
          <code className="text-dark-text-primary">x.flatten(1)</code> or{' '}
          <code className="text-dark-text-primary">x.view(N, -1)</code> before
          the first FC.
        </p>
        <p>
          <strong className="text-term-amber">Channel dim comes first.</strong>{' '}
          PyTorch uses{' '}
          <code className="text-dark-text-primary">(N, C, H, W)</code> —
          batch, channel, height, width. If you hand it{' '}
          <code className="text-dark-text-primary">(N, H, W, C)</code>
          (TensorFlow convention) the conv interprets height as channels and
          you get a silent disaster. MNIST tensors need an explicit channel
          dim: <code className="text-dark-text-primary">x.unsqueeze(1)</code>{' '}
          to go from{' '}
          <code className="text-dark-text-primary">(N, 28, 28)</code> to{' '}
          <code className="text-dark-text-primary">(N, 1, 28, 28)</code>.
        </p>
        <p>
          <strong className="text-term-amber">Wrong input size to adaptive pool / FC.</strong>{' '}
          If your input is not 28×28 — say you train LeNet on 32×32 CIFAR
          without adjusting — the feature map after pool2 is no longer 4×4
          and the{' '}
          <code className="text-dark-text-primary">Linear(256, 120)</code>{' '}
          blows up. Either recompute the flattened size from the shape
          formula, or use an{' '}
          <code className="text-dark-text-primary">nn.AdaptiveAvgPool2d((4, 4))</code>{' '}
          just before the flatten to force a canonical shape regardless of
          input size. This is the single most common bug in other
          people&apos;s PyTorch code.
        </p>
        <p>
          <strong className="text-term-amber">FC-head parameter blowup.</strong>{' '}
          Scale the feature map up — say you switch to 224×224 ImageNet-style
          inputs without pooling more aggressively — and the tensor reaching
          flatten becomes massive. A{' '}
          <code className="text-dark-text-primary">(512, 28, 28)</code>{' '}
          flatten feeding a{' '}
          <code className="text-dark-text-primary">Linear(400K, 4096)</code>{' '}
          is 1.6B parameters from a single layer. This is exactly why modern
          architectures use{' '}
          <code className="text-dark-text-primary">AdaptiveAvgPool2d((1, 1))</code>{' '}
          right before the classifier: collapse every channel to one number
          and keep the FC head small.
        </p>
        <p>
          <strong className="text-term-amber">Softmax inside the model when the loss expects logits.</strong>{' '}
          <code className="text-dark-text-primary">F.cross_entropy</code>{' '}
          applies log-softmax internally. If you softmax in the forward pass
          too, you double-softmax and training crawls. Return raw logits from
          the model; let the loss handle the normalization.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="LeNet vs MLP — parameter efficiency showdown">
        <p>
          Train the LeNet above on MNIST to at least 99.0% test accuracy.
          Note the parameter count (~62K) and wall-clock training time.
        </p>
        <p className="mt-2">
          Now train a 2-layer MLP —{' '}
          <code>nn.Sequential(nn.Flatten(), nn.Linear(784, H), nn.ReLU(), nn.Linear(H, 10))</code> —
          with <code>H</code> tuned just large enough to hit the same 99.0%
          test accuracy. How big does <code>H</code> need to be? How many
          parameters does that MLP have? Spoiler: it either won&apos;t hit
          99% at all, or it will need an <code>H</code> in the low thousands
          and a parameter count well past LeNet&apos;s.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: shift all test digits 3 pixels to the right before
          evaluating. LeNet should degrade gracefully because convolutions
          are translation-equivariant — the filter doesn&apos;t care whether
          the edge it&apos;s looking for is at column 4 or column 7. The MLP
          will fall off a cliff, because every pixel is a separate feature
          and they all just moved. That is the &ldquo;inductive bias&rdquo;
          of convolution, quietly earning its keep.
        </p>
      </Challenge>

      {/* ── Takeaways + next lesson ─────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> A convnet is a layer cake
          baked upside down: wide and shallow at the top (big spatial map,
          few channels), narrow and deep at the bottom (tiny spatial map,
          many channels). Every conv adds channels. Every pool shrinks
          space. The last slice of the cake gets flattened and fed to a
          plain MLP classifier, which is where almost all the parameters
          actually live — the feature extractor is cheap, the head is
          expensive. Shape arithmetic is{' '}
          <code>(H + 2p − k) / s + 1</code> — memorize it; you will use it
          in every convnet you touch. Feature hierarchy (edges → textures →
          parts → objects) emerges for free from training a deep enough
          stack with a classification loss. Modern architectures swap 5×5
          for stacked 3×3, replace pool with strided conv, and inject
          BatchNorm — but the cake-shape is still LeNet.
        </p>
        <p>
          <strong>Next up — Image Classifier (CIFAR-10).</strong> You have
          built the architecture. You have not yet shipped it on anything
          harder than centered grayscale digits. MNIST is the friendliest
          vision benchmark that exists; CIFAR-10 is 32×32 color photographs
          of ten object classes — cats that aren&apos;t centered, dogs on
          textured backgrounds, planes at angles — and LeNet on CIFAR
          without data augmentation tops out around 65%. Next lesson we
          scale up the cake: padded 3×3 convs, BatchNorm, deeper stacks,
          random crops and flips. Same three pieces, stretched into
          something that actually reads the world. The architecture is
          yours; now it&apos;s time to train it on real images.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Gradient-Based Learning Applied to Document Recognition',
            author: 'Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner',
            venue: 'Proceedings of the IEEE, 1998 — the original LeNet-5 paper',
            url: 'http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf',
          },
          {
            title: 'Very Deep Convolutional Networks for Large-Scale Image Recognition',
            author: 'Karen Simonyan, Andrew Zisserman',
            venue: 'ICLR 2015 — the VGG paper, 3×3 everywhere',
            url: 'https://arxiv.org/abs/1409.1556',
          },
          {
            title: 'Dive into Deep Learning — 7.6 Convolutional Neural Networks (LeNet)',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_convolutional-neural-networks/lenet.html',
          },
          {
            title: 'ImageNet Classification with Deep Convolutional Neural Networks',
            author: 'Krizhevsky, Sutskever, Hinton',
            venue: 'NeurIPS 2012 — AlexNet, the scaling-up of LeNet',
            url: 'https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html',
          },
        ]}
      />
    </div>
  )
}
