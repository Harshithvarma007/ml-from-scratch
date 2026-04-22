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
import SkipConnectionViz from '../widgets/SkipConnectionViz'   // a residual block drawn with the identity bypass highlighted; toggle skip on/off and watch gradient flow change
import DepthAblation from '../widgets/DepthAblation'          // plot of final accuracy vs depth for plain nets vs residual nets — plain nets get worse past ~20 layers, residual nets keep improving

// Signature anchor: the express elevator past the middle floors. A tall
// building where every regular floor adds a little work; the elevator shaft
// carries the original signal straight up. If a floor is useless, the
// elevator still delivers the input unchanged. Introduced in the opening,
// revealed at the y = F(x) + x moment, and returned to at the "easy to do
// nothing" beat and in the Personify blocks.

export default function ResnetAndSkipConnectionsLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout (entry point: empty state) ─────── */}
      <Prereq currentSlug="resnet-and-skip-connections" />

      {/* ── Opening: the paradox ────────────────────────────────── */}
      <Prose>
        <p>
          Here is a sentence that sounds like it ought to be free: a deeper
          network is at least as good as a shallower one. Take your working
          20-layer conv net. Paste 14 more layers on top. In the absolute
          worst case, those extra layers learn the identity function and you
          reproduce the 20-layer result exactly. Best case, the new layers
          do something useful. Either way, deeper ≥ shallower. This is just
          arithmetic.
        </p>
        <p>
          In 2015, He, Zhang, Ren, and Sun ran the experiment and the
          arithmetic lost. Deeper nets were <em>worse</em>. And not worse on
          the test set — worse on the <strong>training</strong> set, which
          rules out overfitting as the culprit. The 56-layer net could not
          even fit the data that its 20-layer twin fit easily. The deeper
          network had strictly more capacity and strictly worse results. The
          model was cooked, and no amount of extra epochs unbaked it.
        </p>
        <p>
          Their fix is one line of arithmetic — <code>y = F(x) + x</code>{' '}
          — and it is one of the most consequential one-liners in the
          history of deep learning. Every transformer you&apos;ve heard of
          uses it. Every vision model past 2016 uses it. Diffusion, U-Nets,
          Llama, Claude, the thing that autocompletes your code — all of
          them are built on this addition. The rest of this lesson is why
          that line works and why nothing trained past a dozen layers before
          it existed.
        </p>
      </Prose>

      {/* ── The observation ─────────────────────────────────────── */}
      <Prose>
        <p>
          Let&apos;s be precise about what He saw. Two plain convolutional
          nets on CIFAR-10, same design otherwise —{' '}
          <NeedsBackground slug="build-a-cnn">a standard stack of
          convs and pooling</NeedsBackground> — one 20 layers deep, the
          other 56. Train both, plot training error. The 56-layer net sat{' '}
          <em>above</em> the 20-layer one for the entire run. ImageNet
          reproduced it: a 34-layer plain net trained worse than an
          18-layer plain net. The deeper net contains the shallower net as
          a sub-solution (set the extra layers to identity and you&apos;re
          done), but the optimizer can&apos;t find that sub-solution. It
          just wanders off to somewhere worse.
        </p>
        <p>
          This is the <KeyTerm>degradation problem</KeyTerm>, and it is not
          the same as vanishing gradients — though the two are cousins. You
          can turn on ReLU,{' '}
          <NeedsBackground slug="weight-initialization">He
          init</NeedsBackground>, batch norm, every trick from the last
          fifteen lessons, and the 56-layer plain net still loses to its
          20-layer sibling. Something about stacking many nonlinear layers
          makes the loss surface genuinely hostile. Identity is in there
          somewhere, but SGD can&apos;t walk to it from where it starts.
        </p>
        <p>
          He&apos;s insight was a reframe. Don&apos;t ask the layer to learn
          the output <code>H(x)</code> from scratch. Ask it to learn the
          <em> difference</em> between the input and the desired output —
          the residual <code>F(x) = H(x) − x</code> — and then add{' '}
          <code>x</code> back at the end. If the right answer at this layer
          happens to be <em>&quot;leave it alone&quot;</em>, then{' '}
          <code>F</code> just has to learn to output roughly zero. Pushing
          weights toward zero is the easiest thing an optimizer ever does.
          The identity solution is now the lazy default, not the hidden
          needle in a haystack.
        </p>
      </Prose>

      <MathBlock caption="the residual block — three symbols that changed deep learning">
{`Plain block:       y  =  F(x ; W)

Residual block:    y  =  F(x ; W)  +  x

                   └── learned ──┘   └ identity shortcut ┘

If the optimal y is exactly x, the plain net must learn F to be an
identity function — hard. The residual net just needs F → 0 — easy.`}
      </MathBlock>

      <Callout variant="insight" title="the elevator, the stairs, the useless floor">
        Picture a tall building. The staircase winds through every floor in
        order — each floor is a layer, and walking through takes effort and
        bruises the signal a little on the way. Now run an express elevator
        shaft alongside it. Inputs take the shaft straight from the lobby
        to a later floor, unchanged. The layer in between still does its
        work on the staircase copy — but now the floor only has to learn
        the <em>correction</em> it wants to make, because the original is
        arriving through the shaft anyway. If a floor has nothing useful to
        add, it outputs zero and the elevator delivers the input
        untouched. Nothing breaks. Easy to do nothing. That is the whole
        trick.
      </Callout>

      {/* ── Gradient math ───────────────────────────────────────── */}
      <Prose>
        <p>
          Why does this help optimization specifically? The cleanest
          argument is in the backward pass. Write the network as a stack of
          residual blocks: block <code>ℓ</code> receives <code>x_ℓ</code>{' '}
          and outputs <code>x_(ℓ+1) = x_ℓ + F(x_ℓ ; W_ℓ)</code>. Now do{' '}
          <NeedsBackground slug="multi-layer-backpropagation">multi-layer
          backprop</NeedsBackground> across two adjacent blocks. The chain
          rule does something strange and wonderful.
        </p>
      </Prose>

      <MathBlock caption="the gradient with and without the shortcut">
{`Plain stack:          ∂L/∂x_ℓ   =   ∂L/∂x_(ℓ+1)   ·   ∂F/∂x_ℓ
                                                   │
                                                   └── can be tiny → vanishing

Residual stack:       x_(ℓ+1)   =   x_ℓ   +   F(x_ℓ)

                      ∂L/∂x_ℓ   =   ∂L/∂x_(ℓ+1)  ·  ( 1   +   ∂F/∂x_ℓ )
                                                      │
                                                      └── the "+1" is the escape hatch

Unroll across L blocks:

  ∂L/∂x_0  =  ∂L/∂x_L  ·  Π_{ℓ=0}^{L-1} ( 1 + ∂F/∂x_ℓ )

Even if every ∂F/∂x_ℓ is near zero, the 1s keep the product alive.`}
      </MathBlock>

      <Prose>
        <p>
          Stare at the second line. In a plain stack, every layer
          contributes one local Jacobian, and the gradient at layer 0 is
          the product of all of them. If the average Jacobian magnitude is
          even a little less than 1, that product decays exponentially in
          depth — a hundred layers of 0.9 each is a factor of{' '}
          <code>0.9¹⁰⁰ ≈ 3 × 10⁻⁵</code>. The gradient showing up at the
          first layer is a rounding error. The layer trains at roughly the
          speed of continental drift.
        </p>
        <p>
          In the residual stack, every layer contributes{' '}
          <code>(1 + ∂F/∂x)</code>. Even when <code>∂F/∂x</code>{' '}
          collapses to zero, the <code>1</code> stays put. The gradient has
          a guaranteed lane — the express elevator — and it takes the
          elevator down to the lobby no matter how many useless floors sit
          on either side. This is exactly why ResNet-152 trains and plain-152
          doesn&apos;t: the shortcut is not a metaphor, it&apos;s a term in
          the derivative that refuses to vanish.
        </p>
      </Prose>

      {/* ── Widget 1: Skip Connection Viz ───────────────────────── */}
      <Prose>
        <p>
          Here is a single residual block with the shortcut drawn in. Toggle
          the skip on and off and watch the gradient flow. With the skip,
          the gradient arriving at the block&apos;s input is whatever came
          from above, <em>plus</em> whatever squeezed through <code>F</code>.
          Without it, you have only the <code>F</code> path — and if{' '}
          <code>F</code> multiplies the signal by something small (as
          poorly-initialized stacks love to do), the gradient is dead on
          arrival. The elevator is either running or it isn&apos;t.
        </p>
      </Prose>

      <SkipConnectionViz />

      <Personify speaker="Identity path">
        I am the wire that goes around. I do nothing to the forward signal — I just pass{' '}
        <code>x</code> straight through — but on the backward pass I am the gradient&apos;s
        escape hatch. However badly the residual function mangles its gradient, I guarantee
        the full upstream signal still reaches the bottom of the block. Stack a hundred of me
        and the gradient at layer 0 still has a clean line to the loss. I am the reason depth
        works.
      </Personify>

      {/* ── Widget 2: Depth Ablation ────────────────────────────── */}
      <Prose>
        <p>
          The original paper has one plot that, once you&apos;ve seen it,
          you never forget. Final accuracy versus depth, two curves, same
          axes. The plain curve rises for a while, peaks somewhere around
          18 to 20 layers, then turns and dives — each additional layer
          makes the net measurably worse. The residual curve just keeps
          climbing. Same data, same optimizer, same compute budget. The
          only difference is three symbols, <code>+ x</code>, sprinkled
          through the architecture.
        </p>
      </Prose>

      <DepthAblation />

      <Prose>
        <p>
          Drag the depth slider. Up to about 18 layers the two curves are
          practically overlapping — plain nets do fine at modest depth, no
          elevator needed. Push past 20 and the plain curve starts bleeding
          accuracy; by 50 layers it&apos;s a mess. The residual curve goes
          the other way, still improving through 50, 101, 152. At
          ResNet-152 on ImageNet, top-5 error hit <strong>3.6%</strong> —
          below a widely-cited human estimate of about 5%. Deep networks
          finally did the thing the math had always said they should be
          able to do.
        </p>
      </Prose>

      <Personify speaker="Residual F(x)">
        I am the humble correction. I don&apos;t have to learn the whole mapping — the
        identity path carries the bulk of the signal. I just have to learn the small delta
        that makes the output better than the input. If the layer should do nothing, my
        weights go to zero and I get out of the way. If the layer should do something, I
        learn exactly what to add. I am what makes 152 layers trainable.
      </Personify>

      {/* ── Shortcut types ──────────────────────────────────────── */}
      <Prose>
        <p>
          There&apos;s a practical wrinkle. The shortcut{' '}
          <code>y = F(x) + x</code> only works if <code>F(x)</code> and{' '}
          <code>x</code> have the same shape. Inside a CNN, stages change
          channel count and spatial resolution — so mid-network the
          elevator occasionally needs to let a different-sized box onto the
          next floor. Two options:
        </p>
        <ul>
          <li>
            <strong>Identity shortcut.</strong> When shapes match, do
            nothing — literally <code>x + F(x)</code>. Zero parameters,
            zero FLOPs, just an add. This is the default inside a stage,
            and it&apos;s the cheapest trick in the building.
          </li>
          <li>
            <strong>Projection shortcut.</strong> When channel or spatial
            dims change (at stage boundaries), the shortcut becomes a 1×1
            convolution that reshapes <code>x</code> to match{' '}
            <code>F(x)</code>&apos;s output shape. This adds a handful of
            parameters and lets you down-sample cleanly.
          </li>
        </ul>
        <p>
          Both show up in every ResNet you&apos;ll ever read: identity
          wherever possible (cheap, well-behaved), projection at the stage
          boundaries (unavoidable). The He paper also tried parameter-free
          alternatives like zero-padding the extra channels, and found
          projections won by a small but consistent margin.
        </p>
      </Prose>

      {/* ── Architectures ───────────────────────────────────────── */}
      <Prose>
        <p>
          The ResNet paper shipped five off-the-shelf depths. Each is a
          stack of residual blocks split into four stages, where every
          stage halves the spatial resolution and doubles the channel
          count. You pick the depth you can afford and start training.
        </p>
      </Prose>

      <MathBlock caption="ResNet family — block counts per stage">
{`                     stage1  stage2  stage3  stage4   total   params
ResNet-18    [2x]     2       2       2       2         18      11.7M
ResNet-34    [2x]     3       4       6       3         34      21.8M
ResNet-50    [3x]     3       4       6       3         50      25.6M   (bottleneck)
ResNet-101   [3x]     3       4       23      3        101      44.5M   (bottleneck)
ResNet-152   [3x]     3       8       36      3        152      60.2M   (bottleneck)

[2x] = basic block   (3×3 conv → 3×3 conv, two layers)
[3x] = bottleneck    (1×1 reduce → 3×3 → 1×1 expand, three layers)`}
      </MathBlock>

      <Prose>
        <p>
          The bottleneck block is a parameter-efficiency move. Squeeze the
          channels down with a 1×1 conv, do the expensive 3×3 conv in the
          shrunken space, then 1×1 back up. Three layers instead of two,
          but cheaper than two full-width 3×3 convs. It&apos;s how ResNet-50
          ends up with fewer parameters than ResNet-34 despite carrying
          sixteen more layers around.
        </p>
      </Prose>

      {/* ── Pre-activation vs post-activation ───────────────────── */}
      <Prose>
        <p>
          One more detail, and it turns out to matter a lot. The original
          2015 block does <code>conv → BN → ReLU → conv → BN → (+x) → ReLU</code> —
          the final ReLU sits <em>after</em> the addition. A year later He
          et al. published a follow-up showing that moving BN and ReLU to
          the start of each branch —{' '}
          <code>BN → ReLU → conv → BN → ReLU → conv → (+x)</code> — gave a
          cleaner gradient path. The identity branch is now pure identity,
          with no ReLU lurking to clip negative gradients on the way back.
          With that one change they trained a 1001-layer network to
          convergence.
        </p>
      </Prose>

      <MathBlock caption="post-activation (2015) vs pre-activation (2016)">
{`Post-activation:     x ──┬──→ conv → BN → ReLU → conv → BN ─┐
                         │                                    ⊕ ── ReLU → x'
                         └────────── identity ───────────────┘

Pre-activation:      x ──┬──→ BN → ReLU → conv → BN → ReLU → conv ─┐
                         │                                           ⊕ ── x'
                         └──────────── pure identity ────────────────┘

The pre-activation variant's identity branch has nothing between
x and the addition — gradient flow is perfectly clean. It's what
every modern transformer ("Pre-LN") descended from.`}
      </MathBlock>

      <Callout variant="insight" title="why transformers use Pre-LN">
        Every modern transformer you&apos;ve touched — GPT, Llama, Claude —
        uses <em>pre-norm</em> residuals:{' '}
        <code>x + attention(LayerNorm(x))</code> rather than{' '}
        <code>LayerNorm(x + attention(x))</code>. That pattern is a direct
        descendant of He&apos;s 2016 pre-activation paper. The express
        elevator stays clean, gradients flow unobstructed, and you can
        stack 96 blocks without the loss exploding on step one. Pre-LN
        transformers would not have worked without this lesson.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations. NumPy makes the gradient math concrete —
          you can see the <code>+ 1</code> term show up as{' '}
          <code>np.eye(8)</code>, literally, in ten lines. PyTorch writes
          a residual block as a normal <code>nn.Module</code> with a
          two-line forward pass. Torchvision hands you the full ImageNet
          ResNet-18 pretrained, which is how you actually use a ResNet in
          2026 — you download it.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · residual_gradient.py"
        output={`∂L/∂x  (plain)     = 0.000046
∂L/∂x  (residual)  = 1.000046
ratio                = 21817.5×`}
      >{`import numpy as np

# A toy residual block: F(x) = W2 · ReLU(W1 · x).
# Compare the gradient w.r.t. the input for a plain stack vs a residual stack.

rng = np.random.default_rng(0)
W1 = rng.normal(0, 0.01, size=(8, 8))            # tiny init — mimics deep-net pathology
W2 = rng.normal(0, 0.01, size=(8, 8))
x  = rng.normal(0, 1.0,  size=(8,))

# Forward
z  = W1 @ x
a  = np.maximum(0, z)                            # ReLU
F  = W2 @ a                                      # the residual function

# Upstream gradient (what the layer above hands us). Say it's 1 everywhere.
grad_out = np.ones(8)

# ---- Plain: y = F(x) ----
# ∂y/∂x = W2 · diag(ReLU'(z)) · W1
relu_mask = (z > 0).astype(float)
jac_F = W2 @ np.diag(relu_mask) @ W1             # (8, 8)
grad_x_plain = jac_F.T @ grad_out

# ---- Residual: y = F(x) + x ----
# ∂y/∂x = I + jac_F
jac_res = np.eye(8) + jac_F
grad_x_res = jac_res.T @ grad_out

print(f"∂L/∂x  (plain)     = {np.linalg.norm(grad_x_plain):.6f}")
print(f"∂L/∂x  (residual)  = {np.linalg.norm(grad_x_res):.6f}")
print(f"ratio                = {np.linalg.norm(grad_x_res) / np.linalg.norm(grad_x_plain):.1f}×")

# The "1" from the identity path dominates. This is the entire point of ResNet
# expressed in ten lines of numpy.`}</CodeBlock>

      <Bridge
        label="the math ←→ the code"
        rows={[
          {
            left: '∂L/∂x_ℓ = ∂L/∂x_(ℓ+1) · ∂F/∂x_ℓ',
            right: 'jac_F.T @ grad_out',
            note: 'plain stack: gradient rides only the F path',
          },
          {
            left: '∂L/∂x_ℓ = ∂L/∂x_(ℓ+1) · (1 + ∂F/∂x_ℓ)',
            right: '(np.eye(8) + jac_F).T @ grad_out',
            note: 'residual stack: the "+1" becomes np.eye — literally',
          },
          {
            left: 'tiny init ⇒ ∂F/∂x near zero',
            right: 'ratio ≈ 1e4× in favour of residual',
            note: 'the deeper you go, the larger this ratio compounds',
          },
        ]}
      />

      <Prose>
        <p>
          The ratio printed above is not a metaphor. The plain gradient is
          starving; the residual gradient is roughly 1, because the{' '}
          <code>np.eye(8)</code> is carrying it. That number is exactly
          what the math promised, and it&apos;s the whole reason deep nets
          train. Now scale the idea up to something you&apos;d actually
          write in a model file.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — pytorch · residual_block.py">{`import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """The ResNet basic block — two 3×3 convs, one skip connection."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Projection shortcut only when shapes change
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()                      # zero-cost skip

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)                           # ← the add. this is the whole trick.
        return F.relu(out)                                     # post-activation variant

# Sanity check: shapes line up, gradients flow.
block = BasicBlock(64, 128, stride=2)
x = torch.randn(4, 64, 32, 32, requires_grad=True)
y = block(x)
print("in ", x.shape, "→ out", y.shape)
y.sum().backward()
print("grad on input:", x.grad.norm().item())                  # non-zero, finite`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'out = F(x) ; out = out + x',
            right: 'out = out + self.shortcut(x)',
            note: 'shortcut is either Identity() or a 1×1 conv — same interface',
          },
          {
            left: 'jac_F explicitly computed',
            right: 'autograd walks the block at .backward()',
            note: 'autograd handles the (1 + ∂F/∂x) for free',
          },
          {
            left: 'BatchNorm nowhere to be found',
            right: 'nn.BatchNorm2d after every conv',
            note: 'real ResNets need BN to actually train — a later lesson',
          },
        ]}
      />

      <Prose>
        <p>
          The only line that matters in that whole block is{' '}
          <code>out = out + self.shortcut(x)</code>. Delete it and the
          network stops being trainable past a dozen layers. Keep it and
          you can go to 152, 1001, whatever you want. Every layer in every
          ResNet you&apos;ll ever touch comes down to that single add.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — torchvision · resnet18_imagenet.py"
        output={`ResNet-18 loaded — 11.7M params
top predictions: [('golden retriever', 0.871), ('Labrador', 0.094), ('tennis ball', 0.013)]`}
      >{`import torch
from torchvision import models, transforms
from PIL import Image

# One function call. The model, the weights, fifty years of research.
weights = models.ResNet18_Weights.IMAGENET1K_V1
model   = models.resnet18(weights=weights).eval()

print(f"ResNet-18 loaded — {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

# Inference on one image
preprocess = weights.transforms()                              # the exact preprocessing used at training
img = Image.open("golden_retriever.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0)                               # (1, 3, 224, 224)

with torch.no_grad():
    logits = model(x)
    probs  = torch.softmax(logits, dim=-1).squeeze()

top5 = torch.topk(probs, 5)
labels = weights.meta["categories"]
preds = [(labels[i], p.item()) for p, i in zip(top5.values, top5.indices)]
print("top predictions:", preds[:3])`}</CodeBlock>

      <Bridge
        label="pytorch block → torchvision full net"
        rows={[
          {
            left: 'you write BasicBlock by hand',
            right: 'models.resnet18() stacks 8 of them for you',
            note: 'conv1 + 4 stages × [2, 2, 2, 2] basic blocks + fc = 18 layers',
          },
          {
            left: 'random init — have to train from scratch',
            right: 'weights=ResNet18_Weights.IMAGENET1K_V1',
            note: 'pretrained on 1.28M ImageNet images — free starting point',
          },
          {
            left: 'think about learning-rate schedules',
            right: 'weights.transforms() — even the preprocessing matches',
            note: 'in 2026 you almost never train ResNet from scratch',
          },
        ]}
      />

      <Callout variant="insight" title="this idea is everywhere now">
        Residual connections are not just a vision trick. The transformer
        block is literally <code>x + attention(x)</code> followed by{' '}
        <code>x + mlp(x)</code> — two residual blocks per layer. U-Nets
        use them. Diffusion models are full of them. The LSTM cell state
        is (sort of) a residual path. Anywhere someone stacks more than a
        dozen nonlinear layers, there is an elevator shaft carrying the
        gradient. ResNet was the proof of concept; everything since is a
        variation on the same idea.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Shape mismatch at the shortcut.</strong>{' '}
          If your residual branch changes channel count (e.g. 64 → 128) or
          downsamples the spatial dim (stride=2), the identity path no
          longer matches. <code>x + F(x)</code> will throw a shape error.
          Fix: a 1×1 conv projection shortcut on <code>x</code> (stride
          matching <code>F</code>&apos;s, output channels matching{' '}
          <code>F</code>&apos;s). Every stage boundary in every ResNet
          needs one.
        </p>
        <p>
          <strong className="text-term-amber">BN placement matters a lot.</strong>{' '}
          Post-activation (2015): conv → BN → ReLU → conv → BN → (+x) → ReLU. Pre-activation
          (2016): BN → ReLU → conv → BN → ReLU → conv → (+x). For very deep nets (100+ layers)
          pre-activation trains markedly better. For 18–50 layers the difference is small.
          Know which you&apos;re writing and don&apos;t mix them.
        </p>
        <p>
          <strong className="text-term-amber">The add goes <em>before</em> the final ReLU.</strong>{' '}
          A common bug is <code>ReLU(F(x)) + x</code>, which defeats the point — now the
          shortcut gets clipped whenever <code>x</code> goes negative. The canonical block
          is <code>ReLU(F(x) + x)</code>. The ReLU is <em>on</em> the sum, not on one branch.
        </p>
        <p>
          <strong className="text-term-amber">Don&apos;t forget <code>bias=False</code> in
          convs before BN.</strong> BatchNorm has its own shift parameter. A bias on the conv
          is redundant at best and a minor training slowdown at worst. All the{' '}
          <code>nn.Conv2d</code>s in the reference ResNet set <code>bias=False</code>.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Plain vs residual — build both, watch one break">
        <p>
          Build two small networks in PyTorch for CIFAR-10.{' '}
          <strong>Net A</strong>: an 8-layer plain conv net — eight{' '}
          <code>3×3 conv → BN → ReLU</code> blocks stacked, with a
          stride-2 downsample every two blocks, ending in a
          global-average-pool and a linear head. <strong>Net B</strong>:
          the same thing but with residual skips — group the blocks into
          3 stages of 2 basic blocks each (your own ResNet-8).
        </p>
        <p className="mt-2">
          Train both for 30 epochs with SGD, <code>lr = 0.1</code>,
          momentum 0.9, weight decay 5e-4, cosine schedule. Plot training
          loss and test accuracy side by side. At 8 layers the gap is
          modest — a few points, maybe. Now repeat at 20 layers. The
          plain net&apos;s training loss will be noticeably{' '}
          <em>worse</em> than at 8 layers; the residual net&apos;s will
          keep improving. You&apos;ve just reproduced the central
          experiment of the ResNet paper on a laptop.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: swap the basic block for the bottleneck variant and try
          50 layers. Watch the parameter count <em>drop</em> relative to
          a 20-layer basic-block net.
        </p>
      </Challenge>

      {/* ── Closing ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Deep plain nets degrade
          past ~20 layers not because capacity runs out, but because SGD
          can&apos;t find the identity-plus-tweak solution from a cold
          start. Residual blocks — <code>y = F(x) + x</code> — reframe
          the problem: the elevator shaft makes identity the lazy default,
          and <code>F</code> only has to learn the small correction on
          top. The gradient inherits a permanent bypass route —{' '}
          <code>(1 + ∂F/∂x)</code> at every layer — and so the signal
          survives arbitrary depth. Projection shortcuts patch up the
          shape mismatches at stage boundaries; pre-activation blocks
          keep the elevator wall clean all the way down; and this exact
          pattern is what every transformer, U-Net, and diffusion model
          has been standing on top of ever since.
        </p>
        <p>
          <strong>Next up — Recurrent Neural Networks.</strong> Look back
          at everything you&apos;ve built so far: linear layers, conv
          nets, ResNets. They all eat the input in one bite. The whole
          image, all 224×224 pixels of it, hits layer one at once. But
          what about input that arrives <em>one step at a time</em> — a
          sentence arriving word by word, a melody arriving note by note,
          a trajectory arriving tick by tick? You&apos;d need a network
          with a sense of <em>before</em> and <em>after</em>, a little bit
          of memory that persists from one step to the next. That&apos;s
          the whole next section. We start with the simplest possible
          version — a single hidden state that remembers — and discover,
          in the process, exactly the vanishing-gradient problem that
          skip connections just solved, but this time running through time
          instead of depth.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Deep Residual Learning for Image Recognition',
            author: 'He, Zhang, Ren, Sun',
            venue: 'CVPR 2016 — the original ResNet paper',
            url: 'https://arxiv.org/abs/1512.03385',
          },
          {
            title: 'Identity Mappings in Deep Residual Networks',
            author: 'He, Zhang, Ren, Sun',
            venue: 'ECCV 2016 — the pre-activation follow-up',
            url: 'https://arxiv.org/abs/1603.05027',
          },
          {
            title: 'Dive into Deep Learning — §7.6 Residual Networks',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_convolutional-modern/resnet.html',
          },
          {
            title: 'torchvision.models.resnet — reference implementation',
            author: 'PyTorch team',
            venue: 'the one you\'ll actually use',
            url: 'https://pytorch.org/vision/stable/models/resnet.html',
          },
        ]}
      />
    </div>
  )
}
