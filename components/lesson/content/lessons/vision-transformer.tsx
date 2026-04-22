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
import ImageToPatches from '../widgets/ImageToPatches'
import PatchAttention from '../widgets/PatchAttention'

// Signature anchor: cut the image into puzzle pieces and read them like a
// sentence. A 224×224 image sliced into 16×16 patches becomes a 196-token
// "paragraph" — each patch is a "word," the transformer reads them in order.
// Return at the opening (the puzzle box), the patch-embedding reveal (each
// piece flattened and linearly projected like a word-embedding), and the
// "vision as language" consolidation (attention doesn't care what the tokens
// are, only that they're tokens).
export default function VisionTransformerLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="vision-transformer" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Picture a jigsaw puzzle box. You tip it onto the table and out fall a
          few hundred little squares — a sliver of sky here, a corner of an
          ear there, a chunk of wheel, a patch of fur. You don&apos;t see the
          image yet. You see <em>pieces</em>. That&apos;s the move this whole
          lesson rests on: take an image, cut it into puzzle pieces, and read
          the pieces like a sentence. A 224×224 photo sliced into 16×16 patches
          is a 196-token paragraph. Each patch is a word. The transformer reads
          them left-to-right-top-to-bottom and figures out what the picture is
          the same way it figures out what a sentence means.
        </p>
        <p>
          For thirty years, that was heresy. Vision meant{' '}
          <NeedsBackground slug="convolution-operation">convolutions</NeedsBackground>
          — grids, kernels, pooling, hand-tuned inductive biases that told a
          network <em>images are 2D and nearby pixels are related</em>. Language
          meant recurrence, then attention — sequences, tokens, transformers.
          Two architectures, two worlds, two PhD tracks.
        </p>
        <p>
          Then in October 2020 a Google team submitted a paper called{' '}
          <em>An Image Is Worth 16×16 Words</em> and quietly knocked the wall
          down. Their claim: take the{' '}
          <NeedsBackground slug="transformer-block">transformer block</NeedsBackground>{' '}
          that was eating NLP, feed it pixels in puzzle-piece chunks instead of
          words, and if you scale the data far enough it beats every convnet
          you can name. This is the{' '}
          <KeyTerm>Vision Transformer</KeyTerm> — ViT — and it&apos;s the reason
          your 2025 multimodal model has one architecture for both modalities
          instead of two.
        </p>
        <p>
          This lesson is the bridge. You&apos;ve seen CNNs in this section.
          You&apos;ve heard of transformers in the wild but haven&apos;t built
          one — that&apos;s a whole section by itself, coming later. For today
          the transformer encoder is a black box. The part we care about — the
          genuinely novel piece — is the one at the front: how do you cut an
          image into a paragraph the transformer can actually read?
        </p>
      </Prose>

      <Callout variant="insight" title="the great convergence">
        Before ViT, a multimodal model meant gluing a CNN to a language model
        with tape. After ViT, both halves are <em>the same block</em> —{' '}
        <NeedsBackground slug="self-attention">self-attention</NeedsBackground>{' '}
        plus an MLP, repeated <code>N</code> times. One codebase. One kernel.
        One set of optimizations. Pixels in one tower, words in the other,
        same machinery chewing on tokens — no one inside the block knows or
        cares where those tokens came from. That indifference is why CLIP,
        Flamingo, GPT-4V, and every Gemini sibling share a single trunk.
      </Callout>

      {/* ── The hypothesis ─────────────────────────────────────── */}
      <Prose>
        <p>
          The ViT hypothesis, in one sentence:{' '}
          <em>an image is a sequence of patches</em>. Not a 2D grid with spatial
          structure the network needs to respect via convolutions — a sequence,
          flat, ordered, just like a sentence of words. The transformer then
          treats each patch the way BERT treats a word: embed it, add a
          positional code so it knows where the piece sits in the grid, mix all
          the tokens with self-attention, and read out a prediction.
        </p>
        <p>
          This is more radical than it sounds. A CNN is <em>structurally</em>{' '}
          forced to care about locality — a 3×3 kernel literally cannot look at
          two pixels that are a hundred pixels apart on the first layer. A ViT
          has no such constraint. On layer one, every patch can attend to every
          other patch. The sky piece in the top corner can shake hands with the
          grass piece in the bottom corner on step one. The network learns
          locality (or doesn&apos;t) from data alone.
        </p>
      </Prose>

      {/* ── Widget 1: Image → Patches ──────────────────────────── */}
      <Prose>
        <p>
          Start with the surgery. Drag the patch-size slider and watch the
          image get diced. A 224×224 image with 16×16 patches gives you a 14×14
          grid — 196 puzzle pieces, each a little 16×16×3 tile. Those 196
          pieces <em>are</em> your sequence. From here on, the transformer
          doesn&apos;t see an image. It sees a list of 196 vectors, laid out
          end to end like words on a page, each one carrying the contents of
          one piece.
        </p>
      </Prose>

      <ImageToPatches />

      <Prose>
        <p>
          Three things to notice as you slide:
        </p>
        <ul>
          <li>
            <strong>Patch size is a hyperparameter with quadratic cost.</strong>{' '}
            Halving the patch side quadruples the sequence length, and
            attention is <code>O(N²)</code> in sequence length. Going from{' '}
            <code>16×16</code> to <code>8×8</code> patches makes the model 16×
            more expensive. Smaller pieces, longer paragraph, bigger bill.
          </li>
          <li>
            <strong>Each patch is flattened to a vector.</strong> A 16×16×3 RGB
            piece becomes a 768-dimensional vector (just{' '}
            <code>16 · 16 · 3 = 768</code>). No averaging, no pooling — the raw
            pixels <em>are</em> the token.
          </li>
          <li>
            <strong>The 2D structure is lost at this step.</strong> The grid
            collapses into a flat list; the puzzle pieces come out of the box
            in a line. We&apos;ll bolt position back on via a positional
            encoding — the transformer has no other way to know that patch 42
            is above patch 56.
          </li>
        </ul>
      </Prose>

      {/* ── Patch embedding math ───────────────────────────────── */}
      <Prose>
        <p>
          Now the quiet reveal — the step that explains why the whole
          vision-as-language thing even works. Input image{' '}
          <code>x ∈ ℝ^(H×W×C)</code>. Reshape into <code>N</code> patches of
          size <code>P×P×C</code>, where <code>N = HW/P²</code>. Flatten each
          patch, stack into a matrix, multiply by a single learned matrix{' '}
          <code>E</code> to project to model dimension <code>d_model</code>.
          That last line — patch-times-matrix — is the piece to stare at.
          It&apos;s the <em>exact same operation</em> as a word-embedding
          lookup. NLP tokenizes text into symbols and maps each symbol to a
          vector. ViT tokenizes an image into pieces and maps each piece to a
          vector. Same machine, different side of the puzzle box.
        </p>
      </Prose>

      <MathBlock caption="patch embedding — the one-matrix version">
{`x  ∈  ℝ^(H × W × C)                           input image

→ reshape to patches:
xₚ ∈  ℝ^(N × (P² · C))       where  N = HW / P²

→ linear projection:
z₀ = xₚ · E         ,    E ∈ ℝ^((P² · C) × d_model)

→ prepend [CLS] token, add positional encoding:
z₀ = [ x_CLS ; xₚ · E ] + E_pos

final shape:  z₀  ∈  ℝ^((N + 1) × d_model)`}
      </MathBlock>

      <Prose>
        <p>
          For ViT-Base on 224×224 images: <code>P=16</code>, <code>N=196</code>
          , <code>d_model=768</code>. After patch embedding + CLS + positional
          encoding you have a <code>197 × 768</code> tensor. That&apos;s a
          sequence of length 197 — 197 tokens, a paragraph with one puzzle
          piece per word plus one &ldquo;whole image&rdquo; slot up front.
          Every transformer block that follows sees exactly that. It neither
          knows nor cares that the tokens came from pixels. To the attention
          machinery, this is indistinguishable from a sentence.
        </p>
        <p>
          The{' '}
          <NeedsBackground slug="positional-encoding">positional encoding</NeedsBackground>{' '}
          <code>E_pos</code> is a learned <code>(N+1) × d_model</code> table —
          one row per sequence position, including the CLS slot. It&apos;s the
          grid reference that tells patch 42 &ldquo;you&apos;re in row 3,
          column 0&rdquo; without re-introducing 2D structure to the network.
          ViT uses learned 1D position embeddings, not the sinusoidal ones from
          the original Attention Is All You Need paper; the authors found
          little difference in practice and 1D-learned is the simplest thing
          that works.
        </p>
      </Prose>

      <Personify speaker="Patch">
        I am a 16×16 square of pixels — a scrap of fur, a slice of sky, a
        corner of a wheel. Alone I mean very little: 768 numbers that could be
        anything. But in the sequence I live in, my neighbors and I will shout
        at each other through attention until we agree on what the whole
        picture is. I am a <em>token</em>. Treat me like one.
      </Personify>

      {/* ── Widget 2: Patch Attention ──────────────────────────── */}
      <Prose>
        <p>
          Now the interesting part. Once the puzzle pieces are embedded and
          positionally-coded, every transformer block does self-attention —
          each patch computes a <em>query</em>, a <em>key</em>, and a{' '}
          <em>value</em>, and every patch&apos;s output is a weighted sum of
          every other patch&apos;s value, with weights given by query·key
          similarity. You haven&apos;t built attention yet, so take that on
          faith for now. What matters here is the consequence: patches can,
          and do, attend to <em>anywhere</em> in the image. The ear piece can
          ask the nose piece a question on step one.
        </p>
      </Prose>

      <PatchAttention />

      <Prose>
        <p>
          Click a query patch. The heatmap shows how strongly it attends to
          every other piece of the grid in a trained ViT. Early layers often
          attend locally — ViT has to <em>learn</em> the CNN-style locality
          bias from data, rediscovering the &ldquo;nearby pieces go together&rdquo;
          prior that a convolution gets for free. Deeper layers often go
          long-range, hooking a background patch to a foreground object, or
          linking the left and right side of a symmetric thing. A probing
          paper by Raghu et al. (2021) showed ViT&apos;s early heads actually
          include both local-only heads (similar to 3×3 convolutions) and
          global heads — the network <em>builds</em> locality where it&apos;s
          useful and discards it where it isn&apos;t.
        </p>
        <p>
          This is a different failure mode from a CNN&apos;s. A CNN can&apos;t
          look far without stacking depth or dilating kernels. A ViT can look
          anywhere from layer one but has to learn from scratch what to pay
          attention to. The trade is <em>inductive bias for expressive power</em>{' '}
          — and it&apos;s only a good trade when you have enough data to fill
          that freedom with signal.
        </p>
      </Prose>

      <Callout variant="note" title="inductive bias, in one paragraph">
        A CNN has two biases baked into the architecture:{' '}
        <strong>locality</strong> (a 3×3 kernel sees only nearby pixels) and{' '}
        <strong>translation equivariance</strong> (the same filter runs at
        every position, so a cat in the top-left is recognized the same way as
        a cat in the bottom-right). These are huge priors — they&apos;re what
        let a CNN learn ImageNet with &ldquo;only&rdquo; 1M images. A ViT has
        neither baked in. Every puzzle piece starts with the same freedom to
        attend anywhere, which sounds great until you realize the network has
        to discover &ldquo;pixels near each other are usually related&rdquo;
        from scratch. That&apos;s why the original ViT needed pretraining on
        JFT-300M — 300 million images — before it beat ResNets on ImageNet. No
        free lunch.
      </Callout>

      {/* ── The [CLS] token ────────────────────────────────────── */}
      <Prose>
        <p>
          One last trick before we code. The transformer outputs a sequence —{' '}
          <code>197</code> vectors in, <code>197</code> vectors out, same
          paragraph in, same paragraph out, contents rewritten. But
          classification wants a single vector to feed a linear head. How do
          you pool a paragraph down to one word?
        </p>
        <p>
          ViT borrows BERT&apos;s move. Prepend a learned token, called{' '}
          <KeyTerm>[CLS]</KeyTerm>, to the sequence. It has no corresponding
          patch — it&apos;s a free-floating vector in the embedding, the same
          on every example, trained like any other parameter. A blank sticker
          stuck to the front of the puzzle pieces. As it passes through the
          transformer blocks it attends to every patch and every patch attends
          to it. By the final block its embedding has soaked up information
          from the entire grid. You read it off, feed it to a linear
          classifier, and that&apos;s your logits.
        </p>
      </Prose>

      <Personify speaker="[CLS] token">
        I don&apos;t represent any part of the picture. I represent the{' '}
        <em>whole</em> picture. I&apos;m a learned sponge, prepended at
        position zero, that spends twelve transformer blocks asking every
        patch what they think. By the time the final layer hands me off to the
        classifier head, I am the image, compressed into 768 numbers. Use me,
        then discard me — my only job is to pool.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, same progression you&apos;ve seen everywhere in this
          series. NumPy to show the patch mechanics with nothing up our
          sleeves. PyTorch to show a single ViT block we can actually
          back-prop. <code>timm</code> to show what a real ViT call looks like
          in a production repo.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · patch_embed_numpy.py"
        output={`image:           (3, 224, 224)
patch grid:      (14, 14)
flat patches:    (196, 768)
projected:       (196, 768)
with [CLS]+pos:  (197, 768)`}
      >{`import numpy as np

# Fake image: 3 channels, 224×224.
rng = np.random.default_rng(0)
img = rng.normal(size=(3, 224, 224)).astype(np.float32)
C, H, W = img.shape
P = 16                                    # patch side
D = 768                                   # d_model

assert H % P == 0 and W % P == 0, "image must divide by patch size"

# (1) Reshape into an (N_h, N_w) grid of (C, P, P) patches, then flatten.
#     Trick: reshape + transpose. It's the same data in a different order.
Nh, Nw = H // P, W // P                   # 14, 14
patches = img.reshape(C, Nh, P, Nw, P)    # (3, 14, 16, 14, 16)
patches = patches.transpose(1, 3, 0, 2, 4) # (14, 14, 3, 16, 16)
patches = patches.reshape(Nh * Nw, C * P * P)  # (196, 768)

# (2) Linear projection to d_model. For a 16×16×3 patch this is a 768→768 map;
#     the matrix is learned in a real model, random here for illustration.
E = rng.normal(size=(C * P * P, D)).astype(np.float32) * 0.02
tokens = patches @ E                      # (196, 768)

# (3) Prepend [CLS], add positional encoding. Both learned in real ViT.
cls = rng.normal(size=(1, D)).astype(np.float32) * 0.02
tokens = np.concatenate([cls, tokens], axis=0)          # (197, 768)
E_pos = rng.normal(size=(Nh * Nw + 1, D)).astype(np.float32) * 0.02
z0 = tokens + E_pos                       # (197, 768) — ready for transformer

print(f"image:           {img.shape}")
print(f"patch grid:      ({Nh}, {Nw})")
print(f"flat patches:    {patches.shape}")
print(f"projected:       {tokens[1:].shape}")
print(f"with [CLS]+pos:  {z0.shape}")`}</CodeBlock>

      <Prose>
        <p>
          Into PyTorch. Two things to notice. First, patch extraction is
          usually written as a <code>Conv2d</code> with stride=kernel=P — which
          looks like a convolution, but it isn&apos;t, really. Non-overlapping
          stride-equals-kernel is algebraically the same reshape-and-project
          you just did in NumPy; cuDNN just has a faster kernel for it. The
          pieces don&apos;t overlap, so no information is shared between them
          — that&apos;s what keeps this &ldquo;tokenization&rdquo; and not
          &ldquo;feature extraction.&rdquo; Second, the transformer encoder
          block comes pre-built as <code>nn.TransformerEncoderLayer</code>. We
          call it, we don&apos;t define it.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — pytorch · vit_block.py">{`import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Image → flat token sequence via a strided convolution."""
    def __init__(self, img_size=224, patch=16, in_ch=3, d_model=768):
        super().__init__()
        self.n_patches = (img_size // patch) ** 2
        # Conv2d(kernel=patch, stride=patch) = non-overlapping patch projection.
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, x):                                # x: (B, 3, 224, 224)
        x = self.proj(x)                                 # (B, D, 14, 14)
        x = x.flatten(2).transpose(1, 2)                 # (B, 196, D)
        return x

class ViTLite(nn.Module):
    """One-block ViT skeleton — swap N=12 for the real ViT-Base."""
    def __init__(self, img_size=224, patch=16, d_model=768, n_heads=12, n_blocks=1, n_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, 3, d_model)
        N = self.patch_embed.n_patches
        # Learned [CLS] and 1D positional encoding — both are nn.Parameter.
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, N + 1, d_model))
        # Transformer encoder — we get attention + MLP + LayerNorm as one call.
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation='gelu', norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=n_blocks)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):                                # (B, 3, 224, 224)
        B = x.size(0)
        tokens = self.patch_embed(x)                     # (B, 196, D)
        cls = self.cls.expand(B, -1, -1)                 # (B, 1, D)
        z = torch.cat([cls, tokens], dim=1) + self.pos   # (B, 197, D)
        z = self.encoder(z)                              # (B, 197, D)
        z = self.norm(z)
        return self.head(z[:, 0])                        # CLS → logits  (B, 1000)

model = ViTLite()
x = torch.randn(2, 3, 224, 224)
print(model(x).shape)                                    # torch.Size([2, 1000])`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'img.reshape(...).transpose(...).reshape(...)',
            right: 'nn.Conv2d(stride=patch, kernel=patch)',
            note: 'same linear map, but differentiable and CUDA-fused',
          },
          {
            left: 'cls = rng.normal(size=(1, D))',
            right: 'self.cls = nn.Parameter(torch.zeros(1, 1, D))',
            note: 'nn.Parameter registers it for gradient descent',
          },
          {
            left: 'hand-rolled self-attention loop',
            right: 'nn.TransformerEncoderLayer(...)',
            note: 'multi-head attention + MLP + LayerNorm in one line',
          },
          {
            left: 'np.concatenate([cls, tokens]) + E_pos',
            right: 'torch.cat([cls, tokens], dim=1) + self.pos',
            note: 'same op, different dtype and device story',
          },
        ]}
      />

      <Prose>
        <p>
          Layer three is what you&apos;d actually ship. <code>timm</code> — the{' '}
          PyTorch-image-models library — carries every ViT variant pretrained
          on ImageNet-21k, JFT, LAION, and more. In practice nobody trains ViT
          from scratch on their own data; they finetune a <code>timm</code>{' '}
          checkpoint. The pieces (patch embed, CLS, pos embed) you just wrote
          by hand are sitting right there, with the same names and the same
          shapes, just with weights that actually work.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — timm · vit_timm.py"
        output={`model.default_cfg: crop_pct=0.9, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
patch embed weight shape:  torch.Size([768, 3, 16, 16])
CLS token shape:           torch.Size([1, 1, 768])
pos embed shape:           torch.Size([1, 197, 768])
logits shape:              torch.Size([1, 1000])`}
      >{`import timm
import torch

# One call loads architecture + weights pretrained on ImageNet-21k,
# finetuned on ImageNet-1k. The model is ready for inference.
model = timm.create_model('vit_base_patch16_224', pretrained=True).eval()

# timm exposes the same pieces we built by hand.
print("patch embed weight shape: ", model.patch_embed.proj.weight.shape)
print("CLS token shape:          ", model.cls_token.shape)
print("pos embed shape:          ", model.pos_embed.shape)

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits = model(x)                     # (1, 1000) — ImageNet classes
print("logits shape:             ", logits.shape)`}</CodeBlock>

      <Bridge
        label="hand-rolled → timm"
        rows={[
          {
            left: 'ViTLite(n_blocks=1)',
            right: "timm.create_model('vit_base_patch16_224')",
            note: 'same architecture at N=12, plus pretrained weights',
          },
          {
            left: 'model.patch_embed.proj',
            right: 'model.patch_embed.proj',
            note: 'timm uses the exact same Conv2d trick — ours lines up 1:1',
          },
          {
            left: 'self.cls / self.pos',
            right: 'model.cls_token / model.pos_embed',
            note: 'same parameters, same shapes — naming converges',
          },
        ]}
      />

      <Callout variant="insight" title="one architecture, every modality">
        Stop here and appreciate the punchline: the model you just sketched
        is, structurally, <em>the same model</em> as GPT-2. Both are a stack
        of transformer blocks chewing on a sequence of embedded tokens with
        positional codes. Attention doesn&apos;t care what the tokens{' '}
        <em>are</em> — patches of fur, subword fragments, audio frames — it
        only cares that they&apos;re tokens. Vision becomes language the
        moment you cut the image into pieces. The only real differences
        between a ViT and a GPT are (a) the tokenizer — puzzle pieces vs BPE
        —  and (b) whether self-attention is causal-masked. That&apos;s why a
        CLIP image tower and a CLIP text tower share 80% of their code.
        That&apos;s why you can glue a ViT encoder onto a language-model
        decoder and get a multimodal model with one training loop. One
        architecture. Every modality.
      </Callout>

      {/* ── Scale requirements + variants ──────────────────────── */}
      <Callout variant="warn" title="ViT eats data for breakfast">
        The 2020 paper&apos;s ugliest truth: on ImageNet-1k alone, ViT loses
        to a ResNet. You need ImageNet-21k (14M images) to draw even, and
        JFT-300M (300M images) to win. Pretraining regime matters more than
        architecture at this scale — a network with no locality prior needs
        mountains of data to discover that nearby puzzle pieces belong
        together. Budget accordingly: if you have 50k labeled images and no
        pretrained backbone, a ResNet50 will likely beat a from-scratch ViT.
      </Callout>

      <Prose>
        <p>
          Two follow-up architectures worth knowing, because they&apos;re the
          pragmatic compromises you&apos;ll actually meet in 2026 codebases:
        </p>
        <ul>
          <li>
            <strong>DeiT (Touvron et al. 2021).</strong>{' '}
            &ldquo;Data-efficient ViT.&rdquo; Same architecture, trained with
            heavy augmentation + a distillation token that mimics a CNN
            teacher. Matches ViT quality with <em>only</em> ImageNet-1k — no
            JFT required. This is what made ViT accessible to non-Google labs.
          </li>
          <li>
            <strong>Hybrid ViTs.</strong> Replace the patchify-by-Conv2d with a
            small CNN stem (say, a ResNet&apos;s first three stages). The CNN
            does the low-level feature extraction — which is what CNNs are
            good at — and the transformer does long-range mixing over the
            resulting tokens. ConvNeXt, Swin, and most modern vision backbones
            sit somewhere on this spectrum.
          </li>
        </ul>
      </Prose>

      {/* ── Gotchas ────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Image size must divide patch size.</strong>{' '}
          A ViT with <code>P=16</code> on a 225×225 image will error (or
          silently crop). If you want arbitrary resolution, you must either
          pad, resize, or pick a patch size that divides your shortest side.{' '}
          <code>timm</code> uses <code>bicubic</code> resize by default — not
          free; it blurs edges.
        </p>
        <p>
          <strong className="text-term-amber">Positional encoding doesn&apos;t transfer to new resolutions.</strong>{' '}
          If you pretrain at 224 (196 patches) and finetune at 384 (576
          patches), the positional encoding table is the wrong size. The fix
          is a 2D bicubic interpolation of <code>E_pos</code>, treating it as
          a 14×14 image and upsampling to 24×24. Every ViT codebase implements
          this and every ViT codebase has had a subtle bug in it.
        </p>
        <p>
          <strong className="text-term-amber">CLS position matters — and isn&apos;t the only pooling choice.</strong>{' '}
          ViT uses CLS-at-position-0. Some variants use mean-pooling over all
          patch tokens (&ldquo;GAP&rdquo;) instead, which can be a few tenths
          of a point better on ImageNet. If you swap pooling strategies you
          must retrain the head. Don&apos;t mix.
        </p>
        <p>
          <strong className="text-term-amber">Normalization stats are patch-level.</strong>{' '}
          The input normalization (ImageNet mean/std) is applied pixel-wise
          before patching — it is not per-patch. A common bug is to forget
          this and normalize twice or not at all.
        </p>
      </Gotcha>

      {/* ── Challenge ──────────────────────────────────────────── */}
      <Challenge prompt="Build patch embedding and verify against timm">
        <p>
          Write a <code>PatchEmbedScratch</code> module that takes an image{' '}
          <code>(B, 3, 224, 224)</code> and returns <code>(B, 197, 768)</code>{' '}
          — patches + CLS + learned positional encoding — without using{' '}
          <code>nn.Conv2d</code>. Use pure reshape + transpose +{' '}
          <code>nn.Linear</code>, following the NumPy code above.
        </p>
        <p className="mt-2">
          Then load <code>timm.create_model(&apos;vit_base_patch16_224&apos;,
          pretrained=True)</code>, copy its <code>patch_embed.proj</code>{' '}
          Conv2d weights into your <code>nn.Linear</code> (you&apos;ll need a
          reshape — the Conv2d weight is <code>(768, 3, 16, 16)</code>, your
          Linear weight is <code>(768, 768)</code>), and confirm that your
          output matches timm&apos;s to within floating-point noise:{' '}
          <code>torch.allclose(mine, theirs, atol=1e-5)</code>.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: time both on a 32-image batch. The Conv2d version will be
          faster by a noticeable margin — cuDNN has an optimized kernel for
          stride=kernel, and a plain Linear doesn&apos;t. This is the usual{' '}
          &ldquo;equivalent math, unequal hardware&rdquo; lesson.
        </p>
      </Challenge>

      {/* ── Closing + section teaser ────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> ViT&apos;s one-sentence
          contribution: an image is a sequence of puzzle pieces, and you can
          run a transformer on it. The patch embedding is a linear projection
          (a <code>Conv2d</code> with stride=kernel under the hood, same math
          as a word-embedding lookup), the CLS token pools the paragraph into
          a single vector, and the positional encoding puts the 2D grid back
          after you flattened it away. The trade versus a CNN is{' '}
          <em>less inductive bias, more data hunger, higher ceiling</em>. At
          ImageNet-21k and above, ViT wins. Below, CNNs or hybrids win. The
          deeper lesson — the one that pays off for the rest of this
          curriculum — is that attention doesn&apos;t care what the tokens
          <em> are</em>. Pixels, words, audio frames, protein residues — if
          you can tokenize it, a transformer can eat it. Vision is just
          language with different puzzle pieces.
        </p>
        <p>
          <strong>Up next — Build GPT.</strong> You&apos;ve seen two
          tokenizers now: one that cuts images into fixed 16×16 squares, and
          one that handles{' '}
          <NeedsBackground slug="word-embeddings">embeddings</NeedsBackground>{' '}
          for words. Image tokens are easy to eyeball — you can literally see
          them on the grid. But what actually counts as a token for a language
          model? &ldquo;Cat&rdquo; is a token. Is &ldquo;cats&rdquo;? Is{' '}
          &ldquo;cat&rsquo;s&rdquo;? Is the space before it? The answer
          changes how everything downstream behaves — vocab size, context
          window, which spellings the model can even express. Next lesson:{' '}
          <strong>Tokenizer (Byte Pair Encoding)</strong>. We stop taking the
          word &ldquo;token&rdquo; for granted, build the tokenizer GPT
          actually uses from scratch, and see why &ldquo; hello&rdquo; and
          &ldquo;hello&rdquo; are not the same token.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale',
            author: 'Dosovitskiy et al.',
            venue: 'ICLR 2021 — the original ViT paper',
            url: 'https://arxiv.org/abs/2010.11929',
          },
          {
            title: 'Training Data-Efficient Image Transformers & Distillation Through Attention',
            author: 'Touvron et al.',
            venue: 'ICML 2021 — DeiT',
            url: 'https://arxiv.org/abs/2012.12877',
          },
          {
            title: 'Do Vision Transformers See Like Convolutional Neural Networks?',
            author: 'Raghu, Unterthiner, Kornblith, Zhang, Dosovitskiy',
            venue: 'NeurIPS 2021 — representation probing of ViT',
            url: 'https://arxiv.org/abs/2108.08810',
          },
          {
            title: 'Dive into Deep Learning — Chapter 11.8: Transformers for Vision',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html',
          },
          {
            title: 'PyTorch Image Models (timm)',
            author: 'Wightman',
            venue: 'reference implementation — every ViT variant, pretrained',
            url: 'https://github.com/huggingface/pytorch-image-models',
          },
        ]}
      />
    </div>
  )
}
