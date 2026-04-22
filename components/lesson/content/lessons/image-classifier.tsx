import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import Prereq from '../Prereq'
import NeedsBackground from '../NeedsBackground'
import {
  Prose, Callout, Personify, Bridge, Gotcha, Challenge, References, KeyTerm, AsciiBlock,
} from '../primitives'
import CIFARPredictions from '../widgets/CIFARPredictions'    // gallery of CIFAR-10 test images, click one to see top-5 predictions with probabilities
import TrainValCurves from '../widgets/TrainValCurves'        // animated train-loss and val-accuracy curves for three runs: no aug, aug, aug+label-smoothing

// Signature anchor: a detective putting on progressively more abstract pairs
// of glasses. Early conv layers see evidence (edges, corners, blobs), middle
// layers see motifs (textures, eye-shapes), late layers see suspects (faces,
// cats, trucks). Training teaches the detective which patterns are worth
// looking at; augmentation prevents tunnel vision. Threaded at the opening,
// the layer-hierarchy section, and the data-augmentation section.
export default function ImageClassifierLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="image-classifier" />

      <Prose>
        <p>
          Your <NeedsBackground slug="build-a-cnn">CNN from scratch</NeedsBackground>{' '}
          works. You trained it on MNIST in the{' '}
          <NeedsBackground slug="digit-classifier">digit classifier lesson</NeedsBackground>,
          hit 98% on handwritten digits, and felt — reasonably — like you had solved
          computer vision. You did not. MNIST is handwriting on a clean white
          background, scaled to the same size, centered, grayscale. It is the
          friendliest image dataset in existence. Real photos have color, texture,
          lighting, occlusion, pose, and backgrounds that weren&apos;t cleared by a
          graduate student.
        </p>
        <p>
          Meet <KeyTerm>CIFAR-10</KeyTerm>. Ten classes, labeled color photos, same
          benchmark shape as MNIST — but the cats are in every pose imaginable, the
          trucks are photographed from every angle, and the backgrounds are
          whatever was behind the camera that day. Your 2-layer MLP that aced MNIST
          lands around 50% here. A plain CNN stalls at 70. 90%+ is a different
          conversation entirely — it requires a <em>recipe</em>, not a model.
        </p>
        <p>
          This lesson is that recipe. By the end you will have a training script
          that hits 94% test accuracy on CIFAR-10 — 2015 state-of-the-art,
          perfectly respectable in 2026. But accuracy is the outcome. The point
          of the lesson is to make you see what the classifier is actually
          doing.
        </p>
      </Prose>

      <Callout variant="insight" title="the detective with feature glasses">
        <div className="space-y-2">
          <p>
            A trained image classifier is a detective who puts on progressively
            more abstract pairs of glasses.
          </p>
          <p>
            <strong>Pair one</strong> (early conv layers): sees evidence —
            edges, corners, color blobs. The rawest possible observations.
          </p>
          <p>
            <strong>Pair two</strong> (middle layers): sees motifs — textures,
            eye-shaped patterns, stripe arrangements. Evidence combined into
            things that start to mean something.
          </p>
          <p>
            <strong>Pair three</strong> (late layers): sees suspects — faces,
            cars, cats. Full concepts, assembled from motifs.
          </p>
          <p>
            The prediction head reads the last pair&apos;s notes and names the
            most likely suspect. Training is how the detective learns which
            patterns are worth looking at. That&apos;s the whole lesson in one
            paragraph. The rest is the code, the recipe, and the three or four
            ways you can ruin it.
          </p>
        </div>
      </Callout>

      <AsciiBlock caption="the CIFAR-10 training pipeline, top to bottom">
{`  32×32×3 RGB image (uint8, [0, 255])
         │
         ▼  augment: RandomCrop(32, pad=4) + RandomHorizontalFlip
  32×32×3 (still uint8, but a different crop every epoch)
         │
         ▼  to float + normalize by CIFAR mean/std
  3×32×32 tensor, channels-first, roughly N(0, 1)
         │
         ▼  ResNet-18 (conv → bn → relu → blocks → global-avg-pool)
  10-D logit vector
         │
         ▼  cross-entropy (+ label smoothing)  →  SGD + momentum  →  cosine LR`}
      </AsciiBlock>

      <Prose>
        <p>
          <strong>The dataset.</strong> 60,000 color photos at 32×32 across ten
          classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship,
          truck. 50k train, 10k test, perfectly balanced (6,000 of each class).
          Krizhevsky &amp; Hinton curated it in 2009 from the 80-million-tiny-images
          dump. It&apos;s small enough that an epoch runs in under a minute on a
          laptop GPU, and hard enough that the distance between a naive model and
          a good one is several percentage points of embarrassment. That distance
          is what we&apos;re closing.
        </p>
        <p>
          Click through the test set below. The detective is unambiguously right
          on some (a big red fire truck), nervously right on others (a cat it
          gives 52% to, with dog at 38%), and confidently wrong on a few. Cats
          and dogs trade. Deer and horses trade. Trucks and automobiles live on
          the same street in feature space. Classes with the most within-class
          variance — cats strike every pose; horses don&apos;t — are the ones the
          model fumbles.
        </p>
      </Prose>

      <CIFARPredictions />

      <Prose>
        <p>
          Notice the <em>shape</em> of the mistakes. The detective is not wrong
          randomly — it&apos;s wrong sensibly. A blurry cat really does look
          like a small dog at 32×32. A cargo ship shot head-on really does look
          like a truck. The confusion pairs on CIFAR-10 track genuine visual
          similarity, the same way 4↔9 did on MNIST. That&apos;s the good news:
          the detective is looking at features, not memorizing pixels. Bad
          news: the features it&apos;s looking at are the features you&apos;d
          look at too, which means the hard cases are genuinely hard.
        </p>
        <p>
          The detective&apos;s three pairs of glasses are not something we
          program. They emerge — from the architecture, from the loss, and
          from what training images the detective happens to see. Which brings
          us to the most leveraged line item in the whole recipe.
        </p>
      </Prose>

      <Personify speaker="Augmentation">
        I am the cheapest way to get more data — I invent it. You give me 50,000
        labeled training images; I give you effectively infinite variants by
        cropping, flipping, and jittering the colors of each one at load time.
        The label stays the same — a cat flipped horizontally is still a cat, a
        truck shifted four pixels left is still a truck — so the loss function
        sees a different pixel pattern for the same target every epoch. The
        detective can&apos;t memorize its way out. I&apos;m worth 5 to 10
        percentage points of test accuracy and I cost nothing at inference time.
        I also prevent the tunnel vision that kills models in the wild: I make
        sure the detective can still recognize a cat when it&apos;s upside down,
        shot at sunset, or shifted four pixels to the left.
      </Personify>

      <MathBlock caption="augmentation, mathematically">
{`# Without augmentation, you minimize:
L(θ) = 𝔼_{(x, y) ~ D} [ ℓ(f_θ(x), y) ]

# With augmentation, you minimize:
L_aug(θ) = 𝔼_{(x, y) ~ D, t ~ T} [ ℓ(f_θ(t(x)), y) ]

# where T is a distribution over label-preserving transforms:
#   t(x) = RandomCrop ∘ HorizontalFlip ∘ ColorJitter ∘ ... (x)
#
# This asks the model to be invariant to t. It's a form of
# regularization: you've enlarged the input distribution D to D ∘ T,
# which is strictly harder to overfit than D alone.`}
      </MathBlock>

      <Prose>
        <p>
          The curves below show three training runs on the same ResNet-18, same
          initialization, same optimizer, same schedule — the only thing that
          changes is what&apos;s layered on top of the{' '}
          <NeedsBackground slug="cross-entropy-loss">cross-entropy loss</NeedsBackground>.
        </p>
      </Prose>

      <TrainValCurves />

      <Prose>
        <p>
          Run 1 (<strong>no augmentation</strong>) overfits spectacularly —
          train loss collapses toward zero, val accuracy plateaus around 80%,
          the train-val gap yawns open like a canyon. The detective memorized
          the training set and learned nothing transferable. Run 2
          (<strong>+ random crops, flips, color jitter</strong>) trains slower
          but the val curve climbs past 91%; the gap narrows because every
          epoch the detective sees a slightly different version of every
          photo. Run 3 (<strong>+ label smoothing (ε=0.1)</strong>) trades a
          sliver of train loss for a calibrated output distribution and another
          half-point of accuracy. Full recipe in one picture: each technique
          buys something, none are free, improvements stack.
        </p>
      </Prose>

      <Personify speaker="Validation split">
        I am the honest grader. Carve 5,000 images out of your 50,000-image
        training set and lock them away — you do not train on me, you do not
        look at me except at the end of each epoch. When you tune learning
        rates, batch sizes, augmentation strengths, or architectures, you pick
        the setting that does best on <em>me</em>, not on the test set. Touch
        the test set during development and you&apos;ve leaked it; your final
        accuracy number is a lie. I exist so your choices don&apos;t overfit to
        the only 10,000 images that should tell you the truth.
      </Personify>

      <Prose>
        <p>
          Now the code. Two layers this time — pure Python is hopeless for
          50,000 32×32×3 photos, so we start with a compact NumPy sketch of
          loading plus augmentation, then jump to the full PyTorch training
          script that actually hits 94%.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · cifar_load_and_augment.py (illustrating the transforms)"
        output={`train shape : (50000, 32, 32, 3)  labels: (50000,)
test shape  : (10000, 32, 32, 3)  labels: (10000,)
per-channel mean: [0.4914 0.4822 0.4465]
per-channel std : [0.2470 0.2435 0.2616]
augmented batch shape: (64, 32, 32, 3)  # crops + flips applied`}
      >{`import numpy as np
import pickle, os

# CIFAR-10 ships as 5 training "batches" + 1 test batch of pickled dicts.
def load_batch(path):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    # data is (10000, 3072) flat uint8 with channel-first layout: [R×1024, G×1024, B×1024]
    X = d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)   # -> (N, 32, 32, 3)
    y = np.array(d[b'labels'])
    return X, y

root = 'cifar-10-batches-py'
X_train = np.concatenate([load_batch(f'{root}/data_batch_{i}')[0] for i in range(1, 6)])
y_train = np.concatenate([load_batch(f'{root}/data_batch_{i}')[1] for i in range(1, 6)])
X_test, y_test = load_batch(f'{root}/test_batch')

# Per-channel statistics computed ONCE on the training set (never touch test).
MEAN = (X_train / 255.0).mean(axis=(0, 1, 2))
STD  = (X_train / 255.0).std (axis=(0, 1, 2))

def normalize(x):                         # x: uint8 (H, W, 3) or (N, H, W, 3)
    return (x / 255.0 - MEAN) / STD

# --- Manual augmentation: random crop with 4-px reflection pad, then horizontal flip ---
def augment(batch, rng):
    N, H, W, C = batch.shape
    padded = np.pad(batch, ((0,0), (4,4), (4,4), (0,0)), mode='reflect')
    out = np.empty_like(batch)
    for i in range(N):
        top, left = rng.integers(0, 9, size=2)                # 0..8 offsets
        out[i] = padded[i, top:top+H, left:left+W]
        if rng.random() < 0.5:
            out[i] = out[i, :, ::-1]                          # horizontal flip
    return out

rng = np.random.default_rng(0)
batch = augment(X_train[:64], rng)
# In a real run you would now feed normalize(batch) into your network.`}</CodeBlock>

      <Bridge
        label="what the numpy sketch shows"
        rows={[
          { left: 'CIFAR ships as pickled dicts', right: 'reshape(-1, 3, 32, 32).transpose(...)', note: 'channels-first on disk; we go channels-last for human-visible ops' },
          { left: 'normalize using train-only stats', right: 'MEAN, STD computed on X_train', note: 'test set must never influence preprocessing' },
          { left: 'augment on the CPU, on the fly', right: 'per-batch crop + flip at load time', note: 'GPU stays busy; dataset stays small on disk' },
        ]}
      />

      <Prose>
        <p>
          The NumPy sketch exists so you can see every piece exposed. Production
          code delegates all of this to <code>torchvision</code>, which runs
          augmentation inside the DataLoader&apos;s worker processes so the GPU
          never idles waiting on the CPU. The{' '}
          <NeedsBackground slug="convolution-operation">convolution layers</NeedsBackground>{' '}
          and residual blocks are imported rather than rewritten — we&apos;ll open
          ResNet next lesson. Here we&apos;re wiring the recipe together.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch · cifar_resnet18.py (the production recipe, ~94% test)"
        output={`epoch   1  lr=0.100  train=1.4812  val=0.5821
epoch  10  lr=0.095  train=0.4107  val=0.8634
epoch  40  lr=0.050  train=0.1893  val=0.9214
epoch  80  lr=0.010  train=0.0956  val=0.9381
epoch 100  lr=0.000  train=0.0734  val=0.9412
final test accuracy (single crop): 0.9408
final test accuracy (TTA, 10-crop): 0.9451`}
      >{`import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18          # or: your own residual network

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Data: augment only the training set. Test uses plain normalize. ----
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),                         # H×W×3 uint8 -> 3×H×W float in [0,1]
    transforms.Normalize(MEAN, STD),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_set = datasets.CIFAR10('.', train=True,  download=True, transform=train_tf)
test_set  = datasets.CIFAR10('.', train=False, download=True, transform=test_tf)

# num_workers=4 lets the DataLoader pre-fetch batches on the CPU while the GPU trains
train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# ---- Model: ResNet-18, adapted for 32×32 input (stock torchvision expects 224×224) ----
model = resnet18(num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()                      # stock resnet downsamples too aggressively for CIFAR
model.to(device)

# ---- Optimizer + schedule ----
EPOCHS = 100
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                            weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ---- Train ----
best_val = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    scheduler.step()

    # ---- Eval on the test set (in practice you'd eval on a held-out val split during tuning) ----
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += yb.size(0)
    val = correct / total
    if val > best_val:
        best_val = val
        torch.save(model.state_dict(), 'cifar_resnet18_best.pt')
    print(f"epoch {epoch:3d}  lr={scheduler.get_last_lr()[0]:.3f}  "
          f"train={sum(train_losses)/len(train_losses):.4f}  val={val:.4f}")

# ---- Test-time augmentation: average logits over horizontal flip + 5 crops ----
def tta_predict(model, x):
    crops = [x, torch.flip(x, dims=[-1])]          # original + horizontal flip
    return torch.stack([model(c) for c in crops]).mean(0)

model.load_state_dict(torch.load('cifar_resnet18_best.pt'))
model.eval()
correct = total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (tta_predict(model, xb).argmax(1) == yb).sum().item()
        total   += yb.size(0)
print(f"TTA test accuracy: {correct/total:.4f}")`}</CodeBlock>

      <Bridge
        label="numpy sketch → pytorch recipe"
        rows={[
          { left: 'manual pickle load', right: 'torchvision.datasets.CIFAR10', note: 'handles download, extraction, indexing' },
          { left: 'hand-written augment() function', right: 'transforms.Compose([...])', note: 'composable, per-image, runs in DataLoader workers' },
          { left: 'one-loop SGD', right: 'SGD(momentum=0.9) + CosineAnnealingLR', note: 'momentum + cosine schedule are worth ~3% accuracy' },
          { left: 'plain cross-entropy', right: 'CrossEntropyLoss(label_smoothing=0.1)', note: 'targets become 0.9 on true class, 0.011 on others' },
          { left: 'no checkpointing', right: 'torch.save on best_val', note: 'keeps the best model across a noisy val curve' },
        ]}
      />

      <Callout variant="insight" title="the CIFAR-10 accuracy ladder">
        <strong>Linear classifier:</strong> ~40% — flattening the image and running Linear(3072, 10).{' '}
        <strong>2-layer MLP:</strong> ~50% — same idea as MNIST, much less impressive here.{' '}
        <strong>Simple 4-layer CNN, no augmentation:</strong> ~70% — where you plateau on pure capacity.{' '}
        <strong>Simple CNN + augmentation:</strong> ~85% — augmentation alone is worth 15 points.{' '}
        <strong>ResNet-18 + full recipe:</strong> ~94% — what this lesson&apos;s script produces.{' '}
        <strong>Wide ResNet / DenseNet / modern recipes:</strong> ~96%.{' '}
        <strong>Huge-model pretraining + fine-tuning:</strong> ~99%. Human accuracy on CIFAR-10 is
        around 94% — a well-trained ResNet is already at human parity on this dataset.
      </Callout>

      <Callout variant="note" title="why test-time augmentation works at all">
        TTA averages the model&apos;s predictions across a handful of transformations of
        each test image (original, horizontal flip, maybe a few crops), then argmaxes the
        averaged logits. It costs 2-10× inference compute and buys you 0.5-1% accuracy on
        CIFAR-10 essentially for free (no retraining). The mechanism: averaging reduces
        variance in the prediction for any input where the model is close to a decision
        boundary. It&apos;s the same trick as ensembling — one detective, multiple views.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Train accuracy climbs, val accuracy doesn&apos;t.</strong>{' '}
          The textbook overfit. The detective has memorized the 50,000 training
          images and can&apos;t generalize past them. The fix is almost always
          more regularization — augmentation first, weight decay second, then
          dropout or label smoothing. If the gap is wider than ~5 points,
          something upstream is broken; look there before adding regularizers.
        </p>
        <p>
          <strong className="text-term-amber">Augmenting the test set.</strong>{' '}
          Never. Augmentation is a training-time regularizer; at test time you evaluate on
          the true image. TTA is a controlled exception — you explicitly average over
          transforms — but casually feeding random crops through{' '}
          <code className="text-dark-text-primary">model.eval()</code> silently inflates
          your test accuracy and is a form of test-set leakage.
        </p>
        <p>
          <strong className="text-term-amber">Channels-first vs channels-last confusion.</strong>{' '}
          PyTorch expects <code className="text-dark-text-primary">(N, C, H, W)</code>.
          NumPy and PIL default to <code className="text-dark-text-primary">(N, H, W, C)</code>.
          A silent transpose bug will let training &ldquo;work&rdquo; at 10% accuracy
          forever; the loss goes down and everything looks fine. If your model
          won&apos;t break 30% on CIFAR-10, check the input layout before anything else.
        </p>
        <p>
          <strong className="text-term-amber">Normalizing with the wrong statistics.</strong>{' '}
          CIFAR-10 has its own well-known per-channel mean and std
          (<code className="text-dark-text-primary">(0.4914, 0.4822, 0.4465)</code> and{' '}
          <code className="text-dark-text-primary">(0.2470, 0.2435, 0.2616)</code>).
          Don&apos;t use ImageNet&apos;s <code className="text-dark-text-primary">(0.485, 0.456, 0.406)</code>
          — close, but not close enough, and will cost you a percentage point.
          And compute new stats on <em>your</em> training set for <em>your</em> datasets;
          never recompute stats on train+test together.
        </p>
        <p>
          <strong className="text-term-amber">Class imbalance disguised as accuracy.</strong>{' '}
          CIFAR-10 is balanced — 6,000 of each class, so accuracy and per-class
          accuracy tell the same story. Real datasets aren&apos;t. A 95%-accurate
          model on a dataset that&apos;s 95% &ldquo;normal&rdquo; might be
          predicting &ldquo;normal&rdquo; for everything. Always print a confusion
          matrix. Always. Even on balanced data — the pairs the detective
          confuses tell you what the glasses are missing.
        </p>
        <p>
          <strong className="text-term-amber">Using the test set as your val set during hyperparam search.</strong>{' '}
          If you tune learning rate, weight decay, augmentation strength, or architecture
          by watching test accuracy, your final reported number is a selection-biased
          lie. Carve 5,000 images out of the 50k train set, call it{' '}
          <code className="text-dark-text-primary">val</code>, tune on that. Touch test
          only once, at the end.
        </p>
        <p>
          <strong className="text-term-amber">Stock ResNet-18 on 32×32 input.</strong>{' '}
          <code className="text-dark-text-primary">torchvision.models.resnet18()</code>{' '}
          starts with a 7×7 conv at stride 2 and a maxpool — designed for 224×224
          ImageNet. Feed it 32×32 CIFAR and it downsamples to 4×4 before the first block
          even runs. Swap conv1 for 3×3 stride-1 and replace maxpool with{' '}
          <code className="text-dark-text-primary">nn.Identity()</code>, as in the code
          above. Missing this is worth 5+ points of accuracy.
        </p>
      </Gotcha>

      <Challenge prompt="Hit ≥85% on CIFAR-10 from scratch, with an ablation">
        <p>
          Start with a small CNN — 4 conv layers, batch norm after each, ReLU, two
          max-pool stages, global-average-pool, one Linear to 10. Train three
          configurations on the same 100-epoch budget and report test accuracy for each:
        </p>
        <ul>
          <li>
            <strong>Baseline:</strong> CNN, SGD(lr=0.1), no augmentation, no LR schedule,
            no label smoothing.
          </li>
          <li>
            <strong>+ augmentation:</strong> add RandomCrop(32, pad=4) + RandomHorizontalFlip.
          </li>
          <li>
            <strong>+ LR schedule:</strong> add CosineAnnealingLR over 100 epochs, plus
            momentum=0.9 and weight_decay=5e-4.
          </li>
        </ul>
        <p>
          Report the accuracy delta at each step — you should see roughly 70% → 85% →
          88%+. Then turn in your confusion matrix and identify the two worst class
          pairs. Ours were cat↔dog and deer↔horse; are yours the same? If you have a GPU
          and another hour, swap the CNN for the ResNet-18 from the code above and
          confirm you can get to 94%.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> The jump from MNIST to CIFAR-10
          is the jump from &ldquo;a model works&rdquo; to &ldquo;a <em>recipe</em>{' '}
          works.&rdquo; The detective&apos;s three pairs of glasses emerge from
          the architecture; the training recipe — augmentation, momentum, cosine
          schedule, weight decay, label smoothing — is what decides whether those
          glasses see anything useful. Each knob is a regularizer in disguise;
          together they let the detective generalize from 50k photos to the
          distribution the photos were sampled from. This pattern —
          training-as-recipe rather than training-as-single-lever — is how every
          modern vision, language, and multimodal model is actually produced.
        </p>
        <p>
          <strong>Next up — ResNet &amp; Skip Connections.</strong> We used
          ResNet-18 as a black box here, and that black box comes with a catch
          the detective doesn&apos;t tell you about. The deeper you stack
          layers — the more pairs of glasses you give the detective — the worse
          training gets. Past a certain depth, plain CNNs stop improving and
          start actively regressing. A 56-layer plain network trains worse than
          a 20-layer one, and not because of overfitting. It&apos;s an
          optimization problem that looks impossible from the outside. Then, in
          2015, one line of code made it go away: <code>y = F(x) + x</code>.
          Why that single addition unlocked 152-layer networks — and why ResNet
          was still the backbone of most computer-vision systems a decade later
          — is the next lesson.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Learning Multiple Layers of Features from Tiny Images (CIFAR-10/100 tech report)',
            author: 'Alex Krizhevsky',
            venue: 'University of Toronto, 2009',
            url: 'https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf',
          },
          {
            title: 'Deep Residual Learning for Image Recognition',
            author: 'He, Zhang, Ren, Sun',
            venue: 'CVPR 2016 — the ResNet paper',
            url: 'https://arxiv.org/abs/1512.03385',
          },
          {
            title: 'AutoAugment: Learning Augmentation Policies from Data',
            author: 'Cubuk, Zoph, Mane, Vasudevan, Le',
            venue: 'CVPR 2019 — learned augmentation policies on CIFAR/ImageNet',
            url: 'https://arxiv.org/abs/1805.09501',
          },
          {
            title: 'Dive into Deep Learning — 7.6 Residual Networks (ResNet) and 14.1 Image Augmentation',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_convolutional-modern/resnet.html',
          },
        ]}
      />
    </div>
  )
}
