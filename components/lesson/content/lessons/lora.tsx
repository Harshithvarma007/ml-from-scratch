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
import LoRADecomposition from '../widgets/LoRADecomposition'
import LoRATargetSweep from '../widgets/LoRATargetSweep'

// Signature anchor: the sticky-note correction layer. The pretrained weights
// are a 700-page textbook you can't afford to reprint. LoRA pastes tiny
// sticky notes in the margins that override a handful of equations. At
// inference you read the book + the stickies. Low-rank = each sticky is two
// skinny strips (rank-r); their product is the full-size correction.
//
// Returns: opening (reprinting the book = full fine-tuning = insane),
// decomposition reveal (W + BA, where BA has rank r is the sticky), and the
// "how many stickies is enough" rank-selection section.
export default function LoRALesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="lora" />

      {/* ── Opening: the 700-page textbook you can't reprint ─────── */}
      <Prose>
        <p>
          Picture the base model as a 700-page textbook. Every page is packed
          with equations the pretrained weights have memorised — grammar,
          world facts, the shape of code, the rhythm of dialogue. You want to{' '}
          <NeedsBackground slug="supervised-fine-tuning">fine-tune</NeedsBackground>{' '}
          it on your own data. Fine — except fine-tuning in the textbook
          analogy means <em>reprinting the entire book</em>. New ink for
          every page, even the ones you had no quarrel with. For a 70B model
          that&apos;s roughly 280GB of optimizer state — fp16 parameters plus
          two fp32 AdamW moments for every weight. A single H100 has 80GB. A
          consumer RTX 4090 has 24GB. The math does not care what you want.
          Reprinting the book, on your hardware, is not happening.
        </p>
        <p>
          There is a saner move. Leave the textbook alone. Paste a handful of
          tiny sticky notes in the margins that override a handful of the
          equations. At inference, the reader reads the book{' '}
          <em>plus</em> the stickies — and the stickies win wherever they sit.
          You&apos;ve changed the effective contents of the book without
          touching a single printed page. That is the entire pitch behind{' '}
          <KeyTerm>parameter-efficient fine-tuning</KeyTerm>, and the cleanest
          version of it — the one that now runs inside basically every open
          model on HuggingFace — is <KeyTerm>LoRA</KeyTerm>, Low-Rank
          Adaptation, from Hu et al. in 2021.
        </p>
        <p>
          One sentence: instead of updating a{' '}
          <NeedsBackground slug="mlp-from-scratch">weight matrix</NeedsBackground>{' '}
          <code>W</code> directly, learn a small low-rank correction that sits
          in the margin next to it. The book stays frozen. Ninety-nine-plus
          percent of the parameters never move. And you can fine-tune a 70B
          model on a single GPU.
        </p>
        <p>
          This lesson derives the sticky, walks the parameter arithmetic,
          shows why picking which pages to annotate matters as much as how
          big each note is, and builds a <code>LoRALinear</code> module from
          scratch in three layers — pure NumPy, a hand-rolled PyTorch{' '}
          <code>Module</code>, then the one-liner you&apos;d actually use via
          the <code>peft</code> library.
        </p>
      </Prose>

      <Personify speaker="Full fine-tuning">
        I reprint the book. Every page, every equation, fresh ink. For a 70B
        model that means holding 70B parameters in fp16 plus 70B first-moment
        and 70B second-moment estimators in fp32 for AdamW — roughly 280GB
        just to take one step with{' '}
        <NeedsBackground slug="gradient-descent">gradient descent</NeedsBackground>.
        If you can&apos;t afford the printing press, you can&apos;t afford
        me. I am not sorry.
      </Personify>

      {/* ── The LoRA insight — the reveal ────────────────────────── */}
      <Prose>
        <p>
          The observation that makes the sticky-note trick work is empirical.
          When you fine-tune a pretrained language model, the{' '}
          <em>correction</em> you&apos;d apply to each weight matrix — call
          it <code>ΔW</code> — turns out to be very close to low-rank. The
          textbook has already learned most of what it needs during
          pretraining; fine-tuning is a small, structured override on top.
          Aghajanyan et al. (2020) called this the{' '}
          <KeyTerm>intrinsic dimensionality</KeyTerm> of fine-tuning and
          showed you could adapt a BERT model by tuning as few as 200 scalar
          parameters along the right direction. That&apos;s not a margin
          note. That&apos;s a Post-it.
        </p>
        <p>
          So here&apos;s the LoRA decomposition — the sticky, opened up. A
          weight matrix <code>W</code> is <code>d × d</code>. A full-rank
          correction would also be <code>d × d</code>, <code>d²</code>{' '}
          trainable scalars per matrix, a fresh page of ink per matrix. Don&apos;t
          do that. Write the sticky as two skinny strips — one tall, one wide
          — whose <em>product</em> is the full-size correction:
        </p>
      </Prose>

      <MathBlock caption="LoRA decomposition — the sticky is two skinny strips">
{`W_effective   =   W₀        +        B · A
                ─────────         ─────────────
               frozen book         sticky note
                                  (two strips)

where        A ∈ ℝ^(r × d)      ← one thin horizontal strip
             B ∈ ℝ^(d × r)      ← one thin vertical strip
             r ≪ d              typically r ∈ {4, 8, 16, 32, 64}`}
      </MathBlock>

      <Prose>
        <p>
          <code>W₀</code> is the original pretrained weight — the printed
          page. You freeze it, you never touch it. The only things that move
          during training are the two strips <code>A</code> and <code>B</code>.
          Their product <code>B · A</code> is a <code>d × d</code> matrix
          just like <code>ΔW</code> would have been — same shape as the page
          it overrides — but by construction it has rank at most{' '}
          <code>r</code>. You&apos;ve swapped a full-page rewrite for a
          margin note whose information content is deliberately bottlenecked.
          The savings are enormous.
        </p>
      </Prose>

      {/* ── Widget 1: LoRA Decomposition ─────────────────────────── */}
      <LoRADecomposition />

      <Prose>
        <p>
          Drag the rank slider and watch the two strips grow. At{' '}
          <code>r = d</code> the sticky is the same size as the page —
          no savings, you&apos;ve just reinvented full fine-tuning with extra
          steps. At <code>r = 1</code> the sticky collapses to a single
          outer product — one column times one row — and the parameter count
          is <code>2d</code>. Real LoRA sits in the middle: small enough to
          be cheap, wide enough to carry the override.
        </p>
      </Prose>

      <Personify speaker="Low-rank update (B·A)">
        I am the nudge you&apos;d have learned anyway. Your task is easy
        relative to what the textbook already knows — a correction, not a
        rewrite. I can fit that correction into a few million parameters
        instead of a few billion. At inference the reader sees book plus
        sticky — same shape, same behavior. You just paid 125× less to get
        here.
      </Personify>

      {/* ── Parameter count math ─────────────────────────────────── */}
      <Prose>
        <p>
          Let&apos;s do the arithmetic on a real page. A Llama-2 7B has a
          hidden size of <code>d = 4096</code>. One attention weight matrix
          in that model has:
        </p>
      </Prose>

      <MathBlock caption="book vs sticky — the savings are not subtle">
{`reprint the page:   d × d      =   4096 × 4096    =   16,777,216   params

sticky (r = 16):    2 · r · d  =   2 · 16 · 4096  =      131,072   params

                                                     ─────────────
ratio:              16,777,216 / 131,072            =     128 × fewer`}
      </MathBlock>

      <Prose>
        <p>
          One hundred and twenty-eight times fewer trainable scalars{' '}
          <em>per sticky</em>. A 7B model has hundreds of weight matrices
          across its attention and MLP layers, and LoRA typically targets a
          subset of them. In practice you end up training something like
          0.1% – 1% of the original parameter count. The optimizer state —
          the thing that actually kills you during full fine-tuning — shrinks
          by the same factor. Suddenly the 280GB beast fits in a consumer
          GPU.
        </p>
        <p>
          Here&apos;s where it gets more interesting. You don&apos;t have to
          stick a note on every page — you choose which pages to annotate.
          Attention has four projections per layer (<code>q_proj</code>,{' '}
          <code>k_proj</code>, <code>v_proj</code>, <code>o_proj</code>) and
          the MLP has three more (<code>gate</code>, <code>up</code>,{' '}
          <code>down</code> in a SwiGLU block). Each target you mark costs
          parameters and adds override capacity. Toggle them below and watch
          the tradeoff.
        </p>
      </Prose>

      {/* ── Widget 2: Target Sweep ───────────────────────────────── */}
      <LoRATargetSweep />

      <Prose>
        <p>
          The original LoRA paper only annotated <code>q_proj</code> and{' '}
          <code>v_proj</code> — query and value — and got nearly full
          fine-tuning quality. That convention stuck for years. More recent
          work (QLoRA, and the HuggingFace PEFT defaults) sticks a note on{' '}
          <em>every</em> linear layer in the transformer, MLP included. It
          costs more parameters but reliably picks up a couple of points on
          harder benchmarks. Rule of thumb: start with attention-only at{' '}
          <code>r = 16</code>; if the task is underfit, extend the stickies
          to the MLP before cranking rank higher.
        </p>
      </Prose>

      {/* ── How many stickies is enough (rank-selection beat) ───── */}
      <Callout variant="insight" title="how many stickies is enough">
        The rank <code>r</code> is the width of each margin strip — how much
        information a single sticky can carry. Set it too low and the
        override is starved: the note can&apos;t express what your task
        needs, and fine-tuning underfits. Set it too high and you&apos;re
        paying for ink you never use — the sticky has headroom it never
        fills, because the intrinsic dimension of the correction was always
        smaller than you gave it. The quality curve against rank is
        pleasingly flat past the elbow. For most instruction-tuning tasks
        the elbow lives near <code>r = 8</code>–<code>16</code>. Doubling
        past that is mostly accounting, not learning.
      </Callout>

      <Personify speaker="Rank r">
        I&apos;m the width of the sticky. Set me to 4 and you get an
        aggressive override — fast, tiny notes, fine for style transfer or
        simple instruction-following. Crank me to 64 and I approach
        full-reprint quality at 2% of the cost. Most people leave me at 8 or
        16 and never look back. Doubling me doubles the trainable strip
        area, but the quality curve flattens fast. More rank is not more
        better.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, one module. Start with the forward pass in pure
          NumPy so you can see every matrix multiply that reads the book and
          applies the sticky; then wrap it in a PyTorch{' '}
          <code>nn.Module</code> that replaces <code>nn.Linear</code>{' '}
          one-for-one; then swap the whole thing for <code>peft</code>, which
          sticks the notes for you.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure numpy · lora_scratch.py"
        output={`W0 output shape:    (batch=4, d=128)
LoRA output shape:  (batch=4, d=128)
trainable params:   A=(8,128)=1024, B=(128,8)=1024  →  2048 (vs full 16384)
scale α/r = 2.0 — the knob that controls how loud the sticky is`}
      >{`import numpy as np

np.random.seed(0)
d, r, batch = 128, 8, 4
alpha = 16                                      # LoRA scaling hyperparameter

# The frozen book — pretrained weight we never reprint
W0 = np.random.randn(d, d) * 0.02

# The sticky note, written as two skinny strips
# A is Gaussian, B is zeros so the sticky starts at EXACTLY zero override
A = np.random.randn(r, d) * 0.01                # (r, d) — horizontal strip
B = np.zeros((d, r))                            # (d, r) — vertical strip, zero-init

x = np.random.randn(batch, d)                   # inputs

# Plain book output:   y₀ = x W₀ᵀ
# Book + sticky:       y  = x W₀ᵀ + (α/r) · x Aᵀ Bᵀ
y0      = x @ W0.T
sticky  = (alpha / r) * (x @ A.T) @ B.T         # the low-rank override
y       = y0 + sticky

print("W0 output shape:   ", y0.shape)
print("LoRA output shape: ", y.shape)
print(f"trainable params:   A={A.shape}={A.size}, B={B.shape}={B.size}  →  {A.size + B.size} (vs full {d*d})")
print(f"scale α/r = {alpha/r} — the knob that controls how loud the sticky is")`}</CodeBlock>

      <Prose>
        <p>
          Three things to notice. <strong>A is Gaussian, B is zero.</strong>{' '}
          That init is on purpose — at step zero, <code>B · A = 0</code>, so
          the sticky is blank and the annotated model is bit-identical to the
          base book. You start from exactly the pretrained behavior and move
          away from it. Initialise both Gaussian and the sticky arrives
          pre-scribbled with random noise; the first forward pass corrupts
          the book&apos;s output and training spends hundreds of steps
          clawing back what you broke.
        </p>
        <p>
          <strong>The scale factor <code>α/r</code>.</strong> This is the
          LoRA paper&apos;s <code>α</code> hyperparameter divided by rank.
          It keeps the effective volume of the sticky roughly constant as
          you change <code>r</code>, so you don&apos;t have to re-tune
          learning rate when you sweep rank. Most configs use{' '}
          <code>α = 2r</code>, which makes the scale a clean <code>2</code>.
        </p>
        <p>
          <strong>We compute <code>x @ A.T @ B.T</code>, never form{' '}
          <code>B @ A</code> directly.</strong> Forming the <code>d × d</code>{' '}
          product would defeat the whole point — you&apos;d materialise the
          full-size correction in memory and have accidentally reprinted the
          page. Left-to-right through the low-rank bottleneck keeps memory
          at <code>O(batch · r)</code>. The strip stays a strip.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — hand-rolled pytorch · lora_module.py"
        output={`LoRALinear(d_in=4096, d_out=4096, r=16)
  trainable: 131072 / 16908288 (0.78%)
loss before step: 1.2418  |  after:  1.2006`}
      >{`import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear: frozen book + trainable sticky."""
    def __init__(self, in_features, out_features, r=8, alpha=16, bias=True):
        super().__init__()
        self.r, self.alpha, self.scale = r, alpha, alpha / r

        # The book — frozen page, never reprinted
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        for p in self.linear.parameters():
            p.requires_grad = False              # FREEZE — this is the point

        # The sticky — two trainable strips
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))         # zero-init B
        nn.init.kaiming_uniform_(self.A, a=5**0.5)                  # Gaussian-ish A

    def forward(self, x):
        return self.linear(x) + self.scale * (x @ self.A.T) @ self.B.T

# ---- use it ----
layer = LoRALinear(4096, 4096, r=16, alpha=32)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total     = sum(p.numel() for p in layer.parameters())
print(f"LoRALinear(d_in=4096, d_out=4096, r=16)")
print(f"  trainable: {trainable} / {total} ({100*trainable/total:.2f}%)")

# one fake training step
x = torch.randn(8, 4096)
target = torch.randn(8, 4096)
opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=1e-3)
loss0 = ((layer(x) - target)**2).mean(); loss0.backward(); opt.step()
with torch.no_grad():
    loss1 = ((layer(x) - target)**2).mean()
print(f"loss before step: {loss0.item():.4f}  |  after:  {loss1.item():.4f}")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch LoRALinear"
        rows={[
          {
            left: 'W0 = np.random.randn(d, d) * 0.02',
            right: 'self.linear = nn.Linear(...); freeze()',
            note: 'the frozen book — requires_grad=False on every page',
          },
          {
            left: 'A = np.random.randn(r, d), B = np.zeros',
            right: 'nn.Parameter + kaiming_uniform / zeros',
            note: 'same two strips, now tracked by autograd',
          },
          {
            left: 'y = x @ W0.T + (α/r) * x @ A.T @ B.T',
            right: 'self.linear(x) + self.scale * (x @ A.T) @ B.T',
            note: 'book + sticky — parens are load-bearing (rank bottleneck)',
          },
        ]}
      />

      <Prose>
        <p>
          In real life you don&apos;t hand-write the sticky holder.
          HuggingFace&apos;s <code>peft</code> library walks the model graph,
          pattern-matches module names, and swaps in LoRA wrappers at runtime
          — automated sticky placement. You declare the config, call{' '}
          <code>get_peft_model</code>, train normally.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — production · lora_peft.py"
        output={`trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062`}
      >{`from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                                         # rank — width of each sticky
    lora_alpha=32,                                # α = 2r keeps scale = 2
    lora_dropout=0.05,                            # small dropout on the sticky path
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",                                  # don't also train biases
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062

# From here, train with any Trainer / custom loop. At the end,
# model.save_pretrained("adapter/") writes ONLY the A and B strips —
# the book stays on disk once, shared across any number of stickies.`}</CodeBlock>

      <Bridge
        label="hand-rolled → peft"
        rows={[
          {
            left: 'LoRALinear(q_proj); LoRALinear(v_proj); ...',
            right: 'LoraConfig(target_modules=[...])',
            note: 'peft walks the graph and sticks a note on each matched page',
          },
          {
            left: 'for p in base.parameters(): p.requires_grad=False',
            right: 'get_peft_model(model, config)',
            note: 'freezing the book + placing the stickies in one call',
          },
          {
            left: 'torch.save({"A": ..., "B": ...})',
            right: 'model.save_pretrained("adapter/")',
            note: 'the stickies ship as a ~20MB file — the book stays home',
          },
        ]}
      />

      <Callout variant="insight" title="the point of the three layers">
        Layer 1 shows you the sticky is five lines of matrix algebra — a
        book read and an override added. Layer 2 shows you that a
        LoRALinear module is a twenty-line drop-in for nn.Linear. Layer 3
        shows you that in practice it&apos;s a config object. When a paper
        reports results with &ldquo;LoRA r=16 on q and v,&rdquo; you now
        know exactly which pages got stickied, how wide each strip is, and
        what that implies in trainable parameters, memory, and
        representational capacity.
      </Callout>

      {/* ── The merge trick ─────────────────────────────────────── */}
      <Callout variant="note" title="the merge trick — press the sticky into the page">
        At training time you carry <code>W₀</code>, <code>A</code>, and{' '}
        <code>B</code> separately — book and sticky live apart. At inference
        you can optionally <em>merge</em> them:{' '}
        <code>W_merged = W₀ + (α/r) · B · A</code>. The override gets pressed
        into the page and you&apos;re left with a regular transformer at a
        slightly different weight matrix — same FLOPs, same latency, zero
        extra parameters. PEFT exposes this as{' '}
        <code>model.merge_and_unload()</code>. Production teams almost
        always merge before shipping.
      </Callout>

      <Callout variant="insight" title="sticky stacking — one book, many notes">
        Because the stickies are ~20MB files, you can keep dozens of them
        for a single book — one per user, one per task, one per domain. At
        serve time you load the book once into GPU memory and hot-swap
        sticky sets (or, with tricks like S-LoRA, serve many concurrently).
        This is how companies like Predibase and Together serve thousands of
        fine-tuned models from a handful of base checkpoints. Full
        fine-tuning cannot do this — every variant would be a 14GB blob.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">r = 0 or r = d:</strong> both
          are pathological. <code className="text-dark-text-primary">r = 0</code>{' '}
          means no sticky at all — you&apos;re just reading the book.{' '}
          <code className="text-dark-text-primary">r = d</code> means the
          sticky is the size of the page — no savings, you&apos;ve just
          added two strips whose product equals a full reprint; use full
          fine-tune instead.
        </p>
        <p>
          <strong className="text-term-amber">Initialising both A and B as Gaussian:</strong>{' '}
          one of the most common bugs in hand-rolled implementations. If{' '}
          <code>B</code> is not zero, the sticky arrives pre-scribbled with
          random noise, the first forward pass corrupts the book&apos;s
          output, the loss spikes, and training is unstable for hundreds of
          steps. Zero-init <code>B</code> means a blank sticky — you start
          exactly at the pretrained model.
        </p>
        <p>
          <strong className="text-term-amber">Shipping without merging:</strong>{' '}
          keeping <code className="text-dark-text-primary">W₀</code> and{' '}
          <code className="text-dark-text-primary">B · A</code> separate at
          inference means two matmuls per layer — the reader opens the book
          and then reads the sticky. Roughly 1.2–1.5× latency for no reason.
          Always <code className="text-dark-text-primary">merge_and_unload()</code>{' '}
          before production-serving — unless you genuinely need per-request
          sticky swapping.
        </p>
        <p>
          <strong className="text-term-amber">Same rank across every page:</strong>{' '}
          the default, but not always optimal. Lower layers (early in the
          network) often need less override than upper layers for downstream
          tasks; recent work uses wider{' '}
          <code className="text-dark-text-primary">r</code> stickies in the
          top third of the book. If you&apos;re bottlenecked on quality and
          have already tried larger uniform rank, try non-uniform.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Sweep rank on Llama-7B and find the elbow">
        <p>
          Grab a small instruction dataset (<code>tatsu-lab/alpaca</code>,
          52k examples, is the canonical toy benchmark). Load Llama-2-7B in
          bf16 with <code>peft</code> and fine-tune with LoRA at{' '}
          <code>r ∈ &#123;4, 8, 16, 32&#125;</code>, keeping everything else
          fixed (<code>α = 2r</code>, learning rate <code>2e-4</code>, 1
          epoch, stickies on q and v only).
        </p>
        <p className="mt-2">
          Log the final eval loss for each. Plot loss vs <code>r</code>. You
          should see a clear elbow somewhere between 8 and 16 — past the
          elbow, doubling the sticky width barely moves loss. That elbow is
          the intrinsic dimensionality of your task showing up in your own
          training run.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: at your best <code>r</code>, extend{' '}
          <code>target_modules</code> to include the MLP (<code>gate_proj</code>,{' '}
          <code>up_proj</code>, <code>down_proj</code>) and rerun. How much
          does sticking every page buy you over attention-only, and is the
          extra parameter cost worth it?
        </p>
      </Challenge>

      {/* ── Takeaways + cliffhanger ─────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Fine-tuning updates are
          empirically low-rank, so we don&apos;t have to reprint the book.
          LoRA freezes the base, learns two skinny strips <code>A</code> and{' '}
          <code>B</code> whose product <em>is</em> the margin note, and
          captures 95–100% of full fine-tuning quality at well under 1% of
          the parameter cost. The two knobs you pick are the sticky&apos;s
          width (<em>rank</em>) and which pages get annotated (<em>target
          modules</em>). Initialise <code>A</code> Gaussian and <code>B</code>{' '}
          zero so the sticky starts blank, use scale <code>α/r</code>, press
          the sticky into the page before production unless you need
          hot-swap.
        </p>
        <p>
          <strong>Next up — <KeyTerm>qlora</KeyTerm>.</strong> LoRA shrank
          the optimizer state and the gradient storage — the sticky is tiny.
          But the book itself — <code>W₀</code>, the frozen base — is still
          sitting in memory in fp16, and for a 70B model that&apos;s 140GB
          of printed ink. QLoRA (Dettmers 2023) does one more thing:
          quantise the frozen book down to 4 bits per weight — compress
          the page without losing what it says. The base now fits in ~35GB,
          the LoRA stickies train on top, and you can fine-tune Llama-70B
          on a single 48GB GPU. It is, unreasonably, the state of the art
          for accessible fine-tuning.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'LoRA: Low-Rank Adaptation of Large Language Models',
            author: 'Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen',
            venue: 'arXiv 2021 — the original LoRA paper',
            url: 'https://arxiv.org/abs/2106.09685',
          },
          {
            title: 'Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning',
            author: 'Aghajanyan, Zettlemoyer, Gupta',
            venue: 'ACL 2021 — why fine-tuning updates are low-rank',
            url: 'https://arxiv.org/abs/2012.13255',
          },
          {
            title: 'QLoRA: Efficient Finetuning of Quantized LLMs',
            author: 'Dettmers, Pagnoni, Holtzman, Zettlemoyer',
            venue: 'NeurIPS 2023',
            url: 'https://arxiv.org/abs/2305.14314',
          },
          {
            title: 'PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods',
            author: 'HuggingFace',
            venue: 'library — LoraConfig, get_peft_model, merge_and_unload',
            url: 'https://github.com/huggingface/peft',
          },
          {
            title: 'S-LoRA: Serving Thousands of Concurrent LoRA Adapters',
            author: 'Sheng, Cao, Li, Zhu, Zheng, Gonzalez, Stoica',
            venue: 'MLSys 2024',
            url: 'https://arxiv.org/abs/2311.03285',
          },
        ]}
      />
    </div>
  )
}
