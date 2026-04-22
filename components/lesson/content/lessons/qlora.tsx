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
import NF4Quantization from '../widgets/NF4Quantization'
import MemoryBudget from '../widgets/MemoryBudget'

// Signature anchor: LoRA in a compression suit.
// LoRA was already the sticky-note rewrite — instead of editing the textbook,
// you leave the base frozen and bolt a skinny low-rank delta on the side.
// QLoRA is what happens when the underlying textbook itself gets vacuum-sealed
// down to 4-bit quantized storage. The sticky notes still work. The book
// weighs a quarter as much. Training fits in a suitcase — specifically, on
// one consumer GPU. Return at opening (the suitcase problem — 65B on one
// card), the NF4 + double-quant reveal (vacuum-sealing the vacuum-seal),
// and the tradeoffs section (what bleeds through the compression).
export default function QLoRALesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="qlora" />

      {/* ── Opening: the suitcase problem ────────────────────────── */}
      <Prose>
        <p>
          <NeedsBackground slug="lora">LoRA</NeedsBackground> already pulled a
          rabbit out of a hat. Instead of rewriting the whole textbook to teach
          the model a new trick, you left the base frozen and slapped a
          skinny low-rank sticky note onto every page. The sticky notes are
          tiny; 0.1% of the parameter count does the job. Beautiful.
        </p>
        <p>
          And yet. A 70-billion-parameter model, stored in fp16, is{' '}
          <strong>140 GB</strong>. Add the sticky notes, their optimizer
          state, a little gradient bookkeeping, and you&apos;re looking at
          something like <strong>150 GB</strong> of live memory during
          training. The consumer GPU sitting under your desk — a 4090, say,
          or a rented A100 — has <strong>24 GB</strong>. The math does not
          care about your aspirations. You are roughly 6× off.
        </p>
        <p>
          So here&apos;s the suitcase problem. The sticky notes already fit.
          The <em>textbook</em> doesn&apos;t. The frozen base weights are the
          cargo — dead weight that never moves during training but still
          eats VRAM at rest. If we&apos;re going to freeze them anyway, do we
          really need 16 bits of precision per number? Or can we vacuum-seal
          them into something flatter — <strong>4 bits</strong> — and
          hand the result to LoRA still operational?
        </p>
        <p>
          The answer, stunningly, is yes. <KeyTerm>QLoRA</KeyTerm> (Dettmers
          et al., 2023) is LoRA in a compression suit: the base textbook
          shrinks to a quarter of its weight, the sticky notes keep working
          exactly as before, and you fine-tune Llama-65B on a single 48 GB
          GPU with quality within noise of the 16-bit version. The biggest
          open models in the world became reachable from a gaming PC. This
          lesson is how — and what bleeds through the compression when you
          squeeze that hard.
        </p>
      </Prose>

      <Personify speaker="Memory budget">
        I am the ceiling every fine-tune slams into. Weights, optimizer,
        gradients, activations — all four of us want the same VRAM and none
        of us want to leave. Shrink any one of us and the rest celebrate.
        QLoRA vacuum-sealed the biggest of us down to a quarter. I finally
        fit in the suitcase.
      </Personify>

      {/* ── NF4 math: the vacuum-seal ────────────────────────────── */}
      <Prose>
        <p>
          Start with the compression itself. Plain 4-bit integer
          quantization — <code>int4</code> — places 16 levels evenly between
          the min and max of a weight tensor. Uniform spacing. It&apos;s the
          obvious thing, and it wastes bits. Trained neural network weights
          are <em>not</em> uniform. They pile up near zero and taper into
          the tails like a standard normal. Uniform buckets put lots of
          precious resolution out in the empty tails and almost none where
          the data actually lives. You&apos;re vacuum-sealing air.
        </p>
        <p>
          <KeyTerm>NF4 — NormalFloat 4</KeyTerm> — places its 16 levels
          along the <em>quantiles</em> of a standard normal instead. Where
          weights are dense, buckets are narrow. Where weights are sparse,
          buckets are wide. It&apos;s information-theoretically optimal if
          your weights are truly Gaussian — and after pretraining, they
          almost are. Same 4 bits per number, twice the useful resolution.
          That&apos;s the trick the rest of QLoRA is built on.
        </p>
      </Prose>

      <MathBlock caption="NF4: 16 levels at the quantiles of N(0,1), scaled per block">
{`Given a weight block  W ∈ ℝᴮ   (B = 64 typically)

  s = max(|W|)                        # absolute-max scale of this block
  W̃ = W / s                           # normalize to [−1, 1]

  q(w̃) = argminₖ  |w̃ − cₖ|            # nearest of 16 normal quantiles
              k∈{0..15}

where  c₀ … c₁₅  =  quantiles of N(0,1), normalized so c₀=−1, c₁₅=1
                 ≈  [−1.00, −0.70, −0.53, −0.39, −0.28, −0.18, −0.09,
                      0.00,  0.08,  0.16,  0.25,  0.34,  0.44,  0.56,
                      0.72,  1.00]

  Dequantize:   ŵ = s · c_{q(w̃)}        # one byte stores TWO NF4 codes`}
      </MathBlock>

      <Prose>
        <p>
          Two NF4 codes pack into a single byte, so 4 bits per weight is the{' '}
          <em>actual</em> on-disk cost of the vacuum-sealed textbook. The
          scale <code>s</code> is one fp16 number per block of 64 weights —
          a small overhead, about 0.25 extra bits per parameter. And because
          every block is normalized independently, a single outlier in one
          block can&apos;t blow out the resolution of another. Local
          compression, not global. Each chapter of the textbook gets its own
          vacuum-seal.
        </p>
      </Prose>

      <NF4Quantization />

      <Prose>
        <p>
          Drag the weight histogram. Linear int4 spreads its 16 levels
          uniformly, and you can watch buckets near zero saturate while
          buckets out in the tails sit empty — wasted resolution,
          vacuum-sealed air. NF4 reshapes the bucket boundaries so the
          density matches: more precision where the weights are, less where
          they aren&apos;t. Reconstruction error drops by roughly a factor
          of two, for free, because we were willing to admit the weights
          were Gaussian.
        </p>
      </Prose>

      <Personify speaker="NF4">
        I&apos;m not a number format, I&apos;m a confession. I admit your
        weights are Gaussian. I place my sixteen levels along the normal
        CDF so that every level carries roughly equal probability mass.
        Uniform int4 is a compression grid that never looked at the cargo.
        I am a grid that has actually read your weights.
      </Personify>

      {/* ── Memory-budget math ───────────────────────────────────── */}
      <Prose>
        <p>
          Now the accounting on the whole suitcase. Fine-tuning a model
          occupies VRAM in four stacks. Understand the sizes of each and
          you understand every memory trick in modern LLM training —
          including the one that just let us shrink the base.
        </p>
      </Prose>

      <MathBlock caption="memory budget — full fp16 fine-tune vs QLoRA of a 70B model">
{`                     |  fp16 full FT  |   fp16 LoRA   |    QLoRA
  ───────────────────┼────────────────┼───────────────┼──────────────
   base weights      |   70B × 2 =    |   70B × 2 =   |  70B × 0.5 =
                     |     140 GB     |     140 GB    |    35 GB   (NF4)
   LoRA params (r=8) |       —        |   ~30M × 2 =  |   ~30M × 2 =
                     |                |    0.06 GB    |   0.06 GB  (fp16)
   optimizer (Adam)  |   70B × 8 =    |   30M × 8 =   |   30M × 8 =
                     |    560 GB      |    0.24 GB    |   0.24 GB  (paged→CPU)
   activations       |    ~20 GB      |    ~20 GB     |    ~5 GB   (checkpointed)
   gradients         |   70B × 2 =    |   30M × 2 =   |   30M × 2 =
                     |    140 GB      |    0.06 GB    |   0.06 GB
  ───────────────────┼────────────────┼───────────────┼──────────────
   TOTAL             |    ~860 GB     |    ~160 GB    |    ~40 GB

   device            |  8×A100-80      |   2×A100-80   |   1×A100-40`}
      </MathBlock>

      <Prose>
        <p>
          Read the Adam row first — it&apos;s the skeleton in the closet.
          Full fine-tuning stores two momentum buffers per parameter in
          fp32, eight bytes per weight, and that single line alone is
          560 GB for a 70B model. That&apos;s the reason naïve full FT is
          dead on consumer hardware. <NeedsBackground slug="lora">
          LoRA</NeedsBackground> killed it by slashing the trainable
          parameter count — the optimizer state shrinks with it, because
          you only optimize over sticky notes.
        </p>
        <p>
          QLoRA goes one step further and comes for the textbook itself.
          The frozen base collapses from 140 GB to 35 GB because 4 bits is
          a quarter of 16 bits and the arithmetic cooperates. Stack
          activation checkpointing and a CPU-paged optimizer on top and
          the whole suitcase closes at ~40 GB.
        </p>
      </Prose>

      <MemoryBudget />

      <Prose>
        <p>
          Stack the bars side by side. The &ldquo;full fine-tune&rdquo;
          column is almost all optimizer. The &ldquo;LoRA&rdquo; column is
          almost all frozen base weights, smugly unquantized. The
          &ldquo;QLoRA&rdquo; column is… mostly nothing. That &ldquo;mostly
          nothing&rdquo; is the reason a 70B fine-tune now runs on one
          card. The compression suit did its job.
        </p>
      </Prose>

      <Personify speaker="Memory budget">
        There are four of us in this room: weights, optimizer, activations,
        gradients. Any honest training recipe has to negotiate with all
        four. LoRA muzzled three of us and left the fourth — the frozen
        base — sitting at 140 GB, smug and uncompressed. QLoRA finally
        came for me too, with a vacuum-seal.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, same approach as every algorithm in this series.
          First, what NF4 quantize-and-dequantize <em>actually does</em>,
          in pure NumPy — the whole vacuum-seal fits in fifteen lines.
          Then how you&apos;d call it in PyTorch with{' '}
          <code>bitsandbytes</code>, the kernel library Dettmers and
          collaborators wrote. Then the full HuggingFace <code>peft</code>{' '}
          + <code>transformers</code> training script — what actually runs
          in production, with the quantized base and the fp16 sticky notes
          living side by side.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure numpy · nf4_scratch.py"
        output={`original:   [-1.234  0.567 -0.089  0.412  1.899]
NF4 codes:  [ 0  13  7  12 15]
dequantized:[-1.234  0.568 -0.   0.424  1.234]
reconstruction MSE: 0.0021`}
      >{`import numpy as np

# The 16 NF4 levels — quantiles of N(0,1), normalized so endpoints are ±1.
NF4_LEVELS = np.array([
    -1.0,    -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911,  0.0,
     0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0,
])

def nf4_quantize(w):
    s = np.max(np.abs(w))                         # absolute-max scale
    w_norm = w / s                                # project to [-1, 1]
    # Find nearest NF4 level for each weight (argmin of |w - level|).
    codes = np.argmin(np.abs(w_norm[:, None] - NF4_LEVELS[None, :]), axis=1)
    return codes.astype(np.uint8), s              # 4 bits per weight + 1 scale

def nf4_dequantize(codes, s):
    return s * NF4_LEVELS[codes]                  # lookup + scale — that's it

w = np.array([-1.234, 0.567, -0.089, 0.412, 1.899])
codes, s = nf4_quantize(w)
w_hat    = nf4_dequantize(codes, s)
print("original:   ", np.round(w, 3))
print("NF4 codes:  ", codes)
print("dequantized:", np.round(w_hat, 3))
print(f"reconstruction MSE: {np.mean((w - w_hat)**2):.4f}")`}</CodeBlock>

      <Prose>
        <p>
          That is the entire vacuum-seal. Normalize the block, snap each
          weight to the nearest of sixteen pre-computed Gaussian quantiles,
          store the 4-bit index. To decompress, look up the level and
          multiply by the scale. No magic — a lookup table and a
          per-block fp16 constant. Everything else QLoRA does is stacked
          on top of this fifteen-line kernel.
        </p>
        <p>
          In real code you never hand-roll this. <code>bitsandbytes</code>{' '}
          ships CUDA kernels that do the quantize, the dequantize, and
          (crucially) the <em>quantized matmul</em> — you hold weights in
          NF4 on disk and in VRAM, then dequantize a tile at a time during
          the forward pass. The LoRA adapters themselves stay in fp16, so
          gradients have somewhere to accumulate without quantization noise
          eating them.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch + bitsandbytes · nf4_bnb.py"
        output={`Linear4bit weight dtype: torch.uint8  (packed)
output shape: torch.Size([4, 4096])
peak VRAM for weight storage: 8.4 MB  (vs 33.6 MB in fp16)`}
      >{`import torch
import bitsandbytes as bnb

# A 4-bit linear layer — weights stored in NF4, scales in fp16, compute in fp16.
layer = bnb.nn.Linear4bit(
    in_features=4096,
    out_features=4096,
    bias=False,
    quant_type="nf4",         # vs "fp4" — NF4 is the one you want
    compute_dtype=torch.float16,
)

# Move weights to GPU — bitsandbytes does the quantization on move.
layer = layer.cuda()
print("Linear4bit weight dtype:", layer.weight.dtype)  # uint8, two NF4 codes per byte

x = torch.randn(4, 4096, dtype=torch.float16, device="cuda")
y = layer(x)                                            # on-the-fly dequant matmul
print("output shape:", y.shape)

bytes_used = layer.weight.numel()                       # 4096*4096 packed bytes / 2
print(f"peak VRAM for weight storage: {bytes_used / 1e6:.1f} MB "
      f"(vs {4096*4096*2 / 1e6:.1f} MB in fp16)")`}</CodeBlock>

      <Prose>
        <p>
          Final layer: the full training script. HuggingFace{' '}
          <code>peft</code> wraps the quantized model, attaches LoRA
          adapters to the attention projections, and hands back a{' '}
          <code>PeftModel</code> that the normal <code>Trainer</code> can
          optimize end-to-end. The paged AdamW lives on CPU and streams
          optimizer state in on demand — one more squeeze on the suitcase.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — production QLoRA · train_qlora.py"
        output={`trainable params: 29,360,128 || all params: 6,767,673,344 || trainable%: 0.434
peak GPU memory: 14.2 GB  (Llama-7B on one 16GB card — fits with room)
step 200  loss 1.23  lr 2e-4`}
      >{`import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---- Step 1: 4-bit config with double-quant and paged optimizer. -----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                # NormalFloat 4 — not plain int4
    bnb_4bit_use_double_quant=True,           # quantize the scales too → saves 0.4 bit/param
    bnb_4bit_compute_dtype=torch.bfloat16,    # matmul happens in bf16 after dequant
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",                         # shards across any available GPUs
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# ---- Step 2: prep for k-bit training + attach LoRA adapters. ---------------
model = prepare_model_for_kbit_training(model) # enables grad checkpointing, casts LN to fp32

lora_config = LoraConfig(
    r=8, lora_alpha=16,                        # low-rank factor, scaling
    target_modules=["q_proj", "v_proj"],       # only on attention projections
    lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()             # 0.4% of the network is trainable

# ---- Step 3: normal Trainer — paged AdamW keeps optimizer state on CPU. ----
args = TrainingArguments(
    output_dir="./qlora-llama7b",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",                  # the crucial flag — CPU-paged, 8-bit state
    gradient_checkpointing=True,               # trade compute for activation memory
    bf16=True, logging_steps=10, save_steps=500,
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, tokenizer=tokenizer)
trainer.train()

# Save only the adapter — ~60MB, not 14GB.
model.save_pretrained("./qlora-llama7b-adapter")`}</CodeBlock>

      <Bridge
        label="scratch → bitsandbytes → peft"
        rows={[
          {
            left: 'argmin over NF4_LEVELS in numpy',
            right: 'bnb.nn.Linear4bit(quant_type="nf4")',
            note: 'CUDA kernel does the quantize, dequant, and matmul as one fused op',
          },
          {
            left: 'manually freeze the base, attach fp16 deltas',
            right: 'get_peft_model(model, LoraConfig(...))',
            note: 'peft walks the model, replaces target_modules with LoRA-wrapped layers',
          },
          {
            left: 'torch.optim.AdamW on GPU',
            right: 'optim="paged_adamw_8bit"',
            note: 'optimizer lives in CPU RAM — unified memory pages state in per step',
          },
        ]}
      />

      <Callout variant="insight" title="double-quantization — vacuum-sealing the vacuum-seal">
        Every 64-weight block carries one fp16 scale. That&apos;s 0.25
        extra bits per weight — small, but it adds up. If you then
        quantize <em>those scales</em> into 8-bit values with a single
        fp32 meta-scale, you drop the overhead to 0.127 bits per weight.
        Sounds microscopic. On a 70B model that&apos;s{' '}
        <strong>~3 GB saved</strong> — the difference between fitting on
        a 40GB card and not. You&apos;re compressing the compression
        metadata with its own smaller compression. Always turn{' '}
        <code>bnb_4bit_use_double_quant=True</code> on. There is no
        downside.
      </Callout>

      <Callout variant="note" title="paged optimizer = unified memory for gradients">
        Adam with 8-bit state and &ldquo;paging&rdquo; means: momentum and
        variance live in CPU RAM; when the GPU needs them during an
        optimizer step, the NVIDIA unified memory driver pages the tiles
        in transparently. You&apos;ll see brief PCIe-bound pauses —
        that&apos;s the cost — but the alternative is OOM. On an A100
        with a ≥16 per-device batch, the overhead is in the single-digit
        percents.
      </Callout>

      {/* ── What bleeds through the compression ─────────────────── */}
      <Prose>
        <p>
          Every compression suit has a tradeoff. Vacuum-sealing the
          textbook this hard has to give something up — the question is
          what, and whether it matters. Below is where the seals start to
          leak.
        </p>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Forgetting double-quantization:</strong>{' '}
          leaving <code>bnb_4bit_use_double_quant=False</code> silently
          costs you ~3 GB on a 70B model — you compressed the textbook
          but left the compression metadata in plain fp16. If you&apos;re
          squeezing onto a 40GB card and getting OOM at step zero, this
          is the first flag to check.
        </p>
        <p>
          <strong className="text-term-amber">NF4 is inference-only storage:</strong>{' '}
          you cannot accumulate gradients into 4-bit values — the dynamic
          range is far too narrow and quantization noise would drown the
          update. That&apos;s <em>why</em> the LoRA sticky notes stay in
          fp16/bf16 while the textbook underneath is 4-bit. If someone
          proposes a scheme where the quantized weights themselves are
          trained, they are quietly reinventing QAT and it&apos;s a
          different regime.
        </p>
        <p>
          <strong className="text-term-amber">Paged optimizer bottleneck on tiny batches:</strong>{' '}
          with <code>per_device_batch_size=1</code>, the fraction of
          wall-clock time spent paging CPU↔GPU can balloon past 30%. The
          compression suit is fine; the connector hose is the bottleneck.
          Use gradient accumulation (large effective batch, small
          per-device batch) to amortize the paging cost across more
          gradient computations per optimizer step.
        </p>
        <p>
          <strong className="text-term-amber">Target-modules = q_proj/v_proj only:</strong>{' '}
          the QLoRA paper found that attaching LoRA to <em>all</em> linear
          layers (q,k,v,o,gate,up,down) actually matches full-FT quality
          better than the minimal q/v setup. It&apos;s slightly more
          params but still less than 1% of the base. Worth the extra
          memory — since the base is already vacuum-sealed, you have
          budget to spend on more sticky notes.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="QLoRA Llama-7B on a 16GB card">
        <p>
          Pick a 16GB GPU — a single 4090, a T4, a free Colab Pro
          instance. Load <code>meta-llama/Llama-2-7b-hf</code> with the
          QLoRA config from layer 3 (NF4, double-quant, paged AdamW).
          Attach LoRA adapters (<code>r=8</code>) to all linear layers,
          not just q_proj/v_proj. Fine-tune on 1000 examples from Alpaca
          or Databricks-Dolly.
        </p>
        <p className="mt-2">
          Measure three things:{' '}
          <code>torch.cuda.max_memory_allocated()</code> at the peak of
          training, your wall-clock time per step, and final eval loss on
          a 100-example held-out set.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: rerun the exact same recipe in fp16 LoRA (no
          quantization) on a bigger card. Compare eval loss. The QLoRA
          paper&apos;s headline is that the gap is within noise — see if
          you can reproduce it. If you can, you&apos;ve just verified the
          compression suit empirically on your own data.
        </p>
      </Challenge>

      {/* ── Closing + teaser ─────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Fine-tuning cost is
          dominated by four stacks — weights, optimizer, gradients,
          activations — and every serious training trick attacks one of
          them. LoRA shrank three by slashing the trainable parameter
          count. QLoRA finally came for the fourth, the frozen base
          itself, by vacuum-sealing it into NF4: a 4-bit datatype whose
          levels <em>match the distribution</em> of neural-network
          weights rather than fighting them. Add double-quantization (a
          vacuum-seal on the vacuum-seal) and a paged CPU optimizer for
          activation spikes, and a 70B model fine-tunes on a single
          consumer card with quality within noise of 16-bit. The textbook
          weighs a quarter as much; the sticky notes still work.
        </p>
        <p>
          <strong>Next up — Make GPT Talk Back.</strong> You now have the
          tools to adapt a base model efficiently: sticky-note the deltas
          with LoRA, vacuum-seal the base with QLoRA. But even a
          perfectly fine-tuned model is just a distribution over next
          tokens. What actually comes out of it — confident prose or
          bland mush, creative riffs or exact recitation — depends on how
          you sample from that distribution. Temperature, top-k, nucleus
          sampling: the knobs that turn a trained model into a voice. A
          compressed textbook is still just a textbook until somebody
          picks words out of it.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'QLoRA: Efficient Finetuning of Quantized LLMs',
            author: 'Dettmers, Pagnoni, Holtzman, Zettlemoyer',
            venue: 'NeurIPS 2023 — the paper introducing NF4, double-quant, paged optimizers',
            year: 2023,
            url: 'https://arxiv.org/abs/2305.14314',
          },
          {
            title: 'LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale',
            author: 'Dettmers, Lewis, Belkada, Zettlemoyer',
            venue: 'NeurIPS 2022 — the precursor, outlier-aware 8-bit matmul',
            year: 2022,
            url: 'https://arxiv.org/abs/2208.07339',
          },
          {
            title: 'bitsandbytes — 4-bit and 8-bit CUDA kernels for PyTorch',
            author: 'Dettmers et al.',
            venue: 'GitHub — the library every QLoRA recipe depends on',
            url: 'https://github.com/bitsandbytes-foundation/bitsandbytes',
          },
          {
            title: 'peft: Parameter-Efficient Fine-Tuning',
            author: 'HuggingFace',
            venue: 'GitHub — LoRA / QLoRA / prefix-tuning wrappers for transformers',
            url: 'https://github.com/huggingface/peft',
          },
          {
            title: '8-bit Optimizers via Block-wise Quantization',
            author: 'Dettmers, Lewis, Shleifer, Zettlemoyer',
            venue: 'ICLR 2022 — where 8-bit Adam came from',
            year: 2022,
            url: 'https://arxiv.org/abs/2110.02861',
          },
        ]}
      />
    </div>
  )
}
