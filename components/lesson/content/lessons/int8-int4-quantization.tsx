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
import GPTQDemo from '../widgets/GPTQDemo'                  // step-through quantization layer-by-layer, updating remaining layers with Hessian-based error compensation
import OutlierHandling from '../widgets/OutlierHandling'      // histogram of activations showing long tails; LLM.int8() technique separates outlier channels into fp16

// Signature anchor: a climber going lower on the cliff face. Each dropped bit
// is a rung down — the view gets cheaper but the holds get sketchier. int8 is
// a solid ledge most models land on with negligible loss; int4 is a fingertip
// crack that needs calibration, group-wise scales, and outlier tricks; below
// int4 is free-solo territory. Returns at the opening question ("how low can
// you go?"), the bit-budget reveal, and the outlier failure-mode section.
export default function Int8Int4QuantizationLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="int8-int4-quantization" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          The question on the table is simple and slightly reckless:{' '}
          <em>how low can you go?</em> Picture a climber on a cliff face, roped
          in at the top and working their way down. At every ledge they ask the
          same question — do I drop another rung, or do I stop here? Each rung
          down is one bit dropped from the weight. The view gets cheaper —
          smaller file, less memory, more throughput — but the holds get
          sketchier. Step too low and the rock crumbles. That&apos;s this
          lesson.
        </p>
        <p>
          A <code>fp16</code> Llama-70B weight file is 140 GB. That does not fit
          on your GPU, or your friend&apos;s GPU, or probably any GPU
          you&apos;ve ever met. Drop to <code>int8</code> and it&apos;s 70 GB —
          tight but plausible on a single A100. Drop to <code>int4</code> and
          it&apos;s 35 GB, which fits on a single consumer 4090 with room for a
          KV cache and a browser tab. Two rungs lower on the cliff, same model
          still talking.
        </p>
        <p>
          <NeedsBackground slug="quantization-basics">Quantization</NeedsBackground>{' '}
          is the art of storing{' '}
          <NeedsBackground slug="mlp-from-scratch">weights</NeedsBackground>{' '}
          (and sometimes activations) in fewer bits than the training pipeline
          used. Fewer bits means less memory, less bandwidth, and — on hardware
          that knows what to do with integers — more throughput. The catch is
          that 8-bit integers have 256 possible values, 4-bit integers have 16,
          and a trained transformer has weights that really, really want to be
          16-bit floats. Squeeze them wrong and the model produces word salad.
        </p>
        <p>
          This lesson is about squeezing them right. We&apos;ll cover the two
          families of quantization (<strong>PTQ</strong>, cheap and fast;{' '}
          <strong>QAT</strong>, expensive and best). Then the three algorithms
          that actually show up in production today: <strong>LLM.int8()</strong>{' '}
          for the outlier problem, <strong>GPTQ</strong> for Hessian-based
          error compensation, and <strong>AWQ</strong> for activation-aware
          scaling. By the end, turning a 140 GB model into a 35 GB model should
          feel routine rather than alarming.
        </p>
      </Prose>

      <Personify speaker="Quantization">
        I round your weights to the nearest integer. That&apos;s it. The
        interesting part is that if you round naively — uniform steps, no
        per-channel scales, no outlier protection — your 70B model becomes a
        random n-gram generator. Everything below is how to round without
        breaking things.
      </Personify>

      {/* ── PTQ vs QAT ──────────────────────────────────────────── */}
      <Prose>
        <p>
          Two philosophies. You can either quantize <em>after</em> training
          (treat it as a compression step applied to a finished model) or
          quantize <em>during</em> training (make the network learn with the
          quantization error already baked in).
        </p>
        <ul>
          <li>
            <KeyTerm>Post-Training Quantization (PTQ)</KeyTerm> takes a trained
            fp16 checkpoint, runs a tiny calibration pass over ~256 samples to
            estimate activation ranges, and writes out an int8 or int4 version.
            No gradient updates. Minutes to hours. This is what LLM.int8(),
            GPTQ, and AWQ all are — PTQ methods with increasingly clever tricks.
          </li>
          <li>
            <KeyTerm>Quantization-Aware Training (QAT)</KeyTerm> inserts
            fake-quantize ops into the forward pass during training. The
            network sees the rounding error on every step and learns weights
            that survive it. Best quality, roughly the cost of another training
            run. Rare for 70B-scale LLMs because nobody wants to re-pretrain.
          </li>
        </ul>
        <p>
          For large language models PTQ won by default — the pretraining bill
          is too big to redo. The rest of this lesson is PTQ.
        </p>
      </Prose>

      {/* ── LLM.int8() math + widget ────────────────────────────── */}
      <Prose>
        <p>
          Back to the cliff. The climber stepped down from fp16 to int8 —
          roughly a shoulder-width drop to a solid ledge. Most models land
          here with negligible loss and nobody thinks about it again.
          That&apos;s the good news. The bad news: it took a specific trick to
          make that ledge stable, and the trick comes from Dettmers 2022. If
          you look at the activation tensors inside a 6B+ transformer, a
          handful of hidden dimensions have values roughly 20× larger than
          everything else. They look like spikes poking out of a flat plain.
          Uniformly quantizing that tensor to int8 means the 256 levels get
          stretched to cover the spikes — and the flat plain, which is 99% of
          the information, gets squashed into three or four levels. The model
          is now rounding the signal, not the noise.
        </p>
        <p>
          The fix is called <KeyTerm>mixed-precision decomposition</KeyTerm>.
          Identify the outlier columns (say, any column whose max absolute
          activation exceeds 6.0), pull them out, run them in fp16. Run the
          other 99% of columns in int8. Concatenate the results at the end.
        </p>
      </Prose>

      <MathBlock caption="LLM.int8() — outlier decomposition">
{`Y  =  X · Wᵀ

     =  X_fp16[:, O]  ·  W_fp16[:, O]ᵀ          ← outlier columns, full precision
        +
        dequant( X_int8[:, R] · W_int8[:, R]ᵀ, s_X · s_W )   ← the rest, int8

where  O = { i :  max |X[:, i]|  >  6.0 }
       R = { 0, …, d } \\ O
       s_X, s_W  =  per-row, per-column scaling factors`}
      </MathBlock>

      <OutlierHandling />

      <Prose>
        <p>
          Drag the threshold. At <code>6.0</code> you route ~0.1% of the
          columns to fp16 and the model is within 1% perplexity of the fp16
          baseline. Lower the threshold and more columns go to fp16 — higher
          quality, less compression. Raise it and the int8 path has to absorb
          the spikes, which corrupts the quantization grid for everything
          else. The sweet spot is narrow and worth finding.
        </p>
        <p>
          LLM.int8() is the default <code>load_in_8bit=True</code> flag in{' '}
          <code>bitsandbytes</code>. It is the reason you can run a 175B model
          on a 4-way A100 node with no retraining, no calibration, and no
          perplexity you&apos;d notice.
        </p>
      </Prose>

      <Personify speaker="Outlier channel">
        I&apos;m one hidden dimension out of 12,288 and I run 20× hotter than
        my peers. Transformers love me — I carry position-y, attention-sink-y,
        BOS-ish information the rest of the network relies on. Round me to
        int8 and you flatten me. Round me to fp16 and we all live. Give me
        the floats.
      </Personify>

      {/* ── GPTQ math + widget ──────────────────────────────────── */}
      <Prose>
        <p>
          LLM.int8() keeps the int8 ledge stable. It does <em>not</em> help
          when you want to keep dropping — at 4 bits, the quantization grid is
          so coarse that even well-behaved weights lose information the
          network actually needed. This is the fingertip crack: the climber
          can hang here, but they need to place every hold deliberately. You
          need to compensate for the rounding error as you go.
        </p>
        <p>
          <strong>GPTQ</strong> (Frantar et al. 2023) does this one layer at
          a time. Quantize column <code>j</code> of the weight matrix. That
          introduces an error in the layer&apos;s output. Distribute that
          error across the <em>remaining unquantized</em> columns by nudging
          them slightly — using the layer&apos;s Hessian to figure out which
          nudges hurt the output least. Then quantize the next column. By the
          time every column is quantized, the accumulated error has been
          partially absorbed by weights that hadn&apos;t been touched yet.
        </p>
      </Prose>

      <MathBlock caption="GPTQ — Hessian-based update for the remaining columns">
{`Layer minimises    L(W)  ≈  ‖ X W  −  X Ŵ ‖²

Hessian            H     =  2 · Xᵀ X

For each column j (in order):
  1. quantize:      ŵ_j   =  quant(w_j)
  2. error:         e_j   =  (w_j  −  ŵ_j) / [H⁻¹]_{j,j}
  3. update rest:   w_k  ←  w_k  −  e_j · [H⁻¹]_{j,k}     for k > j
  4. mark j done, move on.`}
      </MathBlock>

      <GPTQDemo />

      <Prose>
        <p>
          Step through column-by-column. Notice how every time you quantize a
          column, the still-unquantized columns (to the right) shift slightly.
          Those shifts are the error compensation — the algorithm is
          pre-baking future quantization errors into the current weights so
          the layer&apos;s overall output doesn&apos;t drift. At the end the
          weights are all int4 and the layer output barely moved.
        </p>
        <p>
          Two things to internalize. First, GPTQ is <em>per-layer</em> — the
          Hessian is estimated from calibration data passed through up to
          that layer, not the whole network. Second, inverting the Hessian is
          the expensive part, but you only do it once per layer, and
          there&apos;s a Cholesky trick that makes it tractable for{' '}
          <code>d ≈ 12,000</code> dimensions. Frantar&apos;s implementation
          quantizes a 175B model in about four GPU-hours.
        </p>
      </Prose>

      <Personify speaker="GPTQ">
        I&apos;m a greedy algorithm with regret. Every column I quantize, I
        apologise to the columns I haven&apos;t touched yet by adjusting them.
        By the time I finish the matrix, the error I introduced early has
        been absorbed by edits I made later. The Hessian tells me how to
        apologise most efficiently.
      </Personify>

      {/* ── AWQ callout ─────────────────────────────────────────── */}
      <Callout variant="insight" title="AWQ — the third way">
        <p>
          <strong>Activation-aware Weight Quantization</strong> (Lin et al.
          2023) takes a different angle on the same problem. Instead of fixing
          errors after the fact like GPTQ, it{' '}
          <em>scales the weights before quantizing them</em>. Find the channels
          whose activations are large — those are the &ldquo;salient&rdquo;
          channels, the ones the model actually uses. Multiply their weights
          by a scale <code>s &gt; 1</code> before quantization, so they land
          on more of the int4 grid and keep their precision. Divide the
          corresponding activation by the same <code>s</code> at runtime to
          cancel it out. The effect: salient weights get fine-grained int4,
          non-salient ones get coarse int4, and the math still works out.
        </p>
        <p className="mt-2">
          AWQ is simpler than GPTQ (no Hessian, no iterative updates), runs
          faster, and in many benchmarks matches or beats GPTQ on perplexity.
          It&apos;s what the <code>AutoAWQ</code> library ships and what most
          modern int4 LLaMA releases use under the hood.
        </p>
      </Callout>

      {/* ── Ladder + serving reality ────────────────────────────── */}
      <Prose>
        <p>
          Time for the bit-budget reveal — the whole cliff face laid out, rung
          by rung. Here&apos;s what each drop actually costs and buys.
        </p>
        <ul>
          <li>
            <strong>fp16 / bf16</strong> — the top of the cliff. 2 bytes per
            weight, no quality loss because this is what the model was trained
            in. 16 bits; 65,536 possible values per weight.
          </li>
          <li>
            <strong>int8</strong> — one rung down, a wide solid ledge. 2×
            smaller than fp16, nearly lossless with LLM.int8(). 256 possible
            values per weight. Every modern GPU has native int8 Tensor Cores.
            The no-brainer default.
          </li>
          <li>
            <strong>int4</strong> — another rung lower, a fingertip crack. 4×
            smaller than fp16. 16 possible values per weight. Needs GPTQ or
            AWQ to keep quality; typical perplexity hit is 0.1–0.3 on WikiText.
            The standard for consumer and local inference.
          </li>
          <li>
            <strong>int3, int2, binary</strong> — free-solo territory. 8
            values, 4 values, 2 values. Quality degrades sharply, and consumer
            hardware doesn&apos;t have a fast kernel for 3-bit anything.
            You&apos;ll see papers but not production deployments. It works —
            if you know what you&apos;re doing, and if you accept that the
            bottom of the cliff is a different sport.
          </li>
        </ul>
        <p>
          In 2024–2026 serving reality:{' '}
          <strong>fp16 is the training/baseline</strong>,{' '}
          <strong>int8 is standard for production inference</strong>, and{' '}
          <strong>
            int4 (GPTQ, AWQ, or GGUF) is common for anything that has to run on
            a laptop or a single consumer card
          </strong>. GGUF in particular — the format used by{' '}
          <code>llama.cpp</code> — has a zoo of mixed-precision schemes (Q4_K_M,
          Q5_K_S, Q8_0, etc.) that pack different bit widths for different
          tensors in the same file.
        </p>
      </Prose>

      <Callout variant="note" title="GGUF and MLX — the quiet infrastructure">
        <p>
          <strong>GGUF</strong> is the on-disk format for <code>llama.cpp</code>:
          a single self-contained file with quantized weights, tokenizer, and
          metadata. Its K-quants (Q4_K_M, Q5_K_M) use a block-wise scheme with
          per-block scales and per-super-block min/max — effectively a
          mixed-precision quantization that outperforms uniform int4 on the
          same bit budget.
        </p>
        <p className="mt-2">
          <strong>MLX</strong> is Apple&apos;s native ML framework for M-series
          chips. Its quantization is conceptually similar (group-wise
          int4/int8) but the kernels are tuned for unified memory and the
          Apple Neural Engine. If you&apos;re running local models on a
          MacBook, MLX is likely what&apos;s under the hood.
        </p>
      </Callout>

      {/* ── Calibration data ────────────────────────────────────── */}
      <Prose>
        <p>
          All of these PTQ methods — LLM.int8(), GPTQ, AWQ — need{' '}
          <KeyTerm>calibration data</KeyTerm>. Not to train anything, just to
          estimate activation ranges and (for GPTQ) the layer Hessian. Typical
          size is 128–512 samples. The dataset matters more than the size:
          calibrate a code model on Wikipedia and you&apos;ll get a model that
          quantizes its most confident domain badly.
        </p>
        <p>
          In practice: use a subset of the pretraining mix, or a held-out
          chunk of your target inference distribution. C4 and WikiText are the
          standard defaults for research; for production, whatever you&apos;ll
          actually be serving is better.
        </p>
      </Prose>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers. A numpy GPTQ sketch on a single linear layer, to see
          the update rule working on something tiny. Then <code>bitsandbytes</code>{' '}
          for LLM.int8() in one flag. Then <code>auto-gptq</code> for real int4
          quantization of a transformer.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · gptq_sketch.py"
        output={`fp16 output norm : 4.8213
int4 naive  norm : 5.1740   (error = 0.3527)
int4 gptq   norm : 4.8301   (error = 0.0088)`}
      >{`import numpy as np

def quantize_int4(w, scale):
    q = np.clip(np.round(w / scale), -8, 7)       # 4-bit signed, [-8, 7]
    return q * scale                              # dequantized fp value

# Single linear layer — X @ W, shapes (N, d) × (d, d)
np.random.seed(0)
N, d = 256, 64
X = np.random.randn(N, d).astype(np.float32)
W = np.random.randn(d, d).astype(np.float32) * 0.1

# Per-column scale (standard int4 choice).
scale = np.max(np.abs(W), axis=0) / 7

# --- naive int4: just round every column independently ---------------
W_naive = np.stack([quantize_int4(W[:, j], scale[j]) for j in range(d)], axis=1)

# --- GPTQ: quantize column-by-column, compensate with Hessian -------
H    = X.T @ X + 1e-3 * np.eye(d)                 # (d, d), SPD
Hinv = np.linalg.inv(H)
W_q  = W.copy()
for j in range(d):
    w_j     = W_q[:, j].copy()
    w_q_j   = quantize_int4(w_j, scale[j])
    err     = (w_j - w_q_j) / Hinv[j, j]
    W_q[:, j] = w_q_j
    # distribute error to remaining columns via Hessian row
    W_q[:, j+1:] -= np.outer(err, Hinv[j, j+1:])

def frob(A): return np.linalg.norm(A)
print(f"fp16 output norm : {frob(X @ W):.4f}")
print(f"int4 naive  norm : {frob(X @ W_naive):.4f}   "
      f"(error = {frob(X @ (W - W_naive)):.4f})")
print(f"int4 gptq   norm : {frob(X @ W_q):.4f}   "
      f"(error = {frob(X @ (W - W_q)):.4f})")`}</CodeBlock>

      <Prose>
        <p>
          Naive int4 rounding blows the output norm off by ~7%. GPTQ&apos;s
          column-by-column compensation brings it back to within 0.2%. Same
          bit budget, same weights, much better model — purely because we
          spread the rounding error across columns that hadn&apos;t been
          rounded yet.
        </p>
        <p>
          Now the production version. <code>bitsandbytes</code> takes any
          HuggingFace model and converts it to LLM.int8() in one flag. No
          calibration code, no custom kernels.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch + bitsandbytes · load_int8.py"
      >{`from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-2-7b-hf"

# LLM.int8() — outlier decomposition handled internally.
# 7B model drops from ~13 GB to ~7 GB of VRAM.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,                  # the whole trick
    device_map="auto",                  # split across GPUs if needed
)
tok = AutoTokenizer.from_pretrained(model_id)

x = tok("Quantization is", return_tensors="pt").to(model.device)
out = model.generate(**x, max_new_tokens=20)
print(tok.decode(out[0]))`}</CodeBlock>

      <Prose>
        <p>
          And int4 via <code>auto-gptq</code>. One calibration pass over 128
          samples, then save a quantized checkpoint that loads like any other
          HuggingFace model.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch + auto-gptq · quantize_int4.py"
      >{`from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset

model_id = "meta-llama/Llama-2-7b-hf"
out_dir  = "llama2-7b-gptq-int4"

qcfg = BaseQuantizeConfig(
    bits=4,                            # int4
    group_size=128,                    # per-group scales — standard choice
    desc_act=False,                    # column-order heuristic; False is faster
)

tok   = AutoTokenizer.from_pretrained(model_id)
model = AutoGPTQForCausalLM.from_pretrained(model_id, qcfg)

# ~128 calibration samples; c4 is the community default for GPTQ.
c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
calib = [tok(next(iter(c4))["text"][:2048], return_tensors="pt")
         for _ in range(128)]

model.quantize(calib)                  # Hessian + column-wise compensation inside
model.save_quantized(out_dir)          # 7B model → ~3.5 GB on disk
tok.save_pretrained(out_dir)`}</CodeBlock>

      <Bridge
        label="numpy sketch → bitsandbytes → auto-gptq"
        rows={[
          {
            left: 'W_q[:, j+1:] -= np.outer(err, Hinv[j, j+1:])',
            right: 'model.quantize(calib)',
            note: 'the loop lives inside — you supply data, it does the math',
          },
          {
            left: 'manual per-column scale',
            right: 'group_size=128',
            note: 'production uses per-group scales — finer than per-tensor, cheaper than per-column',
          },
          {
            left: 'np.linalg.inv(H)',
            right: 'internal Cholesky on block-diagonal H',
            note: 'naive inversion is O(d³); auto-gptq uses a Cholesky trick to scale',
          },
        ]}
      />

      <Callout variant="insight" title="the point of the three layers">
        The numpy sketch shows what GPTQ is <em>doing</em> — a rank-one error
        update, column by column, weighted by a precomputed Hessian inverse.{' '}
        <code>bitsandbytes</code> shows how int8 becomes a one-line deployment
        concern when the kernel handles the outlier split for you.{' '}
        <code>auto-gptq</code> shows the full PTQ flow as a real ML engineer
        runs it: model in, calibration data in, quantized checkpoint out. Same
        algorithm, three levels of ceremony.
      </Callout>

      {/* ── Outlier failure-mode section ────────────────────────── */}
      <Prose>
        <p>
          One more return to the cliff, because this is where naive int4 gets
          people killed. You&apos;ve dropped a rung past the int8 ledge onto
          the fingertip crack, and the holds look fine. Then a single outlier
          weight flicks your boot off. Here&apos;s the failure mode, step by
          step.
        </p>
        <p>
          Int4 has 16 buckets. Your per-tensor scale is set by the maximum
          absolute weight. In a well-behaved matrix that maximum is maybe{' '}
          <code>0.3</code>, every bucket covers ~<code>0.04</code>, and most
          weights land with a quantization error well under a percent. Now
          add one outlier weight at <code>5.0</code>. The maximum just jumped
          16×. Every bucket now covers ~<code>0.6</code>. Your well-behaved
          weights, the ones doing the actual work, are all being rounded to
          either <code>0</code> or <code>±0.6</code>. The signal is gone. The
          model is now a random text generator. One bit too low, one bad
          hold, and the climber is at the bottom of the cliff.
        </p>
        <p>
          The three techniques from this lesson are three different ways to
          not fall. <strong>LLM.int8()</strong> pulls the outlier columns out
          and runs them in fp16 — the climber leaves their fingertips on the
          crack but clips the pro onto a higher anchor. <strong>GPTQ</strong>{' '}
          keeps rounding naively, then nudges the remaining weights to
          compensate — lower the boot, but shift your weight to a different
          hold on the way down. <strong>AWQ</strong> pre-scales the salient
          channels before they get rounded so they land on more of the int4
          grid — chalk up before you even touch the rock. All three exist
          because naive int4 on a raw transformer doesn&apos;t work.
        </p>
      </Prose>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Calibration data mismatch:</strong>{' '}
          calibrate a code model on Wikipedia and its perplexity on code will
          crater. The activation ranges estimated during calibration are the{' '}
          <em>only</em> thing the quantizer knows about your data. Use samples
          from the actual inference domain.
        </p>
        <p>
          <strong className="text-term-amber">Quantizing activations without outlier handling:</strong>{' '}
          works fine for CNNs and small transformers. For 6B+ LLMs the outlier
          channels break the grid and the model degenerates into repetition
          loops. If you&apos;re rolling your own int8 activation quantization,
          you <em>must</em> do mixed-precision decomposition.
        </p>
        <p>
          <strong className="text-term-amber">Expecting speed gains without hardware support:</strong>{' '}
          int4 on a GPU that has no native int4 kernel is often <em>slower</em>{' '}
          than fp16, because every matmul dequantizes first. Know what your
          hardware actually accelerates: Ampere and newer do int8 natively;
          Hopper and newer do fp8; int4 speedups come from bespoke kernels
          (Marlin, ExLlama) not the standard libraries.
        </p>
        <p>
          <strong className="text-term-amber">Not re-quantizing after fine-tuning:</strong>{' '}
          if you QLoRA-finetune an int4 base and then merge the adapters, the
          merged weights are fp16 again. Ship the int4 + adapter separately,
          or re-quantize after merging.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Int4 a 7B model and measure the damage">
        <p>
          Quantize Llama-7B to int4 with GPTQ using the <code>auto-gptq</code>{' '}
          snippet above. Calibrate on 128 samples of C4. Save the checkpoint.
        </p>
        <p className="mt-2">
          Load both the fp16 model and the int4 model. Compute perplexity on
          the first 40k tokens of WikiText-2 test split for each. Report:
        </p>
        <ul className="mt-2 ml-4 list-disc">
          <li>fp16 perplexity</li>
          <li>int4 perplexity (and the delta)</li>
          <li>VRAM used at inference time, fp16 vs int4</li>
          <li>tokens/second for a 256-token generation, fp16 vs int4</li>
        </ul>
        <p className="mt-2 text-dark-text-muted">
          Expect a perplexity delta somewhere around <code>+0.1</code> to{' '}
          <code>+0.3</code>, a VRAM drop of ~4×, and tokens/sec that depends
          heavily on whether you&apos;re using a kernel like Marlin or ExLlama.
          If your perplexity delta is above <code>+1.0</code>, your
          calibration set is probably mismatched — try a different subset.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> Quantization is rounding,
          and rounding is a problem when the thing being rounded has a
          long-tailed distribution. LLM.int8() handles that by routing the
          tails to fp16. GPTQ handles that by distributing rounding error
          across columns that haven&apos;t been touched yet, weighted by a
          Hessian that tells it which distribution hurts output least. AWQ
          handles that by scaling salient weights before quantizing so they
          land on more of the grid. All three need calibration data from the
          distribution you actually care about. All three are PTQ — no
          retraining required — which is why they won at the scale of modern
          LLMs. Int8 is the ledge, int4 is the crack, lower is a sport.
        </p>
        <p>
          <strong>Next up — Speculative Decoding.</strong> Quantization makes
          the model smaller so the same GPU can hold more of it. Speculative
          decoding makes the model <em>faster</em> by having a tiny draft
          model propose several tokens at a time while the big model verifies
          them in parallel. A 2–3× throughput win with zero quality loss,
          because the big model still has the final say on every token. Same
          goal (&ldquo;serve this model cheaper&rdquo;), completely different
          mechanism — no more rungs to drop; instead, a faster climb.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale',
            author: 'Dettmers, Lewis, Belkada, Zettlemoyer',
            venue: 'NeurIPS 2022',
            url: 'https://arxiv.org/abs/2208.07339',
          },
          {
            title: 'GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers',
            author: 'Frantar, Ashkboos, Hoefler, Alistarh',
            venue: 'ICLR 2023',
            url: 'https://arxiv.org/abs/2210.17323',
          },
          {
            title: 'AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration',
            author: 'Lin, Tang, Tang, Yang, Zhang, Dang, Han',
            venue: 'MLSys 2024',
            url: 'https://arxiv.org/abs/2306.00978',
          },
          {
            title: 'QLoRA: Efficient Finetuning of Quantized LLMs',
            author: 'Dettmers, Pagnoni, Holtzman, Zettlemoyer',
            venue: 'NeurIPS 2023',
            url: 'https://arxiv.org/abs/2305.14314',
          },
          {
            title: 'bitsandbytes — 8-bit optimizers and quantization for PyTorch',
            author: 'Tim Dettmers et al.',
            url: 'https://github.com/bitsandbytes-foundation/bitsandbytes',
          },
          {
            title: 'llama.cpp — GGUF format and K-quants',
            author: 'Georgi Gerganov et al.',
            url: 'https://github.com/ggerganov/llama.cpp',
          },
        ]}
      />
    </div>
  )
}
