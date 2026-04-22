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
import RMSvsLayerNorm from '../widgets/RMSvsLayerNorm'
import NormOpCost from '../widgets/NormOpCost'

// Signature anchor: "the step we found we didn't need." RMSNorm is LayerNorm
// with the mean-subtraction dropped. Revisited at the opening, the math
// reveal, and the why-Llama-uses-it section — three load-bearing beats.

export default function RMSNormalizationLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="rms-normalization" />

      <Prose>
        <p>
          <NeedsBackground slug="layer-normalization">LayerNorm</NeedsBackground>{' '}
          does two things in one breath. It subtracts the mean of the feature
          vector (the <em>center</em> step), then divides by the standard
          deviation (the <em>scale</em> step). Two moves, one op, and every
          transformer from 2017 onwards took it as gospel.
        </p>
        <p>
          Then someone stress-tested the first half. Turns out the mean
          subtraction is the part that nobody bothered to ablate. Keep the
          scale, drop the center, retrain — and the model is fine. Not
          equivalent-on-paper fine, but within-noise-of-your-seed fine, which
          is the only kind of fine that actually matters when you&apos;re
          spending eight-figure sums on training runs.
        </p>
        <p>
          That&apos;s <KeyTerm>RMSNorm</KeyTerm>. A 2019 paper by Zhang and
          Sennrich. It&apos;s LayerNorm minus a step. The step we found we
          didn&apos;t need. It ships in Llama, PaLM, Gemma, Mistral, and
          basically every open large-language model published after 2022. The
          rest of this lesson is why that step was safe to skip, what the math
          looks like after you skip it, and why &ldquo;a step we didn&apos;t
          need&rdquo; translates into real wall-clock savings at scale.
        </p>
      </Prose>

      <MathBlock caption="RMSNorm vs LayerNorm">
{`# LayerNorm (what we already know)
μ    =   (1/D) · Σⱼ xⱼ                              # ← mean subtracted
x̂ⱼ  =   (xⱼ − μ) / √(σ² + ε)                       # ← std divided
yⱼ   =   γⱼ · x̂ⱼ  +  βⱼ                             # ← γ scale + β shift

# RMSNorm
rms  =   √( (1/D) · Σⱼ xⱼ²  + ε)                    # ← just RMS (no mean)
yⱼ   =   γⱼ · (xⱼ / rms)                            # ← γ scale, no β`}
      </MathBlock>

      <Prose>
        <p>
          Two lines vanish. The mean <code>μ</code> never gets computed, and
          the learnable shift <code>β</code> goes with it — there&apos;s
          nothing to shift around once you&apos;re no longer centering. What
          survives is the divide-by-RMS and the learnable scale{' '}
          <code>γ</code>. That&apos;s the entire op. Drag the widget below
          and you&apos;ll see both layers chew on the same input with the
          same dial.
        </p>
      </Prose>

      <RMSvsLayerNorm />

      <Prose>
        <p>
          Slide the offset. LayerNorm&apos;s output barely flinches — the
          mean subtraction cancels any bulk shift in the input before it hits
          the scale step. RMSNorm&apos;s output moves with the shift, because
          there&apos;s nothing left to cancel it. On paper that sounds like a
          real property loss. In practice the next Linear layer has a bias,
          and it learns to eat whatever offset the upstream tensor arrives
          with. The invariance wasn&apos;t doing much; the downstream weights
          were doing the work anyway.
        </p>
      </Prose>

      <Callout variant="note" title="the empirical claim">
        Zhang and Sennrich (2019) trained matched transformer encoders on WMT
        translation with LayerNorm and RMSNorm and found no meaningful quality
        gap — but RMSNorm ran 7–64% faster depending on implementation, with
        less compute and less memory traffic. Llama-1 (2023) reproduced this
        at scale and the rest of the field followed. Today RMSNorm is the
        default for every open LLM that values inference latency, which is
        every open LLM.
      </Callout>

      <Personify speaker="RMSNorm">
        I skip the mean. If your feature vector is <code>[1, 2, 3, 4]</code>,
        LayerNorm would subtract 2.5 from each element before rescaling. I
        just rescale. Turns out the mean subtraction wasn&apos;t earning its
        keep — your network compensates through other parameters — and I
        save you one full read and subtract over the activation tensor. At{' '}
        <em>D=4096, L=32 layers, batch of millions of tokens</em>, those
        saved passes turn into real wall-clock time.
      </Personify>

      <Prose>
        <p>
          Let&apos;s quantify what the skipped step actually buys you. Norm
          layers aren&apos;t compute-bound — they&apos;re{' '}
          <em>memory-bound</em>. The clock time you pay is mostly the time
          spent shuffling activations in and out of HBM, not the arithmetic
          inside. Drop one pass over the tensor and you drop roughly that
          fraction of the runtime.
        </p>
      </Prose>

      <NormOpCost />

      <Prose>
        <p>
          Crank the dimensions up to pre-training scale (<code>B = 8</code>,{' '}
          <code>S = 2048</code>, <code>D = 4096</code> is a realistic batch).
          The op-count delta is around 20% — which tracks with the wall-clock
          speedups teams report in practice. Multiply that by 32 transformer
          blocks, each with two norm layers, and the mean-subtract step you
          deleted just saved you hundreds of millions of operations per
          forward pass. Per forward pass. You do a lot of forward passes.
        </p>
      </Prose>

      <Callout variant="insight" title="why the savings are real, not theoretical">
        GPUs have enormous arithmetic throughput but limited memory bandwidth.
        The bottleneck for a norm layer isn&apos;t the math — it&apos;s
        reading the 4096-element activation from HBM and writing it back.
        LayerNorm has to read every element <em>twice</em> (once for the
        mean, once for the variance — or clever fused tricks that still need
        multiple passes). RMSNorm reads once, computes mean(x²), writes back.
        Fewer trips to memory = faster in real life, regardless of the FLOP
        count your textbook cites.
      </Callout>

      <Prose>
        <p>
          When should you pick which? Rough guide:
        </p>
        <ul>
          <li>
            <strong>New transformer architecture:</strong> use RMSNorm.
            It&apos;s the modern default and you&apos;ll match the codebase
            conventions of every recent paper.
          </li>
          <li>
            <strong>Reproducing an older paper:</strong> use whatever it
            used. GPT-2/3 and BERT used LayerNorm. Llama, PaLM, Gemma,
            Mistral use RMSNorm.
          </li>
          <li>
            <strong>Non-transformer networks:</strong> stick with LayerNorm
            or <NeedsBackground slug="batch-normalization">BatchNorm</NeedsBackground>.
            RMSNorm hasn&apos;t been extensively validated outside the
            transformer setting and the savings matter less without hundreds
            of norm layers stacked back-to-back.
          </li>
          <li>
            <strong>Super-long context:</strong> the relative savings grow
            with sequence length. For million-token contexts, RMSNorm is
            essentially required.
          </li>
        </ul>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">No β in RMSNorm.</strong> If
          you&apos;re porting a LayerNorm layer to RMSNorm, don&apos;t forget
          the learnable bias is gone. The next Linear or attention projection
          usually has a bias of its own, so it absorbs whatever shift{' '}
          <code>β</code> would have learned. If you port it naively and leave
          a dangling <code>β</code> parameter around, it will sit there
          unused and you&apos;ll wonder why your param count is off.
        </p>
        <p>
          <strong className="text-term-amber">
            Pre-norm placement still matters.
          </strong>{' '}
          RMSNorm is a drop-in for LayerNorm, but you still want{' '}
          <em>pre-norm</em> ordering — normalize before the sublayer, then
          add the residual. Every modern transformer codebase does it this
          way. Post-norm will train; it will also fight you the whole way.
        </p>
        <p>
          <strong className="text-term-amber">
            Precision matters for the rsqrt.
          </strong>{' '}
          <code className="text-dark-text-primary">
            1 / √(mean(x²) + ε)
          </code>{' '}
          must be computed in fp32 even in fp16 or bfloat16 models, or
          numerical underflow gives you NaNs mid-training and a long debugging
          afternoon. PyTorch&apos;s built-in{' '}
          <code className="text-dark-text-primary">nn.RMSNorm</code> (2.4+)
          handles this correctly; hand-rolled versions often do not.
        </p>
        <p>
          <strong className="text-term-amber">
            <code className="text-dark-text-primary">nn.RMSNorm</code> needs
            PyTorch 2.4+.
          </strong>{' '}
          For older versions, write it by hand in five lines. The tradeoff is
          no CUDA kernel fusion — the built-in is meaningfully faster on
          modern hardware.
        </p>
      </Gotcha>

      <Prose>
        <p>
          Now the code. Two layers — NumPy and PyTorch. The op is simple
          enough that the NumPy version reads like the formula, which is
          exactly the point.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure numpy · rms_norm.py"
        output={`input  rms = [1.1456 0.8732]
output rms per row ≈ 1.0000 for all`}
      >{`import numpy as np

def rms_norm(x, gamma=None, eps=1e-6):
    """RMSNorm over the last dim."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    x_hat = x / rms
    if gamma is not None:
        x_hat = x_hat * gamma
    return x_hat

rng = np.random.default_rng(0)
x = rng.normal(loc=[0.3, -0.2], scale=[1, 1], size=(8, 2)).T
print("input  rms =", np.round(np.sqrt(np.mean(x ** 2, axis=-1)), 4))
y = rms_norm(x)
print("output rms per row ≈", round(np.sqrt(np.mean(y ** 2, axis=-1)).mean(), 4), "for all")`}</CodeBlock>

      <Bridge
        label="RMSNorm ↔ LayerNorm — what drops out"
        rows={[
          { left: 'x.mean(-1, keepdims=True)', right: '[removed]', note: 'no mean subtraction' },
          { left: 'x - mean', right: '[removed]', note: 'no centering' },
          { left: 'x.var(-1)', right: 'np.mean(x ** 2, -1)', note: 'mean(x²) instead of variance around μ' },
          { left: '(x - mean) / sqrt(var + eps)', right: 'x / sqrt(mean(x**2) + eps)', note: 'RMS divide' },
          { left: 'gamma * x_hat + beta', right: 'gamma * x_hat', note: 'no β parameter' },
        ]}
      />

      <Prose>
        <p>
          Five lines removed, one line changed. That&apos;s the whole
          simplification. The PyTorch version is the same thing with{' '}
          <code>rsqrt</code> for the divide (faster and more numerically
          friendly) and an explicit fp32 cast for mixed-precision sanity.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch · rms_norm_pytorch.py"
        output={`torch.Size([8, 12, 768])
mean output rms ≈ 1.000`}
      >{`import torch
import torch.nn as nn

# PyTorch 2.4+ ships nn.RMSNorm natively. For older versions:
class RMSNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))

    def forward(self, x):
        # Cast to float for numerical stability, cast back after.
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms * self.gamma).to(dtype)

norm = RMSNorm(features=768)
x = torch.randn(8, 12, 768)
y = norm(x)
print(y.shape)
print(f"mean output rms ≈ {y.pow(2).mean(-1).sqrt().mean().item():.3f}")`}</CodeBlock>

      <Bridge
        label="pre-norm transformer block — the modern standard"
        rows={[
          {
            left: 'x = x + Attn(LayerNorm(x))',
            right: 'x = x + Attn(RMSNorm(x))',
            note: 'drop-in replacement — same position, same residual',
          },
          {
            left: 'x = x + MLP(LayerNorm(x))',
            right: 'x = x + MLP(RMSNorm(x))',
            note: 'second norm point, also pre-norm',
          },
        ]}
      />

      <Callout variant="insight" title="the norm-layer family, summarized">
        <strong>LayerNorm:</strong> feature-axis mean + variance, plus learned
        γ and β. The safe default. <strong>BatchNorm:</strong> batch-axis
        mean + variance, plus γ and β, plus running stats that switch between
        train and eval. Excellent for large-batch convnets, fragile
        elsewhere. <strong>RMSNorm:</strong> no mean subtraction, just the
        RMS divide and γ. Faster, matches LayerNorm quality empirically,
        standard in LLMs.
      </Callout>

      <Challenge prompt="Port a LayerNorm transformer to RMSNorm">
        <p>
          Pick any small transformer you have lying around, or spin up a
          two-block one from <code>nn.TransformerEncoderLayer</code>. Replace
          every <code>nn.LayerNorm</code> with the <code>RMSNorm</code>{' '}
          module above. Train both on a small language-modeling objective
          (WikiText-2 is fine) for 500 steps. Plot training loss for each.
        </p>
        <p className="mt-2">
          Expected observations: (1) the loss curves are basically identical,
          (2) RMSNorm is ~10% faster end-to-end on a GPU, and (3) the
          difference in final validation perplexity is smaller than your
          run-to-run seed variance. If that last one surprises you, welcome
          to the empirical side of deep learning.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: benchmark just the norm layer in isolation with{' '}
          <code>torch.cuda.synchronize()</code> and{' '}
          <code>time.perf_counter()</code>. You&apos;ll see the 15–25%
          speedup on the op itself, which is where the end-to-end gains come
          from.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> RMSNorm is LayerNorm with
          one step removed — the mean subtraction — and the learnable bias
          that goes with it. Trains as well as LayerNorm on transformer
          workloads, runs meaningfully faster because it skips one pass over
          the activation tensor. Modern LLMs default to it. Old codebases
          still use LayerNorm. The two are drop-in replacements in either
          direction; the quality gap is below noise. This is the quiet
          architectural win — the kind of optimization you only notice
          because Llama uses it.
        </p>
        <p>
          <strong>End of section.</strong> You now have the parts.
          Tensors, autograd, modules, a normalization layer that ships in
          every frontier model on the open web. The next section wires them
          into the four-line ritual that makes a model actually learn —
          forward, loss, backward, step. Every training run on every GPU in
          every data center you&apos;ve heard of is that loop, with a lot of
          bookkeeping bolted on. We&apos;ll build it next.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Root Mean Square Layer Normalization',
            author: 'Biao Zhang, Rico Sennrich',
            venue: 'NeurIPS 2019 — the original RMSNorm paper',
            url: 'https://arxiv.org/abs/1910.07467',
          },
          {
            title: 'LLaMA: Open and Efficient Foundation Language Models',
            author: 'Touvron et al.',
            venue: 'Meta, 2023 — first major open LLM to standardize on RMSNorm',
            url: 'https://arxiv.org/abs/2302.13971',
          },
          {
            title: 'PyTorch documentation — nn.RMSNorm',
            author: 'PyTorch core team',
            venue: 'pytorch.org (available from 2.4)',
            url: 'https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html',
          },
        ]}
      />
    </div>
  )
}
