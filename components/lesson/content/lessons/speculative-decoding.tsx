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
import SpecDecodingFlow from '../widgets/SpecDecodingFlow'
import AcceptanceRate from '../widgets/AcceptanceRate'

// Signature anchor: the intern who drafts and the editor who approves.
// A cheap intern model drafts the next four tokens in the time the big
// editor takes to produce one; the editor then runs ONCE on all four
// drafts in parallel and either approves them (free speedup) or rejects
// at the first disagreement. Returns at the opening, at the parallel-
// verification reveal, and at the mathematical-equivalence section.
export default function SpeculativeDecodingLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="speculative-decoding" />

      {/* ── Opening: the intern / editor frame ───────────────────── */}
      <Prose>
        <p>
          Picture a magazine newsroom at deadline. The senior editor — brilliant,
          slow, thorough — can polish one sentence in the time the cub intern can
          draft four. So you pair them up. The intern races ahead and types out a
          guess at the next four sentences. The editor reads all four at once,
          nods at the ones they would have written themselves, and strikes through
          the first sentence that sounds wrong. That paragraph ships. Everything
          after the strike is thrown away, the intern starts again, and the
          editor moves on — <em>having read four sentences in the time it used
          to take them to write one</em>.
        </p>
        <p>
          That&apos;s the whole lesson. The editor is a big <NeedsBackground slug="code-gpt">autoregressive decoding</NeedsBackground>{' '}
          language model — Llama-70B, say. The intern is a small cheap model — Llama-7B, or
          a 160M-parameter toy — doing the same job at a fraction of the cost. The
          draft is the intern&apos;s guess at the next <code>K</code> tokens. The approval
          is a single parallel forward pass of the editor. And the magic — the
          part that took a DeepMind paper to nail down — is that the tokens you
          ship under this scheme are <em>indistinguishable</em> from what the
          editor would have typed alone. Not approximate. Not &ldquo;good enough.&rdquo;
          Identical in distribution.
        </p>
        <p>
          Here is the most uncomfortable number in modern ML deployment. When Llama-70B generates
          a single token on an H100, the GPU spends roughly <code>97%</code> of the time reading
          weights out of HBM and about <code>3%</code> actually doing arithmetic on them. You
          bought an $30,000 matrix-multiply machine and you&apos;re using it as a glorified
          memory-copy engine.
        </p>
        <p>
          The reason is mechanical. Autoregressive decoding generates tokens one at a time. To
          produce token <code>t+1</code> you need the entire model weights — all 140 GB of
          them — streamed from HBM into the SMs. You then do a batch-of-one forward pass, which
          touches those weights once and throws them out. The arithmetic is trivial. The memory
          read is everything. Inference is{' '}
          <KeyTerm>memory-bandwidth-bound</KeyTerm>, not compute-bound.
        </p>
        <p>
          Which means: if you could somehow do useful work on <em>more</em> tokens per weight-read,
          you&apos;d be getting that work for free. The compute is sitting there idle anyway. This
          lesson is about <KeyTerm>speculative decoding</KeyTerm> — a beautifully cheeky trick
          from Leviathan and Chen (both 2023) that turns idle compute into a 2-3x wall-clock
          speedup, with zero quality loss, and it&apos;s become the default in every serious
          inference stack.
        </p>
      </Prose>

      <Personify speaker="LLM inference">
        I am a sports car idling in a parking lot. My engine is rated for 300 km/h of arithmetic,
        but I am bottlenecked by the size of the fuel line bringing weights from memory. Give me
        more tokens to compute per trip to the fuel pump and I will reward you with almost-free
        speed.
      </Personify>

      <Callout variant="note" title="the arithmetic intensity story">
        A matrix multiply with a batch of <code>1</code> has arithmetic intensity around{' '}
        <code>2</code> FLOPs per byte. An H100 wants around <code>300</code> FLOPs per byte to
        saturate its tensor cores. You are two orders of magnitude below the roofline. Running
        the same matrix multiply on a batch of <code>5</code> uses only slightly more memory —
        the weights are the same — but does <code>5x</code> the compute. Every technique in this
        section is a creative answer to: &ldquo;how do I get the batch size up without waiting?&rdquo;
      </Callout>

      {/* ── The idea ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Speculative decoding&apos;s answer is: <em>guess</em>. Keep a small, cheap{' '}
          <KeyTerm>draft model</KeyTerm> — the intern — next to the big one — the editor. Every
          step, let the intern cheaply draft <code>K</code> tokens autoregressively (say{' '}
          <code>K = 4</code>). Then run the big <KeyTerm>target model</KeyTerm> — the editor —{' '}
          <em>once</em> on all <code>K</code> tokens in parallel. That single forward pass gives
          you the editor&apos;s probability distribution at every position simultaneously. Compare.
          Approve the draft tokens the editor would also have produced. Reject and resample at
          the first disagreement.
        </p>
        <p>
          The numbers are what make this work. Running the editor on one token: one weight
          read, one token produced. Running the editor on <code>K</code> tokens: one weight
          read, up to <code>K</code> tokens approved. If the intern&apos;s guesses agree
          with the editor <code>~70%</code> of the time, you average <code>~3</code> approved
          tokens per editor pass — and the editor&apos;s pass cost barely budged because it was
          memory-bound all along.
        </p>
      </Prose>

      <MathBlock caption="the acceptance rule — modified rejection sampling">
{`for each draft token x with draft prob q(x) and target prob p(x):

    if  p(x) ≥ q(x):        accept unconditionally
    else:                    accept with probability  p(x) / q(x)

on first rejection, resample from the residual distribution:

    p_resid(x)  ∝  max(0, p(x) − q(x))`}
      </MathBlock>

      {/* ── Parallel-verification reveal (anchor return #2) ─────── */}
      <Prose>
        <p>
          <strong>Back to the newsroom for a second.</strong> The part that surprises
          people — even people who write inference code for a living — is that the
          editor reads all four intern-drafted sentences <em>simultaneously</em>.
          Not sentence one, then sentence two, then sentence three, then sentence
          four. All four, at once, in a single pass of the eyes. That&apos;s only
          possible because reading is cheaper than writing. Writing one sentence
          forces the editor to produce a <NeedsBackground slug="softmax">softmax</NeedsBackground>{' '}
          over the vocabulary, pick a token, condition on it, do it again. Reading
          four drafted sentences just asks: &ldquo;for each of these positions, what
          probability would I have assigned to this token?&rdquo; That&apos;s one
          parallel forward pass — the same shape a training batch has. The editor
          is getting four answers for the price of one weight read.
        </p>
      </Prose>

      <SpecDecodingFlow />

      <Prose>
        <p>
          Watch the animation. The intern (small, fast) sprints ahead and drafts four
          candidate tokens. The editor (large, slow) takes those four tokens and runs{' '}
          <em>one</em> forward pass that produces predictions at all four positions in parallel
          — the same way a training batch works. If the first three match, we approve them. The
          fourth disagrees, so we reject it, resample that position from the editor, and discard
          anything after. Net gain: three tokens produced in the time it would have taken to
          produce one, plus a small intern-model tax.
        </p>
      </Prose>

      <Personify speaker="Intern (draft model)">
        I am the guesser. I am <code>10x</code> smaller than my editor and I get things wrong
        constantly — but I get them wrong cheaply, and when I guess right my editor doesn&apos;t
        have to re-do the work. I am not trying to be correct. I am trying to be correct{' '}
        <em>often enough</em>.
      </Personify>

      {/* ── Expected tokens math ─────────────────────────────────── */}
      <Prose>
        <p>
          How much speedup does this actually give you? It depends on the per-token acceptance
          probability <code>α</code>. If each draft token is approved independently with
          probability <code>α</code>, and the intern guesses <code>K</code> tokens, the expected
          number of approved tokens per verification step is a simple geometric-series calculation.
        </p>
      </Prose>

      <MathBlock caption="expected accepted tokens per target pass">
{`E[accepted]  =   (1 − α^(K+1))  /  (1 − α)

         α = 0.7,  K = 4   →    ≈ 2.93 tokens/pass
         α = 0.8,  K = 4   →    ≈ 3.36 tokens/pass
         α = 0.9,  K = 4   →    ≈ 3.78 tokens/pass
         α = 0.5,  K = 4   →    ≈ 1.94 tokens/pass`}
      </MathBlock>

      <Prose>
        <p>
          Translate that to wall-clock speedup. Let <code>c</code> be the cost ratio
          intern-over-editor (say <code>c = 0.1</code> for a 10x smaller intern). Each speculative
          step costs <code>K·c + 1</code> editor-equivalents and produces on average{' '}
          <code>E[accepted]</code> tokens. The ratio is your speedup over plain autoregressive
          decoding:
        </p>
      </Prose>

      <MathBlock caption="end-to-end wall-clock speedup">
{`speedup  =   E[accepted]  /  (K · c + 1)

        α = 0.7, K = 4, c = 0.1   →    2.93 / 1.4   ≈  2.1x
        α = 0.8, K = 4, c = 0.1   →    3.36 / 1.4   ≈  2.4x
        α = 0.9, K = 4, c = 0.05  →    3.78 / 1.2   ≈  3.2x`}
      </MathBlock>

      <AcceptanceRate />

      <Prose>
        <p>
          Slide the acceptance rate up and down. Two things to notice. First, the curve is{' '}
          <em>very</em> sensitive to <code>α</code> in the 0.5-0.9 range — an intern that
          goes from mediocre to good doesn&apos;t linearly improve your throughput, it improves
          it dramatically. Second, cranking <code>K</code> past 4 or 5 rarely helps: once the
          geometric series has mostly converged you&apos;re just paying more draft cost for
          diminishing marginal approved tokens. The sweet spot for most deployments is{' '}
          <code>K ∈ [3, 7]</code>.
        </p>
      </Prose>

      <Personify speaker="Editor (verification)">
        I am a single forward pass of the giant model, and I am the only one allowed to
        stamp tokens into your output. The intern handed me four guesses; I will either approve
        them as &ldquo;yes, I would have said that too&rdquo; or reject them — cryptographically
        preserving the exact distribution you would have gotten without me. I am paranoid about
        correctness so you can be relaxed about speed.
      </Personify>

      <Callout variant="insight" title="why compute is free here">
        Running the editor on <code>K = 4</code> tokens in parallel costs essentially the same
        wall-clock as running it on <code>1</code> token. Both reads use the same weights, and
        the matrix multiply to extend the <NeedsBackground slug="kv-cache">KV cache</NeedsBackground> by{' '}
        <code>4</code> instead of <code>1</code> is a rounding error on an SM that was 97% idle
        to begin with. You&apos;re filling up the unused compute. Speculative decoding is, in
        effect, raising your arithmetic intensity by <code>K</code> almost for free.
      </Callout>

      {/* ── Mathematical-equivalence (anchor return #3) ─────────── */}
      <Prose>
        <p>
          <strong>Now the part that sounds too good to be true.</strong> Every sentence you just
          read is about speed. What about <em>correctness</em>? An intern who guesses wrong 30%
          of the time is shipping drafts into your output stream. How is the final text the same
          as what the editor would have written alone?
        </p>
        <p>
          Go back to the acceptance rule. Look at it as an editor would. Two cases. If the
          editor&apos;s own probability for the drafted token <code>p(x)</code> is already
          at least as high as the intern&apos;s probability <code>q(x)</code>, the editor
          approves unconditionally — of course they would have written that. If the editor&apos;s
          probability is <em>lower</em> than the intern&apos;s, the editor flips a biased coin
          with probability <code>p(x) / q(x)</code>: approve sometimes, reject the rest of the
          time. And on the first rejection, the editor resamples from the leftover mass{' '}
          <code>max(0, p − q)</code>, normalized. Two lines of algebra on the cases give you
          the miracle:
        </p>
      </Prose>

      <MathBlock caption="why speculative decoding is mathematically equivalent to sampling from the editor">
{`Pr[output token = x]
  =  Pr[intern drafts x] · Pr[editor approves x | intern drafted x]
   + Pr[intern drafts any y, rejected] · Pr[resample gives x]

  =  q(x) · min(1, p(x)/q(x))                                     ← approval path
   + (Σ_y q(y) · max(0, 1 − p(y)/q(y))) · p_resid(x)              ← rejection path

  =  min(q(x), p(x))   +   (1 − Σ_y min(q(y), p(y))) · (max(0, p(x)−q(x))) / Z

  =  p(x)                                                          ← the two pieces
                                                                     sum pointwise`}
      </MathBlock>

      <Prose>
        <p>
          That last line is the whole ballgame. The probability that speculative decoding emits
          token <code>x</code> is exactly <code>p(x)</code> — the same as sampling from the
          editor directly. The intern&apos;s distribution <code>q</code> vanishes from the
          final answer. It only ever affected <em>speed</em>, never the output. If you would
          have sampled the string <code>&quot;the cat sat&quot;</code> from the editor at
          temperature <code>0.7</code>, you get the same string with the same probability under
          speculative decoding. An intern who guesses well makes you faster; an intern who
          guesses badly makes you barely faster than the editor alone. Neither one ever makes
          you <em>wrong</em>. That&apos;s why this isn&apos;t a quality/speed tradeoff, and that&apos;s
          why every production inference stack shipped it the moment the paper landed.
        </p>
      </Prose>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, as always. Pure Python on a toy bigram editor to make the acceptance
          rule itself totally concrete. NumPy to verify the distribution is preserved by
          sampling a million times. PyTorch with HuggingFace&apos;s <code>assistant_model</code>{' '}
          argument, which is how you&apos;d actually ship this.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · speculative_scratch.py"
        output={`prompt: [0]
draft:     [3, 7, 2, 9]
verified:  [3, 7, 2]       # 3 accepted
resampled: 5                # first rejection replaced
final seq: [0, 3, 7, 2, 5]`}
      >{`import random, math

# Toy "models": a fast draft q(x) and a slow target p(x).
# Both map a token id (context) to a probability distribution over 10 tokens.
def draft_dist(ctx):      return [max(0.01, math.sin(ctx + i) ** 2) for i in range(10)]
def target_dist(ctx):     return [max(0.01, math.cos(ctx + i) ** 2) for i in range(10)]

def normalize(d):
    s = sum(d)
    return [x / s for x in d]

def sample(dist):
    r, c = random.random(), 0.0
    for i, p in enumerate(dist):
        c += p
        if r < c: return i
    return len(dist) - 1

def speculative_step(prefix, K=4):
    # 1. Draft: roll K tokens autoregressively from the cheap model.
    drafts, ctx = [], prefix[-1]
    for _ in range(K):
        tok = sample(normalize(draft_dist(ctx)))
        drafts.append(tok)
        ctx = tok
    # 2. Target: score all K positions in one "parallel" call (faked by a loop here).
    accepted, ctx = [], prefix[-1]
    for tok in drafts:
        q = normalize(draft_dist(ctx))[tok]
        p = normalize(target_dist(ctx))[tok]
        if random.random() < min(1.0, p / q):
            accepted.append(tok)
            ctx = tok
        else:
            # 3. Resample from residual p' ∝ max(0, p - q).
            p_full = normalize(target_dist(ctx))
            q_full = normalize(draft_dist(ctx))
            resid  = [max(0.0, a - b) for a, b in zip(p_full, q_full)]
            resample = sample(normalize(resid))
            return prefix + accepted + [resample]
    # If all K accepted, sample one extra directly from target.
    return prefix + accepted + [sample(normalize(target_dist(ctx)))]

random.seed(0)
out = speculative_step([0], K=4)
print("final seq:", out)`}</CodeBlock>

      <Prose>
        <p>
          Now the payoff. Run the above many times, collect the output distribution, and
          compare it to a million draws from the editor directly. They match. This is the
          correctness guarantee — speculative decoding&apos;s entire reason for being welcomed
          into production stacks.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · speculative_verify.py"
        output={`target-only empirical:  [0.104 0.099 0.102 0.098 0.099 0.101 0.100 0.097 0.100 0.100]
speculative empirical:  [0.103 0.100 0.103 0.097 0.100 0.099 0.101 0.098 0.099 0.100]
max abs difference:     0.003  (within sampling noise)`}
      >{`import numpy as np

rng = np.random.default_rng(42)

# Target distribution we want to preserve exactly.
p = rng.dirichlet(np.ones(10))
# A deliberately bad draft — lots of disagreement to stress-test correctness.
q = rng.dirichlet(np.ones(10) * 0.3)

def speculative_sample(p, q):
    x = rng.choice(len(q), p=q)             # draft proposes
    if rng.random() < min(1.0, p[x] / q[x]):
        return x                            # accept
    resid = np.maximum(0.0, p - q)
    return rng.choice(len(p), p=resid / resid.sum())   # resample from residual

N = 1_000_000
target_only = np.bincount(rng.choice(len(p), size=N, p=p), minlength=10) / N
specced    = np.bincount([speculative_sample(p, q) for _ in range(N)], minlength=10) / N

print("target-only empirical: ", np.round(target_only, 3))
print("speculative empirical: ", np.round(specced, 3))
print("max abs difference:    ", round(np.abs(target_only - specced).max(), 3))`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for tok in drafts: accept/reject',
            right: 'vectorized p[x] / q[x] comparisons',
            note: 'batch the ratios across all K positions in one go',
          },
          {
            left: 'resid = [max(0, a-b) for ...]',
            right: 'np.maximum(0, p - q)',
            note: 'residual distribution in one broadcast op',
          },
          {
            left: 'sample(normalize(dist))',
            right: 'rng.choice(n, p=dist / dist.sum())',
            note: 'the native sampler — orders of magnitude faster',
          },
        ]}
      />

      <Prose>
        <p>
          PyTorch / HuggingFace has shipped this since <code>transformers 4.30</code>. Hand it
          an <code>assistant_model</code> — the intern — and <code>generate()</code> does the
          rest: draft rollout, editor verification, KV-cache splicing, early-exit on the first
          rejection. You never touch the acceptance math; what you do touch is the model-pair
          selection and the tokenizer compatibility check.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · speculative_hf.py"
        output={`greedy:       The capital of France is Paris, a city of roughly 2.2 million people ...
speculative:  The capital of France is Paris, a city of roughly 2.2 million people ...
outputs identical?  True
wall-clock: greedy 8.34s · speculative 3.61s  →  2.31x speedup`}
      >{`import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok  = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
tgt  = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", torch_dtype=torch.float16, device_map="auto")
drft = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",  torch_dtype=torch.float16, device_map="auto")

prompt = tok("The capital of France is", return_tensors="pt").to(tgt.device)

# Plain greedy decode — one token per forward pass.
t0 = time.time()
out_greedy = tgt.generate(**prompt, max_new_tokens=128, do_sample=False)
t_greedy = time.time() - t0

# Speculative decode — Llama-7B drafts, Llama-13B verifies.
t0 = time.time()
out_spec = tgt.generate(**prompt, max_new_tokens=128, do_sample=False,
                        assistant_model=drft)           # that's it — one kwarg
t_spec = time.time() - t0

print("greedy:      ", tok.decode(out_greedy[0], skip_special_tokens=True)[:90], "...")
print("speculative: ", tok.decode(out_spec[0],   skip_special_tokens=True)[:90], "...")
print("outputs identical? ", torch.equal(out_greedy, out_spec))
print(f"wall-clock: greedy {t_greedy:.2f}s · speculative {t_spec:.2f}s  →  {t_greedy / t_spec:.2f}x speedup")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch (production)"
        rows={[
          {
            left: 'manual draft loop + accept/reject',
            right: 'generate(..., assistant_model=drft)',
            note: 'HuggingFace does the whole dance in one kwarg',
          },
          {
            left: 'recompute target_dist per token',
            right: 'KV cache reuse across draft + verify',
            note: 'the thing that actually makes it fast on GPU',
          },
          {
            left: 'K fixed at 4 in code',
            right: 'adaptive K — grows after long accept runs',
            note: 'vLLM / TGI dynamically resize the speculation window',
          },
        ]}
      />

      <Callout variant="insight" title="the three layers, recapped">
        Pure Python makes the acceptance rule unmistakable — no framework magic, just the
        intern proposing and the editor approving with <code>p/q</code> arithmetic. NumPy
        proves the distribution is preserved, because until you&apos;ve seen <code>1M</code>{' '}
        samples match the editor&apos;s empirical distribution exactly, the whole thing smells
        too good to be true. And PyTorch / HF is the one-line reality: a grown-up inference stack
        already implements this, and your job is picking good model pairs and measuring
        acceptance rate — not re-deriving the math.
      </Callout>

      {/* ── Variants ────────────────────────────────────────────── */}
      <Callout variant="note" title="Medusa — the intern IS the editor">
        Instead of running a separate smaller model, Cai et al. (2024) bolt <code>k</code> tiny
        prediction heads onto the editor&apos;s final hidden state. Head <code>i</code> predicts
        the token <code>i</code> steps ahead — one editor, <code>k</code> interns built into
        its own head. No second model to host, no tokenizer mismatch, and the &ldquo;draft&rdquo;
        is conditioned on the editor&apos;s own representation so acceptance rates are typically
        high (70-90%). Trade-off: you need to fine-tune the heads, which most open weights
        haven&apos;t done out of the box.
      </Callout>

      <Callout variant="note" title="Eagle — guessing features, not tokens">
        Eagle (Li et al. 2024) takes Medusa&apos;s &ldquo;use the editor&apos;s own features&rdquo;
        idea one step further: the intern guesses the editor&apos;s next <em>hidden state</em>{' '}
        (not the next token). Since the hidden state contains far more information than a
        discrete token id, the intern is a better guesser — reported acceptance rates cross 0.9
        on instruction data, pushing real-world speedup closer to <code>3x</code>. It&apos;s the
        current SOTA on the Spec-Bench leaderboard.
      </Callout>

      <Callout variant="note" title="lookahead decoding — no intern at all">
        A different branch of the family tree: have the editor itself produce <code>K</code>{' '}
        speculative n-grams from its recent output, then verify them in a single parallel pass.
        No second model to train or host — the editor does its own guessing on the side. The
        acceptance rates are lower but you get a modest 1.5-2x for free. Nice when memory for
        a second model is not available.
      </Callout>

      <Prose>
        <p>
          You will not implement any of these from scratch in 2026. <strong>vLLM</strong>,{' '}
          <strong>TGI</strong> (Hugging Face&apos;s Text Generation Inference),{' '}
          <strong>TensorRT-LLM</strong>, and <strong>ExLlamaV2</strong> all support some mix of
          vanilla speculative decoding, Medusa, Eagle, and lookahead. Your job as a deployment
          engineer is choosing which one, measuring acceptance on <em>your</em> traffic
          distribution, and sizing the intern model correctly.
        </p>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Temperature must match:</strong> the mathematical
          guarantee relies on sampling the intern at the same temperature as the editor. Mix a
          <code> t=1.0</code> draft with a <code>t=0.7</code> target and you break the
          distribution-preservation proof. Some stacks silently fix this; some don&apos;t. Check.
        </p>
        <p>
          <strong className="text-term-amber">Tokenizer mismatch is fatal:</strong> intern and
          editor must share a tokenizer. Llama-7B drafts for Llama-13B cleanly because they
          share vocab. A Llama intern for a Mistral editor is nonsense — token ids don&apos;t
          align. If you must cross families, use a distilled intern trained on the editor&apos;s
          vocab.
        </p>
        <p>
          <strong className="text-term-amber">End-of-sequence handling:</strong> if the intern
          drafts <code>&lt;eos&gt;</code> mid-speculation, the editor must get a chance to
          <em> reject</em> it — otherwise a bad guess can truncate legitimate outputs. Every
          production stack handles this; hand-rolled implementations forget it constantly.
        </p>
        <p>
          <strong className="text-term-amber">Batch-size interaction:</strong> at batch size 32+,
          your editor is no longer memory-bound — it&apos;s actually using its compute.
          Speculative decoding&apos;s speedup shrinks or disappears because you&apos;re not
          filling idle compute anymore. It shines on latency-sensitive serving (batch 1-4) and
          dims on throughput-oriented batch serving.
        </p>
        <p>
          <strong className="text-term-amber">Acceptance rate is workload-dependent:</strong> an
          intern that approves 80% on Wikipedia-like prose may drop to 50% on code or math.
          Measure on your actual traffic, not on benchmarks from the paper.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Speculative-decode Llama-13B with Llama-7B as the intern">
        <p>
          Load <code>meta-llama/Llama-2-13b-hf</code> as the editor and{' '}
          <code>meta-llama/Llama-2-7b-hf</code> as the intern. Run a batch of 20 prompts —
          mix of prose, code, and math — to <code>max_new_tokens=256</code>. For each prompt,
          time plain greedy decode versus <code>generate(..., assistant_model=drft)</code> and
          record the per-prompt speedup.
        </p>
        <p className="mt-2">
          Then <em>verify correctness</em>: assert <code>torch.equal(out_greedy, out_spec)</code>{' '}
          for every prompt. If any prompt disagrees, you have a tokenizer, temperature, or
          sampling-config bug — this is not a &ldquo;well, it&apos;s mostly the same&rdquo;
          thing, the outputs must match byte-for-byte when <code>do_sample=False</code>.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: log the per-prompt approve/reject rate (HF exposes it via{' '}
          <code>generation_config.output_scores</code>). Notice how prose typically hits 0.7-0.8
          and code drops to 0.4-0.6. Plot it. This histogram is the most important chart for
          sizing a spec-decode deployment.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Double-bonus: swap in a 1B-parameter intern (<code>TinyLlama</code>-style) and compare.
          You trade a lower acceptance rate for a much cheaper draft — often a net win.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> LLM inference is memory-bandwidth-bound, so
          any trick that increases useful work per weight-read is essentially free speed.
          Speculative decoding is the cleanest example: the intern drafts <code>K</code> tokens,
          the editor approves or rejects all <code>K</code> in one parallel verification pass,
          and a careful acceptance rule preserves the editor&apos;s output distribution exactly.
          Typical speedup is 2-3x on real workloads with no quality loss. Medusa and Eagle push
          this further by making the intern a head of the editor itself. Every production
          inference stack (vLLM, TGI, TensorRT-LLM) supports this — you will configure it, not
          implement it.
        </p>
        <p>
          <strong>Next up — Continuous Batching.</strong> The intern/editor trick fills idle
          compute inside <em>one</em> request. But a real production server isn&apos;t handling
          one request — it&apos;s handling dozens at once, each at a different point in its
          own decode. Naive batching forces the whole group to wait for the slowest conversation
          to finish before starting a new one. Continuous batching is the serving-system trick
          that slots new requests into an ongoing batch at every step, and when you combine it
          with speculative decoding you unlock the throughput numbers that make LLM APIs
          economically viable. That&apos;s the lesson after this one.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Fast Inference from Transformers via Speculative Decoding',
            author: 'Leviathan, Kalman, Matias',
            venue: 'ICML 2023',
            url: 'https://arxiv.org/abs/2211.17192',
          },
          {
            title: 'Accelerating Large Language Model Decoding with Speculative Sampling',
            author: 'Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper',
            venue: 'DeepMind 2023',
            url: 'https://arxiv.org/abs/2302.01318',
          },
          {
            title: 'Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads',
            author: 'Cai, Li, Geng, Peng, Lee, Chen, Dao',
            venue: 'ICML 2024',
            url: 'https://arxiv.org/abs/2401.10774',
          },
          {
            title: 'EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty',
            author: 'Li, Wei, Zhang, Zhang',
            venue: 'ICML 2024',
            url: 'https://arxiv.org/abs/2401.15077',
          },
          {
            title: 'Break the Sequential Dependency of LLM Inference Using Lookahead Decoding',
            author: 'Fu, Bailis, Stoica, Zhang',
            year: 2024,
            url: 'https://arxiv.org/abs/2402.02057',
          },
          {
            title: 'vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention',
            author: 'Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica',
            venue: 'SOSP 2023 — speculative decoding landed in v0.3',
            url: 'https://arxiv.org/abs/2309.06180',
          },
        ]}
      />
    </div>
  )
}
