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
import SamplingStrategies from '../widgets/SamplingStrategies'
import DistributionShaper from '../widgets/DistributionShaper'

// Signature anchor: teaching a monologuer to have a conversation. A raw GPT
// only knows how to keep talking — it has never encountered the rhythm of
// turn-taking. This lesson teaches it the user: / assistant: beat by showing
// it thousands of transcripts that follow that beat, then treats the special
// tokens as stage directions so it knows when to stop. The monologuer shows
// up at the opening, the chat-template reveal, and the "runaway-mouth"
// failure-mode section.
export default function MakeGptTalkBackLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="make-gpt-talk-back" />

      {/* ── Opening: the monologuer ─────────────────────────────── */}
      <Prose>
        <p>
          You trained a GPT. You prompt it with <code>&quot;What&apos;s the capital of
          France?&quot;</code> and it replies: <em>&ldquo;What&apos;s the capital of
          Germany? What&apos;s the capital of Italy? What&apos;s the capital of
          Spain?…&rdquo;</em> It does not stop. It will never stop. You trained a
          monologuer — a model that only knows one move, which is to keep talking.
          It has read the entire internet and learned exactly one social skill:
          continue the document.
        </p>
        <p>
          The base model has never, in its entire training life, seen a
          conversation as a <em>conversation</em>. It has seen conversations as
          flat text — long strips of prose where one sentence follows another
          with no sense of whose turn it is. It does not know that you and it
          are two separate speakers. It does not know that it is supposed to
          finish a thought and then hand the microphone back. It is a guest at
          a dinner party who has been holding forth for three hours and has
          not, as far as anyone can tell, noticed that nobody else has spoken.
        </p>
        <p>
          This lesson is about teaching that monologuer to have a conversation.
          Two ingredients, and only two. First, a <KeyTerm>chat template</KeyTerm>{' '}
          — a fixed rhythm of <code>user:</code> / <code>assistant:</code> turns
          marked with special tokens the model learns to treat as stage
          directions (&ldquo;enter, speak, exit&rdquo;). Second, a dataset of a
          few thousand transcripts that follow that rhythm exactly, fine-tuned
          in via{' '}
          <NeedsBackground slug="supervised-fine-tuning">SFT</NeedsBackground>.
          After that the model has learned one new thing: when it sees the
          closing stage cue, it stops.
        </p>
      </Prose>

      <Personify speaker="A freshly pretrained GPT">
        I have read the entire internet. I have 175 billion parameters.
        I will happily continue any text you give me — your email, your shopping
        list, the dinner you are currently still eating — until you cut the
        power. Turn-taking? What is turn-taking? Am I supposed to be waiting
        for something?
      </Personify>

      {/* ── The chat template: stage directions ─────────────────── */}
      <Prose>
        <p>
          Here is the idea, stripped of all jargon. Every transcript in the
          fine-tuning set looks the same. There is a <em>system</em> stage cue
          (the director whispering &ldquo;be helpful, be honest&rdquo;), a{' '}
          <em>user</em> stage cue (&ldquo;the user speaks now&rdquo;), and an{' '}
          <em>assistant</em> stage cue (&ldquo;you speak now&rdquo;). Each cue
          is a{' '}
          <NeedsBackground slug="tokenizer-bpe">special token</NeedsBackground>{' '}
          — a single vocabulary entry the tokenizer refuses to split, so the
          model always sees the full cue as one unit. At the end of the
          assistant&apos;s turn there is a closing cue: <em>stop talking</em>.
          That is the whole chat template. A transcript looks like this:
        </p>
      </Prose>

      <MathBlock caption="the chat template — stage directions around a turn">
{`<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What's the capital of France?<|im_end|>
<|im_start|>assistant
Paris.<|im_end|>`}
      </MathBlock>

      <Prose>
        <p>
          Look at each line as theater. <code>&lt;|im_start|&gt;</code> is the
          curtain rising. <code>system</code>, <code>user</code>,{' '}
          <code>assistant</code> are the roles. <code>&lt;|im_end|&gt;</code> is
          the curtain falling — &ldquo;this speaker is done, next speaker
          please.&rdquo; During fine-tuning you show the model ten thousand
          transcripts in exactly this shape, and the gradient descent does what
          it always does: every special token accumulates a meaning from the
          company it keeps. <code>&lt;|im_end|&gt;</code> ends up meaning{' '}
          <em>&ldquo;I have finished my thought; the other speaker goes
          now&rdquo;</em> — because that is the only context in which it ever
          appeared during training. The stage directions become real.
        </p>
        <p>
          This is the entire reason chat models &ldquo;know&rdquo; how to
          take turns. There is no separate module, no special &ldquo;turn
          controller,&rdquo; no heuristic in the sampling code. There is a
          pretrained next-token predictor, a handful of new vocabulary entries
          with strong and narrow meanings, and the same{' '}
          <NeedsBackground slug="supervised-fine-tuning">supervised
          fine-tuning</NeedsBackground> recipe you already met. That&apos;s it.
          The monologuer became a conversationalist by being shown what a
          conversation looks like — in one consistent format — and told
          &ldquo;be like this.&rdquo;
        </p>
      </Prose>

      <Personify speaker="The <|im_end|> token">
        Before training, I meant nothing. A random vector in embedding space,
        cluttering up the vocabulary. After training, I mean exactly one thing:{' '}
        <em>stop talking</em>. Every time the model sampled me, the loss said
        &ldquo;correct, the next thing is another speaker&apos;s cue.&rdquo;
        Now my embedding is a cliff. The model walks up to me and steps off.
      </Personify>

      {/* ── Why it works: the mechanical story ──────────────────── */}
      <Prose>
        <p>
          Let&apos;s slow down on the mechanical picture, because this is where
          a lot of tutorials wave hands. The model is a conditional
          distribution: given the tokens so far, predict the next one. Nothing
          more. When you ask it &ldquo;how does the assistant know to
          stop?&rdquo; the honest answer is: it doesn&apos;t know anything. It
          just assigns a very high probability to the{' '}
          <code>&lt;|im_end|&gt;</code> token at the end of a natural
          assistant turn, because every such turn in the fine-tuning data
          ended with that token. Your inference loop watches the samples go by,
          and the instant that token shows up, you stop pulling new ones.
        </p>
        <p>
          So turn-taking is split across two places. The <em>model</em> learns
          to <em>predict</em> the end-of-turn token. The <em>inference loop</em>{' '}
          learns to <em>respect</em> it — when the sampler returns{' '}
          <code>&lt;|im_end|&gt;</code>, the loop breaks. The model never stops
          itself; it just nominates a place to stop, and the surrounding code
          honors the nomination. That division of labor is subtle and it will
          come back to bite us in about three paragraphs.
        </p>
      </Prose>

      {/* ── Widget 1: sampling strategies ───────────────────────── */}
      <Prose>
        <p>
          Setting the template aside for a moment: once the model is
          predicting the next token in a conversation, you still have to turn
          the probability distribution into an actual token. That decision is
          the <KeyTerm>decoding strategy</KeyTerm>. Pick the wrong one and the
          fanciest transformer in the world sounds like a broken keyboard.
          Pick the right one and the same weights produce fluent, surprising,
          useful prose. Play with the hero widget below — type a prompt,
          toggle between greedy / temperature / top-k / nucleus, and watch
          tokens fall out one at a time beside the live probability
          distribution the strategy is sampling from.
        </p>
      </Prose>

      <SamplingStrategies />

      <Prose>
        <p>
          Two things to notice. Greedy always produces the same output for the
          same prompt — run it twice and the transcript is identical. The
          other three diverge every run because they&apos;re stochastic. And
          the <em>shape</em> of the distribution is what each strategy is
          manipulating — greedy picks the peak, temperature rescales the
          whole thing, top-k and top-p truncate the tail. Let&apos;s derive
          each one.
        </p>
      </Prose>

      {/* ── Greedy & Temperature math ────────────────────────────── */}
      <Prose>
        <p>
          Set notation first. After the forward pass, the model emits a vector
          of <KeyTerm>logits</KeyTerm> <code>z ∈ ℝᵛ</code> over a vocabulary of
          size <code>V</code> (typically 50k–250k). A softmax turns logits into
          a probability distribution <code>p</code>. We sample a token from{' '}
          <code>p</code>, append, repeat. The decoding strategy is{' '}
          <em>a choice of how to turn <code>z</code> into the next token</em>.
        </p>
        <p>The simplest choice — <strong>greedy decoding</strong>:</p>
      </Prose>

      <MathBlock caption="greedy — the dumbest thing that sometimes works">
{`tₜ  =  argmaxᵢ  zᵢ

(equivalently: argmaxᵢ  pᵢ   — softmax is monotone, so argmax doesn't care)`}
      </MathBlock>

      <Prose>
        <p>
          Pick the highest-scoring token, every time. Deterministic, fast, and
          famously prone to repetition loops: once the model latches onto a
          high-probability bigram (<code>&quot;the cat&quot;</code> →{' '}
          <code>&quot;cat sat&quot;</code> → <code>&quot;sat on&quot;</code> →{' '}
          <code>&quot;on the&quot;</code> → <code>&quot;the cat&quot;</code>),
          greedy rides the cycle forever. Right call for short, factual
          outputs (code completion, yes/no questions, structured extraction).
          Wrong call for anything that needs to feel like a reply.
        </p>
        <p>
          The fix is to put randomness back in. Scale the logits by a{' '}
          <KeyTerm>temperature</KeyTerm> <code>T</code> before softmaxing, then
          sample from the resulting distribution:
        </p>
      </Prose>

      <MathBlock caption="temperature sampling — softmax with a knob">
{`pᵢ(T)  =        exp(zᵢ / T)
           ─────────────────────
            Σⱼ exp(zⱼ / T)

tₜ  ~  Categorical(p(T))

T = 1     →  raw model distribution        (as-trained)
T → 0⁺    →  all mass collapses to argmax  (= greedy)
T < 1     →  distribution sharpens         (safer, more confident)
T > 1     →  distribution flattens         (weirder, more creative)
T → ∞     →  uniform over vocabulary       (random keysmash)`}
      </MathBlock>

      <Prose>
        <p>
          Dividing by a small <code>T</code> blows up the differences between
          logits, so the softmax becomes peakier — the top token dominates.
          A large <code>T</code> shrinks differences, so the softmax flattens
          — even unlikely tokens get a real chance. <code>T</code> is a single
          scalar that interpolates between &ldquo;greedy&rdquo; and &ldquo;uniform
          random.&rdquo; Most chat models default to <code>T ≈ 0.7–1.0</code>.
        </p>
      </Prose>

      <Personify speaker="Temperature">
        I&apos;m the creativity knob. Crank me low and the assistant says the
        safe thing every time — boring, reliable, repetitive. Crank me high
        and the assistant rolls dice on words it&apos;s barely confident
        about — surprising, occasionally ungrammatical. I&apos;m a single
        scalar, and I&apos;m probably the most important inference
        hyperparameter you&apos;ll ever tune.
      </Personify>

      {/* ── Top-k & Top-p math ───────────────────────────────────── */}
      <Prose>
        <p>
          Temperature has a problem. Even at sensible values like{' '}
          <code>T = 1</code>, the tail of the vocabulary distribution is long —
          thousands of tokens each with probability <code>10⁻⁵</code> or
          smaller. Sum them and they make up a non-trivial chunk of probability
          mass, so occasionally you&apos;ll sample one. When the top 20 guesses
          are all sensible continuations and the 100,000th token is a Unicode
          oddity that breaks your JSON parser, you don&apos;t want a 2% chance
          of picking from the tail. You want to <em>cut the tail off</em>.
        </p>
        <p>
          Two standard ways. <strong>Top-k</strong> cuts a fixed number of
          tokens from the top:
        </p>
      </Prose>

      <MathBlock caption="top-k — fixed-size truncation">
{`V_k  =  indices of the k highest values in p

p'ᵢ  =  { pᵢ / Σⱼ∈V_k pⱼ     if i ∈ V_k
        { 0                   otherwise

tₜ  ~  Categorical(p')

(Fan, Lewis, Dauphin — 2018.  k = 50 is the common default.)`}
      </MathBlock>

      <Prose>
        <p>
          Keep the <code>k</code> highest-probability tokens, zero out
          everything else, renormalize so the surviving mass sums to 1, sample
          from that. Simple, fast, effective. The problem: <code>k</code> is
          fixed, but the distribution&apos;s sharpness changes from token to
          token. Sometimes the model is <em>very</em> confident (one or two
          tokens matter); sometimes it&apos;s deeply uncertain (50 tokens share
          the mass). A fixed <code>k = 50</code> is too wide in the first case
          and too narrow in the second.
        </p>
        <p>
          <strong>Nucleus sampling</strong> — also called <strong>top-p</strong>{' '}
          — fixes that by cutting at a fixed cumulative probability instead:
        </p>
      </Prose>

      <MathBlock caption="nucleus (top-p) — adaptive truncation">
{`Sort p descending, giving p₍₁₎ ≥ p₍₂₎ ≥ … ≥ p₍ᵥ₎.

Find the smallest n such that  Σᵢ₌₁ⁿ p₍ᵢ₎  ≥  p.

V_p  =  { (1), (2), …, (n) }      ← the "nucleus"

p'ᵢ  =  { pᵢ / Σⱼ∈V_p pⱼ    if i ∈ V_p
        { 0                  otherwise

tₜ  ~  Categorical(p')

(Holtzman et al. — 2019.  p = 0.9 or 0.95 standard.)`}
      </MathBlock>

      <Prose>
        <p>
          Sort tokens by probability, walk down the list accumulating mass,
          stop the moment you&apos;ve covered <code>p</code> of the total.
          Keep that set (the &ldquo;nucleus&rdquo;), zero the rest,
          renormalize, sample. When the distribution is peaky the nucleus
          contains one or two tokens. When it&apos;s flat it contains
          hundreds. <em>The size adapts to the model&apos;s actual
          uncertainty</em>, which is exactly the thing fixed <code>k</code>{' '}
          misses.
        </p>
        <p>
          In production you usually stack all three: apply temperature first
          to shape the distribution, then top-p to cut the tail adaptively,
          then top-k as an absolute cap so you never consider more than{' '}
          <code>k</code> tokens no matter how flat the tail is. OpenAI,
          Anthropic, and Google all ship some version of this pipeline. The
          panel below shows it happening live — start with a raw next-token
          distribution, apply <code>T</code>, then top-k, then top-p, and
          watch which tokens survive each stage.
        </p>
      </Prose>

      <DistributionShaper />

      <Callout variant="insight" title="the order matters, a little">
        Most reference implementations (HuggingFace, vLLM) apply{' '}
        <em>temperature → top-k → top-p</em> in that order, but the difference
        vs <em>temperature → top-p → top-k</em> is usually negligible when
        both thresholds are reasonable. What matters is that you apply
        temperature <em>before</em> truncation — if you sharpen the
        distribution after cutting its tail, you&apos;re cooking the
        proportions of the surviving tokens, not the full distribution.
      </Callout>

      <Personify speaker="Top-p">
        I&apos;m the adaptive chooser. When the model is confident — one
        obvious next word — I keep just that word. When the model is
        uncertain — a paragraph could go fifty different ways — I open up
        and let fifty candidates in. I don&apos;t care about absolute counts.
        I care about <em>covering enough of your beliefs to sample honestly
        from them</em>.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Four strategies, three phases of your coding life. Start with pure
          Python — no numpy, no tensors, just loops and lists — so every step
          is visible. This is also how you&apos;d implement{' '}
          <NeedsBackground slug="code-gpt">generation</NeedsBackground> in a
          tutorial before reaching for a library.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · sampling_scratch.py"
        output={`logits:   [2.5, 1.8, 1.2, 0.9, 0.3, -0.2, -1.1]
greedy → 0
T=1.0  → 2     (sampled)
T=0.5  → 0     (sampled — sharpened)
top-k=3 (k=3) → 1     (sampled from {0,1,2})
top-p=0.7     → 0     (sampled from {0,1})  nucleus size=2`}
      >{`import math, random

def softmax(z, T=1.0):
    z = [zi / T for zi in z]
    m = max(z)                               # subtract max for numerical stability
    exps = [math.exp(zi - m) for zi in z]
    Z = sum(exps)
    return [e / Z for e in exps]

def greedy(z):
    return max(range(len(z)), key=lambda i: z[i])      # argmax

def sample_temperature(z, T=1.0):
    p = softmax(z, T)
    return random.choices(range(len(p)), weights=p, k=1)[0]

def sample_topk(z, k=50, T=1.0):
    p = softmax(z, T)
    # keep indices of k largest probabilities
    top = sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:k]
    mass = sum(p[i] for i in top)
    weights = [p[i] / mass if i in set(top) else 0.0 for i in range(len(p))]
    return random.choices(range(len(p)), weights=weights, k=1)[0]

def sample_topp(z, p_thresh=0.9, T=1.0):
    p = softmax(z, T)
    order = sorted(range(len(p)), key=lambda i: p[i], reverse=True)
    # walk the sorted list, accumulate mass, keep until you cross p_thresh
    nucleus, cum = [], 0.0
    for i in order:
        nucleus.append(i)
        cum += p[i]
        if cum >= p_thresh:
            break
    mass = sum(p[i] for i in nucleus)
    weights = [p[i] / mass if i in set(nucleus) else 0.0 for i in range(len(p))]
    return random.choices(range(len(p)), weights=weights, k=1)[0]

logits = [2.5, 1.8, 1.2, 0.9, 0.3, -0.2, -1.1]
print("logits:  ", logits)
print("greedy →", greedy(logits))
print("T=1.0  →", sample_temperature(logits, T=1.0),   "   (sampled)")
print("T=0.5  →", sample_temperature(logits, T=0.5),   "   (sampled — sharpened)")
print("top-k=3 →", sample_topk(logits, k=3),           "   (sampled from {0,1,2})")
print("top-p=0.7 →", sample_topp(logits, p_thresh=0.7),"   (sampled from {0,1})")`}</CodeBlock>

      <Prose>
        <p>
          Now vectorize with NumPy. Same algorithms, but every <code>for</code>{' '}
          over the vocabulary becomes an array op — the difference between
          200 ms per token and 2 ms per token when <code>V = 50,000</code>.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · sampling_numpy.py"
      >{`import numpy as np

def softmax(z, T=1.0):
    z = z / T
    z = z - z.max()                               # stability — shift, softmax invariant
    e = np.exp(z)
    return e / e.sum()

def greedy(z):
    return int(np.argmax(z))

def sample_temperature(z, T=1.0, rng=np.random):
    p = softmax(z, T)
    return int(rng.choice(len(p), p=p))

def sample_topk(z, k=50, T=1.0, rng=np.random):
    p = softmax(z, T)
    # find the k-th largest probability, zero out anything smaller
    kth = np.partition(p, -k)[-k]
    p = np.where(p >= kth, p, 0.0)
    p = p / p.sum()                               # renormalize — must not forget this
    return int(rng.choice(len(p), p=p))

def sample_topp(z, p_thresh=0.9, T=1.0, rng=np.random):
    p = softmax(z, T)
    order = np.argsort(p)[::-1]                   # indices sorted descending
    sorted_p = p[order]
    cum = np.cumsum(sorted_p)
    cutoff = np.searchsorted(cum, p_thresh) + 1   # first index where cum >= p
    keep = order[:cutoff]
    mask = np.zeros_like(p); mask[keep] = 1.0
    p = p * mask
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))

rng = np.random.default_rng(0)
logits = np.array([2.5, 1.8, 1.2, 0.9, 0.3, -0.2, -1.1])
print("greedy   →", greedy(logits))
print("T=0.7    →", sample_temperature(logits, T=0.7, rng=rng))
print("top-k=3  →", sample_topk(logits, k=3, rng=rng))
print("top-p=0.9 →", sample_topp(logits, p_thresh=0.9, rng=rng))`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'sum(exp(zi - m) for zi in z)',
            right: 'np.exp(z - z.max()).sum()',
            note: 'vector softmax — one line, same numerical trick',
          },
          {
            left: 'sorted(range(V), key=p.__getitem__)[:k]',
            right: 'np.partition(p, -k)[-k]   # the k-th threshold',
            note: 'O(V) instead of O(V log V) — partition, don\u2019t sort',
          },
          {
            left: 'manual loop accumulating mass',
            right: 'np.cumsum + np.searchsorted',
            note: 'binary-search the cumulative — nucleus size in O(log V)',
          },
        ]}
      />

      <Prose>
        <p>
          And in PyTorch — what you&apos;ll actually call in production.{' '}
          <code>F.softmax</code> does the softmax with the stability trick
          baked in; <code>torch.multinomial</code> does the Categorical
          sample; the top-k / top-p work happens through a small mask applied
          to the logits <em>before</em> softmax, which is the clean pattern
          used in every reference transformer implementation.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · sampling_pytorch.py"
        output={`greedy token   : 0
temp-sampled   : 2
top-k sampled  : 1
top-p sampled  : 0`}
      >{`import torch
import torch.nn.functional as F

@torch.no_grad()                                   # inference — no gradients
def sample(logits, T=1.0, top_k=None, top_p=None):
    # logits: (V,) or (B, V). Work in 1D here for clarity.
    logits = logits / T

    # top-k filter — keep only the k largest logits
    if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        # anything strictly below the k-th largest value → -inf
        logits = torch.where(logits < v[-1], torch.tensor(float('-inf')), logits)

    # top-p filter — sort, compute cumulative softmax, mask below-threshold tokens
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p                        # everything strictly past the nucleus
        remove[1:] = remove[:-1].clone()            # shift right — keep the token that crossed
        remove[0] = False                           # always keep the top token
        sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)               # stability handled internally
    return torch.multinomial(probs, num_samples=1).item()

logits = torch.tensor([2.5, 1.8, 1.2, 0.9, 0.3, -0.2, -1.1])
torch.manual_seed(0)
print("greedy token   :", int(torch.argmax(logits)))
print("temp-sampled   :", sample(logits, T=0.7))
print("top-k sampled  :", sample(logits, T=1.0, top_k=3))
print("top-p sampled  :", sample(logits, T=1.0, top_p=0.9))`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'p = np.where(p >= kth, p, 0); p /= p.sum()',
            right: 'logits.masked_fill_(mask, -inf); F.softmax(logits)',
            note: 'mask in logit space before softmax — no renormalize needed',
          },
          {
            left: 'rng.choice(V, p=p)',
            right: 'torch.multinomial(probs, num_samples=1)',
            note: 'GPU-resident Categorical sample — no cpu round-trip',
          },
          {
            left: '@staticmethod helpers, manual masks',
            right: '@torch.no_grad()  wrapping everything',
            note: 'inference idiom — tell autograd not to build a graph',
          },
        ]}
      />

      <Callout variant="insight" title="mask in logit space, not probability space">
        Setting filtered-out logits to <code>-inf</code> before softmax is
        cleaner than zeroing out probabilities after. The{' '}
        <code>exp(-inf) = 0</code> identity means softmax automatically
        renormalizes over the surviving tokens — no separate divide-by-sum,
        no risk of numerical drift. Every production sampler does it this
        way.
      </Callout>

      {/* ── When the monologuer forgets to stop ─────────────────── */}
      <Prose>
        <p>
          Back to the monologuer, because here is where the stage-direction
          story gets its most common failure mode. The model nominates the
          stop. The inference loop respects the stop. Both halves must
          actually happen. When either one breaks, you get a runaway mouth —
          the assistant finishes its turn, then keeps going, writes a fake{' '}
          <code>user:</code> message to itself, answers it, writes another
          one, answers that. You have invited one guest to dinner and it
          somehow became three, all of them the same guest.
        </p>
        <p>
          Three ways this happens, roughly in order of how often they bite
          people in the wild:
        </p>
        <ul>
          <li>
            <strong>The stop token isn&apos;t in the sampler&apos;s stop
            list.</strong> You trained the model to emit{' '}
            <code>&lt;|im_end|&gt;</code>, but your generation loop only stops
            on <code>&lt;|endoftext|&gt;</code>. The curtain falls; nobody
            notices; the model keeps going because nothing told it not to, and
            the next token it samples is the stage cue for a new user turn.
            Now it&apos;s hallucinating the other half of the transcript. Fix:
            pass <em>every</em> end-of-turn token id to the sampler&apos;s
            stop list.
          </li>
          <li>
            <strong>The chat template at inference doesn&apos;t match the one
            from fine-tuning.</strong> You trained on{' '}
            <code>&lt;|im_start|&gt;user\n…&lt;|im_end|&gt;</code> but at
            inference you&apos;re handing the model a plain{' '}
            <code>&quot;User: …&quot;</code> prefix with no special tokens.
            The model is a stickler for its cues. If the tokens aren&apos;t
            there, it doesn&apos;t know which scene it&apos;s in, and the
            learned stopping behavior quietly dissolves. Fix: use the exact
            tokenizer&apos;s <code>apply_chat_template</code>.
          </li>
          <li>
            <strong>The fine-tuning data was inconsistent.</strong> Half the
            transcripts ended with <code>&lt;|im_end|&gt;</code>, half
            didn&apos;t. The model learned the end-of-turn cue half-heartedly
            — a probability of 0.3 where it should be 0.95 — and now greedy
            decoding walks right past it. Fix: audit the data; every
            assistant turn must end in the same closing cue, no exceptions.
          </li>
        </ul>
      </Prose>

      <Callout variant="note" title="repetition penalty — the other anti-loop hack">
        Even with top-p and decent temperature, long generations drift into
        repetition. The standard fix is a <strong>repetition penalty</strong>:
        for every token already in the prompt or output, divide its logit by{' '}
        <code>θ &gt; 1</code>. Typical <code>θ = 1.1–1.3</code>. It&apos;s
        crude — it penalizes <em>&ldquo;the&rdquo;</em> appearing twice as
        much as a niche token appearing twice — but it reliably kills{' '}
        <em>&ldquo;the the the&rdquo;</em> loops without incoherence. More
        recent alternatives: <em>frequency penalty</em>,{' '}
        <em>presence penalty</em> (OpenAI&apos;s two knobs), and{' '}
        <em>DRY (Don&apos;t Repeat Yourself)</em>, which penalizes only
        exact n-gram repetition.
      </Callout>

      <Callout variant="note" title="beam search — the thing you don't want for chat">
        One strategy we haven&apos;t covered: <strong>beam search</strong>.
        Instead of committing to one token at each step, keep the top{' '}
        <code>B</code> partial sequences ranked by cumulative
        log-probability; at each step, extend each beam by its top-<code>B</code>{' '}
        continuations, then re-rank and keep only the overall top{' '}
        <code>B</code>. It provably finds higher-likelihood sequences than
        greedy or sampling — which is exactly why machine translation uses
        it (BLEU rewards matching a reference, likelihood is a good proxy).
        For <em>open-ended</em> generation (chat, story writing, code) it
        produces famously bland output: sequences that are maximally likely
        tend to be maximally generic. Rule of thumb: beam for translation,
        summarization, and structured outputs; nucleus for everything else.
      </Callout>

      {/* ── Stopping conditions: the full stage-management layer ── */}
      <Prose>
        <p>
          Pulling all the stopping mechanisms together, here is the full
          stage-management layer sitting around the sampler. Three mechanisms,
          usually stacked:
        </p>
        <ul>
          <li>
            <strong><code>max_tokens</code>.</strong> Hard cap on the length
            of the generation. Always set one — cost and latency are both
            linear in tokens, and it saves you when the first two mechanisms
            fail.
          </li>
          <li>
            <strong>EOS / end-of-turn token.</strong> Most modern tokenizers
            have a special <code>&lt;|endoftext|&gt;</code> or{' '}
            <code>&lt;|im_end|&gt;</code> token; the model is trained to emit
            it when it&apos;s done speaking. Stop sampling the moment it&apos;s
            produced. This is the end-of-scene cue from earlier — the thing
            that teaches the monologuer it&apos;s someone else&apos;s turn.
          </li>
          <li>
            <strong>Stop sequences.</strong> User-provided strings (e.g.{' '}
            <code>&quot;\n\nUser:&quot;</code>, <code>&quot;```&quot;</code>).
            Check after each token whether the running output ends with any
            of them; if so, stop and trim. Handy as a safety net when the
            model hallucinates the next user turn in plain text, ignoring the
            special-token template entirely.
          </li>
        </ul>
      </Prose>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Log-probs vs probs:</strong>{' '}
          softmax can underflow to zero for very negative logits. Do
          arithmetic in log-space (use <code>F.log_softmax</code>) and only{' '}
          <code>exp</code> at the last step. Matters most with long sequences
          where you&apos;re summing log-probs across many tokens (e.g. beam
          search scoring).
        </p>
        <p>
          <strong className="text-term-amber">Forgetting to
          renormalize:</strong> after zeroing out filtered tokens in{' '}
          <em>probability</em> space, the remaining probabilities don&apos;t
          sum to 1 — you have to divide by the new sum. In <em>logit</em>{' '}
          space (mask with <code>-inf</code> before softmax) this is handled
          for free. Do the logit thing.
        </p>
        <p>
          <strong className="text-term-amber">Sampling from the training-loss
          output:</strong> during training, models often use{' '}
          <em>label smoothing</em> or <em>teacher forcing</em> — the logits
          you see in the training forward pass are not the logits you should
          sample from. Run generation with the same forward path as eval
          (<code>model.eval()</code>, no dropout, no label smooth).
        </p>
        <p>
          <strong className="text-term-amber">Stop-sequence matching on text
          vs tokens:</strong> if your stop sequence is{' '}
          <code>&quot;User:&quot;</code> but the tokenizer splits it as{' '}
          <code>[&quot;User&quot;, &quot;:&quot;]</code> vs{' '}
          <code>[&quot;Us&quot;, &quot;er:&quot;]</code> depending on context,
          matching on token IDs will miss cases. Match on the decoded string,
          not the token IDs — slower but correct.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Implement nucleus from scratch and hear the difference">
        <p>
          Start with a frozen GPT-2 small (<code>from transformers import
          GPT2LMHeadModel</code>). Write your own{' '}
          <code>nucleus_sample(logits, p)</code> — no library helpers, just
          sort, cumsum, mask, renormalize, <code>torch.multinomial</code>.
        </p>
        <p className="mt-2">
          Generate 200 tokens three times from the same prompt (<em>&ldquo;Once
          upon a time, in a forest&rdquo;</em>), with <code>p = 0.1</code>,{' '}
          <code>p = 0.5</code>, and <code>p = 0.9</code>. Keep the temperature
          fixed at 1.0.
        </p>
        <p className="mt-2">
          Write one sentence each describing the output&apos;s style. You
          should see something like: <code>p=0.1</code> reads almost like
          greedy, stilted and repetitive; <code>p=0.5</code> is fluent but
          safe; <code>p=0.9</code> is varied, sometimes surprising,
          occasionally off-topic.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: plot the <em>size of the nucleus</em> (how many tokens
          survived) at each generation step for <code>p=0.9</code>.
          You&apos;ll see it spikes after commas and periods (high
          uncertainty) and collapses mid-word (one obvious next piece).
        </p>
      </Challenge>

      {/* ── Closing ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> A base GPT is a monologuer.
          Conversation isn&apos;t a property of the model — it&apos;s a
          property of the <em>transcript format</em> you fine-tune into it,
          plus special tokens that act as stage cues for turn-taking. Once
          the model is in the habit of predicting the closing cue, decoding
          is a separate design decision with its own hyperparameters:
          greedy is deterministic and loops; temperature adds calibrated
          randomness; top-k and top-p truncate the tail; repetition penalty
          kills degenerate loops; beam is for likelihood-maximizing tasks
          like translation, not chat. In real systems you stack{' '}
          <em>temperature → top-p → top-k → repetition penalty</em>, call{' '}
          <code>torch.multinomial</code>, and watch the stop list like a
          hawk.
        </p>
        <p>
          <strong>Next up — Reward Modeling.</strong> SFT got the monologuer
          to take turns. It didn&apos;t get it to be <em>good</em>. The
          fine-tuned model can now converse, but it still has no notion of
          which of its possible replies a human would actually prefer — only
          which one looks most like the transcripts in its training set.
          Reward modeling is the next move: show humans two candidate
          replies, ask them which is better, and train a tiny network to
          predict that preference from the text alone. The output is a scalar
          &ldquo;this reply is good, this one isn&apos;t&rdquo; score — and
          it&apos;s the ingredient every RLHF pipeline runs on. We&apos;ll
          derive the Bradley–Terry loss, train a reward model from pairwise
          data, and set up the scoring head that PPO and DPO both consume.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'The Curious Case of Neural Text Degeneration',
            author: 'Holtzman, Buys, Du, Forbes, Choi',
            venue: 'ICLR 2020 — the nucleus sampling paper',
            year: 2019,
            url: 'https://arxiv.org/abs/1904.09751',
          },
          {
            title: 'Hierarchical Neural Story Generation',
            author: 'Fan, Lewis, Dauphin',
            venue: 'ACL 2018 — origin of top-k sampling',
            year: 2018,
            url: 'https://arxiv.org/abs/1805.04833',
          },
          {
            title: 'Language Models are Unsupervised Multitask Learners',
            author: 'Radford, Wu, Child, Luan, Amodei, Sutskever',
            venue: 'OpenAI 2019 — GPT-2 technical report',
            year: 2019,
            url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
          },
          {
            title: 'CTRL: A Conditional Transformer Language Model for Controllable Generation',
            author: 'Keskar, McCann, Varshney, Xiong, Socher',
            venue: 'Salesforce 2019 — the repetition-penalty formulation',
            year: 2019,
            url: 'https://arxiv.org/abs/1909.05858',
          },
        ]}
      />
    </div>
  )
}
