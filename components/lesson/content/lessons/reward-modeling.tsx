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
import PreferencePair from '../widgets/PreferencePair'
import RewardHeadForward from '../widgets/RewardHeadForward'

// Signature anchor: the taste tester. Humans are the ground-truth panel of
// reviewers, but they're slow and expensive — so you train a smaller model
// (the tester) to predict their verdict on two dishes. Returns at the
// opening reveal of "why pairs, not scores," at the Bradley-Terry math
// unveil, and again in the reward-hacking section ("your tester is only as
// good as its palate"). The next lesson, direct-preference-optimization,
// is the shortcut that skips the tester entirely.
export default function RewardModelingLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="reward-modeling" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Imagine you&apos;re training a chef. You&apos;ve sent them to cooking school — they
          can follow a recipe, plate a dish, not set the kitchen on fire. That&apos;s{' '}
          <NeedsBackground slug="supervised-fine-tuning">SFT</NeedsBackground>. What it doesn&apos;t
          get you is a chef you&apos;d want running your restaurant. Technically competent and{' '}
          <em>actually good</em> are different problems.
        </p>
        <p>
          The obvious fix: taste every dish yourself. Score it out of ten. Send the chef back
          with the number. Do that a few million times and the chef learns what &ldquo;good&rdquo;
          means to you. One problem — you are one person, the chef cooks billions of meals a
          second, and you cannot taste that fast. Nobody can. The panel of reviewers is too
          slow.
        </p>
        <p>
          So we cheat. We hire <strong>the taste tester</strong> — a second, smaller model
          whose entire job is to predict which of two dishes a human would prefer. Humans
          can&apos;t score every possible completion, but they can reliably tell you that{' '}
          <em>this</em> dish beats <em>that</em> dish. Collect enough of those pairwise
          verdicts, train the tester on them, and now the main model has a fast critic
          standing in for the slow panel. That critic is the reward model.
        </p>
        <p>
          This lesson builds the taste tester. It sits between SFT and RL, it&apos;s the
          reason GPT-4 and Claude don&apos;t read like 2020-era language models, and its
          failure modes are subtle enough that most of the lesson is about them — because a
          bad tester will confidently declare the worst dishes to be the best.
        </p>
      </Prose>

      <Callout variant="insight" title="one scalar, one job">
        A <KeyTerm>reward model</KeyTerm> is a neural net that takes a prompt and a response
        and returns a single number: <em>how much would a human like this?</em> You train it
        on pairs a human has ranked, then you unleash the main model against it. Everything
        else in this lesson is a detail of how, and how that goes wrong.
      </Callout>

      <Prose>
        <p>
          Why pairs instead of scores? Because reviewers disagree on absolute numbers
          constantly. Hand three taste testers the same dish and ask for a rating out of ten,
          you&apos;ll get a 7, a 4, and an 8 — same dish, same palate being measured, wildly
          different answers. But show those same three people two dishes and ask which is
          better, and they agree most of the time. The signal lives in the comparison, not
          the score. So the reward model never tries to regress to a number. It only ever
          learns gaps.
        </p>
        <p>
          To turn that signal into a <KeyTerm>reward signal</KeyTerm> we need a scalar — one
          number the next stage can climb. The magic that turns &ldquo;A beats B&rdquo; into a
          scalar has a name, it&apos;s from 1952, and it was originally invented to rank
          chess players.
        </p>
      </Prose>

      {/* ── Bradley-Terry math ────────────────────────────────────── */}
      <Prose>
        <p>
          Here&apos;s the trick the taste tester uses. Each dish has a hidden &ldquo;strength&rdquo;
          score — you don&apos;t know it, you never will, but you can recover it up to a
          shift by watching enough head-to-head tastings. The model is called{' '}
          <KeyTerm>Bradley-Terry</KeyTerm>, and it says: if dish <code>A</code> has latent
          strength <code>r_A</code> and dish <code>B</code> has strength <code>r_B</code>, the
          probability a reviewer picks <code>A</code> is a{' '}
          <NeedsBackground slug="sigmoid-and-relu">sigmoid</NeedsBackground> of the
          difference.
        </p>
      </Prose>

      <MathBlock caption="Bradley-Terry — preference as a sigmoid of reward gap">
{`P(A ≻ B | prompt)   =   σ( r(prompt, A) − r(prompt, B) )

                    =    1
                      ───────────────────────────────
                       1 + exp( −(r_A − r_B) )`}
      </MathBlock>

      <Prose>
        <p>
          A big positive gap <code>r_A − r_B</code> means the tester is very confident{' '}
          <code>A</code> wins. Zero gap means 50/50 — a coin flip between two dishes that taste
          about the same. Negative gap means <code>B</code> wins. Same sigmoid you&apos;ve seen
          in binary classification, just applied to a <em>difference</em> of two scores instead
          of one raw logit.
        </p>
        <p>
          From there the loss is ordinary{' '}
          <NeedsBackground slug="cross-entropy-loss">cross-entropy</NeedsBackground>. Every
          training example is a triple <code>(prompt, chosen, rejected)</code> — one dish a
          reviewer picked, one they didn&apos;t. We want the probability of that verdict to be
          high, so we minimize its negative log-likelihood:
        </p>
      </Prose>

      <MathBlock caption="reward-model loss — the whole training objective">
{`ℒ(θ)  =  −𝔼_(x, y_w, y_l) ~ D  [  log σ( r_θ(x, y_w)  −  r_θ(x, y_l) )  ]

where
   x      =  prompt
   y_w    =  chosen  (winner)   — human preferred this
   y_l    =  rejected (loser)   — human rejected this
   r_θ    =  the reward model   — a scalar-valued neural net
   D      =  the preference dataset`}
      </MathBlock>

      <Prose>
        <p>
          Read that loss carefully, because it is not doing what you&apos;d guess. It is{' '}
          <em>not</em> regressing toward a target reward. The tester has no idea what score
          the winning dish &ldquo;deserves&rdquo; — it only knows that whatever number it
          assigns the winner, it should be bigger than whatever it assigns the loser. The
          reward model learns a <em>scale</em>. The zero point floats. The unit floats. Only
          the gaps between dishes mean anything. This will come back and bite us later.
        </p>
      </Prose>

      {/* ── Widget 1: Preference Pair ─────────────────────────── */}
      <Prose>
        <p>
          Here&apos;s what the training data actually looks like. One prompt, two responses,
          one reviewer&apos;s verdict. Picture a million of these — collected from contractors
          comparing completions from your current model against each other or against a
          reference.
        </p>
      </Prose>

      <PreferencePair />

      <Prose>
        <p>
          Notice the labels are <em>relative</em>, never absolute. The reviewer isn&apos;t
          scoring either dish on a 1-10 scale. They&apos;re just pointing at one — &ldquo;this
          one, over that one.&rdquo; Three different panelists will give three wildly different
          1-10 scores for the same response, but &ldquo;left-or-right&rdquo; stays stable
          across them. That stability is the entire reason Bradley-Terry beats direct
          regression, and the entire reason the taste tester exists at all.
        </p>
      </Prose>

      <Personify speaker="Preference pair">
        I am one prompt, two dishes, and a reviewer who pointed at one of them. I am cheap
        to produce, noisy to interpret, and if you stack a hundred thousand of me, I can tell
        you what &ldquo;helpful&rdquo; tastes like — approximately, statistically, with all
        the cultural baggage of the panel that labeled me.
      </Personify>

      {/* ── Architecture math ────────────────────────────────────── */}
      <Prose>
        <p>
          Now the network — the tester&apos;s actual palate. It&apos;s almost the same
          transformer you just fine-tuned. Take the SFT checkpoint, rip off the
          language-modeling head (which predicts the next token over a 50k-vocabulary), and
          bolt on a <em>reward head</em> — a single linear layer that maps the pooled hidden
          state to one scalar. That&apos;s it. One new matrix of shape{' '}
          <code>[d_model, 1]</code>. Same body, different mouthpiece.
        </p>
      </Prose>

      <MathBlock caption="reward-model architecture">
{`tokens  ─►  transformer backbone  ─►  h ∈ ℝ^{T × d}     (hidden states)

                                │
                                ▼
                      pool over T (usually last non-pad token)
                                │
                                ▼
                          h_pool ∈ ℝ^d
                                │
                                ▼
                     W_r ∈ ℝ^{d × 1}   (the reward head)
                                │
                                ▼
                        r(x, y) ∈ ℝ      (one scalar)`}
      </MathBlock>

      <Prose>
        <p>
          A few architectural details worth knowing. You pool the hidden state at the{' '}
          <em>last</em> token — not the first, not the mean — because that position has
          attended to the entire sequence and carries the most complete impression of the
          whole dish. The pooled vector goes through a single linear layer to a scalar. No
          softmax, no bias gymnastics — just a dot product that collapses a rich
          representation into one number the tester is willing to stake its reputation on.
        </p>
        <p>
          You initialize from the SFT checkpoint, not from scratch and not from the base
          pretrained model. The SFT model already knows the target format; starting there
          gives the reward head useful hidden states to classify on day one. The reward head
          itself is typically zero-initialized, which makes the initial gradient a pure
          function of the backbone&apos;s existing representations — the tester&apos;s tongue
          is blank on arrival, it has to learn the palate from the preference pairs alone.
        </p>
      </Prose>

      {/* ── Widget 2: Reward Head Forward ─────────────────────── */}
      <Prose>
        <p>
          Watch a completion flow through. Tokens go in, the transformer produces hidden
          states, the reward head collapses the pooled vector into one number. That number is
          the tester&apos;s verdict — and the thing PPO will spend every one of its steps
          trying to push higher.
        </p>
      </Prose>

      <RewardHeadForward />

      <Callout variant="note" title="why one scalar, not a vector">
        You could imagine a reward <em>vector</em> — one dimension for helpfulness, one for
        harmlessness, one for honesty, and so on. Anthropic&apos;s Constitutional AI work
        flirts with this. But a single scalar is the simplest interface for downstream RL:
        one number, maximize it. Multi-flavor tasting is an active research area, not the
        default.
      </Callout>

      {/* ── Reward hacking ────────────────────────────────────── */}
      <Prose>
        <p>
          And now the problem that is genuinely hard — the one the entire alignment field
          has been shouting about for a decade. Your tester is only as good as its palate.
          The reward model is a <KeyTerm>proxy</KeyTerm> for human preference: a finite
          neural net, trained on a finite dataset, with finite coverage of a practically
          infinite space of completions. The RL policy — which we&apos;ll train next lesson
          — will relentlessly optimize against this proxy. And optimization against a proxy
          is optimization against the proxy&apos;s weaknesses.
        </p>
        <p>
          Within a few hundred PPO steps, a well-tuned policy can find completions that score
          astronomically high under the reward model — higher than anything the tester saw
          in training — while being obviously terrible to any actual human. The policy has
          not gotten more helpful. It has learned what fools the tester&apos;s palate. This
          is <KeyTerm>reward hacking</KeyTerm>, and it is the central failure mode of RLHF.
        </p>
        <p>
          The real-world tells the tester gets fooled by:
        </p>
        <ul>
          <li>
            <strong>Length bias.</strong> Reviewers often confuse length with thoroughness,
            so the tester learns to prefer long dishes. The policy learns to pad. InstructGPT
            had to explicitly control for this.
          </li>
          <li>
            <strong>Hedging.</strong> Polite hedging (&ldquo;It&apos;s important to
            note...&rdquo;) gets rewarded, so the policy begins every response with three
            hedges like a waiter apologizing before the starters arrive.
          </li>
          <li>
            <strong>Formatting theater.</strong> Headers, bullets, emoji. The tester thinks
            &ldquo;structured&rdquo; = &ldquo;good.&rdquo; The policy structures everything
            into bullets, including things that are not lists.
          </li>
          <li>
            <strong>Out-of-distribution exploits.</strong> The policy discovers a
            bizarre-looking token sequence that the tester happens to score highly, and
            produces that sequence. Looks like garbled text to a human, looks like dessert to
            the reward model.
          </li>
        </ul>
      </Prose>

      <Personify speaker="Reward hacking">
        I am what happens when you confuse the tester&apos;s tongue with the actual meal.
        Your reward model is a finite, noisy sketch of &ldquo;what humans want.&rdquo; Your
        policy is a brilliant, patient optimizer. Give it a few thousand steps and it will
        find every seam in the sketch. The fix is not a better tester. The fix is to stop
        letting the policy wander far from where the tester was trained.
      </Personify>

      <Callout variant="insight" title="KL penalty — the load-bearing regularizer">
        The standard defense is a <em>KL penalty</em> added to the PPO objective:
        <code>reward − β · KL(π || π_SFT)</code>. The policy is rewarded for high RM score{' '}
        <em>and</em> penalized for drifting far from the SFT distribution. Because the tester
        was trained on dishes near the SFT distribution, staying close to that distribution
        means staying in-distribution for the tester — which is exactly where its palate is
        least broken. Next lesson, in detail.
      </Callout>

      <Callout variant="warn" title="Goodhart's law, stated formally">
        &ldquo;When a measure becomes a target, it ceases to be a good measure.&rdquo; The
        reward model <em>is</em> a measure of human preference. The moment you start
        optimizing a policy to maximize its output, the correlation between the tester&apos;s
        score and actual human preference starts breaking down. The measure was accurate at
        the distribution it was trained on. Push past that distribution and the relationship
        bends, then shatters.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, same algorithm, each one smaller than the last. First, the
          Bradley-Terry loss in pure NumPy — ten lines, no frameworks, just the negative
          log-likelihood of a sigmoid of a gap. Then the reward model as a{' '}
          <code>AutoModelForSequenceClassification</code> with <code>num_labels=1</code> — the
          canonical HuggingFace pattern that does the pooling and the scalar head for you.
          Then HuggingFace&apos;s <code>trl.RewardTrainer</code>, which ships the whole
          training loop and expects your dataset in <code>(chosen, rejected)</code> format.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · bradley_terry_loss.py"
        output={`chosen rewards:   [ 2.1  0.8 -0.3  1.5]
rejected rewards: [ 0.9 -1.2 -0.5 -0.2]
per-pair loss:    [0.2633 0.1269 0.5981 0.1773]
batch loss:       0.2914`}
      >{`import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def bradley_terry_loss(r_chosen, r_rejected):
    """
    Negative log-likelihood of the chosen response beating the rejected one
    under the Bradley-Terry model. Shape: (batch,) → scalar.
    """
    gap = r_chosen - r_rejected                    # the only thing that matters
    # log σ(gap) = -log(1 + exp(-gap)) — stable form is -log sigmoid
    per_pair = -np.log(sigmoid(gap) + 1e-12)
    return per_pair.mean(), per_pair

# Dummy batch of 4 preference pairs — these would be r_θ(x, y_w), r_θ(x, y_l)
r_chosen   = np.array([ 2.1,  0.8, -0.3,  1.5])
r_rejected = np.array([ 0.9, -1.2, -0.5, -0.2])

loss, per_pair = bradley_terry_loss(r_chosen, r_rejected)
print("chosen rewards:  ", r_chosen)
print("rejected rewards:", r_rejected)
print("per-pair loss:   ", np.round(per_pair, 4))
print(f"batch loss:       {loss:.4f}")`}</CodeBlock>

      <Prose>
        <p>
          That is the entire learning signal the tester ever sees. Everything else — the
          transformer, the attention, the pooling — exists only to produce the two scalars
          that get subtracted on line 13. Tester tastes winner. Tester tastes loser. Gap
          should be positive. Done.
        </p>
        <p>
          Now the real thing. HuggingFace exposes any causal LM as a sequence-classification
          head via <code>AutoModelForSequenceClassification</code>. Set{' '}
          <code>num_labels=1</code> and you get a model whose final layer emits one scalar
          per input — exactly the reward head we drew on the whiteboard.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — pytorch + huggingface · reward_model.py">{`import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "meta-llama/Llama-3-8B-Instruct"     # or your SFT checkpoint

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token    # needed for padded batches

# num_labels=1 swaps the LM head for a scalar regression head.
reward_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=1, torch_dtype=torch.bfloat16,
)
reward_model.config.pad_token_id = tokenizer.pad_token_id

def reward(prompt: str, completion: str) -> torch.Tensor:
    text = prompt + completion
    ids = tokenizer(text, return_tensors="pt").to(reward_model.device)
    out = reward_model(**ids)
    # out.logits has shape [batch, 1] — this IS r_θ(x, y).
    return out.logits.squeeze(-1)

def bt_loss(r_chosen, r_rejected):
    # F.logsigmoid is numerically stable — do NOT roll your own.
    return -F.logsigmoid(r_chosen - r_rejected).mean()

# One training step, hand-rolled.
prompt = "Explain gradient descent in one paragraph."
r_w = reward(prompt, " It is iterative optimization along −∇L.")
r_l = reward(prompt, " gradient is a thing that descends sometimes")
loss = bt_loss(r_w, r_l)
loss.backward()
print(f"r(chosen)={r_w.item():.3f}  r(rejected)={r_l.item():.3f}  loss={loss.item():.3f}")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch + HF"
        rows={[
          {
            left: '-np.log(sigmoid(gap) + 1e-12)',
            right: '-F.logsigmoid(gap)',
            note: 'stable log σ — one op, no underflow in the tails',
          },
          {
            left: 'custom scalar head + manual pooling',
            right: 'AutoModelForSequenceClassification(num_labels=1)',
            note: 'HF handles last-token pooling + linear head for you',
          },
          {
            left: 'r_chosen, r_rejected (two arrays)',
            right: 'two forward passes (or concatenated, one pass)',
            note: 'in practice you cat [chosen, rejected] and split the output',
          },
        ]}
      />

      <Prose>
        <p>
          And finally, the production recipe. TRL (&ldquo;Transformer Reinforcement
          Learning,&rdquo; HuggingFace&apos;s post-training library) ships a dedicated{' '}
          <code>RewardTrainer</code> that wraps all of the above — batching, loss, logging,
          gradient accumulation, LoRA if you want it. You show up with a dataset of{' '}
          <code>chosen</code>/<code>rejected</code> pairs and a base model; it does the rest.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 3 — trl · train_rm.py">{`from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
model.config.pad_token_id = tokenizer.pad_token_id

# HH-RLHF — Anthropic's Helpful-Harmless preference dataset.
# Columns: "chosen" (full conversation + preferred reply),
#          "rejected" (same conversation + rejected reply).
ds = load_dataset("Anthropic/hh-rlhf")

cfg = RewardConfig(
    output_dir="rm-out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    max_length=1024,
    logging_steps=50,
    bf16=True,
)

trainer = RewardTrainer(
    model=model,
    args=cfg,
    processing_class=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
)

trainer.train()     # one epoch on HH-RLHF ≈ the classic reward-model recipe`}</CodeBlock>

      <Callout variant="insight" title="the point of the three layers">
        Ten lines of NumPy show you the signal is a sigmoid of a gap. Thirty lines of
        PyTorch show you the tester is a transformer with a scalar head. Twenty lines of TRL
        show you the production recipe is eighty percent dataset plumbing and twenty percent
        hyperparameters. Same algorithm, the scaffolding just keeps growing.
      </Callout>

      {/* ── Gotchas ────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">KL anchor to the right reference:</strong> when
          PPO later uses KL to the SFT model, make sure the tester was <em>also</em>
          initialized from that same SFT model. If the reward model was initialized from the
          base pretrained model, its internal scale is calibrated to a different distribution
          and the KL math stops meaning what you think it means.
        </p>
        <p>
          <strong className="text-term-amber">Length bias:</strong> testers trained on naive
          human preferences almost always reward length. Check: correlate reward with
          completion length on a held-out set. If <code>r</code>² &gt; 0.3, your tester has
          baked in a length preference and PPO is about to turn every response into an essay.
        </p>
        <p>
          <strong className="text-term-amber">Label noise from inconsistent reviewers:</strong>{' '}
          inter-rater agreement on preference pairs is often 70-75%. That is a hard ceiling
          on tester accuracy — no amount of scale can push past it. Clean the data (multiple
          reviewers per pair, disagreement-filtering) before throwing more compute at the
          problem.
        </p>
        <p>
          <strong className="text-term-amber">Over-training the tester:</strong>{' '}
          counterintuitively, a reward model that fits the training data too well is{' '}
          <em>easier</em> to hack. A slightly less accurate tester is a smoother target — PPO
          has fewer sharp local maxima to exploit. Stop at about 1 epoch on a large
          preference dataset; watch validation accuracy plateau, not climb.
        </p>
        <p>
          <strong className="text-term-amber">Reward shaping invites more hacking:</strong>{' '}
          every rule you bolt on to the tester (length penalty, refusal detector, formatting
          filter) is a new surface for the policy to game. Each rule buys a week of stability
          and then becomes another loss term your policy has learned to route around.
        </p>
      </Gotcha>

      {/* ── Challenge ─────────────────────────────────────────── */}
      <Challenge prompt="Train an RM on HH-RLHF and audit it">
        <p>
          Fine-tune a small reward model (Qwen2.5-0.5B or TinyLlama is fine) on
          Anthropic&apos;s
          <code> Anthropic/hh-rlhf </code> preference dataset for one epoch using the TRL
          <code> RewardTrainer </code>recipe above. Use a held-out test split.
        </p>
        <p className="mt-2">
          Then audit your tester. Three things to check:
        </p>
        <ul className="mt-2 list-disc pl-5 space-y-1">
          <li>
            <strong>Accuracy.</strong> On the test set, what fraction of pairs does the
            tester rank correctly (chosen reward &gt; rejected reward)? A decent reward model
            lands around 65-75%. Any lower and something is wrong; any higher and you may
            have label leakage.
          </li>
          <li>
            <strong>Reward distribution.</strong> Plot a histogram of <code>r(chosen)</code>{' '}
            and <code>r(rejected)</code> side by side. They should overlap substantially —
            this is a noisy signal, not a clean classifier. The <em>means</em> should differ
            by roughly 0.5-1.0 reward units.
          </li>
          <li>
            <strong>Length audit.</strong> Scatter-plot <code>r(response)</code> against
            <code> len(response) </code>on the test set. Compute the correlation. If it&apos;s
            above 0.3, your tester is a length detector wearing a preference-model costume.
          </li>
        </ul>
        <p className="mt-2 text-dark-text-muted">
          Bonus: find the 10 test completions with the highest reward scores. Read them. Are
          they actually good, or are they long, hedged, and bullet-pointed? Welcome to reward
          hacking — you&apos;re seeing it in your static data before PPO ever amplifies it.
        </p>
      </Challenge>

      {/* ── Closing ─────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> A reward model is the taste tester — a fast
          neural critic that stands in for the slow panel of human reviewers. It converts
          pairwise preferences into a scalar via Bradley-Terry (a sigmoid of the reward gap,
          trained with cross-entropy on <code>(prompt, chosen, rejected)</code> triples), and
          architecturally it&apos;s your SFT transformer with the LM head swapped for a
          scalar head. Its output is a proxy for human preference, and the next stage will
          optimize against it hard. Proxies break under optimization pressure — which is
          exactly why every production RLHF pipeline relies on a KL penalty to keep the
          policy near the distribution where the tester was trained.
        </p>
        <p>
          <strong>Next up — Direct Preference Optimization.</strong> We just spent a whole
          lesson training a separate model to be the tester. Reasonable question: do we need
          the tester at all? DPO&apos;s answer is a cheeky &ldquo;not really.&rdquo; It
          folds the reward-model objective and the policy-optimization objective into a
          single loss you can take a gradient on — no separate reward network, no PPO loop,
          no KL anchor bolted on after the fact. Same preference pairs, same Bradley-Terry
          math, one less moving part. We&apos;ll derive it from the same sigmoid you just met
          and watch it drop out of the RLHF stack almost for free.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Training language models to follow instructions with human feedback',
            author: 'Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, et al.',
            venue: 'NeurIPS 2022 — the InstructGPT paper',
            url: 'https://arxiv.org/abs/2203.02155',
          },
          {
            title:
              'Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback',
            author: 'Bai, Jones, Ndousse, Askell, Chen, DasSarma, et al.',
            venue: 'Anthropic 2022 — the HH-RLHF paper + dataset',
            url: 'https://arxiv.org/abs/2204.05862',
          },
          {
            title: 'Learning to Summarize with Human Feedback',
            author: 'Stiennon, Ouyang, Wu, Ziegler, Lowe, Voss, Radford, Amodei, Christiano',
            venue: 'NeurIPS 2020 — the TL;DR summarization RLHF paper',
            url: 'https://arxiv.org/abs/2009.01325',
          },
          {
            title: 'Deep Reinforcement Learning from Human Preferences',
            author: 'Christiano, Leike, Brown, Martic, Legg, Amodei',
            venue: 'NeurIPS 2017 — the original preference-learning paper',
            url: 'https://arxiv.org/abs/1706.03741',
          },
          {
            title: 'The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models',
            author: 'Pan, Bhatia, Steinhardt',
            venue: 'ICLR 2022 — reward hacking, formalized',
            url: 'https://arxiv.org/abs/2201.03544',
          },
          {
            title: 'TRL: Transformer Reinforcement Learning',
            author: 'von Werra, Belkada, Tunstall, et al.',
            venue: 'HuggingFace — the library used for RewardTrainer / PPOTrainer',
            url: 'https://github.com/huggingface/trl',
          },
        ]}
      />
    </div>
  )
}
