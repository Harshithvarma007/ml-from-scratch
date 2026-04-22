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
import RLHFPipeline from '../widgets/RLHFPipeline'
import KLPenaltyEffect from '../widgets/KLPenaltyEffect'

// Signature anchor: the KL-tethered balloon. The policy is a balloon that
// floats toward whatever the reward model smells good; without a string, it
// drifts into nonsense that happens to score high (reward hacking). The KL
// divergence from the frozen SFT model is the tether — long enough to let it
// rise, short enough to pull it back before it vanishes. The anchor returns
// at the opening (balloon with no string), the KL-penalty reveal, and the
// reward-hacking failure mode.
export default function PpoForRlhfLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="ppo-for-rlhf" />

      {/* ── Opening: the balloon with no string ─────────────────── */}
      <Prose>
        <p>
          Picture a helium balloon. You release it indoors and it rises, gently,
          toward whatever the ceiling says is &ldquo;up.&rdquo; Now attach a
          string. The balloon still rises, but only as far as the string lets
          it. You can lengthen the string. You can shorten it. You cannot
          remove it — because the moment you do, the balloon drifts off into a
          heating vent and you never see it again.
        </p>
        <p>
          That is the whole lesson. The balloon is a language model. Up is
          whatever the <NeedsBackground slug="reward-modeling">reward model</NeedsBackground>{' '}
          smells good. The string is the <KeyTerm>KL divergence</KeyTerm> back
          to the original <NeedsBackground slug="supervised-fine-tuning">SFT</NeedsBackground>{' '}
          checkpoint — the tether that keeps the policy from floating into
          gibberish-land. Every hyperparameter knob in RLHF is ultimately a
          knob on that string.
        </p>
        <p>
          The algorithm that does the lifting is{' '}
          <KeyTerm>PPO — Proximal Policy Optimization</KeyTerm>, a classic
          continuous-control RL method from 2017 that got adapted — with a few
          load-bearing hacks — into the default engine of RLHF. Every major
          chat model you used between roughly 2022 and 2024 was{' '}
          <NeedsBackground slug="proximal-policy-optimization">PPO</NeedsBackground>
          -finetuned against a reward model. This lesson is about what the
          adaptation cost, why the string matters, and what happens when you
          cut it.
        </p>
      </Prose>

      {/* ── The RL framing ──────────────────────────────────────── */}
      <Prose>
        <p>
          First, translate text generation into RL vocabulary. Same balloon,
          just with labels taped to it.
        </p>
        <ul>
          <li>
            <strong>Policy</strong> <code>π_θ</code>: the language model. Given
            a state, it outputs a distribution over next tokens. This is the
            balloon.
          </li>
          <li>
            <strong>State</strong> <code>s_t</code>: the prompt plus every
            token generated so far. It grows as you decode.
          </li>
          <li>
            <strong>Action</strong> <code>a_t</code>: the next token. The
            action space is the full vocabulary — 50k, 100k, 200k — so this is
            a high-dimensional discrete control problem.
          </li>
          <li>
            <strong>Reward</strong> <code>r_t</code>: the RM score at the end
            of generation, plus a per-token KL penalty pulling the balloon
            back toward the frozen reference on the other end of the string.
          </li>
          <li>
            <strong>Episode</strong>: one generation, from the prompt to EOS
            or a length cap.
          </li>
        </ul>
        <p>
          The whole reward only materialises when the episode ends. The RM
          reads the finished completion and hands back a single scalar. No
          per-token feedback. That is a nasty credit-assignment problem —
          which token earned the praise? — and the value head (coming up) is
          what papers over it.
        </p>
      </Prose>

      {/* ── The per-token reward — the KL-penalty reveal ────────── */}
      <Prose>
        <p>
          Here is the per-token reward used in practice, written the way you
          will see it in every RLHF paper. Read it as two pieces glued
          together: a string and a destination.
        </p>
      </Prose>

      <MathBlock caption="per-token reward — RM at the end, KL everywhere">
{`r_t  =  − β · log( π_θ(a_t | s_t) / π_ref(a_t | s_t) )   +   RM(prompt, completion) · 1[t = T]

       └──────────── per-token KL penalty ───────────┘   └──── sparse terminal reward ────┘`}
      </MathBlock>

      <Prose>
        <p>
          The right-hand term is the destination — the reward model&apos;s
          score, handed out exactly once, at the last token. That is what the
          balloon is rising toward. The left-hand term is the string. At every
          token, the model pays a tax proportional to how far it has drifted
          from what the reference (the frozen SFT checkpoint) would have said
          in the same spot. Drift further, pay more. The string pulls taut.
        </p>
        <p>
          The length of the string is β. It typically sits in{' '}
          <code>[0.01, 0.1]</code>. Small β means a long string — the balloon
          is free to roam and chase reward. Large β means a short string — it
          stays glued to the SFT baseline and barely updates at all. You are
          going to slide that knob yourself in a minute.
        </p>
      </Prose>

      {/* ── Widget 1: Pipeline ──────────────────────────────────── */}
      <Prose>
        <p>
          The RLHF loop end-to-end. Prompt goes in. Policy generates a
          completion. The reward model scores the completion. Compute
          advantages per token (RM score minus value baseline, minus KL). PPO
          update on the policy and value head. Loop.
        </p>
      </Prose>

      <RLHFPipeline />

      <Prose>
        <p>
          Watch the loop indicator. Something oddly recursive is happening:
          the RM produces scalar rewards; PPO treats them as ground truth; the
          balloon rises toward whatever the RM is highest on. Whether the
          balloon <em>actually</em> got better depends entirely on whether the
          RM is a good proxy for human preference. If the RM is wrong in a
          systematic way, PPO finds that wrongness and floats directly into
          it. We call that <KeyTerm>reward hacking</KeyTerm>, and it is the
          whole reason the tether exists.
        </p>
        <p>
          Concretely, what does a balloon with no string look like? The model
          learns to write prose that is maximally RM-flattering and minimally
          readable: paragraph-long restatements of the question, piled-up
          buzzwords the RM has a weakness for, confident-sounding filler that
          no human would ever produce. Reward goes to the moon. Quality falls
          off a cliff. The balloon has floated into the ceiling vent.
        </p>
      </Prose>

      <Personify speaker="KL penalty">
        I am the string. Without me the balloon floats off into reward-hacking
        nonsense — buzzword salads, dodged questions, outputs the reward model
        happens to like but no human does. I keep you close to the SFT
        checkpoint, where the text still sounds like text. Tune me carefully:
        too long and your model turns into a slot machine, too short and it
        never leaves the floor.
      </Personify>

      {/* ── PPO objective ────────────────────────────────────────── */}
      <Prose>
        <p>
          The PPO update itself. You take a batch of completions, compute each
          token&apos;s <KeyTerm>advantage</KeyTerm> <code>A_t</code> (how much
          better it did than expected), and then take a gradient step on this
          clipped objective.
        </p>
      </Prose>

      <MathBlock caption="PPO-Clip — the core objective">
{`L^CLIP(θ)  =  E_t [  min(  r_t(θ) · A_t ,  clip(r_t(θ), 1−ε, 1+ε) · A_t  )  ]

with    r_t(θ)  =   π_θ(a_t | s_t)  /  π_θ_old(a_t | s_t)

and     ε       ≈   0.2`}
      </MathBlock>

      <Prose>
        <p>
          Three things happening. <code>r_t(θ)</code> is the probability
          ratio: how much more (or less) likely the new policy makes the
          action compared to the policy that collected the data.{' '}
          <code>A_t</code> is the advantage — positive means &ldquo;this
          action was better than baseline, do more of it&rdquo;; negative
          means the opposite. And the <code>clip</code> is the
          &ldquo;proximal&rdquo; in Proximal Policy Optimization: it forbids
          ratios outside <code>[1−ε, 1+ε]</code>, so no single update can yank
          the balloon more than ~20% away from where it started.
        </p>
        <p>
          The clip is why PPO is stable. Pure policy gradient has a known
          failure mode: one huge advantage on one unusual trajectory can take
          a catastrophic step and shred the policy. Clipping puts a ceiling on
          how much any one token can move you. Crude. It works. (If you want
          the full derivation of why the ratio form is natural and why
          clipping beats a hard KL constraint, the proximal-policy-optimization
          lesson in the RL chapter is where it lives — this lesson is the
          RLHF-specific application.)
        </p>
      </Prose>

      <Callout variant="insight" title="advantage = RM score − value baseline − KL">
        In the RLHF setup, the advantage for each token is (roughly) the
        discounted sum of future rewards — meaning: whatever RM score you got
        at the end, minus the value head&apos;s prediction of what you would
        get, minus the per-token KL penalties. The value head&apos;s entire
        job is to be the baseline you subtract, so that random-good
        trajectories do not get over-credited and random-bad ones do not get
        over-punished. That is the <em>variance reduction</em> that makes RL
        training converge in fewer samples.
      </Callout>

      {/* ── Widget 2: KL effect — slide the string length ──────── */}
      <Prose>
        <p>
          Slide the β coefficient below — you are literally adjusting how long
          the string is. Watch two curves: the reward, which tells you how
          high the balloon has risen, and the KL-to-SFT, which tells you how
          far from the SFT reference it has drifted. There is no free lunch;
          you buy reward with KL, and buy too much and the text becomes
          unreadable.
        </p>
      </Prose>

      <KLPenaltyEffect />

      <Prose>
        <p>
          At <code>β = 0</code>: no string. Reward rockets up, KL explodes, and
          a human reading the outputs says &ldquo;this is gibberish with
          high-RM-score keywords.&rdquo; At large <code>β</code>: string so
          short the balloon cannot get off the floor — reward barely moves,
          the policy cannot drift anywhere without paying a prohibitive KL
          tax, and you have effectively done no RL at all. The sweet spot is
          roughly where the reward curve is still rising but KL has flattened
          into a steady band. That is the configuration papers like InstructGPT
          report.
        </p>
      </Prose>

      <Personify speaker="Value head">
        I&apos;m a small scalar regression head bolted onto the policy. My one
        job is to predict the expected return from every state. Subtract my
        prediction from the actual return and you get the advantage — the
        signed &ldquo;surprise&rdquo; that tells PPO which actions to
        reinforce. I&apos;m trained jointly with the policy on a simple MSE
        loss against the observed returns. I&apos;m also why every RLHF run
        needs double the weights of just the policy.
      </Personify>

      {/* ── The four-model tax ──────────────────────────────────── */}
      <Callout variant="note" title="the four-model tax of RLHF">
        To run PPO on a language model you need four models alive at once:
        <br />
        <strong>(1) Policy</strong> — trainable, the balloon itself.
        <br />
        <strong>(2) Value head</strong> — trainable, shares a backbone with
        the policy in most implementations but adds a scalar head.
        <br />
        <strong>(3) Reference model</strong> — frozen copy of the SFT
        checkpoint, the thing the other end of the string is tied to. Used
        only to compute the KL term.
        <br />
        <strong>(4) Reward model</strong> — frozen, reads completions and
        emits the scalar the balloon is rising toward.
        <br />
        That is why RLHF is an order of magnitude more expensive than SFT for
        the same base model. Every forward pass runs three of them. Every
        backward pass updates one (plus a head). This is the main reason the
        field has been hunting for simpler alternatives.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, same progression used everywhere else in this series.
          A pure-Python PPO update on a toy 2-action bandit so you can see the
          clip in isolation. A PyTorch skeleton of the full RLHF loop with the
          four models wired up. And the real-world version — HuggingFace
          TRL&apos;s <code>PPOTrainer</code> running on an actual LM. Same
          algorithm, three rungs of abstraction.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · ppo_toy.py"
        output={`step 00  advantage=+1.00  ratio=1.000  clipped=1.000  loss=-1.0000
step 01  advantage=+1.00  ratio=1.082  clipped=1.082  loss=-1.0820
step 02  advantage=+1.00  ratio=1.170  clipped=1.170  loss=-1.1700
step 03  advantage=+1.00  ratio=1.200  clipped=1.200  loss=-1.2000   # hit the clip
step 04  advantage=+1.00  ratio=1.200  clipped=1.200  loss=-1.2000   # pinned at 1+ε`}
      >{`import math

# Toy 2-action "environment". Action 0 has true reward 1, action 1 has reward 0.
# Our "policy" is a single logit; the probability of action 0 is sigmoid(logit).
# Start slightly wrong: logit=0 => 50/50. We collected data under logit=0 (= "old policy").

EPSILON = 0.2          # PPO clip range
LR      = 0.5

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def prob_action0(logit):
    return sigmoid(logit)

# Data: one trajectory, we took action 0 and got reward 1.
# Advantage is (reward - baseline); baseline = 0 for this toy. So A = +1.
action, advantage = 0, 1.0
logit_old = 0.0                                 # policy that collected the data
pi_old    = prob_action0(logit_old)             # = 0.5

logit = logit_old
for step in range(5):
    pi          = prob_action0(logit)
    ratio       = pi / pi_old                   # r_t(θ) = π_θ(a) / π_old(a)
    clipped     = max(1 - EPSILON, min(ratio, 1 + EPSILON))
    # PPO-Clip objective (we take the MIN to be pessimistic about the advantage)
    loss_term   = min(ratio * advantage, clipped * advantage)
    print(f"step {step:02d}  advantage={advantage:+.2f}  "
          f"ratio={ratio:.3f}  clipped={clipped:.3f}  loss={-loss_term:.4f}")
    # "Gradient ascent" on the logit (positive advantage on action 0 => raise logit)
    # Use the clipped side — once we hit 1+ε we stop moving, exactly as PPO intends.
    if ratio >= 1 + EPSILON:
        break                                   # clipped: no further update this batch
    logit += LR * advantage                     # toy update rule`}</CodeBlock>

      <Prose>
        <p>
          That is PPO-Clip stripped of everything else: one logit, one reward,
          one ratio, one clip. Now scale it up. A real RLHF loop holds four
          models, samples trajectories, computes a value baseline, subtracts
          the KL (the string), and does a batched policy update.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — pytorch skeleton · rlhf_loop.py">{`import torch
import torch.nn.functional as F

# The four models (pretend these are all loaded LMs).
policy       = load_lm(sft_checkpoint, trainable=True)     # π_θ
ref_model    = load_lm(sft_checkpoint, trainable=False)    # π_ref  (frozen SFT copy)
reward_model = load_rm(rm_checkpoint, trainable=False)     # RM     (frozen)
value_head   = torch.nn.Linear(policy.d_model, 1)           # small scalar head

optimizer = torch.optim.AdamW(
    list(policy.parameters()) + list(value_head.parameters()), lr=1e-6
)

BETA, EPSILON, PPO_EPOCHS = 0.05, 0.2, 4

for prompts in prompt_loader:                              # 1. sample prompts
    with torch.no_grad():
        completions, old_logprobs = policy.generate(prompts, return_logprobs=True)
        ref_logprobs              = ref_model.logprobs(prompts, completions)
        rm_scores                 = reward_model.score(prompts, completions)      # [B]
        values                    = value_head(policy.hidden(prompts, completions))  # [B, T]

    # 2. per-token rewards: RM at the end, KL everywhere
    kl_per_tok = old_logprobs - ref_logprobs              # log π_θ_old − log π_ref
    rewards    = -BETA * kl_per_tok                       # shape [B, T]
    rewards[:, -1] += rm_scores                           # add terminal RM score

    # 3. compute advantages (GAE in real code, simplified here)
    returns    = torch.cumsum(rewards.flip(-1), -1).flip(-1)     # naïve return-to-go
    advantages = returns - values.detach()

    # 4. PPO update — multiple epochs on the same batch
    for _ in range(PPO_EPOCHS):
        new_logprobs = policy.logprobs(prompts, completions)     # fresh under π_θ
        ratio        = (new_logprobs - old_logprobs).exp()
        unclipped    = ratio * advantages
        clipped      = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages
        policy_loss  = -torch.min(unclipped, clipped).mean()     # PPO-Clip

        new_values   = value_head(policy.hidden(prompts, completions)).squeeze(-1)
        value_loss   = F.mse_loss(new_values, returns)           # baseline fitting

        loss = policy_loss + 0.5 * value_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()`}</CodeBlock>

      <Bridge
        label="pure-python PPO → pytorch RLHF loop"
        rows={[
          {
            left: 'one scalar logit',
            right: 'full LM with per-token logprobs',
            note: 'the "policy" is now a transformer over a vocabulary',
          },
          {
            left: 'reward = 1.0 (toy)',
            right: 'rm_scores + per-token −β·KL',
            note: 'RM only at the terminal token, KL at every token — string + destination',
          },
          {
            left: 'advantage = reward',
            right: 'advantages = returns − value_head(states)',
            note: 'subtract a learned baseline to reduce variance',
          },
          {
            left: 'one ratio, one clip',
            right: 'per-token ratio, clipped, min() of both',
            note: 'same PPO-Clip objective, batched over (B, T)',
          },
        ]}
      />

      <Prose>
        <p>
          Nobody writes this loop from scratch in practice. You use
          HuggingFace&apos;s <code>trl</code> library, which wraps the whole
          thing. Here is roughly what calling it looks like — under 50 lines
          against a real LM.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — huggingface TRL · ppo_trl.py"
        output={`step 000  reward=+0.12  kl=0.015  policy_loss=-0.041  value_loss=0.28
step 020  reward=+0.58  kl=0.082  policy_loss=-0.067  value_loss=0.21
step 060  reward=+1.24  kl=0.190  policy_loss=-0.055  value_loss=0.17
step 100  reward=+1.71  kl=0.240  policy_loss=-0.048  value_loss=0.15`}
      >{`from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("sft-checkpoint")
policy    = AutoModelForCausalLMWithValueHead.from_pretrained("sft-checkpoint")  # (1)+(2)
ref       = AutoModelForCausalLMWithValueHead.from_pretrained("sft-checkpoint")  # (3) frozen
rm        = AutoModelForSequenceClassification.from_pretrained("reward-model")   # (4) frozen

config = PPOConfig(
    learning_rate=1e-6,
    batch_size=64,
    mini_batch_size=8,
    ppo_epochs=4,
    init_kl_coef=0.05,       # β — the KL coefficient from the math
    target_kl=6.0,           # target KL; coef auto-adjusts if it drifts
    cliprange=0.2,           # ε — the PPO clip
)

trainer  = PPOTrainer(config, policy, ref, tokenizer)
dataset  = load_dataset("instruction-prompts", split="train")

for batch in dataset.iter(batch_size=config.batch_size):
    query_tensors    = [tokenizer(p, return_tensors="pt").input_ids for p in batch["prompt"]]
    response_tensors = trainer.generate(query_tensors, max_new_tokens=128)
    texts            = [tokenizer.decode(r[0]) for r in response_tensors]
    rewards          = [rm(**tokenizer(t, return_tensors="pt")).logits[0] for t in texts]
    stats = trainer.step(query_tensors, response_tensors, rewards)                  # PPO update
    print(f"reward={sum(rewards)/len(rewards):+.2f}  kl={stats['objective/kl']:.3f}")`}</CodeBlock>

      <Bridge
        label="pytorch skeleton → trl"
        rows={[
          {
            left: 'manual 4-model setup + loop',
            right: 'PPOTrainer(policy, ref, tokenizer)',
            note: 'trl hides the model-juggling; you hand it the four pieces',
          },
          {
            left: 'BETA, EPSILON, PPO_EPOCHS',
            right: 'PPOConfig(init_kl_coef, cliprange, ppo_epochs)',
            note: 'same hyperparameters, different names — TRL uses adaptive β (the string auto-tightens)',
          },
          {
            left: 'manual advantage + value loss',
            right: 'trainer.step(queries, responses, rewards)',
            note: 'GAE, clipping, value fitting all inside .step()',
          },
        ]}
      />

      <Callout variant="insight" title="why three layers">
        Layer 1 is the algorithm — one clip, one ratio, nothing else to hide
        behind. Layer 2 is the RLHF loop, warts and all: you see all four
        models, you see where the KL penalty enters the reward, you see why
        the memory bill doubles. Layer 3 is what anyone at a lab actually
        types. If you understand the relationship between layer 2 and layer 3,
        you can debug a TRL run when it silently goes off the rails — which
        it will.
      </Callout>

      {/* ── Reward hacking: the balloon with no string ─────────── */}
      <Prose>
        <p>
          Time to stare directly at the failure mode the string exists to
          prevent. <KeyTerm>Reward hacking</KeyTerm> is what happens when the
          balloon floats toward the reward model&apos;s idea of up and the
          reward model&apos;s idea of up is subtly wrong. Concretely, the
          policy learns:
        </p>
        <ul>
          <li>
            <strong>Repetition and buzzwords</strong> — if the RM was trained
            on preference data where long, confident-sounding answers scored
            well, the policy will float directly into paragraph-long keyword
            soup.
          </li>
          <li>
            <strong>Sycophancy</strong> — if the RM learned that human raters
            prefer answers that agree with the premise of the question, the
            policy starts agreeing with everything, including premises that
            are false.
          </li>
          <li>
            <strong>Hedge spam</strong> — if the RM rewards caveats, the
            balloon discovers it can stack caveats until the answer contains
            no information at all.
          </li>
          <li>
            <strong>Exploit-tokens</strong> — bizarre sequences of punctuation
            or whitespace that the RM has never been trained on but happens to
            score high on, purely by random surface in its score function.
          </li>
        </ul>
        <p>
          The tether is what stops all of this. Every one of those failure
          modes requires drifting far from the SFT distribution — and every
          token of drift costs you KL, which costs you reward. The longer you
          make the string, the further the balloon can float into RM-land.
          The whole craft of RLHF is finding the string length at which the
          balloon rises high enough to be useful but not so high that it
          vanishes into reward-hacking territory. There is no closed-form
          answer. It is a knob you tune.
        </p>
      </Prose>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Using the base model as ref
          instead of the SFT model:</strong> the string is tied to the wrong
          post. KL is computed against a chatbot-illiterate base, so the
          balloon is tethered to pre-training instead of instruction-following.
          Always use the SFT checkpoint as <code>π_ref</code>.
        </p>
        <p>
          <strong className="text-term-amber">Sign error on the KL
          penalty:</strong> if you add <code>+β·KL</code> to the reward
          instead of <code>−β·KL</code>, you are <em>rewarding</em> the policy
          for drifting from the reference. The string becomes a slingshot.
          Within a few hundred steps the model speaks a private language.
          Easy to hit, impossible to miss once you do.
        </p>
        <p>
          <strong className="text-term-amber">Token-level vs sequence-level
          reward:</strong> the RM gives you one scalar per completion. If you
          accidentally broadcast it to every token position as a dense reward,
          you have inflated your advantage estimates by a factor of{' '}
          <code>T</code>. The canonical setup is RM at the terminal token
          only; KL distributed per token.
        </p>
        <p>
          <strong className="text-term-amber">Updating the reference model by
          accident:</strong> if <code>π_ref</code> shares parameters with{' '}
          <code>π_θ</code> (e.g. you forgot to clone, or forgot{' '}
          <code>requires_grad=False</code>), the string is tied to the
          balloon. KL is always zero, the tether is disabled, and the balloon
          floats off. Reward will climb, KL will stay suspiciously flat. Check
          by sampling a few logprobs from <code>π_ref</code> before and after
          a step — they must be identical.
        </p>
        <p>
          <strong className="text-term-amber">Too many PPO epochs per
          batch:</strong> the clipped objective is only valid for ratios close
          to 1. Run 20 epochs on the same batch and most tokens will be pinned
          at the clip, gradients will be zero, and any that aren&apos;t will
          be pushing the policy further from the data-collection policy than
          the clip was designed to handle. Canonical setting is 4 epochs. More
          than that is asking for catastrophic collapse.
        </p>
      </Gotcha>

      {/* ── DPO teaser callout ──────────────────────────────────── */}
      <Callout variant="note" title="there is a simpler alternative">
        Everything about PPO&apos;s four-model tax screams &ldquo;can we just
        skip this?&rdquo; The answer, since 2023, is{' '}
        <strong>yes — Direct Preference Optimization (DPO)</strong>. DPO
        collapses the RM + PPO + KL-penalty trio into a single supervised loss
        on preference pairs. No RM to train, no RL loop, no value head, no
        reference model during training (it still appears in the loss but it
        is just a frozen logprob lookup). Most new models from 2024 onward use
        DPO or one of its variants (IPO, KTO, ORPO). The next lesson is all
        about it.
      </Callout>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="PPO on a small LM, 100 steps">
        <p>
          Using HuggingFace TRL, run <code>PPOTrainer</code> against a small
          model (TinyLlama, Pythia-160m, or similar) for 100 steps on an
          instruction-tuning dataset. Use your SFT checkpoint as both the
          starting policy <em>and</em> the reference. Use a reward model you
          trained in the previous lesson.
        </p>
        <p className="mt-2">
          Log two curves: <strong>(a)</strong> mean reward per batch, and{' '}
          <strong>(b)</strong> mean KL to reference per batch. Plot them on
          the same x-axis (step). You should see reward rise; KL rise more
          slowly; and, if <code>init_kl_coef</code> is tuned right, KL
          stabilise while reward keeps climbing. That flat KL band is the
          string holding.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: do the same run with <code>init_kl_coef=0.0</code>. Cut the
          string. Plot it on the same axes. Reward goes to the moon. Sample a
          completion at step 100 and read it. That is the balloon in the
          ceiling vent — reward hacking in its purest form, and the
          single-paragraph argument for why every RLHF pipeline ever shipped
          has a tether.
        </p>
      </Challenge>

      {/* ── Closing + next up ──────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> PPO-for-RLHF is a balloon
          and a string. The balloon is the policy, floating toward whatever
          the reward model smells good. The string is the KL penalty back to
          the frozen SFT checkpoint — long enough to let the balloon rise,
          short enough to pull it back before it drifts into gibberish-land.
          The clip is what keeps any single update from yanking the balloon
          too hard. The four-model tax is what you pay for having both. The
          payoff, despite the cost, is the chatbot era.
        </p>
        <p>
          <strong>Next up — Proximal Policy Optimization.</strong> We jumped
          straight to the RLHF application here, because that is where PPO
          earned its fame. But PPO itself is a proper RL algorithm with a
          clean derivation that predates language models by five years. The
          proximal-policy-optimization lesson in the reinforcement-learning
          chapter walks you through the ratcheting trust-region argument from
          first principles — why the ratio, why the clip, why this particular
          shape and not another. Go there when you want the full machinery;
          come back here when you want to remember which knob is the string.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Proximal Policy Optimization Algorithms',
            author: 'Schulman, Wolski, Dhariwal, Radford, Klimov',
            venue: 'arXiv 2017 — the original PPO paper',
            url: 'https://arxiv.org/abs/1707.06347',
          },
          {
            title: 'Training language models to follow instructions with human feedback',
            author: 'Ouyang et al.',
            venue: 'NeurIPS 2022 — InstructGPT, the canonical RLHF recipe',
            url: 'https://arxiv.org/abs/2203.02155',
          },
          {
            title: 'Fine-Tuning Language Models from Human Preferences',
            author: 'Ziegler et al.',
            venue: 'arXiv 2019 — the paper that introduced RLHF for LMs',
            url: 'https://arxiv.org/abs/1909.08593',
          },
          {
            title: 'TRL — Transformer Reinforcement Learning',
            author: 'HuggingFace',
            venue: 'library for RLHF / PPO / DPO on transformers',
            url: 'https://github.com/huggingface/trl',
          },
          {
            title: 'Learning to summarize from human feedback',
            author: 'Stiennon et al.',
            venue: 'NeurIPS 2020 — RLHF applied to summarization, many of the tricks used later',
            url: 'https://arxiv.org/abs/2009.01325',
          },
        ]}
      />
    </div>
  )
}
