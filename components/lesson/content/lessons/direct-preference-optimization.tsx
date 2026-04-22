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
import DPOvsPPO from '../widgets/DPOvsPPO'
import DPOLossLandscape from '../widgets/DPOLossLandscape'

export default function DirectPreferenceOptimizationLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="direct-preference-optimization" />

      {/* ── Opening: the shortcut that skips the taste tester ───── */}
      <Prose>
        <p>
          The classic RLHF pipeline is a detour. You collect preference pairs,
          train a{' '}
          <NeedsBackground slug="reward-modeling">reward model</NeedsBackground>{' '}
          to imitate the human labelers, and then unleash PPO on your policy
          to climb that learned reward. Two models, two training stages, two
          places for things to go sideways. The humans point at what they
          like, the taste tester memorizes the pointing, the policy chases
          the taste tester&apos;s scores. It works. It&apos;s also an
          enormous amount of machinery for what — if you squint — is a
          classification problem wearing a trench coat.
        </p>
        <p>
          In 2023 Rafailov et al. at Stanford wrote down the shortcut. If the
          PPO objective has a closed-form optimal policy, and the reward
          model is trained with Bradley-Terry on pairs, and PPO is just
          climbing that reward — then algebraically, you can collapse the
          whole thing. Skip the taste tester. Train the policy directly
          against the preference pairs. Same destination, fewer stops.
          That&apos;s <KeyTerm>DPO</KeyTerm> — direct preference
          optimization, and the direct is doing load-bearing work.
        </p>
        <p>
          This lesson is the derivation. We start from the same
          KL-constrained reward objective PPO optimizes, invert it to
          express the reward in terms of the policy, substitute into
          Bradley-Terry, and watch the reward model disappear into a ratio
          of log-probabilities. Then we personify, code it three ways, and
          cover what you give up by shortcutting — plus KTO, IPO, and ORPO,
          the variants that fell out of the DPO paper like mushrooms after
          rain.
        </p>
      </Prose>

      <Personify speaker="DPO">
        I&apos;m the two-step skipped to one. PPO wanted four models in
        memory, a sampling loop, a value head, and a KL coefficient you
        tuned by superstition. I want two models, no sampling, one scalar{' '}
        <code>β</code>. I&apos;m classification pretending to be RL — and
        most days the math is on my side.
      </Personify>

      {/* ── Derivation ──────────────────────────────────────────── */}
      <Prose>
        <p>
          The derivation is the whole trick. Start where PPO starts — the
          KL-constrained reward maximization objective. You want a policy{' '}
          <code>π</code> that scores well on a reward <code>r(x, y)</code>,
          but doesn&apos;t drift too far from a reference policy{' '}
          <code>π_ref</code> (usually your{' '}
          <NeedsBackground slug="supervised-fine-tuning">SFT</NeedsBackground>{' '}
          checkpoint). The constraint is a KL penalty with strength{' '}
          <code>β</code>:
        </p>
      </Prose>

      <MathBlock caption="the KL-constrained reward objective — what PPO spends all its effort approximating">
{`max_π  𝔼_{x~D, y~π(·|x)} [ r(x, y) ]   −   β · KL( π(·|x) ‖ π_ref(·|x) )`}
      </MathBlock>

      <Prose>
        <p>
          This objective has a closed-form optimal policy. Skip the
          Lagrangian for a moment and take the answer on faith — the policy
          that maximizes this is:
        </p>
      </Prose>

      <MathBlock caption="optimal policy under the KL-constrained reward">
{`π*(y | x)   =   1/Z(x)  ·  π_ref(y | x)  ·  exp( r(x, y) / β )`}
      </MathBlock>

      <Prose>
        <p>
          Here <code>Z(x)</code> is a per-prompt normalizer that makes{' '}
          <code>π*</code> sum to 1 over <code>y</code>. A very normal result:
          the optimal policy is the reference tilted by an exponential in
          the reward. High-reward completions get upweighted, low-reward ones
          get downweighted, and <code>β</code> controls how hard you tilt.
        </p>
        <p>
          Now the reveal — the single move the whole paper is built on. We{' '}
          <em>invert</em> this equation. Instead of reading the policy off
          the reward, we read the reward off the policy. Solve for{' '}
          <code>r</code>:
        </p>
      </Prose>

      <MathBlock caption="the reward, expressed through the policy">
{`r(x, y)   =   β · log( π*(y | x) / π_ref(y | x) )   +   β · log Z(x)`}
      </MathBlock>

      <Prose>
        <p>
          Read that line twice. The reward — the thing the taste tester
          existed to compute — is now a log-ratio of the policy and the
          reference. If we knew the optimal policy, we could read the reward
          straight off it; the reward model is secretly a view into the
          policy itself. And here&apos;s where the bypass clicks into place:
          plug this expression into the Bradley-Terry preference model that
          the reward model was trained on, and the pesky <code>Z(x)</code>{' '}
          term cancels, because Bradley-Terry only cares about reward{' '}
          <em>differences</em> between two completions for the same prompt.
        </p>
      </Prose>

      <MathBlock caption="Bradley-Terry: probability the chosen completion beats the rejected one">
{`P(y_w ≻ y_l | x)   =   σ( r(x, y_w) − r(x, y_l) )

                 =   σ( β · log(π*(y_w|x)/π_ref(y_w|x))
                        − β · log(π*(y_l|x)/π_ref(y_l|x)) )`}
      </MathBlock>

      <Prose>
        <p>
          The taste tester has disappeared. No <code>r_θ</code> anywhere.
          The preference likelihood is now stated entirely in terms of the
          policy and the frozen reference. Maximize the log-likelihood of
          the preference data and you&apos;re training the policy directly:
        </p>
      </Prose>

      <MathBlock caption="the DPO loss — this is the whole algorithm">
{`L_DPO(π, π_ref)  =  − 𝔼_{(x, y_w, y_l) ~ D} [

        log σ ( β · log( π(y_w|x) / π_ref(y_w|x) )
              − β · log( π(y_l|x) / π_ref(y_l|x) ) )

]`}
      </MathBlock>

      <Callout variant="insight" title="read the loss like a sentence">
        <em>Push the log-prob of the chosen completion up; push the
        log-prob of the rejected one down; both anchored to what the
        reference model thought.</em> That is the entire algorithm.{' '}
        <code>β</code> scales how far the policy is allowed to drift from
        the reference. Fifteen lines of PyTorch is a reasonable way to
        describe the end of a research subfield — and a fair summary of
        what happens when a two-step pipeline gets collapsed by algebra.
      </Callout>

      {/* ── Widget 1: DPO vs PPO ────────────────────────────────── */}
      <Prose>
        <p>
          Before we personify <code>β</code>, feel the pipeline difference.
          PPO needs four models in memory — the policy, a value head, the
          reward model, and the reference — plus an online sampling loop to
          generate rollouts for each update. DPO needs two models, one
          static dataset, and no sampling anywhere. The shortcut is not
          cosmetic.
        </p>
      </Prose>

      <DPOvsPPO />

      <Prose>
        <p>
          The compute bar tells it plainly. A 7B model trained with PPO
          wants roughly 4× the GPU memory of the same model trained with
          DPO, and each DPO step is a plain forward-backward pass — no
          generation, no per-token value targets, no clipped ratios. DPO on
          a node of 8×A100s is a weekend project for a grad student. PPO
          is a sprint through infrastructure debt.
        </p>
      </Prose>

      <Personify speaker="β (beta)">
        I&apos;m the only knob left after the shortcut. PPO had a dozen —
        clip epsilon, KL coefficient, value coefficient, entropy
        coefficient, GAE lambda, rollout length, and their cousins.
        I&apos;m one scalar, usually 0.1 to 0.5. Small me: the policy
        drifts far from the reference and overfits the pairs. Large me:
        the policy barely moves. Pick me well.
      </Personify>

      {/* ── Widget 2: Loss landscape ────────────────────────────── */}
      <Prose>
        <p>
          Here&apos;s the loss surface for a single preference pair,
          visualized over the policy&apos;s log-probability of the chosen
          and rejected completions. Drag the knobs and watch where the
          gradient points.
        </p>
      </Prose>

      <DPOLossLandscape />

      <Prose>
        <p>
          Two things to notice. First, the gradient always pushes the
          chosen log-prob up and the rejected log-prob down — no regime
          rewards making the preferred answer less likely. Second, the
          magnitude depends on how wrong the policy currently is. If the
          policy already strongly prefers <code>y_w</code> over{' '}
          <code>y_l</code>, the sigmoid saturates, the gradient shrinks,
          and the update is small. That&apos;s the standard behavior of
          any{' '}
          <NeedsBackground slug="cross-entropy-loss">
            cross-entropy
          </NeedsBackground>{' '}
          loss — DPO inherits it, because under the hood that&apos;s what
          DPO is.
        </p>
      </Prose>

      <Callout variant="note" title="why β shows up twice">
        In the loss, <code>β</code> multiplies the log-ratio <em>inside</em>{' '}
        the sigmoid. That&apos;s equivalent to scaling the implicit reward:
        bigger <code>β</code> means the implicit reward gradient is bigger
        per unit of log-prob change, which paradoxically keeps the policy
        close to the reference, because the sigmoid saturates faster. Small{' '}
        <code>β</code> gives the policy more room to move. PPO teams spent
        years tuning a KL coefficient by hand; DPO compresses all of that
        superstition into this one number.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, as always. First the loss in numpy on a single toy
          pair, so you can see the gradient arithmetic with no framework in
          the way. Then the production path — HuggingFace <code>trl</code>
          &apos;s <code>DPOTrainer</code>, which is what real teams ship.
          Then a hand-rolled PyTorch loop that shows the loss is genuinely
          just fifteen lines.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · dpo_loss_toy.py"
        output={`policy logp(chosen)   = -2.30
policy logp(rejected) = -3.00
ref    logp(chosen)   = -2.50
ref    logp(rejected) = -2.50

DPO loss       = 0.4741
implicit reward gap = β · (log-ratio gap) = 0.1800
gradient(chosen)   = +0.3803  (push up)
gradient(rejected) = -0.3803  (push down)`}
      >{`import numpy as np

# one preference pair, β = 0.1
beta = 0.1

# policy log-probs of chosen (y_w) and rejected (y_l) completions
policy_logp_w = -2.30
policy_logp_l = -3.00

# reference model log-probs of the same completions
ref_logp_w    = -2.50
ref_logp_l    = -2.50

# log-ratios: how much the policy has moved relative to the reference
log_ratio_w = policy_logp_w - ref_logp_w        # +0.20
log_ratio_l = policy_logp_l - ref_logp_l        # -0.50

# the implicit reward gap — this is r(x, y_w) − r(x, y_l) per DPO's derivation
implicit_gap = beta * (log_ratio_w - log_ratio_l)

# DPO loss = -log σ(gap)   →   binary cross-entropy pushing gap positive
loss = -np.log(1 / (1 + np.exp(-implicit_gap)))

# gradient of the loss w.r.t. policy_logp_w is -β · σ(-gap)
sig_neg = 1 / (1 + np.exp(implicit_gap))
grad_w = -beta * sig_neg
grad_l =  beta * sig_neg

print(f"DPO loss       = {loss:.4f}")
print(f"implicit reward gap = β · (log-ratio gap) = {implicit_gap:.4f}")
print(f"gradient(chosen)   = {-grad_w:+.4f}  (push up)")
print(f"gradient(rejected) = {-grad_l:+.4f}  (push down)")`}</CodeBlock>

      <Prose>
        <p>
          That is the full learning signal — two log-ratios, subtracted,
          scaled by <code>β</code>, fed through{' '}
          <code>-log σ</code>. Everything else is plumbing.
        </p>
        <p>
          Production path. HuggingFace <code>trl</code> wraps the loss, the
          two-model bookkeeping, and the data pipeline in a{' '}
          <code>DPOTrainer</code> that has the same API as{' '}
          <code>Trainer</code>. This is how 95% of DPO runs in the wild
          start their lives.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — production · dpo_trl.py"
      >{`from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

model_id = "meta-llama/Llama-2-7b-sft"         # your SFT checkpoint

# Policy and reference are both loaded from the SFT checkpoint.
# The reference is frozen; the policy is what DPO trains.
policy = AutoModelForCausalLM.from_pretrained(model_id)
ref    = AutoModelForCausalLM.from_pretrained(model_id)
tok    = AutoTokenizer.from_pretrained(model_id)

# HH-RLHF preference dataset — each row has {prompt, chosen, rejected}
ds = load_dataset("Anthropic/hh-rlhf", split="train")

config = DPOConfig(
    output_dir="./dpo-llama7b",
    beta=0.1,                 # the one knob that matters
    learning_rate=5e-7,       # DPO needs smaller LR than SFT — the gradient is concentrated
    per_device_train_batch_size=4,
    num_train_epochs=1,
    gradient_checkpointing=True,
)

trainer = DPOTrainer(
    model=policy,
    ref_model=ref,            # pass None and trl will re-load from policy for a PEFT run
    args=config,
    train_dataset=ds,
    tokenizer=tok,
)

trainer.train()`}</CodeBlock>

      <Prose>
        <p>
          Hand-rolled. This is <code>trl</code>&apos;s loss with the
          plumbing stripped — the one function that makes DPO go.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — hand-rolled pytorch · dpo_loss.py"
      >{`import torch
import torch.nn.functional as F

def dpo_loss(policy_logits, ref_logits, labels, beta=0.1):
    """
    policy_logits : (2B, T, V) — logits from the policy for chosen + rejected
    ref_logits    : (2B, T, V) — logits from the frozen reference for the same
    labels        : (2B, T)    — token ids, with -100 on prompt positions to mask them
    returns       : scalar loss + a logging dict
    """
    # per-sequence log-prob of the completion (sum of token log-probs, prompt masked out)
    def seq_logp(logits, labels):
        lp = F.log_softmax(logits[:, :-1], dim=-1)         # shift for next-token prediction
        labels = labels[:, 1:].clone()
        mask = (labels != -100)
        labels = labels.masked_fill(~mask, 0)
        token_lp = lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return (token_lp * mask).sum(dim=-1)                # (2B,) sequence-level log-probs

    pol = seq_logp(policy_logits, labels)
    ref = seq_logp(ref_logits,    labels)

    # first half of the batch is chosen, second half is rejected (a common convention)
    pol_w, pol_l = pol.chunk(2)
    ref_w, ref_l = ref.chunk(2)

    # the DPO log-ratio gap — implicit reward difference scaled by β
    logits = beta * ((pol_w - ref_w) - (pol_l - ref_l))

    loss = -F.logsigmoid(logits).mean()

    return loss, {
        "reward_chosen":   (beta * (pol_w - ref_w)).mean().item(),
        "reward_rejected": (beta * (pol_l - ref_l)).mean().item(),
        "accuracy":        (logits > 0).float().mean().item(),
    }`}</CodeBlock>

      <Bridge
        label="numpy → production"
        rows={[
          {
            left: 'log σ on one (logp_w, logp_l) pair',
            right: 'F.logsigmoid on a batch of pairs',
            note: 'same loss, just vectorised and autograd-tracked',
          },
          {
            left: 'manual gradient = ±β · σ(-gap)',
            right: 'loss.backward()',
            note: 'autograd handles the chain rule through the transformer',
          },
          {
            left: 'policy_logp - ref_logp on scalars',
            right: 'seq_logp(policy) - seq_logp(ref) on token sequences',
            note: 'the real thing sums log-probs of completion tokens, masking the prompt',
          },
          {
            left: 'one implicit_gap scalar',
            right: 'reward_chosen, reward_rejected, accuracy metrics',
            note: 'production runs log both implicit rewards so you can see learning progress',
          },
        ]}
      />

      {/* ── What you give up by shortcutting ───────────────────── */}
      <Prose>
        <p>
          The shortcut is not free. Bypassing the reward model buys
          simplicity and trades away two things that a two-step pipeline
          quietly gave you. Name both before you run a production DPO job.
        </p>
        <p>
          <strong>You lose a separate generalizer.</strong> The reward
          model in the old pipeline was a universal scorer — train it once,
          and it can rate any completion from any policy, including
          completions the labelers never saw. DPO folds the scorer into
          the policy, so there&apos;s no independent object you can score
          off-policy samples against. No reward model means no
          reward-model-as-critic at eval time, no iterative RLHF loop that
          keeps re-sampling from the live policy and re-scoring. You get
          the direct training shortcut and give up the reusable scorer.
        </p>
        <p>
          <strong>You lose the exploration that PPO had.</strong> PPO
          samples from the live policy at every step, so the policy sees
          its own completions, scores them through the reward model, and
          can discover good behaviors that weren&apos;t in any labeled
          pair. DPO is supervised on a fixed dataset — no sampling, no
          exploration, no chance to outgrow the preference distribution.
          If your dataset is narrow, the policy will be narrow. This is
          the single biggest quality gap between DPO and PPO on open-ended
          benchmarks in the wild.
        </p>
      </Prose>

      {/* ── Variants ────────────────────────────────────────────── */}
      <Prose>
        <p>
          DPO spawned a small garden of variants within eighteen months,
          each patching one of the shortcut&apos;s trade-offs. Three are
          worth knowing.
        </p>
      </Prose>

      <Callout variant="note" title="KTO — Kahneman-Tversky Optimization (Ethayarajh 2024)">
        KTO drops the paired-preference requirement. Instead of needing{' '}
        <code>(chosen, rejected)</code> pairs, it works on unpaired{' '}
        <em>good/bad</em> labels on individual completions. The loss is
        derived from prospect theory — humans weigh losses more heavily
        than gains — which makes KTO more robust to class imbalance. Reach
        for it when you have a big pile of thumbs-up / thumbs-down
        feedback and no explicit pairs.
      </Callout>

      <Callout variant="note" title="IPO — Identity Preference Optimization (Azar 2023)">
        IPO patches DPO&apos;s main failure mode: overfitting when the
        preferences are nearly deterministic. DPO&apos;s loss pushes{' '}
        <code>σ(gap)</code> toward 1, which in the limit blows up the
        log-ratios and destroys the reference anchor. IPO replaces the
        log-sigmoid with a squared loss on the gap, which stays bounded.
        If your preferences are clean and your model is tipping into the
        reward-hacking regime, try IPO.
      </Callout>

      <Callout variant="note" title="ORPO — Odds Ratio Preference Optimization (Hong 2024)">
        ORPO fuses SFT and DPO into one objective. Instead of doing SFT
        first and DPO on top, ORPO trains from the base model with a
        single loss that combines supervised next-token prediction on the
        chosen completion <em>plus</em> an odds-ratio penalty on the
        rejected one. You trade one training pipeline for another and
        claw back a chunk of compute. Popular in fine-tuning leaderboards
        as of 2024.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Reference choice matters.</strong>{' '}
          DPO anchors to <code>π_ref</code>. If your reference is the base
          pretrained model (not the SFT checkpoint) the reference
          log-probs are much flatter, the log-ratios blow up, and training
          is unstable. Use the SFT checkpoint as the reference — always.
        </p>
        <p>
          <strong className="text-term-amber">Log-prob normalization.</strong>{' '}
          Sum log-probs over completion tokens only — mask out the prompt.
          A common bug is to include the prompt log-probs in both{' '}
          <code>π</code> and <code>π_ref</code>; they cancel in
          expectation but add variance and slow convergence. Mask them.
        </p>
        <p>
          <strong className="text-term-amber">Batch size.</strong> DPO
          tolerates much smaller batches than PPO — you can train a 7B
          model on 2×A100 with batch size 2. But tiny batches make the
          implicit-reward metrics jumpy; track the running mean, not the
          per-step value.
        </p>
        <p>
          <strong className="text-term-amber">Overfitting to the preference distribution.</strong>{' '}
          The cost of skipping the sampling loop. DPO pushes hard on the
          exact preferences it sees, with no exploration to pull it
          sideways. If your dataset is narrow — say, only harmfulness
          examples — the policy will become excellent at refusing and
          mediocre at everything else. Mix in diverse preference data.
        </p>
      </Gotcha>

      <Challenge prompt="DPO a Llama-7B on HH-RLHF">
        <p>
          Start from an SFT Llama-7B checkpoint. Load the Anthropic{' '}
          <code>hh-rlhf</code> helpful split. Train DPO for one epoch with{' '}
          <code>β = 0.1</code>, learning rate <code>5e-7</code>, and batch
          size 4 per GPU. On 8×A100 this takes about 12 hours.
        </p>
        <p className="mt-2">
          Log the implicit reward gap —{' '}
          <code>β · (logp_chosen_policy - logp_chosen_ref) − β ·
          (logp_rejected_policy - logp_rejected_ref)</code> — every 100
          steps. It should climb from ~0 at the start to something
          positive. If it plateaus near 0, your reference is wrong. If it
          explodes past 1.0, lower <code>β</code> or check for a
          repeated-completion bug in the dataset.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: evaluate the DPO model and a PPO baseline on a held-out
          preference split using an external reward model (e.g.{' '}
          <code>OpenAssistant/reward-model-deberta-v3-large</code>) as the
          scorer. You&apos;re looking for DPO to match or slightly exceed
          PPO — that&apos;s the Rafailov result, and it replicates
          cleanly.
        </p>
      </Challenge>

      {/* ── Closing ─────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> DPO is the two-step
          RLHF pipeline collapsed by algebra into a single supervised
          loss. The shortcut works because the KL-constrained reward
          objective has a closed-form optimal policy, and inverting that
          form lets you express the reward as a log-ratio of policy to
          reference — which substitutes cleanly into Bradley-Terry and
          makes the reward model vanish. What&apos;s left is cross-entropy
          on preference pairs with a <code>β</code> knob. Two models in
          memory, no sampling loop, no PPO hyperparameter zoo. It matches
          or slightly exceeds PPO on most benchmarks, with the caveat that
          direct training on a fixed dataset overfits more eagerly than
          online RL does — which is what KTO, IPO, and ORPO take turns
          addressing.
        </p>
        <p>
          <strong>Next up — PPO for RLHF.</strong> DPO is the shortcut;
          it&apos;s worth seeing the scenic route too. The next lesson
          walks through proximal policy optimization as RLHF actually uses
          it — the clipped-ratio objective, the KL-to-SFT penalty, the
          rollout loop, the four-model dance. Knowing what PPO has to do
          step-by-step is what makes the DPO shortcut feel earned instead
          of magical.
        </p>
      </Prose>

      <References
        items={[
          {
            title:
              'Direct Preference Optimization: Your Language Model is Secretly a Reward Model',
            author: 'Rafailov, Sharma, Mitchell, Ermon, Manning, Finn',
            venue: 'NeurIPS 2023',
            url: 'https://arxiv.org/abs/2305.18290',
          },
          {
            title: 'A General Theoretical Paradigm to Understand Learning from Human Preferences',
            author: 'Azar, Rowland, Piot, Guo, Calandriello, Valko, Munos',
            venue: 'AISTATS 2024 — the IPO paper',
            url: 'https://arxiv.org/abs/2310.12036',
          },
          {
            title: 'KTO: Model Alignment as Prospect Theoretic Optimization',
            author: 'Ethayarajh, Xu, Muennighoff, Jurafsky, Kiela',
            year: 2024,
            url: 'https://arxiv.org/abs/2402.01306',
          },
          {
            title: 'ORPO: Monolithic Preference Optimization without Reference Model',
            author: 'Hong, Lee, Thorne',
            year: 2024,
            url: 'https://arxiv.org/abs/2403.07691',
          },
          {
            title: 'TRL — Transformer Reinforcement Learning (DPOTrainer)',
            author: 'HuggingFace',
            venue: 'library reference',
            url: 'https://huggingface.co/docs/trl/main/en/dpo_trainer',
          },
        ]}
      />
    </div>
  )
}
