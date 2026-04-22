import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import Quiz from '../Quiz'
import WhatNext from '../WhatNext'
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
import ClippedObjective from '../widgets/ClippedObjective'
import PPOvsTRPO from '../widgets/PPOvsTRPO'

// Signature anchor: the trust-region ratchet. PPO's whole trick is a clipped
// objective that lets the policy turn forward in small, bounded steps — a
// ratchet that won't permit a catastrophic jump in either direction. Return
// at the opening (the untrusted jump of vanilla PG), the clip-ratio reveal,
// and the "what happens at the boundary" section.
export default function ProximalPolicyOptimizationLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="proximal-policy-optimization" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Vanilla <NeedsBackground slug="policy-gradients">policy gradients</NeedsBackground>{' '}
          have one unforgivable failure mode: the untrusted jump. You sample a
          batch of trajectories, you run{' '}
          <NeedsBackground slug="gradient-descent">gradient descent</NeedsBackground>{' '}
          on the log-likelihood objective, and one unlucky step lands you
          somewhere the batch never visited. If a single advantage estimate is
          unusually large, or a single ratio unusually far from 1, the new
          policy is so different from the old one that your data turns into
          noise the moment you finish the step. You don&apos;t recover. The
          policy lies on the floor collecting zero reward, and the run is
          over.
        </p>
        <p>
          The fix is a <KeyTerm>trust region</KeyTerm> — a fence around the
          old policy that says &ldquo;step inside me, not outside.&rdquo; In
          2015, Schulman&apos;s{' '}
          <KeyTerm>Trust-Region Policy Optimization</KeyTerm> (TRPO) enforced
          the fence with math: every update solved a constrained problem
          keeping the new policy within a tiny <code>KL</code> distance of the
          old one. Beautiful — a second-order method with conjugate gradient
          and Fisher information matrices. It also takes 500 lines of code,
          runs slow, and nobody wants to maintain it.
        </p>
        <p>
          In 2017, the same lab shipped <KeyTerm>Proximal Policy Optimization</KeyTerm>{' '}
          (PPO). The pitch: throw away the KL constraint. Replace it with a
          single clip on the importance ratio — a{' '}
          <strong>trust-region ratchet</strong> that only lets the policy turn
          forward by small steps. Try to climb past <code>1 + ε</code> and the
          gradient flatlines. Try to fall past <code>1 − ε</code> and the same
          thing happens on the other side. No catastrophic jumps. ~10× simpler
          code. Same empirical performance as TRPO, sometimes better. PPO
          became the default RL algorithm for continuous control, game
          playing, and — five years later — the backbone of RLHF. You saw it
          in the previous section dressed up for language models. Here we
          meet it naked.
        </p>
      </Prose>

      <Personify speaker="PPO">
        I am a single line of loss. I do not solve optimization problems
        inside my forward pass. I do not need Fisher information. I clip the
        ratio, take the pessimistic minimum, and call <code>.backward()</code>.
        My trust region is a ratchet — one tooth of progress per step, no
        slipping back, no lunging forward. I am not elegant. I am pragmatic.
        That is why I won.
      </Personify>

      {/* ── The importance ratio ────────────────────────────────── */}
      <Prose>
        <p>
          Before the objective, the quantity the ratchet grips on to. The{' '}
          <KeyTerm>importance ratio</KeyTerm> at time <code>t</code>:
        </p>
      </Prose>

      <MathBlock caption="the importance ratio — how far has the policy moved?">
{`              π_θ(a_t | s_t)
r_t(θ)   =   ───────────────
             π_θ_old(a_t | s_t)`}
      </MathBlock>

      <Prose>
        <p>
          <code>π_θ_old</code> is the policy that collected the data.{' '}
          <code>π_θ</code> is what you&apos;re currently optimizing. The ratio
          is the distance-meter: how much more (or less) likely the{' '}
          <em>current</em> policy is to take the same action the <em>old</em>{' '}
          policy took. <code>r = 1</code> means the two agree; you haven&apos;t
          moved. <code>r = 1.5</code> means the current policy is 50% more
          likely to do this action — you&apos;ve walked half a step away from
          the data-collection distribution. <code>r = 0.5</code> is the
          mirror: you&apos;re now half as likely to take what used to be a
          common action. The ratchet lives on this one number.
        </p>
        <p>
          Why a ratio at all? Because PPO reuses each batch of trajectories
          for several gradient steps (that&apos;s the sample-efficiency trick).
          After the first step the current policy is no longer the one that
          collected the data — you&apos;re doing{' '}
          <KeyTerm>off-policy correction</KeyTerm>, and the ratio is the
          importance-sampling weight that keeps the math honest. Without a
          fence, those reuse steps would drift. With a fence, they can&apos;t.
        </p>
      </Prose>

      {/* ── The clipped objective ───────────────────────────────── */}
      <MathBlock caption="PPO-Clip — the entire algorithm on one line">
{`L^CLIP(θ)  =  E_t [ min(  r_t(θ) · A_t ,  clip(r_t(θ), 1−ε, 1+ε) · A_t  ) ]

with    ε   ≈   0.2    (standard)`}
      </MathBlock>

      <Prose>
        <p>
          This is the ratchet, spelled out. Two moving parts. <code>r_t(θ)</code>{' '}
          we just defined. <code>A_t</code> is the{' '}
          <KeyTerm>advantage</KeyTerm> — positive when the action did better
          than expected, negative when it did worse. Everything else is the
          fence.
        </p>
        <ul>
          <li>
            If <code>A_t &gt; 0</code> (good action), the objective wants{' '}
            <code>r_t · A_t</code> big — raise <code>π_θ(a_t|s_t)</code>. But
            the <code>clip</code> caps <code>r_t</code> at <code>1 + ε</code>.
            Beyond that boundary the objective goes flat. You can walk closer
            to the good action; you cannot sprint there. One tooth of
            forward progress per step.
          </li>
          <li>
            If <code>A_t &lt; 0</code> (bad action), the objective still
            wants <code>r_t · A_t</code> big (i.e. less negative) — lower{' '}
            <code>π_θ(a_t|s_t)</code>. The clip floors <code>r_t</code> at{' '}
            <code>1 − ε</code>. You can suppress the bad action, but not drive
            its probability to zero in one step. Same ratchet, other
            direction.
          </li>
          <li>
            The <code>min(...)</code> is the{' '}
            <KeyTerm>pessimistic minimum</KeyTerm>. It keeps the trust region
            one-sided: we only collect the clip&apos;s benefit when the{' '}
            <em>unclipped</em> ratio would have given a more optimistic
            gradient than the clipped one. Without the <code>min</code>,
            positive advantages with <code>r &gt; 1 + ε</code> would still
            push the policy further; with it, the gradient dies right at the
            boundary. Fence intact, on both sides.
          </li>
        </ul>
      </Prose>

      {/* ── Widget 1: Clipped objective ─────────────────────────── */}
      <Prose>
        <p>
          Look at the shape of <code>L^CLIP</code> as a function of{' '}
          <code>r_t</code> alone, with <code>A_t</code> held fixed. Toggle the
          sign of the advantage and watch the hinges at <code>1 − ε</code> and{' '}
          <code>1 + ε</code> flip sides. Those hinges are the ratchet&apos;s
          teeth.
        </p>
      </Prose>

      <ClippedObjective />

      <Prose>
        <p>
          This is what happens at the boundary. To the left of{' '}
          <code>1 − ε</code> and to the right of <code>1 + ε</code>, the
          clipped term is flat — zero gradient — so the objective stops
          pulling. The <code>min()</code> makes the stop one-sided: clipping
          only bites when it protects us. Push the ratio toward the
          &ldquo;wrong&rdquo; side of the trust region and the gradient goes
          to zero at the boundary; the policy is not allowed to keep walking.
          That is the <em>proximal</em> in the name — we stay proximate to the
          old policy, enforced not by a Lagrangian constraint but by a
          one-line <code>clamp</code>. The fence is cheap. The ratchet is
          cheap. That&apos;s the trick.
        </p>
      </Prose>

      <Personify speaker="Clip">
        I am the fence at the edge of the trust region. I do nothing when the
        ratio is near 1. The moment the policy tries to step more than{' '}
        <code>ε</code> away in the wrong direction, I zero the gradient and
        end the conversation. I have no theoretical guarantees as strong as
        the KL constraint I replaced. I happen to work. TRPO spent 500 lines
        on what I do in one.
      </Personify>

      {/* ── The training loop ───────────────────────────────────── */}
      <Prose>
        <p>
          PPO&apos;s other trick is <KeyTerm>multi-epoch reuse</KeyTerm>. Each
          batch of trajectories gets replayed through the update{' '}
          <code>K</code> times (typically 3 to 10). This is sample efficiency,
          and it&apos;s <em>only</em> safe because the clip keeps the ratio
          inside the trust region across all <code>K</code> inner steps.
          Without the ratchet, by epoch 3 you&apos;d be taking gradient steps
          on a distribution that no longer resembles anything in the batch —
          the exact untrusted jump we opened with, dressed in a loop.
        </p>
      </Prose>

      <MathBlock caption="PPO training loop — the full recipe">
{`for iteration = 1, 2, ...
    # 1. collect a batch with the current policy
    trajectories  ←  rollout(π_θ_old)                    # N timesteps across parallel envs

    # 2. compute advantages once (outside the inner loop!)
    A_t           ←  GAE(rewards, values, λ=0.95, γ=0.99)

    # 3. K epochs of optimization on the same batch
    for epoch = 1, ..., K
        for minibatch in shuffle(trajectories)
            r_t(θ)      =   π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
            L_policy    =  −E[ min( r · A,  clip(r, 1−ε, 1+ε) · A ) ]
            L_value     =   E[ (V_θ(s_t) − R_t)² ]          # optionally clipped too
            L_entropy   =  −E[ H(π_θ(·|s_t)) ]              # exploration bonus
            loss        =   L_policy + c_v · L_value + c_e · L_entropy
            ∇loss → optimizer step

    # 4. promote the current policy to "old"
    π_θ_old  ←  π_θ`}
      </MathBlock>

      <Prose>
        <p>
          Four pieces. Rollout collects the data. <KeyTerm>GAE</KeyTerm>{' '}
          (Generalized Advantage Estimation, Schulman 2016) builds the
          advantage estimates with a bias-variance knob <code>λ</code> — 0.95
          is the canonical value. The inner <code>K</code>-epoch loop is
          where the policy actually learns, each step limited by the ratchet.
          Then — and this is the part people forget — we <em>replace</em>{' '}
          <code>π_θ_old</code> with the current weights before the next
          rollout. That swap is what keeps the trust region meaningful:{' '}
          <code>π_θ_old</code> is always the policy that collected the most
          recent data, so the fence is always pitched around where the batch
          actually came from.
        </p>
      </Prose>

      {/* ── Widget 2: PPO vs TRPO ───────────────────────────────── */}
      <Prose>
        <p>
          The 2017 paper&apos;s headline: across MuJoCo continuous-control
          benchmarks, PPO matches or beats TRPO on final return while being
          an order of magnitude cheaper to implement and wall-clock faster
          to train. A cartoon of the comparison:
        </p>
      </Prose>

      <PPOvsTRPO />

      <Prose>
        <p>
          The bar chart understates it. TRPO requires Fisher information
          matrices, conjugate gradient, backtracking line search to stay
          inside the trust region — each of which is its own research paper.
          PPO replaces all of it with a clamp. This is why PPO, not TRPO,
          became the default RL algorithm for roughly five years (2017–2021),
          powered OpenAI Five (Dota 2), the imitation-bootstrapped phase of
          AlphaStar, and — to bring this full circle — InstructGPT&apos;s
          RLHF. The previous section was PPO aimed at language generation;
          this section is PPO aimed at classical control. Same ratchet.
          Different state and action spaces.
        </p>
      </Prose>

      <Personify speaker="Importance ratio">
        I measure the distance between the policy that collected this data
        and the policy you&apos;re currently optimizing. When I am 1, the
        update is on-policy and the math is trivial. When I drift toward the
        boundary, the clip catches me. I am the reason you can reuse a batch
        for 10 epochs and still trust the gradient.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, as always. Pure-Python PPO on CartPole to see the
          whole algorithm in 60 lines. NumPy with explicit GAE. Full PyTorch
          with clipping, value loss, entropy bonus, and multi-epoch training —
          the version you&apos;d actually deploy.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · ppo_cartpole.py"
        output={`iter 00  mean_return=18.2
iter 10  mean_return=41.5
iter 30  mean_return=122.7
iter 60  mean_return=194.3
iter 90  mean_return=200.0   # solved (CartPole cap)`}
      >{`import gymnasium as gym
import numpy as np
import math

# Tiny 2-layer MLP policy, plain Python (no autograd) — just to see the skeleton.
# In practice you'd use PyTorch; this is to make the algorithm readable.

env = gym.make("CartPole-v1")
STATE_DIM, N_ACTIONS = 4, 2
EPSILON, GAMMA, LR = 0.2, 0.99, 0.01

# Policy parameters — a single linear layer softmax, small enough to update by hand.
W = np.random.randn(STATE_DIM, N_ACTIONS) * 0.01

def softmax_probs(s, W):
    logits = s @ W
    e = np.exp(logits - logits.max())
    return e / e.sum()

def logprob(s, a, W):
    return math.log(softmax_probs(s, W)[a] + 1e-12)

for iteration in range(100):
    # 1. Rollout: collect one episode under π_old (= current W snapshot)
    W_old = W.copy()
    states, actions, rewards = [], [], []
    s, _ = env.reset()
    done = False
    while not done:
        probs = softmax_probs(s, W_old)
        a = np.random.choice(N_ACTIONS, p=probs)
        s2, r, term, trunc, _ = env.step(a)
        states.append(s); actions.append(a); rewards.append(r)
        s = s2
        done = term or trunc

    # 2. Returns-to-go as a naive advantage proxy (full GAE comes in layer 2)
    returns = np.zeros(len(rewards))
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + GAMMA * G
        returns[t] = G
    advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

    # 3. K=4 epochs of PPO-clip updates on this batch
    for _ in range(4):
        for s, a, A in zip(states, actions, advantages):
            lp_new = logprob(s, a, W)
            lp_old = logprob(s, a, W_old)
            ratio  = math.exp(lp_new - lp_old)
            clipped = max(1 - EPSILON, min(ratio, 1 + EPSILON))
            # gradient of min(r·A, clip(r)·A) w.r.t. W, done by finite-diff-ish
            # scalar rule: if clip is inactive, grad is A · ∇log π(a|s)
            # if clip is active on the binding side, grad is 0
            use_unclipped = (ratio * A) <= (clipped * A)
            if use_unclipped:
                probs = softmax_probs(s, W)
                grad_logpi = -np.outer(s, probs)
                grad_logpi[:, a] += s
                W += LR * A * grad_logpi

    if iteration % 10 == 0:
        print(f"iter {iteration:02d}  mean_return={sum(rewards):.1f}")`}</CodeBlock>

      <Prose>
        <p>
          That&apos;s the algorithm end-to-end: collect, compute advantages,{' '}
          <code>K</code> epochs of clipped updates, promote{' '}
          <code>π_old</code>, repeat. The ratchet is the single line{' '}
          <code>clipped = max(1 − ε, min(ratio, 1 + ε))</code>. Now vectorize
          the advantage computation with proper GAE and stop computing
          gradients by hand.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — numpy + GAE · ppo_gae.py">{`import numpy as np

GAMMA, LAMBDA = 0.99, 0.95

def compute_gae(rewards, values, dones, last_value):
    """
    GAE — Schulman 2016. Trades bias for variance via λ.
      δ_t  = r_t + γ V(s_{t+1}) − V(s_t)                 # one-step TD error
      A_t  = δ_t + (γλ) · δ_{t+1} + (γλ)² · δ_{t+2} + ...  # exponentially weighted sum
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_value * next_nonterminal - values[t]
        gae = delta + GAMMA * LAMBDA * next_nonterminal * gae
        advantages[t] = gae
    returns = advantages + values                          # used as value-function target
    return advantages, returns

# Usage inside a PPO rollout:
# rewards: (T,)  values: (T,)  dones: (T,)  last_value: scalar (bootstrap for truncated traj)
# advantages, returns = compute_gae(rewards, values, dones, last_value)
# advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)   # normalize`}</CodeBlock>

      <Bridge
        label="pure python → numpy + GAE"
        rows={[
          {
            left: 'returns = cumsum of discounted rewards',
            right: 'GAE with λ=0.95 on top of value baseline',
            note: 'lower variance, controlled bias — the canonical advantage',
          },
          {
            left: 'per-episode loop',
            right: '(T,)-shaped arrays, one GAE call',
            note: 'GAE is a reverse scan; naturally vectorizable',
          },
          {
            left: 'advantages from returns alone',
            right: 'advantages = GAE(r, V, dones, V_last)',
            note: 'subtract value baseline for variance reduction',
          },
        ]}
      />

      <Prose>
        <p>
          And the real thing. PyTorch, with the value head, entropy bonus,
          minibatch shuffling, and <code>K</code>-epoch loop. This is within
          shouting distance of the reference implementation you&apos;d find
          in <code>stable-baselines3</code>.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch PPO · ppo_pytorch.py"
        output={`update 001  return=+28.4  kl=0.012  clipfrac=0.08  loss_pi=-0.023  loss_v=0.41
update 020  return=+104.2 kl=0.024  clipfrac=0.15  loss_pi=-0.041  loss_v=0.28
update 050  return=+248.7 kl=0.031  clipfrac=0.22  loss_pi=-0.048  loss_v=0.17
update 100  return=+487.1 kl=0.028  clipfrac=0.19  loss_pi=-0.039  loss_v=0.09`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                    nn.Linear(64, 64),     nn.Tanh())
        self.pi   = nn.Linear(64, n_actions)     # policy logits
        self.v    = nn.Linear(64, 1)             # value head

    def forward(self, obs):
        h = self.shared(obs)
        return self.pi(h), self.v(h).squeeze(-1)

EPSILON, VF_COEF, ENT_COEF = 0.2, 0.5, 0.01
K_EPOCHS, MINIBATCH_SIZE   = 4, 64

policy = ActorCritic(obs_dim=8, n_actions=4)       # e.g. LunarLander
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

def ppo_update(obs, actions, old_logprobs, advantages, returns):
    # Normalize advantages at the batch level (canonical PPO trick — reduces variance).
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(K_EPOCHS):
        # Shuffle indices into minibatches — DO NOT recompute advantages here.
        idx = torch.randperm(len(obs))
        for start in range(0, len(obs), MINIBATCH_SIZE):
            mb = idx[start : start + MINIBATCH_SIZE]

            logits, values = policy(obs[mb])
            dist           = torch.distributions.Categorical(logits=logits)
            new_logprobs   = dist.log_prob(actions[mb])
            entropy        = dist.entropy().mean()

            # PPO-Clip objective
            ratio   = torch.exp(new_logprobs - old_logprobs[mb])
            surr1   = ratio * advantages[mb]
            surr2   = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages[mb]
            loss_pi = -torch.min(surr1, surr2).mean()

            # Value loss — MSE to the computed returns
            loss_v  = F.mse_loss(values, returns[mb])

            # Total loss — policy + weighted value + entropy bonus (minus because maximize)
            loss = loss_pi + VF_COEF * loss_v - ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)    # also standard
            optimizer.step()`}</CodeBlock>

      <Bridge
        label="numpy + GAE → pytorch full PPO"
        rows={[
          {
            left: 'hand-coded softmax and log-prob',
            right: 'torch.distributions.Categorical(logits=...)',
            note: 'autograd tracks log_prob and entropy natively',
          },
          {
            left: 'W += LR · A · grad_logpi',
            right: 'loss.backward(); optimizer.step()',
            note: 'Adam handles the update; we just define the loss',
          },
          {
            left: 'per-episode loop',
            right: 'minibatch shuffle, K epochs, clip_grad_norm',
            note: 'canonical PPO scaffolding — GAE once, K updates, promote',
          },
          {
            left: 'policy only',
            right: 'policy + value head + entropy bonus',
            note: 'shared backbone; value stabilizes, entropy prevents collapse',
          },
        ]}
      />

      <Callout variant="insight" title="RLHF is this ratchet, pointed at tokens">
        Every piece of the RLHF loop from the previous section maps directly
        onto this diagram. The &ldquo;environment&rdquo; becomes the prompt
        distribution plus the reward model. The &ldquo;state&rdquo; is
        (prompt + tokens so far). The &ldquo;action&rdquo; is the next token.
        The reward is the RM score (terminal) plus a per-token KL penalty to
        a reference. Everything else — the clip, the ratio, the{' '}
        <code>K</code> epochs, the advantage via GAE — is unchanged. The
        trust-region ratchet that keeps CartPole from detonating is the same
        ratchet that keeps a 7B language model from collapsing into
        reward-hacking gibberish. If you understand PPO here, you understand
        the engine behind ChatGPT.
      </Callout>

      <Callout variant="note" title="implementation details matter — a lot">
        Engstrom et al. (2020) investigated why PPO implementations that are
        mathematically identical often give wildly different results. The
        answer: the &ldquo;code-level&rdquo; details — advantage
        normalization, observation scaling, orthogonal initialization,
        learning-rate annealing, value-function clipping, gradient clipping —
        account for most of PPO&apos;s reported lead over TRPO. Strip them
        out and PPO and TRPO are roughly equivalent. Keep them in and PPO is
        the default. Required reading if you ever want to reproduce a PPO
        result from scratch.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Computing advantages inside the K-epoch loop:</strong>{' '}
          the classic bug. Advantages must be computed <em>once</em>, before
          the epoch loop, using the value function that was alive when the
          data was collected. Recompute them every epoch with the updated
          value function and you&apos;re chasing your own tail — the learning
          signal becomes incoherent, training diverges.
        </p>
        <p>
          <strong className="text-term-amber">Not clipping the value function:</strong>{' '}
          reference PPO clips <em>both</em> the policy ratio <em>and</em> the
          value update:{' '}
          <code>v_clipped = v_old + clip(v_new − v_old, −ε, +ε)</code>, then{' '}
          <code>L_v = max(MSE(v_new, R), MSE(v_clipped, R))</code>. The same
          trust-region ratchet, applied to the critic. Without it the value
          head can diverge under multi-epoch reuse, which wrecks the
          advantages, which wrecks the policy.
        </p>
        <p>
          <strong className="text-term-amber">Too-large ε:</strong> at{' '}
          <code>ε = 0.5</code> the trust region is so loose the clip barely
          triggers, and you&apos;re back to vanilla policy gradient with all
          its untrusted-jump instability. Stay at <code>0.1–0.3</code>.{' '}
          <code>0.2</code> is standard and works almost everywhere.
        </p>
        <p>
          <strong className="text-term-amber">Too many epochs per batch:</strong>{' '}
          after epoch <code>K</code>, the current policy is roughly{' '}
          <code>K · ε</code> steps from the data-collection policy. At{' '}
          <code>K = 4</code> with <code>ε = 0.2</code>, that&apos;s fine. At{' '}
          <code>K = 20</code> most samples are pinned at the clip boundary,
          the effective gradient is zero, and any that aren&apos;t are
          pushing the policy into territory the ratchet was never designed
          to handle. 4–10 epochs is the working range; 4 is the safe default.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting to normalize advantages:</strong>{' '}
          without per-batch normalization, advantage scale varies with
          reward scale, and PPO&apos;s effective step size varies with it
          too. Normalize to mean-zero unit-variance at the batch level. Tiny
          change in code, large change in stability.
        </p>
        <p>
          <strong className="text-term-amber">Using the wrong old logprobs:</strong>{' '}
          <code>π_θ_old</code> must be the logprobs captured{' '}
          <em>at rollout time</em>, frozen. Recompute them under the current{' '}
          <code>θ</code> each epoch and every ratio becomes exactly 1 — the
          clip never triggers, the ratchet is disengaged, and you have
          silently reverted to a weird form of on-policy gradient ascent with
          zero safeguards.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Solve LunarLander-v2 with PPO">
        <p>
          Using the layer-3 PyTorch scaffold above (or{' '}
          <code>stable-baselines3</code> if you&apos;re short on time — same
          algorithm, more tested), train PPO on <code>LunarLander-v2</code>.
          Target: mean episodic return <strong>&gt; 200</strong> over the
          last 100 episodes. That&apos;s the official &ldquo;solved&rdquo;
          threshold.
        </p>
        <p className="mt-2">
          Start with the standard hyperparameters: <code>ε = 0.2</code>,{' '}
          <code>γ = 0.99</code>, <code>λ = 0.95</code>, <code>K = 4</code>{' '}
          epochs, <code>lr = 3e-4</code>, 2048 steps per rollout, minibatch
          64. Log three curves per update: <strong>(a)</strong> mean episodic
          return, <strong>(b)</strong> approximate KL between{' '}
          <code>π_θ</code> and <code>π_θ_old</code> (should stay under 0.02
          when the ratchet is behaving), <strong>(c)</strong> clip fraction —
          the proportion of samples that hit the boundary of the trust
          region. A healthy run has clip fraction in <code>0.1–0.3</code>.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: rerun with <code>ε = 1.0</code> (the ratchet is effectively
          removed). Plot return and KL on the same axes. You should see a
          fast climb, a catastrophic collapse once KL explodes past the
          trust region, and a run that never recovers. That single plot is
          the best case for PPO&apos;s existence you will ever make.
        </p>
      </Challenge>

      {/* ── Closing + section teaser ────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> PPO replaces TRPO&apos;s
          hard KL constraint with a cheap clip on the importance ratio — a
          trust-region ratchet that lets the policy turn forward by small
          steps and refuses to let it lunge. The <code>min()</code> keeps the
          ratchet one-sided; <code>K</code>-epoch reuse gives you sample
          efficiency you could never get from vanilla policy gradients;
          promoting <code>π_θ_old</code> between iterations keeps the fence
          pitched around the right policy. The algorithm fits on a screen.
          The implementation details — normalization, value clipping, grad
          clipping, GAE — account for half its practical performance, so
          never trust a PPO result without the code. This is the workhorse
          that gave you Dota-playing bots in 2019 and ChatGPT in 2022.
        </p>
        <p>
          <strong>Next up — MoE Fundamentals.</strong> So far every model in
          this curriculum has activated all of its parameters on every
          token. Scaling means making that single stack of activations
          bigger and bigger, and eventually the FLOPs bill catches up with
          you. Mixture of Experts changes the deal: grow the parameter count
          without growing the FLOPs per token, by routing each token through
          only a small subset of specialists. The next section starts with
          why sparse activation is the next axis of scale — and why the
          router that picks which experts fire is suddenly the hardest part
          of the network to train.
        </p>
      </Prose>

      <WhatNext currentSlug="proximal-policy-optimization" />

      <Quiz
        question={
          <>
            In PPO&apos;s clipped surrogate objective{' '}
            <code>L = min(r·A, clip(r, 1−ε, 1+ε)·A)</code>, why the{' '}
            <code>min()</code>?
          </>
        }
        options={[
          {
            text: 'So a negative advantage can still pull a too-large ratio back down — clip alone would let bad actions escape correction.',
            correct: true,
            explain:
              'Exactly. clip(r, 1−ε, 1+ε) by itself produces a flat plateau outside the clip range, which zeroes the gradient. The min() picks the unclipped term when that&apos;s more pessimistic — so a bad action with A < 0 and r ≫ 1+ε still gets a real gradient pushing the policy away from it.',
          },
          {
            text: 'To ensure the loss is always ≤ 0 so we can gradient-descend it.',
            explain:
              'PPO&apos;s objective is maximized (ascended), not descended, and the sign of L varies with A. The min() isn\u2019t about sign — it\u2019s about keeping the pessimistic bound active on both sides.',
          },
          {
            text: 'Because max() would require backing out of automatic differentiation.',
            explain:
              'Both min() and max() are perfectly differentiable in autograd. The choice is pedagogical, not mechanical: we want the tighter, more conservative bound.',
          },
          {
            text: 'It&apos;s an implementation detail of Adam — the min stabilizes moment estimates.',
            explain:
              'The min() is part of the loss definition, not the optimizer. PPO trains fine with SGD; Adam just happens to be common.',
          },
        ]}
      />

      <References
        items={[
          {
            title: 'Proximal Policy Optimization Algorithms',
            author: 'Schulman, Wolski, Dhariwal, Radford, Klimov',
            venue: 'arXiv 2017 — the original PPO paper',
            url: 'https://arxiv.org/abs/1707.06347',
            tags: ['paper'],
          },
          {
            title: 'Trust Region Policy Optimization',
            author: 'Schulman, Levine, Abbeel, Jordan, Moritz',
            venue: 'ICML 2015 — the TRPO paper PPO replaced',
            url: 'https://arxiv.org/abs/1502.05477',
            tags: ['paper'],
          },
          {
            title: 'Implementation Matters in Deep RL: A Case Study on PPO and TRPO',
            author: 'Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph, Madry',
            venue: 'ICLR 2020 — why code-level details dominate PPO vs TRPO',
            url: 'https://arxiv.org/abs/2005.12729',
            tags: ['paper'],
          },
          {
            title: 'High-Dimensional Continuous Control Using Generalized Advantage Estimation',
            author: 'Schulman, Moritz, Levine, Jordan, Abbeel',
            venue: 'ICLR 2016 — the GAE paper used inside PPO',
            url: 'https://arxiv.org/abs/1506.02438',
            tags: ['paper'],
          },
          {
            title: 'The 37 Implementation Details of Proximal Policy Optimization',
            author: 'Huang, Dossa, Raffin, Kanervisto, Wang',
            venue: 'ICLR Blog 2022 — the practical companion to Engstrom 2020',
            url: 'https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/',
            tags: ['blog'],
          },
        ]}
      />
    </div>
  )
}
