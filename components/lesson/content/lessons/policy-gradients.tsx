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
import PolicyGradientDerivation from '../widgets/PolicyGradientDerivation'
import ReturnToGradient from '../widgets/ReturnToGradient'

// Signature anchor: the sensitivity dial. A policy is a dashboard of knobs
// (θ); the policy gradient tells you which knob, nudged a hair, pays off
// most — a sensitivity reading, averaged over episodes. Returned to at the
// opening (the knobs on the dashboard), the score-function-gradient reveal
// (∇ log π × return is the sensitivity reading), and the "why this
// generalizes REINFORCE" section. Peer anchors we do not reuse: MDP's
// board game, REINFORCE's casino tracker, Q-learning's cheat sheet,
// actor-critic's playwright+critic.
export default function PolicyGradientsLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="policy-gradients" />

      {/* ── Opening: the dashboard of knobs ─────────────────────── */}
      <Prose>
        <p>
          Picture the policy as a dashboard. Every weight in the network is a
          knob. Your agent twists a particular set of those knobs and spits
          out a distribution over actions. It takes an action, the world
          answers with reward, the episode ends, and now the question staring
          you in the face is: <em>which knob, if I turned it a hair, would
          make the payoff bigger next time?</em>
        </p>
        <p>
          That question has a name. It&apos;s called a <KeyTerm>policy
          gradient</KeyTerm>. It&apos;s a <em>sensitivity dial</em> — one
          reading per knob, telling you which direction of twist tends to
          pay off, and how strongly. You don&apos;t need the rulebook of the
          world. You don&apos;t need to differentiate through the
          environment. You just need to notice which direction of
          knob-twiddling correlates with more reward, averaged over a pile of
          rollouts. Every policy-based algorithm in reinforcement learning —
          vanilla <NeedsBackground slug="reinforce">REINFORCE</NeedsBackground>,
          A2C, TRPO, PPO, the thing tuning ChatGPT at this very moment — is
          a refinement of that single sensitivity reading.
        </p>
        <p>
          <NeedsBackground slug="markov-decision-processes">MDP</NeedsBackground>{' '}
          gave us the contract: states, actions, rewards. Q-learning answered
          the control problem through a cheat-sheet-shaped back door — learn
          the value of every state-action, then <code>argmax</code>. That
          works until your actions live in <code>ℝⁿ</code> — steering a
          joint to 0.314159 radians, throttling a thrust to 0.42 — and the
          <code>argmax</code> turns into an optimisation problem per step.
          It also fails when the <em>optimal</em> policy is stochastic (mix
          rock/paper/scissors; any deterministic policy gets exploited).
          Value-based RL cannot, by construction, commit to anything other
          than the current best-scoring action.
        </p>
        <p>
          Policy-based RL throws the cheat sheet out and walks up to the
          dashboard. Parameterise <code>π_θ(a | s)</code> as a neural net
          that reads a state and outputs a distribution over actions.
          Do <NeedsBackground slug="gradient-descent">gradient
          ascent</NeedsBackground> on expected return. No value function, no
          <code>argmax</code>, no requirement that the environment be
          differentiable. This lesson derives the one line of calculus that
          makes that possible — the sensitivity dial — stares at it until
          it stops being magic, and then writes it three times in code.
        </p>
      </Prose>

      <Callout variant="insight" title="value-based vs policy-based, one sentence">
        Value-based RL learns <code>Q(s, a)</code> and derives <code>π</code> by{' '}
        <code>argmax</code>; policy-based RL learns <code>π(a | s)</code> directly and
        skips the value function entirely. Both are gradient descent. Only one of them
        works when your actions live in <code>ℝⁿ</code>.
      </Callout>

      {/* ── The policy gradient theorem ──────────────────────────── */}
      <Prose>
        <p>
          The objective is embarrassingly clean: maximise the expected return under the
          policy.
        </p>
      </Prose>

      <MathBlock caption="the objective">
{`J(θ)   =   E_{τ ~ π_θ} [ R(τ) ]

where τ = (s₀, a₀, r₀, s₁, a₁, r₁, …)   is a trajectory
      R(τ) = Σ_t γᵗ rₜ                   is its discounted return`}
      </MathBlock>

      <Prose>
        <p>
          And here is the result that launched the field — the{' '}
          <KeyTerm>policy gradient theorem</KeyTerm>. It says the gradient of this
          expectation has a form you can actually compute:
        </p>
      </Prose>

      <MathBlock caption="the policy gradient theorem (Sutton et al. 2000)">
{`∇_θ J(θ)   =   E_{s, a ~ π_θ} [ ∇_θ log π_θ(a | s) · Q^π(s, a) ]`}
      </MathBlock>

      <Prose>
        <p>
          Stare at this for a second. It&apos;s the sensitivity reading we
          wanted. On the left, the gradient of expected return with respect
          to every knob on the dashboard — the thing that ordinarily
          requires you to differentiate through a sampling distribution and,
          worse, through the environment&apos;s transition dynamics. On the
          right, an expectation of a product of two quantities we can read
          off directly:
        </p>
        <ul>
          <li>
            <code>∇_θ log π_θ(a | s)</code> — which direction on the dashboard
            would make the policy <em>more likely</em> to pick this exact
            action in this exact state. We own the policy network; this is
            one backward pass through a softmax or Gaussian head.
          </li>
          <li>
            <code>Q^π(s, a)</code> — how good the action actually was, the
            expected return from taking <code>a</code> in <code>s</code> and
            continuing under <code>π</code>. We don&apos;t own the
            environment, but we can <em>estimate</em> this from the rewards
            we collected.
          </li>
        </ul>
        <p>
          No derivative of the transition model appears. The environment can
          be a black box, a physics simulator, a real robot, a chatbot with
          a reward model bolted on — it doesn&apos;t matter, because we
          never differentiate through it. The knob sensitivities live
          entirely on our side of the wall.
        </p>
      </Prose>

      {/* ── Widget 1: Policy Gradient Derivation ─────────────────── */}
      <PolicyGradientDerivation />

      <Prose>
        <p>
          Step through the derivation. The move that makes everything work
          is the <KeyTerm>log-derivative trick</KeyTerm>: we can&apos;t push
          <code>∇_θ</code> inside an expectation that depends on{' '}
          <code>θ</code>, because the <em>distribution itself</em> depends on
          every knob on the dashboard. But the identity
        </p>
      </Prose>

      <MathBlock caption="log-derivative identity">
{`∇_θ p_θ(x)   =   p_θ(x) · ∇_θ log p_θ(x)`}
      </MathBlock>

      <Prose>
        <p>
          lets us rewrite <code>∇_θ ∫ p_θ(x) f(x) dx</code> as{' '}
          <code>∫ p_θ(x) · ∇_θ log p_θ(x) · f(x) dx</code>, which is an
          expectation again — one we can estimate by sampling. That&apos;s
          the whole trick. Every score function estimator, every REINFORCE
          variant, every modern policy gradient algorithm is built on that
          one line of calculus. The <code>∇ log π × return</code> reveal is
          the sensitivity dial in its final form: the direction-of-twist
          times how much reward that twist tends to produce.
        </p>
      </Prose>

      <Personify speaker="Log-prob gradient">
        I am the credit carrier. I am exactly one thing — the direction on
        the dashboard that would make <em>this specific action in this
        specific state</em> more likely. I don&apos;t know whether the action
        was good. I don&apos;t know whether the episode succeeded. Someone
        else multiplies me by the return. My only job is to point toward the
        knob-twist that reinforces this choice. Scale me up, the policy
        commits harder. Scale me negative, it backs off. I am the steering
        wheel; the return is the driver.
      </Personify>

      {/* ── REINFORCE ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          The theorem hands us a gradient in terms of <code>Q^π</code> — which
          we don&apos;t have. The simplest workaround: roll out a whole
          episode, measure the return that actually followed each step, and
          use <em>that</em> as a one-sample Monte Carlo estimate of{' '}
          <code>Q^π</code>. This is <KeyTerm>REINFORCE</KeyTerm> (Williams,
          1992):
        </p>
      </Prose>

      <MathBlock caption="REINFORCE — Monte Carlo policy gradient">
{`   Gₜ     =   Σ_{k=t..T} γ^{k-t} r_k               return from step t onward

∇_θ J(θ)  ≈   Σ_t  ∇_θ log π_θ(aₜ | sₜ) · Gₜ         sum over one trajectory

   θ     ←   θ  +  α · ∇_θ J(θ)                    ascent, not descent`}
      </MathBlock>

      <Prose>
        <p>
          Read that middle line slowly, because it&apos;s the reason
          REINFORCE is a <em>specific instance</em> of the general
          sensitivity dial and not its own new idea. The policy gradient
          theorem said: <em>knob-direction times how-good-the-action-was</em>,
          averaged. REINFORCE just picks the cheapest possible
          how-good-the-action-was estimator — the raw return you observed —
          and plugs it in. Swap that estimator and you get every other
          algorithm in the family. Replace <code>G_t</code> with{' '}
          <code>G_t − V(s_t)</code> and you get REINFORCE-with-baseline.
          Replace it with the TD estimate <code>r_t + γV(s_{`{t+1}`})</code>{' '}
          and you&apos;re doing actor-critic. Replace it with a clipped
          importance-weighted advantage and you&apos;re running PPO. Same
          dashboard, same knob-sensitivity reading, different estimator for
          the scalar we multiply it by.
        </p>
        <p>
          Three things to notice about REINFORCE proper. First, it&apos;s an{' '}
          <KeyTerm>unbiased estimator</KeyTerm> of the true gradient — no
          function approximation for <code>Q</code>, just the actual return
          we observed. Second, it has <em>famously high variance</em> — a
          single episode&apos;s return is one draw from a long chain of
          random events, and the same policy can cough up wildly different
          numbers on back-to-back runs. The direction of the sensitivity
          dial is right on average, but its magnitude jitters like a needle
          in a hurricane. Third, it&apos;s embarrassingly simple: one
          forward pass to get <code>log π</code>, one backward pass scaled
          by <code>G</code>, done.
        </p>
      </Prose>

      {/* ── Widget 2: Return to Gradient ─────────────────────────── */}
      <ReturnToGradient />

      <Prose>
        <p>
          Here&apos;s a trajectory laid out step by step. Watch the
          sensitivity reading at each timestep: the log-prob gradient of the
          action we took, scaled by the return that followed. Late actions
          get a small return (little time left to collect reward). Early
          actions get a big return (everything the agent did afterwards
          feeds into their credit). That asymmetry is exactly the{' '}
          <KeyTerm>credit assignment</KeyTerm> problem in RL — and
          REINFORCE&apos;s answer is blunt: every action gets credit for
          every reward that came after it, discounted by how long it had to
          wait.
        </p>
        <p>
          If the final reward was <code>+1</code>, every action in the
          trajectory gets a positive twist-direction, proportional to its
          discounted share. If the final reward was <code>−1</code>, every
          action gets pushed down. This is unsubtle — even actions that
          were genuinely good early in a losing episode get suppressed —
          and it&apos;s why variance reduction matters so much in practice.
          The dial is pointing the right way on average, but any single
          reading shouts where it should whisper.
        </p>
      </Prose>

      <Personify speaker="Return">
        I am the scalar that decides whether to turn the sensitivity reading
        up or down. I come from the environment, not from your network —
        I am a <em>number you measured</em>, not a quantity you
        differentiate. Detach me before you multiply. I am the judge, not
        the witness. Treat me as a target, carry me through the loss as a
        coefficient, and I&apos;ll tell your policy which of its recent
        choices deserve louder voices and which deserve the silent
        treatment.
      </Personify>

      {/* ── Three-layer code ─────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations. A pure-Python REINFORCE on a 3-arm bandit so the moving
          parts are visible. A NumPy version on a tiny CartPole surrogate that shows how
          returns get computed in a loop. A PyTorch version with autograd, an entropy
          bonus, and a value-function baseline — this is what you&apos;d actually write.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · reinforce_bandit.py"
        output={`step   0:  probs=[0.33 0.33 0.33]  pulled=1  reward=0
step  50:  probs=[0.25 0.41 0.34]  pulled=1  reward=1
step 200:  probs=[0.11 0.72 0.17]  pulled=1  reward=1
step 500:  probs=[0.03 0.93 0.04]  pulled=1  reward=1
converged on arm 1 (true best = arm 1 with p=0.8)`}
      >{`import math, random

# 3-arm bandit — only one "state"; each arm pays +1 with its own probability.
true_probs = [0.2, 0.8, 0.5]

# Policy = softmax over 3 logits. θ = the three logits themselves.
theta = [0.0, 0.0, 0.0]
alpha = 0.1

def softmax(logits):
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]

for step in range(501):
    probs = softmax(theta)
    # Sample action from the policy.
    a = random.choices(range(3), weights=probs)[0]
    # Play it, observe the reward.
    r = 1.0 if random.random() < true_probs[a] else 0.0
    # ∇log π(a|·) for softmax:  e_a − probs
    # (indicator of the chosen action, minus the probability vector).
    grad_logp = [-p for p in probs]
    grad_logp[a] += 1.0
    # REINFORCE update: θ ← θ + α · G · ∇log π(a|·).  Here G = r (one-step).
    for i in range(3):
        theta[i] += alpha * r * grad_logp[i]
    if step in (0, 50, 200, 500):
        ps = [f"{p:.2f}" for p in probs]
        print(f"step {step:>3}: probs=[{' '.join(ps)}]  pulled={a}  reward={int(r)}")

print("converged on arm", max(range(3), key=lambda i: theta[i]),
      "(true best = arm 1 with p=0.8)")`}</CodeBlock>

      <Prose>
        <p>
          One state, one step, a <code>softmax</code>, a hand-rolled gradient. Three
          knobs on the dashboard, one sensitivity reading per step, one twist per
          sample. Scale this up: instead of three logits, a neural net. Instead of
          one step, a whole episode whose return we have to accumulate backward
          through time. NumPy.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · reinforce_cartpole.py"
        output={`ep   0:  steps= 14  return=14.00
ep  50:  steps= 42  return=42.00
ep 150:  steps= 98  return=98.00
ep 300:  steps=196  return=196.00
ep 500:  steps=200  return=200.00   (solved)`}
      >{`import numpy as np
import gym  # or gymnasium

env = gym.make("CartPole-v1")
rng = np.random.default_rng(0)

# Simple 2-layer policy: obs(4) → 16 → 2 (softmax).
W1 = rng.standard_normal((4, 16)) * 0.1
W2 = rng.standard_normal((16, 2)) * 0.1
lr = 1e-2
gamma = 0.99

def forward(obs):
    h = np.tanh(obs @ W1)
    logits = h @ W2
    logits -= logits.max()
    p = np.exp(logits); p /= p.sum()
    return p, h

def compute_returns(rewards):
    """G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …   backward recurrence."""
    G = np.zeros_like(rewards, dtype=float)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    # Normalise within batch — variance reduction, see callout below.
    G = (G - G.mean()) / (G.std() + 1e-8)
    return G

for ep in range(501):
    obs, _ = env.reset()
    obss, acts, rewards, hiddens = [], [], [], []
    done = False
    while not done:
        p, h = forward(obs)
        a = rng.choice(2, p=p)
        obs2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        obss.append(obs); acts.append(a); rewards.append(r); hiddens.append(h)
        obs = obs2

    # Monte Carlo returns, normalised.
    G = compute_returns(np.array(rewards))

    # Gradient accumulation — one REINFORCE step per trajectory.
    dW1 = np.zeros_like(W1); dW2 = np.zeros_like(W2)
    for t in range(len(obss)):
        p, _ = forward(obss[t])
        dlogit = -p; dlogit[acts[t]] += 1.0           # ∇log π w.r.t. logits
        dlogit *= G[t]                                # scale by return
        dW2 += np.outer(hiddens[t], dlogit)           # ∇w.r.t. W2
        dh   = dlogit @ W2.T * (1 - hiddens[t]**2)    # backprop through tanh
        dW1 += np.outer(obss[t], dh)                  # ∇w.r.t. W1

    # Ascent (note the plus).
    W1 += lr * dW1
    W2 += lr * dW2

    if ep in (0, 50, 150, 300, 500):
        print(f"ep {ep:>3}: steps={len(rewards):>3}  return={sum(rewards):.2f}"
              + ("   (solved)" if sum(rewards) >= 195 else ""))`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'scalar reward r on one step',
            right: 'compute_returns(rewards): backward γ-recurrence',
            note: 'multi-step trajectories need the whole return, not just r_t',
          },
          {
            left: 'hand-written softmax over 3 logits',
            right: 'forward(obs): np @ obs @ W1 → tanh → W2 → softmax',
            note: 'policy becomes a 2-layer net; same log-prob gradient shape',
          },
          {
            left: 'θ ← θ + α · r · ∇log π',
            right: 'dW1, dW2 accumulated over the trajectory, scaled by G_t',
            note: 'one update per episode, summed across timesteps',
          },
        ]}
      />

      <Prose>
        <p>
          Now PyTorch. Autograd does the backprop for us, we add a{' '}
          <KeyTerm>value-function baseline</KeyTerm> to cut variance, and we tack on an{' '}
          <KeyTerm>entropy bonus</KeyTerm> to keep the policy from collapsing too fast.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · reinforce_pytorch.py"
        output={`ep   0:  return= 21.0   loss=  1.07   ent=0.69
ep 100:  return= 48.2   loss=  0.41   ent=0.62
ep 300:  return=173.4   loss= -0.11   ent=0.39
ep 500:  return=198.7   loss= -0.08   ent=0.22   (solved)`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

env = gym.make("CartPole-v1")
gamma = 0.99

class ActorCritic(nn.Module):
    """Shared trunk → policy head (logits) and value head (scalar baseline)."""
    def __init__(self):
        super().__init__()
        self.trunk  = nn.Sequential(nn.Linear(4, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh())
        self.policy = nn.Linear(64, 2)
        self.value  = nn.Linear(64, 1)
    def forward(self, obs):
        h = self.trunk(obs)
        return self.policy(h), self.value(h).squeeze(-1)

net = ActorCritic()
opt = torch.optim.Adam(net.parameters(), lr=3e-3)

for ep in range(501):
    obs, _ = env.reset()
    log_probs, values, rewards, entropies = [], [], [], []
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        logits, v = net(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        obs, r, term, trunc, _ = env.step(a.item())
        done = term or trunc
        log_probs.append(dist.log_prob(a))        # ∇log π is handled by autograd
        values.append(v)                          # baseline for variance reduction
        entropies.append(dist.entropy())          # bonus: encourage exploration
        rewards.append(r)

    # Monte Carlo returns (targets — no grad flows through these).
    G, running = [], 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        G.insert(0, running)
    G = torch.tensor(G, dtype=torch.float32)
    G = (G - G.mean()) / (G.std() + 1e-8)          # per-batch normalisation

    log_probs = torch.stack(log_probs)
    values    = torch.stack(values)
    entropies = torch.stack(entropies)

    # Advantage = return − baseline.  Detach G; only V's own loss trains V.
    advantage = G - values.detach()
    policy_loss = -(log_probs * advantage).mean()  # minus, because Adam minimises
    value_loss  = F.mse_loss(values, G)            # critic regresses toward G
    entropy_bonus = entropies.mean()
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

    opt.zero_grad(); loss.backward(); opt.step()

    if ep in (0, 100, 300, 500):
        print(f"ep {ep:>3}: return={sum(rewards):>6.1f}   "
              f"loss={loss.item():>6.2f}   ent={entropy_bonus.item():.2f}"
              + ("   (solved)" if sum(rewards) >= 195 else ""))`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'manual ∇log π: dlogit = -p; dlogit[a] += 1',
            right: 'dist.log_prob(a)  +  loss.backward()',
            note: 'autograd traces the log-prob through the softmax',
          },
          {
            left: 'scale grads by G_t, accumulate dW1/dW2',
            right: '-(log_probs * advantage).mean() then opt.step()',
            note: 'negate because PyTorch optimisers minimise; we want ascent',
          },
          {
            left: 'no baseline — raw return',
            right: 'advantage = G − values.detach()',
            note: 'critic cuts variance; detach so G is a pure target',
          },
          {
            left: 'no exploration incentive',
            right: '- 0.01 * entropies.mean()',
            note: 'small bonus keeps the policy from collapsing prematurely',
          },
        ]}
      />

      <Callout variant="insight" title="the three layers, summarised">
        Pure Python shows REINFORCE on a bandit so the update rule has no hiding place:
        one line is the whole algorithm. NumPy scales it to a real environment and forces
        you to implement discounted returns, softmax backprop, and trajectory
        accumulation by hand. PyTorch throws all of that away — autograd handles it — and
        frees you to think about the <em>real</em> questions: baseline, entropy,
        advantage estimation. Same algorithm, three phases of your understanding.
      </Callout>

      {/* ── Why this generalizes REINFORCE ───────────────────────── */}
      <Callout variant="note" title="why this frame generalizes REINFORCE — and everything after">
        Every policy-gradient algorithm is the same sensitivity dial with a
        different estimator for &ldquo;how good was that action.&rdquo; Raw
        return <code>G_t</code> gives you REINFORCE. Return minus a learned
        baseline <code>G_t − V(s_t)</code> gives you REINFORCE-with-baseline
        — same dial, lower variance. One-step TD{' '}
        <code>r_t + γV(s_{`{t+1}`}) − V(s_t)</code> gives you advantage
        actor-critic. Clipped importance-weighted advantage gives you PPO.
        KL-regularised version with a reward model gives you RLHF. They
        aren&apos;t different ideas — they&apos;re different ways of
        estimating the scalar that multiplies{' '}
        <code>∇ log π</code>. Once you see the dial, the rest of the family
        tree is a list of variance-reduction patches.
      </Callout>

      {/* ── Variance-reduction and baselines ─────────────────────── */}
      <Callout variant="note" title="why subtracting a baseline is free">
        The policy gradient is{' '}
        <code>E[∇log π(a|s) · Q(s, a)]</code>. Replace <code>Q</code> with{' '}
        <code>Q − b(s)</code> for any function <code>b(s)</code> that depends only on the
        state and the expectation is unchanged, because{' '}
        <code>E[∇log π(a|s) · b(s)] = b(s) · E[∇log π(a|s)] = b(s) · 0 = 0</code> (the
        expected score function is zero). But the <em>variance</em> drops a lot if{' '}
        <code>b(s) ≈ E_a[Q(s, a)] = V(s)</code>. Using <code>V(s)</code> is how you get
        actor-critic: the critic <em>is</em> your baseline.
      </Callout>

      <Callout variant="note" title="entropy bonus — the exploration subsidy">
        Without help, a policy gradient converges to something deterministic fast, often
        on the wrong action. Adding <code>+ β · H(π(·|s))</code> to the objective
        (equivalently, <code>− β · entropy</code> to the loss) taxes certainty: the
        policy only commits when it&apos;s really sure. Typical <code>β</code> is{' '}
        <code>0.001–0.01</code>. Too low: premature collapse. Too high: random policy,
        never learns.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Forgetting to detach returns.</strong> The
          return <code>G_t</code> is a <em>target</em>, not a differentiable quantity —
          you computed it from rewards the environment gave you. If you leave it attached
          to the computation graph (easy to do when <code>G = something_involving_V</code>),
          gradients flow back through it in directions you didn&apos;t plan. Always{' '}
          <code>.detach()</code> the thing that multiplies the log-prob.
        </p>
        <p>
          <strong className="text-term-amber">Normalising returns across episodes
          vs within a batch.</strong> Per-batch normalisation{' '}
          <code>(G − mean) / std</code> helps a lot — it absorbs drift in the scale of
          returns as the policy improves. But normalising across <em>all history</em>{' '}
          destroys signal (good returns stop looking good once everything is good). Do
          it per update step.
        </p>
        <p>
          <strong className="text-term-amber">Entropy coefficient too high.</strong> Set{' '}
          <code>β = 0.1</code> by accident and your policy will refuse to commit to
          anything — you&apos;ll be training a glorified uniform distribution for the
          entire run. Watch the entropy curve: it should <em>decrease</em> over training,
          just not all the way to zero.
        </p>
        <p>
          <strong className="text-term-amber">Using V(s) as a baseline without
          detaching.</strong> The actor&apos;s gradient uses{' '}
          <code>advantage = G − V(s)</code>. If you leave <code>V</code> attached, the
          actor&apos;s loss will also try to push <code>V</code> around — usually in the
          wrong direction. Detach <code>V</code> in the policy loss; let the value loss
          train it separately.
        </p>
        <p>
          <strong className="text-term-amber">Treating loss sign carelessly.</strong> The
          theorem gives us a gradient we want to <em>ascend</em>. PyTorch optimisers
          descend. Negate the policy loss. Forgetting this once means your agent actively
          learns to do worse — it&apos;s an almost-silent bug because the numbers look
          plausible. Your sensitivity dial is still pointing at the right knobs;
          you&apos;re just turning them the wrong direction.
        </p>
      </Gotcha>

      {/* ── Challenge ────────────────────────────────────────────── */}
      <Challenge prompt="Train REINFORCE on CartPole with and without a baseline">
        <p>
          Start with the PyTorch snippet above, but strip out the value head and the
          value loss so it&apos;s vanilla REINFORCE — the policy gradient scaled by the
          raw normalised return <code>G</code>. Train for 500 episodes on CartPole-v1 and
          plot the return per episode.
        </p>
        <p className="mt-2">
          Now add the value-function baseline back in. Same network, same hyperparams,
          same seed. Plot the return curve on the same axes. You should see two things:
          the baseline version reaches &ldquo;solved&rdquo; (200 steps) in fewer
          episodes, and the <em>variance</em> of the return curve is visibly lower. That
          gap is the baseline earning its keep — a steadier dial reading per update.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: run both variants across 10 seeds and plot the mean ± std. The
          baseline&apos;s contribution is mostly about variance reduction across seeds,
          not raw final performance — modern actor-critic methods exist because of this
          plot.
        </p>
      </Challenge>

      {/* ── Closing + teaser ─────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Policy gradients are a
          sensitivity dial. For every knob on the policy network,{' '}
          <code>∇ log π × return</code> tells you which direction of twist
          tends to produce more reward — averaged, unbiased, estimable from
          trajectories alone, no environment derivative required. Baselines
          and entropy bonuses are not optional sophistication; they are the
          difference between a method that works and a method that
          theoretically should. Everything downstream in this section —
          REINFORCE as a first-class algorithm, actor-critic, PPO — is this
          same dial, paired with a better way of estimating the scalar that
          scales it.
        </p>
        <p>
          <strong>Next up — REINFORCE.</strong> We&apos;ve derived the
          estimator and glued a demo version together here for pedagogy. The
          next lesson zooms all the way in on REINFORCE as a first-class
          algorithm: the training loop, batching across episodes, when to
          reset, how to read its jagged casino-tracker of a learning curve,
          and the specific hyperparameter traps that make it look broken
          when it isn&apos;t. From there it&apos;s a short step to
          actor-critic, PPO, and the algorithms actually running production
          RL today.
        </p>
      </Prose>

      <References
        items={[
          {
            title:
              'Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning',
            author: 'R. J. Williams',
            venue: 'Machine Learning, 1992 — the REINFORCE paper',
            url: 'https://link.springer.com/article/10.1007/BF00992696',
          },
          {
            title:
              'Policy Gradient Methods for Reinforcement Learning with Function Approximation',
            author: 'Sutton, McAllester, Singh, Mansour',
            venue: 'NeurIPS 2000 — the policy gradient theorem',
            url: 'https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html',
          },
          {
            title: 'Reinforcement Learning: An Introduction (2nd ed.), Ch. 13',
            author: 'Sutton & Barto',
            year: 2018,
            url: 'http://incompleteideas.net/book/the-book-2nd.html',
          },
          {
            title: 'Asynchronous Methods for Deep Reinforcement Learning (A3C)',
            author: 'Mnih et al.',
            venue: 'ICML 2016 — entropy bonus as an actor-critic regulariser',
            url: 'https://arxiv.org/abs/1602.01783',
          },
          {
            title: 'High-Dimensional Continuous Control Using Generalized Advantage Estimation',
            author: 'Schulman, Moritz, Levine, Jordan, Abbeel',
            venue: 'ICLR 2016 — modern baseline / advantage estimators',
            url: 'https://arxiv.org/abs/1506.02438',
          },
        ]}
      />
    </div>
  )
}
