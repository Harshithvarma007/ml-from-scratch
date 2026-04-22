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
import REINFORCETraining from '../widgets/REINFORCETraining'
import VarianceReductionComparison from '../widgets/VarianceReductionComparison'

// Signature anchor: REINFORCE as the casino tracker. A patron plays a full
// night at the slot machine, tallies the total payout at the end, then tells
// every arm they pulled: "pulling you paid off tonight, do more of that" (if
// the tally was positive) or "don't pull you next time" (if negative). Monte
// Carlo credit assignment — wait till the whole episode finishes, then spread
// the return across every action taken along the way. Returned to at the
// opening, the log-prob × return reveal, and the "why the variance is so
// brutal" section.
export default function ReinforceLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="reinforce" />

      {/* ── Opening: the casino tracker ──────────────────────────── */}
      <Prose>
        <p>
          Picture a patron at a casino. Not the movie version — no tuxedo, no
          James Bond. Just someone on a stool in front of a slot machine with
          a handful of arms, each one labeled with a different action. They
          pull an arm. Lights flash. The machine nudges them into a new
          state and coughs up (or swallows) some chips. They pull another.
          And another. They don&apos;t know which arm is best. They don&apos;t
          even know if they&apos;re playing the right kind of slot. They just
          pull, watch the chips move, and keep notes.
        </p>
        <p>
          At the end of the night they close the tab, tally the whole evening
          into one number — total payout, good or bad — and turn around to
          the row of arms they pulled: <em>pulling you paid off tonight, do
          more of that tomorrow. You over there, the one I pulled in the
          third round, same goes for you. Don&apos;t pull this one — that
          whole night was a loss.</em> No coach at their shoulder telling
          them which arm was right at each moment. No instant replay. Just a
          whole-night tally, and a rule for redistributing that tally back to
          every arm they touched.
        </p>
        <p>
          That&apos;s <KeyTerm>REINFORCE</KeyTerm>. It&apos;s the simplest
          thing that can possibly work in reinforcement learning. Run a
          policy, play a full episode, wait till the slot machine stops
          spinning, tally the return, and reinforce every action in
          proportion to how the night went. No critic, no target network, no
          replay buffer. Thirty lines of code. The patron walks out knowing
          slightly better slots to pull tomorrow.
        </p>
        <p>
          You have a policy <code>π_θ(a|s)</code> — a neural network that
          takes a state and spits out a distribution over actions (which arm
          to pull). You want to tune <code>θ</code> so the policy racks up
          more reward. The supervised move would be to ask: what was the{' '}
          <em>right</em> arm at each step? Run backprop on that. But in an{' '}
          <NeedsBackground slug="markov-decision-processes">MDP</NeedsBackground>{' '}
          nobody tells you the right arm. You only find out — later, often
          much later — whether the whole night paid off.
        </p>
        <p>
          REINFORCE is also the conceptual parent of every policy-gradient
          method you&apos;ve ever heard of. PPO, TRPO, A2C, RLHF —
          they&apos;re all REINFORCE with better variance control or more
          stability. If you get the casino tracker in your bones, the rest
          of the family is a list of patches.
        </p>
      </Prose>

      <Personify speaker="REINFORCE">
        I&apos;m the patron at the slot machine. Sample a trajectory, wait
        till the episode&apos;s over, tally the payout, and weight each
        arm&apos;s log-probability by the return that followed. My gradients
        are unbiased and obscenely noisy. Everyone who came after me exists
        to reduce my variance.
      </Personify>

      {/* ── The full algorithm as math ──────────────────────────── */}
      <Prose>
        <p>
          Here&apos;s the whole algorithm on a single page. Everything else
          in this lesson is commentary on these three lines.
        </p>
      </Prose>

      <MathBlock caption="REINFORCE — the full update, one episode at a time">
{`for each episode:

    1.  τ   =  (s₀, a₀, r₁, s₁, a₁, r₂, ... , s_{T-1}, a_{T-1}, r_T)
            ← sampled by running π_θ in the environment
               (the patron playing the slot machine for a full night)

    2.  G_t =  Σ_{k=t}^{T-1}  γ^{k-t} · r_{k+1}       ← return from step t onward
                                                        (the tally for arm-pull t
                                                         and every pull after it)

    3.  θ   ←  θ  +  α · Σ_{t=0}^{T-1}  ∇_θ log π_θ(a_t | s_t) · G_t
                                                     (do more of the arms whose
                                                      tally came out positive;
                                                      do less of the others)`}
      </MathBlock>

      <Prose>
        <p>
          Read that bottom line slowly, because it&apos;s the whole lesson.
          It&apos;s the <KeyTerm>policy gradient theorem</KeyTerm> turned into
          an update. For every <code>(s_t, a_t)</code> pair in the trajectory
          — every arm the patron pulled, and the state of the slot machine
          when they pulled it — you compute the gradient of{' '}
          <code>log π_θ(a_t | s_t)</code>, the &ldquo;if I wiggled θ, how
          would the log-probability of this arm-pull change?&rdquo; score.
          Then you scale it by <code>G_t</code>, the tally of what actually
          happened after that pull. Sum across the trajectory. That&apos;s
          your gradient. It&apos;s{' '}
          <NeedsBackground slug="gradient-descent">gradient ascent</NeedsBackground>{' '}
          — we&apos;re climbing expected return, not descending a loss — so
          the update adds rather than subtracts.
        </p>
        <p>
          The reveal: if pulling an arm led to a positive tally,{' '}
          <code>G_t</code> is big and positive, so the update shoves{' '}
          <code>log π_θ(a_t|s_t)</code> up — the policy becomes more likely
          to pull that arm next time in that state. If the tally was tiny or
          negative, the update pushes it down. The casino never has to tell
          the patron which arm was the right one. The payout at the end of
          the night tells them how hard to reinforce the arm they happened
          to choose. That&apos;s the whole game. That&apos;s why this thing
          works at all.
        </p>
      </Prose>

      <Callout variant="note" title="why log π, not π">
        The <code>log</code> in <code>∇log π_θ(a|s)</code> is not a choice —
        it&apos;s what falls out of the policy gradient derivation (the
        &ldquo;log-derivative trick&rdquo;: <code> ∇π / π = ∇log π</code>).
        Practically, it gives you a gradient that&apos;s well-scaled
        regardless of how confident the policy currently is. A 1%
        probability arm-pull can still move — its log-prob is just very
        negative, and the{' '}
        <NeedsBackground slug="softmax">softmax</NeedsBackground> over
        logits absorbs the scale.
      </Callout>

      <Callout variant="insight" title="what &ldquo;Monte Carlo&rdquo; means here">
        REINFORCE is a <strong>Monte Carlo</strong> method — the fancy name
        for &ldquo;wait till the whole episode is over, then look at what
        actually happened.&rdquo; No bootstrapping, no value estimate
        substituted in for the tail of the trajectory. The tally is the real
        tally. That&apos;s what makes the estimator unbiased: you&apos;re
        multiplying each log-prob by the literal return that followed, not a
        guess at the return. It&apos;s also what makes the variance
        ruinous, which we&apos;re about to stare at.
      </Callout>

      {/* ── Widget 1: Live CartPole training ────────────────────── */}
      <Prose>
        <p>
          Watch it learn. The widget below runs REINFORCE on CartPole in
          your browser. Episode return (the length of time the pole stays
          up, capped at 500) is plotted against training episodes. A random
          policy — the patron pulling arms at random — lasts about 20 steps.
          A solved policy lasts 500. In between is a lot of noise, because
          in between is a lot of nights where the tally swung for reasons
          that had very little to do with the arm on any given pull.
        </p>
      </Prose>

      <REINFORCETraining />

      <Prose>
        <p>
          Three things to notice. First, it <em>does</em> learn — episode
          return climbs from ~20 toward 500 over a few hundred episodes.
          The casino tracker is tuning itself. Second, the curve is brutally
          jagged. Two episodes in a row with nearly identical policies can
          return 50 and 400; REINFORCE just eats the variance. Third, the
          slope is impatient in the middle and flattens near the ceiling —
          classic RL learning dynamics. The cheap wins come first, the last
          few percent come slowly.
        </p>
      </Prose>

      <Callout variant="insight" title="REINFORCE loss ≈ weighted cross-entropy">
        Squint at the loss and you&apos;ll see something familiar:{' '}
        <code>−log π_θ(a|s) · G</code>. That&apos;s exactly supervised
        cross-entropy loss on the action <code>a</code>, weighted by the
        tally <code>G</code>. REINFORCE is supervised learning where the
        &ldquo;labels&rdquo; are the arms you actually pulled and the
        &ldquo;loss weights&rdquo; are the payouts they produced. Same
        backprop, different data source. The environment hands you the
        labels and the weights in the same breath.
      </Callout>

      {/* ── The variance problem ────────────────────────────────── */}
      <Prose>
        <p>
          So REINFORCE works. Why does anyone use anything else? Because the
          gradient estimator is <em>unbiased</em> but has{' '}
          <em>catastrophic variance</em>. This is the part of the casino
          story that bites. One full night at the slot machine is one sample
          from a massive distribution over possible nights. Your policy is
          stochastic — it has to be, for exploration — and the slot machine
          is stochastic, and the patron played for a few hundred pulls, and
          the tally at the end is one draw from the joint distribution of
          all of it. The same patron with the same policy on the same slot
          machine can walk out up 400 chips on Monday and down 50 on
          Tuesday. REINFORCE just eats that.
        </p>
        <p>
          It gets worse. A night with a great tally might have one brilliant
          arm-pull followed by a string of lucky random ones that would
          have failed on any other night. REINFORCE doesn&apos;t know which
          pull was the good one — it reinforces <em>every</em> arm pulled
          that night in proportion to the final tally. The signal is in
          there, but it&apos;s buried under &ldquo;the patron got lucky
          seven times in a row on arms that usually lose.&rdquo; You need
          many, many nights to average the luck out. That&apos;s the
          variance problem, and it&apos;s the entire reason the rest of
          policy-gradient research exists.
        </p>
      </Prose>

      <MathBlock caption="the three standard variance-reduction moves">
{`  (1)  baseline b(s_t):

       θ ← θ + α · Σ_t  ∇log π_θ(a_t|s_t) · ( G_t − b(s_t) )

       subtracting b(s) doesn't change the expectation (∇log π sums to zero
       in expectation) but shrinks variance dramatically. Usually b(s) = V_φ(s),
       a learned value function trained by regression on returns.


  (2)  normalized returns:

       Ĝ_t = ( G_t − mean(G) ) / ( std(G) + ε )

       z-score the returns within a batch. Cheap, no extra network.


  (3)  reward-to-go (already baked into G_t above):

       use ONLY the rewards after time t, not the whole trajectory's return.
       arm-pulls at step t have no causal effect on payouts BEFORE step t,
       so including them is just noise.`}
      </MathBlock>

      <Prose>
        <p>
          The baseline is the big one. Replace the raw tally{' '}
          <code>G_t</code> with <code>G_t − b(s_t)</code> and you&apos;re now
          reinforcing an arm-pull based on how much better it did{' '}
          <em>than expected at that state</em>. If <code>b(s)</code> is the
          value function <code>V(s)</code>, the thing you&apos;re multiplying
          the log-prob by is the <KeyTerm>advantage</KeyTerm>{' '}
          <code>A(s, a) = Q(s, a) − V(s)</code> — literally &ldquo;how much
          better is this arm than the average arm at this state on this
          slot machine.&rdquo; That&apos;s the conceptual leap to
          actor-critic and to PPO.
        </p>
      </Prose>

      {/* ── Widget 2: Variance reduction comparison ─────────────── */}
      <Prose>
        <p>
          Three training curves on CartPole, same seed: vanilla REINFORCE,
          REINFORCE with a learned baseline, and REINFORCE with a baseline
          plus normalized returns. Same algorithm at the core — but the
          second and third curves are meaningfully smoother and converge
          faster.
        </p>
      </Prose>

      <VarianceReductionComparison />

      <Prose>
        <p>
          The gap isn&apos;t about the final score — all three can solve
          CartPole eventually. The gap is about how many episodes it takes
          and how confidently you can trust the learning curve. With a
          baseline, the gradient variance drops and the policy updates in a
          straighter line toward the optimum. Stack both tricks and you get
          the practical version of REINFORCE that shows up in real
          codebases, usually as a baseline comparison for PPO.
        </p>
      </Prose>

      <Personify speaker="Reward-to-go">
        I am the edit that stopped paying out arm-pulls for things they
        couldn&apos;t possibly have caused. The reward you got on pull 3?
        Arm 7, pulled on pull 7, had nothing to do with it. Cross it out.
        You&apos;d think this is obvious, but early policy-gradient code
        routinely multiplied log-probs by the full-trajectory tally. I cut
        the variance in half for free.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, as always. Pure Python to see the casino tracker&apos;s
          skeleton — run an episode, compute returns, nudge θ. NumPy to
          vectorise the returns across a batch of episodes. PyTorch to hand
          the gradient work to autograd via{' '}
          <code>Categorical.log_prob</code>. Same algorithm, three idioms,
          each shorter than the last.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · reinforce_scratch.py"
        output={`episode 0   return=17
episode 50  return=43
episode 100 return=78
episode 150 return=142
...`}
      >{`import random, math

# Assume a tiny environment and a policy π_θ parameterised by θ.
# For each state s, π(a|s) is e.g. a softmax over logits = θ @ features(s).

def run_episode(env, policy):
    traj = []
    s = env.reset()
    while True:
        probs  = policy.probs(s)                       # π(·|s) as a list
        a      = random.choices(range(len(probs)), weights=probs)[0]
        s2, r, done = env.step(a)
        traj.append((s, a, r))
        if done: return traj
        s = s2

def returns_from(traj, gamma=0.99):
    G, out = 0.0, []
    for (_, _, r) in reversed(traj):                   # reward-to-go, backwards
        G = r + gamma * G
        out.append(G)
    return list(reversed(out))

def reinforce_step(policy, traj, alpha=1e-2, gamma=0.99):
    Gs = returns_from(traj, gamma)
    for (s, a, _), G in zip(traj, Gs):
        grad_logp = policy.grad_log_prob(s, a)         # ∇_θ log π(a|s)
        policy.theta += alpha * G * grad_logp          # θ ← θ + α·G·∇log π`}</CodeBlock>

      <Prose>
        <p>
          Thirty lines of arithmetic. No replay buffer, no target network, no
          importance weights, no clipping. The patron plays the night,
          tallies the payout backwards from the end (reward-to-go), and
          nudges θ once per arm-pull. Now vectorise — run <code>N</code>{' '}
          episodes in parallel, pack them into arrays, do the tallies in one
          sweep.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — numpy · reinforce_numpy.py">{`import numpy as np

def compute_returns(rewards, gamma=0.99):
    """rewards: shape (T,)  →  returns: shape (T,)   reward-to-go."""
    T = len(rewards)
    G = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        G[t] = running
    return G

def normalize(x, eps=1e-8):
    return (x - x.mean()) / (x.std() + eps)            # variance-reduction trick

def reinforce_update(grad_log_probs, returns, baseline, alpha=1e-2):
    """
    grad_log_probs : (T, D)   — stacked ∇log π for each step
    returns        : (T,)     — G_t
    baseline       : (T,)     — V(s_t), or zeros for vanilla REINFORCE
    """
    advantages = normalize(returns - baseline)         # (G_t − b(s_t)), z-scored
    grad = (advantages[:, None] * grad_log_probs).sum(axis=0)
    return alpha * grad                                # Δθ`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for (s,a,_),G in zip(traj,Gs): θ += α·G·∇log π',
            right: 'Δθ = α · (advantages[:,None] * grad_log_probs).sum(0)',
            note: 'sum over timesteps becomes one matrix reduction',
          },
          {
            left: 'returns computed in a Python for-loop',
            right: 'reverse-scan once, writing into a preallocated array',
            note: 'same O(T), but no per-step Python overhead',
          },
          {
            left: 'raw G_t',
            right: '(G − b) / std(G)',
            note: 'subtract baseline, z-score — two one-liners of variance reduction',
          },
        ]}
      />

      <Prose>
        <p>
          Now PyTorch. The gradient bookkeeping disappears — you just form
          the loss and call <code>.backward()</code>. The critical idiom is{' '}
          <code>torch.distributions.Categorical</code> (or{' '}
          <code>Normal</code> for continuous actions): you sample arm-pulls
          from it at rollout time, and at update time you ask it for{' '}
          <code>log_prob(a)</code>, which is a differentiable tensor
          autograd can trace through.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · reinforce_pytorch.py"
        output={`ep 0    return= 22.0
ep 100  return= 58.4   (avg-last-10)
ep 250  return=197.8
ep 500  return=498.6   solved`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, n_actions))
    def forward(self, s):
        return Categorical(logits=self.net(s))          # π(·|s)

class Value(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 1))
    def forward(self, s):
        return self.net(s).squeeze(-1)                  # V_φ(s)

def train_step(policy, value, opt_p, opt_v, states, actions, returns):
    # --- policy update ----------------------------------------
    dist     = policy(states)
    log_prob = dist.log_prob(actions)                   # differentiable
    baseline = value(states).detach()                   # DETACH: no grad into V here
    advantage = returns - baseline
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    loss_pi = -(log_prob * advantage).mean()            # minus: we're ascending
    opt_p.zero_grad(); loss_pi.backward(); opt_p.step()

    # --- value update (regression on returns) ------------------
    loss_v = F.mse_loss(value(states), returns)
    opt_v.zero_grad(); loss_v.backward(); opt_v.step()`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'policy.grad_log_prob(s, a)  — hand-rolled',
            right: 'Categorical(logits=net(s)).log_prob(a)',
            note: 'autograd traces the ∇log π for you',
          },
          {
            left: 'Δθ = α · Σ_t advantage_t · ∇log π_t',
            right: 'loss = -(log_prob * advantage).mean(); loss.backward()',
            note: 'minus sign: PyTorch minimizes, policy gradient ascends',
          },
          {
            left: 'baseline = V_φ(s) as a numpy array',
            right: 'value(states).detach()',
            note: 'detach so the policy loss doesn\u2019t flow gradients into V',
          },
        ]}
      />

      <Callout variant="insight" title="the point of the three layers">
        Pure Python shows you the algorithm is actually small. NumPy shows
        you where the vectorization hides. PyTorch shows you how little
        code you need in real life — eight lines of real logic inside{' '}
        <code>train_step</code>, because autograd and{' '}
        <code>Categorical</code> do the heavy lifting. Same casino tracker,
        three levels of ceremony.
      </Callout>

      {/* ── Advantage / PPO lineage callout ─────────────────────── */}
      <Callout variant="note" title="REINFORCE → PPO, in one paragraph">
        Once you&apos;re subtracting a learned baseline, you&apos;re
        computing an advantage{' '}
        <code>A(s, a) ≈ G_t − V_φ(s_t)</code>. That&apos;s exactly what
        actor-critic methods use. PPO takes it one step further: instead of
        just one gradient step per batch of trajectories, it runs multiple
        gradient steps but clips the ratio{' '}
        <code>π_new(a|s)/π_old(a|s)</code> to stop the policy from
        wandering off. The gradient being clipped is, underneath, the
        REINFORCE-with-baseline gradient. RLHF — which you met in the
        fine-tuning lesson — is PPO applied to language models with a
        reward model standing in for the environment. You already know the
        math. The rest is engineering.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Off-policy bug via stale log-probs:</strong>{' '}
          the trajectory must be sampled from the <em>current</em> policy,
          and the <code>log_prob</code> you use in the loss must come from
          the <em>current</em> policy as well. If you cache log-probs from
          rollout and then update θ multiple times, you&apos;re using stale
          log-probs and the gradient is biased. This is exactly the bug PPO
          solves with its importance ratio; vanilla REINFORCE just does one
          update per rollout.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting to detach the baseline:</strong>{' '}
          in the policy loss, <code>advantage = returns − V_φ(s)</code>. If
          you don&apos;t <code>.detach()</code> the baseline, gradients
          from the policy loss flow into <code>V_φ</code> and confuse its
          objective (which is MSE against returns, not policy improvement).
          Detach.
        </p>
        <p>
          <strong className="text-term-amber">Sign errors:</strong> policy
          gradient is gradient <em>ascent</em> (maximize return). PyTorch
          optimisers do gradient <em>descent</em>. Your loss must be{' '}
          <code>−(log_prob * advantage).mean()</code> — the minus sign is
          not optional. Half the &ldquo;why isn&apos;t my agent
          learning&rdquo; bugs are this.
        </p>
        <p>
          <strong className="text-term-amber">Return computation direction:</strong>{' '}
          returns are cumulative from the end backwards —{' '}
          <code>G_t = r_{`{t+1}`} + γ·G_{`{t+1}`}</code>. The patron tallies
          the night by walking backwards from the last pull. If you
          accidentally sum forward or forget to reset at episode boundaries,
          the gradients are nonsense.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Measure the baseline's worth">
        <p>
          Run two agents on CartPole with the same seed and hyperparameters:
          (a) vanilla REINFORCE, (b) REINFORCE with a learned value
          baseline. Define &ldquo;solved&rdquo; as average return ≥ 475 over
          the last 100 episodes.
        </p>
        <p className="mt-2">
          Record <em>episodes-to-solve</em> for each. Plot the two learning
          curves on the same axes. The baseline version should solve it in
          roughly half as many episodes, with visibly less jitter along the
          way.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: add normalized returns to (b) and measure again. Then
          disable reward-to-go (use the full-trajectory tally for every
          timestep) in (a) and watch training get dramatically worse —
          that&apos;s how much free variance reduction you were getting.
        </p>
      </Challenge>

      {/* ── Closing / teaser ────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> REINFORCE is the base case
          of every policy-gradient method: play an episode at the slot
          machine, tally the payout, weight log-probabilities by returns,
          step. Its unbiased gradient comes with ruinous variance, which is
          why every modern variant adds a baseline, an advantage estimator,
          and sometimes a trust region. Understanding REINFORCE + baseline
          is most of what you need to understand PPO, and PPO is most of
          what you need to understand RLHF.
        </p>
        <p>
          <strong>Next up — Actor-Critic.</strong> The casino tracker has
          one glaring problem: it has to wait until the end of the night to
          learn anything. What if the patron could lean over to a seasoned
          regular at the next stool and ask, mid-pull, <em>&ldquo;is this
          state usually a good spot?&rdquo;</em> That regular is the{' '}
          <em>critic</em> — a value function <code>V_φ(s)</code> trained
          alongside the policy, quietly estimating how good each state is
          so the policy doesn&apos;t have to finish the whole episode to get
          feedback. You get tighter advantage estimates, bootstrapped
          returns, and the door opens to A2C, A3C, and PPO at scale. The
          critic is what turns the casino tracker into something that can
          actually ship.
        </p>
      </Prose>

      <References
        items={[
          {
            title:
              'Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning',
            author: 'Ronald J. Williams',
            venue: 'Machine Learning, 1992 — the REINFORCE paper',
            url: 'https://link.springer.com/article/10.1007/BF00992696',
          },
          {
            title: 'Reinforcement Learning: An Introduction (2nd ed.), Chapter 13',
            author: 'Sutton & Barto',
            venue: 'Policy Gradient Methods',
            url: 'http://incompleteideas.net/book/RLbook2020.pdf',
          },
          {
            title: 'Spinning Up — Vanilla Policy Gradient',
            author: 'OpenAI / Josh Achiam',
            venue: 'the canonical pedagogical implementation',
            url: 'https://spinningup.openai.com/en/latest/algorithms/vpg.html',
          },
          {
            title: 'High-Dimensional Continuous Control Using Generalized Advantage Estimation',
            author: 'Schulman et al.',
            year: 2015,
            venue: 'GAE — the modern advantage estimator',
            url: 'https://arxiv.org/abs/1506.02438',
          },
          {
            title: 'Proximal Policy Optimization Algorithms',
            author: 'Schulman et al.',
            year: 2017,
            venue: 'PPO — REINFORCE + baseline + trust region, the workhorse',
            url: 'https://arxiv.org/abs/1707.06347',
          },
        ]}
      />
    </div>
  )
}
