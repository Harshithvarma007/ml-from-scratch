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
import QTableUpdate from '../widgets/QTableUpdate'
import EpsilonGreedyCurve from '../widgets/EpsilonGreedyCurve'

// Signature anchor: the cheat sheet that grades every square-action pair.
// A giant lookup table, one row per square, one column per move, each cell
// holding "how much candy you eventually get if you play this entry". The
// lesson returns to the cheat sheet at the opening (the one you don't have
// but want), at the Bellman-update reveal (how one row learns from the
// next), and at the exploration section (you can't fill in the sheet if
// you never visit certain cells).
export default function QLearningLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="q-learning" />

      {/* ── Opening: the cheat sheet you don't have but want ─────── */}
      <Prose>
        <p>
          Imagine you&apos;re mid-game on a giant candy-grabbing board. You&apos;re
          standing on some square, four moves are legal, and you want the one
          that eventually hands you the most candy — not right now, but by the
          end of the game. What you would kill for is a <KeyTerm>cheat sheet</KeyTerm>{' '}
          — a massive lookup table with one row per square and one column per
          move, and inside each cell a single number: &ldquo;from square X, if
          I take action A and then play smart forever after, the total candy I
          eventually collect is <em>this much</em>.&rdquo;
        </p>
        <p>
          Call that number <code>Q(s, a)</code> and that table the{' '}
          <KeyTerm>Q-table</KeyTerm>. If you had it, the game would be trivial:
          look up the row for your current square, scan the columns, pick the
          cell with the biggest number, play that move. Repeat. You&apos;d be
          optimal. You&apos;d also be cheating, because of course you
          don&apos;t have the table — nobody handed it to you.
        </p>
        <p>
          Q-learning is the algorithm that <em>fills in the cheat sheet by
          playing</em>. You start with every entry scribbled at zero. You take
          actions, land in new squares, bank rewards. Each transition nudges
          exactly one cell — one row, one column — a little closer to truth.
          Do this ten million times and the lookup table becomes the lookup
          table you wish you&apos;d had at the start.
        </p>
      </Prose>

      <Callout variant="note" title="why this lesson exists">
        Value iteration — the algorithm from the{' '}
        <NeedsBackground slug="markov-decision-processes">MDP</NeedsBackground>{' '}
        lesson — also fills in a table. But it needs the environment&apos;s
        transition kernel <code>P(s&apos; | s, a)</code> and reward function{' '}
        <code>R(s, a)</code> as inputs. In the real world you almost never have
        those. A robot does not ship with a differential equation for its own
        dynamics. A game engine does not hand you a probability table. All you
        get is a stream of experience — states, actions, rewards — and a
        problem. Q-learning is the fix: same table, but filled in from samples
        instead of math.
      </Callout>

      <Prose>
        <p>
          So we flip the question. Instead of <em>planning</em> with a known
          model, can we <em>learn</em> the optimal action-value function
          directly, by doing? That is the promise of{' '}
          <KeyTerm>model-free reinforcement learning</KeyTerm>, and the
          canonical algorithm is <strong>Q-learning</strong> — Chris
          Watkins&apos; 1989 thesis, still one of the most widely used ideas in
          the field. It is four lines of code. It converges to the optimal
          policy. It does not need the model. It is the algorithm you
          implement first and the algorithm DeepMind scaled to Atari
          twenty-four years later.
        </p>
      </Prose>

      <Personify speaker="Q-learning">
        You do not know how the world works. That is fine. Take an action. See
        what happens. Nudge the entry in your cheat sheet toward what you just
        experienced. Do this ten million times. At the end you have a policy
        that is provably optimal. I require no map, no oracle, no teacher —
        just honest samples, a little patience, and one well-behaved table.
      </Personify>

      {/* ── The Bellman update reveal ──────────────────────────────── */}
      <Prose>
        <p>
          Here&apos;s the cheat sheet in detail. Rows are states, columns are
          actions, and each cell holds your current estimate of{' '}
          <code>Q*(s, a)</code> — the expected discounted return if you take
          action <code>a</code> in state <code>s</code> and then behave
          optimally forever after. You start the table at zero (or anywhere,
          really — it all washes out). Each time you act in the world you see
          a transition <code>(s, a, r, s&apos;)</code>. Then you apply exactly
          one update — to exactly one cell in the table, the one in row{' '}
          <code>s</code> and column <code>a</code>. That surgical locality is
          the point.
        </p>
      </Prose>

      <MathBlock caption="the Q-learning update — one cell, one correction, memorise this line">
{`Q(s, a)  ←  Q(s, a)  +  α · [ r  +  γ · max Q(s', a')  −  Q(s, a) ]
                                      a'
              └──────┘     └────────────────────────┘
              old cell     TD target (what this entry should be now)`}
      </MathBlock>

      <Prose>
        <p>
          Read it from the inside. The bracket is a <KeyTerm>temporal-difference error</KeyTerm>{' '}
          — the gap between the old entry <code>Q(s, a)</code> and a fresher,
          data-informed estimate <code>r + γ · max Q(s&apos;, a&apos;)</code>.
          The fresher number uses one real sample (the reward <code>r</code>)
          and one bootstrapped guess (the best entry in the <em>next</em> row,
          row <code>s&apos;</code>). You drag the old cell a step{' '}
          <code>α</code> toward the new one. That is it. That is Q-learning,
          and it&apos;s the Bellman equation from the MDP lesson reshaped into
          a sampled, per-cell correction instead of a full planning sweep.
        </p>
        <p>
          Three things are happening at once, and they are all load-bearing:
        </p>
        <ul>
          <li>
            <strong>Bootstrapping.</strong> We use the current cheat sheet to
            estimate future returns <em>while we are still filling it in</em>.
            This is circular but it works, because every update pulls each
            entry a little closer to truth, and the errors average out down
            the table.
          </li>
          <li>
            <strong>The <code>max</code> over the next row.</strong> We score
            the next state by the best cell in row <code>s&apos;</code>,
            regardless of what we actually did next. This is why Q-learning
            learns <code>Q*</code>, not <code>Q</code> of whatever noisy
            policy we happened to be using.
          </li>
          <li>
            <strong>Step size <code>α</code>.</strong> Small <code>α</code> =
            slow, stable, cell converges smoothly. Large <code>α</code> =
            fast, bouncy. Typical values: 0.1, 0.01. Same intuition as any{' '}
            <NeedsBackground slug="gradient-descent">SGD</NeedsBackground>-style
            method — the update rule even looks like gradient descent on a
            single-cell loss, because that&apos;s basically what it is.
          </li>
        </ul>
      </Prose>

      <Prose>
        <p>
          Here is a 5x5 gridworld with a goal cell (+1) and a pit (−1). The
          agent starts somewhere random, runs episodes, and you watch the
          cheat sheet fill in. Each cell on the grid shows the max Q-value at
          that state as a heatmap — think of it as peeking at the brightest
          entry in each row of the lookup table. Early episodes produce
          scattered, noisy estimates. Keep stepping. Watch the goal&apos;s
          value propagate backward, one cell per episode, until every row
          knows the way home.
        </p>
      </Prose>

      <QTableUpdate />

      <Prose>
        <p>
          This <em>backward propagation</em> of value is the entire vibe of TD
          learning. The goal cell&apos;s row knows its value first (it
          collects a real reward). Its neighbors&apos; rows learn about it
          through the <code>max</code> term — their update looks one step
          ahead, finds a brighter cell, and pulls their own entry upward.
          Their neighbors learn about those. Every episode nudges information
          one more cell outward from the source. After enough episodes, every
          row in the lookup table knows — through the chain of bootstrapped
          maxes — how to get to the goal.
        </p>
      </Prose>

      <Personify speaker="Q-table">
        I am the cheat sheet, assembled one honest guess at a time. Ask me{' '}
        <code>Q(s, a)</code> and I&apos;ll point you at row <code>s</code>,
        column <code>a</code>, and read out the number — how much return to
        expect if you take that entry and then behave like I told you to. I
        start out wrong about every cell. Every transition you feed me makes
        one of my entries slightly less wrong. Eventually argmax-ing my rows
        gives you the optimal policy, no planner required.
      </Personify>

      {/* ── On-policy vs off-policy ─────────────────────────────── */}
      <Callout variant="insight" title="off-policy — why Q-learning scales">
        Notice the <code>max</code> in the update. It does <em>not</em> care
        what action the agent actually took next. Q-learning updates the{' '}
        <code>(s, a)</code> cell toward the value of the <em>best possible</em>{' '}
        entry in the next row, regardless of which column got played. That
        single design choice makes Q-learning <KeyTerm>off-policy</KeyTerm>:
        the policy it is <em>learning</em> (greedy w.r.t. the table) can be
        different from the policy it is <em>behaving with</em> (usually
        ε-greedy, sometimes random, sometimes a human). You can fill in the
        cheat sheet from a replay buffer of old experience, from human
        demonstrations, from another agent&apos;s logs — anything that looks
        like <code>(s, a, r, s&apos;)</code>. That is a big deal. Its
        on-policy cousin SARSA cannot do this; SARSA updates toward{' '}
        <code>Q(s&apos;, a&apos;)</code> for the action it actually took, so
        it is stuck learning about its own current behavior.
      </Callout>

      {/* ── Exploration vs exploitation ─────────────────────────── */}
      <Prose>
        <p>
          Here&apos;s the catch nobody warns you about. The cheat sheet has
          thousands of cells. If your behavior policy always picks the
          brightest entry in the current row, you will revisit the same
          handful of cells forever — and <em>the rest of the table stays at
          zero, because you never visit those rows</em>. You can&apos;t fill
          in an entry you never touch. The algorithm is mathematically
          obligated to sometimes guess.
        </p>
        <p>
          That&apos;s the <KeyTerm>exploration vs exploitation</KeyTerm>{' '}
          trade-off. <em>Exploit</em> means &ldquo;play the best cell in the
          current row&rdquo; — cash in on what the table already claims.{' '}
          <em>Explore</em> means &ldquo;pick a random column&rdquo; — visit a
          cell you have low confidence in, collect a fresh sample, improve
          that entry. If we only exploit, the table freezes around whatever
          looked good first. If we only explore, we waste every step on
          flailing and the table is accurate but useless. The standard
          compromise is <KeyTerm>ε-greedy</KeyTerm>.
        </p>
      </Prose>

      <MathBlock caption="ε-greedy — the exploration knob">
{`π(a | s)   =   { random action                with probability ε
                { argmax Q(s, a)               with probability 1 − ε
                         a
                         └── pick the brightest cell in row s ──┘

schedule:  ε_t   =   max(ε_min,  ε_0 · decay^t)          typical: ε_0=1.0, ε_min=0.05`}
      </MathBlock>

      <Prose>
        <p>
          Flip a biased coin. With probability <code>ε</code>, do something
          random to find out if any untouched entry is better than what the
          table currently claims. With probability <code>1 − ε</code>, cash in
          on the brightest cell in the row. Early in training <code>ε</code>{' '}
          should be high — the table is blank, you know nothing, so exploit
          nothing. Late in training, <code>ε</code> should be low — the table
          is mostly filled in and the brightest entries are probably right. The
          schedule is a hyperparameter and, as we are about to see, it matters
          a lot.
        </p>
      </Prose>

      <EpsilonGreedyCurve />

      <Prose>
        <p>
          Three schedules, one task. <strong>Constant ε = 0.1</strong> is a
          common textbook default — it learns fast initially, then plateaus
          because it keeps throwing away 10% of its actions on random noise
          forever. <strong>Decaying ε</strong> starts wide and narrows to a
          small floor — best of both worlds in practice.{' '}
          <strong>Decay to zero</strong> looks tempting (pure exploitation at
          the end!) but is dangerous: if the agent latches onto a suboptimal
          row of cells early, it can never escape because it stops exploring
          entirely — the entries in the unvisited rows remain forever at their
          initial zero. The practical answer almost always: decay to a small
          non-zero floor.
        </p>
      </Prose>

      <Personify speaker="ε-greedy">
        I am the knob that trades curiosity for confidence. Turn me up and
        your agent wanders into unexplored rows — useful when the cheat sheet
        is mostly blank. Turn me down and your agent commits to the brightest
        cell it knows — useful when the table is mostly filled in. Turn me to
        zero and your agent freezes its opinions forever; hope the unvisited
        cells weren&apos;t hiding anything good. The art of reinforcement
        learning is mostly tuning me over time.
      </Personify>

      {/* ── Convergence ─────────────────────────────────────────── */}
      <Callout variant="note" title="the convergence guarantee (and its asterisks)">
        Tabular Q-learning converges to <code>Q*</code> with probability 1,
        provided two conditions hold. <strong>(1)</strong> Every state-action
        pair — every cell in the cheat sheet — is visited infinitely often,
        which is why ε must stay strictly positive. <strong>(2)</strong> The
        learning-rate sequence <code>α_t</code> satisfies the Robbins–Monro
        conditions: <code>Σα_t = ∞</code> (enough total learning) and{' '}
        <code>Σα_t² &lt; ∞</code> (decreasing noise). Constant <code>α</code>{' '}
        technically violates this, which is why small constant learning rates
        oscillate around <code>Q*</code> rather than converging to it. In
        practice people use small constants anyway and it works fine. The
        theorem is Watkins &amp; Dayan 1992.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Q-learning is famously short. Three layers, same algorithm, each one
          a little more production-ready than the last. Pure Python on a
          gridworld with a literal 2D list as the lookup table, NumPy with a
          replay buffer and the table as an array, PyTorch with a{' '}
          <NeedsBackground slug="mlp-from-scratch">neural network</NeedsBackground>{' '}
          standing in for the table — the last of which is DQN, and the
          bridge between the tabular toy and modern deep RL.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · q_learning_gridworld.py"
        output={`episode   0: return = -1.00  ε=0.90
episode 200: return =  1.00  ε=0.37
episode 500: return =  1.00  ε=0.06
learned policy: → → → → G
                ↑         ↑
                ↑ (pit)   ↑
                ↑ ←       ↑
                ↑ ← ← ← ←`}
      >{`import random

# 5x5 gridworld; state = (row, col); 4 actions: up, down, left, right.
# Q is our cheat sheet: 25 rows (one per square), 4 columns (one per move).
N_STATES, N_ACTIONS = 25, 4
Q = [[0.0] * N_ACTIONS for _ in range(N_STATES)]   # every cell starts at zero

alpha, gamma = 0.1, 0.95
eps, eps_min, eps_decay = 1.0, 0.05, 0.995

def step(s, a):                             # pretend-environment; returns (s', r, done)
    ...                                     # mechanical gridworld transitions

for episode in range(1000):
    s = env_reset()
    done = False
    while not done:
        # ε-greedy action selection — explore a random column, or exploit the row's best cell
        if random.random() < eps:
            a = random.randrange(N_ACTIONS)           # explore: random entry in row s
        else:
            a = max(range(N_ACTIONS), key=lambda a: Q[s][a])   # exploit: brightest cell

        s_next, r, done = step(s, a)

        # Q-learning update — the one line that matters.
        # Nudge the (s, a) cell toward "reward + γ * best cell in next row".
        td_target = r + (0 if done else gamma * max(Q[s_next]))
        Q[s][a]   = Q[s][a] + alpha * (td_target - Q[s][a])

        s = s_next

    eps = max(eps_min, eps * eps_decay)     # decay exploration each episode`}</CodeBlock>

      <Prose>
        <p>
          Now scale the two things that break in pure Python: batched updates
          and off-policy reuse of data. The idea of a{' '}
          <KeyTerm>replay buffer</KeyTerm> is simple — stash every transition
          you see in a ring buffer, and on each training step sample a random
          minibatch from it. You decorrelate consecutive samples (which are
          highly correlated in time), and you reuse every transition many
          times (which is how you actually learn efficiently from limited
          data). The cheat sheet becomes a NumPy array, and each step updates
          64 cells at once.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — numpy + replay buffer · q_learning_buffer.py">{`import numpy as np
from collections import deque

Q = np.zeros((N_STATES, N_ACTIONS))         # the cheat sheet, as an array
buffer = deque(maxlen=50_000)               # fixed-size replay memory

alpha, gamma = 0.1, 0.95

def update_batch(batch):
    s, a, r, s_next, done = batch
    # vectorised TD target — one line, all 64 transitions (64 cells) at once
    td_target = r + gamma * (1.0 - done) * Q[s_next].max(axis=1)   # max over each next row
    td_error  = td_target - Q[s, a]
    Q[s, a]  += alpha * td_error             # elementwise cell updates, no python loop

for t in range(100_000):
    # 1) interact with the env (behavior policy = ε-greedy on the table)
    a = np.argmax(Q[s]) if np.random.rand() > eps else np.random.randint(N_ACTIONS)
    s_next, r, done = env.step(a)
    buffer.append((s, a, r, s_next, float(done)))

    # 2) learn from a random minibatch of past experience
    if len(buffer) >= 64:
        batch = map(np.array, zip(*random.sample(buffer, 64)))
        update_batch(tuple(batch))

    s = env.reset() if done else s_next`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'Q[s][a] = Q[s][a] + α*(td - Q[s][a])',
            right: 'Q[s, a] += α * (td - Q[s, a])',
            note: 'same cell update, vectorised over a whole minibatch of rows',
          },
          {
            left: 'update on-the-fly after every step',
            right: 'buffer.append(...) then sample later',
            note: 'replay buffer = decorrelated, reusable transitions',
          },
          {
            left: 'max(Q[s_next])',
            right: 'Q[s_next].max(axis=1)',
            note: 'argmax over each next row — 64 lookups at once',
          },
        ]}
      />

      <Prose>
        <p>
          The tabular version cannot scale past toy problems — the cheat sheet
          has one row per state, and state spaces in the real world are huge
          (or continuous). You literally cannot allocate a table with a row
          per Atari frame. The fix, due to Mnih et al.&apos;s 2013 DQN paper,
          is obvious in retrospect: replace the lookup table with a neural
          net <code>Q_θ(s, a)</code> that <em>generalizes</em> across similar
          states instead of storing each entry by hand. The update stays
          structurally identical; only now the targets are fit by gradient
          descent on an MSE (or Huber) loss, and the bag of tricks gets
          bigger.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch DQN on CartPole · dqn_cartpole.py"
        output={`episode  50: return =  32.1  ε=0.74
episode 200: return = 141.6  ε=0.27
episode 500: return = 200.0  ε=0.05   ← CartPole solved
`}
      >{`import torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
import random, gymnasium as gym

# The "cheat sheet" is now a neural net: input a state, output one Q-value per action.
class QNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(),
                                 nn.Linear(128, 128),  nn.ReLU(),
                                 nn.Linear(128, n_out))
    def forward(self, s): return self.net(s)

env      = gym.make("CartPole-v1")
q, q_tgt = QNet(4, 2), QNet(4, 2)           # online net + target net
q_tgt.load_state_dict(q.state_dict())
opt      = torch.optim.Adam(q.parameters(), lr=1e-3)
buffer   = deque(maxlen=100_000)
gamma, eps = 0.99, 1.0

for episode in range(600):
    s, _ = env.reset()
    done, ep_return = False, 0.0
    while not done:
        # ε-greedy on the online Q-net (the learnable cheat sheet)
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(q(torch.tensor(s, dtype=torch.float32)).argmax())

        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buffer.append((s, a, r, s_next, float(done)))
        s, ep_return = s_next, ep_return + r

        # DQN learning step
        if len(buffer) >= 64:
            batch = random.sample(buffer, 64)
            s_, a_, r_, sn_, d_ = map(torch.tensor, zip(*batch))
            s_, sn_ = s_.float(), sn_.float()

            with torch.no_grad():
                td_target = r_ + gamma * (1 - d_) * q_tgt(sn_).max(1).values   # target net!
            td_pred = q(s_).gather(1, a_.unsqueeze(1)).squeeze(1)
            loss    = F.smooth_l1_loss(td_pred, td_target)                     # Huber loss

            opt.zero_grad(); loss.backward(); opt.step()

    # slow-moving target: hard copy every N episodes (or Polyak every step)
    if episode % 10 == 0:
        q_tgt.load_state_dict(q.state_dict())
    eps = max(0.05, eps * 0.995)`}</CodeBlock>

      <Bridge
        label="numpy → pytorch (DQN)"
        rows={[
          {
            left: 'Q[s, a]  (a table lookup)',
            right: 'q(s).gather(1, a)  (a forward pass)',
            note: 'function approximator replaces the lookup table — same semantics, infinite rows',
          },
          {
            left: 'Q[s, a] += α * td_error',
            right: 'loss.backward(); opt.step()',
            note: 'gradient descent on (td_pred − td_target)² does the same cell nudge',
          },
          {
            left: 'max Q(s_next)',
            right: 'q_tgt(sn_).max(1).values',
            note: 'target network — a stale copy of the cheat sheet to keep the target stable',
          },
          {
            left: 'one buffer + one Q-table',
            right: 'buffer + online net + target net',
            note: 'two nets because bootstrapping off a moving target diverges',
          },
        ]}
      />

      {/* ── DQN callouts ────────────────────────────────────────── */}
      <Callout variant="insight" title="what DQN actually added">
        The neural-net Q-function is the headline, but the reason DQN worked
        where naïve function-approximation Q-learning had failed for decades
        is three engineering choices bolted on around it.{' '}
        <strong>Replay buffer</strong> — decorrelate temporally-adjacent
        transitions so gradient descent sees roughly-i.i.d. batches instead of
        highly autocorrelated trajectories.{' '}
        <strong>Target network</strong> — bootstrap from a slow-moving copy of{' '}
        <code>Q_θ</code> so your target does not change every step; this
        single trick turns divergence into convergence.{' '}
        <strong>Huber (smooth-L1) loss</strong> — behaves like MSE near zero,
        L1 on large errors, which clips the damage from occasional huge TD
        errors early in training. Each piece looks mundane. Removing any one
        breaks the algorithm.
      </Callout>

      <Callout variant="note" title="Double DQN — the overestimation fix">
        Plain DQN&apos;s <code>max</code> operator systematically
        over-estimates — whenever Q has noise, <code>max</code> picks the
        noisy-high cell preferentially. Double DQN (van Hasselt et al., 2015)
        decouples action selection from action evaluation: use the online net
        to pick <code>a* = argmax_a Q_θ(s&apos;, a)</code>, then use the
        target net to <em>score</em> it. Two-line code change, measurably
        more stable policies, now the standard.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">ε never decays:</strong> you
          will exploration your way to a mediocre plateau forever. Symptom:
          learning curve rises, then stops climbing suspiciously early.
          Always decay ε, always to a small positive floor.
        </p>
        <p>
          <strong className="text-term-amber">Target network never updates:</strong>{' '}
          Q-values drift to ∞ because the target and the prediction are the
          same network and the whole thing is one giant positive-feedback
          loop. Symptom: losses explode, returns collapse. Always copy{' '}
          <code>q → q_tgt</code> on a fixed schedule (or use Polyak
          averaging).
        </p>
        <p>
          <strong className="text-term-amber">Too-small replay buffer:</strong>{' '}
          the distribution of transitions in the buffer becomes dominated by
          the agent&apos;s recent (narrow) policy. Old skills get overwritten
          — <KeyTerm>catastrophic forgetting</KeyTerm>. Typical DQN buffer:
          10⁵–10⁶ transitions.
        </p>
        <p>
          <strong className="text-term-amber">Unnormalised rewards:</strong> a
          reward of +100 once per episode and 0 otherwise makes the TD-target
          range enormous and destabilises gradient descent. Clip rewards to{' '}
          <code>[−1, 1]</code> or scale them; the original DQN paper does
          this on all Atari games and it is not a detail.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Solve CartPole with DQN, then look at your ε curve">
        <p>
          Copy the Layer-3 code, run it for 600 episodes on{' '}
          <code>CartPole-v1</code>, and plot the per-episode return. You
          should see a messy upward sweep — noisy for the first 100 episodes,
          climbing sharply around episode 150–300, plateauing at 200 (the max
          episode length) once the policy is solid.
        </p>
        <p className="mt-2">
          Then plot <code>ε</code> on the same axis. Notice the correlation:
          returns start climbing seriously once <code>ε</code> has decayed
          below ~0.3, i.e. once the agent mostly exploits the entries it
          trusts. Run again with <code>ε_min = 0.0</code>; you&apos;ll often
          see returns freeze mid-climb because the policy commits too early
          and the unvisited cells never get touched.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: implement Double DQN (two-line change in the target
          calculation) and check that the max Q-values grow more slowly than
          plain DQN. That slow growth is the whole point.
        </p>
      </Challenge>

      {/* ── Closing + teaser ─────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Q-learning is a cheat sheet
          you fill in by playing — one row per state, one column per action,
          one cell updated per transition toward a Bellman-style target. It
          is off-policy by construction, provably convergent in the tabular
          case, and generalises to deep RL by swapping the lookup table for a
          neural net. The three DQN tricks — replay buffer, target network,
          Huber loss — are not cosmetic; each one patches a specific failure
          mode of naïve function-approximation Q-learning. ε-greedy is the
          behavior policy that makes the whole table fillable at all: no
          exploration, no visits, no samples, no entries. Decaying ε on a
          sensible schedule is the single most common tuning lever.
        </p>
        <p>
          <strong>Next up — Policy Gradients.</strong> Q-learning learns a{' '}
          <em>value</em> table and extracts a policy by argmax-ing each row.
          Policy gradients skip the cheat sheet entirely and{' '}
          <em>directly</em> parameterise the policy itself as a neural net,
          optimising it by following the gradient of expected return. That
          buys us two huge things: it handles continuous action spaces
          natively (argmax over a continuous column is awkward), and it
          learns stochastic policies when they are optimal (Q-learning can
          only represent deterministic greedy policies). REINFORCE,
          actor-critic, PPO — everything modern — lives on the other side of
          that door.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Learning from Delayed Rewards',
            author: 'Christopher J. C. H. Watkins',
            venue: 'PhD thesis, University of Cambridge — the original Q-learning',
            year: 1989,
            url: 'https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf',
          },
          {
            title: 'Q-learning',
            author: 'Watkins, Dayan',
            venue: 'Machine Learning 8(3–4), 279–292 — the convergence proof',
            year: 1992,
            url: 'https://link.springer.com/article/10.1007/BF00992698',
          },
          {
            title: 'Playing Atari with Deep Reinforcement Learning',
            author: 'Mnih et al.',
            venue: 'arXiv preprint — the first DQN paper',
            year: 2013,
            url: 'https://arxiv.org/abs/1312.5602',
          },
          {
            title: 'Human-level control through deep reinforcement learning',
            author: 'Mnih et al.',
            venue: 'Nature 518, 529–533 — DQN on 49 Atari games',
            year: 2015,
            url: 'https://www.nature.com/articles/nature14236',
          },
          {
            title: 'Deep Reinforcement Learning with Double Q-learning',
            author: 'van Hasselt, Guez, Silver',
            venue: 'AAAI 2016 — the overestimation fix',
            year: 2015,
            url: 'https://arxiv.org/abs/1509.06461',
          },
          {
            title: 'Reinforcement Learning: An Introduction (2nd ed.), Chapter 6',
            author: 'Sutton, Barto',
            venue: 'MIT Press — the canonical TD-learning reference',
            year: 2018,
            url: 'http://incompleteideas.net/book/RLbook2020.pdf',
          },
        ]}
      />
    </div>
  )
}
