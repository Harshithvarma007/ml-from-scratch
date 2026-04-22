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
  AsciiBlock,
} from '../primitives'
import GridWorldMDP from '../widgets/GridWorldMDP'
import DiscountedReturn from '../widgets/DiscountedReturn'

// Signature anchor: the board game with no memory. You and the game are
// standing on a square (state); you pick a move (action); the dice roll
// hands you a new square (next state) and maybe a candy (reward). The
// game doesn't care how you got to this square — only where you are now.
// That's what "Markov" means. Returns at the opening (the board), the
// (S, A, R, P) tuple reveal (square, move, candy, dice), and the
// discount-factor section (a candy today vs a candy in ten turns).
export default function MarkovDecisionProcessesLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="markov-decision-processes" />

      {/* ── Pre-primer: the board game with no memory ──────────── */}
      <Prose>
        <p>
          Picture a board game. You and the game are standing on a{' '}
          <strong>square</strong>. You pick a <strong>move</strong>. The{' '}
          <strong>dice</strong> roll, and they hand you a new square — maybe
          the one you aimed for, maybe a sideways one because the board is
          slippery — and sometimes the square you land on gives you a{' '}
          <strong>candy</strong>. That is the whole game. Square, move, dice,
          candy. Repeat forever, or until someone wins.
        </p>
        <p>
          Here&apos;s the weird part. The game doesn&apos;t remember how you
          got to this square. It doesn&apos;t care whether you walked in
          straight from the start or ricocheted off five other squares first.
          All it sees is where your piece is right now. That is <em>all</em>{' '}
          it sees. Your future depends on your present square and nothing
          else. That&apos;s what &ldquo;Markov&rdquo; means — the board game
          with no memory.
        </p>
        <p>
          Every reinforcement learning algorithm you&apos;ve heard of — the
          thing that beat Go, the thing that turned a base language model
          into ChatGPT — is, underneath, a strategy for this board game. The
          rest of this page is what the square, the move, the dice, and the
          candy become when the board has ten billion squares and the dice
          are stochastic physics.
        </p>
      </Prose>

      <Callout variant="insight" title="four plain-English words before we start">
        <div className="space-y-2">
          <p>Four plain words you&apos;ll see over and over.</p>
          <p>
            <strong>State</strong> — the square your piece is on. Every fact
            about the situation that matters for what happens next.
          </p>
          <p>
            <strong>Action</strong> — the move you pick. Up, down, left,
            right. Raise, fold, call.
          </p>
          <p>
            <strong>Reward</strong> — the candy. One number the game hands
            you after your move. Bigger = better.
          </p>
          <p>
            <strong>Policy</strong> — your strategy. A rule that says: from
            this square, here&apos;s the move I&apos;ll make.
          </p>
        </div>
      </Callout>

      {/* ── Opening: problem-first reframe ──────────────────────── */}
      <Prose>
        <p>
          Supervised learning assumes somebody else already did the hard part:
          labeled the data. You get <code>(x, y)</code> pairs and your job is
          to learn the map. Reinforcement learning throws that away. There are
          no labels. There is a board, a player, and a candy jar that clinks
          or doesn&apos;t depending on what you do. The only feedback is that
          clink. Figure it out.
        </p>
        <p>
          The contract is almost shockingly simple. You see a{' '}
          <KeyTerm>state</KeyTerm> <code>s</code> — your square. You pick an{' '}
          <KeyTerm>action</KeyTerm> <code>a</code> — your move. The
          environment rolls the <strong>dice</strong> and replies with a{' '}
          <KeyTerm>reward</KeyTerm> <code>r</code> — the candy — and a new
          state <code>s&apos;</code> — your new square. The loop repeats,
          possibly forever. That&apos;s it. Every piece of RL — Q-learning,
          policy gradients, PPO, AlphaGo, ChatGPT&apos;s RLHF stage — is a
          different way of squeezing useful behavior out of that tiny
          contract.
        </p>
        <p>
          To reason about the loop we need a formalism. The field settled on
          the <KeyTerm>Markov Decision Process</KeyTerm> — a 5-tuple that
          pins down what the board is, what the moves are, how the dice roll,
          and what &ldquo;doing well&rdquo; means. This lesson is an honest
          build-up of that tuple, plus the one equation (Bellman&apos;s) that
          every RL algorithm in the next five lessons will solve, copy, or
          approximate.
        </p>
      </Prose>

      <Personify speaker="Reinforcement learning">
        I don&apos;t need labels. Give me a board, a move set, and a candy
        that clinks when things are good. I&apos;ll figure out the rest. It
        will take me a hundred million tries, but I will figure out the rest.
      </Personify>

      {/* ── The agent-environment loop ───────────────────────────── */}
      <Prose>
        <p>
          Before any math, the picture. Every RL textbook draws this diagram
          because it is genuinely the whole setup — the board game loop with
          the players relabeled.
        </p>
      </Prose>

      <AsciiBlock caption="the agent-environment loop — the board game, formalized">
{`   ┌──────────────┐   action aₜ    ┌──────────────────┐
   │              │   (your move)  │                  │
   │    AGENT     │ ─────────────▶ │   ENVIRONMENT    │
   │   (you)      │                │   (the board     │
   │              │ ◀───────────── │    + the dice)   │
   └──────────────┘   rₜ₊₁, sₜ₊₁   └──────────────────┘
         ▲           (candy, new square)       │
         │         state sₜ (your square)      │
         └─────────────────────────────────────┘

   time →  s₀, a₀, r₁, s₁, a₁, r₂, s₂, a₂, r₃, s₃,  …`}
      </AsciiBlock>

      <Prose>
        <p>
          The agent is whatever decides moves — you, a lookup table, a neural
          network. The environment is everything else: the board, the dice,
          the opponent, the physics. A <KeyTerm>trajectory</KeyTerm> is just
          the alternating sequence of squares, moves, and candies that the
          loop produces. The goal is to pick moves so that the candies along
          the trajectory add up to something large.
        </p>
        <p>
          To make &ldquo;add up to something large&rdquo; precise we need to
          define what the game <em>is</em>. That&apos;s the five-tuple.
        </p>
      </Prose>

      {/* ── The 5-tuple ──────────────────────────────────────────── */}
      <MathBlock caption="MDP — the 5-tuple that pins down the board game">
{`M   =   ⟨ S,  A,  P,  R,  γ ⟩

S   =   set of states             every square your piece could be on
A   =   set of actions            every move you could make
P(s'|s,a)   =   transition prob.  the dice — chance of landing on s' from (s, a)
R(s, a)     =   reward            the candy — scalar you get for doing a in s
γ ∈ [0, 1)  =   discount factor   candy-today vs candy-in-ten-turns`}
      </MathBlock>

      <Prose>
        <p>
          <code>S</code> is every square — they can be a tiny discrete set (9
          squares on a tic-tac-toe board) or an enormous continuous one (every
          possible 224×224 RGB frame of an Atari game).{' '}
          <code>A</code> is the move set — four arrows on a grid, every
          possible torque on a robot arm. <code>P</code> is the dice: given a
          square and a move, it says probabilistically which square comes
          next. <code>R</code> is the candy jar: scalar in, nothing fancy.
          And <code>γ</code> is the strange one — a number between 0 and 1
          that controls how much you care about candy you could collect ten
          turns from now.
        </p>
        <p>
          One subtle constraint binds all of this together: the{' '}
          <KeyTerm>Markov property</KeyTerm>. The next square depends only on
          the current square and move, not on the full history of how you got
          here. Formally{' '}
          <code>P(s&apos; | s, a, s_{'{t-1}'}, a_{'{t-1}'}, …) = P(s&apos; | s, a)</code>.
          The present screens off the past. The board game has no memory.
        </p>
        <p>
          If the real world isn&apos;t Markov, <em>engineer the state until
          it is</em>. A single Atari frame isn&apos;t Markov — you can&apos;t
          tell which way the ball is moving from one still image — so DQN
          stacks the last 4 frames into the state. Poker isn&apos;t Markov
          from your cards alone — you also need the betting history — so put
          the betting history in the state. Rewrite the square until it
          carries every fact the future depends on. The trick to RL
          engineering is usually not the algorithm; it&apos;s the state
          representation.
        </p>
      </Prose>

      {/* ── Widget 1: GridWorldMDP ───────────────────────────────── */}
      <Prose>
        <p>
          Abstract enough. Let&apos;s make it concrete. Below is a literal
          board — a 5×5 grid with walls, a goal, and candies. Click a move;
          watch the dice roll you into a new square and the candy land on
          the scoreboard. This is a fully-specified MDP — you can read off{' '}
          <code>S</code>, <code>A</code>, <code>P</code>, and <code>R</code>{' '}
          from the UI directly.
        </p>
      </Prose>

      <GridWorldMDP />

      <Prose>
        <p>
          A few things to notice. The square is just your{' '}
          <code>(row, col)</code> — nothing about past moves is carried
          forward, and nothing needs to be (this game is exactly Markov by
          construction). The move set is tiny: four directions. The dice are
          stochastic (with some slip probability you skid sideways instead of
          forward), and the candy is usually a small negative per step plus a
          big positive at the goal. That&apos;s how the board nudges you
          toward short paths — every wasted step costs you a sliver of candy.
        </p>
        <p>
          Every gridworld you&apos;ll read about in Sutton &amp; Barto is a
          tiny variant of this. The reason the field keeps using them is
          that they&apos;re small enough to solve exactly — you can literally
          enumerate every square and every move — which makes them the ideal
          lab for intuition about algorithms that then scale up to Go and
          Dota.
        </p>
      </Prose>

      {/* ── Return and discount ─────────────────────────────────── */}
      <Prose>
        <p>
          You want to maximize total candy over a trajectory. But
          &ldquo;total&rdquo; is ambiguous — over what horizon? If the game
          runs forever and candies are bounded positive, the sum diverges.
          That&apos;s where <code>γ</code> comes in. A candy today is not
          worth the same as a candy in ten turns. The formal quantity you
          actually maximize is called the <KeyTerm>return</KeyTerm>:
        </p>
      </Prose>

      <MathBlock caption="the discounted return — candy, weighed by how soon you get it">
{`                     ∞
G_t   =   r_{t+1}  +  γ · r_{t+2}  +  γ² · r_{t+3}  +  …   =   Σ  γᵏ · r_{t+k+1}
                                                             k=0

γ = 0   →   myopic; only the next candy matters.
γ = 1   →   infinite horizon; every candy weighted equally (sum may diverge).
γ = 0.9 →   effective horizon ≈ 1 / (1 − γ) = 10 turns.
γ = 0.99 →  effective horizon ≈ 100 turns.`}
      </MathBlock>

      <Prose>
        <p>
          Two things are doing work here. First, the math: any{' '}
          <code>γ &lt; 1</code> combined with bounded candy makes the infinite
          sum finite (geometric series). A potentially-undefined objective
          becomes a well-defined optimization problem — which is the whole
          point. Second, the intuition: <code>γ</code> is a dial for how much
          you care about candy you could collect later versus candy you could
          grab right now. A low <code>γ</code> produces a short-sighted player
          that pockets whatever candy is closest. A high <code>γ</code>{' '}
          produces a patient player willing to eat a small penalty now to
          unlock a big candy ten squares from here.
        </p>
      </Prose>

      <Personify speaker="Discount factor γ">
        I am the horizon dial — candy-today versus candy-in-ten-turns. Crank
        me to 0.99 and your player plans a hundred moves ahead. Crank me to
        0.5 and it will sell its grandmother for a candy it can grab in two
        moves. Pick me badly and your player is either a lunatic or a
        goldfish.
      </Personify>

      {/* ── Widget 2: DiscountedReturn ───────────────────────────── */}
      <Prose>
        <p>
          Drag the γ slider. The plot shows a 50-turn trajectory with a fixed
          candy sequence; the bars show how much each turn&apos;s candy
          contributes to <code>G₀</code> after discounting. At{' '}
          <code>γ = 0.5</code> the bars past turn 10 are visually nothing —
          you effectively don&apos;t see them. At <code>γ = 0.99</code> the
          50th turn still contributes meaningfully. This is the horizon
          effect you&apos;ll feel in every algorithm from here on.
        </p>
      </Prose>

      <DiscountedReturn />

      <Callout variant="note" title="the 1 / (1 − γ) rule of thumb">
        For planning purposes, treat <code>1 / (1 − γ)</code> as your
        &ldquo;effective horizon.&rdquo; Atari DQN uses <code>γ = 0.99</code>,
        so the agent weighs roughly the next 100 frames. A continuing-control
        problem like locomotion might use <code>γ = 0.995</code> (horizon ≈
        200). A short game might use <code>γ = 0.9</code>. It&apos;s the
        single most impactful knob you don&apos;t tune often.
      </Callout>

      {/* ── Policy, V, Q ─────────────────────────────────────────── */}
      <Prose>
        <p>
          The thing we&apos;re learning is called a <KeyTerm>policy</KeyTerm>{' '}
          — your strategy, a rule for picking moves. Two flavors:
        </p>
        <ul>
          <li>
            <strong>Deterministic:</strong> <code>π(s) = a</code>. &ldquo;On
            square <code>s</code>, make move <code>a</code>.&rdquo;
          </li>
          <li>
            <strong>Stochastic:</strong> <code>π(a | s)</code> is a
            probability distribution over moves. &ldquo;On square{' '}
            <code>s</code>, flip this weighted coin.&rdquo;
          </li>
        </ul>
        <p>
          To compare strategies we need a score. The score of being on square{' '}
          <code>s</code> while following policy <code>π</code> is the
          expected return from there — the expected candy total, discounted:
        </p>
      </Prose>

      <MathBlock caption="value functions — V and Q">
{`V^π(s)     =   E_π [ G_t  |  s_t = s ]

Q^π(s, a)  =   E_π [ G_t  |  s_t = s,  a_t = a ]

V and Q relate by:
     V^π(s)  =  Σ_a  π(a | s) · Q^π(s, a)
     Q^π(s, a)  =  R(s, a)  +  γ · Σ_{s'}  P(s' | s, a) · V^π(s')`}
      </MathBlock>

      <Prose>
        <p>
          <code>V</code> is the <em>square</em> value — &ldquo;how good is
          this square?&rdquo; <code>Q</code> is the <em>square-and-move</em>{' '}
          value — &ldquo;how good is this square <em>if I move right
          first</em>?&rdquo; Most modern deep RL learns <code>Q</code>{' '}
          directly because the player can act greedily:{' '}
          <code>a = argmax_a Q(s, a)</code> — just look up every move from
          this square and take the best one.
        </p>
      </Prose>

      {/* ── Bellman equation ─────────────────────────────────────── */}
      <Prose>
        <p>
          Here&apos;s the recursion that holds everything together. The
          value of a square can be written in terms of the values of the
          squares one move ahead:
        </p>
      </Prose>

      <MathBlock caption="Bellman equation — the recursive identity of V">
{`V^π(s)   =   Σ  π(a | s)  ·  [ R(s, a)  +  γ · Σ  P(s' | s, a) · V^π(s') ]
             a                                  s'
             └──────────┘    └──────┘    └──────────────────────────────┘
             weight over    immediate      γ-discounted value of
             moves          candy          where we'll land next`}
      </MathBlock>

      <Prose>
        <p>
          Read it out loud. &ldquo;The value of this square under strategy{' '}
          <code>π</code> is the average, over moves <code>π</code> might
          take, of the candy I pick up plus the discounted value of the
          square I&apos;ll land on.&rdquo; The recursion folds the whole
          infinite-horizon return into a one-step relationship. That is
          enormous. It means we can solve for <code>V</code> by iterating a
          local update instead of computing an infinite sum.
        </p>
        <p>
          Analogously there&apos;s a <KeyTerm>Bellman optimality equation</KeyTerm>,
          where instead of averaging over moves the player picks the best
          one:
        </p>
      </Prose>

      <MathBlock caption="Bellman optimality — if you play perfectly">
{`V*(s)   =   max  [ R(s, a)  +  γ · Σ P(s' | s, a) · V*(s') ]
             a                       s'

π*(s)   =   argmax  [ R(s, a)  +  γ · Σ P(s' | s, a) · V*(s') ]
              a                         s'`}
      </MathBlock>

      <Prose>
        <p>
          An <KeyTerm>optimal policy</KeyTerm> <code>π*</code> is one that
          maximizes <code>V^π(s)</code> for every square <code>s</code>{' '}
          simultaneously. A remarkable theorem (due to Bellman, then Puterman
          tidied it up) says such a strategy exists for every finite MDP —
          and while there can be ties between multiple optimal strategies,
          the optimal <em>value</em> is unique. Put differently: there is a
          single right answer in value-space, and solving the Bellman
          optimality equation recovers it.
        </p>
      </Prose>

      <Personify speaker="Bellman equation">
        I turn an infinite sum into a one-step recursion. Give me the values
        of the squares you&apos;ll be on next, and I&apos;ll give you the
        value of the square you&apos;re on. Iterate me and I converge.
        Neural-network me and I become deep Q-learning. I am the backbone of
        this entire field.
      </Personify>

      <Callout variant="insight" title="why everyone keeps writing about Bellman">
        Look at any RL algorithm you&apos;ve heard of — value iteration,
        policy iteration, Q-learning, SARSA, DQN, TD(λ), A2C, PPO. Every
        single one is a trick for either evaluating or improving a strategy
        by exploiting the Bellman recursion. Value iteration applies it as
        an exact update on a table. Q-learning applies it as a sampled update
        on a single move. DQN applies it inside a neural net. It&apos;s the
        same equation all the way down.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Enough theory — let&apos;s run it. Same three-layer build as usual:
          pure Python first (a single Bellman backup on a 3-square toy), NumPy
          next (full value iteration on our 5×5 board), and finally a PyTorch
          sketch where the value function is a neural net trained by{' '}
          <NeedsBackground slug="gradient-descent">gradient descent</NeedsBackground>{' '}
          (a preview of deep RL).
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · bellman_backup.py"
        output={`iter 00  V = [0.0, 0.0, 0.0]
iter 01  V = [-1.0, -1.0, 10.0]
iter 05  V = [6.561, 7.29, 10.0]
iter 20  V ≈ [7.2898, 8.0998, 10.0]
converged — optimal policy: {s0: 'right', s1: 'right', s2: 'stay'}`}
      >{`# A 3-square toy MDP:  s0 ──right──▶ s1 ──right──▶ s2 (terminal, +10 candy)
# step-candy of -1 for every non-terminal move; γ = 0.9.

gamma = 0.9
states = ["s0", "s1", "s2"]
actions = {"s0": ["right"], "s1": ["right", "left"], "s2": ["stay"]}

# transition model: T[s][a] = (s', r) — where the dice land, what candy you get
T = {
    "s0": {"right": ("s1", -1.0)},
    "s1": {"right": ("s2", +10.0), "left": ("s0", -1.0)},
    "s2": {"stay":  ("s2",  0.0)},            # absorbing
}

V = {s: 0.0 for s in states}                  # initial guess

def bellman_backup(V):
    V_new = {}
    for s in states:
        # Bellman optimality: V*(s) = max_a [ candy + γ · V(next square) ]
        V_new[s] = max(r + gamma * V[s_next] for (s_next, r) in
                       (T[s][a] for a in actions[s]))
    return V_new

for i in range(21):
    V = bellman_backup(V)
    if i in (0, 1, 5, 20):
        print(f"iter {i:02d}  V = {[round(V[s], 4) for s in states]}")

# extract the greedy strategy — on each square, pick the move with highest value
policy = {s: max(actions[s], key=lambda a: T[s][a][1] + gamma * V[T[s][a][0]])
          for s in states}
print("converged — optimal policy:", policy)`}</CodeBlock>

      <Prose>
        <p>
          Vectorise it. On a 5×5 board we have 25 squares and 4 moves. We can
          stuff the whole MDP into NumPy arrays — <code>P</code> is shape{' '}
          <code>(S, A, S)</code>, <code>R</code> is <code>(S, A)</code> — and
          express one sweep of value iteration as a handful of tensor
          contractions.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · value_iteration.py"
        output={`sweep 000  Δ = 10.0000
sweep 010  Δ = 1.3421
sweep 050  Δ = 0.0074
sweep 112  Δ = 9.7e-05  ✓ converged
V* grid:
 [[ 5.32  6.58  8.10  9.83 10.00]
  [ 4.27  5.32  6.58  8.10  9.83]
  [ 3.34  4.27  5.32  6.58  8.10]
  [ 2.51  3.34   -∞   5.32  6.58]
  [ 1.76  2.51  3.34  4.27  5.32]]`}
      >{`import numpy as np

# 5x5 board: moves = {0:up, 1:right, 2:down, 3:left}
# candy +10 at the goal square (0, 4), step cost -0.04 elsewhere, wall at (3, 2).
S, A = 25, 4
gamma = 0.95

# build P[s, a, s'] and R[s, a] from the grid layout
P = np.zeros((S, A, S))
R = np.full((S, A), -0.04)
# ... (populate P, R from grid topology — omitted for brevity) ...

V = np.zeros(S)
for sweep in range(500):
    # Q(s, a) = candy(s, a) + γ · Σ_{s'} dice(s, a, s') · V(s')
    Q = R + gamma * P @ V                                 # shape (S, A)
    V_new = Q.max(axis=1)                                 # Bellman optimality
    delta = np.abs(V_new - V).max()
    V = V_new
    if delta < 1e-4:
        print(f"sweep {sweep:03d}  Δ = {delta:.2e}  ✓ converged")
        break

pi = Q.argmax(axis=1)                                     # greedy policy
print("V* grid:\\n", V.reshape(5, 5).round(2))`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for s in states: max over actions ...',
            right: 'Q = R + γ · (P @ V); V = Q.max(axis=1)',
            note: 'one matmul replaces the nested Python loop — the entire Bellman sweep is 2 lines',
          },
          {
            left: 'dict of transitions T[s][a] = (s\', r)',
            right: 'arrays P[s, a, s\'] and R[s, a]',
            note: 'dense tensors — wasteful for sparse boards, but vectorizable and GPU-friendly',
          },
          {
            left: 'extract policy with argmax in a loop',
            right: 'pi = Q.argmax(axis=1)',
            note: 'one call, whole strategy',
          },
        ]}
      />

      <Prose>
        <p>
          On the 5×5 board value iteration converges in ~100 sweeps. But what
          if the board has 10⁴⁸ squares (Go) or continuous squares (a robot
          arm)? You can&apos;t store a table.{' '}
          <KeyTerm>Function approximation</KeyTerm>: replace the table with a
          neural network that maps <code>square → value</code>. The Bellman
          equation becomes a loss function, and you train by{' '}
          <NeedsBackground slug="gradient-descent">gradient descent</NeedsBackground>{' '}
          on it. This is the entire premise of deep RL — and it&apos;s the
          bridge to the next few lessons.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · bellman_as_loss.py"
        output={`step 0000  bellman loss = 23.8412
step 0500  bellman loss =  0.6194
step 2000  bellman loss =  0.0381
V_net(start square) ≈ 5.34  (tabular V* = 5.32)`}
      >{`import torch
import torch.nn as nn

# Neural value approximator: square features → scalar V(s).
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,  hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, s):
        return self.net(s).squeeze(-1)

# Assume we have a replay of (s, a, r, s') transitions from a random policy.
V_net = ValueNet(state_dim=25)                         # one-hot square features
optim = torch.optim.Adam(V_net.parameters(), lr=3e-3)

for step in range(2001):
    s, r, s_next = sample_batch()                      # from replay buffer
    with torch.no_grad():
        target = r + gamma * V_net(s_next)             # Bellman target — TD(0)
    pred = V_net(s)
    loss = ((pred - target) ** 2).mean()               # regression on Bellman residual
    optim.zero_grad(); loss.backward(); optim.step()
    if step % 500 == 0:
        print(f"step {step:04d}  bellman loss = {loss.item():7.4f}")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'V = np.zeros(S)  # a table, one entry per square',
            right: 'V_net = ValueNet(...)  # a parametric function',
            note: 'from a value per square to a function that generalizes across squares',
          },
          {
            left: 'V_new = (R + γ P @ V).max(axis=1)',
            right: 'target = r + γ · V_net(s_next); MSE(V_net(s), target)',
            note: 'same Bellman equation, now a regression target for gradient descent',
          },
          {
            left: 'sweep until Δ < ε',
            right: 'train until the loss plateaus',
            note: 'fixed-point iteration becomes stochastic optimization — welcome to deep RL',
          },
        ]}
      />

      <Callout variant="insight" title="the point of the three layers">
        The pure-Python version shows Bellman is really just &ldquo;candy plus
        γ times the value of where the dice take you.&rdquo; The NumPy version
        shows that, when the board is enumerable, this is a trivially
        vectorizable dynamic-programming problem. The PyTorch version shows
        what to do when <em>the board is too large to enumerate</em> — you
        give up exactness for generalization, and the Bellman equation
        becomes a loss. DQN, A2C, PPO, SAC: all variants of layer 3.
      </Callout>

      {/* ── Episodic vs continuing ──────────────────────────────── */}
      <Callout variant="note" title="episodic vs. continuing games">
        Some boards have a natural stopping point — one chess game, one level
        of Breakout, one episode of a dialog. Those are <em>episodic</em>:
        trajectories end, you reset, start over. Others never stop — a
        thermostat, an HVAC controller, a market maker. Those are{' '}
        <em>continuing</em>. For episodic games you can technically use{' '}
        <code>γ = 1</code> because the sum is finite. For continuing games
        you must use <code>γ &lt; 1</code> or switch to an average-reward
        formulation (outside scope here). In practice people use{' '}
        <code>γ = 0.99</code> for both and it just works.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Undiscounted infinite returns:</strong>{' '}
          if you set <code>γ = 1</code> on a continuing game with non-zero
          candy, <code>G_t</code> is divergent and <em>nothing is optimal</em>{' '}
          — the Bellman equation has no finite fixed point. Your loss will
          look fine for a while, then explode. Keep <code>γ &lt; 1</code>.
        </p>
        <p>
          <strong className="text-term-amber">Confusing V and Q:</strong>{' '}
          <code>V(s)</code> is &ldquo;how good is this square,&rdquo;{' '}
          <code>Q(s, a)</code> is &ldquo;how good is this square <em>if I
          make move a first</em>.&rdquo; Q-learning learns <code>Q</code>{' '}
          because you can act greedily from it without knowing the dice.
          V-learning needs a model of <code>P</code> to act.
        </p>
        <p>
          <strong className="text-term-amber">Expected vs. realized return:</strong>{' '}
          <code>V^π(s) = E[G_t]</code> is an <em>expectation</em>. Any single
          rollout gives a single realized <code>G_t</code>, which can be
          wildly off the mean in a stochastic game. Never judge a strategy
          on one episode.
        </p>
        <p>
          <strong className="text-term-amber">Unseeded environments:</strong>{' '}
          gym-style environments roll stochastic dice. Without{' '}
          <code className="text-dark-text-primary">env.seed(k)</code> and{' '}
          <code className="text-dark-text-primary">np.random.seed(k)</code>{' '}
          your experiments are non-reproducible — and RL is famously brittle
          enough that seed-to-seed variance can swallow real algorithmic
          improvements. Seed everything, log the seed, report median across
          at least 5 seeds.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Solve a 5×5 board with value iteration">
        <p>
          Build the MDP: a 5×5 board with a goal square at <code>(0, 4)</code>{' '}
          giving candy <code>+10</code> (terminal), a wall at{' '}
          <code>(3, 2)</code> that can&apos;t be entered, a step cost of{' '}
          <code>-0.04</code> on every other square, and <code>γ = 0.95</code>.
          Moves are <code>{'{up, right, down, left}'}</code>; bumping into a
          wall or the board edge leaves your piece in place.
        </p>
        <p className="mt-2">
          Fill in the <code>(25, 4, 25)</code> transition tensor, then run
          value iteration until max <code>Δ &lt; 1e-4</code>. Reshape{' '}
          <code>V*</code> back to <code>(5, 5)</code> and render it as a
          heatmap (matplotlib <code>imshow</code>). Overlay the greedy
          strategy as arrows pointing out of each square using{' '}
          <code>plt.quiver</code>.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: add a 10% slip probability (the dice send you perpendicular
          to your intended direction with prob 0.05 each side). Watch the
          value function and strategy change — the optimal route now avoids
          corridors next to the cliff, because when the dice misbehave,
          playing close to the edge is risky even if it&apos;s shorter.
        </p>
      </Challenge>

      {/* ── Closing ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> An MDP is a 5-tuple{' '}
          <code>⟨S, A, P, R, γ⟩</code> — squares, moves, dice, candy,
          horizon dial — plus the Markov property, which says the game has
          no memory. Your goal is to maximize the discounted return{' '}
          <code>G_t = Σ γᵏ r_{'{t+k+1}'}</code>. A strategy{' '}
          <code>π</code> has a value function <code>V^π</code>, which
          satisfies the Bellman recursion — a one-step equation that folds
          up the whole infinite horizon. Solving or approximating this
          recursion <em>is</em> reinforcement learning. Value iteration does
          it exactly on a table; deep RL does it approximately with a neural
          net.
        </p>
        <p>
          <strong>Next up — Q-Learning.</strong> We&apos;ve been assuming we
          know how the dice roll — that <code>P(s&apos; | s, a)</code> is
          handed to us. In real life you don&apos;t get to peek at the dice.
          Q-learning is the first RL algorithm that doesn&apos;t need{' '}
          <code>P</code> — it learns the value of every square-and-move pair
          purely from watching actual rolls happen, building up a kind of
          cheat sheet square by square. It&apos;s the step from planning to
          actually learning, and it&apos;s the direct ancestor of DQN.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Reinforcement Learning: An Introduction (2nd ed.)',
            author: 'Sutton & Barto',
            venue: 'MIT Press, 2018 — the canonical RL textbook',
            url: 'http://incompleteideas.net/book/the-book-2nd.html',
          },
          {
            title: 'Markov Decision Processes: Discrete Stochastic Dynamic Programming',
            author: 'Puterman',
            venue: 'Wiley, 1994 — the formal measure-theoretic treatment',
            url: 'https://www.wiley.com/en-us/Markov+Decision+Processes%3A+Discrete+Stochastic+Dynamic+Programming-p-9780471727828',
          },
          {
            title: 'Spinning Up in Deep RL — Key Concepts',
            author: 'OpenAI (Joshua Achiam)',
            venue: 'the cleanest modern intro to the MDP formalism',
            url: 'https://spinningup.openai.com/en/latest/spinningup/rl_intro.html',
          },
          {
            title: 'Dynamic Programming',
            author: 'Bellman',
            venue: 'Princeton University Press, 1957 — where the equation came from',
            url: 'https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming',
          },
        ]}
      />
    </div>
  )
}
