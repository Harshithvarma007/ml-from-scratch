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
import ActorCriticPipeline from '../widgets/ActorCriticPipeline'
import A2CTraining from '../widgets/A2CTraining'

// Signature anchor: the playwright and the theater critic. The actor (policy)
// performs; the critic (value function) grades the performance in real time.
// Without the critic, the actor has to wait until the end of the night to hear
// if they were good. With the critic, every scene gets an immediate review,
// and the actor adjusts mid-performance. Returned at opening (the actor
// performing blind), the advantage-function reveal (critic says: "that scene
// was better than average"), and the "both learning simultaneously" section.
export default function ActorCriticLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="actor-critic" />

      {/* ── Opening: the actor performing blind ─────────────────── */}
      <Prose>
        <p>
          Picture an actor on stage, performing a new play for the first time. The house lights
          are down, the audience is silent, and the actor has no idea whether the scene
          they&apos;re in the middle of is landing or dying. They commit to the line, the
          blocking, the pause. Only at the very end of the night — curtain down, applause or
          polite coughing — do they find out if the performance worked.
        </p>
        <p>
          That is <NeedsBackground slug="reinforce">REINFORCE</NeedsBackground>. The actor
          improvises an entire episode, reads the total reward at the end, and tries to
          retro-engineer which scenes were good and which were embarrassing. One episode you
          stumble into a standing ovation, the next you fall off a cliff, and the{' '}
          <NeedsBackground slug="policy-gradients">policy gradient</NeedsBackground> swings
          wildly to match. It&apos;s the high-variance problem, and every performer knows it:
          you can&apos;t adjust mid-performance if the only feedback is at opening night.
        </p>
        <p>
          Textbook fix: subtract a <KeyTerm>baseline</KeyTerm> from the return. The gradient
          stays unbiased, the variance drops. Fine. But <em>what</em> baseline? The best one,
          the one the math actually wants, is the state-value{' '}
          <NeedsBackground slug="q-learning">value function</NeedsBackground>{' '}
          <code>V(s)</code>: the expected return from state <code>s</code> under the current
          policy. Subtract that from
          each episode&apos;s return and the update only moves when the action was{' '}
          <em>better or worse</em> than average from the state the actor happened to be in.
          Which is the whole point of grading a scene.
        </p>
        <p>
          You don&apos;t know <code>V(s)</code>. So you hire someone to figure it out while the
          actor is still on stage. That&apos;s <KeyTerm>actor-critic</KeyTerm>: a playwright and
          a theater critic sharing one rehearsal room. Train a second network — the{' '}
          <strong>critic</strong>, <code>V_φ</code> — whose entire job is to evaluate the
          state the actor is standing in, and use its review as the baseline for the{' '}
          <strong>actor</strong>, <code>π_θ</code>. Actor performs the scene. Critic grades it
          in real time. Actor adjusts on the fly. Critic calibrates against what actually
          happens next. They rehearse together. That pairing is so useful that
          &ldquo;actor-critic&rdquo; stopped being an optional add-on and became the default
          shape of every modern policy-gradient algorithm — A2C, A3C, PPO, SAC, TRPO. Same
          bones underneath.
        </p>
      </Prose>

      <MathBlock caption="two networks, two updates, one trajectory">
{`Actor   (policy):     θ ← θ + α_θ · ∇log π_θ(a|s) · A(s, a)

Critic  (value):      φ ← φ + α_φ · ∇V_φ(s) · (r + γ · V_φ(s') − V_φ(s))

                                  ╰──────── δ, TD error ────────╯

             A(s, a) ≈ δ    (advantage ≈ TD error in the 1-step case)`}
      </MathBlock>

      <Prose>
        <p>
          Read those two lines like a script with stage directions. Both updates are driven by
          the same signal — <code>r + γV_φ(s&apos;) − V_φ(s)</code>, the{' '}
          <KeyTerm>TD error</KeyTerm>. The critic reads it as a prediction error (&ldquo;I
          said this scene was worth <code>V(s)</code>; the real performance says it was worth{' '}
          <code>r + γV(s&apos;)</code>; here&apos;s my miscalibration&rdquo;). The actor reads
          the <em>same</em> number as an advantage (&ldquo;this line beat the critic&apos;s
          expectation by <em>this much</em> — do more of that&rdquo;). One scalar, two
          readings, two gradient steps on two networks. That&apos;s the whole trick.
        </p>
      </Prose>

      {/* ── Widget 1: Actor-Critic Pipeline ─────────────────────── */}
      <ActorCriticPipeline />

      <Prose>
        <p>
          The pipeline. State of the stage goes into both heads. Actor emits{' '}
          <code>π(a|s)</code>, a distribution over next actions, samples one, and delivers the
          line to the environment. The environment returns <code>(r, s&apos;)</code> — the
          audience&apos;s instantaneous reaction and the shape of the next scene. Critic looks
          at <code>V(s)</code> and <code>V(s&apos;)</code>; the advantage{' '}
          <code>A = r + γV(s&apos;) − V(s)</code> is just &ldquo;how much better was the scene
          than my forecast.&rdquo; Then the same <code>A</code> flows into two places: scaled
          by <code>∇log π(a|s)</code> it updates the actor, and as a plain scalar target it
          updates the critic. One forward pass, one environment step, two gradient steps.
          Repeat until opening night.
        </p>
      </Prose>

      <Personify speaker="Actor">
        I perform the scene. I don&apos;t know whether the beat landed until the Critic in the
        back of the house tells me. My gradient is whatever the Critic hands me, multiplied by
        the log-probability of the line I just delivered. If the review comes back{' '}
        <em>better than expected</em>, I lean into that choice. If it comes back <em>worse</em>,
        I lean out. I don&apos;t have opinions about the set. I only have opinions about my
        actions, conditioned on whichever stage I find myself on.
      </Personify>

      {/* ── Advantage + GAE math: the critic's review ───────────── */}
      <Prose>
        <p>
          Unpack the critic&apos;s review. What we <em>really</em> want is{' '}
          <code>A(s, a) = Q(s, a) − V(s)</code> — how much better was this particular action
          than the average action from the same scene. We can&apos;t compute{' '}
          <code>Q(s, a)</code> directly without replaying the whole performance, but we can
          estimate it with a <KeyTerm>one-step bootstrap</KeyTerm>: take the immediate
          reaction plus the discounted value the critic assigns to the next stage. Plug in and
          simplify.
        </p>
      </Prose>

      <MathBlock caption="advantage via one-step bootstrapping">
{`A(s, a)   =   Q(s, a)   −   V(s)

           ≈  ( r + γ · V_φ(s') )   −   V_φ(s)            Q estimated by 1-step TD

           =    r + γ · V_φ(s')   −   V_φ(s)              =: δ, the TD error`}
      </MathBlock>

      <Prose>
        <p>
          This is the review in mathematical form: <em>that scene was better than average by
          δ</em>. Low-variance (one step of the real performance, most of the signal comes
          from the critic&apos;s forecast) but biased — <code>V_φ</code> is only an
          approximation and any miscalibration in it contaminates the grade. At the other
          extreme, the pure Monte-Carlo advantage waits for the curtain and uses the full
          episode return instead of <code>r + γV(s&apos;)</code> — unbiased, but noisy. The
          obvious question: can we interpolate between &ldquo;grade the scene right now&rdquo;
          and &ldquo;wait for the reviews in the morning paper&rdquo;?
        </p>
        <p>
          Yes. That&apos;s <KeyTerm>Generalized Advantage Estimation</KeyTerm>. Schulman et
          al. (2015) wrote down a clean geometric sum over multi-step TD errors with a decay
          parameter <code>λ</code>.
        </p>
      </Prose>

      <MathBlock caption="GAE — interpolating between TD and Monte Carlo">
{`δ_t   =   r_t + γ · V_φ(s_{t+1}) − V_φ(s_t)                    single-step TD error

A_t^GAE(γ, λ)   =   Σ (γλ)^k · δ_{t+k}
                    k=0

                =   δ_t  +  γλ · δ_{t+1}  +  (γλ)² · δ_{t+2}  +  …

                     λ = 0   →   A_t = δ_t                    pure TD, low var, biased
                     λ = 1   →   A_t = G_t − V(s_t)            pure Monte Carlo, high var
                     λ = 0.95  ≈  most PPO / A2C code          sweet spot`}
      </MathBlock>

      <Prose>
        <p>
          <code>λ</code> is the dial between &ldquo;trust the critic&rdquo; and &ldquo;trust
          the audience.&rdquo; At <code>λ = 0</code> the actor takes the critic&apos;s one-step
          grade at face value after every scene (biased but smooth). At <code>λ = 1</code> the
          actor ignores mid-performance reviews and reads the morning paper (unbiased but
          noisy). In practice <code>λ ≈ 0.95</code> is what everybody ships — most of the
          variance reduction, most of the time, with bias kept tolerable.
        </p>
      </Prose>

      {/* ── Widget 2: the critic's effect on training ───────────── */}
      <A2CTraining />

      <Prose>
        <p>
          Two learning curves on CartPole. Vanilla REINFORCE on the left — the actor
          performing blind, total reward climbing but with spikes and reversals every handful
          of episodes, every scene graded only at the end. A2C on the right — same
          environment, same policy architecture, but now there&apos;s a critic in the room
          delivering real-time reviews. Smoother. Faster. The gap between the two curves is{' '}
          <em>variance reduction from a competent critic</em>, full stop. No new policy-gradient
          theorem, no fancier optimizer. Just: hire the critic, listen to the review, move on.
        </p>
      </Prose>

      <Callout variant="insight" title="why A2C isn't cheating">
        You might worry: if we train the actor with a signal that depends on the critic&apos;s
        current (imperfect) review of <code>V(s)</code>, aren&apos;t we introducing bias?
        Slightly, yes — but the baseline <em>itself</em> doesn&apos;t bias the policy gradient
        at all (any function of state only leaves the expectation unchanged). What introduces
        bias is using <code>V_φ(s&apos;)</code> to bootstrap the return. That bias is the
        price we pay for variance reduction, and GAE&apos;s <code>λ</code> is the knob that
        sets the rate.
      </Callout>

      <Personify speaker="Critic">
        I don&apos;t perform. I don&apos;t pick lines. I just sit in the back of the house and
        put a number on the stage. My loss is boring — MSE between my guess and a bootstrapped
        target. But every advantage the Actor sees flows through me, so if my reviews are
        sharp, the Actor&apos;s gradients are clean. If I&apos;m miscalibrated, the Actor
        rehearses lies. I&apos;m the quieter of the two, but I set the signal-to-noise ratio
        for the whole production.
      </Personify>

      {/* ── Both learning simultaneously ────────────────────────── */}
      <Prose>
        <p>
          This is the part that feels paradoxical the first time you see it: both networks are
          learning <em>at the same time</em>. The actor is adjusting its performance based on
          reviews from a critic who is still figuring out how to review. The critic is
          calibrating its reviews against a performance that keeps changing. Neither one ever
          converges to a fixed target — they&apos;re both moving, and each one&apos;s movement
          reshapes the other&apos;s gradient. It should spiral into nonsense, and in some
          implementations (wrong learning rates, no detach, terminal-state bugs) it does. In
          practice, when you wire it carefully, the two processes <em>co-evolve</em>: the
          actor&apos;s performances give the critic cleaner targets, the critic&apos;s
          sharpening reviews give the actor cleaner gradients, and the whole system settles
          into a working rehearsal room. That dance — two networks training each other on the
          same trajectory — is the signature of every actor-critic method.
        </p>
      </Prose>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations of the same rehearsal on CartPole. Pure Python first — two
          tiny linear heads, actor and critic, trained with manual gradients so you can see
          every term. Then NumPy with GAE. Then a full PyTorch A2C that would pass for a
          minimal library implementation.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python actor-critic · ac_scratch.py"
        output={`ep  50 | return  42.3
ep 100 | return  87.6
ep 150 | return 162.4
ep 200 | return 198.1`}
      >{`import random, math
import gymnasium as gym

env = gym.make("CartPole-v1")
obs_dim, n_act, gamma = 4, 2, 0.99

# actor: logits over 2 actions;  critic: scalar V(s).  both linear.
W_pi = [[0.0]*obs_dim for _ in range(n_act)]
W_v  = [0.0]*obs_dim
lr_pi, lr_v = 1e-2, 5e-3

def softmax(z):
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
    return [x / s for x in e]

def policy(s):
    logits = [sum(W_pi[a][i]*s[i] for i in range(obs_dim)) for a in range(n_act)]
    return softmax(logits)

def value(s):
    return sum(W_v[i]*s[i] for i in range(obs_dim))

for ep in range(200):
    s, _ = env.reset()
    done, G = False, 0.0
    while not done:
        p = policy(s)
        a = 0 if random.random() < p[0] else 1
        s2, r, term, trunc, _ = env.step(a); done = term or trunc
        G += r

        # one-step advantage:   A = r + γV(s') − V(s)    (zero the bootstrap if terminal)
        v_s, v_s2 = value(s), 0.0 if done else value(s2)
        adv = r + gamma * v_s2 - v_s

        # critic:  φ ← φ + α · adv · ∇V(s)          (∇V(s) = s, since V is linear)
        for i in range(obs_dim):
            W_v[i] += lr_v * adv * s[i]

        # actor:   θ ← θ + α · adv · ∇log π(a|s)    (softmax grad: (1_a − p) · s)
        for act in range(n_act):
            grad = ((1.0 if act == a else 0.0) - p[act])
            for i in range(obs_dim):
                W_pi[act][i] += lr_pi * adv * grad * s[i]
        s = s2
    if (ep + 1) % 50 == 0:
        print(f"ep {ep+1:3d} | return {G:6.1f}")`}</CodeBlock>

      <Prose>
        <p>
          Even at this size it learns. The actor is two lines of softmax; the critic is a dot
          product. They share a state and a scene-by-scene review and that&apos;s enough. Now
          NumPy, with the upgrade to GAE — collect a rollout, compute the λ-weighted
          advantage, batch the update. This is the shape of every A2C/PPO training loop
          you&apos;ll ever read.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy + GAE · ac_gae_numpy.py"
      >{`import numpy as np
import gymnasium as gym

env = gym.make("CartPole-v1")
OBS, ACT, GAMMA, LAM = 4, 2, 0.99, 0.95

W_pi = np.zeros((ACT, OBS))
W_v  = np.zeros(OBS)

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)

def compute_gae(rewards, values, dones, last_val):
    """δ_t = r_t + γV_{t+1}(1-d) − V_t ;  A_t = δ_t + γλ(1-d)·A_{t+1}"""
    T = len(rewards)
    adv = np.zeros(T); gae = 0.0
    values = np.append(values, last_val)                   # pad with bootstrap
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * values[t+1] * nonterm - values[t]
        gae   = delta + GAMMA * LAM * nonterm * gae
        adv[t] = gae
    returns = adv + values[:-1]                            # target for critic
    return adv, returns

# one update cycle after collecting an N-step rollout
def update(states, actions, advs, returns, lr_pi=1e-2, lr_v=5e-3):
    global W_pi, W_v
    # critic: MSE against bootstrapped returns
    preds = states @ W_v
    W_v += lr_v * ((returns - preds)[:, None] * states).mean(0)
    # actor: policy gradient with GAE advantage (normalize — the field-standard trick)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    probs = softmax(states @ W_pi.T)                       # (N, ACT)
    onehot = np.eye(ACT)[actions]
    grad_logp = onehot - probs                             # ∇log π for softmax
    W_pi += lr_pi * (advs[:, None, None] * grad_logp[:, :, None] * states[:, None, :]).mean(0)`}</CodeBlock>

      <Prose>
        <p>
          Two things to notice. <code>compute_gae</code> runs <em>backward</em> through the
          rollout — each <code>A_t</code> depends on <code>A_{`{t+1}`}</code>, so you sweep
          right-to-left and accumulate. It&apos;s the critic reading the performance in
          reverse, from the last scene outward, so each scene&apos;s review can borrow from
          the future. And we normalize advantages before the actor update:{' '}
          <code>(A − mean) / std</code>. This isn&apos;t in the math — it&apos;s an empirical
          stabilizer that the entire field ships. Don&apos;t fight it.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch A2C · a2c_torch.py"
      >{`import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_act, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden),  nn.Tanh())
        self.actor  = nn.Linear(hidden, n_act)   # logits
        self.critic = nn.Linear(hidden, 1)       # V(s)

    def forward(self, s):
        h = self.shared(s)
        return self.actor(h), self.critic(h).squeeze(-1)

env = gym.make("CartPole-v1")
net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
GAMMA, LAM, N_STEPS = 0.99, 0.95, 2048

def rollout():
    obs, _ = env.reset()
    S, A, R, D, LP, V = [], [], [], [], [], []
    for _ in range(N_STEPS):
        s_t = torch.as_tensor(obs, dtype=torch.float32)
        logits, v = net(s_t)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        nxt, r, term, trunc, _ = env.step(a.item()); done = term or trunc
        S.append(s_t); A.append(a); R.append(r); D.append(float(done))
        LP.append(dist.log_prob(a)); V.append(v)                     # both tracked
        obs = env.reset()[0] if done else nxt
    _, last_v = net(torch.as_tensor(obs, dtype=torch.float32))
    return S, A, R, D, LP, V, last_v

def compute_gae(R, D, V, last_v):
    adv, gae = [0.0]*len(R), 0.0
    V = V + [last_v]
    for t in reversed(range(len(R))):
        nt = 1.0 - D[t]
        delta = R[t] + GAMMA * V[t+1].detach() * nt - V[t].detach()  # detach! critic is a
        gae   = delta + GAMMA * LAM * nt * gae                       # target here, not a fn
        adv[t] = gae
    return torch.stack([torch.tensor(a) for a in adv]).float()

for it in range(500):
    S, A, R, D, LP, V, last_v = rollout()
    adv = compute_gae(R, D, V, last_v)
    returns = adv + torch.stack(V).detach()                          # critic target
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    log_probs = torch.stack(LP)
    values    = torch.stack(V)

    actor_loss  = -(log_probs * adv).mean()                          # gradient ascent → −
    critic_loss = F.mse_loss(values, returns)
    entropy     = -torch.stack([
        torch.distributions.Categorical(logits=net(s)[0]).entropy() for s in S
    ]).mean()                                                        # bonus: explore

    loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy
    opt.zero_grad(); loss.backward(); opt.step()`}</CodeBlock>

      <Bridge
        label="scratch → numpy → pytorch"
        rows={[
          {
            left: 'manual log-softmax gradient: (1_a − p) · s',
            right: 'dist.log_prob(a); loss.backward()',
            note: 'autograd handles the policy gradient — you only write the loss',
          },
          {
            left: 'hand-rolled V = w·s',
            right: 'self.critic = nn.Linear(hidden, 1)',
            note: 'critic is just a second head on a shared trunk — one network, two outputs',
          },
          {
            left: 'adv = r + γV(s\') − V(s)',
            right: 'compute_gae(R, D, V, last_v)',
            note: 'swap one-step TD for λ-weighted multi-step — same advantage role',
          },
          {
            left: 'separate lr_pi and lr_v',
            right: 'actor_loss + 0.5·critic_loss + 0.01·entropy',
            note: 'one optimizer, weighted sum of three losses — standard A2C recipe',
          },
        ]}
      />

      <Callout variant="note" title="A2C vs A3C — the asynchronous detour">
        The original Mnih et al. 2016 paper was <strong>A3C</strong>: Asynchronous Advantage
        Actor-Critic. Multiple worker processes each run their own copy of the environment and
        the network, compute gradients locally, and push them into a shared parameter
        server — no batching, no waiting. In 2016 this was revolutionary; GPUs hadn&apos;t
        taken over RL yet, and CPU-parallel A3C was faster than single-GPU alternatives.{' '}
        <strong>A2C</strong> — the synchronous version, where all workers step in lockstep and
        their experiences are batched into one big gradient update — came shortly after. On a
        GPU, A2C is simpler, equally sample-efficient, and easier to debug. Everyone ships
        A2C now. A3C is mostly of historical interest.
      </Callout>

      <Callout variant="insight" title="the lineage: A2C → PPO">
        Look at the A2C loss above. Three terms: policy gradient, value MSE, entropy bonus.
        That&apos;s 95% of PPO. PPO adds exactly two things: (a) run <em>multiple epochs</em>{' '}
        of SGD on the same batch instead of one, and (b) clip the importance-sampled policy
        ratio so the actor can&apos;t drift too far from the performance the critic was
        grading. That&apos;s it. If you understand A2C, you understand PPO structurally — the
        next lesson just fills in the clipping and the sample reuse.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Using the TD target for the actor:</strong> the
          actor&apos;s gradient multiplies <code>∇log π</code> by the <em>advantage</em>, not
          by the raw target <code>r + γV(s&apos;)</code>. If you pass in the target, every
          action at a good stage looks good — the policy gradient becomes &ldquo;do more of
          whatever you did in scenes with high value,&rdquo; which is noise. Subtract{' '}
          <code>V(s)</code>. Always.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting to detach the critic for the actor:</strong>{' '}
          when computing <code>A = r + γV(s&apos;) − V(s)</code> and feeding <code>A</code>{' '}
          into the actor loss, you must <code>.detach()</code> it (or its components).
          Otherwise the actor loss&apos;s backward pass flows into the critic too, and
          you&apos;re training the critic to make the advantage small — the opposite of what
          you want. From the actor&apos;s perspective the critic is a <em>review</em>, not a
          learnable quantity.
        </p>
        <p>
          <strong className="text-term-amber">Wrong sign:</strong> gradient ascent on{' '}
          <code>log π(a|s) · A</code> is gradient descent on <code>−log π(a|s) · A</code>. In
          PyTorch you minimize losses, so the actor loss has a <em>minus</em>. Flip the sign
          accidentally and your actor rehearses how to lose.
        </p>
        <p>
          <strong className="text-term-amber">Bootstrapping past a terminal state:</strong>{' '}
          at the end of an episode, <code>V(s&apos;)</code> is conceptually zero — the
          curtain came down, there is no next scene. If you forget the <code>(1 − done)</code>{' '}
          mask in <code>r + γ(1−d)V(s&apos;) − V(s)</code>, you&apos;ll feed in the value of
          the <em>new</em> episode&apos;s opening scene as if it belonged to the previous
          performance. Invisible bug, catastrophic in practice.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="A2C on CartPole, then GAE on top">
        <p>
          Start from a working REINFORCE implementation on <code>CartPole-v1</code> (you have
          one from the last lesson). Add a second head to your network for <code>V(s)</code>{' '}
          — the critic&apos;s seat in the house — and a value-loss term to the optimizer. Use
          a one-step TD advantage: <code>A = r + γV(s&apos;)(1−d) − V(s)</code>, detached from
          the critic graph when it feeds into the actor loss.
        </p>
        <p className="mt-2">
          Plot REINFORCE vs A2C on the same axes, averaged over 5 seeds. You should see A2C
          solve CartPole (reward &ge; 195) in roughly half the episodes, with visibly lower
          variance between seeds — the actor getting real-time reviews instead of waiting for
          the curtain.
        </p>
        <p className="mt-2">
          Then upgrade to GAE with <code>λ = 0.95</code>: collect a 2048-step rollout, compute
          advantages backward through the trajectory, normalize them, and apply one big
          batched update. Compare three curves now — REINFORCE, A2C (1-step), A2C + GAE. The
          spacing between them is the whole story of variance reduction.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: set <code>λ = 0</code> and <code>λ = 1</code> explicitly and re-run. You
          should see <code>λ = 0</code> learn fastest but hit a lower ceiling (critic bias),{' '}
          <code>λ = 1</code> match Monte-Carlo noise, and <code>λ = 0.95</code> comfortably
          win.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> Every serious policy-gradient algorithm
          after REINFORCE has a critic. The critic&apos;s job is to be a good reviewer; the
          actor&apos;s job is to move in the direction the critic points. The advantage —{' '}
          <code>r + γV(s&apos;) − V(s)</code> — is the shared scalar they negotiate over, and
          GAE gives you a knob (<code>λ</code>) to trade bias against variance in how
          aggressively you trust the review. Normalize your advantages. Detach your critic
          outputs when they feed the actor. Mask terminal bootstraps. Those three lines of
          discipline separate a working production from a silently broken one.
        </p>
        <p>
          <strong>Next up — Proximal Policy Optimization.</strong> A2C has one remaining
          fragility: a single bad batch can blow up the policy, because there&apos;s nothing
          stopping the actor from making a huge step away from the performance the critic was
          grading. The critic&apos;s reviews become stale instantly; the actor wanders into
          scenes nobody has evaluated; the training spirals. PPO fixes that with a clipped
          importance ratio that lets you safely take <em>multiple</em> gradient steps on the
          same batch without drifting too far from the old policy. It&apos;s the algorithm
          behind ChatGPT&apos;s RLHF, the default baseline in robotics, and — structurally —
          nothing more than A2C plus a <code>min(ratio · A, clip(ratio) · A)</code>. You
          already have the bones. After PPO, the curriculum leaves learning behind and turns
          to <strong>inference and serving</strong>: once you have a trained model, how do
          you ship it cheaply and fast? Opens with <em>Quantization Basics</em> — the first
          of the (almost) free lunches.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Actor-Critic Algorithms',
            author: 'Konda, Tsitsiklis',
            venue: 'NeurIPS 2000 — the original formalization',
            url: 'https://proceedings.neurips.cc/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html',
          },
          {
            title: 'Asynchronous Methods for Deep Reinforcement Learning',
            author: 'Mnih et al.',
            venue: 'ICML 2016 — the A3C paper',
            url: 'https://arxiv.org/abs/1602.01783',
          },
          {
            title: 'High-Dimensional Continuous Control Using Generalized Advantage Estimation',
            author: 'Schulman, Moritz, Levine, Jordan, Abbeel',
            venue: 'ICLR 2016 — GAE',
            url: 'https://arxiv.org/abs/1506.02438',
          },
          {
            title: 'Reinforcement Learning: An Introduction (2nd ed.)',
            author: 'Sutton, Barto',
            venue: 'Chapter 13 — Policy Gradient Methods',
            url: 'http://incompleteideas.net/book/the-book-2nd.html',
          },
        ]}
      />
    </div>
  )
}
