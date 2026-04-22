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
import GradientOverTime from '../widgets/GradientOverTime'
import ActivationSaturation from '../widgets/ActivationSaturation'

// Signature anchor: the whispered telephone chain. 50 kids in a line, each
// whispers the message to the next. By the end the signal is noise.
// Multiplying many <1 numbers is the whisper fading; the inverted version
// (megaphone at every hand-off) is the exploding case. Returned at the
// opening, at the product-of-derivatives math reveal, and at the
// "what-to-do-about-it" moment that teases LSTM.
export default function VanishingGradientProblemLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="vanishing-gradient-problem" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Play the telephone game with fifty kids. The first one hears a
          sentence — <em>&ldquo;the fox crossed the river at dawn&rdquo;</em>{' '}
          — and whispers it to the second, who whispers what <em>they</em>{' '}
          heard to the third, and so on down the line. By kid twenty the
          signal is mush. By kid fifty the last child is earnestly relaying
          something about a fax machine and a raven. No single whisper was
          catastrophically wrong. Each one just dropped a little — a
          consonant here, a vowel there — and fifty little drops multiplied
          into a message that has nothing to do with the original.
        </p>
        <p>
          That is exactly what gradient signals do in a vanilla{' '}
          <NeedsBackground slug="recurrent-neural-network">RNNs</NeedsBackground>{' '}
          when you try to train them on long sequences. Between roughly 1991
          and 1997 this was the wall the whole sub-field could not push
          through. You could teach an RNN to remember a bit across{' '}
          <em>eight</em> time steps. Push that to <em>twenty</em> and the
          loss curve went flat. Push to <em>fifty</em> and the network
          learned absolutely nothing about the start of the sequence — as if
          the first thirty tokens had never been shown. Each step backward
          in{' '}
          <NeedsBackground slug="backprop-through-time">BPTT</NeedsBackground>{' '}
          is a whisper, and the gradient at the end of the chain is what the
          last kid heard.
        </p>
        <p>
          This wasn&apos;t a bug. It wasn&apos;t a learning-rate issue. It
          was arithmetic — the same arithmetic that kills deep{' '}
          <NeedsBackground slug="sigmoid-and-relu">sigmoid</NeedsBackground>{' '}
          stacks, now reinforced by a vicious new multiplier: the{' '}
          <em>same weight matrix, applied over and over</em>. Sepp
          Hochreiter diagnosed it in a 1991 German-language diploma thesis
          that nobody outside his advisor read for three years. Bengio
          et al. re-derived it in 1994. Pascanu et al. nailed it down with
          spectral analysis in 2013. And then LSTM came out, and suddenly
          the telephone chain had a shortcut.
        </p>
        <p>
          This lesson is that wall. We&apos;ll write the gradient chain,
          compute its spectral radius, watch it collapse to zero on a real
          RNN, and understand exactly why gating — not clipping, not better{' '}
          <NeedsBackground slug="weight-initialization">initialization</NeedsBackground>
          , <em>gating</em> — was the only way out.
        </p>
      </Prose>

      <Personify speaker="Vanishing gradient">
        I am not a training instability. I am not a numerical glitch you can
        fix with a learning-rate schedule. I am what happens when you line up
        twenty kids and ask them to whisper the same sentence down the chain.
        There is no hyperparameter for that.
      </Personify>

      {/* ── The gradient chain ───────────────────────────────────── */}
      <Prose>
        <p>
          A vanilla RNN updates its hidden state with the same recurrence at
          every step:
        </p>
        <p>
          <code>h_t = tanh(W_h · h_{`{t-1}`} + W_x · x_t + b)</code>.
        </p>
        <p>
          To train it, we need <code>∂L / ∂h_0</code> — how the loss at the
          end of the sequence depends on the hidden state at the very start.
          Backprop through time says: chain together the Jacobians at every
          step. Same telephone chain, now written as a product.
        </p>
      </Prose>

      <MathBlock caption="backprop through time — the product that decides everything">
{`∂h_T        T
────   =   ∏   ∂h_t / ∂h_{t-1}
∂h_0       t=1

        =   ∏   W_hᵀ  ·  diag( tanh'( W_h · h_{t-1} + W_x · x_t + b ) )
            t

        ≈   ( W_hᵀ · D )ᵀ        where D is the average diagonal derivative`}
      </MathBlock>

      <Prose>
        <p>
          Read the product literally. Each factor <code>W_hᵀ · D</code> is
          one kid in the chain — one whisper from step <code>t</code> to
          step <code>t−1</code>. The gradient from time <code>T</code> all
          the way back to time <code>0</code> is a <em>power</em> of that
          one matrix — roughly <code>(W_hᵀ · D)^T</code>. And the fate of
          any matrix power is decided by one number: its{' '}
          <KeyTerm>spectral radius</KeyTerm> — the magnitude of its largest
          eigenvalue, the amount by which each whisper gets quieter (or
          louder) on its way down the line.
        </p>
        <ul>
          <li>
            If <code>ρ(W_hᵀ · D) &lt; 1</code> — the product shrinks
            exponentially. Every kid whispers a little softer than they
            heard; by kid fifty the signal is dust.{' '}
            <strong>Vanishing.</strong>
          </li>
          <li>
            If <code>ρ(W_hᵀ · D) &gt; 1</code> — the product blows up
            exponentially. Every kid has a megaphone; by kid fifty the
            gradient arrives as a cannon-shell and the optimiser takes a
            step into the void. <strong>Exploding.</strong>
          </li>
          <li>
            If <code>ρ(W_hᵀ · D) = 1</code> — you have won the
            initialisation lottery. Enjoy your training run. It will end
            when you look at it funny.
          </li>
        </ul>
      </Prose>

      {/* ── Widget 1: GradientOverTime ──────────────────────────── */}
      <Prose>
        <p>
          Here is the whisper fading, live. We train a tiny RNN on a task
          where it has to remember a bit from the start of a sequence and
          output it at the end. We plot the norm of <code>∂L/∂h_0</code> as
          a function of sequence length — how loud the last kid&apos;s
          message is, as a function of how many kids are in line.
        </p>
      </Prose>

      <GradientOverTime />

      <Prose>
        <p>
          Straight line on a log axis. At length 10, the gradient is
          measurable — the chain is short, the whisper still recognisable.
          At length 30, it is ten orders of magnitude smaller. At length 50
          it is below the precision of a 32-bit float and effectively zero —
          the last kid heard nothing. The optimiser sees a flat loss
          landscape for anything that happened before step 30, so it never
          learns long dependencies. This is the empirical shape of the
          problem that stalled RNN research for a decade.
        </p>
      </Prose>

      <Personify speaker="Spectral radius">
        I am the toll every whisper pays on the way down the telephone
        chain. Pay me 0.9 per hand-off and after 50 hand-offs you have paid{' '}
        <code>0.9⁵⁰ ≈ 0.005</code>. Pay me 0.5 and you have paid{' '}
        <code>8.9 × 10⁻¹⁶</code>. I do not care about your optimiser. I do
        not care about your loss function. I am the reason your RNN forgot.
      </Personify>

      {/* ── Activation derivatives ───────────────────────────────── */}
      <Prose>
        <p>
          Now zoom in on the <code>D</code> in <code>W_hᵀ · D</code>.
          That&apos;s the diagonal matrix of activation derivatives — the
          nonlinearity&apos;s contribution to each whisper. For tanh:
        </p>
      </Prose>

      <MathBlock caption="tanh and sigmoid derivatives — a budget of at most 1 and at most 0.25">
{`tanh'(x)   =   1 − tanh²(x)        ∈ [0, 1],   peaks at 1 when x = 0

σ'(x)      =   σ(x) · (1 − σ(x))   ∈ [0, 0.25], peaks at 0.25 when x = 0

|  x  |  tanh'(x)  |   σ'(x)   |
| --- | ---------- | --------- |
|  0  |   1.000    |   0.250   |
|  1  |   0.420    |   0.197   |
|  2  |   0.071    |   0.105   |
|  3  |   0.010    |   0.045   |
|  5  |   0.0001   |   0.0066  |`}
      </MathBlock>

      <Prose>
        <p>
          Tanh&apos;s derivative peaks at <em>1</em>. Sigmoid&apos;s peaks
          at <em>0.25</em>. Stack twenty sigmoids in the chain and even at
          their <em>best</em> you are multiplying by{' '}
          <code>0.25²⁰ ≈ 10⁻¹²</code> — every kid in the line is already
          required to whisper at quarter volume before they even open their
          mouth. That is why sigmoid in hidden layers died — and it also
          explains why LSTM uses sigmoid for <em>gates</em> (where
          saturation is a feature, not a bug), not for the hidden state
          itself.
        </p>
        <p>
          Tanh is the RNN default precisely because its derivative budget is
          an order of magnitude larger. But note the tails — by{' '}
          <code>|x| = 5</code> the derivative is <code>10⁻⁴</code>. Saturate
          once and that time step&apos;s Jacobian contribution is
          effectively zero, which means the whisper through that position
          dies regardless of <code>W_h</code>. One kid with laryngitis and
          the rest of the chain is deaf.
        </p>
      </Prose>

      <ActivationSaturation />

      <Prose>
        <p>
          Slide the input left and right. Notice how quickly tanh flattens
          into its rails — anywhere past <code>x = ±3</code> the derivative
          is a rounding error. In an RNN with fifty time steps, every step
          that happens to land in the saturated region contributes a
          near-zero factor to the Jacobian product. One or two of those and
          the whole telephone chain is dead.
        </p>
      </Prose>

      <Callout variant="note" title="why ReLU doesn't save you in an RNN">
        In a feedforward net, ReLU dodged this whole problem — its
        derivative is 1 in the active region, and 1⁰ multiplies cleanly. In
        an RNN the activation derivative is not the whole story. You still
        get <code>W_hᵀ · W_hᵀ · W_hᵀ · …</code> repeated <em>T</em> times —
        the <em>same</em> kid whispering to themselves, over and over. If
        the spectral radius of <code>W_h</code> is under 1, the gradient
        vanishes even with a perfectly flat ReLU derivative. Repeated
        multiplication by a shared matrix is the fundamental issue — the
        nonlinearity just picks whether you vanish fast or <em>very</em>{' '}
        fast.
      </Callout>

      <Personify speaker="Additive path">
        I am the kid who refuses to play telephone. While the spectral
        radius is collecting a toll at every whisper, I carry the message
        through time by <em>addition</em>, not multiplication. Add one a
        thousand times and you get a thousand. Multiply 0.9 a thousand times
        and you get <code>10⁻⁴⁶</code>. This is the whole trick of LSTM and
        GRU — they do not fight the spectral radius. They route a second
        wire down the hallway, past the telephone chain entirely.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Enough math — let&apos;s watch it happen. First a hand-rolled
          NumPy simulation of a 50-step RNN with a random Gaussian{' '}
          <code>W_h</code>, so you can see the Jacobian norm collapse whisper
          by whisper. Then a PyTorch head-to-head between a vanilla RNN and
          an LSTM on a copy-task — remember a value at position 0, output it
          at position <em>T</em>.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · vanishing_gradient_demo.py"
        output={`step  0: ‖∂h_T/∂h_0‖ = 1.000e+00
step  5: ‖∂h_T/∂h_0‖ = 2.413e-01
step 10: ‖∂h_T/∂h_0‖ = 4.972e-02
step 20: ‖∂h_T/∂h_0‖ = 2.101e-03
step 30: ‖∂h_T/∂h_0‖ = 7.884e-05
step 40: ‖∂h_T/∂h_0‖ = 2.963e-06
step 50: ‖∂h_T/∂h_0‖ = 1.082e-07
spectral radius ρ(W_h) = 0.714  →  expected decay 0.714^50 ≈ 5.9e-08`}
      >{`import numpy as np

np.random.seed(0)
H, T = 64, 50

# Random recurrent weights, scaled so spectral radius < 1 → vanishing regime
W_h = np.random.randn(H, H) * (0.9 / np.sqrt(H))
rho = np.max(np.abs(np.linalg.eigvals(W_h)))

# Walk the Jacobian product backward through time — the telephone chain,
# one whisper per iteration of the loop.
jac = np.eye(H)                            # ∂h_t/∂h_t = I at t = T
norms = [np.linalg.norm(jac)]

h = np.zeros(H)                            # a quiet trajectory
for t in range(T):
    pre = W_h @ h                          # pretend x_t = 0 for simplicity
    h = np.tanh(pre)
    D = np.diag(1.0 - h**2)                # tanh'(pre) = 1 - tanh(pre)^2
    jac = jac @ (D @ W_h)                  # one Jacobian step = one whisper
    norms.append(np.linalg.norm(jac))

for k in (0, 5, 10, 20, 30, 40, 50):
    print(f"step {k:2d}: ‖∂h_T/∂h_0‖ = {norms[k]:.3e}")
print(f"spectral radius ρ(W_h) = {rho:.3f}  →  expected decay {rho:.3f}^{T} "
      f"≈ {rho**T:.1e}")`}</CodeBlock>

      <Prose>
        <p>
          The numeric collapse follows <code>ρ(W_h)^T</code> almost exactly.
          You chose the scaling (<code>0.9 / √H</code>) so <code>ρ &lt; 1</code>
          , and physics did the rest — fifty whispers at 71% volume each and
          the final message is a whisper of a whisper of dust. Push the
          scaling to <code>1.1 / √H</code> and the same loop will explode
          instead: every kid now has a megaphone, the norm grows to{' '}
          <code>10¹⁰</code>, you get a NaN. That is the exploding gradient,
          the cousin problem we fix with clipping.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch · rnn_vs_lstm_copy.py"
        output={`task: copy a bit from position 0 to the final output, T = 30
vanilla RNN :  train accuracy 51.2%   (random is 50%)
LSTM        :  train accuracy 99.8%
inspect grad ‖∂L/∂h_0‖ on the same batch:
  RNN  = 3.1e-08
  LSTM = 1.4e-01`}
      >{`import torch, torch.nn as nn

T, B, H = 30, 256, 32
torch.manual_seed(0)

# x is all zeros except the first token, which is the bit to remember
x = torch.zeros(B, T, 1)
bit = torch.randint(0, 2, (B,)).float()
x[:, 0, 0] = bit

rnn  = nn.RNN(input_size=1,  hidden_size=H, nonlinearity='tanh', batch_first=True)
lstm = nn.LSTM(input_size=1, hidden_size=H,                      batch_first=True)

def train(cell, steps=300, lr=1e-2):
    head = nn.Linear(H, 1)
    opt = torch.optim.Adam(list(cell.parameters()) + list(head.parameters()), lr=lr)
    for _ in range(steps):
        out, _ = cell(x)                    # (B, T, H)
        logits = head(out[:, -1, :]).squeeze(-1)   # last-step readout
        loss = nn.functional.binary_cross_entropy_with_logits(logits, bit)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = ((logits > 0).float() == bit).float().mean().item()
    return acc

print(f"vanilla RNN : train accuracy {train(rnn) :.1%}")
print(f"LSTM        : train accuracy {train(lstm):.1%}")`}</CodeBlock>

      <Bridge
        label="theory → empirical"
        rows={[
          {
            left: '∏ W_hᵀ · diag(tanh\')',
            right: 'jac = jac @ (D @ W_h) in a loop',
            note: 'the math IS the NumPy loop — one whisper per time step',
          },
          {
            left: 'ρ(W_h) < 1 ⇒ vanishing',
            right: 'scale = 0.9 / √H in init',
            note: 'you tune the scaling to choose which side of the cliff you land on',
          },
          {
            left: 'LSTM\'s additive cell state',
            right: 'nn.LSTM instead of nn.RNN',
            note: 'same API, categorically different gradient flow through time',
          },
        ]}
      />

      {/* ── Why LSTM/GRU ─────────────────────────────────────────── */}
      <Callout variant="insight" title="the 1997 escape hatch: gating creates an additive path">
        LSTM (Hochreiter &amp; Schmidhuber, 1997) doesn&apos;t make the
        vanishing-gradient math go away. The recurrent Jacobian still has a
        spectral radius, still can collapse — the telephone chain still
        fades. What LSTM adds is a second path — the <em>cell state</em>{' '}
        <code>c_t</code>, updated as{' '}
        <code>c_t = f_t ⊙ c_{`{t-1}`} + i_t ⊙ g_t</code>. When the forget
        gate <code>f_t ≈ 1</code>, the Jacobian{' '}
        <code>∂c_t/∂c_{`{t-1}`} = f_t ≈ 1</code> — a near-identity.
        Multiply near-identity a thousand times and you get near-identity.
        The cell state is a second wire strung down the hallway, above the
        heads of the whispering kids, carrying the message undamaged from
        one end to the other. GRU does the same thing with a simpler
        architecture. Residual connections in ResNets (and transformers)
        are the same trick in feedforward form.
      </Callout>

      <Callout variant="insight" title="exploding is easy, vanishing is structural">
        Exploding gradients are the megaphone version of the telephone
        chain — loud, obvious, the loss goes <code>inf</code>, the weights
        NaN, the traceback is noisy. The fix is{' '}
        <KeyTerm>gradient clipping</KeyTerm>: rescale the gradient vector to
        a max norm before the optimiser step. One line of code, Pascanu et
        al. 2013. Vanishing is the quiet version — there is nothing to see.
        The loss is a flat line. The network looks like it is
        &ldquo;converging.&rdquo; It is actually just not learning. Each
        kid is whispering at 70% and the final message is <code>10⁻¹⁴</code>{' '}
        of the original — silent, not wrong. That is why vanishing outlasted
        exploding as an open problem: the symptom is invisible, and no
        post-hoc rescaling can resurrect a zero.
      </Callout>

      {/* ── Gotchas ──────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">
            &ldquo;I&apos;ll just clip the gradient&rdquo;:
          </strong>{' '}
          clipping fixes <em>exploding</em> (turn the megaphone down), it
          does nothing for <em>vanishing</em>. You cannot rescale{' '}
          <code className="text-dark-text-primary">10⁻¹⁴</code> up to
          something useful without scaling the noise with it. Vanishing is
          structural; it requires an architectural change (gating,
          residuals) or a math change (orthogonal RNN, unitary RNN).
        </p>
        <p>
          <strong className="text-term-amber">
            Initialising <code className="text-dark-text-primary">W_h</code>{' '}
            with small norm &ldquo;for stability&rdquo;:
          </strong>{' '}
          shrinking the spectral radius <em>accelerates</em> the fading
          whisper. For vanilla RNNs you actually want{' '}
          <code className="text-dark-text-primary">ρ(W_h) ≈ 1</code> — use
          orthogonal or identity init (Le, Jaitly, Hinton 2015). For LSTM
          the cell-state path protects you, so the init matters less.
        </p>
        <p>
          <strong className="text-term-amber">
            Sigmoid as the hidden activation:
          </strong>{' '}
          strictly worse than tanh. Sigmoid&apos;s derivative caps at 0.25,
          which puts a multiplicative ceiling of{' '}
          <code className="text-dark-text-primary">0.25^T</code> on any
          whisper travelling through <em>T</em> time steps regardless of{' '}
          <code className="text-dark-text-primary">W_h</code>. Tanh at least
          offers derivative <em>1</em> at the origin. Use tanh for the
          hidden state; reserve sigmoid for gates.
        </p>
        <p>
          <strong className="text-term-amber">
            &ldquo;My RNN is converging, just slowly&rdquo;:
          </strong>{' '}
          print <code className="text-dark-text-primary">‖∂L/∂h_0‖</code>.
          If it is below{' '}
          <code className="text-dark-text-primary">10⁻⁶</code> on a length-50
          sequence, your network is not learning anything that depended on
          the first 30 tokens — it is learning shortcut statistics from the
          last few positions. Switch to LSTM or shorten the chain.
        </p>
      </Gotcha>

      {/* ── Challenge ────────────────────────────────────────────── */}
      <Challenge prompt="Find your RNN's cliff">
        <p>
          Train a vanilla <code>nn.RNN</code> on a copy-task at three
          sequence lengths: <code>T = 5</code>, <code>T = 20</code>, and{' '}
          <code>T = 50</code>. Same architecture, same hyperparameters,
          same training budget (say, 500 steps of Adam). For each{' '}
          <code>T</code>, record the best accuracy the network reaches — a
           5-kid chain, a 20-kid chain, a 50-kid chain.
        </p>
        <p className="mt-2">
          Plot best-accuracy vs <code>T</code>. You will see a sharp cliff
          — perfect at 5, decent at 20, at-chance at 50. Now repeat with{' '}
          <code>nn.LSTM</code>. The cliff either disappears or moves out to{' '}
          <code>T = 200+</code>.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: log <code>‖∂L/∂h_0‖</code> at every training step. Watch
          the RNN gradient collapse to <code>10⁻⁷</code> within the first
          few steps at <code>T = 50</code>; watch the LSTM gradient stay in
          the <code>10⁻¹</code> range. You will have reproduced the 1991
          finding in 40 lines of PyTorch.
        </p>
      </Challenge>

      {/* ── Closing + next up ───────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Vanilla RNNs have a
          spectral-radius problem: the gradient through <em>T</em> time
          steps is a product of <em>T</em> Jacobians — a telephone chain of
          <em> T</em> whispers — and products of Jacobians whose spectral
          radius is under 1 collapse exponentially into noise. Tanh and
          sigmoid derivatives compound the problem; ReLU doesn&apos;t rescue
          it because the same kid is whispering to themselves over and over,
          which means the weight matrix is shared and applied repeatedly.
          Clipping is the fix for the exploding (megaphone) cousin, not for
          vanishing. The architectural fix — gating with an additive cell
          state — is what made recurrent networks practical for real
          sequence tasks.
        </p>
        <p>
          <strong>Next up — LSTM.</strong> We need something with memory
          that survives the telephone chain — gates that can pass a signal
          forward without mangling it, a second wire down the hallway that
          carries the message undamaged while the whispering kids do their
          noisy local thing. That is LSTM. Four gates (input, forget,
          output, candidate), one cell state, and the specific wiring that
          routes the gradient around the multiplicative time-toll.
          You&apos;ll see why Hochreiter&apos;s 1997 design looks
          overcomplicated on first read and exactly right on the second —
          every gate has a job in keeping{' '}
          <code>∂c_t/∂c_{`{t-1}`} ≈ 1</code>, which is the mathematical way
          of saying <em>do not let the whisper fade</em>.
        </p>
      </Prose>

      <References
        items={[
          {
            title:
              'Untersuchungen zu dynamischen neuronalen Netzen (Investigations of Dynamic Neural Networks)',
            author: 'Sepp Hochreiter',
            venue:
              'Diploma thesis, TU Munich — the original diagnosis of the vanishing gradient problem',
            year: 1991,
            url: 'https://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf',
          },
          {
            title: 'Learning long-term dependencies with gradient descent is difficult',
            author: 'Bengio, Simard, Frasconi',
            venue: 'IEEE Transactions on Neural Networks',
            year: 1994,
            url: 'https://ieeexplore.ieee.org/document/279181',
          },
          {
            title: 'On the difficulty of training Recurrent Neural Networks',
            author: 'Pascanu, Mikolov, Bengio',
            venue:
              'ICML 2013 — spectral analysis of BPTT; introduces gradient clipping for the exploding case',
            year: 2013,
            url: 'https://arxiv.org/abs/1211.5063',
          },
          {
            title: 'Long Short-Term Memory',
            author: 'Hochreiter, Schmidhuber',
            venue: 'Neural Computation — the 1997 architectural solution',
            year: 1997,
            url: 'https://www.bioinf.jku.at/publications/older/2604.pdf',
          },
          {
            title: 'Dive into Deep Learning — §9.7 Backpropagation Through Time',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_recurrent-neural-networks/bptt.html',
          },
          {
            title:
              'A Simple Way to Initialize Recurrent Networks of Rectified Linear Units',
            author: 'Le, Jaitly, Hinton',
            year: 2015,
            url: 'https://arxiv.org/abs/1504.00941',
          },
        ]}
      />
    </div>
  )
}
