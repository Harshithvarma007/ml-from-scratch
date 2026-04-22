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
import GRUGateViz from '../widgets/GRUGateViz'
import GRUvsLSTM from '../widgets/GRUvsLSTM'

// Signature anchor: the streamlined filing cabinet. The LSTM had three gates
// and a separate long-term cabinet; the GRU is the IKEA version — two gates,
// one shared drawer. Reset gate = "how much of the old file is still
// relevant?" Update gate = "how much of the new memo replaces the old?"
// Returns at the opening LSTM comparison, the gate reveal, and the "same
// answers, fewer parts" consolidation.
export default function GruLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="gru" />

      {/* ── Opening: the streamlined cabinet ─────────────────────── */}
      <Prose>
        <p>
          Last lesson we built the <NeedsBackground slug="lstm">LSTM</NeedsBackground>{' '}
          — a filing cabinet with three gates, a separate long-term drawer (the
          cell state), and a working-copy drawer (the hidden state) that shadowed
          it through time. It worked. It ran speech recognition, machine
          translation, and the first generation of neural language models for
          nearly two decades. But look at it sitting on the desk: two drawers,
          four weight matrices, three gates, and every batch you ever train has
          to drag all of that through every time step.
        </p>
        <p>
          Someone was going to ask the obvious question. <em>Do we actually
          need all of this?</em>
        </p>
        <p>
          In 2014 Kyunghyun Cho and collaborators answered: no. They proposed
          the <KeyTerm>Gated Recurrent Unit</KeyTerm>, which is the LSTM cabinet
          sent to IKEA and shipped back flat-packed with fewer parts. Merge the
          two drawers into one — no separate cell state, just a single hidden
          state. Merge the forget and input gates into a single <em>update</em>{' '}
          gate — because what you don&apos;t keep is exactly what you write, so
          why were they ever two decisions? Keep one small extra gate, the{' '}
          <em>reset</em> gate, whose only job is to decide how much of the old
          file still belongs in the new draft. Three weight matrices instead of
          four. One shared drawer. Same answers, fewer parts.
        </p>
        <p>
          This is the last lesson of the RNN section. By the end of it
          you&apos;ll have derived the GRU equations, watched its two gates move
          in a widget, counted the parameters against LSTM line by line, and
          landed on the honest practitioner&apos;s verdict: <em>try both, pick
          the winner, move on</em>.
        </p>
      </Prose>

      <Personify speaker="GRU">
        The LSTM built a filing cabinet with two drawers and three gates. I
        looked at the blueprint and asked: what if one drawer, two gates? Same
        memory, 25% fewer screws. If the LSTM is a full-service archive, I am
        the streamlined version — and for most of what you&apos;ll ever ask a
        sequence model to do, you won&apos;t notice the missing parts.
      </Personify>

      {/* ── ASCII diagram ───────────────────────────────────────── */}
      <Prose>
        <p>
          Before the equations, a picture. One drawer goes in (the hidden state
          from last step), one drawer comes out (the new hidden state). In
          between, two gates make decisions and one <em>candidate</em>{' '}
          calculation proposes what the new drawer contents could look like.
          Watch the gate names: reset decides what of the past is still{' '}
          <em>relevant</em>; update decides how much of the new memo replaces
          the old file.
        </p>
      </Prose>

      <AsciiBlock caption="a GRU cell — one shared drawer, two gates, one candidate">
{`                       ┌──────────────────────────────────────┐
                       │                GRU cell              │
      h_{t-1} ────────►│                                      │──► h_t
                       │                                      │
      x_t ────────────►│   r_t = σ(W_r x + U_r h_{t-1})      │
                       │        ▲ reset gate                  │
                       │                                      │
                       │   z_t = σ(W_z x + U_z h_{t-1})      │
                       │        ▲ update gate                 │
                       │                                      │
                       │   ĥ   = tanh(W_h x + U_h (r ⊙ h_{t-1}))
                       │        ▲ candidate hidden            │
                       │                                      │
                       │   h_t = (1 − z) ⊙ h_{t-1}            │
                       │         + z ⊙ ĥ                      │
                       │        ▲ linear interpolation        │
                       └──────────────────────────────────────┘`}
      </AsciiBlock>

      {/* ── Math ────────────────────────────────────────────────── */}
      <Prose>
        <p>
          Four equations, read top to bottom as one pass through the cell at
          time step <code>t</code>. Input: <code>x_t</code> (the new memo) and{' '}
          <code>h_{'{'}t-1{'}'}</code> (the drawer&apos;s current contents).
          Output: <code>h_t</code> (the drawer&apos;s new contents). The two{' '}
          <NeedsBackground slug="sigmoid-and-relu">sigmoid</NeedsBackground>{' '}
          gates squash into <code>(0, 1)</code> — soft switches. The tanh
          candidate proposes a value in <code>(-1, 1)</code>.
        </p>
      </Prose>

      <MathBlock caption="GRU — all four equations">
{`r_t  =  σ( W_r x_t  +  U_r h_{t-1}  +  b_r )                    reset gate

z_t  =  σ( W_z x_t  +  U_z h_{t-1}  +  b_z )                    update gate

ĥ_t  =  tanh( W_h x_t  +  U_h ( r_t ⊙ h_{t-1} )  +  b_h )       candidate

h_t  =  ( 1 − z_t ) ⊙ h_{t-1}   +   z_t ⊙ ĥ_t                   new hidden state`}
      </MathBlock>

      <Prose>
        <p>
          Stare at that last line. It&apos;s a <em>linear interpolation</em>{' '}
          between the old drawer and the candidate, with <code>z_t</code> as the
          dial. Set <code>z_t = 0</code> and the cell just copies{' '}
          <code>h_{'{'}t-1{'}'}</code> forward — the old file stays, nothing
          gets written. Set <code>z_t = 1</code> and the cell throws out the
          old contents and commits entirely to the new candidate — full
          overwrite. Every value in between is a learned blend.
        </p>
        <p>
          That one equation is doing the work LSTM needed two gates to do. The
          old <code>f · c_{'{'}t-1{'}'} + i · ĉ</code> formula had a forget gate{' '}
          <em>and</em> an input gate that could in principle fight each other —
          forget a lot and write a lot, or keep everything and write nothing,
          weirdly consistent but wasteful. GRU ties them together by
          construction: whatever fraction of the old file you toss is exactly
          the fraction of the new memo you write. One dial, not two.
        </p>
      </Prose>

      {/* ── Widget 1 ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Drag the sliders below. Push <code>z_t</code> toward 0 and watch{' '}
          <code>h_t</code> track <code>h_{'{'}t-1{'}'}</code> exactly — the
          drawer doesn&apos;t change. Push it toward 1 and <code>h_t</code>{' '}
          snaps to the candidate — the drawer gets a new file. Push{' '}
          <code>r_t</code> toward 0 and the candidate forgets its own history:
          it becomes a function of <code>x_t</code> alone, as if the cell
          opened a blank folder and didn&apos;t consult the old one at all. Two
          gates. One drawer. Watch them work.
        </p>
      </Prose>

      <GRUGateViz />

      <Callout variant="note" title="two knobs, one drawer">
        The update gate decides <em>how much</em> of the old file survives. The
        reset gate decides how much of the old file is even <em>visible</em>{' '}
        when drafting the replacement. They act at different stages of the same
        pass. A pattern you&apos;ll see the network learn on real text:{' '}
        <code>r</code> drops near zero at topic switches (&ldquo;new
        paragraph, don&apos;t anchor on what came before&rdquo;),{' '}
        <code>z</code> stays low during routine filler tokens (&ldquo;nothing
        worth writing, keep the drawer as-is&rdquo;). Try to induce both in the
        widget.
      </Callout>

      {/* ── Personify: update gate ──────────────────────────────── */}
      <Personify speaker="Update gate z_t">
        I am the interpolator. Every time step I pick a number between 0 and 1,
        elementwise, for how much of the candidate to let into the drawer — and
        by subtraction, how much of the old file stays. Write too fast and you
        overwrite memory you needed. Write too slow and the drawer never gets
        updated. I am the whole reason this streamlined cabinet can hold a fact
        across a hundred time steps and also swap that fact when it stops being
        true. I used to be two separate gates in the LSTM days. Now I am one.
      </Personify>

      {/* ── Parameter budget vs LSTM ────────────────────────────── */}
      <Prose>
        <p>
          How much cabinet do you save by going IKEA? Count the matrices. With
          input dimension <code>d_x</code> and hidden dimension <code>d_h</code>,
          the weight budget breaks down like this.
        </p>
      </Prose>

      <MathBlock caption="parameter budget — GRU vs LSTM">
{`LSTM                                    GRU

forget gate   W_f ∈ ℝ^{d_h×(d_x+d_h)}   reset gate    W_r ∈ ℝ^{d_h×(d_x+d_h)}
input gate    W_i ∈ ℝ^{d_h×(d_x+d_h)}   update gate   W_z ∈ ℝ^{d_h×(d_x+d_h)}
output gate   W_o ∈ ℝ^{d_h×(d_x+d_h)}   candidate     W_h ∈ ℝ^{d_h×(d_x+d_h)}
candidate     W_c ∈ ℝ^{d_h×(d_x+d_h)}

─────────────────────────────           ─────────────────────────────
4 matrices                              3 matrices

total weights  ≈ 4 · d_h · (d_x + d_h)  total weights ≈ 3 · d_h · (d_x + d_h)

                    ratio GRU / LSTM  =  3/4  =  0.75`}
      </MathBlock>

      <Prose>
        <p>
          Exactly 25% fewer parameters, same input and hidden sizes. Not a
          marketing number — a direct consequence of one fewer gate, one fewer
          matrix. On a <code>d_h = 512</code> hidden size with{' '}
          <code>d_x = 512</code> input, LSTM sits at ~2.1M parameters, GRU at
          ~1.6M. Stack that into a six-layer encoder and the savings compound
          into real megabytes.
        </p>
        <p>
          The parameter ratio ripples outward. 25% less memory for the weights.
          25% less memory for the Adam moments (two extra copies of every
          parameter, which people forget when they budget GPU memory). 25%
          fewer FLOPs per step. On constrained hardware — phones, edge devices,
          real-time audio, anything where you&apos;re fighting for milliseconds
          or milliwatts — that ratio is the whole reason you reach for GRU
          instead of LSTM.
        </p>
      </Prose>

      {/* ── Widget 2 ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Parameter count is one axis. What you actually get for the savings is
          another — and the only honest way to answer that is to run them
          side-by-side. The widget below does exactly the experiment you&apos;d
          run yourself: same task, same hidden size, same optimizer, GRU and
          LSTM bolted up next to each other. Watch the parameter count, the
          wall-clock training time, and the final accuracy tick in parallel.
        </p>
      </Prose>

      <GRUvsLSTM />

      <Prose>
        <p>
          This is the pattern Chung et al. (2014) and Jozefowicz et al. (2015)
          found when they ran this comparison at scale across dozens of
          benchmarks: GRU and LSTM land within a point or two of each other on
          most tasks, and <em>neither one consistently wins</em>. That&apos;s a
          stronger finding than it sounds. It means the LSTM&apos;s extra
          machinery — the separate long-term drawer, the third gate, the fourth
          matrix — bought very little on the kind of problems people were
          actually throwing at sequence models. GRU was paying 75% of the
          compute bill for 99% of the performance. Same answers, fewer parts.
        </p>
      </Prose>

      <Callout variant="insight" title="when LSTM still pulls ahead">
        Jozefowicz et al.&apos;s large-scale architectural search did find one
        regime where LSTM reliably beat GRU: language modeling with very long
        dependencies, where a separate cell state — insulated from the tanh
        that squashes the hidden state — could carry information farther
        without being compressed at every read. For shorter sequences
        (~100 tokens or less) GRU matched or beat it. The rule of thumb:
        &ldquo;long and hard&rdquo; → LSTM gets the extra drawer. &ldquo;short
        and common&rdquo; → GRU&apos;s single drawer is fine. &ldquo;not
        sure&rdquo; → try both and look at the numbers, which is what everyone
        does anyway.
      </Callout>

      {/* ── Personify: reset gate ───────────────────────────────── */}
      <Personify speaker="Reset gate r_t">
        I&apos;m the selective amnesiac. When the input tells me context just
        flipped — end of a sentence, a topic pivot, a new speaker — I snap
        toward zero and the candidate gets drafted as if the old file never
        existed. The drawer is still there, untouched; I&apos;ve just pulled
        the blinds on it for this one round of drafting. No direct analog in
        the LSTM — the forget gate erases memory, but I only ever <em>hide</em>{' '}
        it. After this step the old file is still available for the next gate
        to consult. I&apos;m the lightweight piece of machinery GRU added to
        cover what the merged update gate couldn&apos;t.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations of the same algorithm, each shorter than the
          last. Pure Python walks the equations line by line on a single cell.
          NumPy vectorises across the hidden dimension and the batch. PyTorch
          lands you in <code>nn.GRU</code>, a one-line drop-in that runs fused
          kernels on the GPU. Same story as every other lesson in this course
          — see the mechanics, scale them up, cede them to the library.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · gru_scratch.py"
        output={`h at t=0: [0.1, -0.05, 0.02]
h at t=1: [0.18, -0.11, 0.07]
h at t=2: [0.26, -0.14, 0.11]`}
      >{`import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def tanh(x):
    return math.tanh(x)

def dot(row, vec):
    return sum(a * b for a, b in zip(row, vec))

def gru_step(x_t, h_prev, W_r, U_r, b_r,
                          W_z, U_z, b_z,
                          W_h, U_h, b_h):
    # reset gate — "how much of the old file is still relevant?"
    r = [sigmoid(dot(W_r[i], x_t) + dot(U_r[i], h_prev) + b_r[i])
         for i in range(len(h_prev))]
    # update gate — "how much of the new memo replaces the old?"
    z = [sigmoid(dot(W_z[i], x_t) + dot(U_z[i], h_prev) + b_z[i])
         for i in range(len(h_prev))]
    # candidate — reset gate modulates how much of h_prev enters
    rh = [r[i] * h_prev[i] for i in range(len(h_prev))]
    h_cand = [tanh(dot(W_h[i], x_t) + dot(U_h[i], rh) + b_h[i])
              for i in range(len(h_prev))]
    # interpolate: (1 - z) keeps old, z writes new — one drawer updated
    h_new = [(1 - z[i]) * h_prev[i] + z[i] * h_cand[i]
             for i in range(len(h_prev))]
    return h_new

# Tiny run (weights abbreviated — see repo for full example)
# Returns h at every step, starting from zeros.`}</CodeBlock>

      <Prose>
        <p>
          Every list comprehension above collapses into a matmul plus an
          elementwise op. The three gates become three batched matmuls; the
          interpolation line is one broadcasted expression. Vectorise it.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · gru_numpy.py"
        output={`h shape: (32, 128)     # batch of 32, hidden dim 128
param count: 197,376   # vs LSTM at 263,168 — a 25% savings`}
      >{`import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def gru_step(x, h_prev, params):
    W_r, U_r, b_r = params['r']
    W_z, U_z, b_z = params['z']
    W_h, U_h, b_h = params['h']

    r = sigmoid(x @ W_r.T + h_prev @ U_r.T + b_r)          # (B, d_h)
    z = sigmoid(x @ W_z.T + h_prev @ U_z.T + b_z)          # (B, d_h)
    h_cand = np.tanh(x @ W_h.T + (r * h_prev) @ U_h.T + b_h)
    h_new  = (1 - z) * h_prev + z * h_cand                 # interpolation
    return h_new

# Roll forward over a sequence of length T
def gru_forward(X, h0, params):
    # X: (T, B, d_x)    h0: (B, d_h)
    h = h0
    hs = []
    for t in range(X.shape[0]):
        h = gru_step(X[t], h, params)
        hs.append(h)
    return np.stack(hs)                                    # (T, B, d_h)`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: '[sigmoid(dot(W_r[i], x) + ...) for i in ...]',
            right: 'sigmoid(x @ W_r.T + h @ U_r.T + b_r)',
            note: 'the whole gate in one matmul, batched over B examples at once',
          },
          {
            left: '[r[i] * h_prev[i] for i in ...]',
            right: 'r * h_prev',
            note: 'the Hadamard product ⊙ — broadcast elementwise multiply',
          },
          {
            left: 'one hidden vector per call',
            right: 'shape (B, d_h) — batch-first after broadcasting',
            note: 'no Python-level loop over batch elements',
          },
        ]}
      />

      <Prose>
        <p>
          And the one-liner. <code>nn.GRU</code> mirrors <code>nn.LSTM</code>{' '}
          almost exactly — same constructor, same call signature, same forward
          pass — with one cosmetic difference: the returned state is a single
          tensor <code>h_n</code>, not the <code>(h_n, c_n)</code> tuple
          LSTM gives you. One drawer out, not two.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · gru_pytorch.py"
        output={`y shape:   torch.Size([50, 32, 128])    # (seq_len, batch, hidden)
h_n shape: torch.Size([1, 32, 128])     # (num_layers, batch, hidden)
param count GRU : 197376
param count LSTM: 263168                # same shape, ~25% heavier`}
      >{`import torch
import torch.nn as nn

seq_len, batch, d_x, d_h = 50, 32, 64, 128

gru  = nn.GRU(input_size=d_x, hidden_size=d_h, batch_first=False)
lstm = nn.LSTM(input_size=d_x, hidden_size=d_h, batch_first=False)

x = torch.randn(seq_len, batch, d_x)

# GRU: one hidden tensor
y, h_n = gru(x)                           # h_n: (num_layers, batch, d_h)

# LSTM (for comparison): two — hidden and cell
# y_l, (h_n_l, c_n_l) = lstm(x)

print("y shape:  ", y.shape)
print("h_n shape:", h_n.shape)

def count(m): return sum(p.numel() for p in m.parameters())
print("param count GRU :", count(gru))
print("param count LSTM:", count(lstm))`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'for t in range(T): gru_step(...)',
            right: 'y, h_n = nn.GRU(...)(x)',
            note: 'the time loop is fused into the module — runs on GPU, supports cuDNN',
          },
          {
            left: '3 gate matmuls per step, in Python',
            right: 'one fused kernel per step, in CUDA',
            note: 'cuDNN packs W_r, W_z, W_h into one GEMM — much faster in practice',
          },
          {
            left: 'return h only',
            right: 'y, h_n = gru(x)',
            note: 'y is all hidden states over time, h_n is just the last one',
          },
        ]}
      />

      <Callout variant="note" title="how it differs from nn.LSTM in one line">
        <p>
          <code>nn.LSTM(x)</code> returns <code>(y, (h_n, c_n))</code>.{' '}
          <code>nn.GRU(x)</code> returns <code>(y, h_n)</code>. Same sequence
          output, one fewer state tensor — the streamlined cabinet has one
          drawer, not two. If you&apos;re swapping LSTM → GRU in existing code,
          the only place it matters is wherever you unpack the state tuple.
        </p>
      </Callout>

      {/* ── Practical: just try both ─────────────────────────────── */}
      <Callout variant="insight" title="&ldquo;just try both&rdquo; is the right advice">
        After a decade of benchmarks, the practitioner consensus is this:
        there&apos;s no clean theoretical rule for predicting GRU vs LSTM on
        your specific task. The delta is usually a point or two either way, and
        it&apos;s swamped by everything else you tune — learning rate,
        regularization, init, data augmentation. The honest prescription: run
        both with the same hyperparameters on a short schedule, pick the
        winner, and get on with your life. The 25% parameter savings is a
        second-order argument unless you&apos;re memory-bound. Don&apos;t
        over-think the gate count.
      </Callout>

      <Callout variant="insight" title="why we still teach both">
        GRU didn&apos;t replace LSTM. LSTM didn&apos;t replace GRU. The modern
        answer for the problems they were designed for is that transformers
        replaced both around 2018 — while GRU and LSTM survive wherever
        latency, memory, or power is a binding constraint. On-device speech
        recognition. Keyboard autocompletion that has to run on a cheap phone.
        Real-time signal processing where you can&apos;t wait for a
        200-millisecond attention pass. Knowing the trade-offs between the
        streamlined filing cabinet and its older cousin is part of being able
        to ship.
      </Callout>

      {/* ── Gotchas ──────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Return-shape mismatch.</strong>{' '}
          Code written for <code>nn.LSTM</code> crashes on <code>nn.GRU</code>{' '}
          the first time it tries to unpack <code>(h, c)</code>. GRU has one
          drawer, not two. If you&apos;re swapping, grep your codebase for
          every <code>h, c = ...</code> and <code>(h_n, c_n)</code> and strip
          the cell-state half.
        </p>
        <p>
          <strong className="text-term-amber">Forget-gate bias trick does NOT port over.</strong>{' '}
          The famous LSTM hack — initialise the forget-gate bias to{' '}
          <code>+1</code> so the network defaults to remembering — has no
          direct GRU equivalent. GRU&apos;s update gate controls <em>both</em>{' '}
          retention and writing (they&apos;re tied by <code>1 − z</code>), so a
          positive bias on <code>z</code> encourages <em>writing</em>, not
          keeping. The analog, if you want one, is a <em>negative</em> bias on{' '}
          <code>z</code>: that nudges <code>(1 − z)</code> toward 1 so the
          drawer defaults to pass-through. Most practitioners just leave it at
          zero and trust the optimizer.
        </p>
        <p>
          <strong className="text-term-amber">Reset gate placement.</strong>{' '}
          The reset gate <em>only</em> touches <code>h_{'{'}t-1{'}'}</code>{' '}
          inside the candidate calculation — <code>r ⊙ h_{'{'}t-1{'}'}</code>{' '}
          before multiplying by <code>U_h</code>. A classic from-scratch bug is
          to apply <code>r</code> outside, on the final <code>h_t</code>
          — that breaks the whole architecture and your model trains to
          nothing. Re-read the equations:{' '}
          <code>r</code> is a <em>drafting</em> aid, not an output gate.
        </p>
        <p>
          <strong className="text-term-amber">&ldquo;GRU is always faster.&rdquo;</strong>{' '}
          Fewer FLOPs per step, yes — but on modern GPUs cuDNN has highly
          optimized fused kernels for both. The wall-clock gap is usually
          smaller than the 25% parameter ratio suggests. Measure on your
          hardware before you bet on it.
        </p>
      </Gotcha>

      {/* ── Challenge ────────────────────────────────────────────── */}
      <Challenge prompt="Swap LSTM for GRU on the same task">
        <p>
          Take the LSTM training script from the previous lesson — the one that
          classifies sequences or predicts the next token. Change{' '}
          <code>nn.LSTM</code> to <code>nn.GRU</code>, and fix the two or three
          places that unpack the state tuple (GRU returns one tensor, not two).
        </p>
        <p className="mt-2">
          Run both for the same number of epochs, same optimizer, same learning
          rate. Record three numbers per model: parameter count, wall-clock
          training time, and final accuracy / loss. You should see:
        </p>
        <ul className="mt-2">
          <li>
            GRU has ~25% fewer parameters — confirm with{' '}
            <code>sum(p.numel() for p in model.parameters())</code>.
          </li>
          <li>
            GRU trains a bit faster per epoch, though less than 25% — cuDNN is
            good at both.
          </li>
          <li>
            Final accuracy lands within a point or two of LSTM, in one
            direction or the other.
          </li>
        </ul>
        <p className="mt-2 text-dark-text-muted">
          Bonus: repeat with a very long sequence (<code>seq_len &gt; 500</code>)
          and a short one (<code>seq_len &lt; 50</code>). See if LSTM claws
          back a win on the long one, as Jozefowicz et al. predicted — the
          separate cabinet drawer earning its keep on hard, long-range memory.
        </p>
      </Challenge>

      {/* ── Takeaways + section teaser ───────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> GRU is the LSTM&apos;s filing
          cabinet with one drawer instead of two, one fewer gate, one fewer
          weight matrix — the streamlined IKEA version. The update gate does
          the interpolation between old drawer and candidate; the reset gate
          decides how much of the old contents informs the draft. The
          parameter savings are real (~25%) and the performance delta on
          typical tasks is negligible in either direction. In practice you try
          both, and most of the time the winner is within noise of the loser.
          Same answers, fewer parts.
        </p>
        <p>
          <strong>End of the RNN &amp; LSTM section.</strong> You&apos;ve now
          built — from scratch — a plain{' '}
          <NeedsBackground slug="recurrent-neural-network">RNN</NeedsBackground>,
          derived backprop through time, watched its gradient die in the{' '}
          <NeedsBackground slug="vanishing-gradient-problem">vanishing gradient</NeedsBackground>{' '}
          regime, and met the two gated architectures (LSTM, GRU) that fixed it
          well enough to power a decade of NLP and speech systems. Every one of
          these survives in a latency-sensitive corner of some real product
          shipping right now. The mainstream, though, has moved on.
        </p>
        <p>
          <strong>Next up — NLP.</strong> We&apos;ve squeezed memory down to
          the minimum with GRU — one drawer, two gates, same answers, fewer
          parts. But the whole time we&apos;ve been hand-waving the inputs as{' '}
          <code>x_t</code> vectors falling from the sky. The next section
          confronts what we&apos;re actually feeding these cells. <em>Words.</em>{' '}
          And that turns out to be its own problem — one that unlocks everything
          from word2vec to attention to the transformer. Kick off with{' '}
          <KeyTerm>Intro to NLP</KeyTerm>: tokens, vocabularies, and how you
          turn a page of English into something a neural net can differentiate
          through.
        </p>
      </Prose>

      <References
        items={[
          {
            title:
              'Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation',
            author: 'Cho, van Merriënboer, Gulcehre, Bahdanau, Bougares, Schwenk, Bengio',
            venue: 'EMNLP 2014 — the original GRU paper',
            url: 'https://arxiv.org/abs/1406.1078',
          },
          {
            title: 'Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling',
            author: 'Chung, Gulcehre, Cho, Bengio',
            year: 2014,
            url: 'https://arxiv.org/abs/1412.3555',
          },
          {
            title: 'An Empirical Exploration of Recurrent Network Architectures',
            author: 'Jozefowicz, Zaremba, Sutskever',
            venue: 'ICML 2015',
            url: 'https://proceedings.mlr.press/v37/jozefowicz15.html',
          },
          {
            title: 'Dive into Deep Learning — §9.1 Gated Recurrent Units (GRU)',
            author: 'Zhang, Lipton, Li, Smola',
            url: 'https://d2l.ai/chapter_recurrent-modern/gru.html',
          },
        ]}
      />
    </div>
  )
}
