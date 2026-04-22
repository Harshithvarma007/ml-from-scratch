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
import LSTMGateViz from '../widgets/LSTMGateViz'
import CellStateTracker from '../widgets/CellStateTracker'

// Signature anchor: the filing cabinet with three gates. Forget gate = the
// shredder. Input gate = the intake clerk. Output gate = the clerk reading
// aloud to the hallway. Cell state = the long-term file in the cabinet.
// Returned at opening, at each gate reveal, and at the closing consolidation.
export default function LstmLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="lstm" />

      {/* ── Opening: filing cabinet reveal ──────────────────────── */}
      <Prose>
        <p>
          Last lesson the <NeedsBackground slug="recurrent-neural-network">RNN</NeedsBackground>&apos;s
          memory got whispered down a long chain of clerks, and by clerk fifty the message
          was gibberish. The <NeedsBackground slug="vanishing-gradient-problem">vanishing gradient</NeedsBackground>{' '}
          wasn&apos;t a training hiccup — every step in <NeedsBackground slug="backprop-through-time">BPTT</NeedsBackground>{' '}
          multiplied the signal by <code>W · tanh&apos;(·)</code>, and both factors made it
          smaller. No learning rate fixes that. The architecture itself has to change.
        </p>
        <p>
          So picture a different office. Instead of one clerk trying to carry every memory in
          their head, put a <strong>filing cabinet</strong> in the room. A long drawer running
          alongside the whole hallway of clerks. Give the cabinet three gates — three small
          employees who together decide what gets written, what gets kept, and what gets read
          aloud to the next room. That cabinet, riding quietly past every clerk, is the thing
          Sepp Hochreiter and Jürgen Schmidhuber added to RNNs in 1997. They called it the{' '}
          <KeyTerm>Long Short-Term Memory</KeyTerm> network.
        </p>
        <p>
          The LSTM was the dominant sequence model for two decades — speech recognition,
          handwriting, machine translation, the first neural language models that worked at
          scale. Google Translate was an LSTM stack as late as 2016. Every speech-to-text
          system on every phone until about 2018 ran one of these. The reason is the filing
          cabinet: a dedicated drawer where information can sit unchanged across hundreds of
          time steps, with three gates controlling what goes in, what stays, and what comes
          out.
        </p>
      </Prose>

      <Personify speaker="LSTM">
        The vanilla RNN tried to hold every memory in its hands while running the hallway.
        I installed a filing cabinet. A clerk files the important bits, a shredder trims the
        stale bits, and another clerk reads the relevant drawer aloud to the next room. The
        paper never gets squashed in transit — that&apos;s why my gradient survives the trip.
      </Personify>

      {/* ── Cell diagram ────────────────────────────────────────── */}
      <Prose>
        <p>
          Before the equations, the floor plan. An LSTM cell takes three things in: the input
          at this time step <code>x_t</code>, the previous hidden state{' '}
          <code>h_&#123;t-1&#125;</code> (what the last room shouted over the wall), and the
          previous cell state <code>c_&#123;t-1&#125;</code> (the drawer rolling in from the
          last room). It produces two things out: the new drawer contents <code>c_t</code> and
          the new shouted summary <code>h_t</code>. Inside, four tiny networks stand at the
          cabinet — three gates and one candidate note to file.
        </p>
      </Prose>

      <AsciiBlock caption="one LSTM cell — the filing cabinet and its three gates">
{`                    ┌───────────── c_t ──────────────►
                    │                        (the cabinet drawer, rolling through)
      c_{t-1} ──►──┤ × ──── + ────────┬─────────────►
                    │       │         │
                   f_t     i_t·g_t    │
                    ▲       ▲         │
                    │       │       tanh
                    │       │         │
                    │       │         × ◄──── o_t
                    │       │         │
                    │       │         ▼
                    │       │       h_t ────────────►
                    │       │       (what the clerk reads aloud)
       [ h_{t-1}, x_t ] ───┴─► W_f, W_i, W_g, W_o ─┘

   f_t  forget gate   = the shredder                (sigmoid)
   i_t  input gate    = the intake clerk            (sigmoid)
   g_t  candidate     = the note they want to file  (tanh)
   o_t  output gate   = the clerk reading aloud     (sigmoid)`}
      </AsciiBlock>

      <Prose>
        <p>
          Four small networks, one per role, each reading the same concatenated input{' '}
          <code>[h_&#123;t-1&#125;, x_t]</code> — what the last room shouted, plus the new
          paperwork arriving at the door. Three of them are <NeedsBackground slug="sigmoid-and-relu">sigmoid</NeedsBackground>{' '}
          gates producing numbers in <code>(0, 1)</code> — soft on/off switches, a dial from
          &ldquo;fully closed&rdquo; to &ldquo;wide open.&rdquo; The fourth is a tanh that
          proposes an actual <em>value</em> in <code>(-1, 1)</code> — the note the intake
          clerk wants to drop into the drawer.
        </p>
      </Prose>

      {/* ── Math: four gates ────────────────────────────────────── */}
      <MathBlock caption="the four roles — all read the same [h_{t-1}, x_t]">
{`f_t  =  σ(W_f · [h_{t-1}, x_t] + b_f)          ← shredder         (0,1)

i_t  =  σ(W_i · [h_{t-1}, x_t] + b_i)          ← intake clerk     (0,1)

g_t  =  tanh(W_g · [h_{t-1}, x_t] + b_g)       ← the note to file (-1,1)

o_t  =  σ(W_o · [h_{t-1}, x_t] + b_o)          ← hallway reader   (0,1)`}
      </MathBlock>

      <MathBlock caption="the update — elementwise multiplies, one load-bearing addition">
{`c_t  =  f_t ⊙ c_{t-1}  +  i_t ⊙ g_t            ← update the drawer

h_t  =  o_t ⊙ tanh(c_t)                         ← read out to the hallway`}
      </MathBlock>

      <Prose>
        <p>
          Read <code>c_t</code> aloud as the office scene. &ldquo;Take yesterday&apos;s drawer,
          run it past the shredder (keep what you want, shred the rest), then <em>add</em> the
          intake clerk&apos;s new note.&rdquo; That plus sign is the entire trick. A vanilla
          RNN had a multiply where the filing cabinet has an add — and the derivative of an
          add is <code>1</code>, so the signal rolls through cleanly.
        </p>
        <p>
          And <code>h_t</code>: &ldquo;Squash the drawer contents through tanh to bound them,
          then let the output clerk decide what to read aloud.&rdquo; The cell state holds the
          whole file; the hidden state is a filtered, bounded <em>excerpt</em> — what the
          clerk chose to shout into the hallway. Two channels, two jobs.
        </p>
      </Prose>

      {/* ── Widget 1: Gate Viz ──────────────────────────────────── */}
      <Prose>
        <p>
          Pull the gate sliders. Each gate is a single number in its allowed range — set them
          by hand, feed in an <code>x_t</code> and a <code>h_&#123;t-1&#125;</code>, and watch{' '}
          <code>c_t</code> and <code>h_t</code> get assembled piece by piece. The widget is
          the five lines of math above, made physical.
        </p>
      </Prose>

      <LSTMGateViz />

      <Prose>
        <p>
          Three settings to try, because each one is a different office scene.{' '}
          <strong>Shredder off, intake closed</strong> (<code>f_t = 1, i_t = 0</code>): the
          drawer is perfectly preserved. Whatever was in <code>c_&#123;t-1&#125;</code>{' '}
          becomes <code>c_t</code> unchanged. Nobody touched the cabinet this step — a pure
          pass-through.{' '}
          <strong>Shredder on, intake open</strong> (<code>f_t = 0, i_t = 1</code>):
          completely dump the old drawer and refile with the new candidate note. Useful when
          the sequence hits a hard boundary — a sentence ended, a new clause begins.{' '}
          <strong>Output gate closed</strong> (<code>o_t = 0</code>): the hidden state is
          zero even though the drawer may be full of useful paper. The network is remembering
          but keeping quiet — not telling the hallway yet.
        </p>
      </Prose>

      <Personify speaker="Forget gate (the shredder)">
        I stand between the incoming drawer and the outgoing drawer with a paper shredder. A
        value of 1 means the page goes through untouched. A value of 0 means confetti. The
        rest of the network learns when I should fire. Default me to &ldquo;leave it
        alone&rdquo; on day one — I&apos;ll explain why in the gotchas.
      </Personify>

      <Personify speaker="Input gate (the intake clerk)">
        I&apos;m the clerk at the filing cabinet&apos;s front desk. The tanh next to me has
        already written a candidate note — my job is to decide how much of that note actually
        gets dropped into the drawer. Near 1, I file it. Near 0, I toss it in the recycling.
        Pair of scissors for situational commitment.
      </Personify>

      <Personify speaker="Output gate (the hallway reader)">
        I read the drawer aloud to the next room. Not everything in the cabinet is relevant
        right now — so I cherry-pick. The cell state is the whole archive. The hidden state
        is me, clearing my throat and reading the bits that matter for <em>this</em> decision.
      </Personify>

      {/* ── Gradient through the cell state ─────────────────────── */}
      <Prose>
        <p>
          Here is the payoff. In the vanilla RNN the per-step gradient factor was{' '}
          <code>∂h_t / ∂h_&#123;t-1&#125; = diag(tanh&apos;) · W</code> — a full matrix
          multiplied by a squashing derivative, the two ingredients in the telephone-chain
          decay. Now repeat that calculation for the filing cabinet.
        </p>
      </Prose>

      <MathBlock caption="the drawer&apos;s gradient highway — no W, no tanh'">
{`c_t        =   f_t ⊙ c_{t-1}  +  i_t ⊙ g_t

∂c_t       =   f_t              (+ small terms, since f_t, i_t, g_t depend on h_{t-1})
─────
∂c_{t-1}

Over T steps, treating the gates as approximately piecewise-constant:

∂c_T / ∂c_0   ≈   f_T ⊙ f_{T-1} ⊙ ... ⊙ f_1`}
      </MathBlock>

      <Prose>
        <p>
          Compare the two. The RNN&apos;s per-step factor is a full matrix times a squashing
          derivative. The filing cabinet&apos;s per-step factor along the drawer is just a
          diagonal — the shredder — with nothing squashing on top of it. If the shredder
          stays near <code>1</code> (leave the paper alone), the gradient through the drawer
          survives essentially indefinitely. If the shredder drifts toward zero, yes, the
          gradient shrinks along that dimension — but only because the network <em>chose</em>{' '}
          to forget that piece. The vanilla RNN forgot by accident. The LSTM forgets on
          purpose.
        </p>
        <p>
          The full gradient has more paths than the clean <code>f_t</code> product above —
          the gates themselves depend on <code>h_&#123;t-1&#125;</code>, which depends on{' '}
          <code>c_&#123;t-1&#125;</code>, so there are side channels. But the{' '}
          <em>dominant</em> term is the one written, and in practice it carries gradient over
          hundreds of steps where the vanilla RNN dies inside of twenty.
        </p>
      </Prose>

      <Callout variant="insight" title="addition is the whole trick">
        The drawer gets updated by an addition. The derivative of <code>a + b</code> with
        respect to <code>a</code> is <code>1</code> — no shrinkage term, no squashing factor.
        That is the mathematical root of long-range memory, and it&apos;s why every serious
        long-sequence architecture since has looked for an additive path. ResNets are the
        same idea applied to depth. Transformer residual streams are the same idea applied to
        layers. The filing cabinet was first.
      </Callout>

      {/* ── Widget 2: Cell State Tracker ────────────────────────── */}
      <Prose>
        <p>
          Let&apos;s watch the drawer hold a page across the whole hallway. Fifty time steps,
          one bit of information presented at step 5 with a &ldquo;remember this&rdquo;
          marker. The network&apos;s job is to keep that bit alive in the cabinet until step
          50, when a recall cue asks for it. Scrub through time and watch the cell state.
        </p>
      </Prose>

      <CellStateTracker />

      <Prose>
        <p>
          Notice how the trained LSTM latches the bit. When the marker arrives at step 5, the
          intake clerk opens briefly — one file drop. The shredder then sits at <code>1</code>{' '}
          for the rest of the sequence (leave the drawer alone). The hallway reader stays
          quiet until step 50, when it finally reads the drawer out loud. The cell state at
          step 49 is effectively identical to its value at step 6. Fifty steps of perfect
          retention, trained with plain backprop through time. A vanilla RNN handed the same
          task cannot solve it — its gradient dies long before the loss can teach it what to
          hold.
        </p>
      </Prose>

      {/* ── Parameter count ─────────────────────────────────────── */}
      <Callout variant="note" title="three gates, four weight matrices, 4× the parameters">
        A vanilla RNN with input size <code>d</code> and hidden size <code>h</code> has one
        weight matrix of shape <code>(h, d + h)</code>. The filing cabinet has four — three
        for the gates, one for the candidate note — for roughly <code>4 · h · (d + h)</code>{' '}
        parameters. That&apos;s the price of the gating machinery, and it&apos;s why LSTMs
        are heavy compared to their vanilla cousin. In production code the four matrices are
        usually stacked into one fused matmul for efficiency.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three passes at the filing cabinet, each shorter than the last. Pure Python is the
          equations line-for-line on a single time step — the cabinet drawn by hand. NumPy
          vectorises across the hidden dimension and collapses the four gates into one fused
          matmul. PyTorch drops the whole thing into <code>nn.LSTM</code> with packed
          sequences — the version you&apos;ll actually ship.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · lstm_scratch.py"
        output={`t=1  c=[0.31, -0.12]   h=[0.22, -0.08]
t=2  c=[0.47,  0.05]   h=[0.33,  0.03]`}
      >{`import math

def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
def tanh(x):    return math.tanh(x)

def lstm_cell(x, h_prev, c_prev, W, b):
    """One step past the filing cabinet. Hidden size 2. W is a dict of 4 gate matrices."""
    # concatenate last room's shout with the new paperwork at the door
    z = [*h_prev, *x]                              # [h_{t-1}; x_t]

    # three gates + one candidate note, one dot product each
    def gate(Wg, bg, act):
        return [act(sum(wij * zj for wij, zj in zip(row, z)) + b)
                for row, b in zip(Wg, bg)]

    f = gate(W["f"], b["f"], sigmoid)              # shredder       (forget gate)
    i = gate(W["i"], b["i"], sigmoid)              # intake clerk   (input gate)
    g = gate(W["g"], b["g"], tanh)                 # note to file   (candidate)
    o = gate(W["o"], b["o"], sigmoid)              # hallway reader (output gate)

    # drawer update — the two lines everything rides on
    c = [fi * cp + ii * gi for fi, cp, ii, gi in zip(f, c_prev, i, g)]
    h = [oi * tanh(ci)      for oi, ci              in zip(o, c)]
    return h, c`}</CodeBlock>

      <Prose>
        <p>
          Vectorise. The four gates collapse into one fused matmul of 4× the output size,
          split at the end. This is how every production LSTM kernel actually ships — one big
          matrix multiply beats four small ones on hardware every time.
        </p>
      </Prose>

      <CodeBlock language="python" caption="layer 2 — numpy · lstm_numpy.py">{`import numpy as np

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def lstm_cell(x, h_prev, c_prev, W, b):
    """
    Fused filing cabinet. W has shape (4h, d + h); b has shape (4h,).
    We split the 4h-dimensional pre-activation into the three gates + candidate at once.
    """
    z = np.concatenate([h_prev, x], axis=-1)       # (d + h,)
    pre = W @ z + b                                # (4h,)

    f, i, g, o = np.split(pre, 4, axis=-1)         # each (h,) — one slice per role

    f = sigmoid(f)                                 # shredder
    i = sigmoid(i)                                 # intake clerk
    g = np.tanh(g)                                 # candidate note
    o = sigmoid(o)                                 # hallway reader

    c = f * c_prev + i * g                         # drawer update — the magic line
    h = o * np.tanh(c)
    return h, c

# rolling the cabinet through the whole hallway
def lstm_forward(X, W, b, h_dim):
    T = X.shape[0]
    h = np.zeros(h_dim); c = np.zeros(h_dim)
    H = np.zeros((T, h_dim))
    for t in range(T):
        h, c = lstm_cell(X[t], h, c, W, b)
        H[t] = h
    return H`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'four separate gate() calls',
            right: 'one fused matmul, np.split into four',
            note: 'hardware loves one big GEMM over four small ones',
          },
          {
            left: 'list-comprehensions for elementwise ops',
            right: 'f * c_prev + i * g  (broadcasted)',
            note: 'batch + hidden dims come free once you are in numpy',
          },
          {
            left: 'loops over hidden units inside the cell',
            right: 'only the time loop remains — everything else is vectorised',
            note: 'the hallway stays sequential; the cabinet itself is parallel',
          },
        ]}
      />

      <Prose>
        <p>
          PyTorch packs all four roles, handles batching, runs on GPU, and — critically —
          supports <em>packed sequences</em>: the standard way to feed variable-length
          batches through the cabinet without wasting compute on padding tokens. The whole
          cell becomes a one-liner.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · lstm_pytorch.py"
        output={`output shape:  torch.Size([32, 7, 128])   # (batch, time, hidden)
final h_T:     torch.Size([1, 32, 128])
final c_T:     torch.Size([1, 32, 128])`}
      >{`import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LstmTagger(nn.Module):
    def __init__(self, vocab, embed=64, hidden=128, n_tags=10):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.lstm  = nn.LSTM(embed, hidden, batch_first=True)     # all four roles, fused
        self.head  = nn.Linear(hidden, n_tags)

        # GOTCHA PREVIEW: tell the shredder to leave the drawer alone on day one.
        # PyTorch packs biases as [b_i, b_f, b_g, b_o]; hidden:2*hidden slice is forget.
        for name, p in self.lstm.named_parameters():
            if "bias" in name:
                p.data.fill_(0.0)
                n = p.shape[0] // 4
                p.data[n:2 * n].fill_(1.0)         # forget-gate bias = 1 → default to remember

    def forward(self, x, lengths):
        emb = self.embed(x)                                       # (B, T, E)
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        out, (h_T, c_T) = self.lstm(packed)                       # packed -> packed
        out, _ = pad_packed_sequence(out, batch_first=True)       # (B, T, H)
        return self.head(out), (h_T, c_T)

model = LstmTagger(vocab=5000)
x       = torch.randint(0, 5000, (32, 7))                         # (batch=32, max_len=7)
lengths = torch.tensor([7, 6, 6, 5, 5, 4, 3, 3, 2, 2] + [7]*22)
logits, (h_T, c_T) = model(x, lengths)
print("output shape: ", logits.shape)
print("final h_T:    ", h_T.shape)
print("final c_T:    ", c_T.shape)`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'hand-written time loop',
            right: 'nn.LSTM(..., batch_first=True)',
            note: 'CUDA-optimised fused kernel — orders of magnitude faster',
          },
          {
            left: 'ragged sequences → mask or pad manually',
            right: 'pack_padded_sequence + pad_packed_sequence',
            note: 'no compute wasted on padding positions',
          },
          {
            left: 'autograd for backprop through time',
            right: 'loss.backward() — just works',
            note: 'the messy gradient chains from earlier are handled for you',
          },
        ]}
      />

      {/* ── Variants ────────────────────────────────────────────── */}
      <Prose>
        <p>
          A quick tour of cabinet variants you&apos;ll see in the wild. Most don&apos;t matter
          much in 2026, but they show up in papers and legacy codebases:
        </p>
        <ul>
          <li>
            <strong>Peephole connections.</strong> Gers &amp; Schmidhuber (2000) let each gate
            peek directly at <code>c_&#123;t-1&#125;</code> — the gates read the drawer, not
            just the hallway shout. Small win on some tasks, extra parameters. Mostly gone.
          </li>
          <li>
            <strong>Coupled input-forget.</strong> Enforce <code>i_t = 1 − f_t</code> — the
            intake clerk and shredder share one dial. One fewer gate to learn, cheaper cell.
            Shows up in GRU-adjacent designs (foreshadowing).
          </li>
          <li>
            <strong>Bidirectional LSTMs.</strong> Run two cabinets — one rolling forward down
            the hallway, one rolling backward — and concatenate their hidden states. Standard
            for tagging tasks where the whole sequence is available at once (NER, POS, speech
            recognition). Not useful for streaming generation.
          </li>
        </ul>
      </Prose>

      <Callout variant="insight" title="twenty years of dominance">
        Between 1997 and roughly 2018, LSTMs ran the sequence-learning world. Speech
        recognition (Graves, 2013 → Baidu DeepSpeech, 2014). Handwriting recognition.
        Machine translation (Seq2Seq, Sutskever et al. 2014, was a stack of these cabinets).
        The first character-level text generators that produced recognisable output —
        Karpathy&apos;s &ldquo;Unreasonable Effectiveness&rdquo; blog post was an LSTM. The
        first neural language models that worked at scale. Every phone&apos;s speech-to-text
        until about 2018. Longest-running single architecture in modern deep learning.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Shredder bias at zero is a silent killer.</strong>{' '}
          A freshly-initialised LSTM with zero biases has <code>σ(0) = 0.5</code> forget gates
          — which means every step the drawer loses half its contents. After twenty steps
          that&apos;s <code>0.5²⁰ ≈ 10⁻⁶</code>: you just reintroduced the vanishing problem
          you paid four weight matrices to fix. The fix, from Jozefowicz et al. 2015, is to
          initialise the forget-gate bias to <code>1</code> (or higher). Then{' '}
          <code>σ(1) ≈ 0.73</code> and the cabinet defaults to &ldquo;leave the paper
          alone.&rdquo; Not every framework does this by default — check yours.
        </p>
        <p>
          <strong className="text-term-amber">Packed vs padded sequences.</strong>{' '}
          PyTorch&apos;s <code>nn.LSTM</code> accepts both, but feeding padded tensors means
          the cabinet rolls through padding positions doing real compute — slower and, if you
          forget to mask the output, wrong. Use <code>pack_padded_sequence</code> for any
          batch with variable lengths.
        </p>
        <p>
          <strong className="text-term-amber">Hidden state vs cell state confusion.</strong>{' '}
          <code>h_t</code> is what the hallway reader said aloud — what the next layer sees
          and what you decode from. <code>c_t</code> is the drawer itself — the memory
          channel that doesn&apos;t leave the cabinet directly. If you pass a sequence with{' '}
          <code>hidden=None</code>, PyTorch zeros <em>both</em> <code>h_0</code> and{' '}
          <code>c_0</code>; if you carry state between chunks (stateful LSTM), carry both.
          Mixing them up is the single most common LSTM bug.
        </p>
        <p>
          <strong className="text-term-amber">Gradient clipping is still needed.</strong>{' '}
          The filing cabinet solves vanishing gradients. It does not solve <em>exploding</em>{' '}
          gradients. Clip the gradient norm at 1–5 before calling{' '}
          <code>optimizer.step()</code>, or watch one spike wipe your weights.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="The copy task — T = 100, two architectures">
        <p>
          Generate sequences of length 100. The first 10 positions hold a random pattern of
          symbols (0–9), the next 89 are a blank filler, and the final position is a
          &ldquo;recall&rdquo; marker. Target at the marker: the first symbol of the pattern.
          The network has to remember position 0 across 99 time steps.
        </p>
        <p className="mt-2">
          Train two models: <code>nn.RNN</code> and <code>nn.LSTM</code>, same hidden size,
          same optimiser, same step count. Plot the loss curves.
        </p>
        <p className="mt-2 text-dark-text-muted">
          What you&apos;ll see: the vanilla RNN plateaus at chance — its gradient dies before
          the marker&apos;s signal can reach position 0 to teach it anything. The filing
          cabinet solves the task cleanly. This is not a subtle gap — it&apos;s 90% accuracy
          versus coin-flip — and it&apos;s exactly the failure Hochreiter and Schmidhuber set
          out to fix.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: remove the forget-bias-of-one trick from your LSTM. How much slower does it
          train? Sometimes it doesn&apos;t converge at all. One-line bug — now you&apos;ll
          spot it at a glance.
        </p>
      </Challenge>

      {/* ── Closing: why the cabinet survives the telephone chain ── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> The LSTM is two pieces of furniture: a{' '}
          <em>cell state</em> that carries memory through time on an additive path — the
          filing cabinet drawer, with no matrix and no squashing derivative in its way, so
          the gradient survives — and three <em>gates</em> that decide what to shred, what to
          file, and what to read aloud. That addition is why the cabinet survives the
          whispered telephone chain the RNN couldn&apos;t: every clerk along the hallway can
          open the drawer without rewriting its contents. The idea outlived the architecture.
          Residual connections are the same move applied to depth. Transformer residual
          streams are the same move applied to layers. State-space models are the same move
          with continuous dynamics. All of them are descendants of one realisation:{' '}
          <em>give gradient a highway with a derivative of 1.</em>
        </p>
        <p>
          <strong>Next up — GRU.</strong> Three gates and two states works — but look at the
          roles. The shredder and the intake clerk are almost mirror images: one says
          &ldquo;let the old stuff through,&rdquo; the other says &ldquo;let the new stuff
          in.&rdquo; Are they really doing distinct jobs, or could we collapse them into one
          dial? In 2014 Cho et al. tried exactly that. The <KeyTerm>Gated Recurrent Unit</KeyTerm>{' '}
          is the streamlined cabinet — two gates, one state, roughly the same empirical
          performance on most tasks. We&apos;ll derive it, compare parameter counts
          side-by-side, and figure out when the lighter version wins.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Long Short-Term Memory',
            author: 'Hochreiter, Schmidhuber',
            venue: 'Neural Computation 1997 — the original paper',
            url: 'https://www.bioinf.jku.at/publications/older/2604.pdf',
          },
          {
            title: 'Learning to Forget: Continual Prediction with LSTM',
            author: 'Gers, Schmidhuber, Cummins',
            venue: 'Neural Computation 2000 — introduced the forget gate',
            url: 'https://direct.mit.edu/neco/article/12/10/2451/6415/',
          },
          {
            title: 'An Empirical Exploration of Recurrent Network Architectures',
            author: 'Jozefowicz, Zaremba, Sutskever',
            venue: 'ICML 2015 — forget-bias-of-one and variant search',
            url: 'https://proceedings.mlr.press/v37/jozefowicz15.html',
          },
          {
            title: 'The Unreasonable Effectiveness of Recurrent Neural Networks',
            author: 'Andrej Karpathy',
            venue: 'blog post, 2015 — the character-level LSTM that launched a thousand demos',
            url: 'https://karpathy.github.io/2015/05/21/rnn-effectiveness/',
          },
          {
            title: 'Dive into Deep Learning — §9.2 Long Short-Term Memory',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_recurrent-modern/lstm.html',
          },
        ]}
      />
    </div>
  )
}
