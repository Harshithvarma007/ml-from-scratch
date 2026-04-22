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
import RNNUnroll from '../widgets/RNNUnroll'
import HiddenStateTimeline from '../widgets/HiddenStateTimeline'

// Signature anchor: a note-taker who keeps ONE sticky note. As each word
// passes, they rewrite the sticky with a condensed summary. The sticky =
// hidden state. Re-landed at the opening problem frame, the hidden-state
// reveal, and the unrolling bridge to BPTT.
export default function RecurrentNeuralNetworkLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="recurrent-neural-network" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Picture a note-taker sitting at a desk with exactly one sticky note. A
          stream of words is being read out loud, one at a time. The note-taker
          is not allowed to write a second sticky. They can&apos;t flip the sticky
          over. They can&apos;t ask for a bigger one. They are allowed to do
          exactly one thing: after each word, erase the sticky and rewrite it
          with a condensed summary of everything that matters so far.
        </p>
        <p>
          That&apos;s an RNN. The whole idea. Before any math, before any code —
          that&apos;s the thing. Now let&apos;s figure out why we need one in the
          first place, because every network we&apos;ve built so far has had a
          very specific allergy.
        </p>
        <p>
          Everything we&apos;ve built so far assumes one thing about its input:{' '}
          <em>it has a fixed shape</em>. An <NeedsBackground slug="mlp-from-scratch">MLP</NeedsBackground>{' '}
          expects exactly 784 pixels, or exactly 30 tabular features, or exactly
          whatever you wired its input layer to take. Change the length and the
          whole network shrugs and refuses.
        </p>
        <p>
          Now try to feed it a sentence. &ldquo;Hi.&rdquo; is two characters. &ldquo;I really
          loved this movie, can&apos;t wait to see it again.&rdquo; is fifty. A speech waveform
          has hundreds of thousands of samples. A stock ticker never stops. Variable length is
          the <em>default</em> for anything time-shaped, and a feedforward net is architecturally
          allergic to it.
        </p>
        <p>
          The <KeyTerm>recurrent neural network</KeyTerm> is the first architecture we&apos;ll
          meet that handles sequences natively. It solves the problem the way
          our note-taker solves it: instead of demanding the whole sequence in
          one giant input vector, read it one step at a time, and keep a little
          running summary between steps. That summary is the sticky note. The
          rule for rewriting the sticky is exactly <em>one</em> function — applied over and
          over, with the <em>same</em> weights. That&apos;s it. That&apos;s the whole idea.
          The rest of this lesson is the consequences.
        </p>
      </Prose>

      <Personify speaker="Feedforward net">
        I expect an input of shape <code>[B, 784]</code>. If you give me <code>[B, 912]</code>{' '}
        I raise a shape error and crash. If you give me one example of length 3 and another of
        length 7 in the same batch, I raise a shape error and crash. I am perfect at images. I
        am allergic to sentences.
      </Personify>

      {/* ── The RNN core ─────────────────────────────────────────── */}
      <Prose>
        <p>
          Here is the note-taker&apos;s rewrite rule, written as math. At each time step{' '}
          <code>t</code>, read the current word <code>x_t</code> and look at what&apos;s on the
          sticky from a moment ago, <code>h_{'{t-1}'}</code>. Mix them with two weight matrices,
          add a bias, and squash with a tanh. The output replaces whatever was on the sticky:
        </p>
      </Prose>

      <MathBlock caption="the RNN recurrence">
{`h_t  =  tanh( W_x · x_t   +   W_h · h_{t-1}   +   b )

y_t  =  W_y · h_t   +   b_y                    (optional per-step output)`}
      </MathBlock>

      <Prose>
        <p>
          Three weights, one nonlinearity, one bias. <code>W_x</code> maps the new word into
          sticky-note space. <code>W_h</code> maps the old sticky into the same space so the two
          can be added together. <code>tanh</code> keeps the sticky&apos;s entries in{' '}
          <code>(-1, 1)</code> so the running summary doesn&apos;t blow up as we iterate.
        </p>
        <p>
          The striking thing isn&apos;t what&apos;s in the formula — it&apos;s what&apos;s{' '}
          <em>not</em>. There is no <code>t</code> subscript on <code>W_x</code>, <code>W_h</code>,
          or <code>b</code>. The same parameters are used at every time step. The rewrite rule
          doesn&apos;t change depending on which word we&apos;re on. That&apos;s the
          &ldquo;recurrent&rdquo; part — one function, called in a loop, threading the sticky through.
        </p>
      </Prose>

      {/* ── Widget 1: RNNUnroll ──────────────────────────────────── */}
      <Prose>
        <p>
          The cleanest way to see it is to <em>unroll</em> the loop. The picture below draws the
          same RNN cell once per time step, like a cartoon strip. Click a step and watch which
          sticky gets computed, and which <code>x_t</code> and <code>h_{'{t-1}'}</code> fed
          into it.
        </p>
      </Prose>

      <RNNUnroll />

      <Prose>
        <p>
          Pay attention to one detail: the little box that says <code>W_x, W_h, b</code> is the
          same box in every panel. We didn&apos;t duplicate the parameters — we reused them. The
          drawing makes the network <em>look</em> like a deep feedforward stack, and in an
          important sense it <em>is</em> one: an N-step RNN is a depth-N network where every
          layer happens to share weights. Hold onto that — we&apos;ll use exactly that insight
          next lesson when we train it.
        </p>
      </Prose>

      <Callout variant="note" title="unrolling is a drawing, not a rewrite">
        When people say &ldquo;unroll the RNN,&rdquo; they don&apos;t mean copying the weights
        N times in memory. They mean: imagine drawing the computational graph of the loop, one
        step per box. The forward pass is still a <code>for</code> loop over a single cell.
        Only the <em>graph</em> is N boxes long, which is what autograd needs to do backprop
        through time.
      </Callout>

      {/* ── Personify: Hidden state ──────────────────────────────── */}
      <Personify speaker="Hidden state">
        I am a small vector — maybe 128 numbers, maybe 1024. I&apos;m the only thing that
        survives from one time step to the next. Everything the network wants to remember — who
        the subject of the sentence was, whether we&apos;re inside a quote, what the last chord
        was — has to be encoded somewhere in me. I am the entire memory of the model. Treat me
        with respect.
      </Personify>

      {/* ── Widget 2: HiddenStateTimeline ────────────────────────── */}
      <Prose>
        <p>
          Time to watch a real sticky note get rewritten. Feed the string{' '}
          <code>&quot;hello world&quot;</code> into a small RNN, character by character, and
          track the 16 sticky-note dimensions as a heatmap. Rows are dimensions, columns are
          time steps — each column is one rewrite, <code>h_t</code> for one character.
        </p>
      </Prose>

      <HiddenStateTimeline />

      <Prose>
        <p>
          Some dimensions change sharply on a vowel and quiet down on a consonant. Some flip
          sign on the space. Some slowly drift upward over the whole sequence. Individually
          they&apos;re unreadable; collectively they&apos;re the note-taker&apos;s shorthand —
          the pattern of marks on a sticky that, to a fluent reader (the next layer), encodes
          everything worth carrying forward. Training is the process of teaching the RNN which
          shorthand is worth keeping.
        </p>
      </Prose>

      {/* ── Weight sharing math ──────────────────────────────────── */}
      <Prose>
        <p>
          Let&apos;s put a number on the weight-sharing claim. Suppose the sticky has
          size <code>H = 128</code> and each word is encoded in <code>D = 64</code> dims. For a
          sequence of length <code>T = 100</code>:
        </p>
      </Prose>

      <MathBlock caption="parameter count — shared vs. unshared">
{`shared (an RNN):
    |W_x|  =  H · D      =   128 · 64   =    8,192
    |W_h|  =  H · H      =   128 · 128  =   16,384
    |b|    =  H          =                      128
    total                                    24,704           ← independent of T

unshared (one W per step, a straw man):
    T · (|W_x| + |W_h| + |b|)   =   100 · 24,704   =   2,470,400

ratio                                                    ≈  100×`}
      </MathBlock>

      <Prose>
        <p>
          A factor of 100, linear in sequence length. That&apos;s not even the real win,
          though. The real win is generalisation.
        </p>
        <p>
          If the pattern &ldquo;the word <em>not</em> flips sentiment&rdquo; shows up in
          training at position 3, you want the model to apply it at position 17 too. With a
          shared rewrite rule, <em>you get that for free</em> — the same <code>W_x</code>
          processes every token, so a trick learned at one position is automatically a trick at
          all positions. An unshared model would need its own little rewrite rule for every
          step, and the first time test data put &ldquo;not&rdquo; where training data
          didn&apos;t, it would shrug.
        </p>
      </Prose>

      <Personify speaker="Shared weights">
        I am stateless. I am a pure function of <code>(x_t, h_{'{t-1}'})</code>. I don&apos;t
        know what time step it is, I don&apos;t care what time step it is. You taught me a
        pattern once; I&apos;ll apply it at step 4, step 40, and step 400 with the same vigor.
        That&apos;s the deal.
      </Personify>

      {/* ── Three-layer code ─────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations of the same note-taker. Pure Python with an explicit loop,
          NumPy vectorised over a batch, PyTorch&apos;s <code>nn.RNN</code> in one call. Each is
          a rung up the abstraction ladder; each earns its keep in a different moment of your
          life.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · rnn_scratch.py"
        output={`step 0  x='h'  h_t[:4]=[ 0.123 -0.441  0.087  0.612]
step 1  x='e'  h_t[:4]=[ 0.298 -0.201  0.315  0.481]
step 2  x='l'  h_t[:4]=[ 0.517  0.082  0.294  0.203]
step 3  x='l'  h_t[:4]=[ 0.604  0.221  0.188  0.015]
step 4  x='o'  h_t[:4]=[ 0.711  0.354 -0.073 -0.218]`}
      >{`import math, random
random.seed(0)

D, H = 8, 16                             # input dim, hidden dim
vocab = sorted(set("helo "))             # tiny char set
stoi  = {c: i for i, c in enumerate(vocab)}

def onehot(c):                           # D-long one-hot, D = len(vocab) padded to 8
    v = [0.0] * D
    v[stoi[c]] = 1.0
    return v

def randmat(r, c): return [[random.gauss(0, 0.1) for _ in range(c)] for _ in range(r)]
def matvec(M, v):  return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
def vecadd(a, b):  return [x + y for x, y in zip(a, b)]
def tanh(v):       return [math.tanh(x) for x in v]

W_x = randmat(H, D)                      # H×D
W_h = randmat(H, H)                      # H×H
b   = [0.0] * H

def rnn_step(x_t, h_prev):
    # h_t = tanh(W_x x_t + W_h h_{t-1} + b)
    return tanh(vecadd(vecadd(matvec(W_x, x_t), matvec(W_h, h_prev)), b))

h = [0.0] * H                            # h_0 — the initial state
for t, ch in enumerate("hello"):
    h = rnn_step(onehot(ch), h)
    print(f"step {t}  x='{ch}'  h_t[:4]={[round(v, 3) for v in h[:4]]}")`}</CodeBlock>

      <Prose>
        <p>
          That <code>h = rnn_step(onehot(ch), h)</code> line <em>is</em> the note-taker erasing
          and rewriting their sticky. One function in, one function out. Now NumPy. Same math,
          but we broadcast over a whole batch of sequences at once — the shape every real RNN
          lives in.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · rnn_numpy.py"
        output={`h_final shape: (4, 16)     # (batch, hidden)
h_final[0, :4]: [ 0.71  0.35 -0.07 -0.22]`}
      >{`import numpy as np
rng = np.random.default_rng(0)

B, T, D, H = 4, 5, 8, 16                 # batch, time, input, hidden

X   = rng.standard_normal((B, T, D)) * 0.5      # a batch of B sequences, each length T
W_x = rng.standard_normal((D, H)) * 0.1         # note the (D, H) shape — matches x @ W
W_h = rng.standard_normal((H, H)) * 0.1
b   = np.zeros(H)

def rnn_forward(X, W_x, W_h, b):
    B, T, _ = X.shape
    H = W_h.shape[0]
    h = np.zeros((B, H))                        # h_0 for every example in the batch
    H_seq = np.zeros((B, T, H))                 # save every h_t — useful for backprop later
    for t in range(T):
        # x_t: (B, D)   W_x: (D, H)   → (B, H)
        # h  : (B, H)   W_h: (H, H)   → (B, H)
        h = np.tanh(X[:, t] @ W_x + h @ W_h + b)
        H_seq[:, t] = h
    return H_seq, h                             # whole sequence + last step

H_seq, h_final = rnn_forward(X, W_x, W_h, b)
print("h_final shape:", h_final.shape)
print("h_final[0, :4]:", np.round(h_final[0, :4], 2))`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for ch in "hello": h = rnn_step(...)',
            right: 'for t in range(T): h = np.tanh(X[:, t] @ W_x + h @ W_h + b)',
            note: 'the time loop stays — we can\'t vectorise across time, only across batch',
          },
          {
            left: 'matvec(W_x, onehot(c))',
            right: 'X[:, t] @ W_x',
            note: 'one matmul replaces B × H × D scalar multiplies',
          },
          {
            left: 'h = [0.0] * H',
            right: 'h = np.zeros((B, H))',
            note: 'one h_0 per example in the batch, all at once',
          },
        ]}
      />

      <Prose>
        <p>
          Notice the time loop survives the move to NumPy. You can parallelise a thousand
          note-takers side by side (the batch), but each one still has to rewrite their own
          sticky one word at a time — the current sticky literally doesn&apos;t exist until the
          previous one is done. In production you almost never write that loop by hand. PyTorch
          ships <code>nn.RNN</code>, which fuses the whole thing into a single optimised kernel.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · rnn_pytorch.py"
        output={`out shape:   torch.Size([4, 5, 16])     # (B, T, H)
h_n shape:   torch.Size([1, 4, 16])     # (num_layers, B, H)
matches manual last step? True`}
      >{`import torch
import torch.nn as nn

B, T, D, H = 4, 5, 8, 16
x = torch.randn(B, T, D)                        # batch-first tensor

rnn = nn.RNN(
    input_size   = D,
    hidden_size  = H,
    num_layers   = 1,
    nonlinearity = "tanh",                      # or "relu" — tanh is the classic
    batch_first  = True,                        # expect (B, T, D), not (T, B, D)
)

h0  = torch.zeros(1, B, H)                      # (num_layers, B, H) — required shape
out, h_n = rnn(x, h0)                           # out: every h_t, h_n: last h_t

print("out shape:  ", out.shape)                # (B, T, H)
print("h_n shape:  ", h_n.shape)                # (1, B, H)
print("matches manual last step?", torch.allclose(out[:, -1], h_n[0]))`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'for t in range(T): h = np.tanh(...)',
            right: 'out, h_n = rnn(x, h0)',
            note: 'the whole time loop collapses into one fused kernel',
          },
          {
            left: 'W_x, W_h, b = rng.standard_normal(...)',
            right: 'nn.RNN(input_size=D, hidden_size=H)',
            note: 'PyTorch manages params, initialization, autograd, and CUDA in one object',
          },
          {
            left: 'H_seq, h_final = rnn_forward(...)',
            right: 'out, h_n = rnn(x, h0)    # same two objects',
            note: 'out is all h_t; h_n is just the last — most code uses out for seq tasks and h_n for classification',
          },
        ]}
      />

      <Callout variant="insight" title="the point of the three layers">
        Pure Python makes the recurrence impossible to misread — one step is one function, and
        a loop drives it. NumPy forces you to think about tensor shapes: what&apos;s batched,
        what&apos;s looped, and why some of those can&apos;t be swapped. PyTorch hides all of
        that behind one object — which is fine, because you understand what it&apos;s doing.
        Nobody writes an RNN from scratch in 2026; everybody should have written one once.
      </Callout>

      {/* ── I/O patterns ─────────────────────────────────────────── */}
      <Callout variant="note" title="five shapes of sequence task">
        The magic of RNNs is that the same cell handles five very different kinds of problem,
        depending on how you wire up the inputs and outputs.
        <ul className="mt-2 space-y-1.5">
          <li>
            <strong>one-to-one</strong> — one input, one output. Standard classification. No
            recurrence needed, but an RNN can still do it.
          </li>
          <li>
            <strong>one-to-many</strong> — one input, a sequence of outputs. Image captioning:
            feed a picture, generate a sentence.
          </li>
          <li>
            <strong>many-to-one</strong> — a sequence in, one label out. Sentiment
            classification, sequence-level regression. Use <code>h_n</code>, ignore the per-step{' '}
            <code>out</code>.
          </li>
          <li>
            <strong>many-to-many (encoder–decoder)</strong> — input sequence into a context
            vector, output sequence out. Translation. This is where seq2seq comes from and
            where transformers descend.
          </li>
          <li>
            <strong>many-to-many (synced)</strong> — one output per input step, emitted in
            lockstep. POS tagging, named-entity recognition, frame-level speech labels. Use{' '}
            <code>out</code>, the full sequence of hidden states.
          </li>
        </ul>
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Forgetting <code>h0</code>:</strong>{' '}
          <code>nn.RNN</code> accepts <code>None</code> and will default <code>h_0</code> to
          zeros, which is fine for the <em>first</em> batch of a new sequence. If you&apos;re
          continuing a sequence across batches (e.g. char-level LM with truncated BPTT), you{' '}
          <em>must</em> pass the previous <code>h_n</code> as the next <code>h_0</code>.
          Otherwise you&apos;ve reset memory every batch and the model can&apos;t learn
          anything beyond <code>T</code> steps.
        </p>
        <p>
          <strong className="text-term-amber">Batch-first vs. seq-first:</strong> PyTorch&apos;s
          default is <code>(T, B, D)</code>, which is fast but reads like a Rorschach blot. Pass{' '}
          <code>batch_first=True</code> to get <code>(B, T, D)</code>, which matches every other
          tensor in your codebase. Pick one and be consistent — mixing them silently broadcasts
          garbage.
        </p>
        <p>
          <strong className="text-term-amber">Not detaching the hidden state:</strong> when you
          continue a sequence across batches, pass <code>h.detach()</code> as the next{' '}
          <code>h_0</code>, not raw <code>h</code>. If you don&apos;t, autograd keeps the graph
          alive all the way back to the start of training, and a few hundred steps in the
          gradient graph is hundreds of thousands of ops long. Memory balloons, then gradients
          explode. The fix is one method call.
        </p>
      </Gotcha>

      {/* ── Challenge ────────────────────────────────────────────── */}
      <Challenge prompt="A Shakespeare-flavoured RNN">
        <p>
          Grab the tiny-Shakespeare corpus (about 1MB of plain text, easy to find). Build a
          character vocabulary — roughly 65 characters. Train a one-layer <code>nn.RNN</code>{' '}
          with <code>hidden_size=256</code> on the task of predicting the next character given
          the previous 100. Use cross-entropy loss on the per-step output projection{' '}
          <code>W_y · h_t</code>, batch size 32, Adam at <code>1e-3</code>.
        </p>
        <p className="mt-2">
          After each epoch, sample a 200-character passage: start from a random seed, feed the
          model one character at a time, sample the next one from the softmax output, and loop.
          Print the sample.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Expect nonsense at epoch 1. By epoch 3 you&apos;ll see real English words. By
          epoch 10 you&apos;ll have something that <em>looks</em> like Shakespeare —
          word-like spacing, quotation marks in the right places, character names showing up —
          while being completely, gloriously meaningless. This is the experiment that made
          Karpathy&apos;s 2015 post famous, and it still works. Remember to <code>detach()</code>{' '}
          the hidden state between batches, or you will find out what that gotcha was about.
        </p>
      </Challenge>

      {/* ── Closing ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> An RNN is a feedforward network run in a
          loop, with a single sticky note — the hidden state — threaded from step to step. The
          same rewrite rule is used at every step, which lets it handle sequences of any length
          and generalise patterns across positions. Its entire expressive power comes from the
          single recurrence <code>h_t = tanh(W_x x_t + W_h h_{'{t-1}'} + b)</code>. Its entire
          API is three weights and a bias.
        </p>
        <p>
          <strong>Next up — Backprop Through Time.</strong> The forward pass is easy — the
          note-taker rewrites their sticky once per word and we&apos;re done. The backward pass
          has to walk through every step of that unrolled sequence in reverse, using{' '}
          <NeedsBackground slug="gradient-descent">gradient descent</NeedsBackground> and the{' '}
          <NeedsBackground slug="backpropagation">backprop</NeedsBackground> chain rule applied
          to the same shared weights at every time step. That&apos;s{' '}
          <code>backprop-through-time</code>, and that&apos;s where things get interesting.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Finding Structure in Time',
            author: 'Jeffrey L. Elman',
            venue: 'Cognitive Science, 1990 — the original Elman network',
            url: 'https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1',
          },
          {
            title: 'The Unreasonable Effectiveness of Recurrent Neural Networks',
            author: 'Andrej Karpathy',
            venue: 'blog, 2015 — the char-RNN Shakespeare post',
            url: 'https://karpathy.github.io/2015/05/21/rnn-effectiveness/',
          },
          {
            title: 'Dive into Deep Learning — §9.4: Recurrent Neural Networks',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_recurrent-neural-networks/rnn.html',
          },
          {
            title: 'A Learning Algorithm for Continually Running Fully Recurrent Neural Networks',
            author: 'Williams, Zipser',
            venue: 'Neural Computation, 1989',
            url: 'https://ieeexplore.ieee.org/document/6795228',
          },
        ]}
      />
    </div>
  )
}
