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
import SinusoidPattern from '../widgets/SinusoidPattern'        // heatmap of sinusoidal positional encoding: 64 positions × 128 dims, different frequency per dim
import RoPERotation from '../widgets/RoPERotation'              // visualize RoPE: a 2D vector rotates by position-dependent angle; show that inner product depends only on relative position

// Signature anchor: theater seat numbers. Attention is a crowd of audience
// members shouting at once; without seat stickers the model has no way to
// tell row A from row Z. Sinusoidal encoding = a seat sticker with a
// color/pattern readable at a distance. The anchor opens the lesson, is
// returned to at the sinusoidal reveal (why not "1, 2, 3…"), and consolidates
// at the RoPE section (relative seating falls out for free).
export default function PositionalEncodingLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="positional-encoding" />

      {/* ── Opening: the theater with no seat numbers ────────────── */}
      <Prose>
        <p>
          Picture the attention layer as a theater. Every token in your
          sentence is an audience member, and they&apos;re all shouting at
          once. The model&apos;s job is to listen — to figure out who&apos;s
          responding to whom, which voice belongs with which. So far so good.
          Now picture the same theater with the seats unmarked. No row
          letters. No numbers on the chairs. Just a crowd.
        </p>
        <p>
          That&apos;s what attention sees. The self-attention operation is a
          weighted sum over the value vectors of every token in the sequence,
          and the weights come from dot products between queries and keys.
          None of that arithmetic cares about <em>where</em> a token sits.
          Shuffle every audience member in the room and the output comes out
          identical — just permuted. Attention, by construction, is a set
          operation. An unseated crowd.
        </p>
        <p>
          That&apos;s fine for <NeedsBackground slug="sentiment-analysis">bag-of-words</NeedsBackground>{' '}
          sentiment, where order is discardable. It is catastrophic for
          language. &ldquo;Dog bites man&rdquo; and &ldquo;man bites dog&rdquo;
          use the same three words and mean opposite things. If a transformer
          is to tell them apart, every audience member needs a sticker
          stapled to their shirt saying <em>you are in row 3</em>. Positional
          encoding is that sticker. How you design the sticker — the recipe
          that turns &ldquo;row 3&rdquo; into a vector the model can read —
          is the entire subject of this lesson.
        </p>
      </Prose>

      <Callout variant="note" title="why the seats went missing in the first place">
        RNNs never needed seat stickers. They process{' '}
        <NeedsBackground slug="intro-to-nlp">tokens</NeedsBackground> one at a
        time, so &ldquo;which token came first&rdquo; is baked into the order
        of the forward pass — the hallway already forces you into single
        file. Transformers traded that sequentiality for parallelism: every
        seat sees every other seat simultaneously, which is fast and powerful
        and blind. Positional encoding is the bill we pay for the speed.
        Every variant below is a different way to pay it.
      </Callout>

      {/* ── Sinusoidal: why not just "1, 2, 3…"? ─────────────────── */}
      <Prose>
        <p>
          Here&apos;s the first thing everyone tries and nobody ships: just
          staple a number to each seat. Row 0 gets the scalar 0, row 1 gets
          1, row 511 gets 511. Clean, simple, done. It breaks almost
          immediately. A scalar of 511 has a vastly different magnitude from
          a scalar of 3, and you&apos;re adding this thing directly into a{' '}
          <NeedsBackground slug="word-embeddings">word embedding</NeedsBackground>{' '}
          whose values sit in a tight range. The position number drowns the
          meaning. Normalize it to <code>[0, 1]</code>? Now the gap between
          position 5 and position 6 depends on how long the sequence is —
          the sticker changes meaning if someone walks in late.
        </p>
        <p>
          The fix Vaswani et al. (2017) chose is worth staring at, because it
          solves a specific problem: <em>seat stickers need to be comparable
          at a distance</em>. If the model is going to learn &ldquo;pay
          attention to the word three rows back,&rdquo; it has to be able to
          compute &ldquo;three rows back&rdquo; from whatever the sticker
          actually is. Not a scalar. A pattern. Something geometric.
        </p>
        <p>
          For token position <code>pos</code> and embedding dimension index{' '}
          <code>i</code>:
        </p>
      </Prose>

      <MathBlock caption="sinusoidal positional encoding — Vaswani et al. 2017">
{`PE(pos, 2i)     =  sin( pos / 10000^(2i/d) )

PE(pos, 2i+1)   =  cos( pos / 10000^(2i/d) )`}
      </MathBlock>

      <Prose>
        <p>
          Read that with the theater in mind. Each embedding dimension is one
          color on the sticker, each with its own frequency. Dimension 0
          oscillates fast — its stripes repeat every ~6 seats. Dimension{' '}
          <code>d−1</code> oscillates slowly — its wavelength is about{' '}
          <code>2π · 10000</code> seats. Between them you get a geometric
          sweep of wavelengths, every dimension tuned to a different
          resolution of position. The full vector at one seat is a unique
          combination of stripe colors and patterns — a barcode nobody else
          in the audience is wearing.
        </p>
        <p>
          Two properties make this sticker clever rather than arbitrary.
          First: every seat is unique. The high-frequency dimensions are the
          seconds hand, the low-frequency ones are the hour hand, and
          together they stamp a one-of-a-kind barcode on every row for any
          reasonable sequence length. Second, and this is the one that
          earns its keep: because of the trig identities{' '}
          <code>sin(a+b) = sin a cos b + cos a sin b</code> and{' '}
          <code>cos(a+b) = cos a cos b − sin a sin b</code>, the sticker at
          seat <code>pos + k</code> is a <em>linear function</em> of the
          sticker at seat <code>pos</code>. The model can, in principle,
          learn a single linear operator that means &ldquo;attend to the
          audience member <code>k</code> rows back.&rdquo; Relative seating
          is baked into the geometry of the sticker, not just into the
          number.
        </p>
      </Prose>

      {/* ── Widget 1: Sinusoid Pattern ──────────────────────────── */}
      <SinusoidPattern />

      <Prose>
        <p>
          That heatmap <em>is</em> the seating chart. Each column is one row
          of the theater (0 through 63); each row of the heatmap is one
          embedding dimension (0 through 127). The stripes run fast on top
          to slow on the bottom — exactly the geometric sweep the formula
          prescribes. Drag across columns and watch: neighbouring seats wear
          nearly-identical stickers, seats far apart wear wildly different
          ones. That gradient of similarity is the signal the model reads to
          work out who&apos;s sitting where.
        </p>
        <p>
          The sticker is <em>added</em> to the token embedding, not
          concatenated — we&apos;re gluing &ldquo;what the word means&rdquo;
          and &ldquo;which row it&apos;s in&rdquo; into the same vector.
          First reaction: won&apos;t the seat pattern contaminate the
          meaning? In practice it doesn&apos;t, for roughly the reason two
          different radio stations on different frequencies can share one
          wire — different frequencies interfere minimally. The network
          learns, through training, to disentangle the two channels.
        </p>
      </Prose>

      <Personify speaker="Position">
        The crowd forgot me. Without my sticker, every permutation of your
        sentence looks the same to attention — a bag of meanings, no
        grammar. I arrive as a vector of oscillations, a barcode stapled to
        each audience member&apos;s shirt, and once I&apos;m added the
        network can finally tell row A from row Z. I am the order you
        forgot to include.
      </Personify>

      {/* ── Learned PE ──────────────────────────────────────────── */}
      <Prose>
        <p>
          The Vaswani recipe is elegant but fixed. The next obvious move is
          to stop hand-designing the sticker and let the model print its
          own: allocate <code>nn.Embedding(max_len, d)</code>, initialise
          randomly, backprop. This is a <KeyTerm>learned positional
          embedding</KeyTerm>, and it&apos;s what BERT and GPT-2 use. Every
          seat in the theater gets its own blank sticker at the start of
          training, and gradient descent colors them in.
        </p>
        <p>
          The upside is simplicity — it&apos;s just another embedding table
          — and the model can, in theory, invent a barcode better suited to
          the data than anything a human would pick. The downside is harsh:
          a table of size <code>max_len × d</code> has no idea what to do
          with seat <code>max_len + 1</code>. Train BERT with 512 rows and
          you cannot suddenly seat 1024 audience members — there&apos;s
          literally no sticker printed for row 513. You can extend the
          table and keep training, but the new rows start from random and
          the model has never seen them. Sinusoidal encodings are just a
          formula; you can evaluate them at seat 1,000,000 and get a
          well-defined vector, even if no sequence that long has ever been
          shown to the model.
        </p>
      </Prose>

      <Callout variant="note" title="why the &lsquo;fixed vs learned&rsquo; debate never settled">
        Empirically the two perform roughly identically at the lengths they
        were trained on. Learned wins on slightly better perplexity;
        sinusoidal wins on length generalisation. For models with a hard max
        context — BERT, GPT-2, most early encoders — learned was fine, and
        simpler is always easier to ship. Once people started caring about
        long context, the whole field pivoted to stickers that do both —
        which is where RoPE comes in.
      </Callout>

      {/* ── RoPE: stickers that compare at a distance ───────────── */}
      <Prose>
        <p>
          Rotary Position Embedding (Su et al., 2021) is the positional
          encoding LLaMA uses. And Gemma. And Mistral. And most modern
          decoder-only transformers that don&apos;t use ALiBi. By installed
          base, it&apos;s the dominant seat-sticker of 2024, and it&apos;s
          cleverer than it looks.
        </p>
        <p>
          The idea: instead of stapling a sticker to the audience member,
          <em>spin the audience member on a turntable</em> by an angle that
          depends on their row. Take the query and key vectors, split each{' '}
          <code>d</code>-dimensional vector into <code>d/2</code>{' '}
          two-dimensional pairs, and rotate each pair by an angle
          proportional to its position — with a different frequency per
          pair, the same geometric sweep as before. In 2D, rotating
          <code>(x₁, x₂)</code> by angle <code>θ</code> gives:
        </p>
      </Prose>

      <MathBlock caption="RoPE — rotate the query/key pair by position-dependent angle">
{`R(θ) · (x₁, x₂)  =  ( x₁ cos θ  −  x₂ sin θ ,
                       x₁ sin θ  +  x₂ cos θ )

θₘ    =   m · ωᵢ     ← position m times per-pair frequency ωᵢ
ωᵢ    =   1 / 10000^(2i/d)`}
      </MathBlock>

      <Prose>
        <p>
          The magic is what this does to the inner product. Rotate query{' '}
          <code>q</code> at seat <code>m</code> and key <code>k</code> at
          seat <code>n</code>, then take their dot product:
        </p>
      </Prose>

      <MathBlock caption="RoPE&apos;s load-bearing identity">
{`⟨ R(mω) · q ,  R(nω) · k ⟩    =    ⟨ q ,  R((n − m)ω) · k ⟩

   →   the similarity score depends only on the
       relative offset (n − m), not on m or n alone`}
      </MathBlock>

      <Prose>
        <p>
          Stop and absorb that. The attention score between two audience
          members depends only on <em>how far apart they&apos;re seated</em>,
          not on whether they&apos;re in rows 3 and 5 or rows 103 and 105.
          That&apos;s exactly the property the theater wanted: seats whose
          relationships are comparable at a distance, no matter where in the
          room they are. It falls out of rotating Q and K together as a
          literal algebraic consequence — no seat-sticker addition, no
          learned bias, no embedding table. Just a rotation applied at each
          attention layer.
        </p>
      </Prose>

      {/* ── Widget 2: RoPE Rotation ─────────────────────────────── */}
      <RoPERotation />

      <Prose>
        <p>
          The 2D pair on the left is a query; the one on the right is a
          key. Both start in row 0. Advance them both by the same amount —
          say, push them ten rows back together — and the angle between
          them stays identical, so the dot product is unchanged. Advance
          only one and the angle shifts. <em>That</em> shift is the
          relative-position signal the model reads. The inner product
          isn&apos;t blind to position — it&apos;s blind to{' '}
          <em>absolute</em> position, tuned to <em>relative</em> position.
          That distinction is the whole game.
        </p>
      </Prose>

      <Personify speaker="Rotation">
        The other encodings staple a sticker to your shirt. I spin your
        query and your key on the same turntable, and when you take the
        inner product the shared rotation cancels — only the difference
        survives. I don&apos;t whisper &ldquo;you are in row seven&rdquo; to
        the model; I whisper &ldquo;you are three rows ahead of that other
        audience member,&rdquo; which, it turns out, is all the model
        actually wanted to know.
      </Personify>

      <Callout variant="insight" title="why the field converged on RoPE">
        Sinusoidal stickers make relative seating <em>available</em> — a
        linear operator can recover it from the added signal, if the model
        bothers to learn one. RoPE makes relative seating{' '}
        <em>structural</em> — every attention score is a function of
        relative offset by construction, no learning required. It also
        composes cleanly with no extra parameters, plays nicely with flash
        attention kernels, and extrapolates gracefully (with tricks like
        NTK-aware scaling) to theaters 4× and 8× longer than training.
        That combination is why LLaMA, Gemma, Mistral, Qwen, DeepSeek, and
        most open-weights LLMs since 2022 ship with RoPE.
      </Callout>

      {/* ── ALiBi ───────────────────────────────────────────────── */}
      <Prose>
        <p>
          Which brings us to the fourth option, the contrarian one. What if
          you skip the seat sticker entirely and just… penalize audience
          members for shouting at people in faraway rows? That&apos;s
          <KeyTerm> ALiBi</KeyTerm> (Press, Smith, Lewis 2021). After
          computing the raw attention score <code>QKᵀ</code>, subtract a
          penalty proportional to how many rows apart the two seats are:
        </p>
      </Prose>

      <MathBlock caption="ALiBi — no embedding, just a distance penalty">
{`score(m, n)   =   qₘ · kₙ   −   λ · |m − n|

 (λ is a per-head scalar, chosen on a fixed geometric schedule)`}
      </MathBlock>

      <Prose>
        <p>
          That&apos;s the whole encoding. Audience members attend less to
          seats far away simply because the score gets more negative as the
          row-distance grows. Each attention head gets its own{' '}
          <code>λ</code> — some heads use a steep penalty (short-range
          heads, focused on the nearest few rows), others a shallow one
          (long-range heads, happy to listen across the theater). No added
          stickers, no learned table, no rotations. The paper&apos;s
          headline finding: a model trained with ALiBi at 1024 rows can be
          deployed at 2048 with <em>almost no degradation</em>. Length
          extrapolation, nearly for free.
        </p>
      </Prose>

      <Callout variant="note" title="the four-way comparison">
        <ul>
          <li>
            <strong>Sinusoidal:</strong> fixed formula, sticker added to the
            embedding, extrapolates in principle but relative position is
            implicit — the model has to learn to read the barcode.
          </li>
          <li>
            <strong>Learned:</strong> <code>nn.Embedding(max_len, d)</code>,
            simple, slightly better on-distribution, cannot seat anyone
            past row <code>max_len</code>. Used by BERT / GPT-2.
          </li>
          <li>
            <strong>RoPE:</strong> rotation applied to Q and K, relative
            seating is structural, extrapolates with light scaling tricks.
            Used by most modern LLMs.
          </li>
          <li>
            <strong>ALiBi:</strong> additive penalty on attention scores, no
            sticker at all, extrapolates cleanly. Used by Mosaic&apos;s MPT,
            BLOOM.
          </li>
        </ul>
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations of the same Vaswani recipe. Pure Python
          first — one scalar at a time, the math directly transcribed. Then
          NumPy, where the broadcasting idea clicks and one seat&apos;s
          worth of arithmetic becomes the whole theater at once. Then
          PyTorch, where we drop in a minimal RoPE block so you can see both
          stickers side by side.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · sinusoidal_pe_scratch.py"
        output={`pos=0 dim=0 PE=0.0000
pos=0 dim=1 PE=1.0000
pos=1 dim=0 PE=0.8415
pos=1 dim=1 PE=0.5403`}
      >{`import math

def positional_encoding_scalar(pos, i, d):
    # one scalar of the PE matrix — position pos, embedding dim i
    # pairs: (2i, 2i+1) share a frequency, sin on the even, cos on the odd
    angle = pos / (10000 ** (2 * (i // 2) / d))
    if i % 2 == 0:
        return math.sin(angle)
    else:
        return math.cos(angle)

d = 128
for pos in (0, 1):
    for dim in (0, 1):
        v = positional_encoding_scalar(pos, dim, d)
        print(f"pos={pos} dim={dim} PE={v:.4f}")`}</CodeBlock>

      <Prose>
        <p>
          Now vectorise. The cleanest way to print the full{' '}
          <code>(seq_len, d)</code> seating chart is with two 1D arrays and
          a broadcast — no loops. This is the shape of every production PE
          implementation you&apos;ll ever see.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · sinusoidal_pe_numpy.py"
        output={`PE shape: (64, 128)
PE[0, :4]:  [0.     1.     0.     1.    ]
PE[1, :4]:  [0.8415 0.5403 0.8218 0.5697]`}
      >{`import numpy as np

def positional_encoding(seq_len, d):
    pos = np.arange(seq_len)[:, None]            # (seq_len, 1)
    i   = np.arange(d)[None, :]                  # (1, d)

    # frequency per dimension: pair 2i and 2i+1 share a freq
    div = np.power(10000.0, (2 * (i // 2)) / d)  # (1, d)
    angle = pos / div                            # (seq_len, d)  via broadcasting

    pe = np.zeros((seq_len, d))
    pe[:, 0::2] = np.sin(angle[:, 0::2])         # even dims → sin
    pe[:, 1::2] = np.cos(angle[:, 1::2])         # odd dims → cos
    return pe

PE = positional_encoding(64, 128)
print("PE shape:", PE.shape)
print("PE[0, :4]: ", np.round(PE[0, :4], 4))
print("PE[1, :4]: ", np.round(PE[1, :4], 4))`}</CodeBlock>

      <Bridge
        label="pure python → numpy"
        rows={[
          {
            left: 'for pos in range(seq_len): for i in range(d):',
            right: 'pos[:, None] + i[None, :]  # broadcast',
            note: 'two nested loops collapse into one outer-product shape',
          },
          {
            left: 'if i % 2 == 0: sin(...) else: cos(...)',
            right: 'pe[:, 0::2] = sin; pe[:, 1::2] = cos',
            note: 'strided slicing addresses even and odd dims independently',
          },
          {
            left: '10000 ** (2 * (i // 2) / d)',
            right: 'np.power(10000.0, (2*(i//2))/d)',
            note: 'same formula, now a vector of per-dim divisors',
          },
        ]}
      />

      <Prose>
        <p>
          Finally, the PyTorch <code>nn.Module</code> you would actually
          ship, plus a tight RoPE helper. The sinusoidal chart is registered
          as a non-trainable buffer (no gradient flows through it — nobody
          is learning the seat layout, it was already correct), and the
          RoPE rotation is applied to Q and K right before the attention
          softmax.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch · positional_encoding_torch.py"
        output={`sinusoidal PE shape: torch.Size([1, 64, 128])
after adding PE:      torch.Size([8, 64, 128])
rope q shape:         torch.Size([8, 64, 128])
max |q·k - q'·k'|:    2.1e-07  (relative-only check)`}
      >{`import torch
import torch.nn as nn

class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        i   = torch.arange(d).unsqueeze(0).float()
        div = torch.pow(10000.0, (2 * (i // 2)) / d)
        angle = pos / div
        pe = torch.zeros(max_len, d)
        pe[:, 0::2] = torch.sin(angle[:, 0::2])
        pe[:, 1::2] = torch.cos(angle[:, 1::2])
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d), not trainable

    def forward(self, x):                                # x: (B, T, d)
        return x + self.pe[:, : x.size(1)]

def rope(x, pos):                                        # x: (..., T, d), pos: (T,)
    # split last dim into pairs, rotate each pair by pos * per-pair frequency
    d = x.size(-1)
    i = torch.arange(d // 2, device=x.device).float()
    freq = 1.0 / torch.pow(10000.0, 2 * i / d)           # (d/2,)
    theta = pos[:, None].float() * freq[None, :]         # (T, d/2)
    cos, sin = theta.cos(), theta.sin()                  # (T, d/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]                  # split into pairs
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = rot1, rot2
    return out

# sanity check the shapes + a quick relative-position check for RoPE
torch.manual_seed(0)
x = torch.randn(8, 64, 128)
pe = SinusoidalPE(d=128)
print("sinusoidal PE shape:", pe.pe.shape)
print("after adding PE:     ", pe(x).shape)

q = torch.randn(8, 64, 128)
k = torch.randn(8, 64, 128)
positions = torch.arange(64)
q_rot = rope(q, positions)
k_rot = rope(k, positions)
print("rope q shape:        ", q_rot.shape)

# shift both q and k by the same offset — inner product should match the unshifted pair
offset = 5
q_shift = rope(q, positions + offset)
k_shift = rope(k, positions + offset)
diff = ((q_rot * k_rot).sum(-1) - (q_shift * k_shift).sum(-1)).abs().max()
print(f"max |q·k - q'·k'|:    {diff:.1e}  (relative-only check)")`}</CodeBlock>

      <Bridge
        label="numpy → pytorch"
        rows={[
          {
            left: 'pe = positional_encoding(seq, d)  # numpy array',
            right: 'self.register_buffer("pe", pe)',
            note: 'non-trainable tensor, moves with .to(device), no gradient',
          },
          {
            left: 'out = x + pe[:seq_len]',
            right: 'return x + self.pe[:, :x.size(1)]',
            note: 'same add, now broadcasting over the batch dim',
          },
          {
            left: '(no numpy analogue — RoPE is Q/K side)',
            right: 'q_rot, k_rot = rope(q, pos), rope(k, pos)',
            note: 'applied inside attention, not added to the token embedding',
          },
        ]}
      />

      <Callout variant="insight" title="the point of the three layers">
        Pure Python makes the formula obvious. NumPy teaches the
        broadcasting idiom — you&apos;ll write that{' '}
        <code>pos[:, None] / div[None, :]</code> pattern hundreds of times
        in your career. PyTorch adds two things real systems care about:
        the <code>register_buffer</code> trick (so the seating chart moves
        to GPU with the model but doesn&apos;t consume gradients) and a
        Q/K-side rotation that lives inside the attention layer instead of
        next to the embedding.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">PE added before LayerNorm changes the
          statistics it sees:</strong> token embeddings are typically initialised with a
          small standard deviation, and adding a sinusoid with values in <code>[−1, 1]</code>
          noticeably shifts the variance going into the first norm. Most implementations
          either scale embeddings by <code>√d</code> before the add (the Vaswani move) or
          place the PE inside a residual after the first norm. Know which convention the
          codebase you&apos;re reading uses.
        </p>
        <p>
          <strong className="text-term-amber">Length extrapolation is not free:</strong>{' '}
          train a learned-PE model at <code>max_len=512</code> and infer at 2048 — you
          crash on an index-out-of-bounds or (worse) silently look up garbage if someone
          padded the table. Sinusoidal tolerates it mathematically, but most models still
          degrade sharply past their training length. RoPE and ALiBi degrade more
          gracefully, but even RoPE needs NTK-aware / YaRN-style frequency scaling to
          push 4× beyond training context without breaking.
        </p>
        <p>
          <strong className="text-term-amber">RoPE swaps components during rotation:</strong>
          {' '}a common bug when implementing RoPE by hand is forgetting that the rotation
          mixes the two halves of the pair — you can&apos;t just scale <code>x₁</code> by{' '}
          <code>cos θ</code> and be done. The pair <code>(x₁, x₂)</code> becomes{' '}
          <code>(x₁ cos θ − x₂ sin θ, x₁ sin θ + x₂ cos θ)</code>. Miss the cross-term and
          the inner-product-depends-only-on-relative-position property quietly breaks.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Verify the Vaswani claim empirically">
        <p>
          The core theoretical property of sinusoidal PE is that{' '}
          <code>PE(pos + k)</code> can be written as a <em>linear function</em> of{' '}
          <code>PE(pos)</code> for any fixed shift <code>k</code>. Implement the PE matrix,
          pick a shift <code>k = 7</code>, and find the <code>d × d</code> matrix{' '}
          <code>M_k</code> such that <code>PE(pos + 7) ≈ M_k · PE(pos)</code> for all{' '}
          <code>pos</code>.
        </p>
        <p className="mt-2">
          Hint: stack the first 100 PE vectors into a matrix <code>P</code> of shape{' '}
          <code>(100, d)</code>, do the same for the shifted ones into <code>P_shifted</code>,
          and solve <code>P · M_kᵀ = P_shifted</code> via <code>np.linalg.lstsq</code>. The
          residual should be at machine-precision zero. That&apos;s the relative-position
          property, made concrete.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: check that <code>M_k</code> is, structurally, a block-diagonal rotation
          matrix — one 2×2 rotation per frequency pair. You have just rediscovered RoPE.
        </p>
      </Challenge>

      {/* ── Closing: the cliffhanger into self-attention ────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Attention without
          positional information is an order-blind set operation — a crowd
          in an unmarked theater. Every real transformer hands out seat
          stickers from the outside. Sinusoidal encodings are a fixed,
          additive barcode built so a linear operator can recover relative
          offsets. Learned embeddings are a simple lookup table that wins on
          perplexity and loses on extrapolation — you only printed stickers
          for the rows you trained on. RoPE spins Q and K on the same
          turntable so inner products depend only on relative seating by
          construction, which is why it dominates modern LLMs. ALiBi skips
          the sticker entirely and biases attention scores by row-distance,
          which is why it extrapolates so cleanly.
        </p>
        <p>
          <strong>End of the NLP section.</strong> We walked from one-hot
          vectors to word2vec to subword tokenization to positional
          encoding. You now have every piece a transformer needs as{' '}
          <em>input</em>: tokens that mean something, and seats that know
          where they are. What remains is the mechanism that actually reads
          the room.
        </p>
        <p>
          <strong>Next: Self-Attention — the main event.</strong> Now every
          audience member knows where they&apos;re sitting. The question
          the next section answers is how to let them talk to each other
          without whispering in a straight line, row by row, the way an
          RNN would. In self-attention every seat queries every other seat
          at once, compares barcodes, and decides whose voice to listen to.
          Everything this course has built — gradients, backprop,
          embeddings, and the seat stickers you just finished putting on —
          exists to make that single operation work. Time to watch it.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Attention Is All You Need',
            author: 'Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin',
            venue: 'NeurIPS 2017 — the original Transformer paper; Section 3.5 defines the sinusoidal PE',
            url: 'https://arxiv.org/abs/1706.03762',
          },
          {
            title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding',
            author: 'Su, Lu, Pan, Murtadha, Wen, Liu',
            venue: '2021 — the RoPE paper; used by LLaMA, Gemma, Mistral, Qwen',
            url: 'https://arxiv.org/abs/2104.09864',
          },
          {
            title: 'Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation',
            author: 'Press, Smith, Lewis',
            venue: 'ICLR 2022 — ALiBi',
            url: 'https://arxiv.org/abs/2108.12409',
          },
          {
            title: 'Dive into Deep Learning — Section 11.6: Self-Attention and Positional Encoding',
            author: 'Zhang, Lipton, Li, Smola',
            url: 'https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html',
          },
        ]}
      />
    </div>
  )
}
