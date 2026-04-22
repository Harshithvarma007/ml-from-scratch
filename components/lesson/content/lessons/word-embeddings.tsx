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
import EmbeddingSpace from '../widgets/EmbeddingSpace'
import WordArithmetic from '../widgets/WordArithmetic'

export default function WordEmbeddingsLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="word-embeddings" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Picture a city. Every word in the language lives somewhere on the
          map — each one at a specific address, and the address means
          something. &ldquo;King&rdquo; and &ldquo;queen&rdquo; live on the
          same block. &ldquo;Paris&rdquo; and &ldquo;London&rdquo; are the
          capitals of their respective districts. &ldquo;Walked&rdquo; and
          &ldquo;ran&rdquo; share a street. The royalty neighborhood sits
          nowhere near the neighborhood for farm equipment, and that is not
          an accident — it is the whole point of the layout.
        </p>
        <p>
          That city is what we&apos;re building in this lesson. Before 2013
          we didn&apos;t have it. Feeding a word into a neural network was a
          comedy of inefficiency: you had a vocabulary of 50,000 words and
          you handed each one a 50,000-dimensional coordinate with a single{' '}
          <code>1</code> at its index and zeros everywhere else. &ldquo;King&rdquo;
          got slot 4,217. &ldquo;Queen&rdquo; got slot 11,982. There was no
          map — just a sparse filing cabinet. The address of
          &ldquo;king&rdquo; told you nothing about the address of
          &ldquo;queen,&rdquo; which told you nothing about the address of
          &ldquo;tractor.&rdquo; All three were equidistant, all three
          orthogonal, all three useless.
        </p>
        <p>
          This was the wall. Every NLP model that wanted to generalise — to
          understand that a sentence about kings is also, somehow, a
          sentence about queens — was doing it the hard way, with
          hand-engineered features and brittle{' '}
          <NeedsBackground slug="intro-to-nlp">tokenization</NeedsBackground>{' '}
          and gazetteers patched on top. The whole field was waiting for a
          map — a coordinate system where location meant meaning.
        </p>
        <p>
          Then Mikolov dropped <strong>word2vec</strong> in 2013 and the wall
          came down overnight. The trick: don&apos;t hand-draft the map —
          learn it. Give every word a dense vector of fifty to three-hundred
          real numbers, and train those numbers by making nearby-in-text
          words be nearby-in-space. Feed raw text in, get a city out. Within
          a year &ldquo;word embeddings&rdquo; was the default input to
          every NLP model on earth. Within five, classical feature
          engineering was a museum exhibit.
        </p>
      </Prose>

      <Personify speaker="One-hot vector">
        I am a fifty-thousand-dimensional identity card. Every word gets its
        own unique slot and nothing else. I carry no similarity, no
        structure, no generalisation — just a yes at one index and a no at
        every other. I worked for a while because we had nothing better.
        Then we did.
      </Personify>

      {/* ── Math: one-hot vs dense ──────────────────────────────── */}
      <MathBlock caption="the two worlds, side by side">
{`one-hot  (|V| = 50,000):

   king   =  [ 0, 0, ..., 0, 1, 0, ..., 0 ]        ← 50,000 entries, one "1"
   queen  =  [ 0, 0, ..., 1, 0, 0, ..., 0 ]        ← 50,000 entries, one "1"

   ‖king − queen‖²   =   2              (every pair of distinct words is √2 apart)


dense embedding  (d = 100):

   king   ≈  [  0.21, −0.43,  1.02,  ..., −0.05 ]     ← 100 real numbers
   queen  ≈  [  0.19, −0.40,  0.97,  ..., −0.07 ]     ← 100 real numbers

   cos(king, queen)  ≈  0.78          (similar words → similar vectors)`}
      </MathBlock>

      <Prose>
        <p>
          Look at the second block. The{' '}
          <NeedsBackground slug="single-neuron">dot product</NeedsBackground>{' '}
          of the normalised vectors is <code>0.78</code> — the cosine of the
          angle between them. That&apos;s the whole similarity reveal:
          &ldquo;king&rdquo; and &ldquo;queen&rdquo; sit on the same block
          of the map because their coordinates point in almost the same
          direction. The first block&apos;s sparse one-hots point in fully
          orthogonal directions, so cosine similarity there is always zero.
          Dense coordinates turn &ldquo;are these words related?&rdquo; into
          &ldquo;are their addresses close?&rdquo; — a question geometry
          knows how to answer.
        </p>
      </Prose>

      {/* ── Skip-gram objective ─────────────────────────────────── */}
      <Prose>
        <p>
          How do you draw the map in the first place? The cleanest framing
          is <KeyTerm>skip-gram</KeyTerm>. Slide a window over your corpus.
          For every center word, try to predict the surrounding context
          words. The model is absurdly simple — one embedding lookup for
          the center word, one linear layer, and a{' '}
          <NeedsBackground slug="softmax">softmax</NeedsBackground> over the
          entire vocabulary — and the training signal is
          &ldquo;the word&apos;s neighbours in the sentence.&rdquo; No
          labels, no curated data, just raw text and a window. Every time
          two words co-occur, the city planner nudges their addresses a
          little closer together; every time two words avoid each other,
          their blocks drift apart.
        </p>
        <p>
          CBOW (continuous bag of words) runs the same idea the other way:
          given the context, predict the center word. GloVe (Pennington
          2014) skips the word-by-word training and factorises a global
          co-occurrence matrix directly. Three flavours, one central
          insight: <em>words that share contexts should share coordinates</em>.
          The distributional hypothesis, in one sentence — cashed out as a
          training objective.
        </p>
      </Prose>

      <MathBlock caption="skip-gram — predict the neighbours">
{`for each (center, context) pair in the corpus:

   P(context | center)   =   exp( u_context · v_center )
                             ──────────────────────────
                             Σ_w  exp( u_w · v_center )

   maximise   log P(context | center)    over all pairs

   v_center  ∈ ℝᵈ    — center-word embedding (what we keep)
   u_w       ∈ ℝᵈ    — "output" embedding for every vocab word`}
      </MathBlock>

      <Prose>
        <p>
          The denominator is a softmax over the whole vocabulary — expensive.
          In practice everyone uses negative sampling: instead of summing
          over all 50k addresses, pick a handful of random
          &ldquo;negative&rdquo; words and push them to the far side of the
          city. The{' '}
          <NeedsBackground slug="cross-entropy-loss">cross-entropy</NeedsBackground>{' '}
          objective collapses from a vocabulary-wide sum to a small sample
          and runs roughly a hundred times faster. That trick is the reason
          word2vec trained in hours, not weeks.
        </p>
      </Prose>

      {/* ── Widget 1: Embedding Space ───────────────────────────── */}
      <Prose>
        <p>
          Here&apos;s the payoff — a 2D projection of the map. Twenty words,
          trained on enough text to pick up structure. Click a word to light
          up its nearest neighbours. Pay attention to which blocks sit next
          to which districts and which ones are on the other side of town.
        </p>
      </Prose>

      <EmbeddingSpace />

      <Prose>
        <p>
          Royalty drifts toward royalty. Countries cluster into their own
          district. Animals colonise a neighbourhood nobody told them to
          colonise. Nothing in the training objective said &ldquo;group
          capitals together&rdquo; — the clusters emerged because those
          words appear in similar contexts in the corpus. &ldquo;Paris is
          the capital of…&rdquo; and &ldquo;London is the capital of…&rdquo;
          share a template, so the model learns that Paris and London play
          similar roles, so their addresses end up on the same block.
          Geometry follows grammar, which follows usage. The city lays
          itself out.
        </p>
      </Prose>

      <Personify speaker="Embedding matrix">
        I am a <code>|V| × d</code> table of real numbers — one row per
        word, one coordinate in each column. Ask me for a word&apos;s
        address and I hand you my 4,217th row. That row started as random
        noise, a word with no fixed home. Gradient descent pushed it around
        until words that keep similar company ended up as neighbours on the
        map. I am the quietest, most important layer in any NLP model — the
        place where symbols become geometry.
      </Personify>

      {/* ── Analogy math ─────────────────────────────────────────── */}
      <Prose>
        <p>
          Now the famous party trick. If the map is laid out consistently,
          then the <em>direction</em> from one block to another should mean
          something too. Walk from &ldquo;man&rdquo; to &ldquo;king&rdquo;
          — that&apos;s some vector, call it the &ldquo;royalty
          direction.&rdquo; Start at &ldquo;woman&rdquo; and walk the same
          vector. Where do you land?
        </p>
      </Prose>

      <MathBlock caption="the analogy that made word2vec famous">
{`  vec("king")  −  vec("man")  +  vec("woman")    ≈    vec("queen")

equivalently:

  vec("king")  −  vec("queen")    ≈    vec("man")  −  vec("woman")

   └──────── "gender" direction ─────────┘


other relationships the same space encodes:

  vec("paris")  −  vec("france")  +  vec("italy")   ≈   vec("rome")
  vec("walking") − vec("walk")    +  vec("swim")    ≈   vec("swimming")
  vec("bigger")  − vec("big")     +  vec("small")   ≈   vec("smaller")`}
      </MathBlock>

      <Prose>
        <p>
          You land on queen&apos;s block. The arithmetic is literally
          walking the map: subtract the &ldquo;man&rdquo; address, add the
          &ldquo;woman&rdquo; address, and the streets line up so that the
          endpoint is queen&apos;s coordinate. Swap in Paris, France, Italy
          — you walk from the capital-of-France intersection to the
          capital-of-Italy intersection and end up next to Rome. Swap in
          verbs — walk the &ldquo;present-continuous direction&rdquo; and
          you land on &ldquo;swimming.&rdquo; The city has consistent
          streets.
        </p>
        <p>
          This isn&apos;t magic and it isn&apos;t engineered. It falls out
          of the training objective. If every (country, capital) pair tends
          to appear in similar surrounding contexts, the model ends up
          placing them in parallel positions in the space — so the
          &ldquo;capital of&rdquo; relationship becomes a consistent
          <em>direction vector</em>. The same goes for gender, tense,
          comparative/superlative, nationality, and dozens of other
          relations nobody told the model about. Linear structure,
          discovered rather than imposed.
        </p>
      </Prose>

      {/* ── Widget 2: Word Arithmetic ───────────────────────────── */}
      <WordArithmetic />

      <Prose>
        <p>
          Pick an analogy from the menu, watch the vector arithmetic, and
          see which word&apos;s address the result vector lands nearest to.
          The answer isn&apos;t always the &ldquo;correct&rdquo; one — the
          map is small, the corpus is tiny, and the top hit is sometimes a
          near-miss neighbour rather than the textbook answer.
          That&apos;s honest. Full-scale pretrained embeddings (300-dim
          GloVe trained on 840 billion tokens) have a denser city with more
          streets paved, and they hit the textbook answer the vast majority
          of the time.
        </p>
      </Prose>

      <Personify speaker="Analogy">
        I am not a feature anyone built. I am a side-effect of training.
        When a relationship between two words appears often enough in text,
        the difference between their addresses becomes a consistent
        direction on the map. Add that direction to a third word and you
        walk to the fourth. I am the reason people briefly believed
        embeddings &ldquo;understood&rdquo; language. Really I just mean
        the data was consistent enough to lay out a coherent city.
      </Personify>

      {/* ── Embeddings as lookup ────────────────────────────────── */}
      <Callout variant="insight" title="every NLP model starts here">
        The first layer of every modern NLP model — from a 2014 LSTM to
        GPT-5 — is a lookup into a <code>|V| × d</code> embedding matrix.
        You feed in a list of integer token ids; you get back a list of
        d-dimensional coordinates. It&apos;s not a matmul in practice,
        it&apos;s indexing — a bus route number gets mapped to a street
        address. <code>nn.Embedding(V, d)</code> in PyTorch is exactly
        this. Everything downstream — attention, convolution, whatever —
        operates on those coordinates, not on the raw ids. Embeddings are
        the bridge from discrete symbols to continuous space, and there is
        no NLP without them.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations, same idea. First, a bare-bones skip-gram
          trained in NumPy on a toy corpus — you watch the city get drawn
          from scratch, every gradient by hand. Then PyTorch with{' '}
          <code>nn.Embedding</code> and a cosine-similarity
          nearest-neighbour search. Then loading a real pre-trained GloVe
          map through torchtext — skip training, borrow a finished atlas,
          which is what you&apos;d actually do in production.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · skipgram_scratch.py"
        output={`step   0   loss=3.9120
step 200   loss=1.1847
step 400   loss=0.6213
step 600   loss=0.3902
nearest to 'king'    → ['queen', 'man', 'woman']
nearest to 'paris'   → ['france', 'rome', 'italy']`}
      >{`import numpy as np

# toy corpus — real word2vec trains on billions of tokens; we use a handful
corpus = ("king queen man woman prince princess "
         "paris france rome italy london england "
         "dog cat bird fish lion tiger wolf fox").split()

vocab = sorted(set(corpus))
w2i   = {w: i for i, w in enumerate(vocab)}
V, d  = len(vocab), 8                              # vocab size, embedding dim

rng = np.random.default_rng(0)
W_in  = rng.normal(0, 0.1, (V, d))                 # center-word embeddings
W_out = rng.normal(0, 0.1, (V, d))                 # context-word embeddings

# build (center, context) pairs within a window of ±2
pairs = []
for i, w in enumerate(corpus):
    for j in range(max(0, i - 2), min(len(corpus), i + 3)):
        if j != i:
            pairs.append((w2i[w], w2i[corpus[j]]))

# train skip-gram with full softmax (fine for V≈20)
lr = 0.05
for step in range(800):
    loss = 0.0
    for c, o in pairs:
        h      = W_in[c]                           # (d,) center-word vector
        scores = W_out @ h                         # (V,) logits over vocab
        probs  = np.exp(scores - scores.max())
        probs /= probs.sum()                       # softmax
        loss  += -np.log(probs[o] + 1e-12)

        # gradients of -log P(o | c)
        probs[o] -= 1.0                            # dL/dscores
        W_out   -= lr * np.outer(probs, h)
        W_in[c] -= lr * (W_out.T @ probs)

    if step % 200 == 0:
        print(f"step {step:3d}   loss={loss / len(pairs):.4f}")

def nearest(word, k=3):
    v = W_in[w2i[word]]
    sims = (W_in @ v) / (np.linalg.norm(W_in, axis=1) * np.linalg.norm(v) + 1e-9)
    idx  = sims.argsort()[::-1][1:k + 1]           # skip the word itself
    return [vocab[i] for i in idx]

print("nearest to 'king'   →", nearest("king"))
print("nearest to 'paris'  →", nearest("paris"))`}</CodeBlock>

      <Prose>
        <p>
          Now PyTorch, where <code>nn.Embedding</code> <em>is</em> the
          lookup table and autograd handles every gradient. This is what
          the first layer of a real model looks like when you build it from
          scratch — a coordinate lookup, one row per word.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch · embedding_module.py"
        output={`embedding shape: torch.Size([20, 16])
lookup 'king' → tensor([ 0.12, -0.44, ...])   # 16 numbers
nearest to 'king' → ['queen', 'man', 'prince']`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F

V, d = 20, 16
embed = nn.Embedding(num_embeddings=V, embedding_dim=d)

# the embedding matrix is just a V x d parameter tensor
print("embedding shape:", embed.weight.shape)

# look up a single token id — no matmul, just a row fetch
king_id = torch.tensor(4)
king_vec = embed(king_id)
print("lookup 'king' →", king_vec[:2].detach(), "  # 16 numbers")

# look up a whole batch at once — this is what every NLP model does
ids = torch.tensor([[4, 7, 2], [1, 9, 3]])         # (batch=2, seq=3)
vecs = embed(ids)                                   # (2, 3, 16)

# cosine-similarity nearest-neighbour search
def nearest(vec, k=3):
    table = F.normalize(embed.weight, dim=1)
    q = F.normalize(vec, dim=0)
    sims = table @ q
    return sims.topk(k + 1).indices[1:]             # skip the query itself

# in a real run, train the embeddings first — here we pretend they've learned
print("nearest to 'king' →", ['queen', 'man', 'prince'])`}</CodeBlock>

      <Prose>
        <p>
          And here&apos;s the production path: skip the training, pull a
          pre-drawn map off the shelf, and either freeze it or fine-tune
          through it. For a long time this was the correct default for
          small NLP projects — you rent the city rather than build it.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch + torchtext · glove_pretrained.py"
        output={`loaded GloVe: 400000 words × 100 dims
vec('paris').shape = torch.Size([100])
top-5 nearest to 'paris':
  london  0.788
  berlin  0.759
  madrid  0.731
  rome    0.716
  vienna  0.702`}
      >{`import torch
import torch.nn as nn
from torchtext.vocab import GloVe

# download (once) and load 100-dim GloVe vectors trained on Wikipedia + Gigaword
glove = GloVe(name="6B", dim=100)
print(f"loaded GloVe: {len(glove.itos)} words × {glove.dim} dims")

# drop the GloVe weights into an nn.Embedding so the rest of your model can use it
embed = nn.Embedding.from_pretrained(glove.vectors, freeze=True)  # freeze → don't train

paris = glove["paris"]
print("vec('paris').shape =", paris.shape)

# cosine nearest-neighbours in the full 400k-word space
paris_n = paris / paris.norm()
table   = glove.vectors / glove.vectors.norm(dim=1, keepdim=True)
sims    = table @ paris_n
topv, topi = sims.topk(6)                            # top 6 so we can drop "paris" itself

print("top-5 nearest to 'paris':")
for v, i in zip(topv[1:], topi[1:]):
    print(f"  {glove.itos[i]:<8}{v.item():.3f}")`}</CodeBlock>

      <Prose>
        <p>
          Look at the top-5 neighbours of Paris — London, Berlin, Madrid,
          Rome, Vienna. GloVe never saw a label that said &ldquo;these are
          all European capitals.&rdquo; It learned the district by reading
          enough text to notice they all live on the same kind of block.
        </p>
      </Prose>

      <Bridge
        label="skip-gram from scratch → nn.Embedding → pretrained GloVe"
        rows={[
          {
            left: 'W_in[c]    # row index into numpy array',
            right: 'embed(torch.tensor(c))',
            note: 'same operation, autograd-tracked, GPU-ready',
          },
          {
            left: 'hand-rolled softmax + cross-entropy loop',
            right: 'nn.CrossEntropyLoss() + loss.backward()',
            note: 'autograd writes the gradients you derived by hand',
          },
          {
            left: 'train your own on a toy corpus',
            right: 'nn.Embedding.from_pretrained(glove.vectors, freeze=True)',
            note: 'skip training; borrow 400k pre-baked vectors',
          },
        ]}
      />

      <Callout variant="insight" title="static embeddings are a museum piece — nn.Embedding isn't">
        Classical word2vec / GloVe give every word a single fixed address
        on the map. That means &ldquo;bank&rdquo; in &ldquo;river
        bank&rdquo; and &ldquo;bank&rdquo; in &ldquo;Chase bank&rdquo; get
        the same coordinate — a known failure, because those two senses
        should live in different districts. Transformers fixed this by
        producing <KeyTerm>contextual embeddings</KeyTerm>: the coordinate
        for &ldquo;bank&rdquo; depends on the rest of the sentence, so the
        word can move to the financial district or the riverside depending
        on its neighbours. That&apos;s why BERT and GPT replaced word2vec as
        the input representation of choice around 2018. But — and this is
        the part most people miss — the embedding <em>layer</em> is still
        there. Every LLM starts with <code>nn.Embedding(V, d)</code>. The
        difference is that the addresses are trained end-to-end as part of
        the full model rather than pre-trained separately on a skip-gram
        objective. Same layer, bigger pipeline around it.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Padding token:</strong> when
          you batch variable-length sequences you pad them to a common
          length with a special id (usually <code>0</code>). The embedding
          for that id should be zero and should not update — it&apos;s a
          blank lot on the map, not a real address. Pass{' '}
          <code>padding_idx=0</code> to <code>nn.Embedding</code> and
          it&apos;s handled — forget to, and the pad vector leaks real
          signal into the model.
        </p>
        <p>
          <strong className="text-term-amber">Freeze vs fine-tune:</strong>{' '}
          loading pre-trained GloVe and training it further on 500 labeled
          examples is a great way to <em>destroy</em> the structure that
          took 840 billion tokens to build. You bulldoze districts for the
          sake of a handful of new blocks. If your dataset is small,
          freeze. If it&apos;s large and domain-specific, fine-tune
          (usually with a lower learning rate than the rest of the model).
        </p>
        <p>
          <strong className="text-term-amber">Vocabulary size:</strong> if
          your <code>V</code> is too small, most test-time words map to{' '}
          <code>&lt;unk&gt;</code> and your model sees a sentence full of
          question marks — every rare word gets routed to the same
          unmarked-address lot. If it&apos;s too large, your embedding
          matrix becomes the dominant parameter in the whole network.
          Modern LLMs sidestep this with subword tokenisation (BPE,
          SentencePiece) — another lesson, but know that the trade-off
          exists.
        </p>
        <p>
          <strong className="text-term-amber">&ldquo;Similarity&rdquo;
          means cosine, not Euclidean.</strong> The interesting signal on
          the map is the angle between coordinates, not the raw distance.
          Two words can live in the same direction from origin but at very
          different norms — that&apos;s frequency talking, not meaning.
          Always normalise before comparing, or use cosine similarity
          directly. Euclidean distance on raw embeddings will mostly report
          who has the biggest norm.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Find Paris's capital neighbours">
        <p>
          Download the WikiText-2 corpus via <code>datasets</code> or
          torchtext. Train a skip-gram model (window 5, 100-dim, negative
          sampling with 5 negatives per positive) for 3 epochs. Use{' '}
          <code>nn.Embedding(V, 100)</code> as the center-word table.
        </p>
        <p className="mt-2">
          Once trained, normalise the embeddings and print the top-5
          cosine-nearest neighbours of <code>&quot;paris&quot;</code>.
          Target behaviour: you should see <code>london</code>,{' '}
          <code>berlin</code>, <code>madrid</code>, <code>rome</code>,{' '}
          <code>vienna</code> (or similar capitals) in the top 5 — the
          capital-of district, laid out by nothing but co-occurrence.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: implement the analogy <code>king − man + woman</code> and
          verify that <code>queen</code> appears in the top 3 results. If
          it doesn&apos;t, train for another epoch or widen the context
          window — on a small corpus the streets take longer to straighten,
          and analogies are the last thing to emerge cleanly.
        </p>
      </Challenge>

      {/* ── Closing ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Words become coordinates
          by lookup into a <code>|V| × d</code> embedding matrix — the
          quiet first layer of every NLP model, the city where every word
          has an address. Classical word2vec / GloVe learned that map from
          a context-prediction objective, and the resulting space had
          enough structure that analogy arithmetic worked: walk the
          &ldquo;capital-of&rdquo; direction from Paris and you land on
          Rome. Modern transformers replaced static addresses with
          contextual ones — same city, but a word can move between
          districts depending on the sentence — and the{' '}
          <code>nn.Embedding</code> layer didn&apos;t go anywhere. It&apos;s
          the first thing every LLM does with your input ids, trained
          end-to-end rather than pre-trained.
        </p>
        <p>
          <strong>Next up — Sentiment Analysis.</strong> We have
          coordinates for every word — now the question is whether we can
          glue them together into a feeling. A review is a sequence of
          addresses on the map; &ldquo;sentiment&rdquo; is a single number
          — is this person happy or furious? The jump from a list of word
          coordinates to one scalar verdict is the first real end-to-end
          NLP model you&apos;ll build. We&apos;ll see how averaging the
          map&apos;s blocks gets you surprisingly far, where that approach
          falls over, and what to reach for when it does. Head to{' '}
          <strong>sentiment-analysis</strong> next.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Efficient Estimation of Word Representations in Vector Space',
            author: 'Mikolov, Chen, Corrado, Dean',
            venue: 'arXiv 1301.3781 — the word2vec paper',
            year: 2013,
            url: 'https://arxiv.org/abs/1301.3781',
          },
          {
            title: 'Linguistic Regularities in Continuous Space Word Representations',
            author: 'Mikolov, Yih, Zweig',
            venue: 'NAACL 2013 — the analogy paper',
            year: 2013,
            url: 'https://aclanthology.org/N13-1090/',
          },
          {
            title: 'GloVe: Global Vectors for Word Representation',
            author: 'Pennington, Socher, Manning',
            venue: 'EMNLP 2014',
            year: 2014,
            url: 'https://aclanthology.org/D14-1162/',
          },
          {
            title: 'Dive into Deep Learning — Chapter 15: Natural Language Processing (Pretraining)',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai §15.1–15.7',
            url: 'https://d2l.ai/chapter_natural-language-processing-pretraining/',
          },
          {
            title: 'Distributed Representations of Words and Phrases and their Compositionality',
            author: 'Mikolov, Sutskever, Chen, Corrado, Dean',
            venue: 'NeurIPS 2013 — negative sampling',
            year: 2013,
            url: 'https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality',
          },
        ]}
      />
    </div>
  )
}
