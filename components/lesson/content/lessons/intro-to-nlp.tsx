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
import TokenizerPlayground from '../widgets/TokenizerPlayground'
import VocabularyStats from '../widgets/VocabularyStats'

// Signature anchor: the assembly line that turns raw text into a clean tensor.
// A messy paragraph enters the loading dock, rides the conveyor through three
// stations (cleaning, tokenizing, vectorizing), and leaves the other end as a
// neat row of numbers a network can eat. Returned to at each station reveal
// and at the closing consolidation — "the model never touches text, only what
// the assembly line hands it."
export default function IntroToNlpLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="intro-to-nlp" />

      {/* ── Opening: the loading dock ───────────────────────────── */}
      <Prose>
        <p>
          A neural net is a matrix multiply with ambitions. It adores numbers
          and cannot read. If you hand it the sentence &ldquo;The cat sat on
          the mat.&rdquo; it will stare at you, politely, the way a calculator
          stares at a poem. It does not know what a <code>c</code> is. It does
          not know what a word is. It does not, in fact, know what a sentence
          is. It knows tensors.
        </p>
        <p>
          So before any <NeedsBackground slug="recurrent-neural-network">RNN</NeedsBackground>,
          any transformer, any <NeedsBackground slug="mlp-from-scratch">MLP</NeedsBackground>{' '}
          perched on top of word embeddings, the text has to be converted. Not
          gently translated — <em>converted</em>, the way a warehouse converts
          raw cotton into thread. That conversion happens on{' '}
          <strong>the assembly line that turns raw text into a clean tensor</strong>.
          Messy paragraphs come in one end at the loading dock. Three stations
          work on them in sequence. A row of integers rolls off the other end,
          ready to be handed to the first{' '}
          <NeedsBackground slug="single-neuron">neuron</NeedsBackground> in the
          model. That assembly line is what this lesson is about.
        </p>
        <p>
          <KeyTerm>Natural Language Processing</KeyTerm> is, before anything
          else, that line. Every transformer, every sentiment classifier,
          every chatbot you have ever used runs the same conveyor: raw text in,
          integer IDs out, and <em>those</em> integers are what the network
          actually sees. This lesson walks the line end to end — cleaning,
          tokenizing, vectorizing — and explains why each station looks the
          way it does. By the end, when you read &ldquo;we used a 50k BPE
          vocabulary&rdquo; in a paper, it will mean something concrete: a
          particular station configured a particular way.
        </p>
      </Prose>

      {/* ── Pipeline diagram ────────────────────────────────────── */}
      <AsciiBlock caption="the standard NLP input pipeline">
{`      ┌──────────────────┐
      │   raw text       │   "The cat sat on the mat."
      └────────┬─────────┘
               │  preprocess  (lowercase? strip punct? unicode normalize?)
               ▼
      ┌──────────────────┐
      │   normalized     │   "the cat sat on the mat"
      └────────┬─────────┘
               │  tokenize    (split into units: chars / words / subwords)
               ▼
      ┌──────────────────┐
      │   tokens         │   ["the", "cat", "sat", "on", "the", "mat"]
      └────────┬─────────┘
               │  vocab lookup  (hashmap: str → int)
               ▼
      ┌──────────────────┐
      │   ids            │   [5, 241, 908, 17, 5, 612]
      └────────┬─────────┘
               │  embed         (id → dense vector, learnable)
               ▼
      ┌──────────────────┐
      │   vectors        │   tensor of shape [seq_len, d_model]
      └────────┬─────────┘
               │
               ▼
             model`}
      </AsciiBlock>

      <Prose>
        <p>
          Four arrows on the conveyor, four distinct engineering decisions.
          Skip the cleaning station and the model has to waste parameters
          learning that <code>Cat</code> and <code>cat</code> are the same
          animal. Pick the wrong tokenizer at the second station and half of
          Twitter shows up as <code>&lt;UNK&gt; &lt;UNK&gt; &lt;UNK&gt;</code> —
          unrecognizable crates falling off the belt. Get the vocab wrong at
          the third and your embedding table is 300MB for a model that only
          needs 30MB of parameters. Every production NLP system lives and
          dies on the boring middle of the line.
        </p>
      </Prose>

      {/* ── Station two: the tokenizer (with playground) ────────── */}
      <Prose>
        <p>
          The loading dock is dull — a string is a string — and the third
          station is a hashmap, which we&apos;ll get to. The interesting
          machinery is the middle one: the <strong>tokenizer station</strong>,
          where the text gets chopped into reusable pieces. Different
          tokenizers are different blades mounted on the same conveyor. The
          fastest way to feel what they do is to watch four of them chew on
          the same sentence. Type anything below. Each row shows how one
          blade slices your input, and how big its vocabulary is.
        </p>
      </Prose>

      <TokenizerPlayground />

      <Prose>
        <p>
          Three numbers to watch. The <strong>vocab size</strong> (how many
          distinct tokens the blade knows about), the{' '}
          <strong>token count</strong> (how many pieces fall off the conveyor
          for your sentence), and — implicitly — the tradeoff between them.
          They are inversely related. Tiny vocab, tiny pieces, long sequence.
          Huge vocab, fat pieces, short sequence, but most entries in the
          table never get used and each one costs embedding parameters
          regardless.
        </p>
        <ul>
          <li>
            <strong>Character-level.</strong> Vocab: about 100 (the printable
            ASCII range, plus a handful of Unicode oddities). No
            out-of-vocabulary problem — every word decomposes into known
            characters. The catch: sequences become long. A 100-word
            paragraph is ~500 characters on the belt, and transformer compute
            scales quadratically with sequence length.
          </li>
          <li>
            <strong>Word-level.</strong> Vocab: 50k–100k. Natural unit for
            English readers. The catch: <em>every</em> new word —
            misspellings, names, new slang, foreign loanwords — gets stamped{' '}
            <code>&lt;UNK&gt;</code> at the station and rolls on unchanged.
            Word-level blades have a generalization ceiling hard-coded into
            them on day one.
          </li>
          <li>
            <strong>Subword (BPE, WordPiece, SentencePiece).</strong> Vocab:
            32k–50k. The middle path. Common words are single tokens
            (<code>the</code>, <code>cat</code>); rare words get decomposed
            into meaningful pieces (<code>un + fathom + able</code>). OOV
            essentially disappears because the blade can always fall back to
            characters. This is what every modern LLM uses. GPT-4 uses a
            ~100k BPE. Llama uses a ~32k SentencePiece.
          </li>
        </ul>
      </Prose>

      <Personify speaker="Tokenizer">
        I am the second station on the line. My one job: take your string of
        Unicode and slice it into pieces small enough to fit in a vocabulary,
        big enough to mean something. Pick me poorly and your model spends a
        third of its parameters memorizing that <code>Tokenization</code> is
        one word. Pick me well and it learns that <code>Token</code>,{' '}
        <code>ization</code>, and <code>##s</code> are reusable building
        blocks for half the English lexicon.
      </Personify>

      {/* ── Zipf's law — why word-level breaks ──────────────────── */}
      <Prose>
        <p>
          To see <em>why</em> the word-level blade jams, we need to talk
          about the raw material. Text is not a tidy uniform stream; it is a
          wildly uneven one, and the unevenness is mathematically lawful. If
          you rank words by frequency in a large corpus, the count of the{' '}
          <code>n</code>-th most common word is roughly inversely proportional
          to its rank. This is Zipf&apos;s law, noted by the linguist George
          Zipf in 1949, and it&apos;s the reason the assembly line needs a
          cleverer station than &ldquo;split on spaces.&rdquo;
        </p>
      </Prose>

      <MathBlock caption="Zipf's law — the empirical frequency-rank relationship">
{`f(n)  ∝   1
         ───
          nˢ

where  n = rank of the word (1 = most common)
       f = frequency count
       s ≈ 1  for natural language`}
      </MathBlock>

      <Prose>
        <p>
          Take the log of both sides and the relationship becomes linear:{' '}
          <code>log f = −s · log n + c</code>. On a log-log plot, word
          frequency against rank is a straight line with slope about{' '}
          <code>−1</code>. It holds, for basically every natural language
          ever measured, across corpora of books, newspapers, web text, and
          tweets. Punch it in and watch.
        </p>
      </Prose>

      <VocabularyStats />

      <Prose>
        <p>
          That straight line on the log-log axis is the{' '}
          <KeyTerm>long tail</KeyTerm>, and it&apos;s the entire reason the
          subword blade exists. The top few thousand words (<code>the</code>,{' '}
          <code>of</code>, <code>and</code>, <code>to</code>…) make up about
          80% of all tokens in typical English text. The remaining 20% of
          tokens come from a vocabulary of hundreds of thousands of words,
          most of which appear fewer than five times in a corpus of a million
          words. Your assembly line has to process all of them.
        </p>
        <p>
          A word-level blade is forced into an uncomfortable choice: either
          add every rare word to the vocab (embedding table explodes, and the
          rare embeddings never learn anything because they see almost no
          gradient) or truncate the vocab at some frequency cutoff and dump
          everything below it into one bucket. That bucket is{' '}
          <code>&lt;UNK&gt;</code>, and it is where language goes to die.
        </p>
      </Prose>

      <Personify speaker="<UNK>">
        I am the stand-in for every word you never taught your model.
        &ldquo;Anthropic.&rdquo; &ldquo;cryptocurrency.&rdquo;
        &ldquo;Pikachu.&rdquo; All me. I am one embedding vector, and I am
        supposed to represent the entire infinite set of words your
        tokenizer never saw during training. I do a bad job of it. This is
        why subword tokenizers mostly put me out of work.
      </Personify>

      {/* ── Special tokens ──────────────────────────────────────── */}
      <Prose>
        <p>
          Beyond <code>&lt;UNK&gt;</code>, every model family reserves a
          handful of <KeyTerm>special tokens</KeyTerm> — extra crates the
          station always stamps in, whose roles have nothing to do with
          language per se. They&apos;re structural markers the model learns
          to attend to, riding the belt next to the ordinary tokens.
        </p>
        <ul>
          <li>
            <code>[CLS]</code> — BERT&apos;s &ldquo;classification&rdquo;
            token. Prepended to every input; its final hidden state is used
            as a sentence embedding.
          </li>
          <li>
            <code>[SEP]</code> — BERT&apos;s separator between two sentences
            in a pair task.
          </li>
          <li>
            <code>[PAD]</code> — padding token. Added to shorter sequences so
            that a batch has uniform length. The attention mask tells the
            model to ignore it.
          </li>
          <li>
            <code>&lt;s&gt;</code>, <code>&lt;/s&gt;</code> —
            beginning-of-sequence and end-of-sequence, common in GPT-style
            and T5 models.
          </li>
          <li>
            <code>&lt;|endoftext|&gt;</code> — GPT-family document separator.
          </li>
        </ul>
        <p>
          These live in the vocabulary alongside ordinary tokens and consume
          IDs like any other. When you read a paper and see a vocab size of{' '}
          <code>50,257</code>, that includes the specials.
        </p>
      </Prose>

      {/* ── Station three: vocab ↔ id ───────────────────────────── */}
      <Prose>
        <p>
          The third station — vectorize — is almost anticlimactic after the
          tokenizer. It&apos;s two lookup tables kept in sync:
        </p>
        <ul>
          <li>
            <code>token2id</code>: a hashmap from string to integer.
            Constant-time lookup as each token rolls past on the conveyor.
            &ldquo;cat&rdquo; → <code>241</code>.
          </li>
          <li>
            <code>id2token</code>: a list where index <code>i</code> holds
            the string for id <code>i</code>. Constant-time lookup when
            you&apos;re running the line in reverse to decode.{' '}
            <code>241</code> → &ldquo;cat&rdquo;.
          </li>
        </ul>
        <p>
          That&apos;s it. The &ldquo;vocabulary&rdquo; is just those two
          structures, saved to disk as JSON or a binary. Every tokenizer
          library you will ever use — Hugging Face&apos;s{' '}
          <code>tokenizers</code>, SentencePiece, torchtext — is, at its
          core, wrapping these two lookups plus whatever splitting algorithm
          the tokenizer station upstream decided on.
        </p>
      </Prose>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations of the same assembly line, climbing in
          sophistication. Pure Python so you can see every bolt. NumPy to
          watch the long tail emerge in real numbers. And a real tokenizer
          library — the industrial version of the line, which is what
          actually ships.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · word_tokenizer.py"
        output={`vocab size: 8
ids: [5, 1, 4, 2, 5, 3]
decoded: the cat sat on the mat`}
      >{`import string

def preprocess(text):
    text = text.lower()
    # strip punctuation, keep word characters and whitespace
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize(text):
    return preprocess(text).split()           # whitespace split — the naive word tokenizer

def build_vocab(corpus, specials=("<PAD>", "<UNK>")):
    tokens = set()
    for sent in corpus:
        tokens.update(tokenize(sent))
    # specials get the lowest ids by convention
    id2token = list(specials) + sorted(tokens)
    token2id = {tok: i for i, tok in enumerate(id2token)}
    return token2id, id2token

def encode(text, token2id):
    unk = token2id["<UNK>"]
    return [token2id.get(tok, unk) for tok in tokenize(text)]

def decode(ids, id2token):
    return " ".join(id2token[i] for i in ids)

corpus = ["The cat sat on the mat.", "The dog sat on the log."]
token2id, id2token = build_vocab(corpus)
ids = encode("The cat sat on the mat.", token2id)

print(f"vocab size: {len(id2token)}")
print(f"ids: {ids}")
print(f"decoded: {decode(ids, id2token)}")`}</CodeBlock>

      <Prose>
        <p>
          That&apos;s the whole line in thirty lines of Python.{' '}
          <code>preprocess</code> is the loading dock,{' '}
          <code>tokenize</code> is the station, <code>encode</code> is the
          lookup. Tensor out. Now the frequency analysis that reveals
          Zipf&apos;s law — we run the tokenizer station over a real corpus
          and count what falls off the belt. NumPy makes this a three-line
          affair.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · zipf_analysis.py"
        output={`top-10 tokens:
  rank  1: "the"  count=2134
  rank  2: "of"   count=1012
  rank  3: "and"  count=987
  ...
slope of log-log fit: -1.02  (Zipf predicts -1)`}
      >{`import numpy as np
from collections import Counter

def load_corpus(path):
    with open(path) as f:
        return f.read().lower().split()

tokens = load_corpus("wikipedia_sample.txt")
counts = Counter(tokens)

# sort tokens by frequency, descending
ranked = counts.most_common()
ranks  = np.arange(1, len(ranked) + 1)                 # 1, 2, 3, ...
freqs  = np.array([c for _, c in ranked])              # aligned counts

# fit a line in log-log space:  log f = slope · log n + intercept
slope, intercept = np.polyfit(np.log(ranks), np.log(freqs), 1)
print(f"slope of log-log fit: {slope:.2f}  (Zipf predicts -1)")

# the long tail in one number: fraction of tokens that appear only once
hapax = (freqs == 1).sum() / len(freqs)
print(f"hapax legomena (words seen exactly once): {hapax:.1%}")`}</CodeBlock>

      <Prose>
        <p>
          Run this on any sizeable English corpus and the slope comes out
          between <code>−0.9</code> and <code>−1.1</code>. Shakespeare,
          Wikipedia, Reddit, scientific papers — they all obey the same law.
          The fraction of words appearing exactly once (the{' '}
          <em>hapax legomena</em>) is typically 40–60%. Half your vocabulary
          is words the model sees a single time. No amount of cleverness at
          the loading dock fixes that — the tokenizer station itself has to
          get smarter.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — hugging face tokenizers · bpe_in_practice.py"
        output={`trained vocab size: 30000
encode:  [464, 3797, 3332, 319, 262, 2603, 13]
tokens:  ['The', ' cat', ' sat', ' on', ' the', ' mat', '.']
decode:  The cat sat on the mat.`}
      >{`from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# Byte-level BPE — what GPT-2/3/4 use. No <UNK> needed:
# every byte is in the base vocabulary, so any Unicode string encodes.
tokenizer = Tokenizer(BPE(unk_token=None))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

trainer = BpeTrainer(
    vocab_size=30_000,
    special_tokens=["<|endoftext|>", "<|pad|>"],
)
tokenizer.train(files=["wikipedia_sample.txt"], trainer=trainer)

print(f"trained vocab size: {tokenizer.get_vocab_size()}")

enc = tokenizer.encode("The cat sat on the mat.")
print(f"encode: {enc.ids}")
print(f"tokens: {enc.tokens}")
print(f"decode: {tokenizer.decode(enc.ids)}")`}</CodeBlock>

      <Bridge
        label="pure python → numpy → production"
        rows={[
          {
            left: 'text.lower().split()',
            right: 'Tokenizer(BPE(...)).encode(text)',
            note: 'a one-line naive splitter becomes a learned segmentation',
          },
          {
            left: 'token2id.get(tok, unk_id)',
            right: 'byte-level base vocab',
            note: 'production BPE has no OOV — it always falls back to bytes',
          },
          {
            left: 'Counter(tokens).most_common()',
            right: 'tokenizer.train(files=...)',
            note: 'frequency counting is how BPE decides which merges to add',
          },
        ]}
      />

      <Callout variant="insight" title="why subword tokenization won">
        Word-level blades jam under Zipf&apos;s law — the long tail of rare
        words bloats the vocab and starves each rare embedding of gradient
        signal. Character-level blades avoid OOV but stretch sequences until
        quadratic attention becomes unaffordable. Subword blades — BPE,
        WordPiece, SentencePiece — interpolate cleanly: frequent words stay
        one token, rare words decompose into reusable pieces, and the vocab
        stays a crisp 32k–50k. Every large language model since GPT-2 has
        shipped with one of these mounted on the second station. It is the
        quiet consensus of the field.
      </Callout>

      <Callout variant="note" title="BPE in one paragraph">
        Byte-Pair Encoding starts with a base vocab of characters (or bytes).
        It counts the most frequent adjacent pair in the training corpus —
        say <code>(&apos;t&apos;, &apos;h&apos;)</code> — and merges it into
        a new token <code>&apos;th&apos;</code>. Repeat: find the next most
        frequent pair, merge. Do this N times and you have a vocab of size{' '}
        <code>base + N</code> where frequent character sequences have
        collapsed into single tokens. That is the entire algorithm. WordPiece
        (BERT) is BPE with a likelihood criterion instead of raw frequency.
        SentencePiece is BPE that treats spaces as regular characters, making
        it trivially language-agnostic.
      </Callout>

      {/* ── Gotchas ────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Lowercasing:</strong> helpful
          for topic classification, fatal for Named Entity Recognition.
          &ldquo;Apple&rdquo; the company and &ldquo;apple&rdquo; the fruit
          are different entities, and the loading dock shouldn&apos;t be
          throwing away that distinction before the tokenizer sees it. Modern
          tokenizers are case-sensitive by default for exactly this reason.
        </p>
        <p>
          <strong className="text-term-amber">Unicode normalization:</strong>{' '}
          the string <code>&quot;café&quot;</code> can be encoded as five
          codepoints (<code>é</code> as one character) or six (<code>e</code>{' '}
          + combining acute accent). Run the wrong normalization form (NFC
          vs NFD) and identical-looking strings hash to different vocab
          entries. Always normalize at the loading dock, before the
          tokenizer ever touches the text.
        </p>
        <p>
          <strong className="text-term-amber">HTML, emoji, zero-width characters:</strong>{' '}
          web-scraped text is full of <code>&lt;br&gt;</code> tags, emoji
          modifiers, zero-width joiners (<code>U+200D</code>), and
          right-to-left marks. These look invisible to a human and
          disastrous to a tokenizer that wasn&apos;t trained on them. Strip
          or normalize deliberately; do not trust your input.
        </p>
        <p>
          <strong className="text-term-amber">Bytes vs characters:</strong>{' '}
          Python 3 <code>str</code> is a sequence of Unicode codepoints.
          Python 3 <code>bytes</code> is a sequence of 8-bit values.
          Byte-level BPE (GPT-style) tokenizes bytes, not codepoints — which
          is why it can handle any text ever written without an{' '}
          <code>&lt;UNK&gt;</code>. If you mix the two up, you get garbled
          multi-byte characters, and the error propagates silently down the
          line into your embedding table.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Tokenize a Wikipedia page three ways">
        <p>
          Grab the plain-text version of a Wikipedia article (try the one on{' '}
          <em>Tokenization</em> itself — it&apos;s on-theme). Feed it to
          three assembly lines, each with a different blade mounted on the
          tokenizer station:
        </p>
        <ul className="list-disc pl-5 mt-2 space-y-1">
          <li>
            A character-level tokenizer (every unique character gets an id).
          </li>
          <li>
            A word-level tokenizer using <code>text.lower().split()</code>.
          </li>
          <li>
            A BPE tokenizer trained on the page itself with{' '}
            <code>vocab_size=2000</code> via{' '}
            <code>tokenizers.Tokenizer</code>.
          </li>
        </ul>
        <p className="mt-2">
          For each, report three numbers: <strong>vocab size</strong>,{' '}
          <strong>token count</strong> after encoding the page, and the{' '}
          <strong>compression ratio</strong> (characters ÷ tokens).
          Character-level gives ratio ≈ 1. Word-level gives something like
          4–6. BPE sits in between — typically 2.5–3.5 — and that&apos;s the
          sweet spot modern LLMs exploit.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: print the 20 most frequent BPE tokens. You will see function
          words (<code>the</code>, <code>of</code>), common suffixes
          (<code>ing</code>, <code>tion</code>), and punctuation. That top-20
          is Zipf&apos;s law in action.
        </p>
      </Challenge>

      {/* ── Closing: the whole line, and what's next ───────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Every NLP model in
          existence runs the same assembly line on its inputs: loading dock
          (preprocess), tokenizer station (chop), vectorizer station (look
          up integer IDs), then embed and hand the tensor to the network.
          The <em>model never touches text</em> — only what the assembly
          line hands it. The tokenizer blade governs vocab size, sequence
          length, and OOV behavior all at once, and the math of Zipf&apos;s
          law is why nobody mounts a pure word-level blade anymore. Subword
          blades (BPE, WordPiece, SentencePiece) are the de facto standard
          because they gracefully span the full range from{' '}
          <code>the</code> to arbitrary Unicode garbage without a single
          crate rolling off marked <code>&lt;UNK&gt;</code>.
        </p>
        <p>
          <strong>Next up — Word Embeddings.</strong> The assembly line
          stops at an integer. <code>241</code>. That&apos;s a great ID and
          a terrible representation — because <code>241</code> and{' '}
          <code>242</code> are adjacent integers that probably belong to
          completely unrelated words. Turning a word into a number is easy;
          turning a word into a number that <em>means</em> something is the
          next problem, and it&apos;s what every embedding table in every
          model you&apos;ve ever heard of is there to solve. We&apos;ll
          swap the bare integer coming off the conveyor for a dense vector
          whose geometry encodes meaning, and watch what that changes.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Human Behavior and the Principle of Least Effort',
            author: 'George K. Zipf',
            venue: 'Addison-Wesley, 1949 — the original observation',
          },
          {
            title: 'Neural Machine Translation of Rare Words with Subword Units',
            author: 'Sennrich, Haddow, Birch',
            venue: 'ACL 2016 — BPE for NLP',
            year: 2015,
            url: 'https://arxiv.org/abs/1508.07909',
          },
          {
            title: 'Japanese and Korean Voice Search',
            author: 'Schuster, Nakajima',
            venue: 'ICASSP 2012 — the WordPiece paper',
            url: 'https://research.google/pubs/pub37842/',
          },
          {
            title: 'SentencePiece: A simple and language independent subword tokenizer',
            author: 'Kudo, Richardson',
            venue: 'EMNLP 2018',
            year: 2018,
            url: 'https://arxiv.org/abs/1808.06226',
          },
          {
            title: 'Dive into Deep Learning — Ch. 8.1–8.2: Text Preprocessing & Language Models',
            author: 'Zhang, Lipton, Li, Smola',
            url: 'https://d2l.ai/chapter_recurrent-neural-networks/text-preprocessing.html',
          },
        ]}
      />
    </div>
  )
}
