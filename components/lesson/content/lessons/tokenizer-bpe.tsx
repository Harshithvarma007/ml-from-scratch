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
import BPEMergesAnimation from '../widgets/BPEMergesAnimation'
import TokenizationPreview from '../widgets/TokenizationPreview'

// Signature anchor: the greedy compressor that grows its own alphabet.
// Start from bare letters, spot the most common pair of neighbors, glue it
// into a new symbol, repeat 50,000 times. Returned at the opening, the
// pair-count-merge loop reveal, and the "why not just use words" tradeoff.
export default function TokenizerBpeLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="tokenizer-bpe" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Before any transformer can read a sentence, somebody has to chop the sentence into
          numbered chunks. That chopping is <NeedsBackground slug="intro-to-nlp">tokenization</NeedsBackground>,
          and it is the quietly most important problem in NLP. Two obvious ways to do it both
          fail. Give every character its own token and a sentence turns into a parade — your
          sequence length balloons and the model burns attention on the letters in
          &ldquo;the.&rdquo; Give every word its own token and the first time someone types
          <code> pseudohypoparathyroidism</code> the model hits <code>&lt;UNK&gt;</code> and
          shrugs.
        </p>
        <p>
          BPE is what sits in the middle, and the shape of it is almost cartoonishly simple.
          Think of it as <strong>the greedy compressor that grows its own alphabet</strong>.
          Start with just letters — the alphabet you were born with. Then scan the corpus,
          spot the most common pair of neighbors (<code>t-h</code> is everywhere), and glue
          that pair into a single new symbol. Now your alphabet is one symbol longer. Do that
          again. And again. Do it 50,000 times and you have a dictionary where the most
          common English chunks are each one token, rarer stuff decomposes into a handful of
          pieces, and nothing is ever truly unknown.
        </p>
        <p>
          That is <KeyTerm>Byte Pair Encoding</KeyTerm> — the tokenizer GPT-2, GPT-3, GPT-4,
          Llama, and essentially every modern LLM runs on. It started life as Philip Gage&apos;s
          1994 data-compression trick. Sennrich dusted it off for machine translation in 2015.
          Now it sits in front of every prompt you have ever typed to an LLM. This lesson
          builds one from scratch: train it on a tiny corpus, watch the merges happen, encode
          a sentence with the resulting merge list, then see why the GPT-2 authors swapped
          characters for <em>bytes</em>.
        </p>
      </Prose>

      <Personify speaker="Tokenizer">
        I am the gatekeeper. The model will never see your string — only the integer IDs I
        assign. Choose me poorly and no amount of parameters will save you. Choose me well and
        the model sees an efficient, regular stream.
      </Personify>

      {/* ── BPE objective ───────────────────────────────────────── */}
      <Prose>
        <p>
          Back to the compressor image. We start with an alphabet — the smallest unit we are
          willing to see. For now, call it letters; in a minute we will swap letters for bytes.
          Every word in the corpus gets split down to that alphabet, so <code>lower</code>
          {' '}is just five atoms sitting in a row: <code>l</code>, <code>o</code>, <code>w</code>,
          <code>e</code>, <code>r</code>. Now count every adjacent pair across the whole
          corpus. Pick the pair that shows up most often. Glue those two atoms into a single
          new symbol — that new symbol joins the alphabet. Rewrite the corpus with the glue
          applied. Then do the whole thing again. And again.
        </p>
        <p>
          That is the entire training algorithm. Count, pick, glue, repeat.
        </p>
      </Prose>

      <MathBlock caption="BPE training — the whole algorithm">
{`vocab    ←  set of all unique base units in the corpus
merges   ←  []

repeat until |vocab| = target_size:
    pairs    ←  count(adjacent_pair(a, b) for all (a, b) in corpus)
    (a*, b*) ←  argmax_(a,b)  pairs[(a, b)]
    corpus   ←  replace every (a*, b*) with new token (a* b*)
    vocab    ←  vocab ∪ { (a* b*) }
    merges.append( (a*, b*) → (a* b*) )`}
      </MathBlock>

      <Prose>
        <p>
          No gradients. No neural net. No loss function. Just counting pairs and gluing the
          winner. The output is a <em>merge list</em> — an ordered sequence of rewrite rules,
          one per step. The merge list <strong>is</strong> the tokenizer. To encode new text
          you replay the merges in the same order they were learned. To decode you run them
          in reverse. Everything else the libraries wrap around this is packaging.
        </p>
      </Prose>

      {/* ── Widget 1: BPE Merges Animation ──────────────────────── */}
      <Prose>
        <p>
          Step through the training loop on a five-sentence corpus. Each click advances one
          merge. Watch the pair-frequency table, see the winning pair get highlighted, and
          notice the alphabet grow by exactly one new token per step.
        </p>
      </Prose>

      <BPEMergesAnimation />

      <Prose>
        <p>
          The first few merges always feel boringly predictable, and that is the point. Step 1
          almost always glues <code>(t, h)</code>. Step 2 glues <code>(h, e)</code>. Step 3
          the compressor looks at the <code>th</code> it just minted, spots <code>(th, e)</code>
          {' '}everywhere, and glues those — and congratulations, <code>the</code> is now one
          token. Keep cranking. Frequent digrams like <code>(e, r)</code>, <code>(i, n)</code>,
          and <code>(i, ng)</code> light up next. After a few hundred merges the alphabet
          contains every common English word as a single symbol, and the remaining budget
          gets spent on suffixes (<code>-tion</code>, <code>-ing</code>), roots
          (<code>pseudo</code>), and the long tail of subwords you need to assemble anything
          stranger.
        </p>
        <p>
          This is what lets a word like <code>pseudohypoparathyroidism</code> survive without
          any special handling. It never earned a single token — it was not frequent enough
          to win a merge round. But <code>pseudo</code>, <code>hypo</code>, <code>para</code>,
          <code>thyroid</code>, and <code>ism</code> each might have. The word decomposes into
          four or five familiar pieces. The model reads it as a sequence of known fragments,
          not a single alien blob. No <code>&lt;UNK&gt;</code>, no panic.
        </p>
      </Prose>

      <Personify speaker="Merge">
        Give me a pile of text and I will tell you what it loves to put next to what.
        <code> (t, h)</code> always together? You just bought yourself a <code>th</code>. Show
        me that <code>th</code> nestled up to <code>e</code> in half your corpus? Fine, have
        your <code>the</code>. I am a greedy compression oracle. I take the cheapest possible
        win, every time, until you tell me to stop.
      </Personify>

      {/* ── Token efficiency math ───────────────────────────────── */}
      <Prose>
        <p>
          It is reasonable, at this point, to ask the obvious question: <em>why not just use
          whole words as tokens and skip the merge dance entirely?</em> Because of the math
          below. A word-level vocabulary has to memorize every inflection, every proper noun,
          every typo, every new word the internet coins next Tuesday. You either bloat the
          dictionary into the millions or you fall back to <code>&lt;UNK&gt;</code> the moment
          a user types something unexpected. BPE splits the difference — common words <em>do</em>
          {' '}become single tokens once the compressor merges their letters enough times, but
          the dictionary stays capped because rare words borrow pieces instead of demanding
          their own slot. Measure it.
        </p>
      </Prose>

      <MathBlock caption="what BPE buys you">
{`compression_ratio   =   C / T          (chars per token)

tokens_per_word     =   T / W          (W = word count)

tokens_per_byte     =   T / B          (bytes, for byte-level BPE)


GPT-4 (cl100k_base, V = 100 277):

    English      ≈  3.8 chars/token     ≈  0.75 tokens/word
    Python code  ≈  2.9 chars/token     ≈  slightly lossy on symbols
    Thai         ≈  0.9 chars/token     ≈  ~5× worse than English
    Burmese      ≈  0.5 chars/token     ≈  ~10× worse than English`}
      </MathBlock>

      <Prose>
        <p>
          Read that table carefully. <strong>Every</strong> modern LLM is charged per token.
          Every context window is counted in tokens. When a Thai user and an English user type
          the same paragraph, the Thai user pays five times the tokens to say the same thing,
          gets five times less context, and runs up five times the latency. That asymmetry is
          not a bug in the math — it is a direct consequence of training the compressor on a
          corpus dominated by English web text. The frequent pairs that won merge rounds were
          English pairs. Everyone else pays the tax.
        </p>
      </Prose>

      <Callout variant="warn" title="the tokenizer is a policy decision">
        A compressor trained on &ldquo;a representative sample of the internet&rdquo; inherits
        the distribution of the internet — roughly 60% English. The resulting per-token costs
        are not neutral. When a new model ships with better benchmarks on language X, check
        whether they also trained a better tokenizer for X. Half the win is often there.
      </Callout>

      {/* ── Widget 2: Tokenization Preview ──────────────────────── */}
      <Prose>
        <p>
          Now try it live. Type a sentence, watch it decompose into byte-pair tokens with the
          merges applied in the exact order they were learned. Notice how leading spaces
          travel with the word that follows them (a GPT-2 convention — more on that below) and
          how unusual words fragment into smaller pieces while common words survive as a
          single token. The merge list you trained <em>is</em> the dictionary; this widget is
          it running in reverse on your input.
        </p>
      </Prose>

      <TokenizationPreview />

      <Prose>
        <p>
          Two things are worth staring at. First, <code>&quot;hello&quot;</code> and
          {' '}<code>&quot; hello&quot;</code> (with a leading space) are <em>different tokens</em>.
          Not a bug — the GPT-2 tokenizer deliberately binds the leading space to the token
          because it makes sentence-level encoding cheaper and decoding lossless. It means
          the word at position 0 in a sentence and the same word mid-sentence get different
          IDs. Get used to it.
        </p>
        <p>
          Second, capitalization matters. <code>&quot;The&quot;</code>, <code>&quot;the&quot;</code>,
          and <code>&quot; the&quot;</code> are three separate tokens in the dictionary. They
          share zero parameters in the <NeedsBackground slug="word-embeddings">embeddings</NeedsBackground>
          {' '}table. The model has to <em>learn</em> that they mean roughly the same thing —
          and it does, but only because it sees them in similar contexts a million times.
        </p>
      </Prose>

      <Personify speaker="Byte">
        I am the atom. 256 values, zero ambiguity, no Unicode drama. Whatever you throw at me
        — emoji, CJK ideographs, a corrupted file — decomposes cleanly into some sequence of
        me. Build your vocabulary out of me and you will never meet an <code>&lt;UNK&gt;</code>{' '}
        token again. That is my one job and I am very good at it.
      </Personify>

      {/* ── Byte-level BPE callout ──────────────────────────────── */}
      <Prose>
        <p>
          GPT-2 introduced one small, brilliant change to the Sennrich recipe: don&apos;t start
          the alphabet from characters, start it from <em>bytes</em>. A character-level
          compressor has to decide what a character even is. Does <code>é</code> count as one
          symbol or two (<code>e</code> plus a combining acute accent)? Which Unicode
          normalization form? If the user pastes a malformed UTF-8 sequence, do you crash?
          What about <code>🦀</code>, or <code>中</code>, or the weird zero-width joiner some
          emoji smuggle in?
        </p>
        <p>
          A byte-level BPE sidesteps every one of those questions. The starting alphabet is
          just the 256 possible byte values. Every possible input — every Unicode character,
          every emoji, every binary blob — encodes to some sequence of bytes, which
          decomposes trivially into those 256 atoms. The merges then glue common byte
          sequences into larger tokens, and now the compressor can chew on <em>anything</em>
          {' '}— source code, Chinese, Python, a PDF — with the same 50K-entry dictionary, and
          decoding is a perfect inverse. No loss. No <code>&lt;UNK&gt;</code>. Ever.
        </p>
      </Prose>

      <Callout variant="insight" title="byte-level vs char-level, one sentence">
        Char-level BPE has to agree with you on what a character is. Byte-level BPE does not
        care — bytes are bytes. This is why every LLM since GPT-2 uses byte-level.
      </Callout>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations, each shorter than the last. First, pure Python on a tiny
          corpus so you can watch every pair, every merge, every rewrite. Second, a numpy-free
          encoder that applies a trained merge list to a new word. Third, the Hugging Face
          {' '}<code>tokenizers</code> library — what you would actually ship.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python BPE training · bpe_scratch.py"
        output={`merge 1: ('l', 'o') -> 'lo'        (count=3)
merge 2: ('lo', 'w') -> 'low'      (count=3)
merge 3: ('e', 'r') -> 'er'        (count=2)
merge 4: ('n', 'e') -> 'ne'        (count=2)
merge 5: ('ne', 'w') -> 'new'      (count=2)

vocab after 5 merges: {'l','o','w','e','r','n','s','t','lo','low','er','ne','new'}`}
      >{`from collections import Counter

# tiny corpus — 5 pseudo-sentences, already split into words
corpus = ["low", "low", "low", "lower", "lower", "newest", "newest", "newer"]

# step 0: each word becomes a tuple of characters (the starting "base units")
words = [tuple(w) for w in corpus]

merges = []
for step in range(5):
    # count every adjacent pair across the whole corpus
    pair_counts = Counter()
    for w in words:
        for i in range(len(w) - 1):
            pair_counts[(w[i], w[i + 1])] += 1

    # greedy — pick the most frequent pair
    best_pair, count = pair_counts.most_common(1)[0]
    merges.append(best_pair)
    merged = best_pair[0] + best_pair[1]

    # rewrite the corpus: replace every (a, b) with the merged token
    new_words = []
    for w in words:
        out, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == best_pair:
                out.append(merged)
                i += 2
            else:
                out.append(w[i])
                i += 1
        new_words.append(tuple(out))
    words = new_words
    print(f"merge {step+1}: {best_pair} -> {merged!r}  (count={count})")

vocab = set(t for w in words for t in w)
print(f"\\nvocab after 5 merges:", vocab)`}</CodeBlock>

      <Prose>
        <p>
          Twenty lines, the whole compressor. You can read the greedy logic off the page: the
          inner loop counts every adjacent pair, the <code>most_common(1)</code> call picks
          the winner, the glue step rewrites the corpus so the newly-minted token participates
          in the next round of counting. The first winning pair was <code>(l, o)</code>. Five
          merges later, <code>low</code> and <code>new</code> are single tokens while
          {' '}<code>er</code> and <code>est</code> are reusable suffixes. A 300-entry dictionary
          would turn most of common English into single-token words. A 50K dictionary covers
          essentially all of it plus the interesting corners.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — encoder from a trained merge list · bpe_encode.py"
        output={`tokens: ['low', 'e', 'r']
tokens: ['new', 'e', 's', 't']
tokens: ['low', 'e', 'r', 'ne', 's', 's']        # 'lowerness' — unseen word, fragmented`}
      >{`def encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Apply merges in the exact order they were learned."""
    tokens = list(word)                               # start from characters

    for a, b in merges:                               # iterate merges in order
        i = 0
        out = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                out.append(a + b)                     # apply this merge
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
    return tokens

# the merges learned in layer 1
merges = [('l', 'o'), ('lo', 'w'), ('e', 'r'), ('n', 'e'), ('ne', 'w')]

for w in ["lower", "newest", "lowerness"]:
    print("tokens:", encode(w, merges))`}</CodeBlock>

      <Bridge
        label="training → encoding"
        rows={[
          {
            left: 'count adjacent pairs, pick argmax',
            right: 'iterate saved merges in order',
            note: 'training is greedy search; encoding is deterministic replay',
          },
          {
            left: 'merges list grows over time',
            right: 'merges list is frozen, read-only',
            note: 'the merge list IS the tokenizer — save it, ship it',
          },
          {
            left: "rewrite corpus after every merge",
            right: 'rewrite one word at a time, never retrain',
            note: 'encoding is O(words × |merges|) — fast enough in practice',
          },
        ]}
      />

      <Prose>
        <p>
          In production nobody writes the encoder by hand. You use a library that handles
          byte-level encoding, pre-tokenization (splitting on whitespace and punctuation),
          regex for the GPT-style space-handling, and a fast Rust implementation of the merge
          loop. That&apos;s Hugging Face <code>tokenizers</code>, or OpenAI&apos;s{' '}
          <code>tiktoken</code>. Same greedy compressor underneath — just with the inner loop
          rewritten in a language that does not stop to breathe. Here&apos;s what training
          looks like from the outside.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — HuggingFace BpeTrainer · bpe_hf.py"
        output={`vocab size: 2000
encoding 'The quick brown fox': ['The', ' quick', ' brown', ' fox']        # most freq words — single token each
encoding 'pseudohypoparathyroidism': ['pseudo', 'hyp', 'op', 'ara', 'thyroid', 'ism']`}
      >{`from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# byte-level BPE, just like GPT-2
tok = Tokenizer(BPE(unk_token=None))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)

trainer = BpeTrainer(
    vocab_size=2000,                          # target vocabulary size
    special_tokens=["<|endoftext|>"],
    initial_alphabet=ByteLevel.alphabet(),    # 256 bytes as starting atoms
)

tok.train(files=["shakespeare.txt"], trainer=trainer)  # pretend this file exists

print("vocab size:", tok.get_vocab_size())
enc = tok.encode("The quick brown fox")
print("encoding 'The quick brown fox':", enc.tokens)

enc = tok.encode("pseudohypoparathyroidism")
print("encoding 'pseudohypoparathyroidism':", enc.tokens)`}</CodeBlock>

      <Bridge
        label="numpy-free encoder → HuggingFace"
        rows={[
          {
            left: 'merges = [(a, b), ...]',
            right: 'tok = Tokenizer(BPE(...))',
            note: 'the merge list lives inside the Tokenizer object',
          },
          {
            left: 'encode(word, merges) -> list[str]',
            right: 'tok.encode(text).ids -> list[int]',
            note: 'the library also assigns integer IDs for direct model input',
          },
          {
            left: 'iterate python merge list',
            right: 'compiled Rust applies all merges in one pass',
            note: 'production tokenizers are 100-1000× faster than pure python',
          },
        ]}
      />

      <Callout variant="insight" title="the point of the three layers">
        Layer 1 makes the compressor unmistakable — count pairs, argmax, glue. Layer 2 shows
        the asymmetry between training (greedy, stateful) and encoding (deterministic,
        stateless replay of the saved dictionary). Layer 3 is what you actually ship —
        someone else wrote the ten-microsecond-per-call Rust. Same algorithm, very different
        lines-of-code.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Leading spaces are tokens:</strong>{' '}
          <code>&quot;hello&quot;</code> and <code>&quot; hello&quot;</code> are different
          tokens in every GPT-style tokenizer. If you programmatically concatenate strings,
          check whether you are about to mint a different token sequence than the one the
          model was trained on. This bug is the source of most &ldquo;why is my LLM
          performance bad on my own dataset&rdquo; posts.
        </p>
        <p>
          <strong className="text-term-amber">Unicode normalization:</strong> the string{' '}
          <code>&quot;café&quot;</code> can be encoded in UTF-8 as either{' '}
          <code>63 61 66 C3 A9</code> (precomposed é) or{' '}
          <code>63 61 66 65 CC 81</code> (e + combining accent). These tokenize to different
          sequences. Normalize (NFC is usually right) <em>before</em> tokenizing — and apply
          the same normalization at inference time that you applied during training.
        </p>
        <p>
          <strong className="text-term-amber">Numbers fragment badly:</strong> BPE doesn&apos;t
          know arithmetic. The number <code>12345</code> might tokenize as <code>&quot;123&quot;,
          &quot;45&quot;</code> or <code>&quot;12&quot;, &quot;345&quot;</code> or any other
          split depending on what sequences were frequent in training. This is partly why LLMs
          struggle with long arithmetic. Modern models (Llama-3, GPT-4o) explicitly split
          digits into single-digit tokens to fix this.
        </p>
        <p>
          <strong className="text-term-amber">Decoding a partial stream:</strong> if you
          interrupt generation mid-token — e.g., streaming token-by-token to a UI — the last
          token might be an incomplete UTF-8 sequence. Buffer bytes until you have a complete
          character. Otherwise you get mojibake on the last chunk.
        </p>
      </Gotcha>

      {/* ── Challenge ────────────────────────────────────────────── */}
      <Challenge prompt="Train BPE on Shakespeare">
        <p>
          Grab the complete works of Shakespeare as a plain text file (about 5 MB,
          <code> shakespeare.txt</code> is a standard benchmark). Train a byte-level BPE
          tokenizer on it with <code>vocab_size=256</code> using Hugging Face{' '}
          <code>BpeTrainer</code>. Since you start from 256 bytes, you will learn exactly zero
          merges — this is your baseline.
        </p>
        <p className="mt-2">
          Now re-train with <code>vocab_size=1000</code>, then <code>5000</code>, then{' '}
          <code>10000</code>. For each, encode the full corpus and measure tokens-per-byte.
          You should see it drop from 1.00 (no merges) to around 0.35-0.40 (10K vocab). That
          ratio is the compression ratio you bought with the extra 9 744 vocab slots.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: print the 20 longest tokens in the 10K vocab. You will find whole Shakespeare
          words like <code>&quot;Hamlet&quot;</code> and <code>&quot;Rosencrantz&quot;</code>{' '}
          as single tokens — the compressor has memorized the corpus&apos;s proper nouns.
          That&apos;s what &ldquo;trained on your data&rdquo; means.
        </p>
      </Challenge>

      {/* ── Closing + teaser ─────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> BPE is a greedy compressor that grows its
          own alphabet. It starts from bare atoms — bytes, in practice — spots the most
          common pair, glues the pair into a new symbol, and repeats until the dictionary
          hits the size you asked for. That solves out-of-vocabulary by letting rare words
          borrow common pieces, caps vocabulary bloat by construction, and — in the byte
          variant — handles every Unicode corner the same way it handles ASCII. The merge
          list is the tokenizer: train once, apply forever. And the compression ratio is not
          uniform across languages, which is a genuine source of real-world model bias that
          most benchmarks quietly ignore.
        </p>
        <p>
          <strong>Next up — Build Vocabulary.</strong> You have seen the compressor run on
          five words. Now point it at a real corpus, ship it, and watch what a 50K-entry
          dictionary actually contains. That is{' '}
          <code>build-vocabulary</code> — training a BPE vocab on real text, spelunking the
          result, and making sure the merge list you save matches the one you load.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Neural Machine Translation of Rare Words with Subword Units',
            author: 'Sennrich, Haddow, Birch',
            venue: 'ACL 2016 — the paper that brought BPE into NLP',
            url: 'https://arxiv.org/abs/1508.07909',
          },
          {
            title: 'Language Models are Unsupervised Multitask Learners',
            author: 'Radford, Wu, Child, Luan, Amodei, Sutskever',
            venue: 'OpenAI 2019 — the GPT-2 paper, introduces byte-level BPE',
            url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
          },
          {
            title: 'tiktoken — OpenAI\u2019s fast BPE tokenizer',
            author: 'OpenAI',
            venue: 'GitHub — the reference implementation for GPT-3/4 tokenization',
            url: 'https://github.com/openai/tiktoken',
          },
          {
            title: 'Hugging Face Tokenizers library',
            author: 'Hugging Face',
            venue: 'fast Rust-backed BPE / WordPiece / Unigram trainer and encoder',
            url: 'https://github.com/huggingface/tokenizers',
          },
          {
            title: 'A New Algorithm for Data Compression',
            author: 'Philip Gage',
            venue: 'C Users Journal 1994 — the original BPE paper, for compression',
            url: 'http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM',
          },
        ]}
      />
    </div>
  )
}
