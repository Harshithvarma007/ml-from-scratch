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
import VocabSizeSweep from '../widgets/VocabSizeSweep'
import TokenFrequencyDistribution from '../widgets/TokenFrequencyDistribution'

// Signature anchor: the librarian walking every shelf with a clipboard —
// tallying how many times each token appears across the training corpus.
// The clipboard becomes the vocabulary: keep the N most frequent tokens,
// toss the rest into UNK. Anchor returns at the opening walk, the
// frequency-cutoff reveal, and the "what falls off the clipboard" section.
export default function BuildVocabularyLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="build-vocabulary" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Picture a librarian with a clipboard. Not judging books, not
          shelving them, not recommending summer reads — just walking every
          shelf in the stacks and tallying how many times each token shows up
          across the entire corpus. <code>the</code>: four hundred million.{' '}
          <code>probability</code>: twelve thousand. <code>snorgleplop</code>:
          one, in a fanfic someone scraped by accident. The walk takes a
          while. The clipboard gets heavy. When it&apos;s done, that
          clipboard <em>is</em> your vocabulary.
        </p>
        <p>
          That&apos;s the whole lesson, in cartoon form. Training a
          vocabulary is the least glamorous step in a modern LLM pipeline
          and the first one you actually do. Before you initialize a single
          weight, before you write a line of model code, before you even
          check the GPU quota you don&apos;t have — you send the librarian
          through the shelves with a clipboard. Count everything, rank by
          frequency, keep the top N, save a JSON file. The file is under a
          megabyte. It ships with the model forever, and the model lives
          with whatever the librarian wrote down for the rest of its life.
        </p>
        <p>
          The recipe looks deceptively small: pick a corpus, pick a{' '}
          <KeyTerm>vocab size</KeyTerm>, run{' '}
          <NeedsBackground slug="tokenizer-bpe">BPE</NeedsBackground> until
          the clipboard has that many rows, save the file. Under a megabyte
          of JSON. But almost every decision the librarian makes along the
          way — which shelves to walk, how long to tally, what to do with
          the one-off fanfic token — shows up three months later as a
          latency number, a scaling curve, or a multilingual regression no
          one can explain. This lesson is those decisions.
        </p>
      </Prose>

      <Personify speaker="Vocab size">
        I&apos;m the one number you pick before you pick any other number.
        Too small and every sentence becomes a ribbon of subwords — long
        sequences, slow inference, bloated attention. Too big and most of me
        is dead weight — embedding rows for tokens that appear once in a
        million documents. Somewhere between <code>32k</code> and{' '}
        <code>100k</code>, for English, is where I earn my keep.
      </Personify>

      {/* ── Math: the vocab-size tradeoff ────────────────────────── */}
      <Prose>
        <p>
          The tradeoff is concrete. Every extra row on the clipboard costs
          you one row in the <NeedsBackground slug="word-embeddings">embedding table</NeedsBackground>{' '}
          and one row in the output projection. Every token you{' '}
          <em>don&apos;t</em> keep forces the{' '}
          <NeedsBackground slug="intro-to-nlp">tokenization</NeedsBackground>{' '}
          step to split common words into subwords, which makes sequences
          longer, which makes training and inference both quadratically more
          expensive (attention is <code>O(n²)</code>). Short clipboard:
          cheap memory, expensive compute. Long clipboard: the other way
          around.
        </p>
      </Prose>

      <MathBlock caption="the two costs that fight each other">
{`embedding params   =   V · d       (V = vocab size, d = hidden dim)

sequence length    ∝   1 / (tokens-per-word)      ≈ f(V)

compute per step   ≈   n² · d      (attention dominates at long n)

⇒  small V  →  long n  →  big n²·d    (compute wins)
   big   V  →  short n  →  big V·d    (params win)
   sweet spot lies in between — usually 32k–100k for English.`}
      </MathBlock>

      {/* ── Widget 1: VocabSizeSweep ─────────────────────────────── */}
      <VocabSizeSweep />

      <Prose>
        <p>
          Drag the slider. At <code>V = 100</code> the librarian kept almost
          nothing — the tokenizer is effectively byte-pair on characters,
          most English words take four to eight tokens, and a tweet is 200
          tokens long. At <code>V = 50,000</code> the clipboard holds every
          common word outright; compression is roughly <code>0.75</code>{' '}
          tokens per word. Doubling again to <code>100k</code> (GPT-4) buys
          diminishing returns — the marginal rows the librarian writes down
          are rare proper nouns and code symbols. Past that you&apos;re
          paying an embedding row for tokens that show up once every
          thousand documents.
        </p>
        <p>
          The slider shows you the shape of the tradeoff. It doesn&apos;t
          show you <em>what</em> the librarian was counting. Which is the
          next thing.
        </p>
      </Prose>

      <Callout variant="note" title="corpus = the vocab's worldview">
        The clipboard only ever reflects the shelves the librarian actually
        walked. Send them through Wikipedia and they&apos;ll tally endless
        counts of <code>tion</code>, <code>ing</code>, and{' '}
        <code>United States</code>. Send them through GitHub and they come
        back with <code>def </code>, <code>self.</code>, and <code>=&gt;</code>
        at the top of the frequency list. A BPE merge is just a vote on
        that tally: whichever byte pairs the librarian counted the most win
        early merge slots and become tokens. The vocabulary <em>is</em> the
        training distribution, compressed into 30,000 lines of JSON.
      </Callout>

      {/* ── Multilingual balancing ───────────────────────────────── */}
      <Prose>
        <p>
          Multilingual gets ugly fast. Say the librarian walks Common Crawl:
          about 45% of the shelves are English, 5% Chinese, 3% Spanish, and
          the long tail is everything else. Count raw, rank by frequency,
          and the clipboard is — unsurprisingly — mostly English. Swahili
          gets character-level representation and a 20x longer sequence for
          the same sentence, because its tokens never made the frequency
          cut.
        </p>
        <p>
          The fix is <KeyTerm>upsampling</KeyTerm>: send the librarian
          through the low-resource shelves more than once, so their tally
          for those languages climbs faster than the raw corpus frequency
          would suggest. The mT5 paper uses a temperature-based sampler;
          LLaMA 2 hand-tuned the weights. Either way, the goal is the same —
          move everyone&apos;s tokens-per-word into the same ballpark.
        </p>
      </Prose>

      <AsciiBlock caption="naive vs upsampled multilingual training mix">
{`raw Common Crawl share                upsampled training mix (α = 0.3)

 en  ████████████████████  45 %       en  ████████████  28 %
 zh  ███                    5 %       zh  █████         12 %
 es  ██                     3 %       es  ████           9 %
 ar  █                     1.5%       ar  ███            7 %
 hi  ▌                     0.7%       hi  ███            6 %
 sw  ▏                     0.1%       sw  ██             4 %
  …                                    …

 → English tokens win 90% of merges    → every language gets a fair shot
   rare-language seqs ~20× longer        sequence lengths within 2–3× of en`}
      </AsciiBlock>

      {/* ── Widget 2: TokenFrequencyDistribution ─────────────────── */}
      <Prose>
        <p>
          Whatever mix the librarian walks, the final frequency column on
          the clipboard always looks like this — and this is where the
          frequency-cutoff decision actually bites:
        </p>
      </Prose>

      <TokenFrequencyDistribution />

      <Prose>
        <p>
          Log-log axes, near-perfect straight line. Zipf&apos;s law all over
          again, because language. The top 100 rows on the clipboard —
          spaces, common subwords, punctuation — account for maybe 40% of
          all the librarian&apos;s tally marks. The bottom 10% of the
          vocabulary shows up in fewer than 0.01% of documents. Those rare
          rows still cost one embedding parameter each, and they barely get
          enough gradient signal during training to learn anything useful.
        </p>
        <p>
          This is the frequency cutoff in real life. The librarian keeps
          the top N by count and draws a line. Everything above the line
          gets a token id. Everything below it — the tokens that fell off
          the clipboard — either gets split into smaller pieces the
          tokenizer already knows, or dumped into an <code>&lt;UNK&gt;</code>{' '}
          bucket on classical tokenizers. The librarian never judges
          content. They just count. If a string of bytes didn&apos;t show
          up often enough, it doesn&apos;t make the cut, no matter how
          meaningful it might be to someone somewhere.
        </p>
        <p>
          This is one of the reasons vocab-size tuning matters. Past some
          point, every new row the librarian adds is in the long tail — it
          probably won&apos;t help the model, and it&apos;s guaranteed to
          cost memory.
        </p>
      </Prose>

      {/* ── Special tokens ───────────────────────────────────────── */}
      <Personify speaker="Special token">
        I&apos;m the slot the BPE algorithm never sees. I get a fixed id and
        a fixed string — <code>&lt;|endoftext|&gt;</code>, <code>[PAD]</code>,{' '}
        <code>&lt;|user|&gt;</code> — and I get added <em>after</em> the
        librarian finishes counting, stapled onto the end of the clipboard.
        Don&apos;t ever let BPE learn my bytes. Don&apos;t ever let a user
        sneak my string into their prompt.
      </Personify>

      <Prose>
        <p>
          Special tokens are the hooks. They tell the model when a document
          starts, when a message ends, where padding begins, who is
          speaking. They are not learned by BPE — the librarian doesn&apos;t
          tally them. You reserve specific ids for them, set their
          embedding to something sensible (usually random init), and make
          sure the tokenizer will never produce them from ordinary text.
        </p>
        <p>
          The standard cast, across the industry:
        </p>
        <ul>
          <li>
            <code>[PAD]</code> — padding to make variable-length sequences into rectangular
            batches. Attention mask zeroes them out.
          </li>
          <li>
            <code>[CLS]</code> / <code>[SEP]</code> — BERT-era classification and sentence
            separators. Still used in encoder models.
          </li>
          <li>
            <code>&lt;s&gt;</code> / <code>&lt;/s&gt;</code> — sentence boundaries for T5 /
            BART-style seq2seq.
          </li>
          <li>
            <code>&lt;|endoftext|&gt;</code> — GPT-2&apos;s original document separator,
            inherited by most GPT-family tokenizers.
          </li>
          <li>
            <code>&lt;|user|&gt;</code> / <code>&lt;|assistant|&gt;</code> / <code>&lt;|system|&gt;</code>
             — chat-template roles. Post-GPT-3.5. These are the tokens that make instruction
            tuning tractable.
          </li>
          <li>
            <strong>Reserved slots.</strong> LLaMA 2 stapled roughly 256 unused token ids
            onto the end of the clipboard. They do nothing at pretraining time. When
            someone later wants to add a new role, a tool-use marker, or a modality token,
            they don&apos;t have to resize the embedding matrix — they claim a reserved id
            and keep moving.
          </li>
        </ul>
      </Prose>

      <Callout variant="insight" title="why reserved slots are genius">
        Resizing an embedding matrix means every downstream checkpoint —
        every fine-tune, every LoRA adapter — has to be re-aligned or
        regenerated. Reserving 256 empty ids at the end of the clipboard
        costs you <code>256 · d</code> parameters, maybe 1 MB. It saves you
        from a permanent fork in model-compatibility-land. Every serious
        LLM release since LLaMA has copied the idea.
      </Callout>

      {/* ── The vocab file ───────────────────────────────────────── */}
      <Prose>
        <p>
          What does the librarian&apos;s clipboard actually look like once
          you pour it onto disk? A tokenizer is typically two files: a{' '}
          <code>vocab.json</code> mapping tokens to ids, and a{' '}
          <code>merges.txt</code> listing BPE merges in order. Combined size
          for a 50k vocab: under a megabyte. It ships with the model
          forever. You version it like code.
        </p>
      </Prose>

      <AsciiBlock caption="what a tokenizer actually is, on disk">
{`my-tokenizer/
├── vocab.json                { "<|endoftext|>": 50256, "the": 262, "Ġthe": 11 , … }
├── merges.txt                #  50,000 lines, one per BPE merge, order matters
│                             t h
│                             th e
│                             Ġ t
│                             …
├── special_tokens_map.json   { "bos_token": "<|endoftext|>", "pad_token": "[PAD]" }
└── tokenizer_config.json     { "model_max_length": 2048, "add_prefix_space": false, … }

total on disk:  ~900 KB for 50 000 tokens.   shippable as part of the model repo.`}
      </AsciiBlock>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three ways to send the librarian through the stacks. The first is
          the toy version — a pure-Python BPE loop with pre-tokenization on
          whitespace, useful for understanding and nothing else. The second
          is what you&apos;d actually reach for in a real project. The
          third is the clipboard that&apos;s already loaded on every GPT-4
          inference server on Earth.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · train_bpe.py"
        output={`merges learned: 4000
vocab size:     4256   (256 byte baseline + 4000 merges)
"Hello, world!" → ['H', 'e', 'l', 'lo', ',', ' ', 'wor', 'ld', '!']`}
      >{`from collections import Counter

def pre_tokenize(text):
    # split on whitespace first — BPE only merges within words, never across.
    return [list(w) for w in text.split()]

def get_pair_stats(words, freqs):
    pairs = Counter()
    for word, f in zip(words, freqs):
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += f
    return pairs

def merge_pair(words, pair):
    a, b = pair
    out = []
    for word in words:
        new, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new.append(a + b); i += 2
            else:
                new.append(word[i]); i += 1
        out.append(new)
    return out

def train_bpe(corpus, num_merges=4000):
    words_by_freq = Counter(corpus.split())
    words  = [list(w) for w in words_by_freq.keys()]
    freqs  = list(words_by_freq.values())
    merges = []
    for _ in range(num_merges):
        pairs = get_pair_stats(words, freqs)
        if not pairs: break
        best = max(pairs, key=pairs.get)
        words = merge_pair(words, best)
        merges.append(best)
    return merges

corpus = open("shakespeare.txt").read()
merges = train_bpe(corpus, num_merges=4000)
print(f"merges learned: {len(merges)}")`}</CodeBlock>

      <Prose>
        <p>
          That version is 30 lines, readable end-to-end, and about ten
          thousand times too slow for anything real — picture our librarian
          on a unicycle, stopping at every shelf to rewrite the clipboard
          by hand. In production you use the Hugging Face{' '}
          <code>tokenizers</code> library — Rust under the hood, ByteLevel
          pre-tokenization, trains a 50k vocab on 10 GB of text in under
          ten minutes.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — Hugging Face tokenizers · train_hf_bpe.py"
        output={`[00:04:18] Pre-processing files       (10 GB,  14.2M docs)   ━━━━━━━━ 100 %
[00:09:51] Tokenize words                                    ━━━━━━━━ 100 %
[00:12:07] Count pairs                                       ━━━━━━━━ 100 %
[00:41:02] Compute merges  (50 000)                          ━━━━━━━━ 100 %
vocab size: 50257   (50000 BPE + 1 <|endoftext|> + 256 reserved)`}
      >{`from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder       = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=50_000,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>", "[PAD]"]
                   + [f"<|reserved_{i}|>" for i in range(256)],      # LLaMA-style slots
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),            # full 256-byte base
)

tok.train(files=["corpus_shard_00.txt", "corpus_shard_01.txt", ...], trainer=trainer)
tok.save("my-tokenizer.json")

print("vocab size:", tok.get_vocab_size())
print(tok.encode("Hello, world!").tokens)`}</CodeBlock>

      <Prose>
        <p>
          And because you rarely actually send the librarian through the
          shelves from scratch — most projects start from someone
          else&apos;s clipboard — layer three is what it looks like to{' '}
          <em>use</em> a production tokenizer. This is <code>tiktoken</code>,
          which is what OpenAI ships for GPT-4.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — tiktoken (GPT-4's actual vocab) · use_tiktoken.py"
        output={`gpt-4 vocab size:  100277
encoded tokens:    [9906, 11, 1917, 0]
decoded back:      'Hello, world!'
avg tokens/word in The Raven: 1.41`}
      >{`import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")     # the real cl100k_base vocab
print("gpt-4 vocab size: ", enc.n_vocab)

ids = enc.encode("Hello, world!")
print("encoded tokens:   ", ids)
print("decoded back:     ", repr(enc.decode(ids)))

# quick compression check
poem = open("raven.txt").read()
n_tokens = len(enc.encode(poem))
n_words  = len(poem.split())
print(f"avg tokens/word in The Raven: {n_tokens / n_words:.2f}")`}</CodeBlock>

      <Bridge
        label="pure python → HF tokenizers → tiktoken"
        rows={[
          {
            left: 'split on whitespace; merge pairs in a Python loop',
            right: 'ByteLevel pre-tokenizer + Rust BPE trainer',
            note: 'same algorithm, 10,000× faster, handles raw bytes and Unicode edges',
          },
          {
            left: 'manual special-tokens list in code',
            right: 'special_tokens=[...] + 256 reserved slots',
            note: 'reserving ids at train time = no embedding resize later',
          },
          {
            left: 'pickle.dump(merges)',
            right: 'tokenizer.json  ←→  tiktoken.Encoding',
            note: 'one JSON file ships with the model for its entire lifetime',
          },
        ]}
      />

      <Callout variant="insight" title="real production vocab sizes, for calibration">
        <p>
          <strong>GPT-2:</strong> 50,257 tokens — the cl50k lineage, English-heavy.
          <br />
          <strong>GPT-4:</strong> 100,277 tokens (cl100k_base) — bigger vocab for faster
          inference on long contexts; handles code and non-English markedly better.
          <br />
          <strong>LLaMA / LLaMA 2:</strong> 32,000 tokens — small, English-centric,{' '}
          SentencePiece-BPE, plus those 256 reserved slots.
          <br />
          <strong>LLaMA 3:</strong> 128,256 tokens — a large jump to match GPT-4-class
          multilingual coverage.
          <br />
          <strong>mT5:</strong> 250,112 tokens — 101 languages; when you balance for Swahili
          and Hindi the vocab has to grow.
        </p>
      </Callout>

      {/* ── Gotchas ──────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Forgetting the pre-tokenization rule:</strong>{' '}
          BPE only merges within &ldquo;words&rdquo; as defined by your pre-tokenizer. If the
          librarian tallied with <code>ByteLevel(add_prefix_space=False)</code> and
          inference code later calls <code>add_prefix_space=True</code>, the token ids
          silently shift for every word after the first one. Save the pre-tokenizer config
          with the vocab. No exceptions.
        </p>
        <p>
          <strong className="text-term-amber">Version drift:</strong> Hugging Face{' '}
          <code>tokenizers</code> library updates sometimes change how ties in pair counts are
          broken, or how Unicode normalization is applied. Two librarians with the same
          corpus and same seed can write down subtly different merge orders across library
          versions. Pin the version that trained your production tokenizer.
        </p>
        <p>
          <strong className="text-term-amber">NFD vs NFC normalization:</strong> Unicode has
          multiple ways to write the same character. <code>é</code> can be one codepoint
          (NFC) or two (&ldquo;e&rdquo; + combining accent, NFD). If the librarian counted
          on NFC and you feed it NFD text at inference, accented characters become{' '}
          <em>two separate tokens</em>. Normalize at both ends of the pipe, identically.
        </p>
        <p>
          <strong className="text-term-amber">Special-token injection:</strong> if you
          naively pass user input through the tokenizer as plain text, a user who types{' '}
          <code>&lt;|endoftext|&gt;</code> literally into their prompt can inject the real
          document-boundary token and confuse your model. Use the <code>allowed_special</code>{' '}
          or <code>disallowed_special</code> flags and treat raw user text as untrusted.
        </p>
      </Gotcha>

      {/* ── Challenge ────────────────────────────────────────────── */}
      <Challenge prompt="Train a 4000-token BPE on Shakespeare">
        <p>
          Send the librarian through Tiny Shakespeare (<code>tinyshakespeare.txt</code>,
          about 1 MB). Using the Hugging Face <code>tokenizers</code> library, train a
          BPE with <code>vocab_size=4000</code>, ByteLevel pre-tokenization, and{' '}
          <code>min_frequency=2</code>.
        </p>
        <p className="mt-2">
          Take the 100 most common English words (from any standard list — Google&apos;s
          10k word list works). Run each through your tokenizer and count how many come
          out as a single token id. That&apos;s your &ldquo;single-token coverage&rdquo;
          score — how much of real English made it onto a Shakespearean clipboard.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: retrain with <code>vocab_size=1000</code> and <code>vocab_size=16000</code>.
          Plot single-token coverage vs vocab size. The curve flattens around{' '}
          <code>V ≈ 2000</code> — past that you&apos;re mostly paying for Shakespearean
          proper nouns and archaic spellings, which is exactly the corpus-vs-target-distribution
          point.
        </p>
      </Challenge>

      {/* ── Closing ──────────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> Vocabulary training is a
          librarian with a clipboard, a frequency cutoff, and a set of
          engineering decisions you&apos;ll live with for the life of the
          model. Vocab size trades sequence length against embedding
          parameters — 32k to 100k is the working range for English, larger
          for multilingual. Corpus selection <em>is</em> the vocabulary:
          the shelves you walk decide what gets tallied, which decides
          what&apos;s cheap to say. Upsample rare languages so their
          tokens make the cut. Reserve slots for the specials you&apos;ll
          need later. Special tokens go on after training, never before.
          And the whole artifact fits in a sub-megabyte JSON file you
          version like code.
        </p>
        <p>
          <strong>Next up — Tokenization Edge Cases.</strong> Our librarian
          built a clean clipboard on well-behaved text. Real text is not
          well-behaved. Emoji glued to punctuation, invisible zero-width
          joiners, URLs that end in tracking parameters, source code with
          tabs-vs-spaces holy wars, and the occasional user input that is{' '}
          <em>literally</em> the string <code>&lt;|endoftext|&gt;</code>.
          Next lesson: the places tokenizers break when the real world
          walks into the library, and what the production-tested defenses
          look like.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Neural Machine Translation of Rare Words with Subword Units',
            author: 'Sennrich, Haddow, Birch',
            venue: 'ACL 2016 — the BPE-for-NMT paper that started it all',
            year: 2015,
            url: 'https://arxiv.org/abs/1508.07909',
          },
          {
            title:
              'SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing',
            author: 'Kudo, Richardson',
            venue: 'EMNLP 2018',
            url: 'https://arxiv.org/abs/1808.06226',
          },
          {
            title: 'LLaMA: Open and Efficient Foundation Language Models',
            author: 'Touvron et al.',
            venue: 'Meta AI, 2023 — the reserved-token-slots paper',
            url: 'https://arxiv.org/abs/2302.13971',
          },
          {
            title: 'Hugging Face tokenizers — documentation',
            venue: 'huggingface.co/docs/tokenizers',
            url: 'https://huggingface.co/docs/tokenizers/index',
          },
          {
            title: 'tiktoken — OpenAI’s fast BPE tokenizer',
            venue: 'github.com/openai/tiktoken',
            url: 'https://github.com/openai/tiktoken',
          },
        ]}
      />
    </div>
  )
}
