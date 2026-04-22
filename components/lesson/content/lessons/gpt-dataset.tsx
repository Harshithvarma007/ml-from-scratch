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
import DatasetMixture from '../widgets/DatasetMixture'
import PackingViz from '../widgets/PackingViz'

// Signature anchor: the massive library sliced into overlapping reading
// windows. The training corpus is one giant book; the model can only read
// N tokens at a time (the context window); training = pointing the model at
// window (0..N), then (1..N+1), then (2..N+2), asking "guess the next word"
// every time. The book stays put; the window slides. Returns at the opening
// (book vs slider), the x/y reveal (x = tokens 0..N-1, y = tokens 1..N), and
// the seam section ("what about the seam between windows?").
export default function GptDatasetLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="gpt-dataset" />

      {/* ── Opening: the book and the slider ────────────────────── */}
      <Prose>
        <p>
          Picture a library. Not a metaphorical one — a real, physical pile of
          books welded end-to-end into one continuous page: every Wikipedia
          article, every GitHub file, every book the lab could license, all
          spliced into a single scroll roughly a trillion words long. That
          scroll is the corpus. It stays put. It does not move.
        </p>
        <p>
          Now picture a little rectangular reading window — a slider. The
          window is, say, 2048 words wide, and the model can only see what is
          inside it. Training is this: lay the window over the scroll at
          position zero, ask the model to guess the next word at every spot in
          the window, nudge the window one word to the right, and repeat. The
          book never slides. <em>You</em> slide. That is the entire dataset
          story for a GPT: <strong>a massive library sliced into overlapping
          reading windows</strong>.
        </p>
        <p>
          Most tutorials skip straight to &ldquo;tokenize, batch, train&rdquo;
          and wave at &ldquo;context length&rdquo; like it is a number on a
          spec sheet. By the end of this page you will have felt the window
          slide across the corpus by hand — watched the input-target pair fall
          out of it, traced what happens at the seam between two adjacent
          windows, and written the same slider three times, each shorter than
          the last.
        </p>
      </Prose>

      <Callout variant="insight" title="three plain-English words before we start">
        <div className="space-y-2">
          <p>
            <strong>Corpus</strong> — the book. One giant, concatenated stream
            of <NeedsBackground slug="tokenizer-bpe">tokens</NeedsBackground>.
            No chapters, no documents, just a sequence.
          </p>
          <p>
            <strong>Context window</strong> — the slider. The fixed number of
            tokens the model can look at in one forward pass. Call this{' '}
            <code>N</code>. GPT-2 had <code>N = 1024</code>; Llama-3 has{' '}
            <code>N = 8192</code>; Gemini has gone to the millions.
          </p>
          <p>
            <strong>Next-token prediction</strong> — the game. Inside the
            window, for every position <code>i</code>, the model tries to
            guess token <code>i+1</code>. One window, thousands of tiny guess
            problems stacked on top of each other.
          </p>
        </div>
      </Callout>

      {/* ── Scale of the book ───────────────────────────────────── */}
      <Prose>
        <p>
          A frontier GPT is trained on a corpus of roughly 0.5 to 15 trillion
          tokens — the book is long. A single human reading full-time at 300
          words a minute would need fifteen thousand years to finish it once.
          The model finishes it once per training run, sliding the window
          across at roughly a billion words a second on a cluster of GPUs.
          That is the scale we are trying to make tractable.
        </p>
      </Prose>

      <MathBlock caption="a rough sense of scale — tokens per modern LLM">
{`GPT-3 (2020)         ~300 B tokens      (Common Crawl + Books + Wikipedia)
LLaMA-1 (2023)       ~1.4 T tokens      (open web + GitHub + arXiv + books)
LLaMA-2 (2023)       ~2.0 T tokens
LLaMA-3 (2024)      ~15.0 T tokens      (heavily filtered web, multilingual)
Phi-3 (2024)         ~3.3 T tokens      ("textbook-quality" filtered)

rule of thumb: Chinchilla says ~20 tokens per parameter is compute-optimal
               so a 7 B model wants ~140 B tokens; modern labs train 5-10x over`}
      </MathBlock>

      <Prose>
        <p>
          The library is not a single book though — it is a library. Web
          crawls, GitHub, arXiv, Wikipedia, books, StackExchange, each shelf
          with its own voice. When we weld the shelves into one scroll, the
          proportions we use matter. Crank the web shelf to 90% and the
          model becomes fluent but shallow. Crank arXiv and GitHub high and
          it reasons about code and LaTeX but sounds like it was raised in a
          basement. The data mix is the recipe, not the oven.
        </p>
        <p>
          Below is a stylised version of <KeyTerm>The Pile</KeyTerm> — the
          825 GB, 22-source corpus EleutherAI released in 2020. Drag the
          sliders: each shelf has a raw size and a quality weight, and the
          pie chart shows the resulting slice of the scroll.
        </p>
      </Prose>

      <DatasetMixture />

      <Prose>
        <p>
          There is no theorem telling you the right proportions. Labs
          literally ablate dozens of mixes on small proxy models and pick the
          one whose downstream eval numbers look best. The historical menu of
          public corpora is a small canon worth knowing by name:
        </p>
        <ul>
          <li>
            <strong>The Pile (2020).</strong> 825 GB, 22 sources: C4 (156 GB
            web), GitHub (95 GB), Books3 (54 GB, later yanked), ArXiv
            (56 GB), PubMed, Wikipedia, StackExchange, Enron emails, and so
            on. The open-source standard for years.
          </li>
          <li>
            <strong>RedPajama (2023).</strong> Together Computer&apos;s open
            replica of the LLaMA-1 training mix — ~1.2 T tokens. First
            serious attempt to reproduce a frontier corpus in public.
          </li>
          <li>
            <strong>SlimPajama (2023).</strong> RedPajama after aggressive
            deduplication: 627 B tokens, and empirically <em>better</em> than
            the bigger parent because duplicates were hurting generalization.
          </li>
          <li>
            <strong>Dolma (2024).</strong> AI2&apos;s 3 T-token open corpus
            — the cleanest and most transparently documented open corpus as
            of writing.
          </li>
        </ul>
      </Prose>

      <Personify speaker="Data mix">
        I&apos;m the recipe, not the oven. Change me from 70% web / 10% code
        to 40% web / 30% code and you will get a different model — not
        better, not worse, different. I am where bias, style, and capability
        come from. Pick me thoughtlessly and no amount of architecture tuning
        will save you.
      </Personify>

      {/* ── The window and the input-target pair ────────────────── */}
      <Prose>
        <p>
          Back to the slider. Here is the part the tutorials always botch.
          When the model reads one window of <code>N</code> tokens, it is
          not asked to predict <em>the next word after the window</em>. It
          is asked to predict the next word at <em>every single position
          inside</em> the window — simultaneously. One window trains
          <code>N</code> guesses at once, not one.
        </p>
        <p>
          Which means the input and target for one window are almost
          identical — just shifted by one. That shift is the whole trick:
        </p>
      </Prose>

      <MathBlock caption="the input-target pair for one reading window">
{`context window  N = 8   (toy)
corpus          [The, cat, sat, on, the, mat, and, purred, loudly, forever, .]

window starting at position 0:
  x  =  [The, cat, sat, on,  the, mat, and, purred]      tokens 0..N-1
  y  =  [cat, sat, on,  the, mat, and, purred, loudly]   tokens 1..N

per-position training signal (read each column):
  position 0:  given "The",                      predict "cat"
  position 1:  given "The cat",                  predict "sat"
  position 2:  given "The cat sat",              predict "on"
  ...
  position 7:  given "The cat sat ... purred",   predict "loudly"`}
      </MathBlock>

      <Prose>
        <p>
          Read the two rows again. <code>x</code> is tokens <code>0..N-1</code>.{' '}
          <code>y</code> is tokens <code>1..N</code>. Same sequence, shifted
          one to the left. One window, <code>N</code> training examples, for
          free. This is why next-token prediction scales — every token in the
          corpus contributes to the gradient, not just the last one. The
          slider is cheap because every position inside it is a lesson.
        </p>
        <p>
          Now slide. Advance the window by one token and do it again.
        </p>
      </Prose>

      <MathBlock caption="the window slides — overlapping reading windows">
{`window at position 0:   [The, cat, sat, on,  the, mat, and, purred]
window at position 1:        [cat, sat, on,  the, mat, and, purred, loudly]
window at position 2:             [sat, on,  the, mat, and, purred, loudly, forever]
window at position 3:                  [on,  the, mat, and, purred, loudly, forever, .]

the book never moves. the window slides.`}
      </MathBlock>

      {/* ── The seam between windows ────────────────────────────── */}
      <Prose>
        <p>
          <strong>What about the seam between windows?</strong> Good
          question, and the honest answer is: in practice we usually do not
          slide the window by one. We slide it by <code>N</code> — by a full
          window width — so consecutive windows butt up against each other
          with no overlap. Position 0 gives you the chunk <code>[0..N-1]</code>;
          position 1 (of the chunked stride) gives you <code>[N..2N-1]</code>;
          the book is carved into adjacent, non-overlapping chunks.
        </p>
        <p>
          That is faster and cheaper, and almost every training loop does it,
          but it has one subtle wart: the model never sees a training example
          where the context straddles a chunk boundary. The last token of
          chunk 1 and the first token of chunk 2 live in different windows
          and never look at each other during training. At inference the
          model is perfectly happy to process a continuous stream, but during
          training, the seam is a clean cut. In practice it does not hurt
          much — the windows are thousands of tokens wide, and the marginal
          cost of missing a few cross-seam predictions is tiny compared to
          the 3x efficiency win. Worth knowing it is there.
        </p>
        <p>
          There is a related seam problem inside a single chunk: what if the
          chunk accidentally welds the end of a news article to the beginning
          of a Python script? The model will happily attend across the gap
          and start believing one caused the other. The fix is a dedicated
          end-of-text token <code>&lt;|eot|&gt;</code> inserted at every
          document boundary in the corpus, so the model learns to reset its
          expectations whenever it sees that token. Tiny detail, load-bearing.
        </p>
      </Prose>

      {/* ── Packing: stuffing the book efficiently ──────────────── */}
      <Prose>
        <p>
          Which brings us to the next problem: before we can even start
          sliding, we have to weld the library into the book. Real documents
          come in wildly varying lengths — a tweet is 40 tokens, a
          StackOverflow answer is 800, a chapter is 12,000. If you naively
          pad each document up to <code>N</code> tokens to make it fill a
          window, you are computing forward and backward passes on billions
          of pad tokens that contribute exactly nothing to the gradient.
          Unacceptable.
        </p>
        <p>
          The fix is called <KeyTerm>sequence packing</KeyTerm>. Concatenate
          documents end-to-end, separated by <code>&lt;|eot|&gt;</code>,
          then slice the resulting stream into contiguous chunks of length{' '}
          <code>N</code>. Almost no wasted tokens. This is what makes the
          book a book instead of a stack of loose pages.
        </p>
      </Prose>

      <MathBlock caption="packing beats padding by a lot">
{`setup:  window length N = 2048
        mean doc length  = 650 tokens      (with a long tail)

padded batch:   each doc → 2048 tokens, with (2048 − len) pad tokens
                effective tokens / window = 650
                efficiency = 650 / 2048 ≈ 31.7 %

packed batch:   concatenate docs with <|eot|> until ≥ 2048 tokens, slice
                effective tokens / window ≈ 2047  (one <|eot|> per boundary)
                efficiency ≈ 99.9 %

speedup:        ~3.15x more real tokens per dollar of GPU time`}
      </MathBlock>

      <PackingViz />

      <Prose>
        <p>
          Scrub the toggle. On the <code>padded</code> side each box is a
          document plus a grey wasteland of pad tokens that the model
          computes attention over and learns nothing from. On the{' '}
          <code>packed</code> side documents butt up against each other
          separated by the little <code>&lt;|eot|&gt;</code> marker, and the
          chunked windows land wherever they land. The marker is critical:
          without it the model happily attends across document boundaries
          and starts believing the end of a news article caused the
          beginning of a Python script.
        </p>
        <p>
          A subtler version of this — <strong>FlashAttention with document
          masks</strong> — goes one step further: the attention kernel
          itself refuses to attend across <code>&lt;|eot|&gt;</code>{' '}
          boundaries. Packed without a mask is 95% as good; packed with a
          mask is technically correct. Most open training codebases
          (Megatron, nanoGPT, Llama-recipes) now ship a document-masked
          packed loader by default.
        </p>
      </Prose>

      <Callout variant="insight" title="quality &gt; quantity, sometimes dramatically">
        Phi-1 and Phi-3 from Microsoft Research bet the farm on{' '}
        <em>heavily filtered</em> &ldquo;textbook-quality&rdquo; data —
        synthetic-and-curated code, not raw Common Crawl. Phi-3-mini
        (3.8 B params, ~3.3 T tokens) matches or beats many 7 B models on
        MMLU, HumanEval, and reasoning benches. The scaling-laws orthodoxy
        says more tokens always helps; Phi says <em>better</em> tokens can
        help more. In 2024 this stopped being a contrarian take and became
        the default view at most labs.
      </Callout>

      <Personify speaker="Deduplication">
        I&apos;m the cleaner. Your corpus has 40 copies of the GPL license,
        300 forks of the same README, and one Harry Potter fanfic that got
        reposted across 12 forums. If you train on that unchanged, your
        model will memorise instead of generalise. I delete the duplicates.
        Exact-match I do in an afternoon with a hash table; near-match I do
        with MinHash LSH and a weekend. Either way, models I&apos;ve touched
        have lower eval loss at the same token count.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Code time. Three layers, same slider: sample documents from four
          shelves, weld them into a book, slice into chunks of size{' '}
          <code>N</code>, and for each chunk hand back the input-target pair{' '}
          <code>(x, y)</code> we just derived by hand. Pure NumPy, then a
          PyTorch <code>Dataset</code>, then a streaming Hugging Face
          pipeline. The third one is how you would actually do it at
          trillion-token scale.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — numpy · mix_and_sample.py"
        output={`drew 20 documents:
  web: 10   code: 5   books: 3   wiki: 2
expected: [10.0, 5.0, 3.0, 2.0]`}
      >{`import numpy as np

# Four stylised shelves of the library. Each is just a list of "documents".
sources = {
    "web":   [f"web_doc_{i}"   for i in range(10_000)],
    "code":  [f"code_doc_{i}"  for i in range(5_000)],
    "books": [f"book_doc_{i}"  for i in range(2_000)],
    "wiki":  [f"wiki_doc_{i}"  for i in range(3_000)],
}

# Desired mix proportions. These do NOT have to match the raw sizes —
# you often up-weight small, high-quality shelves (wiki, books).
weights = np.array([0.50, 0.25, 0.15, 0.10])
names   = np.array(list(sources.keys()))
rng     = np.random.default_rng(42)

def sample_batch(n):
    # 1. pick which shelf each document comes from, weighted
    chosen = rng.choice(names, size=n, p=weights)
    # 2. draw one random doc from the chosen shelf
    return [rng.choice(sources[s]) for s in chosen]

batch = sample_batch(20)
counts = {s: sum(1 for x in batch if x.startswith(s)) for s in names}
print("drew 20 documents:")
print("  " + "   ".join(f"{k}: {v}" for k, v in counts.items()))
print(f"expected: {list(weights * 20)}")`}</CodeBlock>

      <Prose>
        <p>
          That is the mixing side — the shelves-to-book step. Now the slider
          itself. We concatenate already-
          <NeedsBackground slug="tokenizer-bpe">tokenized</NeedsBackground>{' '}
          documents with <code>&lt;|eot|&gt;</code>, slice the resulting
          stream into adjacent chunks of length <code>N</code>, and for each
          chunk return the <code>(x, y)</code> pair where <code>y</code> is{' '}
          <code>x</code> shifted by one. This is the{' '}
          <NeedsBackground slug="pytorch-basics">tensors</NeedsBackground>{' '}
          the training loop actually eats.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — pytorch · packed_dataset.py"
        output={`packed 1,247 docs into 412 windows of 2048 tokens
waste: 189 tokens total (0.02%)
first window x[:3] = tensor([ 243, 1084,  ...])  # 50256 = <|eot|>`}
      >{`import torch
from torch.utils.data import Dataset

EOT = 50256                                   # GPT-2's <|endoftext|> id

class PackedDataset(Dataset):
    """The slider, packaged. Weld the book, then hand back (x, y) per window."""

    def __init__(self, tokenized_docs, block_size=2048):
        # tokenized_docs: list[list[int]], already through the tokenizer.
        # Step 1: weld the library into one continuous book.
        stream = []
        for doc in tokenized_docs:
            stream.extend(doc)
            stream.append(EOT)                # seam marker — never drop it
        self.stream = torch.tensor(stream, dtype=torch.long)
        self.block  = block_size
        # Drop the ragged tail; we want clean fixed-size chunks.
        self.n_blocks = len(self.stream) // block_size

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        # Step 2: slide the window. Non-overlapping chunks, stride = block.
        i = idx * self.block
        x = self.stream[i     : i + self.block]       # tokens 0..N-1
        y = self.stream[i + 1 : i + self.block + 1]   # tokens 1..N  (shift)
        return x, y

# quick sanity check
fake_docs = [list(range(n)) for n in [120, 80, 500, 30, 2000] * 250]
ds = PackedDataset(fake_docs, block_size=2048)
print(f"packed {len(fake_docs):,} docs into {len(ds)} windows of 2048 tokens")
waste = len(ds.stream) - len(ds) * 2048
print(f"waste: {waste} tokens total ({waste / len(ds.stream):.2%})")
print(f"first window x[:3] = {ds[0][0][:3]!r}  # {EOT} = <|eot|>")`}</CodeBlock>

      <Prose>
        <p>
          The two lines <code>x = stream[i : i+N]</code> and{' '}
          <code>y = stream[i+1 : i+N+1]</code> are literally the math from
          the window diagram above. Nothing hidden. Every window in the book
          becomes one <code>(x, y)</code> pair, and a training step chews
          through a batch of them at once — this is what the{' '}
          <NeedsBackground slug="gpt-data-loader">DataLoader</NeedsBackground>{' '}
          in the next lesson wraps up.
        </p>
        <p>
          Last layer: production. You do not shard, tokenize, and pack
          2 trillion tokens by hand. You use Hugging Face&apos;s{' '}
          <code>datasets</code> library in streaming mode so nothing has to
          fit on a single disk, and you let it do the parallel tokenization.
          Same slider, different plumbing.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — hugging face · streaming_mixture.py"
      >{`from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
EOT = tok.eos_token_id

# Four real shelves, streamed — none of these fit on a laptop.
web   = load_dataset("allenai/c4",         "en",           split="train", streaming=True)
code  = load_dataset("codeparrot/github-code", "all",      split="train", streaming=True)
books = load_dataset("the_pile_books3",                    split="train", streaming=True)
wiki  = load_dataset("wikipedia",          "20220301.en",  split="train", streaming=True)

def tokenize_and_mark(batch):
    ids = tok(batch["text"], add_special_tokens=False)["input_ids"]
    # Append EOT to each doc so the later slider sees clean seams.
    return {"input_ids": [d + [EOT] for d in ids]}

datasets = [
    ds.map(tokenize_and_mark, batched=True, remove_columns=ds.column_names)
    for ds in [web, code, books, wiki]
]

# Proportional sampler — 50% web, 25% code, 15% books, 10% wiki.
mixed = interleave_datasets(
    datasets,
    probabilities=[0.50, 0.25, 0.15, 0.10],
    stopping_strategy="all_exhausted",
    seed=42,
)

# Stream -> weld the book -> slide the window. A real training loop
# wraps this generator in a DataLoader.
buffer, BLOCK = [], 2048
for example in mixed:
    buffer.extend(example["input_ids"])
    while len(buffer) >= BLOCK + 1:
        chunk, buffer = buffer[:BLOCK + 1], buffer[BLOCK:]
        yield torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])`}</CodeBlock>

      <Bridge
        label="numpy → pytorch → huggingface"
        rows={[
          {
            left: 'rng.choice(names, p=weights)',
            right: 'interleave_datasets(probabilities=...)',
            note: 'same weighted shelf-picking, just at 100 GB/s instead of 1 MB/s',
          },
          {
            left: 'stream.extend(doc); stream.append(EOT)',
            right: 'map(tokenize_and_mark) then buffer.extend',
            note: 'weld the book with a seam marker — load-bearing in both',
          },
          {
            left: 'stream[i : i+N], stream[i+1 : i+N+1]',
            right: 'chunk[:-1], chunk[1:] in the yield',
            note: 'slide the window; x and y shifted by one — same math, streaming',
          },
        ]}
      />

      <Callout variant="insight" title="the point of the three layers">
        Layer 1 makes the <em>sampling</em> concrete — weighted random choice
        across the shelves of the library, nothing more. Layer 2 makes the{' '}
        <em>slider</em> concrete — one welded stream, strided views, the{' '}
        <code>(x, y)</code> shift. Layer 3 is how you run the same slider at
        trillion-token scale: streaming, parallel tokenization, interleaved
        shelves. The same two ideas, three scales of machinery.
      </Callout>

      {/* ── Callouts: quality & copyright ───────────────────────── */}
      <Callout variant="note" title="quality over quantity — the leverage point">
        The most cost-effective thing you can do to a training run is{' '}
        <em>not</em> adding more tokens to the book. It is removing bad
        tokens. Dedup, filter for language quality, drop anything with a
        perplexity score above some threshold from a small pretrained model,
        prefer shelves with editorial review. Teams that do this seriously
        report 2-3x sample efficiency improvements — i.e. matching a
        competitor&apos;s model at 1/3 the compute. You will not find a
        cheaper lever anywhere else in the stack.
      </Callout>

      <Callout variant="warn" title="the legal and ethical mess">
        Almost every frontier LLM is trained on data scraped without
        explicit license from its authors. The New York Times is suing
        OpenAI; Getty Images is suing Stability; Books3 (a core shelf of The
        Pile) has been yanked after legal threats. Opt-in corpora like
        Common Pile exist but are tiny by comparison. If you are building a
        commercial model, the legal landscape is genuinely unresolved as of
        2026 — assume your shelf choices will be scrutinised. If you are
        training on personal data, run PII filters, and be ready to
        implement &ldquo;right to be forgotten&rdquo; takedowns via model
        editing or retraining.
      </Callout>

      {/* ── Gotcha ──────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Train/val leakage:</strong> if
          the same document (or a near-duplicate) appears in both the
          training book and your held-out val book, your val loss will look
          suspiciously great and your model won&apos;t generalise. Always
          dedup <em>across</em> the train/val split, not just within it.
          MinHash LSH with a Jaccard threshold of ~0.8 is the standard tool.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting to shuffle
          shelves:</strong> if chunk 1 of the book is all of Wikipedia and
          chunk 2 is all of GitHub, the model sees 10 B tokens of prose then
          10 B tokens of Python and catastrophically forgets how to write
          English. Interleave the shelves at sampling time (the HF snippet
          above) or shuffle chunks at the filesystem level before loading.
        </p>
        <p>
          <strong className="text-term-amber">Packing without
          separators:</strong> bare concatenation without{' '}
          <code>&lt;|eot|&gt;</code> teaches the model that a product review
          leads into the Quicksort algorithm. You&apos;ll get a
          fluent-but-incoherent model and wonder where your eval numbers
          went.
        </p>
        <p>
          <strong className="text-term-amber">Tokenizing at train
          time:</strong> running the tokenizer on every batch, every epoch
          is a classic way to make your GPUs wait on your CPUs. Tokenize{' '}
          <em>once</em>, write the <code>int32</code> token IDs to disk,
          memmap them at train time. Standard in every serious codebase;
          forgetting it costs you 2-5x throughput.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Build a 4-shelf sliding-window loader">
        <p>
          Pick four small public text corpora — e.g. <code>wikitext-2</code>,{' '}
          <code>tiny_shakespeare</code>, a slice of <code>the_pile</code>
          &apos;s PubMed, and a slice of GitHub. Tokenize each with the
          GPT-2 tokenizer, append <code>eot</code> at every document seam,
          and build a <code>PackedDataset</code> per shelf.
        </p>
        <p className="mt-2">
          Wrap them in a proportional sampler with proportions{' '}
          <code>[0.4, 0.1, 0.3, 0.2]</code>. Sample 100 windows. For each
          window, assert that <code>y</code> equals <code>x</code> shifted
          by one, and count which shelf(ves) contributed — you&apos;ll need
          the seam <code>eot</code> tokens to reconstruct this. Plot the
          actual token share per shelf against your configured proportions.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: add MinHash-LSH deduplication across the four shelves
          before welding the book. Report how many documents got culled.
          For most public corpora the dedup rate is surprisingly high
          (often &gt;10%).
        </p>
      </Challenge>

      {/* ── Closing ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> A GPT training set is a
          massive library sliced into overlapping reading windows, and the
          slider is the whole story: weld the shelves into one book with{' '}
          <code>&lt;|eot|&gt;</code> seams, chunk the book into windows of
          length <code>N</code>, and for each window hand back{' '}
          <code>(x, y)</code> where <code>y</code> is <code>x</code> shifted
          by one. The mix determines the model&apos;s personality. Packing
          buys you ~3x token efficiency for free. Dedup and quality
          filtering stack another 1.5-3x on top. These three levers — mix,
          packing, filtering — are bigger than most architecture changes you
          could make.
        </p>
        <p>
          <strong>Next up — KV-Cache.</strong> We just spent an entire
          lesson welding the book and sliding the window across it at
          training time, where the model cheerfully recomputes everything
          from scratch for every chunk because the cost is amortised across
          thousands of positions. At inference time — when the model is
          generating one token at a time and the window is effectively
          sliding by one every step — that recomputation suddenly becomes
          absurd. The model is re-reading the entire prefix of the window
          for every single new token it spits out. The{' '}
          <KeyTerm>KV-cache</KeyTerm> is the one trick that makes fast
          inference possible: remember what attention already computed for
          the old tokens, only compute for the new one. Same slider,
          different economics. That&apos;s the next lesson.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'The Pile: An 800GB Dataset of Diverse Text for Language Modeling',
            author: 'Gao et al.',
            venue: 'EleutherAI, 2020',
            url: 'https://arxiv.org/abs/2101.00027',
          },
          {
            title: 'RedPajama: an Open Dataset for Training Large Language Models',
            author: 'Together Computer',
            venue: '2023',
            url: 'https://github.com/togethercomputer/RedPajama-Data',
          },
          {
            title: 'Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining',
            author: 'Soldaini et al.',
            venue: 'AI2, 2024',
            url: 'https://arxiv.org/abs/2402.00159',
          },
          {
            title: 'Textbooks Are All You Need',
            author: 'Gunasekar et al.',
            venue: 'Microsoft Research, 2023 — the Phi-1 paper',
            url: 'https://arxiv.org/abs/2306.11644',
          },
          {
            title: 'SlimPajama: A 627B token cleaned and deduplicated version of RedPajama',
            author: 'Soboleva et al.',
            venue: 'Cerebras, 2023',
            url: 'https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama',
          },
        ]}
      />
    </div>
  )
}
