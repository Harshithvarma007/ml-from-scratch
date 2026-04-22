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
import ShardedDataLoader from '../widgets/ShardedDataLoader'
import ContextWindowViz from '../widgets/ContextWindowViz'

// Signature anchor: the cafeteria line. Kitchen = dataset on disk, line = loader,
// table = training loop. Batch size = plates per trip. Shuffle = random plates.
// Workers = extra servers pre-loading the next tray. Introduced at the opening,
// returned to at batching/shuffle/workers, and again at the "stall" gotcha.
export default function GptDataLoaderLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="gpt-data-loader" />

      {/* ── Opening: the cafeteria line ─────────────────────────── */}
      <Prose>
        <p>
          Picture a cafeteria. The kitchen, somewhere in the back, has made an
          absurd amount of food — more than anyone will eat in a lifetime. The
          diner at the table, famished and impatient, can eat exactly one plate
          at a time. Between the two is a line: servers moving trays from the
          kitchen out to the table, at roughly the speed the diner chews. If
          the line keeps up, the table never stops eating. If the line lags,
          the diner stares at an empty tray while the GPU — sorry, while the
          diner — quietly bills you for nothing.
        </p>
        <p>
          That line is a <KeyTerm>data loader</KeyTerm>, and this whole lesson
          is about not starving the table. The kitchen has the food: GPT-3 ate
          roughly 300 billion tokens, Llama 3 ate 15 trillion, and even the
          &ldquo;small&rdquo; fine-tune you&apos;re about to run on a single
          GPU walks in the door with tens of billions. A plain Python list at
          that scale would need half a terabyte of RAM and ten minutes to
          pickle. The kitchen is too big to bring to the table in one trip. So
          you don&apos;t. You build a line.
        </p>
        <p>
          Every serious LLM codebase — nanoGPT, GPT-NeoX, Megatron, the actual
          OpenAI training infrastructure — converges on the <em>same</em> line.
          Tokenize the corpus once, offline. Write the token ids out as raw
          binary shards — flat trays of food the kitchen can hand over without
          ceremony. At train time the loader memory-maps those shards and
          ladles random slices onto the table. No database, no JSON, no
          pickle, no custom serializer. Just <code>numpy.memmap</code> and some
          arithmetic, running fast enough that the diner never stops chewing.
        </p>
        <p>
          Get this wrong and your $30k/hour cluster becomes a $30k/hour
          file-system benchmark — a very expensive cafeteria with an empty
          table. Get it right and you barely think about it again, which is
          how you know the line is doing its job.
        </p>
      </Prose>

      <Callout variant="insight" title="three parts of the cafeteria">
        <div className="space-y-2">
          <p>
            Three nouns you&apos;ll see over and over. Keep them straight and
            the rest of the lesson is bookkeeping.
          </p>
          <p>
            <strong>Kitchen</strong> — the <em>dataset</em>. Bytes on disk. The
            pre-tokenized corpus, sharded into binary files. Huge, cold, and
            unopinionated about what the diner eats next.
          </p>
          <p>
            <strong>Line</strong> — the <em>data loader</em>. Servers shuttling
            plates from the kitchen to the table. Picks which plates, in what
            order, how many at once, and how far ahead to pre-load.
          </p>
          <p>
            <strong>Table</strong> — the <em>training loop</em>. One diner, one
            mouth. Asks for a plate, computes a forward and backward pass, asks
            for the next plate. Never waits politely; billable by the second.
          </p>
        </div>
      </Callout>

      {/* ── Pipeline diagram ────────────────────────────────────── */}
      <AsciiBlock caption="the LLM data pipeline — offline once, then online forever">
{`  ┌─────────────────┐     tokenize     ┌─────────────────────┐
  │  raw corpus     │   ───────────▶   │  shard_0000.bin     │
  │  (500 GB text)  │   tiktoken /     │  shard_0001.bin     │   100 M tokens
  │  books, code,   │   sentencepiece  │  shard_0002.bin     │   each, uint16
  │  CC, arxiv…     │   BPE encoder    │        ⋮            │   ~200 MB file
  └─────────────────┘                  │  shard_9999.bin     │
                                       └──────────┬──────────┘
                                                  │ np.memmap(...)
                                                  ▼
                                       ┌─────────────────────┐
                                       │  OS virtual memory  │   no load cost
                                       │  page cache / mmap  │   OS handles I/O
                                       └──────────┬──────────┘
                                                  │ idx = randint()
                                                  │ x = data[idx:idx+T]
                                                  │ y = data[idx+1:idx+T+1]
                                                  ▼
                                       ┌─────────────────────┐
                                       │  DataLoader batch   │   (B, T) int64
                                       │  B chunks stacked   │   ───▶ GPU
                                       └─────────────────────┘`}
      </AsciiBlock>

      <Prose>
        <p>
          Everything interesting happens once, up front, in the kitchen: the
          raw corpus gets run through a BPE tokenizer (
          <NeedsBackground slug="tokenizer-bpe">tokens</NeedsBackground> are
          just integer ids) and the results get dumped into binary shards.
          After that, training is just <em>reading random trays out of a big
          int array</em>. That&apos;s it. That&apos;s the whole thing.
        </p>
      </Prose>

      {/* ── Widget 1: Sharded loader ────────────────────────────── */}
      <ShardedDataLoader />

      <Prose>
        <p>
          A 100 GB corpus pre-tokenized into 100 M-token shards. Each shard is
          about 200 MB — small enough to fit under every filesystem&apos;s 2 GB
          limit, small enough to live on cheap object storage, small enough to
          download in a minute. The cursor iterating through them is the
          training loop: it&apos;s not streaming in the conventional sense; the
          shards sit on disk and the OS pages in only the bytes the model
          actually touches. The line never carries more food than the next
          plate needs.
        </p>
        <p>
          Why shard at all instead of one giant vat? Three reasons.{' '}
          <strong>One</strong>, portability — every filesystem can handle a
          200 MB tray, not all can handle 100 GB. <strong>Two</strong>,
          parallelism — eight servers can each open their own shard without
          elbowing each other for the same file handle. <strong>Three</strong>,
          cheap shuffle — you pick a random shard, then a random offset inside
          it, and you&apos;ve drawn a uniform sample from the entire kitchen
          without ever carrying the whole kitchen out to the table.
        </p>
      </Prose>

      <Personify speaker="Memory-mapped file">
        I am not loaded. I am not streamed. I am a <em>promise</em>. When your
        code writes <code>data[4_837_291]</code>, the OS walks the page table,
        notices the 4 KB page containing that byte isn&apos;t resident, fetches
        it from disk, and hands you the integer — all in a few microseconds.
        You think you have a giant tray in memory. You have a file descriptor
        and a pointer. That is the entire trick.
      </Personify>

      <Callout variant="insight" title="why memmap instead of torch.load">
        <code>torch.load</code> on a 200 GB tensor reads every byte off disk
        into RAM before your first training step. That&apos;s a 20-minute
        stall at the table — the diner sitting with cutlery in hand while the
        kitchen laboriously wheels out the entire pantry. <code>np.memmap</code>{' '}
        returns instantly; it just registers the file with the kernel&apos;s
        virtual memory. Pages arrive on demand, get paged out under pressure,
        cached for the next access. The OS has spent fifty years making this
        fast. Use it.
      </Callout>

      {/* ── Input / target math ─────────────────────────────────── */}
      <Prose>
        <p>
          Now the actual plating. GPT is a next-token predictor. Every
          position in a sequence is a training example: given tokens up
          through position <code>i</code>, predict token <code>i+1</code>. So
          the input and target for a block of length <code>T</code> are two
          overlapping slices of the same tray, offset by one:
        </p>
      </Prose>

      <MathBlock caption="input/target pairing — just a shift">
{`given a token stream   tokens[0], tokens[1], tokens[2], ..., tokens[N-1]

sample a random index  i ∈ [0, N - T - 1]

input  x  =  tokens[i     : i + T    ]       ← length T
target y  =  tokens[i + 1 : i + T + 1]       ← length T, shifted by 1

loss = cross_entropy( model(x),  y )         ← one CE per position`}
      </MathBlock>

      <Prose>
        <p>
          That&apos;s the entire training signal. No labels, no annotations,
          no human in the loop. The <KeyTerm>self-supervised</KeyTerm> premise
          of language modeling is that every token in the corpus is its own
          label — the &ldquo;correct answer&rdquo; for position <code>i</code>{' '}
          is whatever literally came next. Free supervision on the entire
          internet. The kitchen writes its own answer key.
        </p>
      </Prose>

      {/* ── Widget 2: Context window ────────────────────────────── */}
      <ContextWindowViz />

      <Prose>
        <p>
          Slide the window. The blue row is <code>x</code>; the green row is{' '}
          <code>y</code>. <code>y</code> is just <code>x</code> shifted one
          position to the right — every element of <code>x</code> is looking
          at the element of <code>y</code> immediately next door and asking
          &ldquo;did I predict you?&rdquo;. With a context window of 1024
          tokens, one plate gives the model 1024 independent
          next-token-prediction problems — the per-position loss averages
          over all of them. That density is why transformers train
          efficiently; you&apos;re extracting <em>T loss signals</em> per
          forward pass, not one. Every plate is a buffet.
        </p>
      </Prose>

      <Callout variant="note" title="shuffle = random plates, not a reshuffle pass">
        A traditional DataLoader shuffles once per epoch — it mixes up the
        whole tray, then hands out plates in the new order. LLM training
        doesn&apos;t do epochs; the kitchen is too big, you may never see the
        same token twice. Instead the line draws a random <em>starting
        index</em> every time the table calls for a plate. The distribution
        over possible plates is already uniform, no shuffle pass needed. And
        because the shards are memmapped, picking random indices costs
        nothing — no reshuffle, no data rewrite, just a new offset.
      </Callout>

      <Personify speaker="Shard">
        I am a flat array of <code>uint16</code>s. Two bytes per token, 100
        million tokens, 200 MB on disk. I have no structure beyond
        &ldquo;token, token, token&rdquo; — no sentence boundaries, no
        document boundaries the model can see, just a stream. The separator
        token <code>&lt;|endoftext|&gt;</code> lives inline with everything
        else; the model has to learn what it means. I am boring and I am
        fast, and those two properties are why the table never stalls.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three implementations, each shorter and faster than the last. Pure
          Python with line-by-line file reads (the line a single server can
          barely walk). NumPy with <code>np.memmap</code> (what nanoGPT
          actually does). <NeedsBackground slug="pytorch-basics">PyTorch tensors</NeedsBackground>{' '}
          wrapped in a <code>Dataset</code> so the standard DataLoader can
          run eight servers at once and drop plates directly onto GPU memory
          — what you&apos;d plug into a real{' '}
          <NeedsBackground slug="training-loop">training loop</NeedsBackground>.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python, the slow way · loader_scratch.py"
        output={`batch 0: x[:8]=[ 842  103 1127  577   11   29 4982   13]
batch 0: y[:8]=[ 103 1127  577   11   29 4982   13 2001]
(took 3.4s per batch — GPU will starve)`}
      >{`import random
import time

# Imagine tokens.txt — one integer token id per line, 100 M lines.
# This is the "naive Python list" approach: don't do this.

def load_all_tokens(path):
    with open(path) as f:
        return [int(line) for line in f]           # 1.5 GB of Python ints

def get_batch(tokens, block_size, batch_size):
    xs, ys = [], []
    for _ in range(batch_size):
        i = random.randint(0, len(tokens) - block_size - 1)
        xs.append(tokens[i     : i + block_size])
        ys.append(tokens[i + 1 : i + block_size + 1])
    return xs, ys

t0 = time.time()
tokens = load_all_tokens("tokens.txt")              # 90 s to load
x, y = get_batch(tokens, block_size=1024, batch_size=32)
print(f"batch 0: x[:8]={x[0][:8]}")
print(f"batch 0: y[:8]={y[0][:8]}")
print(f"(took {time.time()-t0:.1f}s per batch — GPU will starve)")`}</CodeBlock>

      <Prose>
        <p>
          Everything about that is wrong. Loading 100 M ints into a Python
          list takes minutes and burns 1.5 GB of RAM on the interpreter&apos;s
          boxed-int overhead. Slicing a Python list copies. The file format
          is text, so every read re-parses digits. One server, dragging the
          whole kitchen to the table on every trip. But the <em>shape</em> of
          the loop — sample index, slice input, slice target — is already
          right, and it&apos;s the thing we&apos;re about to make 1000×
          faster.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy memmap, the real thing · loader_numpy.py"
        output={`shard: data_0003.bin  length=100000000  dtype=uint16
x.shape=(32, 1024)  y.shape=(32, 1024)
y == x shifted by 1:  True
(0.3 ms per batch — disk barely touched)`}
      >{`import numpy as np

# shards were written once, offline, as raw binary:
#   ids = tokenizer.encode(text)                    # list[int]
#   np.array(ids, dtype=np.uint16).tofile("data_0003.bin")

def open_shard(path):
    return np.memmap(path, dtype=np.uint16, mode="r")   # instant, no load

def get_batch(data, block_size, batch_size, rng):
    # one vectorised call draws batch_size random starts
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x  = np.stack([data[i     : i + block_size    ] for i in ix])
    y  = np.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.astype(np.int64), y.astype(np.int64)       # int64 for embeddings

rng  = np.random.default_rng(0)
data = open_shard("data_0003.bin")
print(f"shard: data_0003.bin  length={len(data)}  dtype={data.dtype}")

x, y = get_batch(data, block_size=1024, batch_size=32, rng=rng)
print(f"x.shape={x.shape}  y.shape={y.shape}")
print(f"y == x shifted by 1:  {np.array_equal(y[:, :-1], x[:, 1:])}")`}</CodeBlock>

      <Bridge
        label="pure python → numpy memmap"
        rows={[
          {
            left: 'tokens = [int(l) for l in f]',
            right: 'data = np.memmap(path, dtype=uint16)',
            note: 'no load — OS pages in bytes on first access',
          },
          {
            left: 'xs.append(tokens[i:i+T])',
            right: 'np.stack([data[i:i+T] for i in ix])',
            note: 'slice is a view into memory, not a copy',
          },
          {
            left: 'for _ in range(B): randint(...)',
            right: 'rng.integers(0, N-T-1, size=B)',
            note: 'one call draws B starts — trivially vectorised',
          },
        ]}
      />

      <Prose>
        <p>
          Wrap that same memmap in a <code>torch.utils.data.Dataset</code> and
          the standard PyTorch DataLoader will parallelise it across workers,
          pin memory, and prefetch plates onto the GPU before the table has
          finished the current one. The Dataset itself is ten lines. All the
          heavy lifting already happened offline, in the tokenizer.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch Dataset · loader_pytorch.py"
        output={`step 0  x.shape=torch.Size([16, 1024])  y.shape=torch.Size([16, 1024])
step 1  x.shape=torch.Size([16, 1024])  y.shape=torch.Size([16, 1024])
GPU util: 94%  (was 3% with naive loader)`}
      >{`import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ShardedTokenDataset(Dataset):
    """Random-access next-token dataset over a single memmapped shard."""
    def __init__(self, path, block_size, length=10_000):
        self.data       = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.length     = length          # "virtual" epoch size

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        T  = self.block_size
        i  = np.random.randint(0, len(self.data) - T - 1)
        x  = torch.from_numpy(self.data[i     : i + T    ].astype(np.int64))
        y  = torch.from_numpy(self.data[i + 1 : i + T + 1].astype(np.int64))
        return x, y

ds = ShardedTokenDataset("data_0003.bin", block_size=1024, length=10_000)
dl = DataLoader(ds, batch_size=16, num_workers=4, pin_memory=True)

for step, (x, y) in enumerate(dl):
    print(f"step {step}  x.shape={x.shape}  y.shape={y.shape}")
    if step == 1: break`}</CodeBlock>

      <Bridge
        label="numpy → pytorch Dataset"
        rows={[
          {
            left: 'data = np.memmap(path, …)',
            right: 'self.data = np.memmap(path, …) (in __init__)',
            note: 'each worker opens its own memmap — mmap is process-safe',
          },
          {
            left: 'get_batch(data, T, B, rng)',
            right: '__getitem__ returns one (x, y)',
            note: 'DataLoader handles batching + multi-worker + pin_memory',
          },
          {
            left: 'x.astype(np.int64)',
            right: 'torch.from_numpy(...).long()',
            note: 'embedding lookup wants int64; uint16 is just storage',
          },
        ]}
      />

      <Callout variant="insight" title="the point of three layers (again)">
        Pure Python is the line you&apos;d sketch on a whiteboard — one
        server, slow feet. NumPy + memmap is the nanoGPT load cell — this is
        genuinely the code Karpathy ships; it&apos;s not a simplification.
        PyTorch Dataset wraps that same memmap so the standard DataLoader can
        run a crew of servers in parallel, pre-plating the next tray while
        the table is still eating the current one. No step adds conceptual
        weight — each just adds the machinery of the next scale up.
      </Callout>

      {/* ── Callouts: batching, packing, workers ────────────────── */}
      <Callout variant="note" title="batch size: plates per trip">
        <code>batch_size</code> is the number of plates the line carries on
        one trip. Bigger trips mean the GPU crunches more examples in
        parallel per forward pass — better utilization — but the kitchen has
        to fit more trays in RAM and the gradient noise drops, which can
        actually hurt learning on small datasets. Start at 32 or 64 for small
        models, push up to 512+ on big ones, and watch loss curves and GPU
        memory together. Too small and the table waits between bites; too
        large and the kitchen runs out of counter space.
      </Callout>

      <Callout variant="note" title="packing short documents">
        Your corpus is full of short plates — tweets, stack-exchange answers,
        code snippets. Treat each as its own training example and every short
        one needs padding to block_size, wasting compute on{' '}
        <code>&lt;pad&gt;</code> tokens. Instead you <em>pack</em>: concatenate
        every document into one long tray, separated by a special{' '}
        <code>&lt;|endoftext|&gt;</code> token, and sample fixed-size blocks
        from that stream. A single block might span three full documents and
        half of a fourth. The model learns from{' '}
        <code>&lt;|endoftext|&gt;</code> what a document boundary looks like.
        Zero padding, zero wasted compute.
      </Callout>

      <Callout variant="warn" title="num_workers: extra servers on the line">
        This is the stall section — read it twice. With{' '}
        <code>num_workers=0</code> (the default), data loading runs on the
        same Python thread as your training step: forward, backward, loader,
        forward, backward, loader, serially. One server walking back to the
        kitchen every time the diner finishes a plate. The GPU idles half the
        time, which shows up as 50% utilization and a training run that takes
        twice as long as it should. Set <code>num_workers=4</code> (or 8, or
        whatever keeps utilization high) and PyTorch spins up background
        processes that pre-plate the next batch while the GPU is still eating
        the current one. Combined with <code>pin_memory=True</code>, the
        host→GPU transfer overlaps with computation. This single setting is
        the difference between a three-day run and a seven-day run — the
        difference between a cafeteria and a line at the DMV.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">uint16 vs uint32:</strong>{' '}
          GPT-2&apos;s vocabulary is 50,257 tokens — fits in a uint16 (max
          65,535). Llama&apos;s is 128,000 — needs uint32. Save the wrong
          dtype and your shards are either twice the size they need to be, or
          silently truncating token ids mod 65,536. There is no error. The
          kitchen just starts plating garbage.
        </p>
        <p>
          <strong className="text-term-amber">Endianness across machines:</strong>{' '}
          <code className="text-dark-text-primary">np.memmap</code> reads the
          host byte order by default. Tokenize on x86, train on a weird ARM
          cluster, the bytes swap and you get nonsense ids. Write shards with
          an explicit dtype (<code className="text-dark-text-primary">&apos;&lt;u2&apos;</code>{' '}
          for little-endian uint16) so there&apos;s no ambiguity.
        </p>
        <p>
          <strong className="text-term-amber">Skipping num_workers:</strong>{' '}
          covered above. If GPU utilization is below 80%, the line is the
          bottleneck, full stop. The table is starving and you&apos;re paying
          per second. Profile with{' '}
          <code className="text-dark-text-primary">nvidia-smi dmon</code>{' '}
          before you profile the model.
        </p>
        <p>
          <strong className="text-term-amber">mmap address space on 32-bit systems:</strong>{' '}
          a 32-bit process can only address ~4 GB of virtual memory total, so
          <code className="text-dark-text-primary">np.memmap</code> on a bigger
          shard fails. In 2026 this basically means &ldquo;don&apos;t train
          LLMs on a Raspberry Pi 3,&rdquo; but if you&apos;re on a 32-bit ARM
          edge device and wondering why{' '}
          <code className="text-dark-text-primary">mmap</code> raises{' '}
          <code className="text-dark-text-primary">ENOMEM</code>, that&apos;s
          why.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Ship a real data loader end-to-end">
        <p>
          Download <a href="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt">Tiny
          Shakespeare</a> (about 1 MB of text, 1 M characters). Encode it with{' '}
          <code>tiktoken.get_encoding(&quot;gpt2&quot;)</code> — you&apos;ll
          get roughly 300k BPE tokens. Save them to{' '}
          <code>shakespeare.bin</code> as <code>np.uint16</code>. That&apos;s
          your kitchen.
        </p>
        <p className="mt-2">
          Build a PyTorch <code>Dataset</code> that memmaps the file and
          returns random 256-token <code>(x, y)</code> plates. Wrap it in a{' '}
          <code>DataLoader</code> with{' '}
          <code>batch_size=32, num_workers=2</code> — that&apos;s a line with
          two servers. Pull one batch and assert{' '}
          <code>(y[:, :-1] == x[:, 1:]).all()</code> — every target must be
          the input shifted by one, on every row. If that assert passes, the
          line is calibrated.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: decode <code>x[0]</code> back to text with{' '}
          <code>enc.decode(x[0].tolist())</code> and read it. You should see a
          random chunk of Shakespeare. Congratulations — the table has
          something to eat.
        </p>
      </Challenge>

      {/* ── Closing ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> LLM data loading is a solved
          problem and the solution is boring on purpose. Tokenize once,
          offline. Write shards as raw binary. Memmap them at train time so
          the OS handles paging. Sample random indices — no epochs, no
          reshuffle pass; shuffle <em>is</em> a new random offset. Pair input
          with target by slicing the same tray twice, offset by one. Let the
          DataLoader run a crew of workers so the line never lags behind the
          table. That&apos;s it. That&apos;s the whole cafeteria.
        </p>
        <p>
          <strong>Next up — GPT Dataset.</strong> The loader you just built
          reads from a single shard. Real training rotates through thousands
          of shards, sometimes weighting them by quality (code gets 3×,
          CommonCrawl gets 1×, books get 5×) and sometimes scheduling them by
          phase of training. Next lesson we turn the one-shard Dataset into a{' '}
          <em>curriculum</em> — a weighted, ordered, multi-shard kitchen that
          matches what nanoGPT and GPT-NeoX actually use in production. The
          line gets smarter about which tray to grab next.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'nanoGPT',
            author: 'Andrej Karpathy',
            venue: 'GitHub — the canonical minimal GPT training repo',
            url: 'https://github.com/karpathy/nanoGPT',
          },
          {
            title: 'The Pile: An 800GB Dataset of Diverse Text for Language Modeling',
            author: 'Gao et al.',
            year: 2020,
            url: 'https://arxiv.org/abs/2101.00027',
          },
          {
            title: 'RedPajama: an Open Dataset for Training Large Language Models',
            author: 'Together AI',
            venue: '1.2 T token open reproduction of the Llama pretraining mix',
            url: 'https://github.com/togethercomputer/RedPajama-Data',
          },
          {
            title: 'torch.utils.data — PyTorch Documentation',
            venue: 'Dataset, DataLoader, num_workers, pin_memory',
            url: 'https://pytorch.org/docs/stable/data.html',
          },
          {
            title: 'numpy.memmap — NumPy Documentation',
            venue: 'The primitive every LLM loader is built on',
            url: 'https://numpy.org/doc/stable/reference/generated/numpy.memmap.html',
          },
        ]}
      />
    </div>
  )
}
