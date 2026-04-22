import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
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
import Prereq from '../Prereq'
import NeedsBackground from '../NeedsBackground'
import EdgeCaseExplorer from '../widgets/EdgeCaseExplorer'
import TokenSplitInspector from '../widgets/TokenSplitInspector'

export default function TokenizationEdgeCasesLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="tokenization-edge-cases" />

      {/* ── Opening ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Every tokenizer has its Tower of Babel moments — the places where the tokenizer&apos;s
          grammar breaks down and strings that look identical start speaking different dialects
          to the model. Whitespace that isn&apos;t whitespace. Numbers that chunk weirdly. Unicode
          combining characters that fool byte-level counting. Emoji that are three bytes
          pretending to be one character. Each one is a speaker of a different dialect, walking
          up to the tokenizer&apos;s interpreter and getting translated into something the model
          has never heard before.
        </p>
        <p>
          The tokenizer is the thinnest, most-ignored layer in the entire stack, and it is
          responsible for roughly half of the weird bugs you will ever hit in production LLM
          work. It sits between a human string and the model&apos;s integer interface, and it
          does not behave the way you think strings do.
        </p>
        <p>
          You ship a prompt. It works. Your colleague copy-pastes it into Slack, which helpfully
          adds a leading space — a different dialect, and the completion quality drops 10%. You
          try to do arithmetic with GPT-4 and it gets 7-digit multiplication reliably wrong
          because the digits keep splitting in different places. A Japanese customer&apos;s bill
          is three times larger than an English customer&apos;s for the same conversation
          length. All three stories have the same root cause: the tokenizer broke down at an
          edge case and nobody was watching.
        </p>
        <p>
          This lesson is the Babel tour. The boring corners, the expensive corners, and the{' '}
          <em>haunted</em> corners — the rare tokens that make GPT hallucinate nonsense when you
          mention them by name. Assumes you&apos;ve walked through{' '}
          <NeedsBackground slug="intro-to-nlp">tokenization</NeedsBackground>, that you know{' '}
          <NeedsBackground slug="tokenizer-bpe">BPE</NeedsBackground> well enough to recognize a
          merge when you see one, and that you understand how a{' '}
          <NeedsBackground slug="build-vocabulary">vocabulary</NeedsBackground> gets baked into a
          fixed table. By the end you should trust the tokenizer exactly as much as it deserves,
          which is not very.
        </p>
      </Prose>

      <Personify speaker="Tokenizer">
        I am not the model. I am older, dumber, and permanent. I was trained once, on one
        corpus, and frozen. Every string you will ever send passes through me first. If I map
        your input to a weird sequence of IDs, the model sees weirdness — and it cannot look
        back through me to see your original text. That is the entire contract.
      </Personify>

      {/* ── Widget 1: EdgeCaseExplorer ──────────────────────────── */}
      <Prose>
        <p>
          A curated gallery of strings that embarrass the tokenizer — every dialect in one
          place. Click through each one. You&apos;ll see the same text tokenize{' '}
          <em>completely differently</em> depending on whether it has a leading space, lives in
          an emoji, uses a rare character, or happens to be a long number. Identical-looking
          inputs, different token IDs. The token count on the right is the thing you pay for,
          and the one the model has to reason through.
        </p>
      </Prose>

      <EdgeCaseExplorer />

      <Callout variant="note" title="the leading-space trap, in one picture">
        In GPT-4&apos;s tokenizer, <code>&quot;hello&quot;</code> is one token and{' '}
        <code>&quot; hello&quot;</code> (with the leading space) is a different one token — same
        word, different ID, different embedding. A sentence split as{' '}
        <code>&quot;The&quot; | &quot; cat&quot;</code> is two tokens; the same letters split as{' '}
        <code>&quot;The &quot; | &quot;cat&quot;</code> — with the space on the other side — are
        also two tokens, but <em>different</em> ones. The model trained on the first. The
        second is a small, real distributional shift — two strings that look the same to you
        collide into two different vectors inside the model.
      </Callout>

      <Personify speaker="Leading space">
        I am the silent byte at the front of your word. I am invisible to you, but not to the
        tokenizer — I make <code>&quot; Paris&quot;</code> a totally different vector from{' '}
        <code>&quot;Paris&quot;</code>. Your trim() call just changed the meaning of your
        prompt by a noticeable amount, and you didn&apos;t even notice I existed.
      </Personify>

      {/* ── Multilingual inflation math ──────────────────────────── */}
      <Prose>
        <p>
          Here is the edge case that should bother you most, because it crosses from &ldquo;weird
          bug&rdquo; into &ldquo;structural unfairness.&rdquo; A GPT-4-family tokenizer was
          trained predominantly on English text. English words lump together into single
          tokens. Languages it saw less often do not — they get chopped character by character,
          a foreign dialect that the greedy compressor never learned to pack. The ratio of
          tokens per character you pay for is <em>wildly</em> different by language.
        </p>
      </Prose>

      <MathBlock caption="tokens-per-character, same paragraph, translated — GPT-4 family">
{`English           :  ~0.25 tokens / char        ("the cat sat"            →  3 tokens / 11 chars)
Spanish           :  ~0.30 tokens / char        ~1.2x   cost vs English
Chinese (simpl.)  :  ~0.90 tokens / char        ~3.5x   cost vs English
Japanese          :  ~1.00 tokens / char        ~4.0x   cost vs English
Hindi (Devanagari):  ~1.20 tokens / char        ~4.8x   cost vs English
Burmese           :  ~2.00 tokens / char        ~8.0x   cost vs English`}
      </MathBlock>

      <Prose>
        <p>
          Read that table twice. A Japanese-speaking user of your product, holding a
          conversation identical in meaning to an English user&apos;s, will hit your context
          length window four times faster and pay four times more per API call. The Yenai &amp;
          Petrov 2024 paper documented ratios up to <code>15x</code> for some language /
          tokenizer combinations. This is not a rounding error — it&apos;s a tax on
          non-English dialects baked into the model&apos;s interface, a Babel penalty charged at
          the gate.
        </p>
        <p>
          The same mechanism — common strings merge, rare strings shatter — is the reason
          numbers tokenize like they&apos;re trying to fool you. <code>&quot;1234&quot;</code>{' '}
          might be a single token because it appeared often in training data (it&apos;s a
          common year). <code>&quot;1235&quot;</code> might split into <code>&quot;12&quot;</code>{' '}
          and <code>&quot;35&quot;</code>. <code>&quot;123456&quot;</code> might chunk as{' '}
          <code>&quot;12&quot;</code> + <code>&quot;345&quot;</code> + <code>&quot;6&quot;</code>
          , and <code>&quot;123456789&quot;</code> could be 4 tokens. This is why GPT-4 is worse
          at arithmetic than it &ldquo;should&rdquo; be — digits don&apos;t line up cleanly
          between problems, so the model can&apos;t just reason in a digit-by-digit way. It has
          to learn the algebra of chunks whose boundaries keep shifting.
        </p>
      </Prose>

      {/* ── Widget 2: TokenSplitInspector ───────────────────────── */}
      <Prose>
        <p>
          Type anything into the box. You&apos;ll see the actual GPT-4-style tokenization —
          token IDs, the byte decomposition of each token, and a running cost. Try{' '}
          <code>&quot; hello&quot;</code> vs <code>&quot;hello&quot;</code> — watch two
          identical-looking words collide into two different IDs. Try{' '}
          <code>&quot;1234567890&quot;</code> vs <code>&quot;2024&quot;</code>. Paste a
          sentence in Japanese. Paste some Python. Paste a URL. Watch the tokenizer flail its
          way through each dialect in turn.
        </p>
      </Prose>

      <TokenSplitInspector />

      <Callout variant="insight" title="the one reliable intuition">
        Common strings → few tokens. Rare strings → many tokens. A URL slug like{' '}
        <code>&quot;github.com&quot;</code> is cheap because the tokenizer saw a billion of
        them; a UUID is expensive because it&apos;s never seen any particular UUID. This alone
        explains 90% of the tokenizer&apos;s behavior — it is a frequency-based compressor, and
        every edge case is a string that didn&apos;t make the frequency cut.
      </Callout>

      {/* ── SolidGoldMagikarp section ────────────────────────────── */}
      <Prose>
        <p>
          Now the haunted corner — the deepest Babel moment of them all. In 2023, Rumbelow and
          Watkins went searching for tokens in GPT-2&apos;s vocabulary that had been{' '}
          <em>allocated</em> a token ID but that appeared almost never in the training data. The
          tokenizer had learned them from one corpus — largely Reddit usernames — and the model
          had been trained on a different, cleaner one. So the model saw those token IDs
          roughly zero times during training. Two corpora, two dialects, and one poor model
          trying to read both.
        </p>
        <p>
          What happens when you prompt a model with a token it has never trained on? The
          model&apos;s embedding for that token is basically untouched random noise. The
          behavior goes off the rails. Asking GPT-3 to &ldquo;please repeat the string{' '}
          <em>SolidGoldMagikarp</em> back to me&rdquo; caused it to output{' '}
          <code>&quot;distribute&quot;</code>, insult the user, or refuse to acknowledge that
          the word existed. Dozens of such tokens were found. We now call them{' '}
          <KeyTerm>glitch tokens</KeyTerm>, and every production tokenizer has them in some
          number.
        </p>
      </Prose>

      <Personify speaker="Rare token">
        I was baked into the tokenizer&apos;s vocabulary years ago, during an era of the
        internet that no longer exists. My embedding vector has never been gradient-updated. I
        am a ghost in the vocabulary — a valid ID with no trained meaning. If you summon me
        by name, the model will speak in tongues.
      </Personify>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers of code, smallest to largest. Pure Python to stage the leading-space
          trap with a toy BPE-like vocab so you can read the collision in a dict. NumPy to
          count tokens per language across a real paragraph and turn the dialect tax into
          arithmetic. And <code>tiktoken</code> — OpenAI&apos;s actual GPT-4 tokenizer — to
          measure what you&apos;ll actually be billed for.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · leading_space_demo.py"
        output={`"hello"  → token id 5
" hello" → token id 9
These are different rows in the embedding table.`}
      >{`# a toy vocabulary that models the leading-space behavior
# every word exists twice: with and without the preceding space
vocab = {
    "hello":   5,    " hello":   9,     # same letters, different IDs
    "world":   6,    " world":  10,
    "the":     7,    " the":    11,
    "<unk>":   0,
}

def tokenize(text):
    # walk left-to-right, greedy-matching the longest key that fits
    out, i = [], 0
    while i < len(text):
        for length in range(min(8, len(text) - i), 0, -1):
            piece = text[i : i + length]
            if piece in vocab:
                out.append(vocab[piece])
                i += length
                break
        else:
            out.append(vocab["<unk>"]); i += 1
    return out

print('"hello"  → token id', tokenize("hello")[0])
print('" hello" → token id', tokenize(" hello")[0])
print("These are different rows in the embedding table.")`}</CodeBlock>

      <Prose>
        <p>
          Vectorise it. Count tokens across a multi-language dataset and compute the
          cost-ratio we promised you — the Babel tax, in a single division. Here&apos;s a
          minimal NumPy-y version against a tiny hand-written tokenizer so you can see the
          arithmetic without the BPE table getting in the way.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy · multilingual_ratio.py"
        output={`english  : 47 chars →  11 tokens   ratio 0.23
japanese : 23 chars →  22 tokens   ratio 0.96
--------------------------------------------------
japanese costs 2.00x more tokens for the same meaning.`}
      >{`import numpy as np

# stand-in for a real tokenizer: char-level for non-English,
# word-level for English. Real ratios are similar in spirit.
def fake_tokenize(text, lang):
    if lang == "english":
        return text.split()                    # whole words, 1 token each
    return list(text.replace(" ", ""))          # 1 token per character (ish)

samples = {
    "english":  "the cat sat on the mat in the warm afternoon sun",
    "japanese": "暖かい午後の日差しの中でマットの上に座っている猫",
}

rows = []
for lang, text in samples.items():
    n_chars  = len(text)
    n_tokens = len(fake_tokenize(text, lang))
    rows.append((lang, n_chars, n_tokens, n_tokens / n_chars))

arr = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
for (lang, *_), row in zip(rows, arr):
    print(f"{lang:9s}: {int(row[0]):3d} chars → {int(row[1]):3d} tokens   ratio {row[2]:.2f}")

ratio = arr[1, 1] / arr[0, 1]
print("-" * 50)
print(f"japanese costs {ratio:.2f}x more tokens for the same meaning.")`}</CodeBlock>

      <Prose>
        <p>
          Now the real thing. <code>tiktoken</code> is the actual BPE tokenizer OpenAI ships,
          and <code>cl100k_base</code> is GPT-4&apos;s encoding. Ten lines, and you can measure
          every edge case we&apos;ve talked about — leading spaces, number chunking, the
          non-English tax — against the exact table your API bill is computed from.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — tiktoken · real_tokenizer.py"
        output={`"hello"            → [15339]                              (1 token)
" hello"           → [24748]                              (1 token, different id!)
"The quick brown"  → [791, 4062, 14198]                   (3 tokens)
"1234567890"       → [4513, 10961, 20652]                 (3 tokens)
"123456789"        → [4513, 10961, 2366]                  (3 tokens)
Japanese 12-char   → 28 tokens    (~2.3x vs English)`}
      >{`import tiktoken

enc = tiktoken.get_encoding("cl100k_base")       # the GPT-4 / GPT-3.5 tokenizer

def show(label, s):
    ids = enc.encode(s)
    print(f"{label!r:20s} → {ids!s:40s} ({len(ids)} tokens)".replace(",)", ")"))

show("hello",           "hello")
show(" hello",          " hello")                # leading space ⇒ different token
show("The quick brown", "The quick brown")
show("1234567890",      "1234567890")            # ten digits, 3 tokens
show("123456789",       "123456789")             # nine digits, also 3 tokens

english  = "the cat sat on the mat in the warm afternoon sun"
japanese = "暖かい午後の日差しの中でマットの上に座っている猫"
print(f"Japanese 12-char   → {len(enc.encode(japanese))} tokens    "
      f"(~{len(enc.encode(japanese)) / len(enc.encode(english)):.1f}x vs English)")`}</CodeBlock>

      <Bridge
        label="pure python → numpy → tiktoken"
        rows={[
          {
            left: 'vocab = {"hello": 5, " hello": 9, ...}',
            right: 'tiktoken.get_encoding("cl100k_base")',
            note: 'a toy dict becomes a trained 100k-entry BPE table',
          },
          {
            left: 'tokenize(text)  # longest-match walk',
            right: 'enc.encode(text)  # production BPE, merges in C',
            note: 'same idea — greedy-ish subword matching — but orders of magnitude faster and trained on the open web',
          },
          {
            left: 'n_tokens / n_chars  # per-language ratio',
            right: 'len(enc.encode(text)) / len(text)',
            note: 'the formula that tells you how much more a non-English user pays',
          },
        ]}
      />

      <Callout variant="insight" title="why three layers, for this lesson in particular">
        The toy vocab in layer 1 is the cleanest way to see that <em>the leading space is its
        own token</em> — you can read the dict and the dialect collision is obvious. Layer 2
        makes the Babel tax concrete — ratios are just division. Layer 3 is what you&apos;ll
        actually call in production to estimate cost, budget context length, and debug weird
        model behavior when an edge case slips through. Same mental model, three scales of
        rigor.
      </Callout>

      {/* ── SolidGoldMagikarp / cost callouts ───────────────────── */}
      <Callout variant="warn" title="glitch tokens in production">
        If you&apos;re running a product on top of an LLM and a user pastes an odd-looking
        Unicode string that provokes nonsensical output, your first suspect is a glitch token
        — a dialect the model has genuinely never heard. The mitigation is not fancy: strip or
        replace tokens that decode to obviously-trash byte sequences before they hit the model.
        OpenAI has scrubbed most of the well-known ones from GPT-4, but no tokenizer is clean,
        and yours definitely isn&apos;t if you trained your own BPE.
      </Callout>

      <Callout variant="warn" title="the non-English tax is a product-design issue">
        If your usage-based pricing bills users per token, you are — probably without meaning
        to — charging Japanese users three to five times more than English users for the same
        conversation. That might be defensible (it genuinely costs you more to serve them),
        or it might be a bug the legal team wants to know about. Either way, do the math on
        your traffic before the regulators do.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">&ldquo;1 word ≈ 1 token&rdquo;:</strong>{' '}
          approximately true for short common English, wildly wrong everywhere else.
          Don&apos;t build a token budget from a word count. Use <code>enc.encode</code> and
          count.
        </p>
        <p>
          <strong className="text-term-amber">String equality ≠ token equality:</strong>{' '}
          <code>&quot;hello&quot;.strip() == &quot;hello&quot;</code> is true as a Python
          string; the <em>tokenized</em> versions collide into different IDs. If you&apos;re
          caching on tokenized input, normalize first.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting the end-of-text token:</strong>{' '}
          most models were trained with a special{' '}
          <code>&lt;|endoftext|&gt;</code> token marking the boundary of a document.
          Tokenize your chat history without it, and the model sees one giant document
          instead of a turn-taking dialogue. Behaviors shift subtly.
        </p>
        <p>
          <strong className="text-term-amber">BOS vs no-BOS confusion:</strong> Llama-family
          models prepend a <code>&lt;BOS&gt;</code> token; OpenAI&apos;s BPE doesn&apos;t. If
          you copy a prompt from one ecosystem to another without re-tokenizing, the{' '}
          <em>first</em> position shifts by one and alignment goes bad silently.
        </p>
        <p>
          <strong className="text-term-amber">NFC vs NFD Unicode:</strong>{' '}
          <code>&quot;é&quot;</code> can be encoded as one code point (NFC) or two (NFD: an{' '}
          <code>e</code> plus a combining accent). Copy-pasting from a Mac to a Linux
          terminal can silently switch between them. The tokenizer treats them as different
          strings — another edge case where two things that look identical aren&apos;t.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Measure the non-English tax">
        <p>
          Pick a paragraph of English — say, the first paragraph of a Wikipedia article. Find
          the same paragraph in Japanese (Wikipedia is great for this — articles often
          have translations). Encode both with <code>tiktoken.get_encoding(&quot;cl100k_base&quot;)</code>.
        </p>
        <p className="mt-2">
          Compute <code>tokens_ja / tokens_en</code>. You should see somewhere between{' '}
          <code>2.0</code> and <code>4.5</code> depending on the paragraph. Then do it again
          with Hindi or Arabic. Then one more: try <code>o200k_base</code> (the newer GPT-4o
          tokenizer) and see how much it narrowed the gap.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: write a function <code>estimate_cost(text, lang)</code> that takes a string
          and a language tag and returns a dollar figure, using GPT-4o&apos;s published
          per-token price. Use it to audit your real product&apos;s traffic mix.
        </p>
      </Challenge>

      {/* ── Close + next ────────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> The tokenizer is a frozen, imperfect
          compressor that sits between your text and the model&apos;s integer interface, and
          every edge case is a place where its grammar breaks down. Leading spaces collide
          into different IDs. Numbers chunk weirdly and fool the model&apos;s arithmetic.
          Non-English dialects pay a Babel tax of 2x to 15x. Glitch tokens are real and
          occasionally ruinous. When your LLM-backed product is doing something weird, the
          tokenizer is always your second suspect — right after the prompt itself — and
          sometimes it is genuinely the culprit.
        </p>
        <p>
          <strong>Next up — <code>gpt-data-loader</code>.</strong> Now that we know how
          individual strings tokenize (and misbehave), we can stop talking about one string at a
          time and start talking about <em>datasets</em>. How do you take a folder of text
          files, run them through the tokenizer, chunk the output into training examples of
          fixed context length, and stream it to the GPU without running out of memory?
          That&apos;s the data loader, and it&apos;s the unglamorous plumbing that every real
          training run depends on.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'SolidGoldMagikarp (plus, prompt generation)',
            author: 'Rumbelow, Watkins',
            venue: 'LessWrong, 2023 — discovery of glitch tokens in GPT-2/3',
            url: 'https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation',
          },
          {
            title: 'Language Model Tokenizers Introduce Unfairness Between Languages',
            author: 'Yenai, Petrov et al.',
            year: 2024,
            venue: 'quantifies the tokens-per-character gap across 52 languages',
            url: 'https://arxiv.org/abs/2305.15425',
          },
          {
            title: 'tiktoken — OpenAI\'s fast BPE tokenizer',
            author: 'OpenAI',
            venue: 'cl100k_base (GPT-4) and o200k_base (GPT-4o) encodings',
            url: 'https://github.com/openai/tiktoken',
          },
          {
            title: "Let's build the GPT Tokenizer",
            author: 'Andrej Karpathy',
            venue: 'YouTube, 2024 — 2-hour walkthrough of BPE from scratch',
            url: 'https://www.youtube.com/watch?v=zduSFxRajkE',
          },
        ]}
      />
    </div>
  )
}
