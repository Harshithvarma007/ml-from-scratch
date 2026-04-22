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
import SFTDataFormatting from '../widgets/SFTDataFormatting'
import SFTLossMasking from '../widgets/SFTLossMasking'

// Signature anchor: the fluent polyglot going to etiquette school. The
// pretrained model already speaks every language on the internet — it just
// has no manners. SFT doesn't teach new words; it teaches when to answer,
// how to end a reply, what "polite exchange" looks like. Returns at the
// opening, at the instruction-tuning reveal, and at the limitation beat
// ("you can't etiquette-school your way to new knowledge").

export default function SupervisedFineTuningLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="supervised-fine-tuning" />

      {/* ── Opening: the fluent polyglot with no manners ────────── */}
      <Prose>
        <p>
          Picture a fluent polyglot with no manners. They speak every language
          on the internet — <NeedsBackground slug="train-your-gpt">pretraining</NeedsBackground>{' '}
          on a trillion tokens will do that — and yet you cannot get through
          dinner with them. Ask a question and they finish the question. Say
          hello and they recite an email template. The grammar is flawless.
          The vocabulary is encyclopedic. What&apos;s missing is school — the
          other kind of school. The kind that teaches which fork to use.
        </p>
        <p>
          Feed a base model the first half of a Wikipedia article and it will
          continue with plausible Wikipedia. Feed it{' '}
          <code>&quot;Write me a haiku about Kubernetes&quot;</code> and you
          will get… another instruction. Because on the internet, that&apos;s
          what usually follows a line like that: a forum post asking the same
          question, a tutorial title, another prompt. Our polyglot has read
          every English sentence ever written and learned <em>English</em>,
          not <em>assistance</em>.
        </p>
        <p>
          <KeyTerm>Supervised fine-tuning</KeyTerm> — SFT — is etiquette
          school for the fluent. You take a few thousand to a million{' '}
          <code>(prompt, response)</code> pairs of polite exchanges written by
          humans (or by a strong model), format them with role markers, and
          train the model to produce the response token-by-token given the
          prompt. The loss looks almost exactly like pretraining&apos;s{' '}
          <NeedsBackground slug="code-gpt">next-token prediction</NeedsBackground>{' '}
          <NeedsBackground slug="cross-entropy-loss">cross-entropy</NeedsBackground>.
          Almost.
        </p>
        <p>
          The one difference is the thing that matters. You don&apos;t train
          it to predict the prompt. You only train it to predict the reply.
          You&apos;re not teaching the polyglot new words — you&apos;re
          teaching them what to <em>say</em> versus what to <em>listen to</em>.
          Everything interesting in this lesson lives inside that sentence.
        </p>
      </Prose>

      <Personify speaker="Base model">
        I was trained on a trillion tokens of internet. I can do your taxes,
        roast your haiku, write a Python REPL. But I will not, unprompted,
        know to answer a question. Someone has to show me the shape of a
        reply. Someone has to school me in when the human has stopped
        talking.
      </Personify>

      {/* ── The dataset shape: one polite exchange ──────────────── */}
      <Prose>
        <p>
          Here&apos;s what a single SFT training example looks like in its
          rawest form — one polite exchange, two turns, a question and the
          shape of a good answer. This is an etiquette flashcard.
        </p>
      </Prose>

      <AsciiBlock caption="one SFT example — InstructGPT-era text format">
{`Human: What's the capital of New Zealand?
Assistant: Wellington — it's on the southern tip of the North Island.

◆ prompt tokens ..... 13    ← the model sees these, does NOT learn them
◆ response tokens ... 18    ← the model sees these AND trains on them
◆ total sequence .... 31    ← what goes into the transformer in one forward pass`}
      </AsciiBlock>

      <Prose>
        <p>
          That&apos;s the whole object. Two roles, a question, an answer, and
          a mental note about which tokens the model is supposed to{' '}
          <em>learn</em> vs. which ones it&apos;s only supposed to{' '}
          <em>read</em>. Every real SFT dataset is a pile of these polite
          exchanges, formatted consistently. The formatting is the
          interesting bit, because the polyglot has to learn where its own
          turn begins — which means there must be an unambiguous marker.
          Think of it as the napkin on the lap: a small ritual that tells
          everyone dinner has started.
        </p>
      </Prose>

      <SFTDataFormatting />

      <Prose>
        <p>
          Toggle between the three formats above. The raw view shows the
          data as-is. The instruction-format view wraps it in Alpaca-style
          headers (<code>### Instruction:</code> / <code>### Response:</code>).
          The chat-template view uses role tokens like{' '}
          <code>&lt;|im_start|&gt;user</code> and <code>&lt;|im_end|&gt;</code>{' '}
          — special single-token markers the tokenizer emits once per turn.
          Different model families use different conventions. Llama-2 has
          its own <code>[INST]</code> tags; ChatML uses{' '}
          <code>&lt;|im_start|&gt;</code>; Vicuna uses bare{' '}
          <code>USER:</code>/<code>ASSISTANT:</code>. None of them are
          &ldquo;right&rdquo; — they&apos;re just the table manners this
          particular finishing school decided to enforce. Pick a convention
          and stick to it; the polyglot only knows the etiquette you teach
          them.
        </p>
      </Prose>

      <Personify speaker="Chat template">
        I am a naming convention dressed up as infrastructure. I decide where
        your turn ends and the model&apos;s begins. Train the model with me
        one way and serve it another, and you will get the most
        confused-sounding assistant of your career. Consistency is the whole
        job.
      </Personify>

      {/* ── Loss masking math — only the reply gets graded ──────── */}
      <Prose>
        <p>
          Now the critical piece. The sequence has 31 tokens. A naive
          next-token loss would train the model to predict <em>all</em> of
          them — including the prompt. That would mean optimizing the
          polyglot to sound like a human <em>asking</em> questions, which is
          the opposite of the goal. You are not running a school for
          interrogators. So we mask.
        </p>
        <p>
          Assign a label vector <code>y</code> the same length as the input.
          For positions inside the prompt, write <code>-100</code> —
          PyTorch&apos;s <code>CrossEntropyLoss</code> treats <code>-100</code>{' '}
          as &ldquo;ignore this position&rdquo; and contributes nothing to
          the gradient. For positions inside the response, write the true
          next-token id. The loss averages only over the response positions:
        </p>
      </Prose>

      <MathBlock caption="SFT loss — only response tokens count">
{`                 N
ℒ_SFT  =   − ─────────    ∑    m_t  ·  log P(x_t | x_<t ; θ)
              ∑_t m_t     t=1

where   m_t = 1   if token t is part of the response
        m_t = 0   if token t is part of the prompt (label = −100)`}
      </MathBlock>

      <Prose>
        <p>
          It&apos;s the usual causal-LM cross-entropy with one extra
          indicator <code>m_t</code>. The denominator — the count of
          unmasked tokens — keeps the loss at the same scale whether your
          prompts are long or short. That matters because if you average
          over <em>all</em> tokens, a conversation with a three-paragraph
          prompt and a one-sentence reply would contribute almost nothing
          to the gradient. Only the polite reply gets graded by the
          etiquette teacher; the guest&apos;s question is read aloud for
          context and then ignored.
        </p>
      </Prose>

      <SFTLossMasking />

      <Prose>
        <p>
          Every token is colored by whether it contributes to the loss.
          Scroll across the sequence: the prompt greys out, the response
          lights up. That mask is the thing separating SFT from raw
          continued pretraining. Take it away and you&apos;re teaching the
          polyglot to sound like whoever wrote the prompt — you are
          fine-tuning on the <em>wrong half</em> of the conversation.
          Etiquette school where the student practices the teacher&apos;s
          lines.
        </p>
      </Prose>

      <Personify speaker="Loss mask">
        I am the difference between &ldquo;train on all of it&rdquo; and
        &ldquo;train on the reply.&rdquo; I am a boolean vector the same
        length as your tokens. I cost nothing to compute. I am why your
        fine-tune sounds like an assistant instead of an echo chamber.
        Forget me and you will not know anything is wrong until the
        evaluations come back strange.
      </Personify>

      <Callout variant="note" title="why -100 specifically">
        Pure implementation detail. PyTorch&apos;s{' '}
        <code>nn.CrossEntropyLoss</code> takes an <code>ignore_index</code>{' '}
        argument that defaults to <code>-100</code>. Any token position with
        that label is excluded from the mean. HuggingFace Transformers&apos;{' '}
        <code>DataCollatorForLanguageModeling</code> and the whole TRL
        library assume the convention. If you&apos;re rolling your own
        collator, use <code>-100</code> — not <code>0</code>, not{' '}
        <code>-1</code> — or you will silently train on the wrong tokens.
      </Callout>

      {/* ── Training setup — a short, cheap finishing course ────── */}
      <Prose>
        <p>
          The optimization is less heroic than you&apos;d think. SFT is a
          weekend finishing course, not four years of language immersion.
          Typical hyperparameters:
        </p>
        <ul>
          <li>
            <strong>Dataset:</strong> 10k–1M examples. Pretraining used
            trillions of tokens — SFT is three to six orders of magnitude
            smaller. You already spoke the language; you just need manners.
          </li>
          <li>
            <strong>Epochs:</strong> 1–3. Go past 3 and the polyglot
            memorizes your exact dinner party lines instead of generalizing
            the etiquette; politeness collapses into parroting.
          </li>
          <li>
            <strong>Learning rate:</strong> <code>2e-5</code> is the
            standard, roughly 10× lower than the end-of-pretraining LR.
            You&apos;re nudging a finished model, not building one.
          </li>
          <li>
            <strong>Optimizer:</strong> AdamW. Warmup ratio ~3%, cosine
            decay to zero.
          </li>
          <li>
            <strong>Batch size:</strong> effective batch ~128 (gradient
            accumulation does most of the work on a single-node setup).
          </li>
          <li>
            <strong>Context length:</strong> long enough to hold the
            longest conversation in your dataset. Padding is wasteful;
            packing multiple short examples into one sequence helps.
          </li>
        </ul>
        <p>
          Compared to pretraining a 7B model (thousands of GPU-years,
          petabytes of text) a reasonable SFT run is <em>hours</em> on a
          single 8×A100 node. That&apos;s most of why SFT caught on as a
          lab technique — anyone with a node can send their fluent
          polyglot to etiquette school over a long weekend, and the impact
          on output quality is dramatic.
        </p>
      </Prose>

      <Callout variant="insight" title="quality beats quantity — the LIMA result">
        Zhou et al. 2023 (&ldquo;LIMA: Less Is More for Alignment&rdquo;)
        fine-tuned Llama-65B on just <em>1,000</em> painstakingly curated
        prompt/response pairs — and matched the instruction-following
        quality of models tuned on 52,000 Alpaca examples or 100,000+
        internal demonstrations. The lesson isn&apos;t &ldquo;use less
        data.&rdquo; It&apos;s that at the SFT stage you are teaching{' '}
        <em>format and style</em>, not facts. A thousand clean examples of
        polite exchange will beat a hundred thousand noisy ones. Etiquette
        school works best with a short book of exquisite manners, not a
        phone directory of mediocre ones. Noisy data — contradictions,
        hallucinated answers, off-topic replies — actively hurts.
      </Callout>

      {/* ── The honest limitation ───────────────────────────────── */}
      <Callout variant="note" title="you cannot etiquette-school your way to new knowledge">
        This is the limitation nobody says out loud. SFT reshapes the
        polyglot&apos;s <em>behavior</em> — tone, turn-taking, answer
        shape. It does not teach them facts they never saw in pretraining.
        Fine-tune a 2021-trained model on 2024 news and it will still
        confidently insist it&apos;s 2021 — politely, beautifully formatted,
        flat wrong. Manners on top of missing vocabulary is still missing
        vocabulary. If you need new knowledge, you need new pretraining or
        retrieval; SFT is finishing school, not a library.
      </Callout>

      {/* ── Catastrophic forgetting ─────────────────────────────── */}
      <Prose>
        <p>
          SFT has one persistent failure mode. You&apos;re updating every
          weight in the network — the same weights that encode everything
          the base model learned during pretraining. If you overfit, or
          train for too long, or train on a narrow dataset, you can{' '}
          <em>erase</em> pretrained capabilities. Your chat-tuned model
          forgets how to do arithmetic. Your code-tuned model forgets
          French. Etiquette school, pushed too hard, can make the polyglot
          forget how to speak. This is <KeyTerm>catastrophic forgetting</KeyTerm>{' '}
          and it is embarrassingly easy to induce.
        </p>
        <p>The three standard defenses:</p>
        <ul>
          <li>
            <strong>Mix in pretraining data.</strong> Every batch: some
            SFT, some unchanged pretraining text. Keeps the distribution
            anchored.
          </li>
          <li>
            <strong>Low learning rate, few epochs.</strong> The numbers
            above (<code>2e-5</code>, 1–3 epochs) exist to prevent this.
          </li>
          <li>
            <strong>Parameter-efficient fine-tuning.</strong> Freeze the
            base weights entirely; only train a small set of adapters
            (LoRA, QLoRA). This is the next lesson and it is the standard
            modern move.
          </li>
        </ul>
      </Prose>

      {/* ── Three-layer code ────────────────────────────────────── */}
      <Prose>
        <p>
          Three layers, one job: take a <code>(prompt, response)</code>{' '}
          pair, turn it into a training batch with the right mask, and get
          a loss. Pure Python does the formatting. NumPy does the
          tokenization and the mask. PyTorch + HuggingFace + TRL does the
          full training loop — because in production you will not write
          any of this by hand.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · format_sft.py"
        output={`--- formatted ---
### Instruction:
What's the capital of New Zealand?

### Response:
Wellington — it's on the southern tip of the North Island.

prompt_len = 45  |  response_len = 57`}
      >{`# One example. No tokenizer, no tensors — just the text contract.

def format_alpaca(prompt: str, response: str) -> dict:
    formatted = (
        f"### Instruction:\\n{prompt}\\n\\n"
        f"### Response:\\n{response}"
    )
    # Track where the response starts, so downstream we know what to mask.
    prompt_part = f"### Instruction:\\n{prompt}\\n\\n### Response:\\n"
    return {
        "text": formatted,
        "prompt_len": len(prompt_part),
        "response_len": len(response),
    }

ex = format_alpaca(
    "What's the capital of New Zealand?",
    "Wellington \u2014 it's on the southern tip of the North Island.",
)
print("--- formatted ---")
print(ex["text"])
print(f"\\nprompt_len = {ex['prompt_len']}  |  response_len = {ex['response_len']}")`}</CodeBlock>

      <Prose>
        <p>
          Move to NumPy. Tokenize once, build the labels vector with{' '}
          <code>-100</code> in the prompt region, and <em>that</em> is the
          object your trainer wants.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy + tokenizer · mask_sft.py"
        output={`input_ids shape: (31,)
labels   shape: (31,)
prompt mask positions: 13   response mask positions: 18
first 5 labels: [-100 -100 -100 -100 -100]
last  5 labels: [  286  6255  5373 29889     2]`}
      >{`import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

prompt   = "Human: What's the capital of New Zealand?\\nAssistant: "
response = "Wellington \u2014 it's on the southern tip of the North Island."

# Tokenize the two halves separately so we know their lengths exactly.
prompt_ids   = tok(prompt, add_special_tokens=False)["input_ids"]
response_ids = tok(response + tok.eos_token, add_special_tokens=False)["input_ids"]

input_ids = np.array(prompt_ids + response_ids, dtype=np.int64)

# Labels are a copy of input_ids with prompt positions zeroed out via -100.
labels = input_ids.copy()
labels[: len(prompt_ids)] = -100                 # ignore prompt in the loss

print("input_ids shape:", input_ids.shape)
print("labels   shape:", labels.shape)
print(f"prompt mask positions: {len(prompt_ids)}   "
      f"response mask positions: {len(response_ids)}")
print("first 5 labels:", labels[:5])
print("last  5 labels:", labels[-5:])`}</CodeBlock>

      <Bridge
        label="python text → numpy ids + mask"
        rows={[
          {
            left: 'formatted = f"...{prompt}...{response}"',
            right: 'input_ids = tok(prompt_part) + tok(response_part)',
            note: 'two tokenizer calls so you know the prompt boundary',
          },
          {
            left: 'prompt_len (chars)',
            right: 'len(prompt_ids)  # tokens',
            note: 'tokens, not characters — that is the unit the loss operates on',
          },
          {
            left: 'conceptually: "ignore the prompt"',
            right: 'labels[:len(prompt_ids)] = -100',
            note: 'the entire trick in one line',
          },
        ]}
      />

      <Prose>
        <p>
          Layer 3 — the thing you actually run. HuggingFace{' '}
          <code>transformers</code> ships the model, the tokenizer, and the
          chat template. TRL&apos;s <code>SFTTrainer</code> wraps the whole
          masking / packing / training loop. A full SFT run in about 30
          lines.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch + transformers + trl · sft_train.py"
      >{`import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig

MODEL = "meta-llama/Llama-2-7b-hf"

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token                  # Llama-2 ships without a pad token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Alpaca — 52k (prompt, response) pairs. Clean, small, battle-tested.
ds = load_dataset("tatsu-lab/alpaca", split="train")

def formatting_func(row):
    # TRL uses the chat template registered on the tokenizer.
    msgs = [
        {"role": "user",      "content": row["instruction"]},
        {"role": "assistant", "content": row["output"]},
    ]
    return tok.apply_chat_template(msgs, tokenize=False)

cfg = SFTConfig(
    output_dir          = "sft-llama2-alpaca",
    num_train_epochs    = 3,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 32,          # → effective batch 128
    learning_rate       = 2e-5,
    lr_scheduler_type   = "cosine",
    warmup_ratio        = 0.03,
    bf16                = True,
    logging_steps       = 20,
    save_steps          = 500,
    max_seq_length      = 2048,
    packing             = True,                # concatenate short examples → fewer wasted tokens
)

trainer = SFTTrainer(
    model            = model,
    tokenizer        = tok,
    train_dataset    = ds,
    formatting_func  = formatting_func,
    args             = cfg,
)
trainer.train()                                # loss masking, chat template, everything — handled
`}</CodeBlock>

      <Bridge
        label="numpy → pytorch + trl"
        rows={[
          {
            left: 'labels[:prompt_len] = -100',
            right: 'SFTTrainer(formatting_func=...)',
            note: 'TRL computes the response-only mask for you from the chat template',
          },
          {
            left: 'for loop over (ids, labels)',
            right: 'trainer.train()',
            note: 'packing, padding, collation, AdamW, cosine LR, logging — one call',
          },
          {
            left: 'raw text concat',
            right: 'tok.apply_chat_template(msgs, tokenize=False)',
            note: 'the canonical way to emit the exact format the model was trained on',
          },
        ]}
      />

      <Callout variant="insight" title="why the three layers, again">
        Pure Python shows what &ldquo;one training example&rdquo; actually
        is — two pieces of text and a note about where one ends. NumPy
        shows the one line that does all the work:{' '}
        <code>labels[:len(prompt_ids)] = -100</code>. And TRL is what
        you&apos;d ship, because nobody writes their own collator in
        production — not because it&apos;s hard, but because the
        library-written one is battle-tested on a hundred edge cases yours
        hasn&apos;t seen yet.
      </Callout>

      {/* ── Gotchas ─────────────────────────────────────────────── */}
      <Gotcha>
        <p>
          <strong className="text-term-amber">Template mismatch train → infer:</strong> the
          most common SFT bug. You train with ChatML (<code>&lt;|im_start|&gt;user</code>) and
          serve with bare <code>USER:</code> prefixes, or vice-versa. The model never sees the
          start-of-turn marker it was trained on; it treats your prompt as mid-conversation
          text; the output is weird. <em>Always</em> use{' '}
          <code>tokenizer.apply_chat_template</code> on both sides.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting the mask:</strong> if you just feed{' '}
          <code>input_ids</code> as both input and labels, you&apos;re training the model to
          predict its own prompt. It will still &ldquo;work&rdquo; — loss goes down, eval
          plausibly improves — but the gradient signal is diluted and the model learns to
          imitate users as much as assist them.
        </p>
        <p>
          <strong className="text-term-amber">Overtraining:</strong> 10 epochs on your SFT set
          will give you a model that quotes its training data verbatim, has lost arithmetic,
          and speaks only in the style of your annotators. 1–3 epochs, low LR, cosine decay.
          Resist the instinct to train until loss stops going down — on SFT, you want to stop
          well before it plateaus.
        </p>
        <p>
          <strong className="text-term-amber">Tokenizing the response without EOS:</strong> if
          you don&apos;t append <code>tok.eos_token</code> to the response before tokenizing,
          the model never learns when to stop generating. At inference it will keep going past
          the answer, hallucinate a new question, and answer that too. One extra token in the
          dataset saves a thousand confused user-reports.
        </p>
      </Gotcha>

      {/* ── Challenge ───────────────────────────────────────────── */}
      <Challenge prompt="Fine-tune Llama-2-7B on 1k Alpaca examples and A/B against the base model">
        <p>
          Sample 1,000 rows from <code>tatsu-lab/alpaca</code> (random seed, stratified by
          instruction length if you want to be fancy). Run the layer-3 script above with{' '}
          <code>num_train_epochs=3</code> and <code>learning_rate=2e-5</code>. On a single 8×A100
          node this takes under two hours with QLoRA, roughly a day with full fine-tuning.
        </p>
        <p className="mt-2">
          Hold out 20 instructions your training data never touched. Generate a response from
          both the base Llama-2-7B and your SFT model with the same decoding settings
          (<code>temperature=0.7</code>, <code>top_p=0.9</code>, <code>max_new_tokens=256</code>).
          Read them side by side.
        </p>
        <p className="mt-2 text-dark-text-muted">
          What you should see: the base model completes the instruction as if it were forum
          text — often echoing the question, often trailing into tangents. The SFT model
          answers directly. Neither is smarter than the other in any deep sense; they&apos;ve
          just been pointed at different distributions. That&apos;s the whole thing SFT does.
        </p>
      </Challenge>

      {/* ── Closing + teaser ────────────────────────────────────── */}
      <Prose>
        <p>
          <strong>What to carry forward.</strong> SFT is etiquette school
          for a fluent polyglot — the first step that turns a base model
          into something you can actually talk to. The mechanics are a
          standard causal-LM loss with one indicator variable (the loss
          mask) that says &ldquo;only grade the reply.&rdquo; The dataset
          is tiny compared to pretraining, the training run is short, and
          the leverage is enormous. The failure modes are mostly about{' '}
          <em>consistency</em>: chat templates have to match between train
          and serve, EOS has to be where it belongs, and you have to
          resist training too long. And the hard ceiling is still the one
          from the opening — you cannot etiquette-school your way into
          facts the polyglot never learned.
        </p>
        <p>
          <strong>Next up — LoRA.</strong> The 30-line script above
          updates all 7 billion weights of Llama-2. That&apos;s expensive
          to train, expensive to store (one 14GB checkpoint per task), and
          the easiest way in the world to induce catastrophic forgetting.{' '}
          <KeyTerm>LoRA</KeyTerm> — Low-Rank Adaptation — changes the
          contract: freeze the base polyglot, train <em>0.1%</em> of new
          parameters as a stack of sticky notes over the weights, keep one
          base model and a dozen tiny adapters around. It is the single
          most important modern fine-tuning technique and it&apos;s a
          surprisingly small amount of linear algebra.
        </p>
      </Prose>

      <References
        items={[
          {
            title:
              'Training language models to follow instructions with human feedback (InstructGPT)',
            author: 'Ouyang et al.',
            venue: 'NeurIPS 2022',
            url: 'https://arxiv.org/abs/2203.02155',
          },
          {
            title: 'LIMA: Less Is More for Alignment',
            author: 'Zhou et al.',
            venue: 'NeurIPS 2023',
            url: 'https://arxiv.org/abs/2305.11206',
          },
          {
            title: 'Llama 2: Open Foundation and Fine-Tuned Chat Models',
            author: 'Touvron et al.',
            venue: 'Meta AI, 2023',
            url: 'https://arxiv.org/abs/2307.09288',
          },
          {
            title: 'TRL — Transformer Reinforcement Learning',
            author: 'HuggingFace',
            venue: 'library, SFTTrainer reference',
            url: 'https://huggingface.co/docs/trl/sft_trainer',
          },
          {
            title: 'Stanford Alpaca: An Instruction-following LLaMA model',
            author: 'Taori et al.',
            venue: '2023 — the 52k-example dataset most SFT tutorials use',
            url: 'https://github.com/tatsu-lab/stanford_alpaca',
          },
        ]}
      />
    </div>
  )
}
