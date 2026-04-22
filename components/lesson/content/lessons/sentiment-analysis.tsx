import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import Prereq from '../Prereq'
import NeedsBackground from '../NeedsBackground'
import {
  Prose, Callout, Personify, Bridge, Gotcha, Challenge, References, KeyTerm, AsciiBlock,
} from '../primitives'
import SentimentClassifier from '../widgets/SentimentClassifier'
import AttentionHighlight from '../widgets/AttentionHighlight'

// Signature anchor: the mood ring. A paragraph of text goes in, a single dial
// spins to positive or negative. Returns at the opening (review → dial), the
// pooling / bag-of-vectors reveal (how a paragraph becomes one dial), and the
// failure-mode section (sarcasm fools the ring). Cliffhanger names
// positional-encoding — "not bad" and "bad not" come out identical when the
// ring averages every word's tint and throws the order away.
export default function SentimentAnalysisLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="sentiment-analysis" />

      <Prose>
        <p>
          Here is the task, stripped down to one sentence: you take a paragraph
          of text, you squeeze it through your model, and a single dial spins
          to positive or negative. That&apos;s it. A 400-word Yelp review
          collapses into a number between 0 and 1. A tweet becomes a thumbs-up
          or a thumbs-down. Think of it as a <strong>mood ring</strong> for
          text — whatever you feed it, it reports back one color, one tint, one
          dial position. No translation, no summarization, no parsing, no
          knowledge base. Just a label. This is{' '}
          <KeyTerm>sentiment analysis</KeyTerm>, and it is the MNIST of NLP:
          small, clean, easy to set up, and still the right first real task to
          train a text model on.
        </p>
        <p>
          The ring shows up everywhere once you start looking for it. Product
          reviews on Amazon, tweets mentioning your airline, customer-support
          tickets, Yelp stars, IMDB ratings, app-store feedback. Sometimes the
          dial snaps to one of two positions (pos/neg). Sometimes it ticks
          through five stars. Sometimes it&apos;s a continuous needle between
          −1 and +1. The core task is the same — map a string of characters to
          a scalar or a small set of classes — and the modeling choices you
          make in this lesson transfer to every other text-classification task
          you will ever build.
        </p>
      </Prose>

      <AsciiBlock caption="the sentiment pipeline, in full">
{`    "This movie was a complete waste of two hours."
                    │
                    ▼  tokenize
    [This, movie, was, a, complete, waste, of, two, hours, .]
                    │
                    ▼  featurize  (bag-of-words · avg embedding · LSTM state)
    fixed-size vector  ∈ ℝᵈ
                    │
                    ▼  Linear(d, 1)  +  sigmoid
    p(positive)  ∈  [0, 1]
                    │
                    ▼  threshold at 0.5
    label: NEGATIVE`}
      </AsciiBlock>

      <Prose>
        <p>
          Read that diagram as the anatomy of the ring. Tokens go in at the
          top, get squeezed into a single fixed-size vector in the middle (no
          matter how long the review), and then a linear head plus a{' '}
          <NeedsBackground slug="sigmoid-and-relu">sigmoid</NeedsBackground>{' '}
          collapses the vector down to one number — the dial. Every sentiment
          model you&apos;ll meet in this lesson is a different answer to one
          question: <em>how do you squeeze the paragraph into the vector?</em>
        </p>
        <p>
          The standard benchmark is the{' '}
          <KeyTerm>IMDB Large Movie Review Dataset</KeyTerm> — 25,000 training
          reviews and 25,000 test reviews, evenly balanced between positive and
          negative, each labeled from the reviewer&apos;s own star rating (≤4 =
          negative, ≥7 = positive, middle scores excluded so the signal is
          clean). Released by Maas et al. in 2011, it has been the workhorse of
          sentiment papers ever since. You can download it in two lines, train
          a model in two more, and have a score you can compare against a
          decade of published numbers. That is rare. Most NLP benchmarks need
          dataset cards, license checks, and preprocessing pipelines; IMDB is
          a folder of text files and a CSV of labels.
        </p>
      </Prose>

      <SentimentClassifier />

      <Prose>
        <p>
          Type a sentence above and watch the dial spin. The mood ring behind
          the widget is a hand-coded toy — a fixed dictionary of positive and
          negative keywords, each with a weight, plus a tiny embedding score
          for words not in the dictionary. It is not trained on IMDB. It will
          get sarcasm wrong, miss negation, and be confused by emoji. But it
          captures the mechanical heart of every sentiment model: every word
          contributes a signed tint, the tints are summed, and the sum is
          squashed into a probability. A bag-of-words logistic regression does
          exactly this, just with weights learned from data rather than
          hand-picked.
        </p>
      </Prose>

      <Personify speaker="Bag of words">
        I throw away word order. I throw away grammar. I throw away everything
        except which words appeared and how often. And yet — on IMDB I score
        around 88% with a bit of tf-idf and a logistic regression on top. I
        am the stubborn baseline everyone tries to beat and most barely
        manage to. Before you reach for a transformer, beat me first. If you
        can&apos;t beat me by three points, your fancy model is probably not
        learning what you think it is.
      </Personify>

      <MathBlock caption="log-linear sentiment scoring — the bag-of-words model in one equation">
{`Let V = {w₁, w₂, ..., w_{|V|}}  be the vocabulary.

For a review with word counts  c_i  (or tf-idf values) for each  wᵢ:

        score(review) =  Σᵢ  βᵢ · cᵢ   +   b

        p(positive | review) =  σ(score)  =  1 / (1 + e^{-score})

Training objective (binary cross-entropy over N reviews):

        L(β, b) = −(1/N) Σⱼ [ yⱼ log pⱼ  +  (1 − yⱼ) log(1 − pⱼ) ]

Each  βᵢ  is a learned weight per word:
    βᵢ > 0  → word wᵢ pushes toward positive  (e.g. "brilliant", "loved")
    βᵢ < 0  → word wᵢ pushes toward negative  (e.g. "boring", "waste")
    βᵢ ≈ 0  → word wᵢ is neutral               (e.g. "the", "of", "movie")`}
      </MathBlock>

      <Prose>
        <p>
          Look at the math as the ring&apos;s wiring diagram. Each word carries
          a tiny tint — <code>βᵢ</code>, positive or negative. To score a
          review, you add up every word&apos;s tint, pass the sum through a
          sigmoid, and the dial lands somewhere between 0 and 1. That&apos;s
          how a paragraph becomes one dial: you average its mood. <code>|V|</code>{' '}
          weights, one bias, a sigmoid,{' '}
          <NeedsBackground slug="cross-entropy-loss">cross-entropy</NeedsBackground>.
          You can fit it with scikit-learn in five lines. It runs in
          milliseconds. And until 2013 — when Socher&apos;s recursive models
          and later LSTMs started to matter — this was the state of the art on
          most sentiment benchmarks. When you hear someone say &ldquo;simple
          baselines are shockingly hard to beat,&rdquo; they are usually
          talking about this.
        </p>
      </Prose>

      <AttentionHighlight />

      <Prose>
        <p>
          Same sentence, new view. The widget above highlights which words
          contributed most to the prediction — brighter words pushed the ring
          harder. This is not real attention in the transformer sense
          (that&apos;s a later lesson). It is a visualization of{' '}
          <code>|βᵢ · cᵢ|</code> per word: each token&apos;s signed
          contribution to the score. In a linear model this is exact and
          interpretable — every prediction decomposes cleanly into a sum of
          per-word pushes, and you can always point to the specific words that
          swung the dial. This is the one real advantage a bag-of-words model
          has over an LSTM or a transformer: its reasoning is fully
          transparent.
        </p>
        <p>
          Try typing <em>&ldquo;not bad at all&rdquo;</em> and watch the mood
          ring happily light up <em>bad</em> as a negative tint and call the
          review negative. That is the bag-of-words failure mode in miniature.
          The ring averaged every word&apos;s color and never noticed the{' '}
          <em>not</em> sitting one slot to the left. Word order matters —
          sometimes enormously — and any model that throws it away will get
          these cases wrong forever.
        </p>
      </Prose>

      <Personify speaker="LSTM">
        I read your sentence left-to-right, one token at a time, carrying a hidden state
        that remembers what came before. When I reach <em>bad</em> and my state already
        contains <em>not</em>, I can flip the sign. When I reach <em>hours</em> after{' '}
        <em>waste of two</em>, I know to treat it as a complaint, not a measurement. I
        am slower, harder to train, and harder to interpret than the bag-of-words
        baseline — but I finally understand that <em>not bad</em> means good. On IMDB
        that context-awareness buys me two or three points. On harder datasets it buys
        me much more.
      </Personify>

      <Prose>
        <p>
          Three layers of code, each a different way to build the mood ring.
          Pure Python for the bag-of-words baseline with hand-picked words (to
          build intuition), NumPy with tf-idf + logistic regression (the real
          classical baseline), and finally a PyTorch LSTM with a trained
          embedding layer — the neural upgrade that stops averaging{' '}
          <NeedsBackground slug="word-embeddings">word embeddings</NeedsBackground>{' '}
          blindly and starts reading them in order.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="layer 1 — pure python · sentiment_baseline.py (handcrafted keyword lists)"
        output={`review: "This movie was absolutely brilliant, I loved every minute!"
  score = +3.0  →  POSITIVE  (p = 0.95)
review: "What a complete waste of two hours. Boring and awful."
  score = −4.0  →  NEGATIVE  (p = 0.02)
review: "It was okay I guess, not bad."
  score = −1.0  →  NEGATIVE  (p = 0.27)    ← wrong! "not bad" = positive`}
      >{`import math, re

POSITIVE_WORDS = {
    "brilliant": 2.0, "excellent": 2.0, "loved": 1.5, "amazing": 2.0,
    "great": 1.2, "good": 0.8, "enjoyable": 1.2, "masterpiece": 2.5,
    "wonderful": 1.5, "fantastic": 1.8, "best": 1.5, "perfect": 2.0,
}
NEGATIVE_WORDS = {
    "awful": -2.0, "terrible": -2.0, "boring": -1.5, "waste": -2.0,
    "bad": -1.0, "worst": -2.0, "hated": -1.8, "disappointing": -1.5,
    "dull": -1.2, "horrible": -2.0, "mediocre": -0.8, "poorly": -1.0,
}
BIAS = 0.0

def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())

def score(text):
    s = BIAS
    for tok in tokenize(text):
        s += POSITIVE_WORDS.get(tok, 0.0) + NEGATIVE_WORDS.get(tok, 0.0)
    return s

def predict(text):
    s = score(text)
    p = 1 / (1 + math.exp(-s))
    return ("POSITIVE" if p >= 0.5 else "NEGATIVE"), p

for review in [
    "This movie was absolutely brilliant, I loved every minute!",
    "What a complete waste of two hours. Boring and awful.",
    "It was okay I guess, not bad.",
]:
    label, p = predict(review)
    print(f"{label}  p={p:.2f}  |  {review}")`}</CodeBlock>

      <CodeBlock
        language="python"
        caption="layer 2 — numpy / scikit-learn · sentiment_tfidf.py (the real classical baseline)"
        output={`loaded 25000 train, 25000 test reviews
fitting tf-idf vectorizer over 50000 reviews...
vocabulary size: 74849
fitting logistic regression...
train accuracy: 0.9438
test  accuracy: 0.8832     ← the number to beat
top positive weights:  excellent  +3.81
                       wonderful  +3.14
                       perfect    +2.97
                       loved      +2.71
                       amazing    +2.65
top negative weights:  worst      −4.22
                       awful      −3.61
                       boring     −3.44
                       waste      −3.02
                       terrible   −2.89`}
      >{`import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_files

# Load IMDB (assume aclImdb/train and aclImdb/test downloaded)
train = load_files("aclImdb/train", categories=["pos", "neg"], encoding="utf-8")
test  = load_files("aclImdb/test",  categories=["pos", "neg"], encoding="utf-8")
X_train_text, y_train = train.data, train.target
X_test_text,  y_test  = test.data,  test.target

# Tf-idf: term frequency down-weighted by document frequency.
# Unigrams + bigrams so "not bad" gets its own feature.
vec = TfidfVectorizer(
    lowercase=True, ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True,
)
X_train = vec.fit_transform(X_train_text)
X_test  = vec.transform(X_test_text)

# Logistic regression — the log-linear model from the math block above.
clf = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
clf.fit(X_train, y_train)

print(f"train accuracy: {clf.score(X_train, y_train):.4f}")
print(f"test  accuracy: {clf.score(X_test,  y_test):.4f}")

# Interpret: which words have the most extreme learned weights?
vocab = np.array(vec.get_feature_names_out())
w = clf.coef_[0]
for idx in np.argsort(w)[-5:][::-1]:
    print(f"  +pos  {vocab[idx]:<16s}  {w[idx]:+.2f}")
for idx in np.argsort(w)[:5]:
    print(f"  −neg  {vocab[idx]:<16s}  {w[idx]:+.2f}")`}</CodeBlock>

      <Bridge
        label="pure python → numpy + scikit-learn"
        rows={[
          { left: 'hand-picked words & weights', right: 'TfidfVectorizer + LogisticRegression.fit', note: 'weights learned from 25k labeled reviews, not guessed' },
          { left: 'score = Σ constant weights', right: 'score = w @ x (sparse matrix)', note: 'one matmul over a 75k-dim sparse bag-of-ngrams' },
          { left: 'unigrams only', right: 'ngram_range=(1, 2)', note: 'captures "not bad", "waste of", "must see"' },
          { left: 'raw counts', right: 'tf-idf with sublinear_tf', note: 'down-weights frequent filler words ("the", "movie")' },
        ]}
      />

      <CodeBlock
        language="python"
        caption="layer 3 — pytorch LSTM · sentiment_lstm.py (end-to-end trained embedding)"
        output={`epoch 1  train_loss=0.541  val_acc=0.8342
epoch 2  train_loss=0.312  val_acc=0.8746
epoch 3  train_loss=0.221  val_acc=0.8881
epoch 4  train_loss=0.164  val_acc=0.8934
epoch 5  train_loss=0.122  val_acc=0.8971   ← beats tf-idf by ~1.5 points
test accuracy: 0.8989`}
      >{`import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")

def yield_tokens(iter_):
    for _, text in iter_:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(
    yield_tokens(IMDB(split="train")),
    specials=["<pad>", "<unk>"], min_freq=5,
)
vocab.set_default_index(vocab["<unk>"])

def collate(batch):
    labels, texts, lengths = [], [], []
    for label, text in batch:
        ids = vocab(tokenizer(text))[:400]                    # truncate long reviews
        labels.append(1 if label == "pos" else 0)
        texts.append(torch.tensor(ids))
        lengths.append(len(ids))
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    return texts, torch.tensor(labels, dtype=torch.float), torch.tensor(lengths)

train_loader = DataLoader(list(IMDB(split="train")), batch_size=32, shuffle=True, collate_fn=collate)
test_loader  = DataLoader(list(IMDB(split="test")),  batch_size=64, shuffle=False, collate_fn=collate)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc    = nn.Linear(hidden_dim, 1)
    def forward(self, x, lengths):
        e = self.embed(x)                                     # (B, T, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (h, _) = self.lstm(packed)                         # h: (1, B, H)
        return self.fc(h.squeeze(0)).squeeze(-1)              # logits (B,)

model = SentimentLSTM(len(vocab))
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 6):
    model.train()
    losses = []
    for x, y, L in train_loader:
        optim.zero_grad()
        logits = model(x, L)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward(); optim.step()
        losses.append(loss.item())
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y, L in test_loader:
            preds = (torch.sigmoid(model(x, L)) >= 0.5).float()
            correct += (preds == y).sum().item(); total += len(y)
    print(f"epoch {epoch}  train_loss={sum(losses)/len(losses):.3f}  val_acc={correct/total:.4f}")`}</CodeBlock>

      <Bridge
        label="tf-idf + logistic regression → LSTM + embedding"
        rows={[
          { left: 'sparse bag-of-ngrams vector', right: 'dense sequence of embedding vectors', note: 'word order is preserved — "not bad" and "bad not" look different' },
          { left: 'one learned weight per ngram', right: 'embedding matrix + recurrent weights', note: '~10M parameters vs. 75k; needs more data and more epochs' },
          { left: 'pd.read_csv + TfidfVectorizer', right: 'DataLoader + collate_fn + pad_sequence', note: 'variable-length sequences, padded to the longest in the batch' },
          { left: 'clf.fit(X, y) (seconds)', right: 'full SGD loop with Adam (minutes on CPU, seconds on GPU)', note: 'but every piece is trained end-to-end — including the word vectors' },
        ]}
      />

      <Callout variant="insight" title="the sentiment accuracy ladder on IMDB">
        <strong>Bag of words + logistic regression:</strong> ~87-88% — the
        flat-colored mood ring, fits in seconds.{' '}
        <strong>Averaged word embeddings + linear classifier:</strong> ~88-89%
        — a subtler tint, still linear, still fast (see fastText).{' '}
        <strong>LSTM over embeddings:</strong> ~89-91% — the first jump that
        comes from reading words in order instead of averaging them.{' '}
        <strong>Bidirectional LSTM + attention:</strong> ~92-93% — pre-transformer peak.{' '}
        <strong>Fine-tuned BERT-base:</strong> ~94-95%.{' '}
        <strong>Fine-tuned RoBERTa-large / DeBERTa:</strong> ~96-97% — current
        SOTA. The gap from bag-of-words to BERT is a factor of a thousand in
        parameters for eight points of accuracy. Whether that trade is worth
        it depends on your deployment constraints, not your aesthetics.
      </Callout>

      <Callout variant="note" title="binary, ordinal, or regression?">
        Real sentiment is continuous — a reviewer who gave four stars feels
        differently from one who gave five, even though both would spin the
        dial to &ldquo;positive.&rdquo; Binary classification throws that
        signal away for simplicity; ordinal regression (predict 1-5 stars
        with a rank-aware loss, usually an{' '}
        <NeedsBackground slug="mlp-from-scratch">MLP</NeedsBackground> head
        with a{' '}
        <NeedsBackground slug="softmax">softmax</NeedsBackground> over star
        buckets) or plain regression (predict a real number) both recover it.
        Use binary when you genuinely only care about thumbs-up / thumbs-down;
        use ordinal when the degree matters for downstream decisions (e.g.
        ranking reviews by how scathing they are).
      </Callout>

      <Callout variant="note" title="evaluation: accuracy is not enough">
        IMDB is perfectly balanced (50/50 pos/neg), so accuracy works. In the wild, class
        imbalance is the norm — product reviews skew positive, support tickets skew
        negative, election-eve tweets skew whichever way the day is going. Always report{' '}
        <KeyTerm>precision</KeyTerm>, <KeyTerm>recall</KeyTerm>, and <KeyTerm>F1</KeyTerm>{' '}
        per class, plus the full <KeyTerm>confusion matrix</KeyTerm>. A model with 95%
        accuracy on a 95/5 split has learned to always predict the majority class. A model
        with 80% accuracy and balanced F1 across classes is the one you ship.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Negation handling.</strong>{' '}
          <em>&ldquo;not bad&rdquo;</em> is positive. <em>&ldquo;hardly the worst
          thing I&apos;ve ever seen&rdquo;</em> is mildly positive. The mood
          ring averages every word&apos;s tint independently, so it gets these
          wrong forever. Bigrams help (<code>not_bad</code> as one feature),
          LSTMs do better, transformers do best. If you care about negation, do
          not ship unigram-only logistic regression.
        </p>
        <p>
          <strong className="text-term-amber">Sarcasm fools the ring.</strong>{' '}
          <em>&ldquo;Oh great, another sequel. Just what we needed.&rdquo;</em>{' '}
          Every positive keyword is present; the label is strongly negative.
          The ring spins to &ldquo;positive&rdquo; with full confidence and
          gets it exactly backwards. No current production model handles
          sarcasm reliably — even humans miss it in text without tone cues. If
          you have sarcasm in your data, measure it and set expectations.
        </p>
        <p>
          <strong className="text-term-amber">Emoji and unicode tokenizing.</strong>{' '}
          Regex tokenizers like <code>[a-z]+</code> drop emoji entirely, which is
          catastrophic on tweets where{' '}
          <span role="img" aria-label="emojis">🔥❤️😡</span> carry most of the sentiment.
          Use a tokenizer that keeps emoji as tokens (spaCy, HuggingFace, or a custom
          regex that includes the emoji unicode ranges).
        </p>
        <p>
          <strong className="text-term-amber">Train/test leak via duplicate reviews.</strong>{' '}
          Scraped review datasets often contain near-duplicates across splits — the same
          reviewer posting on multiple sites, bot-generated filler, reposts. A model that
          &ldquo;generalizes&rdquo; to the test set may just be memorizing shared
          duplicates. Always dedupe by text hash (and by paragraph-level overlap for long
          reviews) before trusting a headline accuracy number.
        </p>
        <p>
          <strong className="text-term-amber">Domain shift.</strong>{' '}
          A ring calibrated on IMDB movie reviews will be underwhelming on
          Amazon product reviews, tragic on financial news, and nearly
          useless on medical notes. The vocabulary of sentiment is
          domain-specific — <em>&ldquo;aggressive&rdquo;</em> is negative for
          a movie character and positive for an antibiotic. Retrain,
          fine-tune, or at least reweight your model for the target domain.
        </p>
      </Gotcha>

      <Challenge prompt="LSTM vs. BoW — when do they disagree?">
        <p>
          Train both baselines on IMDB: the tf-idf + logistic regression from layer 2, and
          the LSTM from layer 3. On the 25k test reviews, find the ones where the two
          models disagree — same input, opposite predictions. How many are there? Hand-label
          a sample of 30: which model tends to be right on the disagreement set? Look
          specifically at reviews containing <em>not</em>, <em>but</em>, <em>although</em>, or
          sarcastic praise (<em>&ldquo;such a fantastic waste of time&rdquo;</em>). You
          should find the LSTM wins on negation and contrast, and occasionally loses on
          long reviews where a clear keyword signal gets diluted by the recurrent hidden
          state. Write down the failure modes — you will see them again in every
          sequence model you ever train.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> Sentiment is the hello-world
          of NLP because every part of the pipeline — tokenization,
          featurization, a classifier head, a loss, an evaluation metric — is
          present in its simplest usable form. The mood ring is the right
          mental picture for all of them: a paragraph walks in, a single dial
          walks out. Every text task you tackle next (topic classification,
          intent detection, spam filtering, toxicity scoring) is a variation
          on the same skeleton — turn text into a fixed-size vector, put a
          linear head on top, train with cross-entropy. What changes is how
          you compute that vector, and how much context the computation can
          hold.
        </p>
        <p>
          <strong>Next up — Positional Encoding.</strong> The mood ring
          averaged every word&apos;s color and threw away the order — that&apos;s
          fine for &ldquo;terrible&rdquo; vs &ldquo;great,&rdquo; but{' '}
          <em>&ldquo;not bad&rdquo;</em> and <em>&ldquo;bad not&rdquo;</em>{' '}
          come out identical. LSTMs fixed that by reading the sequence one
          token at a time and carrying a hidden state. That is inherently
          sequential — you cannot parallelize it across time steps, which
          caps how big you can scale. Transformers throw the recurrence away
          and process every token in parallel, which raises the same question
          all over again: how does the model know the order of words when it
          sees them all at once? The answer is{' '}
          <KeyTerm>positional encoding</KeyTerm> — a clever vector you add to
          every embedding that tells the model where in the sequence the
          token sits. That is the bridge to transformers, and it is the next
          lesson.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Learning Word Vectors for Sentiment Analysis',
            author: 'Maas, Daly, Pham, Huang, Ng, Potts',
            venue: 'ACL 2011 — introduced the IMDB dataset',
            year: 2011,
            url: 'https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf',
          },
          {
            title: 'Thumbs up? Sentiment Classification Using Machine Learning Techniques',
            author: 'Pang, Lee, Vaithyanathan',
            venue: 'EMNLP 2002 — the paper that framed sentiment as a ML problem',
            year: 2002,
            url: 'https://www.cs.cornell.edu/home/llee/papers/sentiment.pdf',
          },
          {
            title: 'Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank',
            author: 'Socher, Perelygin, Wu, Chuang, Manning, Ng, Potts',
            venue: 'EMNLP 2013 — SST dataset and the first deep model that beat BoW convincingly',
            year: 2013,
            url: 'https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf',
          },
          {
            title: 'Dive into Deep Learning — 16.2 Sentiment Analysis: Using RNNs',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-rnn.html',
          },
          {
            title: 'Dive into Deep Learning — 16.3 Sentiment Analysis: Using CNNs',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-cnn.html',
          },
          {
            title: 'Bag of Tricks for Efficient Text Classification (fastText)',
            author: 'Joulin, Grave, Bojanowski, Mikolov',
            venue: 'EACL 2017 — averaged embeddings + linear is a remarkably strong baseline',
            year: 2017,
            url: 'https://arxiv.org/abs/1607.01759',
          },
        ]}
      />
    </div>
  )
}
