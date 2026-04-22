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
import TensorPlayground from '../widgets/TensorPlayground'
import AutogradTrace from '../widgets/AutogradTrace'
import TrainingLoopStepper from '../widgets/TrainingLoopStepper'

// Signature anchor: the scribe. PyTorch's autograd is a scribe sitting next
// to you — silently recording every op you perform on a tensor during the
// forward pass and handing you the gradients when you call .backward().
// Threaded at three load-bearing moments: the opening (hand-rolled backprop
// vs. the scribe), the autograd reveal (the ledger walked in reverse), and
// the nn.Module/Optimizer beat (compound entries + the updater). Device is
// where the scribe sits. Tensors are the ink that leaves a faint record.
export default function PyTorchBasicsLesson() {
  return (
    <div className="space-y-6">
      {/* ── Prerequisite callout (entry point: pytorch section) ─── */}
      <Prereq currentSlug="pytorch-basics" />

      {/* ── Opening: the scribe ────────────────────────────────── */}
      <Prose>
        <p>
          You&apos;ve already done the hard version. You wrote a{' '}
          <NeedsBackground slug="mlp-from-scratch">multi-layer perceptron</NeedsBackground>{' '}
          in pure NumPy, derived the{' '}
          <NeedsBackground slug="backpropagation">backward pass</NeedsBackground>{' '}
          by hand, then stacked layers until the{' '}
          <NeedsBackground slug="multi-layer-backpropagation">chain rule through an arbitrarily deep net</NeedsBackground>{' '}
          turned into a bookkeeping nightmare. Every new layer meant a new
          Jacobian to trace by hand. The final 40% of the code was index
          arithmetic with no mathematical content whatsoever.
        </p>
        <p>
          Now meet the scribe.
        </p>
        <p>
          PyTorch ships a scribe that sits next to you while you compute. Every
          operation you perform on a tensor during the forward pass — a matmul,
          an add, a squared error — the scribe quietly writes down in a ledger.
          When you call <code>.backward()</code>, the scribe flips the ledger
          over and walks it in reverse, handing you every gradient you need.
          You never write the backward pass again. You write the forward pass
          like you&apos;re computing something ordinary, and the scribe
          watches.
        </p>
        <p>
          That&apos;s the library. Three moving parts sit around the scribe:
          the <KeyTerm>tensor</KeyTerm> (special ink that leaves a faint
          record), <KeyTerm>autograd</KeyTerm> (the scribe itself — the ledger
          and the reverse walk), and the <KeyTerm>module</KeyTerm> system
          (pre-fabricated chunks of operations the scribe logs as one entry).
          Then an <em>optimizer</em> — the person who reads the gradients and
          turns the dials. Understand those four and you understand 95% of
          what you&apos;ll ever need from any deep-learning framework.
        </p>
      </Prose>

      {/* ── Tensors ─────────────────────────────────────────────── */}
      <Prose>
        <p>
          Start with the ink. A tensor is an n-dimensional array — a
          generalization of a scalar (0-D), vector (1-D), matrix (2-D) into
          arbitrarily many dimensions. Every piece of data in a PyTorch
          program — input images, model weights, loss values, gradients — is
          a tensor. The API is almost identical to NumPy, with four
          additions that matter:
        </p>
        <ul>
          <li>
            <strong>device</strong> — a tensor can live on CPU or GPU. Move it
            with <code>.to(&quot;cuda&quot;)</code>. This is where the scribe
            sits; keep all your tensors on the same device or the scribe has
            to get up and walk.
          </li>
          <li>
            <strong>dtype</strong> — integer, float32, float16, bfloat16, int8
            for quantization. Picking the right dtype can halve your memory
            and double your throughput.
          </li>
          <li>
            <strong>requires_grad</strong> — flip this to <code>True</code>{' '}
            and the scribe starts recording every op you apply. Flip it off
            and the ink is plain.
          </li>
          <li>
            <strong>grad_fn</strong> — the entry in the ledger. An opaque
            handle pointing back to the op that produced this tensor, used
            during the reverse walk.
          </li>
        </ul>
        <p>
          Reading <em>shapes</em> first is the single most valuable debugging
          skill in this business. Cycle through the ops below and notice how
          the output shape falls out of the input shapes before any
          arithmetic happens.
        </p>
      </Prose>

      <TensorPlayground />

      <Callout variant="note" title="if you know numpy, you already know 80% of this">
        <code>a + b</code>, <code>a.reshape(2, 3)</code>, <code>a.sum(axis=0)</code>,{' '}
        <code>a @ b</code> — all identical in PyTorch and NumPy. The gap is the four things
        above (<em>device, dtype, requires_grad, grad_fn</em>) plus slightly different
        function names here and there (<code>torch.cat</code> vs <code>np.concatenate</code>,
        etc.). The mental model is the same.
      </Callout>

      <Personify speaker="Tensor">
        I am an n-dimensional array with an attitude. I know which device I live on (<em>CPU</em> by
        default, <em>CUDA</em> if you ask), what dtype I am (<em>float32</em> by default, half or
        bfloat if memory&apos;s tight), and whether I am a leaf or the product of an op. If you
        flip <code>requires_grad=True</code> on me, I will also silently record a trail of every
        op you apply — so that later, when you call <code>backward()</code>, I can retrace my
        steps.
      </Personify>

      {/* ── Autograd ────────────────────────────────────────────── */}
      <Prose>
        <p>
          That trail is the ledger — in the documentation it&apos;s called the{' '}
          <strong>computation graph</strong>. Every time you apply an op to a
          tensor with <code>requires_grad=True</code>, PyTorch allocates a new
          tensor for the result and attaches a <code>grad_fn</code> — the
          scribe&apos;s entry for that op, carrying the exact local derivative.
          Stacking ops stacks grad_fns. One forward pass, one ledger, page by
          page.
        </p>
        <p>
          Then you call <code>loss.backward()</code>. The scribe walks the
          ledger in reverse — chain rule, applied to every parameter in the
          graph, in linear time. This is the same{' '}
          <NeedsBackground slug="multi-layer-backpropagation">multi-layer backprop</NeedsBackground>{' '}
          algorithm you wrote by hand last section, with two differences: the
          scribe did the bookkeeping, and the scribe doesn&apos;t get tired.
        </p>
        <p>
          Pick an expression below. Step through the forward pass — each node
          in the graph fills in with its numeric value as the scribe takes
          dictation. Then step through the backward pass — each node gets a
          pink <code>∂L</code> bubble as the scribe hands back the gradient.
          Your job is to write the forward expression. The scribe writes the
          backward.
        </p>
      </Prose>

      <AutogradTrace />

      <Callout variant="insight" title="what .backward() actually does">
        It starts at the loss tensor, sets its gradient to 1.0 (implicitly), then calls each
        upstream node&apos;s pre-registered <code>backward</code> function, passing along the
        gradient from the node below. Matrix multiply&apos;s backward is a transposed matmul.
        ReLU&apos;s backward is a mask. Sigmoid&apos;s backward is <code>y(1-y) · grad_out</code>.
        It&apos;s a library of hand-written local derivatives, assembled by the graph
        structure. The whole thing is a few thousand lines of C++/CUDA — you could read it in
        a long afternoon.
      </Callout>

      <Gotcha>
        <p>
          <strong className="text-term-amber">
            <code className="text-dark-text-primary">.backward()</code> can only be called once.
          </strong>{' '}
          By default, autograd frees the computation graph after one backward pass to save
          memory. Calling backward again crashes. If you need to differentiate through the same
          graph twice, pass <code className="text-dark-text-primary">retain_graph=True</code> —
          but 99% of the time you actually want a fresh forward pass.
        </p>
        <p>
          <strong className="text-term-amber">
            Inplace ops break autograd.
          </strong>{' '}
          <code className="text-dark-text-primary">x.add_(1)</code> (with the underscore)
          modifies the tensor in place. If that tensor was part of a computation graph,
          autograd may refuse to compute gradients or silently give wrong ones. Prefer{' '}
          <code className="text-dark-text-primary">x = x + 1</code> unless you know why you
          want inplace.
        </p>
        <p>
          <strong className="text-term-amber">
            <code className="text-dark-text-primary">.detach()</code> breaks the graph.
          </strong>{' '}
          Use it deliberately when you want to stop gradient flow: moving averages, reward
          baselines, fixed targets. The common bug is calling <code>.detach()</code> where you
          didn&apos;t mean to — suddenly your upstream weights are not getting gradients.
        </p>
        <p>
          <strong className="text-term-amber">
            Wrap eval-mode code in <code className="text-dark-text-primary">with torch.no_grad():</code>
          </strong>{' '}
          to skip graph construction entirely — roughly 2× faster inference and you won&apos;t
          accidentally backprop through your test set.
        </p>
      </Gotcha>

      {/* ── Modules + training loop ─────────────────────────────── */}
      <Prose>
        <p>
          The scribe will log every little op you write — but real networks
          have thousands of ops, and you don&apos;t want to re-assemble a
          linear layer out of a matmul and an add every time. Enter{' '}
          <code>nn.Module</code>: a pre-fabricated chunk of operations the
          scribe logs as one entry. A container that owns tensors, knows which
          ones are learnable parameters, and defines a <code>forward</code>{' '}
          method. You compose them by nesting. A <code>Linear</code> is a
          Module. A <code>Sequential</code> of Linears is a Module. A custom
          class you wrote with five sub-modules is also a Module. Every one
          offers <code>.parameters()</code>, <code>.to(device)</code>,{' '}
          <code>.state_dict()</code>, and <code>.train()/.eval()</code> — the
          standard API.
        </p>
        <p>
          The optimizer is the third character. The scribe hands you
          gradients; the optimizer reads them and adjusts the weights. A
          single <code>torch.optim.SGD(model.parameters(), lr=0.1)</code> is
          the packaged version of the update rule from{' '}
          <NeedsBackground slug="gradient-descent">gradient descent</NeedsBackground>.
          Adam is the same loop with a little more bookkeeping. Either way,
          three roles: scribe records, scribe walks backward, optimizer steps.
        </p>
        <p>
          Put them together and every PyTorch training script in existence
          reduces to this five-line dance:
        </p>
      </Prose>

      <MathBlock caption="the 5-line training loop — this never changes">
{`optimizer.zero_grad()         # clear the .grad buffer from last step
yhat = model(x)               # forward: build the graph, get predictions
loss = criterion(yhat, y)     # scalar loss — graph extends to this node
loss.backward()               # chain-rule walk — fills every .grad
optimizer.step()              # θ ← θ − α · ∇L   for every parameter`}
      </MathBlock>

      <Prose>
        <p>
          Click through it line by line below. Watch each piece of tensor
          state change as the line executes. This is the same loop you&apos;ll
          use to train GPT-5. What varies across projects is the model, the
          data, the loss, the optimizer — the loop itself is invariant.
        </p>
      </Prose>

      <TrainingLoopStepper />

      <Callout variant="insight" title="the five lines in order, in plain English">
        &ldquo;Forget the old gradients. Run the model on the current batch. Measure how
        wrong we are. Tell me which direction to nudge every parameter. Nudge them.&rdquo;
        That&apos;s the entire modern ML training pipeline — everything else (schedulers,
        checkpoints, mixed precision, distributed training) is plumbing around this loop.
      </Callout>

      {/* ── Three-layer code (a single concrete example) ───────── */}
      <Prose>
        <p>
          A complete training script, start to finish. One-variable linear
          regression trained by backprop. The number of lines of boilerplate
          is genuinely minimal — and every line you write is a line the
          scribe is watching.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="minimal pytorch training script · linreg.py"
        output={`step   0: loss = 8.1326
step 100: loss = 0.0123
step 200: loss = 0.0019
learned slope: 1.9912  intercept: 0.0451`}
      >{`import torch
import torch.nn as nn

# 1. Data — two vectors with a known linear relationship
torch.manual_seed(0)
x = torch.linspace(-2, 2, 100).unsqueeze(-1)          # (100, 1)
y = 2 * x + 0.05 * torch.randn_like(x)                # y ≈ 2x + noise

# 2. Model — one neuron, one weight, one bias
model = nn.Linear(in_features=1, out_features=1)

# 3. Loss + optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. The five-line loop, repeated.
for step in range(250):
    optimizer.zero_grad()
    yhat = model(x)
    loss = criterion(yhat, y)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"step {step:3d}: loss = {loss.item():.4f}")

print(f"learned slope: {model.weight.item():.4f}  "
      f"intercept: {model.bias.item():.4f}")`}</CodeBlock>

      <Bridge
        label="the anatomy, annotated"
        rows={[
          {
            left: 'nn.Linear(1, 1)',
            right: 'weight + bias, both requires_grad=True',
            note: 'Module handles parameter registration automatically',
          },
          {
            left: 'model.parameters()',
            right: 'a generator over all learnable tensors',
            note: 'what the optimizer iterates over',
          },
          {
            left: 'loss.backward()',
            right: 'fills weight.grad and bias.grad',
            note: 'autograd walks the graph — same recurrence as last section',
          },
          {
            left: 'optimizer.step()',
            right: 'weight -= lr * weight.grad ; bias -= lr * bias.grad',
            note: 'SGD in four lines; Adam is eight; the loop code is the same',
          },
        ]}
      />

      {/* ── GPU, mixed precision, the production additions ─────── */}
      <Prose>
        <p>
          Two short additions turn the above into a production script. Move
          the scribe to a GPU:
        </p>
      </Prose>

      <CodeBlock language="python" caption="same loop, on GPU">{`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = x.to(device)
y = y.to(device)
model = model.to(device)

# The training loop is unchanged — tensors already know which device they live on.
for step in range(250):
    optimizer.zero_grad()
    yhat = model(x)
    loss = criterion(yhat, y)
    loss.backward()
    optimizer.step()`}</CodeBlock>

      <Prose>
        <p>
          Every tensor moves to the GPU; the loop is byte-for-byte identical.
          The scribe doesn&apos;t care where the ink is — it just cares that
          everything stays together. Mix a CPU tensor with a CUDA one and
          you&apos;ll get a polite runtime error telling you to pick a side.
        </p>
        <p>Mixed precision — use fp16 for speed, fp32 for stability:</p>
      </Prose>

      <CodeBlock language="python" caption="automatic mixed precision — roughly 2× faster on modern GPUs">{`from torch.amp import autocast, GradScaler

scaler = GradScaler()

for step in range(250):
    optimizer.zero_grad()
    with autocast(device_type="cuda", dtype=torch.float16):
        yhat = model(x)                       # forward in fp16
        loss = criterion(yhat, y)
    scaler.scale(loss).backward()             # scale the loss up so fp16 grads don't underflow
    scaler.step(optimizer)                    # unscale + step
    scaler.update()                           # adjust the scale factor`}</CodeBlock>

      <Callout variant="insight" title="the frame you'll never lose">
        The five-line loop is the <em>heartbeat</em> of every deep-learning codebase in
        existence. From a 1-line linear regressor to Llama-3 pre-training on ten thousand
        GPUs — it is always this loop. Distributed training wraps it with an all-reduce.
        Mixed precision wraps it with autocast. Gradient accumulation just delays the step.
        But the five verbs — zero, forward, loss, backward, step — never change. Internalize
        them and every deep-learning codebase suddenly feels familiar.
      </Callout>

      <Challenge prompt="Write your own nn.Module">
        <p>
          Subclass <code>nn.Module</code> to implement a two-layer MLP (16 hidden units, ReLU)
          without using <code>nn.Sequential</code>. Register the two linear layers in{' '}
          <code>__init__</code>, implement <code>forward</code>, then train it on{' '}
          <code>y = sin(x)</code> sampled on <code>[-π, π]</code>. Use the five-line loop.
          Plot the fit after 1000 steps.
        </p>
        <p className="mt-2 text-dark-text-muted">
          Bonus: swap SGD for Adam (<code>torch.optim.Adam</code>, default lr 1e-3) and watch
          it converge roughly 3× faster. That&apos;s the single most popular optimizer
          upgrade in practice.
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> The scribe is the whole
          point of PyTorch. Tensors are the ink it records. Autograd is the
          ledger and the reverse walk. <code>nn.Module</code> is a
          pre-fabricated entry. The optimizer is the person who reads the
          gradients and turns the dials. Every training script is a five-line
          loop: zero, forward, loss, backward, step. Debug shape-first,
          value-second. Use <code>.to(device)</code> to move to GPU and{' '}
          <code>autocast</code> for free fp16 speedups.
        </p>
        <p>
          <strong>Next up — Layer Normalization.</strong> There is one tool
          the scribe doesn&apos;t give you for free: a way to keep activation
          distributions stable as they flow through a deep net. Without it,
          activations drift, gradients explode, and early layers quietly stop
          learning. Layer normalization is the fix baked into every modern
          transformer. You&apos;ll derive it, visualize the distribution it
          enforces, and see why the scribe needed help.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'PyTorch — Official documentation',
            author: 'PyTorch core team',
            venue: 'pytorch.org',
            url: 'https://pytorch.org/docs/stable/index.html',
          },
          {
            title: 'PyTorch Internals — How it\'s actually implemented',
            author: 'Edward Z. Yang',
            venue: 'blog post, 2019',
            url: 'http://blog.ezyang.com/2019/05/pytorch-internals/',
          },
          {
            title: 'Automatic differentiation in PyTorch',
            author: 'Paszke et al.',
            venue: 'NeurIPS 2017 AutoDiff Workshop',
            url: 'https://openreview.net/forum?id=BJJsrmfCZ',
          },
          {
            title: 'Dive into Deep Learning — Chapter 5: The Deep Learning Toolchain',
            author: 'Zhang, Lipton, Li, Smola',
            venue: 'd2l.ai',
            url: 'https://d2l.ai/chapter_multilayer-perceptrons/index.html',
          },
        ]}
      />
    </div>
  )
}
