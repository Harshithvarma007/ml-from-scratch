import CodeBlock from '../CodeBlock'
import MathBlock from '../MathBlock'
import Prereq from '../Prereq'
import NeedsBackground from '../NeedsBackground'
import {
  Prose, Callout, Personify, Bridge, Gotcha, Challenge, References, KeyTerm,
} from '../primitives'
import LiveNeuronMonitor from '../widgets/LiveNeuronMonitor'
import ActivationHealthChart from '../widgets/ActivationHealthChart'
import RescueMode from '../widgets/RescueMode'

// Anchor: the electrician hunting broken bulbs in a chandelier. A dead ReLU
// is a bulb stuck off — the neuron's input is always negative, ReLU outputs
// zero, gradient is zero, weights never update, the bulb stays dead. The
// chandelier still lights up (the model still trains, the loss still drops)
// but it's dimmer than it should be because 10–30% of the bulbs are out.
// Threaded at three load-bearing moments: the silent failure at the open,
// the detector code (mechanical check for always-zero outputs), and the
// fix menu (reinit / LR / activation swap).

export default function DeadReluDetectorLesson() {
  return (
    <div className="space-y-6">
      <Prereq currentSlug="dead-relu-detector" />

      <Prose>
        <p>
          You walked into the dining room and the chandelier is on. Great.
          You&apos;d never stop and count bulbs. But a third of them are out
          — not broken glass, not a flicker, just quietly dark — and the
          room is dimmer than the architect drew it. You can&apos;t tell
          because the ones that <em>are</em> lit are doing enough work to
          make the place look fine.
        </p>
        <p>
          That is the failure mode of this lesson. A{' '}
          <KeyTerm>dead ReLU</KeyTerm> neuron is a bulb stuck in the off
          position. Its pre-activation never climbs above zero, so{' '}
          <NeedsBackground slug="sigmoid-and-relu">ReLU</NeedsBackground>{' '}
          outputs zero for every example, the gradient through it is zero,
          and its weights freeze in place for the rest of training. The loss
          curve looks normal. The validation accuracy looks normal. Ninety
          percent of the network is still learning. The problem is invisible
          to every high-level diagnostic — and it&apos;s the single most
          common cause of &ldquo;why is my ReLU net plateauing at an
          accuracy below what a smaller network gets.&rdquo; The chandelier
          is lit. The chandelier is also dimmer than it should be. Nobody
          has checked.
        </p>
        <p>
          This lesson is you with a ladder and a notebook, going bulb by
          bulb. Three tools: a live per-neuron monitor you can watch during
          training, a per-layer health chart over the full run, and a menu
          of rescue strategies for the dead ones. By the end you&apos;ll
          have a reflex — when accuracy plateaus in a ReLU network, check
          the dead-fraction <em>before</em> you reach for fancier
          architectures.
        </p>
      </Prose>

      <Callout variant="note" title="the exact failure condition">
        A ReLU neuron is dead at step <code>t</code> if, for every training
        example <code>i</code>, the pre-activation{' '}
        <code>w·xᵢ + b ≤ 0</code>. Then the output is zero, and the gradient
        through the ReLU is zero, so <code>∂L/∂w = 0</code> and{' '}
        <code>∂L/∂b = 0</code>. With zero gradient,{' '}
        <NeedsBackground slug="gradient-descent">SGD</NeedsBackground>{' '}
        cannot move <code>w</code> or <code>b</code> — they are permanently
        stuck. Every future training step makes no difference to this
        neuron. The filament is cold; no current will ever flow through it
        again.
      </Callout>

      <MathBlock caption="the math of permadeath">
{`z_n  =  w_n · x  +  b_n            pre-activation for neuron n

If for every x in the training set, z_n(x) ≤ 0:

   ReLU(z_n)        =  0              — output zero
   ∂ReLU / ∂z_n     =  0              — gradient through ReLU is zero
   ∂L / ∂w_n, ∂b_n  =  0              — no update signal

Neuron n is permanently frozen. The rest of the network can't help it —
any upstream signal still multiplies by the zero-derivative of ReLU.`}
      </MathBlock>

      <Prose>
        <p>
          That last line is why this failure is terminal instead of
          temporary. A normal inactive neuron — zero on this input, alive on
          the next — has escape velocity. A dead neuron&apos;s own gradient
          is zero, so nothing upstream can push it back above the threshold.
          The rest of the chandelier can blaze; this socket is dark forever.
          Forward pass: the light flicks on. Drop a failed bulb in there and
          watch what the monitor catches.
        </p>
      </Prose>

      <LiveNeuronMonitor />

      <Prose>
        <p>
          Watch the grid. With He{' '}
          <NeedsBackground slug="weight-initialization">init</NeedsBackground>{' '}
          and a sane learning rate, the network converges and most bulbs
          stay lit. Crank the init scale down or the learning rate up and
          cells start going dark — one at a time, never recovering. That
          pattern is the dead-ReLU failure mode in live action. Note the
          word <em>never</em>. You&apos;re not watching a neuron take a
          break; you&apos;re watching one die.
        </p>
      </Prose>

      <Personify speaker="Dead neuron">
        I used to fire. Then my bias got pushed below the minimum of any
        pre-activation I see, and now I output zero on every example. My
        gradient is zero. My weights are frozen. I am, in every practical
        sense, not part of the network anymore — but I still take up memory,
        still cost a multiply-accumulate in the forward pass, and nobody
        has noticed because I&apos;m hidden in a grid of 128 other neurons
        still doing their jobs. Think of me as the bulb you only notice
        when you finally climb the ladder.
      </Personify>

      <ActivationHealthChart />

      <Prose>
        <p>
          Zoom out from one bulb to the whole layer. The metric to log is
          <em> percentage alive</em> — fraction of neurons that were
          non-zero on at least one example in a validation batch. A healthy
          initialization lands near 50% (by ReLU&apos;s symmetry). Anything
          much below that after a few hundred steps is a red flag. Dragging
          the LR above about <code>0.5</code> for this toy network starts
          pushing layers into the dead zone; flipping to Leaky ReLU keeps
          them flat. Same chandelier, different wiring.
        </p>
      </Prose>

      <Callout variant="insight" title="why deeper layers die faster">
        Dead bulbs compound. If layer 1 has 30% dead, then layer 2 sees a
        degraded signal — more of its pre-activations are near-zero, and
        its own bias update pushes more of them below zero. By layer 5 the
        dead fraction can be 80%+. The fix has to happen early; detecting
        dead neurons at the <em>first</em> layer is a leading indicator for
        the whole network. One burned-out bulb near the ceiling and the
        sockets below it never see enough voltage to glow.
      </Callout>

      <Personify speaker="Percent-alive metric">
        I am the one-line metric you should log on every ReLU run. At 50%
        I am happy. At 30% you should worry. Below 20% your network is
        effectively a much smaller network, and the next time you hit an
        accuracy ceiling, I will be why.
      </Personify>

      <Prose>
        <p>
          So. You&apos;ve found the dark sockets. What do you actually{' '}
          <em>do</em>? There are four moves on the electrician&apos;s belt,
          and they trade off differently.
        </p>
      </Prose>

      <RescueMode />

      <Prose>
        <p>
          Four strategies, each with its tradeoffs. <strong>Lower the LR</strong>
          {' '}— turn down the voltage; cheapest move, prevents new deaths,
          won&apos;t resurrect the already-dead. <strong>Swap to Leaky ReLU</strong>
          {' '}— swap the bulb type; tiny code change, revives everything
          because the negative side leaks a sliver of current instead of
          zero, mild loss of sparsity. <strong>Re-initialize dead neurons</strong>
          {' '}— literally replace the bulb; surgical, preserves ReLU,
          requires detection machinery. <strong>Swap to GELU</strong> —
          modern default in transformers, smooth activation with no
          hard-zero region, slightly more compute. For a new project start
          with He init plus a modest LR (you&apos;ll see <code>~5%</code>{' '}
          dead and it&apos;s fine). For an existing project that&apos;s
          plateauing, check the dead-fraction first — the cheapest bug to
          fix is the one you&apos;ve already diagnosed. Now the code that
          does the diagnosing.
        </p>
      </Prose>

      <CodeBlock
        language="python"
        caption="detection · pure python"
        output={`layer h1: 28% dead  (36 / 128 neurons)
layer h2: 41% dead  (53 / 128 neurons)
layer h3: 64% dead  (82 / 128 neurons)`}
      >{`# Given a trained PyTorch-style MLP with dense hidden layers.
# On a held-out batch of validation inputs, record how many neurons
# produced a non-zero output on at least one example.

def count_dead_neurons(model, val_batch):
    dead_per_layer = {}
    activations = val_batch                             # start with inputs
    for name, layer in model.named_modules():
        if not isinstance(layer, LinearReLU):           # pseudo — your Linear+ReLU
            continue
        z = layer.linear(activations)                   # pre-activation
        a = z.maximum(0)                                # post-activation
        active = (a > 0).any(dim=0)                     # True if alive at all
        dead = (~active).sum().item()
        dead_per_layer[name] = (dead, a.shape[-1])
        activations = a
    return dead_per_layer`}</CodeBlock>

      <Prose>
        <p>
          That loop is the electrician&apos;s checklist. For every bulb in
          every layer, did it light up for <em>any</em> example in the
          batch? If not, mark it dead. The <code>.any(dim=0)</code> is the
          whole trick — one non-zero reading anywhere across the batch is
          enough to prove a neuron is alive. Silence across the whole batch
          is the signature of a stuck bulb. Now the library-grade version:
          same checklist, no model surgery required.
        </p>
      </Prose>

      <CodeBlock language="python" caption="detection · pytorch with hooks">{`import torch
import torch.nn as nn

class DeadNeuronProbe:
    def __init__(self, model):
        self.active_seen = {}
        for name, m in model.named_modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            # output shape: (batch, features)   — track which features ever fired
            seen = (output > 0).any(dim=0).detach()
            if name in self.active_seen:
                self.active_seen[name] = self.active_seen[name] | seen
            else:
                self.active_seen[name] = seen
        return hook

    def report(self):
        for name, seen in self.active_seen.items():
            dead = (~seen).sum().item()
            total = seen.numel()
            print(f"{name}: {dead}/{total} dead  ({100 * dead / total:.1f}%)")

# Usage in a training / eval loop:
# probe = DeadNeuronProbe(model)
# for batch in val_loader: model(batch.x)
# probe.report()`}</CodeBlock>

      <Bridge
        label="detection pattern, layered up"
        rows={[
          { left: 'loop through layers manually', right: 'forward hooks on every ReLU', note: 'no model modifications needed' },
          { left: 'per-batch counts', right: 'OR-accumulate across the full val set', note: 'a neuron is "alive" if it ever fired' },
          { left: 'print once', right: 'log to tensorboard per epoch', note: 'watch the dead fraction evolve' },
        ]}
      />

      <Prose>
        <p>
          The probe clamps onto every ReLU module and watches the current
          flow. It doesn&apos;t touch the model. It doesn&apos;t add a
          parameter. It just reports how many bulbs lit at least once
          across your validation pass. That&apos;s the kind of diagnostic
          you leave running on every serious training run — same way{' '}
          <NeedsBackground slug="training-diagnostics">training diagnostics</NeedsBackground>
          {' '}logs loss and gradient norm. Now the easiest rescue: swap the
          bulb type everywhere.
        </p>
      </Prose>

      <CodeBlock language="python" caption="rescue · swap activations in place">{`# Once detection finds too many dead neurons, the cheapest fix is swapping
# in LeakyReLU. This works without re-initialising anything:

def replace_relu_with_leaky(model, negative_slope=0.1):
    for name, m in model.named_children():
        if isinstance(m, nn.ReLU):
            setattr(model, name, nn.LeakyReLU(negative_slope))
        else:
            replace_relu_with_leaky(m)                  # recurse

# replace_relu_with_leaky(model)
# Resume training. Dead neurons will start receiving non-zero gradients.`}</CodeBlock>

      <Gotcha>
        <p>
          <strong className="text-term-amber">Checking on a single batch</strong>{' '}
          gives noisy answers. A neuron that failed on the current batch
          might fire on the next one. Accumulate &ldquo;did this neuron
          ever fire&rdquo; across a whole validation pass (or at minimum a
          few hundred examples). One flicker doesn&apos;t mean dead.
        </p>
        <p>
          <strong className="text-term-amber">Confusing &ldquo;inactive&rdquo; with &ldquo;dead&rdquo;.</strong>{' '}
          Inactive = zero on this input, alive on other inputs — perfectly
          healthy, and the source of ReLU&apos;s sparsity advantage. Dead =
          zero on every input — the pathology. A bulb that&apos;s off right
          now is not the same as a bulb that will never come on again. Only
          the second needs fixing.
        </p>
        <p>
          <strong className="text-term-amber">Forgetting that biases move.</strong>{' '}
          A network can have a perfectly healthy init and still produce
          dead neurons after enough training steps because the bias drifted
          too negative. Detection must run during/after training, not just
          at init. Some bulbs burn out on the way to the minimum, not at
          the factory.
        </p>
        <p>
          <strong className="text-term-amber">Re-init can destabilise.</strong>{' '}
          If the rest of the network has learned around a dead neuron,
          re-initialising it with random weights injects noise upstream.
          Re-init during the first few epochs when the network hasn&apos;t
          calcified yet.
        </p>
      </Gotcha>

      <Challenge prompt="Kill half the network, then save it">
        <p>
          Train a 5-layer, 128-wide ReLU MLP on MNIST with a deliberately
          large LR (0.5) and show that by epoch 2,{' '}
          <code>&gt; 40%</code> of neurons in the last hidden layer are
          dead (log it per epoch using the probe above). Then apply each of
          the four rescue strategies in turn, continuing training, and plot
          the dead-fraction and validation accuracy curves. Which strategy
          wins on accuracy? Which is cheapest? Which would you use in
          production?
        </p>
      </Challenge>

      <Prose>
        <p>
          <strong>What to carry forward.</strong> Dead ReLUs are silent —
          they don&apos;t show up in the loss curve and they don&apos;t
          show up in the aggregate gradient norm. They only appear in
          per-neuron activation statistics. Log the dead-fraction per
          hidden layer on every serious ReLU run. When an accuracy ceiling
          appears, check this metric <em>first</em>. Leaky ReLU or GELU are
          the cheapest permanent fixes; smaller LR is the cheapest
          temporary one. Bring a ladder and a notebook; the chandelier
          won&apos;t tell you which bulbs are out.
        </p>
        <p>
          <strong>Next up — Digit Classifier.</strong> This is the moment
          everything you&apos;ve built gets shipped as a single model. Loss
          functions, the training loop, diagnostics, now activation health
          — all of it converging on MNIST, handwritten digits end-to-end.
          No more toy gradients on <code>x²</code>, no more synthetic
          monitors. Real pixels in, real predictions out, real validation
          accuracy you can brag about. The abstract tools are about to turn
          into a model that actually recognises handwriting.
        </p>
      </Prose>

      <References
        items={[
          {
            title: 'Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification',
            author: 'He, Zhang, Ren, Sun',
            venue: 'ICCV 2015 — the He init paper',
            url: 'https://arxiv.org/abs/1502.01852',
          },
          {
            title: 'Rectified Linear Units Improve Restricted Boltzmann Machines',
            author: 'Nair, Hinton',
            venue: 'ICML 2010 — the original ReLU paper',
            url: 'https://icml.cc/Conferences/2010/papers/432.pdf',
          },
          {
            title: 'Dying ReLU and Initialization: Theory and Numerical Examples',
            author: 'Lu, Shin, Su, Karniadakis',
            venue: 'arXiv 2019',
            url: 'https://arxiv.org/abs/1903.06733',
          },
          {
            title: 'Gaussian Error Linear Units (GELUs)',
            author: 'Hendrycks, Gimpel',
            year: 2016,
            url: 'https://arxiv.org/abs/1606.08415',
          },
        ]}
      />
    </div>
  )
}
