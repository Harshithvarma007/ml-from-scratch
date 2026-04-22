export type Difficulty = 'Easy' | 'Medium' | 'Hard'

export interface Lesson {
  slug: string
  title: string
  difficulty: Difficulty
  blurb: string
  sectionSlug: string
  /**
   * Lesson slugs this lesson depends on. Used by <WhatNext> and could power a
   * topological graph view in the future. Slugs only — section is resolved via
   * `findLessonBySlug()`. Authored sparingly on cornerstone lessons.
   */
  prerequisites?: string[]
  /**
   * Lesson slugs that build on this one. The inverse relation is computed
   * automatically by `getEnabledBy()`, so only hand-author this when you want
   * a curated ordering (e.g. "what you should read next", not "everything
   * that ever mentions this").
   */
  enables?: string[]
  /**
   * Additional search terms that aren't in the title or blurb — synonyms,
   * acronyms, related concepts. Powers the Cmd+K palette so a reader typing
   * "chain rule" finds Backpropagation even though those words don't appear
   * in the title. Optional; sparingly seeded on lessons whose title differs
   * from how readers commonly search for the concept.
   */
  keywords?: string[]
}

export interface Section {
  slug: string
  title: string
  subtitle: string
  accent: SectionAccent
  icon: SectionIcon
  lessons: Omit<Lesson, 'sectionSlug'>[]
}

export type SectionAccent =
  | 'purple'
  | 'cyan'
  | 'green'
  | 'amber'
  | 'pink'
  | 'blue'
  | 'magic'
  | 'teal'
  | 'rose'
  | 'orange'
  | 'indigo'
  | 'fuchsia'
  | 'emerald'
  | 'slate'
export type SectionIcon =
  | 'sigma'
  | 'network'
  | 'flame'
  | 'gauge'
  | 'type'
  | 'eye'
  | 'sparkles'
  | 'image'
  | 'repeat'
  | 'sliders'
  | 'boxes'
  | 'waves'
  | 'target'
  | 'zap'

// Centralized Tailwind class map for each accent.
// Purely presentational — referenced by home cards, section headers, and lesson header accents.
export const accentClasses: Record<
  SectionAccent,
  { text: string; bg: string; border: string; ring: string; dot: string; glow: string }
> = {
  purple: {
    text: 'text-term-purple',
    bg: 'bg-term-purple/10',
    border: 'border-term-purple/30',
    ring: 'ring-term-purple/20',
    dot: 'bg-term-purple',
    glow: 'shadow-[0_0_24px_-4px_rgba(167,139,250,0.25)]',
  },
  cyan: {
    text: 'text-term-cyan',
    bg: 'bg-term-cyan/10',
    border: 'border-term-cyan/30',
    ring: 'ring-term-cyan/20',
    dot: 'bg-term-cyan',
    glow: 'shadow-[0_0_24px_-4px_rgba(103,232,249,0.25)]',
  },
  green: {
    text: 'text-term-green',
    bg: 'bg-term-green/10',
    border: 'border-term-green/30',
    ring: 'ring-term-green/20',
    dot: 'bg-term-green',
    glow: 'shadow-[0_0_24px_-4px_rgba(74,222,128,0.25)]',
  },
  amber: {
    text: 'text-term-amber',
    bg: 'bg-term-amber/10',
    border: 'border-term-amber/30',
    ring: 'ring-term-amber/20',
    dot: 'bg-term-amber',
    glow: 'shadow-[0_0_24px_-4px_rgba(251,191,36,0.25)]',
  },
  pink: {
    text: 'text-term-pink',
    bg: 'bg-term-pink/10',
    border: 'border-term-pink/30',
    ring: 'ring-term-pink/20',
    dot: 'bg-term-pink',
    glow: 'shadow-[0_0_24px_-4px_rgba(244,114,182,0.25)]',
  },
  blue: {
    text: 'text-dark-sql',
    bg: 'bg-dark-sql/10',
    border: 'border-dark-sql/30',
    ring: 'ring-dark-sql/20',
    dot: 'bg-dark-sql',
    glow: 'shadow-[0_0_24px_-4px_rgba(96,165,250,0.25)]',
  },
  magic: {
    text: 'text-dark-magic',
    bg: 'bg-dark-magic/10',
    border: 'border-dark-magic/30',
    ring: 'ring-dark-magic/20',
    dot: 'bg-dark-magic',
    glow: 'shadow-[0_0_24px_-4px_rgba(192,132,252,0.25)]',
  },
  teal: {
    text: 'text-term-teal',
    bg: 'bg-term-teal/10',
    border: 'border-term-teal/30',
    ring: 'ring-term-teal/20',
    dot: 'bg-term-teal',
    glow: 'shadow-[0_0_24px_-4px_rgba(94,234,212,0.25)]',
  },
  rose: {
    text: 'text-term-rose',
    bg: 'bg-term-rose/10',
    border: 'border-term-rose/30',
    ring: 'ring-term-rose/20',
    dot: 'bg-term-rose',
    glow: 'shadow-[0_0_24px_-4px_rgba(251,113,133,0.25)]',
  },
  orange: {
    text: 'text-term-orange',
    bg: 'bg-term-orange/10',
    border: 'border-term-orange/30',
    ring: 'ring-term-orange/20',
    dot: 'bg-term-orange',
    glow: 'shadow-[0_0_24px_-4px_rgba(251,146,60,0.25)]',
  },
  indigo: {
    text: 'text-term-indigo',
    bg: 'bg-term-indigo/10',
    border: 'border-term-indigo/30',
    ring: 'ring-term-indigo/20',
    dot: 'bg-term-indigo',
    glow: 'shadow-[0_0_24px_-4px_rgba(129,140,248,0.25)]',
  },
  fuchsia: {
    text: 'text-term-fuchsia',
    bg: 'bg-term-fuchsia/10',
    border: 'border-term-fuchsia/30',
    ring: 'ring-term-fuchsia/20',
    dot: 'bg-term-fuchsia',
    glow: 'shadow-[0_0_24px_-4px_rgba(232,121,249,0.25)]',
  },
  emerald: {
    text: 'text-term-emerald',
    bg: 'bg-term-emerald/10',
    border: 'border-term-emerald/30',
    ring: 'ring-term-emerald/20',
    dot: 'bg-term-emerald',
    glow: 'shadow-[0_0_24px_-4px_rgba(52,211,153,0.25)]',
  },
  slate: {
    text: 'text-term-slate',
    bg: 'bg-term-slate/10',
    border: 'border-term-slate/30',
    ring: 'ring-term-slate/20',
    dot: 'bg-term-slate',
    glow: 'shadow-[0_0_24px_-4px_rgba(148,163,184,0.25)]',
  },
}

export const roadmap: Section[] = [
  {
    slug: 'math-foundations',
    title: 'Math Foundations',
    subtitle: 'The calculus and linear algebra behind every neural net',
    accent: 'purple',
    icon: 'sigma',
    lessons: [
      {
        slug: 'gradient-descent',
        title: 'Gradient Descent',
        difficulty: 'Easy',
        blurb: 'The workhorse optimizer — derive, implement, and visualize it.',
        // No in-curriculum prerequisite — this is the opening algorithmic
        // lesson; assumed background is high-school calculus.
        enables: ['sigmoid-and-relu', 'linear-regression-training', 'backpropagation'],
        keywords: ['sgd', 'optimizer', 'learning rate', 'convergence', 'step size'],
      },
      {
        slug: 'sigmoid-and-relu',
        title: 'Sigmoid & ReLU',
        difficulty: 'Easy',
        blurb: 'Activation functions, their derivatives, and why ReLU won.',
      },
      {
        slug: 'softmax',
        title: 'Softmax',
        difficulty: 'Easy',
        blurb: 'Turn raw logits into calibrated probabilities.',
        keywords: ['logits', 'probabilities', 'normalize', 'temperature', 'classification head'],
      },
      {
        slug: 'cross-entropy-loss',
        title: 'Cross-Entropy Loss',
        difficulty: 'Medium',
        blurb: 'The canonical classification loss — from KL divergence down.',
        keywords: ['entropy', 'log loss', 'nll', 'negative log likelihood', 'kl divergence'],
      },
      {
        slug: 'linear-regression-forward',
        title: 'Linear Regression (Forward)',
        difficulty: 'Easy',
        blurb: 'Predictions as matrix multiplications.',
      },
      {
        slug: 'linear-regression-training',
        title: 'Linear Regression (Training)',
        difficulty: 'Medium',
        blurb: 'Closed-form vs iterative — when each wins.',
      },
    ],
  },
  {
    slug: 'build-a-neural-net',
    title: 'Build a Neural Net',
    subtitle: 'Neurons, layers, and backprop — wired by hand',
    accent: 'cyan',
    icon: 'network',
    lessons: [
      {
        slug: 'single-neuron',
        title: 'Single Neuron',
        difficulty: 'Easy',
        blurb: 'A weighted sum, a nonlinearity, a prediction.',
      },
      {
        slug: 'backpropagation',
        title: 'Backpropagation',
        difficulty: 'Medium',
        blurb: 'The chain rule, made mechanical.',
        keywords: ['chain rule', 'backprop', 'autograd', 'gradients', 'derivatives'],
      },
      {
        slug: 'multi-layer-backpropagation',
        title: 'Multi-Layer Backpropagation',
        difficulty: 'Hard',
        blurb: 'Chain rule across arbitrarily deep networks.',
      },
      {
        slug: 'backprop-ninja',
        title: 'Backprop Ninja',
        difficulty: 'Hard',
        blurb: 'Derive backward for a 2-layer MLP by hand — checked live against finite differences.',
        prerequisites: ['backpropagation', 'multi-layer-backpropagation'],
        enables: ['mlp-from-scratch', 'weight-initialization'],
      },
      {
        slug: 'mlp-from-scratch',
        title: 'MLP from Scratch',
        difficulty: 'Medium',
        blurb: 'A full multi-layer perceptron in pure NumPy.',
      },
      {
        slug: 'weight-initialization',
        title: 'Weight Initialization',
        difficulty: 'Medium',
        blurb: 'Xavier, He, and the math of exploding gradients.',
      },
    ],
  },
  {
    slug: 'pytorch',
    title: 'PyTorch',
    subtitle: 'Swap NumPy for autograd and GPUs',
    accent: 'amber',
    icon: 'flame',
    lessons: [
      {
        slug: 'pytorch-basics',
        title: 'PyTorch Basics',
        difficulty: 'Easy',
        blurb: 'Tensors, autograd, modules — the mental model.',
      },
      {
        slug: 'layer-normalization',
        title: 'Layer Normalization',
        difficulty: 'Medium',
        blurb: 'Per-sample normalization for transformers.',
      },
      {
        slug: 'batch-normalization',
        title: 'Batch Normalization',
        difficulty: 'Medium',
        blurb: 'The CNN-era normalization trick, fully unpacked.',
      },
      {
        slug: 'rms-normalization',
        title: 'RMS Normalization',
        difficulty: 'Medium',
        blurb: 'LayerNorm without mean subtraction — why it works.',
      },
    ],
  },
  {
    slug: 'training',
    title: 'Training',
    subtitle: 'The loop, the diagnostics, the first real model',
    accent: 'green',
    icon: 'gauge',
    lessons: [
      {
        slug: 'training-loop',
        title: 'Training Loop',
        difficulty: 'Easy',
        blurb: 'Forward, loss, backward, step — the four-line core.',
      },
      {
        slug: 'training-diagnostics',
        title: 'Training Diagnostics',
        difficulty: 'Medium',
        blurb: 'Loss curves, grad norms, and the art of debugging.',
      },
      {
        slug: 'dead-relu-detector',
        title: 'Dead ReLU Detector',
        difficulty: 'Medium',
        blurb: 'Find and fix silently-dying neurons.',
      },
      {
        slug: 'digit-classifier',
        title: 'Digit Classifier',
        difficulty: 'Medium',
        blurb: 'Ship a working MNIST model end-to-end.',
      },
    ],
  },
  {
    slug: 'cnns-and-vision',
    title: 'CNNs & Vision',
    subtitle: 'Filters, feature maps, and the architectures that taught machines to see',
    accent: 'teal',
    icon: 'image',
    lessons: [
      {
        slug: 'convolution-operation',
        title: 'Convolution Operation',
        difficulty: 'Easy',
        blurb: 'The kernel that slides, multiplies, sums.',
      },
      {
        slug: 'pooling',
        title: 'Pooling',
        difficulty: 'Easy',
        blurb: 'Max, average, and why we downsample.',
      },
      {
        slug: 'build-a-cnn',
        title: 'Build a CNN from Scratch',
        difficulty: 'Medium',
        blurb: 'A LeNet-style conv net, one layer at a time in NumPy.',
      },
      {
        slug: 'image-classifier',
        title: 'Image Classifier',
        difficulty: 'Medium',
        blurb: 'CIFAR-10 end-to-end — augmentation, training, evaluation.',
      },
      {
        slug: 'resnet-and-skip-connections',
        title: 'ResNet & Skip Connections',
        difficulty: 'Hard',
        blurb: 'Residuals, identity paths, and why depth stopped hurting.',
      },
      {
        slug: 'vision-transformer',
        title: 'Vision Transformer (ViT)',
        difficulty: 'Hard',
        blurb: 'Treating image patches as tokens — the bridge to attention.',
      },
    ],
  },
  {
    slug: 'rnn-and-lstm',
    title: 'RNN & LSTM',
    subtitle: 'Sequence modeling before attention — and the problems that motivated it',
    accent: 'rose',
    icon: 'repeat',
    lessons: [
      {
        slug: 'recurrent-neural-network',
        title: 'Recurrent Neural Network',
        difficulty: 'Medium',
        blurb: 'Hidden state, shared weights, sequential processing.',
      },
      {
        slug: 'backprop-through-time',
        title: 'Backprop Through Time',
        difficulty: 'Hard',
        blurb: 'Unroll the loop to compute gradients across a sequence.',
      },
      {
        slug: 'vanishing-gradient-problem',
        title: 'Vanishing Gradient Problem',
        difficulty: 'Hard',
        blurb: 'Why long sequences kill plain RNNs — analytically.',
      },
      {
        slug: 'lstm',
        title: 'LSTM',
        difficulty: 'Hard',
        blurb: 'Gates, cell state, and the first real fix for long memory.',
      },
      {
        slug: 'gru',
        title: 'GRU',
        difficulty: 'Medium',
        blurb: 'A lighter LSTM that often matches it.',
      },
    ],
  },
  {
    slug: 'nlp',
    title: 'NLP',
    subtitle: 'From bag-of-words to dense meaning vectors',
    accent: 'pink',
    icon: 'type',
    lessons: [
      {
        slug: 'word-embeddings',
        title: 'Word Embeddings',
        difficulty: 'Medium',
        blurb: 'Meaning as geometry in ℝⁿ.',
      },
      {
        slug: 'intro-to-nlp',
        title: 'Intro to Natural Language Processing',
        difficulty: 'Easy',
        blurb: 'Tokens, vocabs, and the classical NLP pipeline.',
      },
      {
        slug: 'sentiment-analysis',
        title: 'Sentiment Analysis',
        difficulty: 'Medium',
        blurb: 'A minimal end-to-end text classifier.',
      },
      {
        slug: 'positional-encoding',
        title: 'Positional Encoding',
        difficulty: 'Hard',
        blurb: 'Sinusoids, RoPE, and why order matters.',
      },
    ],
  },
  {
    slug: 'attention-and-transformers',
    title: 'Attention & Transformers',
    subtitle: 'The single mechanism that reshaped deep learning',
    accent: 'blue',
    icon: 'eye',
    lessons: [
      {
        slug: 'self-attention',
        title: 'Self Attention',
        difficulty: 'Hard',
        blurb: 'Queries, keys, values — derived and animated.',
        prerequisites: ['word-embeddings', 'positional-encoding', 'recurrent-neural-network'],
        enables: ['multi-headed-self-attention', 'transformer-block'],
        keywords: ['qkv', 'query key value', 'scaled dot product', 'attention is all you need'],
      },
      {
        slug: 'multi-headed-self-attention',
        title: 'Multi Headed Self Attention',
        difficulty: 'Hard',
        blurb: 'Parallel attention heads specializing on different patterns.',
        prerequisites: ['self-attention'],
        enables: ['transformer-block', 'grouped-query-attention'],
      },
      {
        slug: 'transformer-block',
        title: 'Transformer Block',
        difficulty: 'Hard',
        blurb: 'Attention + MLP + norms + residuals — one layer.',
      },
    ],
  },
  {
    slug: 'build-gpt',
    title: 'Build GPT',
    subtitle: 'A working GPT, built lesson by lesson',
    accent: 'magic',
    icon: 'sparkles',
    lessons: [
      {
        slug: 'tokenizer-bpe',
        title: 'Tokenizer (Byte Pair Encoding)',
        difficulty: 'Hard',
        blurb: 'BPE from scratch — merges, vocabs, edge cases.',
        keywords: ['bpe', 'tokenizer', 'subword', 'tiktoken', 'sentencepiece', 'vocabulary'],
      },
      {
        slug: 'build-vocabulary',
        title: 'Build Vocabulary',
        difficulty: 'Medium',
        blurb: 'Train a BPE vocab on real text.',
      },
      {
        slug: 'tokenization-edge-cases',
        title: 'Tokenization Edge Cases',
        difficulty: 'Medium',
        blurb: 'Whitespace, unicode, emoji, and the quirks of tiktoken.',
      },
      {
        slug: 'gpt-data-loader',
        title: 'GPT Data Loader',
        difficulty: 'Medium',
        blurb: 'Streaming tokens into the model efficiently.',
      },
      {
        slug: 'gpt-dataset',
        title: 'GPT Dataset',
        difficulty: 'Medium',
        blurb: 'Context windows, next-token targets, packing.',
      },
      {
        slug: 'code-gpt',
        title: 'Code GPT',
        difficulty: 'Hard',
        blurb: 'Assemble the full GPT architecture.',
      },
      {
        slug: 'train-your-gpt',
        title: 'Train Your GPT',
        difficulty: 'Hard',
        blurb: 'AdamW, warmup, cosine decay — the real recipe.',
      },
      {
        slug: 'make-gpt-talk-back',
        title: 'Make GPT Talk Back',
        difficulty: 'Medium',
        blurb: 'Sampling: temperature, top-k, nucleus.',
      },
      {
        slug: 'kv-cache',
        title: 'KV-Cache',
        difficulty: 'Hard',
        blurb: 'The single trick behind fast inference.',
        keywords: ['kv cache', 'inference', 'generation', 'decoding', 'memory'],
      },
      {
        slug: 'grouped-query-attention',
        title: 'Grouped Query Attention',
        difficulty: 'Hard',
        blurb: 'Llama-style attention: memory savings without accuracy loss.',
      },
    ],
  },
  {
    slug: 'fine-tuning-and-rlhf',
    title: 'Fine-Tuning & RLHF',
    subtitle: 'From a base model to an aligned, instruction-following assistant',
    accent: 'orange',
    icon: 'sliders',
    lessons: [
      {
        slug: 'supervised-fine-tuning',
        title: 'Supervised Fine-Tuning',
        difficulty: 'Medium',
        blurb: 'Turn a base model into an instruction-follower.',
      },
      {
        slug: 'lora',
        title: 'LoRA',
        difficulty: 'Hard',
        blurb: 'Low-rank adapters — fine-tune 0.1% of the parameters.',
        keywords: ['lora', 'peft', 'adapters', 'low rank', 'parameter efficient'],
      },
      {
        slug: 'qlora',
        title: 'QLoRA',
        difficulty: 'Hard',
        blurb: 'LoRA on 4-bit weights — fine-tune a 70B on a single GPU.',
      },
      {
        slug: 'reward-modeling',
        title: 'Reward Modeling',
        difficulty: 'Hard',
        blurb: 'Train a preference model from human pairwise comparisons.',
      },
      {
        slug: 'ppo-for-rlhf',
        title: 'PPO for RLHF',
        difficulty: 'Hard',
        blurb: 'Policy optimization against a learned reward model.',
      },
      {
        slug: 'direct-preference-optimization',
        title: 'Direct Preference Optimization',
        difficulty: 'Hard',
        blurb: 'RLHF without a separate reward model — the elegant alternative.',
      },
    ],
  },
  {
    slug: 'mixture-of-experts',
    title: 'Mixture of Experts',
    subtitle: 'Sparse activation — the next axis of scale',
    accent: 'indigo',
    icon: 'boxes',
    lessons: [
      {
        slug: 'moe-fundamentals',
        title: 'MoE Fundamentals',
        difficulty: 'Medium',
        blurb: 'Why sparse activation lets you scale parameters without scaling FLOPs.',
      },
      {
        slug: 'top-k-routing',
        title: 'Top-k Routing',
        difficulty: 'Hard',
        blurb: 'The gating network that picks which experts see each token.',
      },
      {
        slug: 'load-balancing-loss',
        title: 'Load Balancing Loss',
        difficulty: 'Hard',
        blurb: 'Preventing expert collapse — keep every expert busy.',
      },
      {
        slug: 'expert-parallelism',
        title: 'Expert Parallelism',
        difficulty: 'Hard',
        blurb: 'Distributing experts across GPUs at training time.',
      },
    ],
  },
  {
    slug: 'diffusion-models',
    title: 'Diffusion Models',
    subtitle: 'Generate images by learning to reverse noise',
    accent: 'fuchsia',
    icon: 'waves',
    lessons: [
      {
        slug: 'denoising-intuition',
        title: 'Denoising Intuition',
        difficulty: 'Easy',
        blurb: 'Learn to reverse a staircase of Gaussian noise.',
      },
      {
        slug: 'forward-and-reverse-diffusion',
        title: 'Forward & Reverse Diffusion',
        difficulty: 'Hard',
        blurb: 'The two Markov chains that define the generative process.',
      },
      {
        slug: 'unet-architecture',
        title: 'U-Net Architecture',
        difficulty: 'Medium',
        blurb: 'Skip connections across a contracting-expanding path.',
      },
      {
        slug: 'ddpm-from-scratch',
        title: 'DDPM from Scratch',
        difficulty: 'Hard',
        blurb: 'Train a tiny diffusion model on MNIST, end to end.',
      },
      {
        slug: 'classifier-free-guidance',
        title: 'Classifier-Free Guidance',
        difficulty: 'Hard',
        blurb: 'Steer generation without an auxiliary classifier.',
      },
      {
        slug: 'latent-diffusion',
        title: 'Latent Diffusion',
        difficulty: 'Hard',
        blurb: 'Why Stable Diffusion works in VAE latent space.',
      },
    ],
  },
  {
    slug: 'reinforcement-learning',
    title: 'Reinforcement Learning',
    subtitle: 'Learn from reward signals — the algorithms behind AlphaGo and RLHF',
    accent: 'emerald',
    icon: 'target',
    lessons: [
      {
        slug: 'markov-decision-processes',
        title: 'Markov Decision Processes',
        difficulty: 'Medium',
        blurb: 'States, actions, rewards, transitions — the RL contract.',
      },
      {
        slug: 'q-learning',
        title: 'Q-Learning',
        difficulty: 'Medium',
        blurb: 'Learn a value function from experience, one update at a time.',
      },
      {
        slug: 'policy-gradients',
        title: 'Policy Gradients',
        difficulty: 'Hard',
        blurb: 'Optimize the policy directly via gradient ascent on expected reward.',
      },
      {
        slug: 'reinforce',
        title: 'REINFORCE',
        difficulty: 'Hard',
        blurb: 'The cleanest policy-gradient algorithm — and its variance problem.',
      },
      {
        slug: 'actor-critic',
        title: 'Actor-Critic',
        difficulty: 'Hard',
        blurb: 'Combine policy learning with a value baseline.',
      },
      {
        slug: 'proximal-policy-optimization',
        title: 'Proximal Policy Optimization',
        difficulty: 'Hard',
        blurb: 'The stable RL algorithm behind RLHF.',
        prerequisites: ['policy-gradients', 'reinforce', 'actor-critic'],
        enables: ['ppo-for-rlhf'],
      },
    ],
  },
  {
    slug: 'inference-and-serving',
    title: 'Inference & Serving',
    subtitle: 'Ship the model — make it fast, cheap, and production-ready',
    accent: 'slate',
    icon: 'zap',
    lessons: [
      {
        slug: 'quantization-basics',
        title: 'Quantization Basics',
        difficulty: 'Medium',
        blurb: 'Why lower precision is an (almost) free lunch.',
      },
      {
        slug: 'int8-int4-quantization',
        title: 'INT8 & INT4 Quantization',
        difficulty: 'Hard',
        blurb: 'Post-training quantization and QAT, in detail.',
      },
      {
        slug: 'speculative-decoding',
        title: 'Speculative Decoding',
        difficulty: 'Hard',
        blurb: 'A small model drafts, the big model verifies — 2-3× faster.',
      },
      {
        slug: 'continuous-batching',
        title: 'Continuous Batching',
        difficulty: 'Hard',
        blurb: 'The throughput trick that makes production LLMs economical.',
      },
      {
        slug: 'paged-attention',
        title: 'Paged Attention',
        difficulty: 'Hard',
        blurb: "vLLM's virtual-memory-inspired KV cache.",
      },
    ],
  },
]

// Flat array of every lesson with sectionSlug attached — handy for nav and routing.
export function flatLessons(): Lesson[] {
  const out: Lesson[] = []
  for (const section of roadmap) {
    for (const lesson of section.lessons) {
      out.push({ ...lesson, sectionSlug: section.slug })
    }
  }
  return out
}

export function getSection(sectionSlug: string): Section | null {
  return roadmap.find((s) => s.slug === sectionSlug) ?? null
}

export function getLesson(sectionSlug: string, lessonSlug: string): Lesson | null {
  const section = getSection(sectionSlug)
  if (!section) return null
  const lesson = section.lessons.find((l) => l.slug === lessonSlug)
  if (!lesson) return null
  return { ...lesson, sectionSlug }
}

export function getAdjacent(
  sectionSlug: string,
  lessonSlug: string
): { prev: Lesson | null; next: Lesson | null } {
  const all = flatLessons()
  const idx = all.findIndex((l) => l.sectionSlug === sectionSlug && l.slug === lessonSlug)
  if (idx === -1) return { prev: null, next: null }
  return {
    prev: idx > 0 ? all[idx - 1] : null,
    next: idx < all.length - 1 ? all[idx + 1] : null,
  }
}

export function totalLessons(): number {
  return roadmap.reduce((sum, s) => sum + s.lessons.length, 0)
}

export const firstLesson = (): Lesson => flatLessons()[0]

// Slug-only lookup. Useful for resolving `prerequisites`/`enables` refs which
// only carry the lesson slug (not the section), and for the command palette.
// First match wins — lesson slugs are currently globally unique across the
// roadmap. If we ever allow collisions, extend this to return all matches.
export function findLessonBySlug(slug: string): Lesson | null {
  for (const section of roadmap) {
    const hit = section.lessons.find((l) => l.slug === slug)
    if (hit) return { ...hit, sectionSlug: section.slug }
  }
  return null
}

// Inverse index: which lessons declare this one as a prerequisite? Computed
// at read time so we don't have to keep both sides of the edge in sync by
// hand. Returned in roadmap order.
export function getEnabledBy(slug: string): Lesson[] {
  const out: Lesson[] = []
  for (const section of roadmap) {
    for (const lesson of section.lessons) {
      if (lesson.prerequisites?.includes(slug)) {
        out.push({ ...lesson, sectionSlug: section.slug })
      }
    }
  }
  return out
}
