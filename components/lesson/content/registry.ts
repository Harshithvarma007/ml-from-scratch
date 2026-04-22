import type { ComponentType } from 'react'

// Math Foundations
import GradientDescentLesson from './lessons/gradient-descent'
import SigmoidAndReluLesson from './lessons/sigmoid-and-relu'
import SoftmaxLesson from './lessons/softmax'
import CrossEntropyLossLesson from './lessons/cross-entropy-loss'
import LinearRegressionForwardLesson from './lessons/linear-regression-forward'
import LinearRegressionTrainingLesson from './lessons/linear-regression-training'

// Build a Neural Net
import SingleNeuronLesson from './lessons/single-neuron'
import BackpropagationLesson from './lessons/backpropagation'
import MultiLayerBackpropagationLesson from './lessons/multi-layer-backpropagation'
import BackpropNinjaLesson from './lessons/backprop-ninja'
import MLPFromScratchLesson from './lessons/mlp-from-scratch'
import WeightInitializationLesson from './lessons/weight-initialization'

// PyTorch
import PyTorchBasicsLesson from './lessons/pytorch-basics'
import LayerNormalizationLesson from './lessons/layer-normalization'
import BatchNormalizationLesson from './lessons/batch-normalization'
import RMSNormalizationLesson from './lessons/rms-normalization'

// Training
import TrainingLoopLesson from './lessons/training-loop'
import TrainingDiagnosticsLesson from './lessons/training-diagnostics'
import DeadReluDetectorLesson from './lessons/dead-relu-detector'
import DigitClassifierLesson from './lessons/digit-classifier'

// CNNs & Vision
import ConvolutionOperationLesson from './lessons/convolution-operation'
import PoolingLesson from './lessons/pooling'
import BuildACnnLesson from './lessons/build-a-cnn'
import ImageClassifierLesson from './lessons/image-classifier'
import ResnetLesson from './lessons/resnet-and-skip-connections'
import VisionTransformerLesson from './lessons/vision-transformer'

// RNN & LSTM
import RecurrentNeuralNetworkLesson from './lessons/recurrent-neural-network'
import BackpropThroughTimeLesson from './lessons/backprop-through-time'
import VanishingGradientProblemLesson from './lessons/vanishing-gradient-problem'
import LstmLesson from './lessons/lstm'
import GruLesson from './lessons/gru'

// NLP
import WordEmbeddingsLesson from './lessons/word-embeddings'
import IntroToNlpLesson from './lessons/intro-to-nlp'
import SentimentAnalysisLesson from './lessons/sentiment-analysis'
import PositionalEncodingLesson from './lessons/positional-encoding'

// Attention & Transformers
import SelfAttentionLesson from './lessons/self-attention'
import MultiHeadedSelfAttentionLesson from './lessons/multi-headed-self-attention'
import TransformerBlockLesson from './lessons/transformer-block'

// Build GPT
import TokenizerBpeLesson from './lessons/tokenizer-bpe'
import BuildVocabularyLesson from './lessons/build-vocabulary'
import TokenizationEdgeCasesLesson from './lessons/tokenization-edge-cases'
import GptDataLoaderLesson from './lessons/gpt-data-loader'
import GptDatasetLesson from './lessons/gpt-dataset'
import CodeGptLesson from './lessons/code-gpt'
import TrainYourGptLesson from './lessons/train-your-gpt'
import MakeGptTalkBackLesson from './lessons/make-gpt-talk-back'
import KvCacheLesson from './lessons/kv-cache'
import GroupedQueryAttentionLesson from './lessons/grouped-query-attention'

// Fine-Tuning & RLHF
import SupervisedFineTuningLesson from './lessons/supervised-fine-tuning'
import LoraLesson from './lessons/lora'
import QloraLesson from './lessons/qlora'
import RewardModelingLesson from './lessons/reward-modeling'
import PpoForRlhfLesson from './lessons/ppo-for-rlhf'
import DpoLesson from './lessons/direct-preference-optimization'

// Mixture of Experts
import MoeFundamentalsLesson from './lessons/moe-fundamentals'
import TopKRoutingLesson from './lessons/top-k-routing'
import LoadBalancingLossLesson from './lessons/load-balancing-loss'
import ExpertParallelismLesson from './lessons/expert-parallelism'

// Diffusion Models
import DenoisingIntuitionLesson from './lessons/denoising-intuition'
import ForwardReverseDiffusionLesson from './lessons/forward-and-reverse-diffusion'
import UNetArchitectureLesson from './lessons/unet-architecture'
import DdpmFromScratchLesson from './lessons/ddpm-from-scratch'
import ClassifierFreeGuidanceLesson from './lessons/classifier-free-guidance'
import LatentDiffusionLesson from './lessons/latent-diffusion'

// Reinforcement Learning
import MdpLesson from './lessons/markov-decision-processes'
import QLearningLesson from './lessons/q-learning'
import PolicyGradientsLesson from './lessons/policy-gradients'
import ReinforceLesson from './lessons/reinforce'
import ActorCriticLesson from './lessons/actor-critic'
import PpoLesson from './lessons/proximal-policy-optimization'

// Inference & Serving
import QuantizationBasicsLesson from './lessons/quantization-basics'
import Int8Int4QuantizationLesson from './lessons/int8-int4-quantization'
import SpeculativeDecodingLesson from './lessons/speculative-decoding'
import ContinuousBatchingLesson from './lessons/continuous-batching'
import PagedAttentionLesson from './lessons/paged-attention'

export interface LessonContent {
  Component: ComponentType
  hideTOC?: boolean
}

const lessonContent: Record<string, LessonContent> = {
  // 1 — Math Foundations
  'math-foundations/gradient-descent': { Component: GradientDescentLesson, hideTOC: true },
  'math-foundations/sigmoid-and-relu': { Component: SigmoidAndReluLesson, hideTOC: true },
  'math-foundations/softmax': { Component: SoftmaxLesson, hideTOC: true },
  'math-foundations/cross-entropy-loss': { Component: CrossEntropyLossLesson, hideTOC: true },
  'math-foundations/linear-regression-forward': { Component: LinearRegressionForwardLesson, hideTOC: true },
  'math-foundations/linear-regression-training': { Component: LinearRegressionTrainingLesson, hideTOC: true },
  // 2 — Build a Neural Net
  'build-a-neural-net/single-neuron': { Component: SingleNeuronLesson, hideTOC: true },
  'build-a-neural-net/backpropagation': { Component: BackpropagationLesson, hideTOC: true },
  'build-a-neural-net/multi-layer-backpropagation': { Component: MultiLayerBackpropagationLesson, hideTOC: true },
  'build-a-neural-net/backprop-ninja': { Component: BackpropNinjaLesson, hideTOC: true },
  'build-a-neural-net/mlp-from-scratch': { Component: MLPFromScratchLesson, hideTOC: true },
  'build-a-neural-net/weight-initialization': { Component: WeightInitializationLesson, hideTOC: true },
  // 3 — PyTorch
  'pytorch/pytorch-basics': { Component: PyTorchBasicsLesson, hideTOC: true },
  'pytorch/layer-normalization': { Component: LayerNormalizationLesson, hideTOC: true },
  'pytorch/batch-normalization': { Component: BatchNormalizationLesson, hideTOC: true },
  'pytorch/rms-normalization': { Component: RMSNormalizationLesson, hideTOC: true },
  // 4 — Training
  'training/training-loop': { Component: TrainingLoopLesson, hideTOC: true },
  'training/training-diagnostics': { Component: TrainingDiagnosticsLesson, hideTOC: true },
  'training/dead-relu-detector': { Component: DeadReluDetectorLesson, hideTOC: true },
  'training/digit-classifier': { Component: DigitClassifierLesson, hideTOC: true },
  // 5 — CNNs & Vision
  'cnns-and-vision/convolution-operation': { Component: ConvolutionOperationLesson, hideTOC: true },
  'cnns-and-vision/pooling': { Component: PoolingLesson, hideTOC: true },
  'cnns-and-vision/build-a-cnn': { Component: BuildACnnLesson, hideTOC: true },
  'cnns-and-vision/image-classifier': { Component: ImageClassifierLesson, hideTOC: true },
  'cnns-and-vision/resnet-and-skip-connections': { Component: ResnetLesson, hideTOC: true },
  'cnns-and-vision/vision-transformer': { Component: VisionTransformerLesson, hideTOC: true },
  // 6 — RNN & LSTM
  'rnn-and-lstm/recurrent-neural-network': { Component: RecurrentNeuralNetworkLesson, hideTOC: true },
  'rnn-and-lstm/backprop-through-time': { Component: BackpropThroughTimeLesson, hideTOC: true },
  'rnn-and-lstm/vanishing-gradient-problem': { Component: VanishingGradientProblemLesson, hideTOC: true },
  'rnn-and-lstm/lstm': { Component: LstmLesson, hideTOC: true },
  'rnn-and-lstm/gru': { Component: GruLesson, hideTOC: true },
  // 7 — NLP
  'nlp/word-embeddings': { Component: WordEmbeddingsLesson, hideTOC: true },
  'nlp/intro-to-nlp': { Component: IntroToNlpLesson, hideTOC: true },
  'nlp/sentiment-analysis': { Component: SentimentAnalysisLesson, hideTOC: true },
  'nlp/positional-encoding': { Component: PositionalEncodingLesson, hideTOC: true },
  // 8 — Attention & Transformers
  'attention-and-transformers/self-attention': { Component: SelfAttentionLesson, hideTOC: true },
  'attention-and-transformers/multi-headed-self-attention': { Component: MultiHeadedSelfAttentionLesson, hideTOC: true },
  'attention-and-transformers/transformer-block': { Component: TransformerBlockLesson, hideTOC: true },
  // 9 — Build GPT
  'build-gpt/tokenizer-bpe': { Component: TokenizerBpeLesson, hideTOC: true },
  'build-gpt/build-vocabulary': { Component: BuildVocabularyLesson, hideTOC: true },
  'build-gpt/tokenization-edge-cases': { Component: TokenizationEdgeCasesLesson, hideTOC: true },
  'build-gpt/gpt-data-loader': { Component: GptDataLoaderLesson, hideTOC: true },
  'build-gpt/gpt-dataset': { Component: GptDatasetLesson, hideTOC: true },
  'build-gpt/code-gpt': { Component: CodeGptLesson, hideTOC: true },
  'build-gpt/train-your-gpt': { Component: TrainYourGptLesson, hideTOC: true },
  'build-gpt/make-gpt-talk-back': { Component: MakeGptTalkBackLesson, hideTOC: true },
  'build-gpt/kv-cache': { Component: KvCacheLesson, hideTOC: true },
  'build-gpt/grouped-query-attention': { Component: GroupedQueryAttentionLesson, hideTOC: true },
  // 10 — Fine-Tuning & RLHF
  'fine-tuning-and-rlhf/supervised-fine-tuning': { Component: SupervisedFineTuningLesson, hideTOC: true },
  'fine-tuning-and-rlhf/lora': { Component: LoraLesson, hideTOC: true },
  'fine-tuning-and-rlhf/qlora': { Component: QloraLesson, hideTOC: true },
  'fine-tuning-and-rlhf/reward-modeling': { Component: RewardModelingLesson, hideTOC: true },
  'fine-tuning-and-rlhf/ppo-for-rlhf': { Component: PpoForRlhfLesson, hideTOC: true },
  'fine-tuning-and-rlhf/direct-preference-optimization': { Component: DpoLesson, hideTOC: true },
  // 11 — Mixture of Experts
  'mixture-of-experts/moe-fundamentals': { Component: MoeFundamentalsLesson, hideTOC: true },
  'mixture-of-experts/top-k-routing': { Component: TopKRoutingLesson, hideTOC: true },
  'mixture-of-experts/load-balancing-loss': { Component: LoadBalancingLossLesson, hideTOC: true },
  'mixture-of-experts/expert-parallelism': { Component: ExpertParallelismLesson, hideTOC: true },
  // 12 — Diffusion Models
  'diffusion-models/denoising-intuition': { Component: DenoisingIntuitionLesson, hideTOC: true },
  'diffusion-models/forward-and-reverse-diffusion': { Component: ForwardReverseDiffusionLesson, hideTOC: true },
  'diffusion-models/unet-architecture': { Component: UNetArchitectureLesson, hideTOC: true },
  'diffusion-models/ddpm-from-scratch': { Component: DdpmFromScratchLesson, hideTOC: true },
  'diffusion-models/classifier-free-guidance': { Component: ClassifierFreeGuidanceLesson, hideTOC: true },
  'diffusion-models/latent-diffusion': { Component: LatentDiffusionLesson, hideTOC: true },
  // 13 — Reinforcement Learning
  'reinforcement-learning/markov-decision-processes': { Component: MdpLesson, hideTOC: true },
  'reinforcement-learning/q-learning': { Component: QLearningLesson, hideTOC: true },
  'reinforcement-learning/policy-gradients': { Component: PolicyGradientsLesson, hideTOC: true },
  'reinforcement-learning/reinforce': { Component: ReinforceLesson, hideTOC: true },
  'reinforcement-learning/actor-critic': { Component: ActorCriticLesson, hideTOC: true },
  'reinforcement-learning/proximal-policy-optimization': { Component: PpoLesson, hideTOC: true },
  // 14 — Inference & Serving
  'inference-and-serving/quantization-basics': { Component: QuantizationBasicsLesson, hideTOC: true },
  'inference-and-serving/int8-int4-quantization': { Component: Int8Int4QuantizationLesson, hideTOC: true },
  'inference-and-serving/speculative-decoding': { Component: SpeculativeDecodingLesson, hideTOC: true },
  'inference-and-serving/continuous-batching': { Component: ContinuousBatchingLesson, hideTOC: true },
  'inference-and-serving/paged-attention': { Component: PagedAttentionLesson, hideTOC: true },
}

export function getLessonContent(
  sectionSlug: string,
  lessonSlug: string,
): LessonContent | null {
  return lessonContent[`${sectionSlug}/${lessonSlug}`] ?? null
}
