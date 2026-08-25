export type PassNote = {
  summary: string;
  chapter: string;
  chapterTitle: string;
};

const BOOK = 'https://github.com/slexisvn/mlfw/tree/main/docs';

export const PASS_NOTES: Record<string, PassNote> = {
  CallInlinerPass: {
    summary: 'Pastes called functions into their caller so later passes see one flat graph.',
    chapter: 'part3/ch14-what-a-pass-is',
    chapterTitle: 'What a pass is',
  },
  DecompositionPass: {
    summary: 'Rewrites big ops into primitive ones: softmax becomes exp, sum and divide.',
    chapter: 'part4/ch21-decomposition',
    chapterTitle: 'Decomposition',
  },
  canonicalize: {
    summary: 'Puts equivalent programs in one agreed shape, and folds every identity an op declares about itself — x * 1, neg(neg(x)).',
    chapter: 'part4/ch20-algebra-and-ieee754',
    chapterTitle: 'Algebra and IEEE-754',
  },
  algebraic_simplify: {
    summary: 'Rewrites that pair two different ops — transpose(transpose(x)), (-a) * (-b), a + (-b). It picks up the single-op identities too, when canonicalize is off.',
    chapter: 'part4/ch20-algebra-and-ieee754',
    chapterTitle: 'Algebra and IEEE-754',
  },
  constant_fold: {
    summary: 'Evaluates anything whose inputs are already known at compile time.',
    chapter: 'part4/ch19-fold-cse-dce',
    chapterTitle: 'Fold, CSE, DCE',
  },
  cse: {
    summary: 'Finds two ops computing the same thing and keeps one of them.',
    chapter: 'part4/ch19-fold-cse-dce',
    chapterTitle: 'Fold, CSE, DCE',
  },
  dce: {
    summary: 'Deletes every op whose result nobody reads.',
    chapter: 'part4/ch19-fold-cse-dce',
    chapterTitle: 'Fold, CSE, DCE',
  },
  FusionPass: {
    summary: 'Groups neighbouring ops into one kernel so intermediates stay in registers.',
    chapter: 'part4/ch22-fusion-why',
    chapterTitle: 'Why fusion',
  },
  PriorityFusionPass: {
    summary: 'Fuses whole-graph by benefit: the most profitable merge goes first, then the queue is rescored.',
    chapter: 'part4/ch24-fusion-strategies',
    chapterTitle: 'Fusion strategies',
  },
  DominatorFusionPass: {
    summary: 'Fuses along dominator relationships in the graph rather than by measured benefit.',
    chapter: 'part4/ch24-fusion-strategies',
    chapterTitle: 'Fusion strategies',
  },
  EpilogueFusionPass: {
    summary: 'Glues cheap elementwise work onto the tail of a heavy op like matmul.',
    chapter: 'part4/ch23-fusion-legality',
    chapterTitle: 'Fusion legality',
  },
  FusionMergerPass: {
    summary: 'Merges fusion groups that turned out to be adjacent, up to the size budget.',
    chapter: 'part4/ch24-fusion-strategies',
    chapterTitle: 'Fusion strategies',
  },
  MultiOutputFusionPass: {
    summary: 'Lets one fused kernel produce several results instead of splitting the group.',
    chapter: 'part4/ch24-fusion-strategies',
    chapterTitle: 'Fusion strategies',
  },
  LayoutTransformPass: {
    summary: 'Chooses memory layouts and inserts the transposes the choice implies.',
    chapter: 'part4/ch25-layout',
    chapterTitle: 'Layout',
  },
  QuantizationPass: {
    summary: 'Swaps float ops for integer ones plus the quantize/dequantize pairs around them.',
    chapter: 'part4/ch26-optional-pipelines',
    chapterTitle: 'Optional pipelines',
  },
  CalibrationPass: {
    summary: 'Runs sample data through the graph to learn the ranges quantization needs.',
    chapter: 'part4/ch26-optional-pipelines',
    chapterTitle: 'Optional pipelines',
  },
  MixedPrecisionPass: {
    summary: 'Demotes ops to a narrower float where the numerics survive it.',
    chapter: 'part4/ch26-optional-pipelines',
    chapterTitle: 'Optional pipelines',
  },
  RematerializationPass: {
    summary: 'Trades compute for memory: recompute a value later instead of holding it live.',
    chapter: 'part5/ch30-memory-for-recomputation',
    chapterTitle: 'Memory for recomputation',
  },
  GraphPartitionPass: {
    summary: 'Splits the graph across targets when more than one device is in play.',
    chapter: 'part4/ch26-optional-pipelines',
    chapterTitle: 'Optional pipelines',
  },
  PartitionMaterializationPass: {
    summary: 'Turns each partition into its own function with explicit boundaries.',
    chapter: 'part4/ch26-optional-pipelines',
    chapterTitle: 'Optional pipelines',
  },
  LegalizeConstBuffersPass: {
    summary: 'Rewrites constant buffers into a form the target can actually link.',
    chapter: 'part6/ch34-lowering-rules',
    chapterTitle: 'Lowering rules',
  },
  InlineReindexPass: {
    summary: 'Folds index-only blocks into their consumers so no loop nest exists just to renumber.',
    chapter: 'part6/ch35-index-arithmetic',
    chapterTitle: 'Index arithmetic',
  },
  AutoTensorizePass: {
    summary: 'Spots loop nests shaped like a hardware intrinsic and swaps in the intrinsic.',
    chapter: 'part7/ch43-scheduling-for-gpus',
    chapterTitle: 'Scheduling for GPUs',
  },
  SchedulePass: {
    summary: 'Decides how each loop nest runs: tiling, order, threads — the how, not the what.',
    chapter: 'part7/ch38-separating-what-from-how',
    chapterTitle: 'Separating what from how',
  },
  LoopPartitionPass: {
    summary: 'Splits a loop so the boundary iterations stop paying for a guard on every step.',
    chapter: 'part6/ch37-proving-things-about-indices',
    chapterTitle: 'Proving things about indices',
  },
  SimplifyPass: {
    summary: 'Proves index expressions and drops guards it can show are always true.',
    chapter: 'part6/ch37-proving-things-about-indices',
    chapterTitle: 'Proving things about indices',
  },
  MemorySchedulePass: {
    summary: 'Reorders independent work to lower the peak number of buffers alive at once.',
    chapter: 'part9/ch52-scheduling-for-peak',
    chapterTitle: 'Scheduling for peak',
  },
  MemoryPlanPass: {
    summary: 'Assigns buffers to storage, reusing space whose lifetimes do not overlap.',
    chapter: 'part9/ch49-buffer-lifetimes',
    chapterTitle: 'Buffer lifetimes',
  },
  AccumulatorDetectionPass: {
    summary: 'Recognises a reduction loop and keeps its running total in a register.',
    chapter: 'part10/ch53-lir-the-third-ir',
    chapterTitle: 'LIR, the third IR',
  },
  FlatIndexSimplifyPass: {
    summary: 'Cleans up the flat address arithmetic left behind by lowering to LIR.',
    chapter: 'part10/ch53-lir-the-third-ir',
    chapterTitle: 'LIR, the third IR',
  },
};

export function passNote(name: string): PassNote | null {
  return PASS_NOTES[name] ?? null;
}

export function chapterUrl(note: PassNote): string {
  return `${BOOK}/${note.chapter}`;
}
