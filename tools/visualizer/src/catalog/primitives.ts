export type PrimitiveNote = { decision: string; reason: string };

export const PRIMITIVE_NOTES: Record<string, PrimitiveNote> = {
  split: {
    decision: 'cut one loop into an outer and an inner loop',
    reason: 'the inner loop now covers a tile small enough to stay in cache, and the outer one walks between tiles',
  },
  fuseLoops: {
    decision: 'welded two nested loops into one',
    reason: 'a single flat loop has one induction variable to bind to a thread, and one bound to check instead of two',
  },
  reorder: {
    decision: 'changed which loop is outermost',
    reason: 'the loop order decides the order addresses are touched, and only one order walks memory in the direction it is laid out',
  },
  tile: {
    decision: 'split every named loop and interleaved the halves',
    reason: 'this is split plus reorder in one step: the outer loops walk tiles, the inner ones walk inside a tile',
  },
  vectorize: {
    decision: 'marked a loop to run its iterations as one wide instruction',
    reason: 'the iterations write neighbouring addresses and never read each other, so the hardware can do several per instruction',
  },
  unroll: {
    decision: 'marked a loop to be written out iteration by iteration',
    reason: 'with the trip count known, dropping the loop removes the counter and the branch, and lets the scheduler mix the bodies',
  },
  parallelize: {
    decision: 'marked a loop to run its iterations on separate cores',
    reason: 'no iteration reads what another writes, so they can run in any order and therefore at the same time',
  },
  bindThread: {
    decision: 'bound a loop to a hardware thread axis',
    reason: 'on a GPU the loop does not run — the launch geometry does, so each iteration becomes one thread',
  },
  computeAt: {
    decision: 'moved a producer inside its consumer loop',
    reason: 'computing the producer where it is read keeps the value in registers instead of in a buffer between two loop nests',
  },
  reverseComputeAt: {
    decision: 'moved a consumer inside its producer loop',
    reason: 'the consumer now runs on each tile as it is produced, so the producer never has to materialise the whole tensor',
  },
  computeInline: {
    decision: 'dissolved a producer into every place that reads it',
    reason: 'the producer is cheap elementwise work, so recomputing it per reader beats storing it and loading it back',
  },
  computeInlineBlock: {
    decision: 'dissolved a producer block into its readers',
    reason: 'the producer is cheap elementwise work, so recomputing it per reader beats storing it and loading it back',
  },
  cacheRead: {
    decision: 'staged an input into a faster scope',
    reason: 'the loop reads the same values many times, so one copy into shared memory pays for itself over the reuses',
  },
  cacheWrite: {
    decision: 'staged an output through a faster scope',
    reason: 'accumulating in a local buffer and writing once at the end turns many slow stores into one',
  },
  setScope: {
    decision: 'moved a buffer to another memory scope',
    reason: 'the scope decides who can see the buffer and how fast it is, and this buffer is only read where the faster scope reaches',
  },
  storageAlign: {
    decision: 'padded a buffer stride',
    reason: 'a stride that is a multiple of the bank count makes every thread in a warp hit the same bank, and the padding breaks that up',
  },
  rfactor: {
    decision: 'split one reduction into partial sums plus a final combine',
    reason: 'partial sums are independent, so the reduction can run in parallel — at the cost of a different summation order',
  },
  decomposeReduction: {
    decision: 'lifted the initialization out of the reduction loop',
    reason: 'the init writes once and the update runs every iteration, so separating them lets each be scheduled on its own',
  },
  fuseConsumer: {
    decision: 'pulled a consumer block into the producer loop nest',
    reason: 'one loop nest producing and consuming means the intermediate never reaches memory',
  },
  annotate: {
    decision: 'attached a hint to a loop',
    reason: 'the annotation carries a decision that codegen, not the scheduler, has to act on',
  },
  blockize: {
    decision: 'wrapped a loop nest into one block',
    reason: 'a block is the unit an intrinsic can replace, so the nest has to become one before tensorization can match it',
  },
};
