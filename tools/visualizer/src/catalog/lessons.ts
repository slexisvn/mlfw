import type { CompileOptions } from '../protocol.js';
import type { StageTab } from './glossary.js';

export type Question = {
  asks: string;
  choices: string[];
  answer: number;
  because: string;
};

export type Beat = {
  title: string;
  says: string;
  source?: string;
  options?: Partial<CompileOptions>;
  run?: boolean;
  focus?: { pass: string; tab: StageTab };
  tab?: StageTab;
  asks?: Question;
};

export type Lesson = {
  id: string;
  title: string;
  blurb: string;
  beats: Beat[];
};

const MLP = `const model = new Sequential(
  new Linear(64, 128),
  new ReLU(),
  new Linear(128, 64),
);

const x = randn([32, 64]);

run(model, [x]);
`;

const DEAD = `const forward = (a, b) => {
  const unused = a.sub(b).exp();
  const scaled = a.mul(b);
  return scaled.add(a).tanh();
};

const a = randn([256]);
const b = randn([256]);

run(forward, [a, b]);
`;

const MATMUL = `const forward = (a, b) => a.matmul(b).tanh();

const a = randn([128, 128]);
const b = randn([128, 128]);

run(forward, [a, b]);
`;

export const LESSONS: Lesson[] = [
  {
    id: 'dce',
    title: 'What a pass is',
    blurb: 'One pass, one job, one verdict. Watch dead code elimination decide.',
    beats: [
      {
        title: 'a branch nobody reads',
        says: 'This program computes three things and returns one of them. Nothing reads `unused`, and nothing ever will — it is dead the moment tracing finishes. Run it and watch which pass notices.',
        source: DEAD,
        options: { fusion: false, scheduling: false, backward: 'off', autotune: false },
        run: true,
        tab: 'ir',
      },
      {
        title: 'predict before you look',
        says: 'Dead code elimination is about to run. It deletes an op when nothing reads its result and the op declares no side effect. `unused = a.sub(b).exp()` is two ops.',
        asks: {
          asks: 'How many ops does dce delete on this graph?',
          choices: ['none — the values are still in the code you wrote', 'one — the exp', 'two — the exp and the sub', 'three — it takes the mul too'],
          answer: 2,
          because: 'It deletes the exp first. That leaves the sub with no reader, so the sub goes on the next turn of the same worklist — cascading is why one deletion is never the whole story. The mul survives: `scaled` is read.',
        },
      },
      {
        title: 'the verdict, and the reason',
        says: 'Open Why. Each deletion now says which of the two cases it was — nothing ever read it, or its only reader was deleted a moment ago. That distinction is the whole algorithm.',
        focus: { pass: 'dce', tab: 'why' },
      },
      {
        title: 'turn it off',
        says: 'Press ⊘ next to dce in the timeline. The pipeline runs again without it, and the dead ops go all the way through lowering into the kernel — the compiler will happily generate code nobody calls. That is what the pass was buying you.',
      },
    ],
  },
  {
    id: 'fusion',
    title: 'Why fusion',
    blurb: 'Fusion does not remove arithmetic. Measure what it does remove.',
    beats: [
      {
        title: 'two layers and a relu',
        says: 'Every op here reads its inputs from memory and writes its result back. Run it with fusion off first, and look at the Result tab: the roofline numbers tell you what this program is actually spending its time on.',
        source: MLP,
        options: { fusion: false, scheduling: true, backward: 'off', autotune: false },
        run: true,
        tab: 'result',
      },
      {
        title: 'pin it',
        says: 'Open Compare and pin this run. Everything from here is measured against it.',
        tab: 'compare',
      },
      {
        title: 'predict before you look',
        says: 'Now turn fusion on. The graph will keep doing exactly the same arithmetic — the same multiplies, the same adds, the same relu.',
        asks: {
          asks: 'With fusion on, what should change?',
          choices: ['fewer flops', 'fewer bytes moved', 'both', 'neither — it only reorders code'],
          answer: 1,
          because: 'Fusion never removes arithmetic. It removes the round trip to memory between two ops: the intermediate stays in a register instead of being written out and read back. Flops stay put, bytes drop, and arithmetic intensity — flops per byte — goes up.',
        },
      },
      {
        title: 'see it',
        says: 'Fusion is on and the run is done. Compare shows bytes moved and arithmetic intensity against the pinned run. On a model this small the wall clock may not move much — kernel launch overhead dominates. The traffic number is the honest one.',
        options: { fusion: true },
        run: true,
        tab: 'compare',
      },
      {
        title: 'the decision itself',
        says: 'Find PriorityFusionPass in the timeline and open Why. Every group it formed is there with the reason, and so is every pair it refused — the refusals are where fusion legality lives.',
        focus: { pass: 'PriorityFusionPass', tab: 'why' },
      },
    ],
  },
  {
    id: 'schedule',
    title: 'Separating what from how',
    blurb: 'The loop nest says what. Schedule primitives say how, one at a time.',
    beats: [
      {
        title: 'a matmul, lowered',
        says: 'A matmul and a tanh. After lowering, the graph is gone and loop nests have taken its place — the same arithmetic, now with an explicit order of execution.',
        source: MATMUL,
        options: { fusion: true, scheduling: true, backward: 'off', autotune: false },
        run: true,
        tab: 'graph',
      },
      {
        title: 'one primitive at a time',
        says: 'SchedulePass applies a whole sequence at once. The rows indented under it are that sequence replayed one call at a time, on the nest as it stood before the pass ran. Step through them and watch the nest change shape.',
        focus: { pass: 'SchedulePass', tab: 'graph' },
      },
      {
        title: 'predict before you look',
        says: 'One of those primitives is `split`. It cuts one loop into two: an outer loop over tiles and an inner loop inside a tile.',
        asks: {
          asks: 'What does split change about the program?',
          choices: ['it computes fewer elements', 'it computes the same elements in a different order', 'it computes the same elements in the same order', 'it makes the program parallel'],
          answer: 1,
          because: 'Split alone touches every element exactly once, and in the same sequence — it only changes which loop the index comes from. It is what comes after that matters: once the loops are separate, reorder can put them in a different order, and bind can hand one of them to a thread. Split is the enabling move, not the win.',
        },
      },
      {
        title: 'the reason it fired',
        says: 'Each primitive row explains what that call bought. What none of them can tell you is whether the sizes were right — that is the next lesson.',
      },
    ],
  },
  {
    id: 'memory',
    title: 'Buffer lifetimes',
    blurb: 'Peak memory is not the sum of what you allocate.',
    beats: [
      {
        title: 'the temporaries',
        says: 'This program makes several intermediate tensors. Open Memory: each bar is one buffer, drawn across the statements it is alive for.',
        source: MLP,
        options: { fusion: false, scheduling: true, backward: 'off', autotune: false },
        run: true,
        tab: 'memory',
      },
      {
        title: 'predict before you look',
        says: 'Two buffers can share one slot exactly when their lifetimes never overlap.',
        asks: {
          asks: 'What does peak memory depend on?',
          choices: ['how many temporaries there are', 'how big the biggest one is', 'how many are alive at the same moment', 'the order the passes ran in'],
          answer: 2,
          because: 'The peak is the worst moment, not the total. A chain of ten temporaries where each dies before the next is born fits in one slot. Which is why statement order matters: moving a producer next to its consumer ends a lifetime sooner, and that is exactly what the memory scheduler tries to do.',
        },
      },
      {
        title: 'the gap that is left',
        says: 'The chart draws three lines: what is genuinely live, what the plan reserves, and what one address per buffer would have cost. The distance between the first two is fragmentation — space a live buffer is sitting above that cannot be handed out.',
        tab: 'memory',
      },
    ],
  },
  {
    id: 'autodiff',
    title: 'What the backward pass costs',
    blurb: 'Differentiating a graph is mechanical. Deciding what to keep is not.',
    beats: [
      {
        title: 'a training step',
        says: 'Switch Direction to train. The compiler differentiates the traced graph by rule, then compiles the result — so the timeline now runs twice, once for the forward and once for the backward.',
        source: MLP,
        options: { backward: 'separate', fusion: true, scheduling: true, autotune: false },
        run: true,
        tab: 'result',
      },
      {
        title: 'the evidence',
        says: 'The Result tab compares the compiled gradients against the same step run through eager autograd. That number is the only thing standing between a derivation that is right and one that merely looks right.',
        tab: 'result',
      },
      {
        title: 'predict before you look',
        says: 'The backward needs forward values. It can either hold them in memory from the forward pass, or recompute them when it gets there.',
        asks: {
          asks: 'Which values are worth recomputing instead of keeping?',
          choices: ['the big ones', 'the cheap ones', 'the ones near the output', 'none — recomputing is always slower'],
          answer: 1,
          because: 'Cheap ones. Keeping a value costs memory for the whole forward pass; recomputing it costs its own arithmetic once. For a neg or an abs that trade is obviously worth it, for a matmul it obviously is not, and everything in between is what a remat policy is for.',
        },
      },
      {
        title: 'read the decisions',
        says: 'Open Why on any pass in the backward half and look for the autodiff entries: each forward value is listed as kept alive or recomputed, with the reason. The forward graph also grew — the extra outputs are the saved values being handed across.',
      },
    ],
  },
];

export function lessonById(id: string): Lesson | null {
  return LESSONS.find(lesson => lesson.id === id) ?? null;
}
