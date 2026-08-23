export type Example = {
  id: string;
  title: string;
  blurb: string;
  source: string;
};

export const EXAMPLES: Example[] = [
  {
    id: 'mlp',
    title: 'MLP',
    blurb: 'Two linear layers and a ReLU — the smallest graph that still fuses.',
    source: `const model = new Sequential(
  new Linear(8, 16),
  new ReLU(),
  new Linear(16, 4),
);

const x = randn([1, 8]);

run(model, [x]);
`,
  },
  {
    id: 'dead-code',
    title: 'Dead code',
    blurb: 'A branch nobody reads, so you can watch DCE delete it.',
    source: `const forward = (a, b) => {
  const unused = a.sub(b).exp();
  const scaled = a.mul(b);
  return scaled.add(a).tanh();
};

const a = randn([64]);
const b = randn([64]);

run(forward, [a, b]);
`,
  },
  {
    id: 'algebra',
    title: 'Algebraic identities',
    blurb: 'x + 0, x * 1, and a double negation for the simplifier to eat.',
    source: `const forward = (x) => {
  const zero = zeros([32]);
  const one = ones([32]);
  const shifted = x.add(zero);
  const scaled = shifted.mul(one);
  return scaled.neg().neg().relu();
};

const x = randn([32]);

run(forward, [x]);
`,
  },
  {
    id: 'norm',
    title: 'LayerNorm block',
    blurb: 'A normalization layer decomposes into many small ops, then fuses back.',
    source: `const model = new Sequential(
  new Linear(32, 32),
  new LayerNorm([32]),
  new GELU(),
);

const x = randn([4, 32]);

run(model, [x]);
`,
  },
  {
    id: 'attention',
    title: 'Attention head',
    blurb: 'Projections, a matmul pair and a softmax — a real transformer shape.',
    source: `class Head extends Module {
  constructor(dim) {
    super();
    this.q = new Linear(dim, dim);
    this.k = new Linear(dim, dim);
    this.v = new Linear(dim, dim);
    this.scale = 1 / Math.sqrt(dim);
  }

  forward(x) {
    const q = this.q.forward(x);
    const k = this.k.forward(x);
    const v = this.v.forward(x);
    const scores = q.matmul(k.transpose(-2, -1)).mul(this.scale);
    return softmax(scores, -1).matmul(v);
  }
}

const model = new Head(16);
const x = randn([2, 8, 16]);

run(model, [x]);
`,
  },
  {
    id: 'conv',
    title: 'Conv stack',
    blurb: 'Convolution, normalization and pooling, on the way to a tiled kernel.',
    source: `const model = new Sequential(
  new Conv2d(3, 8, 3, { padding: 1 }),
  new BatchNorm2d(8),
  new ReLU(),
  new MaxPool2d(2),
);

const x = randn([1, 3, 16, 16]);

run(model, [x]);
`,
  },
];
