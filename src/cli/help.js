export const BANNER = `MLFW Tensor Lang

Try:
  x = tensor([[1, 2], [3, 4]])
  w = randn([2, 3])
  y = relu(x @ w)

Type help for examples. Type exit to quit.`;

const HELP = {
  main: `Getting started:
  x = tensor([[1, 2], [3, 4]])
  shape(x)
  y = x * 2 + 1
  x[:, 0]

Neural network:
  model = Sequential(
    Linear(2, 4),
    ReLU(),
    Linear(4, 1),
  )
  output = model(x)

Compiler:
  compiled = compile(model, input=x)
  trace(compiled)

More:
  help tensor
  help model
  help compile
  examples`,

  tensor: `Tensor examples:
  x = tensor([[1, 2], [3, 4]])
  z = zeros([2, 3])
  r = randn([4, 8])

  shape(x)
  x + 1
  x * 2
  x @ randn([2, 3])
  mean(x, axis=0)
  relu(x)
  x[:, 1]
  x[0:2]
  reshape(x, [4])
  transpose(x, 0, 1)

Autograd:
  x = tensor([2], grad=true)
  y = sum(x * x)
  backward(y)
  grad(x)`,

  model: `Neural network examples:
  model = Sequential(
    Linear(2, 4),
    ReLU(),
    Linear(4, 1),
  )

  x = randn([8, 2])
  output = model(x)

Custom model:
  model MLP(input, hidden, output) {
    fc1 = Linear(input, hidden)
    fc2 = Linear(hidden, output)

    forward x {
      x = relu(fc1(x))
      return fc2(x)
    }
  }


  net = MLP(2, 4, 1)
  x = randn([8, 2])
  output = net(x)`,

  compile: `Compile a neural network:
  model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
  x = randn([8, 2])

  compiled = compile(model, input=x)
  trace(compiled)
  graph(compiled)

Options (off by default):
  compile(model, input=x, target=gpu)
  compile(model, input=x, fusion=true)
  compile(model, input=x, scheduling=true, autotune=true)
  compile(model, input=x, quantization=true)
  compile(model, input=x, layout=true, rematerialization=true)
  compile(model, input=x, inplaceReuse=true)
  compile(model, input=x, partition=true)

Targets: cpu (default), gpu, wasm`,
};

const EXAMPLES = {
  tensor: `x = tensor([[1, 2], [3, 4]])
w = randn([2, 3])
y = relu(x @ w)
y`,

  linear: `model = Sequential(
  Linear(2, 4),
  ReLU(),
  Linear(4, 1),
)
x = randn([8, 2])
model(x)`,

  custom: `model MLP(input, hidden, output) {
  fc1 = Linear(input, hidden)
  fc2 = Linear(hidden, output)

  forward x {
    x = relu(fc1(x))
    return fc2(x)
  }
}

model = MLP(2, 4, 1)
x = randn([8, 2])
model(x)`,

  compile: `model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
x = randn([8, 2])
compiled = compile(model, input=x)
trace(compiled)`,
};

export function getHelp(topic = 'main') {
  return HELP[topic] || `Unknown help topic '${topic}'. Try: help tensor, help model, or help compile.`;
}

export function listExamples() {
  return `Available examples:
  tensor       Basic tensor operations
  linear       Sequential neural network
  custom       Custom model with forward
  compile      Compile and inspect a model

Show an example:
  example tensor`;
}

export function getExample(name) {
  return EXAMPLES[name] || `Unknown example '${name}'. Try: examples`;
}

export function handleReplCommand(line) {
  const command = line.trim();
  if (command === 'help') return getHelp();
  if (command.startsWith('help ')) return getHelp(command.slice(5).trim());
  if (command === 'examples') return listExamples();
  if (command.startsWith('example ')) return getExample(command.slice(8).trim());
  return null;
}
