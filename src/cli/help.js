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
  y = compiled(x)
  trace(compiled)

More:
  help tensor
  help model
  help fn
  help control
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
  model MLP(input, hidden, output):
    fc1 = Linear(input, hidden)
    fc2 = Linear(hidden, output)

    forward x:
      x = relu(fc1(x))
      return fc2(x)


  net = MLP(2, 4, 1)
  x = randn([8, 2])
  output = net(x)`,

  fn: `Function examples:
  fn double(x):
    return x * 2
  double(5)

  fn normalize(x):
    return x / sum(x)
  normalize(tensor([3, 1, 2]))

Closure (captures outer variables):
  scale = 10
  fn scaled_add(a, b):
    return (a + b) * scale

Multi-statement:
  fn describe(x):
    print(shape(x))
    print(dtype(x))
    return mean(x)`,

  control: `Conditionals:
  if x > 0:
    print("positive")
  elif x == 0:
    print("zero")
  else:
    print("negative")

For loop:
  for i in range(5):
    print(i)
  for x in [1, 2, 3]:
    total += x

While loop:
  x = 10
  while x > 0:
    x -= 1

Break and continue:
  for i in range(100):
    if i > 10: break
    if i == 5: continue
    print(i)`,

  compile: `Compile and execute a neural network:
  model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
  x = randn([8, 2])

  compiled = compile(model, input=x)
  y = compiled(x)                        # execute compiled model
  trace(compiled)
  graph(compiled)

Lazy compile (no input needed):
  compiled = compile(model)
  y = compiled(x)                        # first call compiles + executes

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

  custom: `model MLP(input, hidden, output):
  fc1 = Linear(input, hidden)
  fc2 = Linear(hidden, output)

  forward x:
    x = relu(fc1(x))
    return fc2(x)

model = MLP(2, 4, 1)
x = randn([8, 2])
model(x)`,

  compile: `model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
x = randn([8, 2])
compiled = compile(model, input=x)
y = compiled(x)
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
