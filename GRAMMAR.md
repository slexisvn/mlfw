# Tensor Lang Grammar

Tensor Lang is a Python-inspired scripting language for tensor computation and neural network definition. It features indentation-based blocks, first-class tensor operations, and a built-in neural network module system.

## Lexical Structure

### Whitespace and Comments

```
Whitespace  →  ' ' | '\t' | '\r'        (ignored)
Newline     →  '\n' | ';'               (statement separator)
Comment     →  '#' <anything> '\n'
            |  '//' <anything> '\n'
```

Newlines inside brackets `()` `[]` are ignored (implicit line continuation).

### Identifiers and Keywords

```
Identifier  →  [A-Za-z_][A-Za-z0-9_]*
```

**Keywords** (recognized contextually as identifiers):

```
model  forward  fn  if  elif  else  for  in  while
break  continue  return  not  and  or  true  false  null
```

### Literals

```
Number  →  [0-9]+ ('.' [0-9]+)?  ([eE] [+-]? [0-9]+)?
        |  '.' [0-9]+             ([eE] [+-]? [0-9]+)?

String  →  '"' <chars> '"'
        |  "'" <chars> "'"
```

**Escape sequences** in strings: `\n` `\t` `\\` `\"` `\'`

### Operators and Symbols

| Length | Tokens |
|--------|--------|
| 3-char | `**=` |
| 2-char | `**` `==` `!=` `<=` `>=` `+=` `-=` `*=` `/=` `@=` |
| 1-char | `(` `)` `[` `]` `,` `.` `=` `:` `+` `-` `*` `/` `@` `<` `>` |

### Indentation

Tensor Lang uses Python-style indentation for blocks:

- After `:` followed by a newline, increased indentation opens a block (`INDENT` token)
- Decreased indentation closes a block (`DEDENT` token)
- Indentation changes inside brackets are ignored

## Grammar

### Program

```ebnf
Program  →  Statement*
```

### Statements

```ebnf
Statement  →  Assignment
           |  CompoundAssign
           |  IfStmt
           |  ForStmt
           |  WhileStmt
           |  FnDecl
           |  ModelDecl
           |  'break'
           |  'continue'
           |  'return' Expression?
           |  ExpressionStmt

Assignment      →  IDENTIFIER '=' Expression
CompoundAssign  →  IDENTIFIER CompoundOp Expression
CompoundOp      →  '+=' | '-=' | '*=' | '/=' | '**=' | '@='
ExpressionStmt  →  Expression
```

### Control Flow

```ebnf
IfStmt    →  'if' Expression ':' Block ElifClause* ElseClause?
ElifClause →  'elif' Expression ':' Block
ElseClause →  'else' ':' Block

ForStmt   →  'for' IDENTIFIER 'in' Expression ':' Block
WhileStmt →  'while' Expression ':' Block
```

### Blocks

```ebnf
Block  →  ':' Statement                              (* one-line form *)
       |  ':' NEWLINE INDENT Statement+ DEDENT        (* multi-line form *)
```

### Functions

```ebnf
FnDecl     →  'fn' IDENTIFIER ParamList ':' Block
ParamList  →  '(' (IDENTIFIER (',' IDENTIFIER)* ','?)? ')'
```

Functions capture their declaring scope (closures). The return value is the last evaluated expression, or an explicit `return`.

### Models

```ebnf
ModelDecl    →  'model' IDENTIFIER ParamList ':' ModelBody
ModelBody    →  NEWLINE INDENT (Assignment | ForwardDecl)+ DEDENT
ForwardDecl  →  'forward' ForwardParams? ':' Block
ForwardParams → IDENTIFIER (',' IDENTIFIER)*
```

A model must contain exactly one `forward` block. Assignments in the model body define submodule fields.

Note: unlike `fn`, `forward` does **not** use parentheses around its parameters. Parameters are listed directly, separated by commas:

```python
forward x:           # single parameter
forward x, y, z:     # multiple parameters
forward:             # no parameters
```

### Expressions

#### Operator Precedence (lowest to highest)

| Precedence | Operators | Associativity |
|------------|-----------|---------------|
| 1 | `or` | left |
| 2 | `and` | left |
| 3 | `==` `!=` `<` `<=` `>` `>=` | left |
| 4 | `+` `-` | left |
| 5 | `*` `/` `@` | left |
| 6 | `**` | **right** |
| 7 | unary `-` `+` `not` | prefix |
| (postfix) | `()` `[]` `.` | left |

#### Expression Rules

```ebnf
Expression  →  Prefix (BinaryOp Expression)*

Prefix  →  Literal
        |  IDENTIFIER
        |  ArrayLiteral
        |  UnaryExpr
        |  '(' Expression ')'

Literal       →  NUMBER | STRING | 'true' | 'false' | 'null'
ArrayLiteral  →  '[' (Expression (',' Expression)* ','?)? ']'
UnaryExpr     →  ('-' | '+' | 'not') Expression
```

#### Postfix Operations

```ebnf
Call     →  Expression '(' ArgList? ')'
ArgList  →  Argument (',' Argument)* ','?
Argument →  (IDENTIFIER '=')? Expression

Member   →  Expression '.' IDENTIFIER

Index      →  Expression '[' IndexItems ']'
IndexItems →  IndexItem (',' IndexItem)* ','?
IndexItem  →  Slice | Expression

Slice  →  Expression? ':' Expression? (':' Expression?)?
```

#### Binary Operators

| Operator | Operation |
|----------|-----------|
| `+` | Addition (numbers), concatenation (strings), element-wise add (tensors) |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division |
| `**` | Exponentiation |
| `@` | Matrix multiplication |
| `==` `!=` `<` `<=` `>` `>=` | Comparison |
| `and` | Logical AND (short-circuits on scalars, element-wise on tensors) |
| `or` | Logical OR (short-circuits on scalars, element-wise on tensors) |

## Type System

| Type | Description |
|------|-------------|
| Number | Integer or floating-point |
| String | Quoted text |
| Boolean | `true` or `false` |
| Null | `null` |
| Array | Ordered heterogeneous list `[1, "a", true]` |
| Tensor | N-dimensional array with dtype and device |
| Function | User-defined or built-in callable |
| Module | Neural network layer (callable via `forward`) |

Scalars are automatically promoted to tensors when used with tensor operands.

## Indexing and Slicing

Tensor indexing supports multi-dimensional access:

```python
x[0]          # select along dim 0 → removes dimension
x[1:4]        # slice dim 0, indices 1..3
x[::2]        # slice dim 0, step 2
x[0, 1:3]     # select dim 0, then slice dim 1
x[-1]         # negative indexing (from end)
```

- Integer index: selects and removes the dimension (`select`)
- Slice `start:end:step`: keeps the dimension (`slice`)
- Slice bounds default to `0:size:1`

## Named Arguments

Functions accept named arguments using `name=value` syntax. Named arguments are collected into an options object:

```python
zeros([2, 3], dtype=f32, device=gpu)
Linear(784, 128, bias=false)
compile(model, input=x, target=gpu, fusion=true)
```

## Built-in Constants

**Devices:** `cpu` `gpu` `wasm` `webgpu`

**Dtypes:** `f16` `f32` `f64` `i32` `i64` `bool`

## Built-in Functions

### Tensor Creation

| Function | Signature |
|----------|-----------|
| `tensor` | `tensor(data, opts?)` |
| `zeros` | `zeros(shape, opts?)` |
| `ones` | `ones(shape, opts?)` |
| `empty` | `empty(shape, opts?)` |
| `full` | `full(shape, value, opts?)` |
| `randn` | `randn(shape, opts?)` |
| `arange` | `arange(start, end?, step?, opts?)` |
| `eye` | `eye(n, m?, opts?)` |
| `linspace` | `linspace(start, end, steps, opts?)` |
| `randperm` | `randperm(n, opts?)` |
| `zerosLike` | `zerosLike(tensor)` |
| `onesLike` | `onesLike(tensor)` |
| `emptyLike` | `emptyLike(tensor)` |
| `fullLike` | `fullLike(tensor, value)` |
| `randnLike` | `randnLike(tensor)` |

Options: `dtype`, `device`, `grad` (alias for `requiresGrad`).

### Arithmetic and Element-wise

```
add  sub  mul  div  neg  pow  remainder  maximum  minimum
exp  log  sqrt  rsqrt  abs  sin  cos  tanh  sigmoid
relu  gelu  silu  sign  floor  ceil
softmax  log_softmax
```

### Comparison

```
eq  ne  lt  le  gt  ge  where
```

### Linear Algebra

```
matmul  dot  cat  stack  clone
```

### Reductions

| Function | Signature |
|----------|-----------|
| `sum` | `sum(input, axis?, keep?)` |
| `mean` | `mean(input, axis?, keep?)` |
| `max` | `max(input, axis?, keep?)` |
| `min` | `min(input, axis?, keep?)` |
| `argmax` | `argmax(input, axis?, keep?)` |
| `argmin` | `argmin(input, axis?, keep?)` |
| `prod` | `prod(input, axis?, keep?)` |

### Shape Operations

```
reshape(tensor, shape)         transpose(tensor, dim0, dim1)
permute(tensor, dims)          expand(tensor, shape)
slice(tensor, dim, start, end, step=1)
unsqueeze(tensor, dim)         squeeze(tensor, dim)
narrow(tensor, dim, start, length)
select(tensor, dim, index)     contiguous(tensor)
detach(tensor)
```

### Autograd

```
requires_grad(tensor, flag=true)
grad(tensor)
backward(tensor, gradient?)
```

### Utility

```
range(start, stop?, step?)     len(value)
shape(tensor)                  dtype(tensor)
print(value)                   compile(model, ...)
trace(compiled)                graph(compiled)
Sequential(...modules)
```

## Neural Network Modules

```
Linear(in, out, bias=true)          ReLU()  GELU()  SiLU()  Sigmoid()  Tanh()
LeakyReLU(negativeSlope=0.01)       ELU(alpha=1.0)
Softmax(dim=-1)                     LogSoftmax(dim=-1)
Flatten(startDim=1, endDim=-1)      Dropout(p=0.5)
LayerNorm(shape, eps=1e-5)          BatchNorm1d(features, eps=1e-5, momentum=0.1)
BatchNorm2d(features, eps=1e-5, momentum=0.1)
Conv1d(in, out, kernel, stride=1, padding=0)
Conv2d(in, out, kernel, stride=1, padding=0)
MaxPool2d(kernel, stride?, padding=0)
AvgPool2d(kernel, stride?, padding=0)
AdaptiveAvgPool2d(outputSize)
Embedding(num, dim, paddingIdx?)
CrossEntropyLoss()  MSELoss()  NLLLoss()  BCELoss()
```

## Examples

### Basic Tensor Operations

```python
x = tensor([[1, 2], [3, 4]], dtype=f32)
y = ones([2, 2])
z = x @ y + 1                # matmul then broadcast add
print(sum(z, axis=0))
```

### Function Definition

```python
fn softmax_manual(x):
  e = exp(x - max(x))
  return e / sum(e)
```

### Model Definition

```python
model MLP(hidden):
  fc1 = Linear(784, hidden)
  fc2 = Linear(hidden, 10)
  forward x:
    x = relu(fc1(x))
    return fc2(x)

net = MLP(128)
out = net(randn([1, 784]))
```

### Control Flow

```python
fn fibonacci(n):
  a = 0
  b = 1
  for i in range(n):
    temp = b
    b = a + b
    a = temp
  return a
```

### Compilation

```python
model Net():
  fc = Linear(4, 2)
  forward x:
    return fc(x)

net = Net()
compiled = compile(net, input=randn([1, 4]), target=cpu)
```

## Deep Model Examples

### Cross-Attention (multi-param forward)

`forward` accepts multiple parameters, enabling models that take separate inputs:

```python
model CrossAttention(d):
  wq = Linear(d, d)
  wk = Linear(d, d)
  wv = Linear(d, d)

  forward query, key, value:
    q = wq(query)
    k = wk(key)
    v = wv(value)
    scores = softmax(q @ transpose(k, 0, 1))
    return scores @ v

attn = CrossAttention(64)
out = attn(randn([8, 64]), randn([16, 64]), randn([16, 64]))
```

### Transformer Encoder Block

Single-layer transformer with self-attention, residual connections, and layer normalization:

```python
model TransformerBlock(d):
  wq = Linear(d, d)
  wk = Linear(d, d)
  wv = Linear(d, d)
  wo = Linear(d, d)
  ln1 = LayerNorm([d])
  ff1 = Linear(d, d * 4)
  ff2 = Linear(d * 4, d)
  ln2 = LayerNorm([d])

  forward x:
    q = wq(x)
    k = wk(x)
    v = wv(x)
    scores = q @ transpose(k, 0, 1)
    attn = softmax(scores)
    ctx = attn @ v
    h = ln1(x + wo(ctx))
    return ln2(h + ff2(gelu(ff1(h))))

net = TransformerBlock(64)
out = net(randn([16, 64]))
compile(net, input=randn([16, 64]))
```

### Stacked Transformer Encoder

Multi-layer encoder with composable sub-models:

```python
model FFN(d):
  fc1 = Linear(d, d * 2)
  fc2 = Linear(d * 2, d)
  forward x:
    return fc2(gelu(fc1(x)))

model EncoderLayer(d):
  wq = Linear(d, d)
  wk = Linear(d, d)
  wv = Linear(d, d)
  ln1 = LayerNorm([d])
  ffn = FFN(d)
  ln2 = LayerNorm([d])

  forward x:
    q = wq(x)
    k = wk(x)
    v = wv(x)
    attn = softmax(q @ transpose(k, 0, 1)) @ v
    h = ln1(x + attn)
    return ln2(h + ffn(h))

model Encoder(d, n):
  layer1 = EncoderLayer(d)
  layer2 = EncoderLayer(d)

  forward x:
    x = layer1(x)
    return layer2(x)

net = Encoder(64, 2)
out = net(randn([16, 64]))
compile(net, input=randn([16, 64]))
```

### ResNet-style CNN

Convolutional network with residual blocks, batch normalization, and pooling:

```python
model ResBlock(ch):
  conv1 = Conv2d(ch, ch, 3, padding=1)
  bn1 = BatchNorm2d(ch)
  conv2 = Conv2d(ch, ch, 3, padding=1)
  bn2 = BatchNorm2d(ch)

  forward x:
    h = relu(bn1(conv1(x)))
    h = bn2(conv2(h))
    return relu(x + h)

model ResNet(num_classes):
  stem = Conv2d(3, 32, 3, padding=1)
  bn0 = BatchNorm2d(32)
  block1 = ResBlock(32)
  block2 = ResBlock(32)
  pool = AdaptiveAvgPool2d([1, 1])
  flat = Flatten()
  fc = Linear(32, num_classes)

  forward x:
    x = relu(bn0(stem(x)))
    x = block1(x)
    x = block2(x)
    x = flat(pool(x))
    return fc(x)

net = ResNet(10)
out = net(randn([4, 3, 32, 32]))
compile(net, input=randn([4, 3, 32, 32]))
```

### Autoencoder

Encoder-decoder architecture with bottleneck:

```python
model Encoder(input_dim, latent_dim):
  fc1 = Linear(input_dim, 256)
  fc2 = Linear(256, 128)
  fc3 = Linear(128, latent_dim)

  forward x:
    x = relu(fc1(x))
    x = relu(fc2(x))
    return fc3(x)

model Decoder(latent_dim, output_dim):
  fc1 = Linear(latent_dim, 128)
  fc2 = Linear(128, 256)
  fc3 = Linear(256, output_dim)

  forward z:
    z = relu(fc1(z))
    z = relu(fc2(z))
    return sigmoid(fc3(z))

model Autoencoder(input_dim, latent_dim):
  encoder = Encoder(input_dim, latent_dim)
  decoder = Decoder(latent_dim, input_dim)

  forward x:
    z = encoder(x)
    return decoder(z)

ae = Autoencoder(784, 32)
out = ae(randn([8, 784]))
```

### Sequential Pipeline with Loss

Using `Sequential` for quick model assembly:

```python
classifier = Sequential(
  Linear(784, 256),
  ReLU(),
  Dropout(p=0.3),
  Linear(256, 128),
  ReLU(),
  Dropout(p=0.3),
  Linear(128, 10)
)

x = randn([32, 784])
logits = classifier(x)
probs = softmax(logits)
compiled = compile(classifier, input=x, target=cpu, debug=true)
```

### Multi-Head Attention (manual)

Explicit multi-head attention with head splitting via tensor operations:

```python
model MultiHeadAttention(d, heads):
  wq = Linear(d, d)
  wk = Linear(d, d)
  wv = Linear(d, d)
  wo = Linear(d, d)

  forward x:
    q = wq(x)
    k = wk(x)
    v = wv(x)
    scores = q @ transpose(k, 0, 1)
    scale = sqrt(tensor(d, dtype=f32))
    attn = softmax(scores / scale)
    return wo(attn @ v)

model TransformerEncoder(d, heads, layers):
  attn1 = MultiHeadAttention(d, heads)
  ln1 = LayerNorm([d])
  ff1_up = Linear(d, d * 4)
  ff1_down = Linear(d * 4, d)
  ln2 = LayerNorm([d])
  attn2 = MultiHeadAttention(d, heads)
  ln3 = LayerNorm([d])
  ff2_up = Linear(d, d * 4)
  ff2_down = Linear(d * 4, d)
  ln4 = LayerNorm([d])

  forward x:
    # Layer 1
    h = ln1(x + attn1(x))
    h = ln2(h + ff1_down(gelu(ff1_up(h))))
    # Layer 2
    h = ln3(h + attn2(h))
    h = ln4(h + ff2_down(gelu(ff2_up(h))))
    return h

net = TransformerEncoder(64, 4, 2)
out = net(randn([16, 64]))
```

### Text Classification Model

Embedding + transformer + classification head:

```python
model TextClassifier(vocab, d, num_classes):
  embed = Embedding(vocab, d)
  wq = Linear(d, d)
  wk = Linear(d, d)
  wv = Linear(d, d)
  ln1 = LayerNorm([d])
  ff1 = Linear(d, d * 4)
  ff2 = Linear(d * 4, d)
  ln2 = LayerNorm([d])
  head = Linear(d, num_classes)

  forward x:
    x = embed(x)
    q = wq(x)
    k = wk(x)
    v = wv(x)
    attn = softmax(q @ transpose(k, 0, 1)) @ v
    h = ln1(x + attn)
    h = ln2(h + ff2(gelu(ff1(h))))
    return head(mean(h, axis=0, keep=true))
```

### Autograd and Training Loop

Manual gradient computation with tensor operations:

```python
w = randn([2, 1], grad=true)
b = zeros([1], grad=true)

for i in range(100):
  x = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  y = tensor([[5.0], [11.0], [17.0]])
  pred = x @ w + b
  loss = mean((pred - y) ** 2)
  backward(loss)
  w_grad = grad(w)
  b_grad = grad(b)
  w = requires_grad(w - 0.01 * w_grad)
  b = requires_grad(b - 0.01 * b_grad)

print(w)
print(b)
```

### Compilation Targets

Compiling models to different backends:

```python
model Net():
  fc = Linear(4, 2)
  forward x:
    return fc(x)

net = Net()
x = randn([1, 4])

cpu_compiled = compile(net, input=x, target=cpu)
gpu_compiled = compile(net, input=x, target=gpu)
wasm_compiled = compile(net, input=x, target=wasm)
webgpu_compiled = compile(net, input=x, target=webgpu)

# With optimization options
optimized = compile(net, input=x, target=cpu, fusion=true, scheduling=true, debug=true)
```
