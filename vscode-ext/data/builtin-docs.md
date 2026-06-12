# Tera Built-in Reference

Canonical signatures + descriptions for every Tera built-in. Sole source of truth for the language server. Each entry:

```
## name(param1, param2=default, ...)
Description.
```

Constants (devices/dtypes) omit the parameter list:

```
## name
Description.
```

---

## tensor(data, opts?)
Construct a tensor from a literal value, array, or nested array. Accepts `dtype`, `device`, `grad` options.

## zeros(shape, opts?)
Create a tensor of the given shape filled with `0`.

## ones(shape, opts?)
Create a tensor of the given shape filled with `1`.

## empty(shape, opts?)
Allocate a tensor of the given shape without initializing its contents.

## full(shape, value, opts?)
Create a tensor of the given shape filled with the provided scalar `value`.

## randn(shape, opts?)
Sample a tensor of the given shape from the standard normal distribution.

## arange(start, end?, step?, opts?)
Half-open integer range tensor `[start, end)` with optional `step`.

## eye(n, m?, opts?)
Identity matrix of size `n × m` (or `n × n` if `m` omitted).

## linspace(start, end, steps, opts?)
Evenly spaced values between `start` and `end`, inclusive, with `steps` points.

## randperm(n, opts?)
Random permutation of integers `0..n-1`.

## zerosLike(tensor)
Zero tensor with the same shape, dtype, and device as the input.

## onesLike(tensor)
Tensor of ones with the same shape, dtype, and device as the input.

## emptyLike(tensor)
Uninitialized tensor with the same shape, dtype, and device as the input.

## fullLike(tensor, value)
Constant-filled tensor matching the shape, dtype, and device of the input.

## randnLike(tensor)
Standard-normal sample with the same shape, dtype, and device as the input.

## where(condition, a, b)
Element-wise conditional selection: pick from `a` where `condition` is true, else from `b`.

## cat(tensors, axis=0)
Concatenate tensors along an existing dimension.

## stack(tensors, axis=0)
Stack tensors along a new dimension.

## sum(column) {function}
Aggregate `Column` computing the sum of a column within a `groupBy(...).agg(...)`.

## max(column) {function}
Aggregate `Column` computing the maximum of a column within a `groupBy(...).agg(...)`.

## min(column) {function}
Aggregate `Column` computing the minimum of a column within a `groupBy(...).agg(...)`.

## range(start, stop?, step?)
Integer range: returns an array `[start..stop)` with optional `step`.

## print(...values, sep=" ")
Print one or more values to the runtime output, separated by `sep`.

## trace(compiled)
Print the runtime execution trace of a compiled program.

## graph(compiled)
Print the IR graph of a compiled program or module.

## compile(model, input?, target=cpu, fusion?, scheduling?, debug?)
Compile a model or function to a backend (`cpu`/`gpu`/`wasm`/`webgpu`). `input` provides an example for shape inference and tuning.

## Sequential(...modules)
Compose modules into a feed-forward pipeline. The output of each module is fed to the next.

## Linear(in, out, bias=true)
Fully-connected layer `y = x @ Wᵀ + b`. Set `bias=false` to disable the bias term.

## ReLU()
Rectified Linear Unit activation module: `max(0, x)`.

## GELU()
Gaussian Error Linear Unit activation module — commonly used in Transformers.

## SiLU()
SiLU/Swish activation module: `x * sigmoid(x)`.

## Sigmoid()
Logistic sigmoid activation module.

## Tanh()
Hyperbolic tangent activation module.

## LeakyReLU(negative_slope=0.01)
Leaky ReLU activation; negative inputs are scaled by `negative_slope` instead of zeroed.

## ELU(alpha=1.0)
Exponential Linear Unit activation. Smooth alternative to ReLU for negative values.

## Softmax(dim=-1)
Softmax module over the specified dimension.

## LogSoftmax(dim=-1)
LogSoftmax module — numerically stable log of softmax.

## Flatten(start_dim=1, end_dim=-1)
Flatten a contiguous range of dimensions into one. Typical use: between conv blocks and a Linear head.

## Dropout(p=0.5)
Randomly zero elements with probability `p` during training. Inactive at eval time.

## LayerNorm(shape, eps=1e-5)
Layer normalization over the given trailing shape. Stabilizes activations independent of batch.

## BatchNorm1d(features, eps=1e-5, momentum=0.1)
Batch normalization for 2-D `(N, C)` or 3-D `(N, C, L)` inputs.

## BatchNorm2d(features, eps=1e-5, momentum=0.1)
Batch normalization for 4-D `(N, C, H, W)` image-like inputs.

## Conv1d(in, out, kernel, stride=1, padding=0)
1-D convolution over an input with `in` channels, producing `out` channels.

## Conv2d(in, out, kernel, stride=1, padding=0)
2-D convolution. Use `padding` to preserve spatial dimensions.

## MaxPool2d(kernel, stride?, padding=0)
2-D max pooling. Downsamples spatial dimensions taking the per-window max.

## AvgPool2d(kernel, stride?, padding=0)
2-D average pooling. Downsamples spatial dimensions averaging per window.

## AdaptiveAvgPool2d(output_size)
2-D adaptive average pooling to a target output spatial shape, independent of input size.

## Embedding(num, dim, padding_idx?)
Lookup table mapping integer ids to dense vectors of size `dim`.

## CrossEntropyLoss()
Combined LogSoftmax + NLL loss — standard for multiclass classification.

## MSELoss()
Mean squared error loss — standard for regression.

## NLLLoss()
Negative log-likelihood loss. Pair with LogSoftmax outputs.

## BCELoss()
Binary cross-entropy loss for sigmoid-activated outputs.

## SGD(params, lr=0.01, momentum=0, weight_decay=0)
Stochastic gradient descent with optional `momentum` and `weight_decay`.

## Adam(params, lr=0.001, betas=[0.9, 0.999], weight_decay=0)
Adaptive moment estimation optimizer. Standard default for deep learning.

## AdamW(params, lr=0.001, betas=[0.9, 0.999], weight_decay=0.01)
Adam variant with decoupled weight decay — preferred for transformer-style models.

## StepLR(optimizer, step_size, gamma=0.1)
Decay the learning rate by `gamma` every `step_size` epochs.

## CosineAnnealingLR(optimizer, t_max, eta_min=0)
Cosine schedule decaying the learning rate to `eta_min` over `t_max` epochs.

## ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.1)
Reduce learning rate when a monitored metric stops improving.

## Trainer(max_epochs=20, accelerator="cpu", logger=true, enable_checkpointing=true, enable_progress=true, callbacks?, fast_dev_run=false, gradient_clip_val?, log_every_n_steps=50)
Drives the training loop: epochs, validation, callbacks, logging, checkpointing.

### fit(model, train_loader, val_loader?)
Run the training loop. Iterates `max_epochs` over `train_loader`, optionally validating on `val_loader` each epoch.

### validate(model, loader)
Run validation only (no gradient updates). Returns logged metrics.

### test(model, loader)
Run the model in eval mode over `loader`. Returns logged metrics.

### predict(model, loader)
Run the model in eval mode and collect outputs into an array.

## log(name, value, on_step?, on_epoch?, prog_bar=false, reduce_fx="mean")
Log a metric value from inside `train`/`validate`. Calls `.compute()` automatically on Metric instances.

## optim_config(optimizer, lr_scheduler?)
Wrap an optimizer (and optionally an LR scheduler) for return from an `optimizer:` block.

## TensorDataset(...tensors)
In-memory dataset zipping one or more tensors along their first dimension.

## DataLoader(dataset, batch_size=32, shuffle=true, drop_last=false)
Iterate over a dataset in mini-batches with optional shuffling and `drop_last`.

### len()
Number of batches per epoch.

## load_csv(path, separator=",")
Load a CSV file into a `DataFrame`. Numeric fields are parsed as numbers; use
the `DataFrame` API (`select`, `filter`, `groupBy`, `to_tensor`, `encode`, …)
to analyse it.

## encode(data)
Encode categorical values to integer ids. Returns `[encoded_tensor, classes_array]`.

## decode(indices, classes)
Map integer ids back to original class labels using the `classes` array from `encode`.

## normalize(tensor, axis=0)
Standardize a tensor along `axis`: subtract mean and divide by standard deviation.

## train_test_split(data, test_size=0.2)
Split a tensor into train/test partitions along the first dimension.

## DataFrame(columns) {data}
Build a lazy `DataFrame` from named column arrays, one named argument per
column: `DataFrame(name=["a", "b"], age=[30, 40])`. Column types are inferred
from the values. The frame records a query plan and is only executed when
materialized with `collect`, `toArray`, `count`, `show`, or `chunks`.

## col(column) {function}
Reference a column by name in a `DataFrame` expression, returning a `Column`
that can be transformed and compared. Use a dotted name (`"t.id"`) to qualify a
table alias.

## lit(value) {function}
Wrap a constant value as a `Column` literal so it can be combined with other
columns in expressions.

## expr(sql) {function}
Parse a scalar SQL string into a `Column`, e.g. `expr("price * 1.1")`. Bound
against the frame's schema at build time.

## avg(column) {function}
Aggregate `Column` computing the mean of a column within a `groupBy(...).agg(...)`.

## count(column) {function}
Aggregate `Column` counting non-null values of a column within `agg(...)`.

## countStar() {function}
Aggregate `Column` counting all rows (`COUNT(*)`) within `agg(...)`.

## EarlyStopping(monitor, patience=3, mode="min")
Stop training when a monitored metric stops improving for `patience` evaluations.

## ModelCheckpoint(monitor, save_top_k=1, mode="min")
Save the best model(s) according to a monitored metric.

## ProgressCallback()
Lightweight progress bar callback for the Trainer.

## LearningRateMonitor()
Log the current learning rate at each step.

## Timer()
Measure and log wall-clock time per epoch and total.

## GradientAccumulationScheduler(scheduling)
Accumulate gradients across multiple steps before updating, on a per-epoch schedule.

## ConsoleLogger()
Send log records to stdout.

## CSVLogger(save_dir="logs", name="experiment")
Append log records to a CSV file under `save_dir/name`.

## Accuracy(task="binary", num_classes?, top_k=1)
Classification accuracy metric. Configure with `task` (`binary`/`multiclass`/`multilabel`).

## Precision(task="binary", num_classes?, average="macro")
Precision metric — fraction of positive predictions that are correct.

## Recall(task="binary", num_classes?, average="macro")
Recall metric — fraction of actual positives that are predicted positive.

## F1Score(task="binary", num_classes?, average="macro")
Harmonic mean of precision and recall.

## ConfusionMatrix(num_classes)
Cumulative confusion matrix over `num_classes`.

## MetricCollection(...metrics)
Group multiple metrics into one callable for convenience.

## cpu
CPU execution backend.

## gpu
Native GPU execution backend (CUDA-like).

## wasm
WebAssembly execution backend.

## webgpu
WebGPU execution backend (browser-friendly).

## f16
Half-precision floating-point dtype (16-bit).

## f32
Single-precision floating-point dtype (32-bit). Default for most models.

## f64
Double-precision floating-point dtype (64-bit).

## i32
32-bit signed integer dtype.

## i64
64-bit signed integer dtype.

## bool
Boolean dtype.

---

# Kind templates

These `## @kind/<kind>` entries define methods auto-injected into every builtin
of the matching kind. A builtin's own `###` methods take precedence; templates
add the rest.

## @kind/module

### forward(x)
Run the module's forward pass. Calling the module directly (`module(x)`) is equivalent to `module.forward(x)`.

### parameters()
Return an array of the module's learnable parameter tensors.

### train()
Set the module to training mode (enables Dropout, updates BatchNorm running stats).

### eval()
Set the module to evaluation mode (disables Dropout, freezes BatchNorm stats).

## @kind/sequential

### forward(x)
Run inputs sequentially through each contained module.

### parameters()
Return parameters of all contained modules concatenated.

### train()
Switch all submodules to training mode.

### eval()
Switch all submodules to evaluation mode.

## @kind/optimizer

### step()
Apply one optimizer update step using the current gradients.

### zero_grad()
Zero out gradients of all tracked parameters before the next backward pass.

### param_groups()
Return the list of parameter groups (each with its own learning rate, weight decay, etc.).

## @kind/scheduler

### step(metric?)
Advance the scheduler by one step. Some schedulers (`ReduceLROnPlateau`) require a monitored metric.

### get_last_lr()
Return the most recently computed learning rate(s).

## @kind/metric

### update(preds, target)
Update internal state with a new batch of predictions and ground-truth labels.

### compute()
Compute the current metric value across all accumulated updates.

### reset()
Clear accumulated state so the next epoch starts fresh.

## @kind/callback

### on_train_start(trainer, model)
Hook fired at the start of training.

### on_train_end(trainer, model)
Hook fired at the end of training.

### on_epoch_start(trainer, model)
Hook fired at the start of each epoch.

### on_epoch_end(trainer, model)
Hook fired at the end of each epoch.

## @kind/logger

### log(name, value, step?)
Record a scalar metric value.

### flush()
Flush buffered records to the underlying sink.

## @kind/trainer

### fit(model, train_loader, val_loader?)
Run the full training loop.

### validate(model, loader)
Run validation only.

### test(model, loader)
Run the model in eval mode and report logged metrics.

### predict(model, loader)
Run the model in eval mode and return collected outputs.

# Pseudo-types

These don't correspond to a builtin call but capture the type of common results.

## $Tensor

### shape
Return the shape (size-per-dimension array) of the tensor.

### dtype
Return the dtype string of the tensor.

### reshape(shape) -> Tensor
Return a view with the given shape; total element count must match.

### transpose(dim0, dim1) -> Tensor
Swap two dimensions.

### permute(dims) -> Tensor
Reorder all dimensions per the permutation list.

### expand(shape) -> Tensor
Broadcast to a larger shape without copying memory.

### slice(dim, start, end, step=1) -> Tensor
View a contiguous slice along the given dimension.

### unsqueeze(dim) -> Tensor
Insert a size-1 dimension at the given position.

### squeeze(dim) -> Tensor
Remove a size-1 dimension at the given position.

### narrow(dim, start, length) -> Tensor
Take `length` elements starting at `start` along `dim`.

### select(dim, index) -> Tensor
Select a single index along `dim`, removing that dimension.

### contiguous() -> Tensor
Return a row-major contiguous copy of the tensor.

### detach() -> Tensor
Return a copy detached from the autograd graph.

### backward(gradient?) -> Tensor
Propagate gradients backward from this tensor.

### requires_grad(flag=true) -> Tensor
Enable or disable gradient tracking on this tensor.

### grad
Read the accumulated gradient of this leaf tensor.

### length
Total number of elements (numel).

### neg() -> Tensor
Element-wise unary negation.

### exp() -> Tensor
Element-wise natural exponential `e^x`.

### log() -> Tensor
Element-wise natural logarithm.

### sqrt() -> Tensor
Element-wise square root.

### rsqrt() -> Tensor
Element-wise reciprocal square root `1/√x`.

### abs() -> Tensor
Element-wise absolute value.

### sin() -> Tensor
Element-wise sine.

### cos() -> Tensor
Element-wise cosine.

### tanh() -> Tensor
Element-wise hyperbolic tangent.

### sigmoid() -> Tensor
Element-wise logistic sigmoid `1/(1+e^-x)`.

### relu() -> Tensor
Element-wise ReLU activation.

### gelu() -> Tensor
Gaussian Error Linear Unit activation.

### silu() -> Tensor
SiLU/Swish activation: `x * sigmoid(x)`.

### sign() -> Tensor
Element-wise sign: `-1`, `0`, or `+1`.

### floor() -> Tensor
Element-wise floor (round toward `-∞`).

### ceil() -> Tensor
Element-wise ceiling (round toward `+∞`).

### clone() -> Tensor
Return a deep copy of the tensor (separate storage).

### add(other) -> Tensor
Element-wise addition; scalars are auto-promoted.

### sub(other) -> Tensor
Element-wise subtraction; scalars are auto-promoted.

### mul(other) -> Tensor
Element-wise multiplication; scalars are auto-promoted.

### div(other) -> Tensor
Element-wise division; scalars are auto-promoted.

### pow(exponent) -> Tensor
Element-wise power `x ** exponent`.

### remainder(other) -> Tensor
Element-wise floored remainder (sign follows divisor).

### maximum(other) -> Tensor
Element-wise maximum of two tensors.

### minimum(other) -> Tensor
Element-wise minimum of two tensors.

### eq(other) -> Tensor
Element-wise equality comparison. Returns a boolean tensor.

### ne(other) -> Tensor
Element-wise inequality comparison.

### lt(other) -> Tensor
Element-wise less-than comparison.

### le(other) -> Tensor
Element-wise less-than-or-equal comparison.

### gt(other) -> Tensor
Element-wise greater-than comparison.

### ge(other) -> Tensor
Element-wise greater-than-or-equal comparison.

### matmul(other) -> Tensor
Matrix multiplication; broadcasts on leading batch dimensions.

### dot(other) -> Tensor
Inner (dot) product of two 1-D tensors.

### sum(axis?, keep?) -> Tensor
Sum over `axis` (or the whole tensor); `keep` retains reduced dims.

### mean(axis?, keep?) -> Tensor
Arithmetic mean over `axis` (or the whole tensor); `keep` retains reduced dims.

### max(axis?, keep?) -> Tensor
Maximum over `axis` (or the whole tensor); `keep` retains reduced dims.

### min(axis?, keep?) -> Tensor
Minimum over `axis` (or the whole tensor); `keep` retains reduced dims.

### argmax(axis?, keep?) -> Tensor
Index of the maximum along `axis`; `keep` retains reduced dims.

### argmin(axis?, keep?) -> Tensor
Index of the minimum along `axis`; `keep` retains reduced dims.

### prod(axis?, keep?) -> Tensor
Product of elements over `axis` (or the whole tensor); `keep` retains reduced dims.

### softmax(axis=-1) -> Tensor
Softmax along `axis`, normalizing to a probability distribution.

### log_softmax(axis=-1) -> Tensor
Logarithm of softmax along `axis`, numerically stable.

## $Model

### parameters()
Return the model's learnable parameter tensors.

### forward(*args)
Run the model's forward block. Calling the model directly is equivalent.

### train() -> Model
Set training mode.

### eval() -> Model
Set evaluation mode.

### state_dict()
Return a serializable dict of parameter tensors.

### load_state_dict(state)
Load parameter tensors from a previously saved dict.

## $DataFrame

### columns()
Return the column names as an array of strings.

### schema()
Return the frame's schema (fields with names and data types).

### explain()
Return the logical query plan as a human-readable string.

### select(...columns) -> DataFrame
Project a new frame from the given columns or `Column` expressions.

### filter(condition) -> DataFrame
Keep only rows matching a boolean `Column` (or SQL string) condition.

### where(condition) -> DataFrame
Alias for `filter`.

### withColumn(name, column) -> DataFrame
Return a new frame with an added or replaced column computed from `column`.

### drop(...columns) -> DataFrame
Return a new frame without the named columns.

### groupBy(...columns) -> GroupedData
Group rows by the given columns, returning a `GroupedData` for aggregation.

### orderBy(...specs) -> DataFrame
Sort rows. Each spec is a column name/`Column`, or `{ col, desc }` for ordering.

### sort(...specs) -> DataFrame
Alias for `orderBy`.

### limit(count, offset=0) -> DataFrame
Return at most `count` rows, skipping the first `offset` rows.

### head(n=5) -> DataFrame
Return the first `n` rows as a new frame (pandas-style preview).

### distinct() -> DataFrame
Return a frame with duplicate rows removed.

### union(other) -> DataFrame
Concatenate the rows of another frame with matching column types.

### unionAll(other) -> DataFrame
Concatenate rows of another frame, keeping duplicates.

### join(other, on, how="INNER") -> DataFrame
Join with another frame on one or more key columns. `how` is one of
`INNER`, `LEFT`, `RIGHT`, or `FULL`.

### collect()
Execute the plan and return all rows as an array of objects.

### toArray()
Alias for `collect`.

### count()
Execute the plan and return the number of rows.

### show(n=20)
Execute and print the first `n` rows as a formatted table; returns the text.

### chunks()
Execute and stream results as an async iterator of data chunks.

### to_tensor(...columns)
Materialize the (optionally selected) numeric columns into a 2-D tensor of
shape `[rows, columns]`. Non-numeric columns raise — encode them first.

### to_array()
Alias for `collect` — execute and return rows as an array of objects.

### encode(column, classes?)
Encode a categorical column to integer ids, returning `[encoded_tensor,
classes_array]`. Pass `classes=` to reuse ids fitted on another frame.

## $GroupedData

### agg(...columns) -> DataFrame
Apply aggregate `Column` expressions (e.g. `sum`, `avg`, `count`) over each
group, returning a `DataFrame` of group keys and aggregates.

## $Column

### alias(name) -> Column
Rename the column's output to `name`.

### as(name) -> Column
Alias for `alias`.

### add(other) -> Column
Arithmetic addition with another column or value.

### sub(other) -> Column
Arithmetic subtraction with another column or value.

### mul(other) -> Column
Arithmetic multiplication with another column or value.

### div(other) -> Column
Arithmetic division with another column or value.

### eq(other) -> Column
Equality comparison, producing a boolean column.

### ne(other) -> Column
Inequality comparison, producing a boolean column.

### lt(other) -> Column
Less-than comparison, producing a boolean column.

### le(other) -> Column
Less-than-or-equal comparison, producing a boolean column.

### gt(other) -> Column
Greater-than comparison, producing a boolean column.

### ge(other) -> Column
Greater-than-or-equal comparison, producing a boolean column.

### and(other) -> Column
Logical AND of two boolean columns.

### or(other) -> Column
Logical OR of two boolean columns.

### not() -> Column
Logical negation of a boolean column.

### isNull() -> Column
True where the column value is null.

### isNotNull() -> Column
True where the column value is not null.

### like(pattern) -> Column
SQL `LIKE` match against a string pattern.

### between(low, high) -> Column
True where the value lies in the inclusive range `[low, high]`.

### isin(...values) -> Column
True where the value is one of the given values.

### cast(targetType) -> Column
Cast the column to another data type.
