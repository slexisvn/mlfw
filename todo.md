# TODO — bug hunt compiler

## Trạng thái hiện tại (2026-06-10)
- Fuzzer `tests/e2e/fuzz-differential.test.js` (seeded, 200 prog × cpu+wasm) đang xanh.
- Đợt vừa rồi fix: WASM i32 `abs` (emit f32.abs), i32 max/min reduce-init (±Infinity → INT_MIN/MAX theo dtype).
  - Code: `src/backend/dtype_map.js` (`reduceInitValue` + `INT_RANGE`), `src/tracing/tracer.js`, `src/dispatcher/jit_cache.js`.
  - Test: `tests/e2e/differential-nn.test.js` (+max_i32_neg, +min_i32_pos), fuzz-differential.test.js.
- **Chưa commit git.** Full e2e + AD: 413 passed.

### Fix batched/broadcast matmul (P0 rank≠2) — đã xong
- Bug: matmul N-D sai cả shape lẫn giá trị khi batch-dim cần broadcast (vd `[1,3,4]@[5,4,2]` ra `[1,3,2]`),
  và `lhs2D @ rhsND` sai thứ tự dim. Eager & compiled CÙNG sai (cùng builder) nên differential eager-vs-compiled
  KHÔNG bắt được — phải dùng oracle độc lập (ref bmm thuần JS).
- Nguyên nhân: `builder.matmul` chỉ xử lý 2D×2D và 3D×3D-equal-batch; còn lại fallback dot không có batch.
  Sau khi cho matmul tự broadcast batch (chèn `broadcast_in_dim`) thì lộ bug 2: `broadcast_in_dim` được hạ
  thành VIEW (set `srcBuf.broadcastDims`, map result→buffer gốc), nhưng `buildDotGeometry` không tôn trọng
  `broadcastDims` → extent contracting = `lhs.shape[dim]` ra undefined → vòng lặp không chạy → ra 0.
- Code:
  - `src/tensor/utils/shape_utils.js`: thêm `matmulOutputShape` (semantics PyTorch, rank-promote 1D + broadcast batch).
  - `src/compiler/ir/graph/builder.js`: `matmul` rewrite — promote 1D, broadcast batch qua `broadcast_in_dim`, dot với batch dims; squeeze lại dim 1D. Thêm helper `bcastBatchDims` + `_broadcastBatch`.
  - `src/dispatcher/jit_dispatch.js` + `src/tensor/native/meta/meta_ops.js`: dùng chung `matmulOutputShape`.
  - `src/compiler/passes/lowering/lowering_registry.js`: `buildDotGeometry` dùng LOGICAL shape (operand type) cho extent/dim, remap chỉ số vật lý qua `broadcastDims` (helper `physicalDotIndices`).
- Test (oracle độc lập, không hardcode): `tests/e2e/compiler/linalg-pipeline.test.js` — 12 case tường minh + fuzz 120 prog so với `bmmRef` thuần JS. Revert fix → RED, apply lại → GREEN.

## Fuzzer mới quét được (phạm vi hẹp)
- Op: unary/binary(+broadcast row/col)/matmul-2D/reduce(sum,mean,max,min).
- Shape: chỉ 2D, dim 2–6, chain 2–6 op. Dtype: f32 + i32. Backend: CPU + WASM.

## Việc tiếp theo — ưu tiên cao → thấp

### P0 — mở rộng generator (mật độ bug lịch sử cao nhất ở đây)
- [x] Rank ≠ 2: batched matmul + broadcast nhiều chiều, + 0D/1D/3D/4D cho unary/binary/reduce + view-ops. Fuzzer `fuzz-differential.test.js` (block "N-D + view-ops", 400 prog × cpu+wasm) xanh.
- [x] View-ops trong chain: reshape/transpose/permute/expand/slice(step)/squeeze/unsqueeze/narrow/select/flatten — đã có trong N-D fuzzer, trộn với compute.
- [x] Backward/autodiff: fuzzer `differential-backward.test.js` (block "fuzz backward ... numerical finite-difference", 200 prog, oracle ĐỘC LẬP = numerical grad), giờ chạy **fusion ON** (workaround đã gỡ sau khi fix bug fusion multi-output). 200/200. CÒN double-backward (grad of grad) chưa fuzz.

### Bug đã fix đợt này (2026-06-10, đợt 2)
> Forward N-D/view fuzzer + backward numerical fuzzer quét ra một loạt bug. Đã fix (ko comment/hardcode/O(n²)):
- **matmul N-D broadcast** (sai shape+value): `shape_utils.matmulOutputShape`, `builder.matmul` (promote 1D + broadcast batch qua broadcast_in_dim + dot), `buildDotGeometry` tôn trọng `broadcastDims`. Test: linalg-pipeline.
- **contiguous()/tensorToContiguous bỏ qua offset/stride của view** (select/slice-step ra data sai/thừa): `view_ops.contiguous`, `jit_dispatch.tensorToContiguous` chỉ fast-path khi offset==0 && storage.len==numel.
- **broadcast_in_dim hạ thành VIEW nhưng consumer ko phải elementwise/dot ko tôn trọng broadcastDims** (slice/reshape... đọc sai → 0/NaN): `graph_to_tensor` chỉ giữ view khi mọi consumer ∈ `BROADCAST_VIEW_SAFE`, còn lại materialize.
- **slice VJP bỏ qua step** (grad sai khi step>1): `vjp_rules/shape.js` pad interior=step-1, high tính theo gradShape.
- **eager autograd view-ops ko ghi backward** (slice/expand/squeeze/unsqueeze/narrow/select share grad của source → grad sai shape/value): wire `_wrapWithAutograd` + `SelectBackward` mới (`autograd/function/view.js`, `view_ops.js`, `dispatch.js`).
- **ReluBackward đọc raw storage của view đã lưu** (mask sai): `unary.js` dùng `.contiguous()`.
- **reduce(sum/mean) VJP dùng broadcast_dimensions có lỗ [1,2]** → kích fusion-bug; đổi sang reshape-về-keepdim rồi broadcast identity (`vjp_rules/reduction.js`).
- **grad của operand bị broadcast ko reduce về shape gốc** (grad sai N lần): `backward_builder.reduceGradToOperandShape` sum các broadcast-dims.
- **2 return-value trỏ cùng buffer → chỉ ghi 1, cái kia = 0**: `graph_to_tensor` cấp buffer mới + copy khi srcBuf trùng (`usedReturnBuffers`).
- **lowering theo block-order, ko topo** → consumer (copy của reshape) emit TRƯỚC producer (reduce) ⇒ đọc buffer chưa ghi = 0: `graph_to_tensor.topologicalOps` hạ ops theo thứ tự topo.

### BUG fusion multi-output khác shape — ĐÃ FIX (2026-06-11)
> Repro tối thiểu (backward): `mean(sub(sum(x,0,keepdim), y), 0).reshape([1])` → grad x phải `[1,1,1]`,
> fusion ON ra `[1,0,0]` (lane 1,2 = 0). Cần đủ sum-keepdim + sub-broadcast-y + mean + reshape; bỏ 1 → đúng.
> Backward numerical fuzzer bật fusion ON: s=152, s=196 đỏ.
- Root cause: `lowerFusion` lấy `outputs[0].shape` làm loop extent (`rules/fusion.js:91-92`). Khi fusion có
  NHIỀU output KHÁC shape (vd grad-y `[1]` và grad-x `[3]` qua broadcast_in_dim trong thân fusion), loop chạy
  theo output đầu `[1]` → các output lớn hơn chỉ ghi lane 0, phần còn lại = 0. Dump PrimFunc xác nhận:
  `for i in 0..1 { buf17[i]=cse; buf15[i]=cse }` với buf15 shape `[3]`.
- Fix (ko comment/hardcode): `canLowerAsElementwiseFusion` thêm guard — chỉ đi đường 1-loop elementwise khi
  MỌI result cùng shape; khác shape → fallback `lowerFusionAsIndividualOps` (mỗi op loop riêng, đúng).
  Helper `shapesEqual`. File: `src/compiler/passes/lowering/rules/fusion.js`.
- Test: gỡ workaround `fusion:{enabled:false}` trong `differential-backward.test.js` (fuzz backward giờ chạy
  fusion ON mặc định = regression test). Revert guard → s=152/196 RED; apply lại → 41/41 GREEN.
- Verify: e2e+compiler+stress 1236/1236 pass.

### Bug đã fix đợt 3 (2026-06-11) — exploratory fuzz nn/composite ops + f64
> Exploratory fuzzer (`_explore` scratch) quét softmax/argmax/where/prod/trig... × {f32,f64} × {cpu,wasm}.
- **prod f64 trên WASM ra ≈0** (denormal rác): `lowerBufferStore` (`tensor_to_lir.js:109`) lấy dtype store từ
  `inferDtype(value)` (FloatImm→f32) thay vì dtype BUFFER đích → init const f64 bị `f32.store` (4 byte) rồi
  reduce đọc `f64.load` (8 byte). Fix: `dtype = node.buffer.dtype || inferDtype(value)`. Kéo theo lộ bug 2:
- **i32 const init dựa byte rác**: `lowerConstant` bọc MỌI số vào `FloatImmNode` kể cả i32 → init min i32
  (INT_MAX) là f32 không biểu diễn nổi → sau fix store-dtype thì `i32.trunc_f32_s` trap. Fix: emit
  `IntImmNode` khi `isDtypeInt(outBuf.dtype)` (`lowering_registry.js`). (Trước đây "may mắn" đúng vì bytes rác
  vẫn là số dương lớn.)
- **argmax/argmin compile fail rồi sai axis**: `_SCALAR_ARG_SPEC` trong `tracing/dispatch.js` thiếu
  argmax/argmin → attrs rỗng → axis undefined (throw "Cannot infer result types"), thêm vào tracer map thì
  default axis 0 (sai). Fix: thêm `argmax/argmin: ['dim','keepdim']` vào dispatch spec + entry tracer
  `_BUILDER_METHOD_MAP` (`b.argmax(args[0], a?.dim ?? 0, a?.keepdim ?? false)`).
- Test: `differential-nn.test.js` (+prod_f64, +prod_f32, +argmax_dim1, +argmin_dim0, +argmax_keepdim). Revert
  → prod_f64 + argmax RED; apply → 46/46 GREEN. Full e2e+compiler+wasm 1572/1573 (1 = stress flaky-under-load).
- GAP còn lại (chưa fix, op chưa wire cho compile — không phải value bug): `cat`, `stack` (No dispatch key),
  `pow` (Cannot infer result types — cả eager cũng ko compile). → P1 wiring.

### Bug đã fix đợt 4 (2026-06-11) — exploratory fuzz batch 2 (negative-dim, comparison, f64 chains)
- **EAGER reduce dim ÂM sai toàn bộ** (sum/mean/max/min/prod với dim=-1,-2): `jit_cache._buildGraphFunc`
  (đường eager) KHÔNG normalize dim âm — truyền thẳng `-1` vào `builder.reduce` trong khi `_inferOutputShape`
  lại normalize → graph reduce nhầm trục → ra `[1,2,3,...]` degenerate. (compiled/tracer ĐÚNG → differential
  eager-vs-compiled bắt được, eager là bên sai.) Fix: `.map(d => d < 0 ? rank + d : d)` trong jit_cache,
  khớp với `_traceReduce`. Test: differential-nn (+mean_negdim, +sum_negdim2, +prod_negdim_f64,
  +max_negdim_keepdim). Revert → 8 RED; apply → 54/54 GREEN. Full e2e+compiler+dispatcher+tensor 1617/1617.
- Đã quét sạch (no fail): comparison/where/select, softmax/log_softmax f64 + dim giữa, matmul f64 2D/3D,
  reduce all-dims, unary f64 (gelu/rsqrt/sqrt/log/exp), div f64, abs/neg i32, broadcast scalar/row.

### Bug đã fix đợt 5 (2026-06-11) — P1 dtype coverage (1 batch lớn)
> Exploratory fuzz integer dtypes (i8/i16/i64/ui8) + bool + f16 × {unary,binary,reduce,matmul} × {cpu,wasm}.
- **WASM narrow-int load/store biến mất** (i16/ui8/bool ra 0/rác, i16 reduce compile-fail "stack fallthru"):
  `wat_encoder.js` INSTR map THIẾU opcode memory narrow-int — codegen emit đúng `i32.load16_s`/`store16`/
  `load8_u`/`load16_u` nhưng encoder drop token lạ → bytes rỗng. Fix: thêm 4 memarg entry
  `[opcode, align=log2(bytes), offset]` vào `wat_encoder.js:33-34`. (ui8/i16 wrap-on-overflow đã đúng sẵn —
  store8/store16 truncate khớp CPU mod-2^n.) [subagent]
- **eager `where` cond bool ra giá trị wrap** (-2→254): output dtype + buffer ctor lấy từ `tensors[0]` (=cond
  bool) thay vì kết quả. Fix: `jit_cache` gắn `entry.outDtype` từ return-op của graph; `jit_dispatch` dùng
  `entry.outDtype` + cấp outData bằng `typedArrayCtor(outDtype)` (không phải `runtimeArgs[0].constructor`).
- Test: `differential-nn.test.js` BOTH2 (+add/mul/sum/max/matmul cho i16 & ui8, +eq_bool) + 1 test standalone
  eager where-bool. `mk()` generalize `Math.round` cho mọi non-float (qua `isDtypeFloat`). Revert → RED; apply
  → 77/77. Full e2e+wasm+compiler+dispatcher+tensor **1960/1960**.
- f16 quét sạch (lưu trữ as-f32, hoạt động). i64 BỎ QUA (cần BigInt host plumbing — niche).

## ====== P1 COI NHƯ XONG (2026-06-11) ======
> Bug-hunt + op/dtype coverage của P1 đã hoàn tất:
> - dtype i8/i16/ui8/bool/f64/f16(as-f32) ✓ + ~16 op mới wire/build ✓ + compiled VJP cho op mới ✓
> - chain-fuzz op mới + softmax/where: 1200/1200 sạch ✓
> - quant path: 52 test sẵn (có numerical/execute) pass ✓
> - Toàn bộ: e2e+compiler+tensor+nn+dispatcher+wasm **2156/2156**
> CÒN LẠI (KHÔNG phải bug — feature/precision/niche, ghi rõ bên dưới): true f16/bf16, eager-autograd parity,
> i64, topk-index, gather/scatter raw, logical_and-wasm (no real user).

### P1 — dtype & op còn trống (cập nhật)
- [x] Integer dtypes i8/i16/ui8 + bool trên compile cpu+wasm — đã fix (encoder narrow-int) + test.
- [x] f64 (quét rộng, ổn). f16 (as-f32, ổn).
- [x] f16/bf16 ĐÚNG NGHĨA (half-precision thật) — XONG đợt 12 (Uint16Array 2-byte, Giesen RNE, cpu+wasm bit-exact).
- [x] i64 (BigInt host marshalling) — XONG đợt 12 (encoder i64 opcodes + BigInt plumbing, >2^53 exact).
- [ ] Quant path (quantize→dequantize observer) fuzz — chưa đụng.

### Feature đã wire đợt 6 (2026-06-11) — frontend ops (IR+lowering ĐÃ có sẵn, chỉ wire frontend) [subagent]
> Phát hiện: IR ops + builder methods + lowering rules cho concat/split/gather/scatter/pad/clamp/pow/where/
> one_hot ĐÃ tồn tại đầy đủ. "Làm feature" = wire frontend (ops.js export + registration schema + index.js +
> dispatch _SCALAR_ARG_SPEC + tracer/jit_cache _BUILDER map). Template: `where` (wire tối thiểu, auto-path lo
> phần còn lại vì builder có method cùng tên).
- [x] `clamp(self,min,max)` — min/max nhận số hoặc tensor (`_asTensor` helper). Map → builder.clamp(min,self,max).
- [x] `pad(self,low,high,value=0)`.
- [x] `one_hot(indices,depth)` → f32.
- [x] `index_select(self,dim,index)` — trên builder.gather, helper `indexSelectGatherOpts` (StableHLO gather attrs).
- [x] `cat`/`stack` COMPILE — root: `computeKeySet` (dispatcher.js) không đệ quy vào array args → TRACING key
  của symbolic tensor TRONG `Tensor[]` bị bỏ sót. Fix computeKeySet đệ quy + flatten tensor-list ở
  jit_dispatch + tracing dispatch (guard để `int[]` như pad.low/high KHÔNG bị nhầm là tensor-list). stack =
  reshape-insert-dim + concat.
- Test: differential-nn.test.js (+clamp_scalar, +pad_2d trong BOTH2; +describe "index ops" cho one_hot/
  index_select/cat/stack cpu+wasm). Sanity eager+cpu+wasm khớp. Full e2e+tensor+dispatcher+compiler+wasm+nn
  **2088/2088**.
- SKIP (lý do rõ): `gather`/`scatter` raw (opts XLA 5-attr + scatter cần combiner region — phức tạp, đã có
  index_select thay); `split` (multi-output `numResults:-1`, cần multi-output plumbing ở JIT/tracing — ngoài
  phạm vi wire frontend); `cumsum`/`flip`/`roll`/`repeat`/`topk`/`group_norm` (CHƯA có IR op — cần build mới).

### Feature đợt 7 (2026-06-11) — composition/decomposition (KHÔNG cần IR op mới)
> Nhận ra nhiều "feature thiếu" làm được bằng compose op có sẵn, không cần multi-output plumbing.
- [x] `group_norm(input,numGroups,weight,bias,eps)` — decompose reshape→mean/var→normalize→reshape (như
  layer_norm). `src/nn/functional/normalization.js` + class `GroupNorm` (`modules/normalization.js`, export
  `nn/index.js`). Test: differential-nn (groupnorm, groupnorm_g1). eager==cpu==wasm.
- [x] `repeat(reps)` / `tile(reps)` — method, compose reshape→expand→reshape (`view_ops.js`). PyTorch/numpy
  semantics. Test: repeat_2d/repeat_1d/tile_promote.
- [x] `split(sizeOrSizes,dim)` / `chunk(chunks,dim)` — method, compose qua `narrow` (mỗi mảnh 1 narrow, trả JS
  array; compile() đã xử lý forward trả array). KHÔNG cần multi-output IR plumbing. Test: split_cat/chunk_cat
  round-trip cpu+wasm.
- Full e2e+tensor+nn+dispatcher+compiler+wasm **2102/2102**.

### Feature đợt 8 (2026-06-11) — flip/roll/cumsum (composition, KHÔNG cần IR op mới)
> Nhận ra cả 3 compose được, không cần scan kernel mới. File mới `src/tensor/ops/composite.js` (import cat/add/
> index_select + zeros + tensor), install method + export top-level (index.js).
- [x] `roll(self,shift,dim)` — cat(narrow(tail), narrow(head)). Circular shift.
- [x] `flip(self,dims)` — index_select với reversed-index constant (capture-constant trong trace OK đã verify).
- [x] `cumsum(self,dim)` — Hillis-Steele scan: out += shiftRight(out, 2^k) cho k=0..log(n). O(n·log n) work,
  O(log n) graph ops — KHÔNG O(n²), không cần scan kernel. shiftRight = cat(zeros, narrow).
- Test: differential-nn (roll_dim1/flip_dims/cumsum_dim0/dim1 differential) + oracle ĐỘC LẬP (roll circular,
  flip reverse, cumsum prefix-sum giá trị tường minh). 127/127. Full e2e+tensor+nn+dispatcher 983/983.

### Feature đợt 9 (2026-06-11) — sort/topk qua static bitonic network [subagent]
- [x] `sort(self,dim,descending)` — bitonic sorting network UNROLLED tại compile-time (compare-exchange tĩnh cho
  n cố định). Pad lên power-of-2 với sentinel ±Infinity; mỗi stage (k,j): partner=index_select(reversed const
  idx), lo=minimum, hi=maximum, x=where(maskKeepLo, lo, hi). Non-last-dim qua transpose. O(n·log²n) element-op,
  KHÔNG O(n²) data-loop. `composite.js`.
- [x] `topk(self,k,dim,largest)` — sort rồi narrow(0,k). **Values-only** (chưa trả index — v1).
- Test: 6 differential (sort_last_pow2/nonpow2/desc/dim0, topk_last/smallest) + 6 oracle ĐỘC LẬP vs
  `Array.prototype.sort`. Verify n=1..32 (pow2 + non), âm + trùng, 2D/3D. Revert (đảo mask) → 10 RED. Full
  e2e+tensor+dispatcher+compiler+nn **1817/1817**.

### Bug đợt 10 (2026-06-11) — compiled VJP cho op mới (backward fuzz numerical oracle)
> Backward smoke-test các op mới: compiled backward (compileWithBackward) lỗi ở clamp/flip; eager .backward()
> còn hỏng nhiều hơn (cat/clamp/flip/roll/index_select/pad ko tạo backward node — EAGER autograd gap riêng).
> Tập trung compiled (training path):
- **VJP thiếu cho `clamp`, `where`/`select`, `gather`** → grad undefined/lỗi. Thêm:
  - `clamp` VJP: nested where (geLo→leHi) mask, trả grad chỉ trong [lo,hi]. `vjp_rules/arithmetic.js`.
  - `where`/`select` VJP: grad routed theo cond. `arithmetic.js`.
  - `gather` VJP: scatterAdd(zeros, indices, grad) qua gather↔scatter duality (offset_dims→update_window_dims...).
    `vjp_rules/shape.js`. → flip/index_select backward chạy.
- **Bonus bug:** `logical_and` (`&&`) trên WASM emit `i32.and` sai khi operand là mask f32 → clamp VJP đầu
  dùng logicalAnd ra grad [0,0,0,0] trên wasm. Đổi clamp VJP sang nested-where (tránh logical_and). `logical_and`/
  `logical_or` KHÔNG có user thật nào khác trong codebase (latent, low-impact) → ghi nhận, chưa fix codegen.
- Test: differential-backward.test.js — `COMPOSITE_BWD` (cat/cumsum/flip/roll/repeat/pad/index_select vs
  numerical, op tuyến tính nên grad exact) + explicit clamp/where (data tránh kink, grad tường minh). Revert →
  clamp/where RED. Full e2e+compiler+tensor 1616/1616, backward 65/65.

### Bug đợt 11 (2026-06-11) — EAGER autograd cho op mới ĐÃ FIX [subagent]
- [x] Thêm autograd Function eager cho 5 primitive: `CatBackward`/`ClampBackward`/`PadBackward`/
  `IndexSelectBackward`/`WhereBackward` (`autograd/function/indexing.js` + register `autograd/registry.js`).
- [x] **dispatch.js load-bearing fix**: `_anyRequiresGrad`/`_extractTensors` đệ quy vào array arg → cat([...])
  với grad-tensor TRONG list giờ kích autograd (trước bị bỏ → cumsum/roll mất grad).
- Composites (flip/roll/cumsum/repeat/split) tự thừa kế. Smoking-gun: eager `cumsum(x,1).backward()` giờ ra
  **[5,4,3,2,1]** (reverse-cumsum) thay vì all-ones. Verify: clamp[1,0,1,1], cat[2,2,2,2], flip[3,2,1] đúng.
- Test: differential-backward.test.js — 15 prog eager-vs-numerical + 4 explicit. Revert → 17 RED (registry) /
  7 RED (dispatch). Full e2e+autograd+tensor+nn+compiler+dispatcher **1913/1913**.
- [x] `logical_and`/`logical_or` WASM codegen — ĐÃ FIX (2026-06-11). Root cause thật (rõ hơn note cũ): bug CHỈ ở
  đường **SIMD**. `f32x4.gt/lt` tạo mask **all-ones** `0xFFFFFFFF` (true) / `0` (false); `v128.and`/`v128.or` giữ
  nguyên; bool lưu dạng f32 nên `convert(bool→f32)` trong vec-path là NO-OP (`_emitVecExpr CastNode` bỏ qua cast) →
  mask all-ones store thẳng ra f32 = **NaN**. (Scalar đúng: `f32.gt→i32 0/1`, `i32.and`, `f32.convert_i32_s→0.0/1.0`.)
  Chỉ lộ khi mask SIMD bị convert-sang-float-numeric (qua `select`/bitselect thì OK vì all-ones là truthy).
  - Fix: `backend/wasm/codegen.js` `_emitVecExpr` case `CastNode` — khi cast mask (CompareNode / `&&`/`||`/`!`) sang
    float thì normalize `v128.and(mask, f32x4.splat(1.0))` (all-ones→1.0, 0→0.0; đúng cả với bool sạch 1.0/0.0).
    Helper `_isVecMaskExpr`. KHÔNG đụng select (bitselect vẫn dùng raw mask).
  - Test: `tests/backend/wasm/wasm-compile.test.js` "logical_{and,or} mask converted to float matches oracle"
    (N=3 scalar + N=8/16 SIMD, cpu+wasm vs oracle 0/1). Revert fix → 4 SIMD case RED, scalar xanh. Full
    e2e+tensor+wasm+compiler+autograd+nn+dispatcher **2342/2342**.

### Feature đợt 12 (2026-06-11) — 4 feature deferred (f16/bf16 thật, i64, topk-index, gather/scatter)
> Tất cả land + test (independent-oracle) + revert-test RED + verify cpu+wasm. Full suite 2939/2939 (trừ gpu segfault).
- [x] **TRUE f16/bf16 half-precision** — lưu trữ 2-byte THẬT (Uint16Array) + rounding chuẩn (Giesen branchless).
  - `src/tensor/utils/half.js` (MỚI): f16/bf16<->f32 (Giesen magic, RNE), `coerceForStorage`/`readFromStorage`,
    đăng ký global `__mlfw_*` cho CPU codegen gọi. Bf16 = top16 + RNE.
  - Compute model: "store half, compute f32" — normalizeDtype đã map f16/bf16→f32 nên math là f32; CHỈ convert ở
    boundary load/store. CPU codegen: wrap load `__mlfw_f16_to_f32(buf[i])`, store `buf[i]=__mlfw_f32_to_f16(v)`.
    WASM codegen: `i32.load16_u`+decode bit-math inline / encode+`i32.store16` (`_emitHalfDecode/Encode`, 3 scratch
    local). SIMD tự tắt cho f16/bf16 (ko có trong SIMD table) + guard `_treeHasHalf` ở vectorized loop.
  - dtype plumbing: types.js (BF16 + bytes 2), dtype.js (typedArrayCtor→Uint16Array), dtype_map.js (js Uint16Array,
    wasm load/store narrow bytes 2), lir/nodes.js normalizeDtype bf16→f32, runtime.js ctors. Input round + read expand
    ở from_ops/tensor.js (toArray/item). CPU vs WASM BIT-EXACT. encoder: +i32 and/or/xor/shl/shr + reinterpret.
- [x] **i64 (BigInt host marshalling)** — `tensor([...],{dtype:'i64'})` nhận JS number→BigInt (coerceForStorage).
  CPU codegen: store i64 wrap `BigInt(...)`, zero/const literal `0n` (BigInt64Array ko nhận Number). WASM: thêm ~25
  i64 opcode vào encoder (load/store/add/mul/div/cmp/extend/wrap/convert/const sleb-BigInt) + i64 local/param parse;
  codegen `_numPrefix/_joinPrefix/_convertTo` + int ops dùng prefix (i64.*). Giữ chính xác >2^53 (test 2^53+1).
- [x] **topk trả INDEX** — carry index payload qua bitonic network: `idx` seed iota, mỗi compare-exchange
  `takeSelf=where(mask, eq(lo,x), eq(hi,x)); idx=where(takeSelf, idx, partnerIdx)`. Trùng → keep-self (deterministic).
  `topk`→`[values,indices]` (PyTorch tuple), `+argsort`. `_bitonicLastDim(self,desc,withIdx)`. (composite.js)
- [x] **gather/scatter RAW (PyTorch dim-indexed)** — `builder.gatherDim/scatterAddDim`: materialize full-coord index
  (iota các dim khác + index ở dim d → concat) rồi `builder.gather/scatterAdd` (XLA 5-attr, sliceSizes=1, collapse
  hết). `gather`/`scatter_add` wire như index_select; `scatter` (overwrite) = composition where+scatter_add+mask
  (unique-idx). VJP: gather có sẵn; thêm `scatter` VJP (grad_operand=grad, grad_updates=gather dual). gather backward
  vs numerical OK.
- Test: differential-nn.test.js (half-precision block, i64 block, topk/argsort/gather/scatter trong index-ops +
  composition oracle), differential-backward.test.js (COMPOSITE_BWD +gather_d1/gather_dup/scatter_add_src).
- GIỚI HẠN ghi rõ: bf16 WASM bỏ nhánh NaN-mantissa (finite OK, khớp half.js cho data hữu hạn); scatter overwrite chỉ
  đúng unique-index (PyTorch duplicate = unspecified); i64 toArray() mất chính xác >2^53 (đọc raw BigInt storage thay).

### Feature CŨ (đã xong ở trên)
- [x] `topk` trả INDEX — xong (đợt 12).
- [x] `gather`/`scatter` raw (advanced indexing) — xong (đợt 12, PyTorch dim-indexed trên builder.gather/scatterAdd).
- [x] grad qua split/flip/roll/cumsum/sort — QUÉT XONG. split/flip/roll/cumsum đã có backward test (COMPOSITE_BWD +
  EAGER_PRIM_BWD). **sort/topk** chưa → quét ra BUG fusion-merger.
  > **BUG fusion-merger re-merge fusion đã xóa** (2026-06-11): backward của `binary_op(sort(x), y)` (vd `mul(sort,c)`,
  > shape multi-row) THROW "Fusion lowering: unmapped operand from 'unknown'". sort alone OK; sort+binary lỗi.
  - Root: `FusionMergerPass.run` (`fusion_merger.js`) sau `_merge(producer,consumer)` xóa CẢ HAI op (dropAllOperands+
    removeOp) nhưng chỉ `merged.add(producer)`, KHÔNG add consumer. Edge sau dùng consumer (đã xóa, numOperands=0) làm
    producer → `pArgRemap` rỗng → block-arg của nó map về `undefined` → `remapOperands` giữ ref cũ = free var trong
    thân fusion → lowering throw. (fusion OFF / strategy dominator: OK → khẳng định bug ở merger.)
  - Fix 1 dòng: thêm `merged.add(consumer)`. Test: `differential-backward.test.js` COMPOSITE_BWD (+sort_mul_const,
    sort_add_const, sort_dim0_mul, relu_mul_sort, topk_mul_const, sort_plus_sort vs numerical). Revert → 6 RED.
- [x] **conv/pool/norm coverage** — QUÉT XONG. Quét ra BUG dilation+groups.
  > **BUG conv2d/conv1d BỎ dilation + groups khi compile** (2026-06-11): `tracer.js:35 _BUILDER_METHOD_MAP.conv2d`
  > gọi `b.conv(...,strides,padding)` THIẾU `{dilation,groups}` (dù `dispatch.js:20` đã trích `dilation,groups`).
  > → compiled luôn dùng dilation=[1,1],groups=1: sai SHAPE (dilation) + sai VALUE (groups). conv op-def + lowering
  > ĐÃ xử lý dilation/groups đúng — chỉ frontend tracer drop. (Eager ĐÚNG, là oracle.)
  - Fix: truyền `{ dilation: a?.dilation, groups: a?.groups }` vào `b.conv`. Test: differential-nn BOTH (+conv2d_dilation2,
    conv2d_groups2, conv2d_dilation_groups, conv1d_dilation) + oracle ĐỘC LẬP (dilated conv hand-ref + grouped per-group).
    Revert → 6 RED. Pool (maxpool/avgpool/adaptive stride/pad), groupnorm/batchnorm-eval/layernorm-nd, softmax/
    log_softmax, embedding(i32) — quét sạch (eager==cpu==wasm).
- [x] **bool/mask, comparison/select/where, index dtype** — QUÉT XONG: where(bool-mask), where(from-compare),
  eq/lt-int→where, comparison-int — eager==cpu==wasm sạch. gather/scatter/index_select/embedding với index **i32**:
  sạch. GAP (chưa fix, niche): index **i64** (BigInt) throw "Cannot mix BigInt" — index ops compute qua JIT, BigInt
  index trộn Number offset trong code sinh. i32-index là chuẩn ở đây (argmax/argsort/topk-idx đều trả i32) → để lại.
- [x] **Quantized i8/u8 (quant path)** — quant là COMPILER PASS (ko phải user op). 51 test sẵn (params-quantization +
  pass-quantization, có numerical/execute) PASS. Coverage ổn; random-graph quant-fuzzer là future-add.

### P2 — edge values & shape biên (2026-06-11, đợt P2)
- [x] **NaN / Inf / -0** (item 1): quét unary/binary/reduce/matmul/softmax/where/clamp/min/max với NaN,±Inf,-0.
  eager==cpu==wasm NHẤT QUÁN ở MỌI case (NaN propagation, Inf arithmetic, max/min-NaN, -0 div/sign, softmax-Inf→NaN).
  Không có divergence — IEEE semantics khớp 3 đường. (Không cần fix.)
- [x] **Empty / degenerate shapes** (item 2): [0,3]/[3,0], reduce trên axis rỗng, matmul 0-contract, cat/broadcast
  size-0 — eager==cpu==wasm khớp. sum-rỗng→0, prod-rỗng→1, max/mean-rỗng→-Inf/NaN (reduce-init leak,
  NHẤT QUÁN 3 đường — giống PyTorch-undefined nhưng không phải differential bug).
- [x] **i32 / narrow overflow** (item 4): add/mul overflow 2^31, abs/neg INT_MIN, ui8/i16/i8 wrap, sum/matmul
  accumulate wrap — eager==cpu==wasm khớp (mod-2^n nhất quán).
- [x] **BUG: i32 div-by-zero & INT_MIN/-1 trap trên WASM** (item 4) — ĐÃ FIX.
  - Repro: `div(i32[6,0,…], i32[0,0,…])` hoặc `div(i32[INT_MIN],i32[-1])`. eager&cpu→0 (resp. wrap INT_MIN),
    WASM `i32.div_s` TRAP ("divide by zero" / "divide result unrepresentable"). Cross-backend divergence.
  - Root: `backend/wasm/codegen.js _emitMathOp` emit thẳng `i32.div_s`/`rem_s` — trap trên /0 và INT_MIN/-1.
  - Fix: `_emitIntDiv`/`_emitIntRem` (scratch local `_idiv_a/b` + prescan) — safe-divisor select(1,b,trapCond) rồi
    div_s, kết quả select(0,q,b==0). Khớp JS ToInt32: /0→0, INT_MIN/-1→wrap INT_MIN. i32+i64, lồng nhau OK.
  - Test: `differential-nn.test.js` "i32 division edge cases" (oracle `(a/b)|0`). Revert→WASM RED, cpu xanh.
  - GHI CHÚ: i64 div-by-zero — eager BigInt `12n/0n` THROW (không silent), compiled wasm→0. Niche, eager-side
    throw nên không phải usable path; để lại.
- [x] **Empty-axis / all-same reductions** (item 5): len-1 axis, argmax/argmin ties (lowest-index), max len-1,
  mean rỗng→NaN — eager==cpu==wasm khớp.
- [x] **Shape động (DYNAMIC / sym_int)** (item 6) — XONG TRỌN (1 batch lớn). Feature gần như ko hoạt động e2e →
  giờ chạy đúng cpu+wasm cho elementwise/chain/broadcast, reduce dim0/dim1/keepdim, mean-trên-trục-dynamic,
  softmax, normalize, matmul (dynM/dynK), **2-dim-cùng-dynamic, và transpose dim dynamic** (compile-once-run-many).
  > Trước fix: `dynamic_shapes:[Set([0])]` cho kết quả SAI (silent) NGAY CẢ trên shape đã trace. Codegen ĐÃ tham số
  > hoá (`_ds_*`) nhưng runtime/tracer/lowering/wasm-layout chưa nối. Chuỗi fix (gốc → ngọn):
  - **runtime ko truyền `_ds` args**: `compile.js _prepareExecution` truyền raw typed-array → `run()` ko đọc shape →
    mọi `_ds` fallback. Fix: bọc allArgs thành `RuntimeTensor(data,shape,dtype)` để `_extractShapeParams` trích dim.
  - **`_extractShapeParams` bufferIndex chết**: key bằng VariableNode nhưng query bằng buffer-name string → luôn miss
    → toàn positional-scan (sai cho transpose/reorder). Fix: build bufferIndex theo buffer NAME (xử lý cả 2 dạng key:
    test dùng string-key, PrimFunc dùng VarNode→Buffer). `runtime.js`.
  - **loop extent leak `-1`** ở MỌI lowering rule gọi `makeLoopNest(ctx,shape)` THIẾU `buf` → bound = DYNAMIC literal.
    Fix: truyền `buf` + thread `extentNodes` ở fusion + shape.js (transpose/reshape/slice/pad/concat/iota/broadcast) +
    control_flow + layout + linalg(gather/scatter) + lowerConstant + copyPairs.
  - **buffer phái sinh ko đăng ký `_ds`** → `_resolveShapeParam` fallback '1' → intermediate cấp size 1, ghi quá biên.
    Fix: `getOrAllocBuffer/allocFreshBuffer` gọi `_registerDynamicDims` (đăng ký mọi DYNAMIC dim của MỌI buffer).
  - **reduce/transpose mất symbol propagation**: thêm `propagateSymbolicShapes` cho `reduce` + `transpose`
    (`ir/graph/ops/reduction.js`, `ops/shape.js`). reshape propagate **return null** khi inShape undefined → tracer
    rơi về generic offset-fallback (đúng cho keepdim reduce→reshape nội bộ).
  - **mean trên trục DYNAMIC chia sai**: rule mean tính `reduceSize *= inBuf.shape[dim]` = -1 (DYNAMIC) → chia /-1.
    Fix: tách static part + `extentNode` cho dynamic reduce-dim, dựng divisor runtime + `MathOp('/')`. `rules/reduction.js`.
  - **WASM memory-layout overlap (2-dim-dynamic)**: `scanner.js computeMemoryLayout` dùng `numel<0` để phát hiện
    dynamic — `(-1)*(-1)=1` (2 dim dynamic) lọt → intermediate cấp 4 byte → ĐÈ input. Fix: phát hiện dynamic qua
    `shape.some(d => d<0)`, over-alloc 65536. (cùng bug ở `wasm/codegen.js _layoutBuffers`).
  - **symbol unification (transpose/reorder intermediate)**: buffer phái sinh đảo chiều (vd transpose [s0,4]→[4,s0])
    có dim s0 ở vị trí KHÁC → positional-scan giải sai. Fix: tracer gắn `value.symbolicShape` lên IR value;
    `LoweringContext._shapeParamVar` UNIFY var theo tên symbol (`symbolToVar`) → mọi buffer cùng symbol dùng CHUNG
    1 `_ds`, giải chính xác qua key của INPUT buffer. dedup `func.shapeParams` theo tên var (graph_to_tensor).
  - Test: `differential-nn.test.js` "dynamic shapes: compile once, run on multiple concrete shapes" (21 case ×
    cpu+wasm + oracle tường minh: sum-reuse, mean-trên-trục-dynamic, transpose). Revert từng fix lõi → RED
    (mean 2, wasm-layout 2, symbol-unify 2). Full e2e+tensor+wasm+compiler+autograd+nn+dispatcher+tracing **2386/2386**.
  - GIỚI HẠN ghi rõ (KHÔNG phải bug): WASM dynamic intermediate over-alloc cứng 65536 byte/buffer (size runtime lớn
    >16K float/intermediate sẽ tràn — heuristic sẵn có); softmax wasm lệch eager ~1e-7 (f32 vs f64, trong tol).
- [x] **Accumulation extremes** (item 7): sum/mean 100k, matmul K=2000, denormal — trong tol (maxrel<1e-5).
  KHÔNG phải bug — chỉ giới hạn precision: WASM dùng kernel f32, eager+CPU-JS dùng f64. 2 case lệch (DOCUMENTED,
  ko fix): (a) `sum([3.4e38,3.4e38,-3.4e38])` WASM f32 acc → +Inf (overflow) vs eager f64 → 3.4e38; (b) catastrophic
  cancel `sum([1e8,1,-1e8])` WASM f32→0 vs f64→1. Cả hai là bản chất f32-compute, CPU-JS(f64)==eager.
- [x] **dim=1 broadcast boundaries** (item 3): scalar-tensor [1,1]/[1,1,1], keepdim-reduce→broadcast (NON-dynamic),
  [1,N]/[N,1] bcast, chain — eager==cpu==wasm khớp (static). (Dynamic keepdim→bcast: xem gap item 6.)

---

## CATALOG FUZZ TOÀN COMPILER — theo tầng + oracle bắt bug

> Khác biệt cốt lõi: differential (eager vs compiled) chỉ là 1 oracle. Compiler còn cần
> **metamorphic** (pass phải bảo toàn ngữ nghĩa), **round-trip**, **invariant/verifier**,
> **property-based**. Mỗi mục dưới ghi rõ oracle dùng.

### TIẾN ĐỘ FUZZ CATALOG (2026-06-11, đợt B→O) — tới mục O thì dừng
> Quét tầng A→O bằng fuzz để SĂN BUG (fuzz = throwaway scratch, KHÔNG commit). Đa số tầng SẠCH
> (đã test kỹ từ trước); chỉ tầng K ra bug thật (3 bug WASM scheduling).
> **Test commit = UNIT TEST cụ thể pin từng fix**, đặt vào file test có sẵn cùng tầng — KHÔNG commit
> file fuzz random. Cụ thể:
> - Bug tầng A (argmax/argmin): unit test trong `tests/e2e/differential-nn.test.js` (oracle độc lập, bảng dim×keepdim).
> - Bug tầng K (3 bug WASM scheduling): unit test trong `tests/backend/wasm/wasm-parallel.test.js`
>   (reduce axis-0 sync+async + max-reduce + matmul [64,8]@[8,16] sync+async, oracle tính tay). Revert guard
>   `_vecAccumOperandsUnitStride` → matmul RED.
> - Các tầng B,C,D,E,F,G,H,I,J,L,M,N,O: fuzz quét SẠCH, KHÔNG commit test (không có bug để pin; vùng đã có
>   test sẵn). Helper `graph_eval.js` đã xóa (chỉ dùng cho fuzz scratch).
> Tóm tắt fuzz đã quét (oracle dùng):
> - **A** ✓ (2 bug argmax/argmin neg-dim + keepdim — xem đợt 13 ở trên).
> (oracle ghi để lần sau quét lại; file fuzz random đã xóa — fuzz chỉ để săn bug, không commit)
> - **B** AD: backward-graph verifier invariant + remat metamorphic (remat on/off → cùng grad) + VJP softmax/matmul
>   vs numerical. SẠCH. (double-backward = feature chưa có `createGraph`, KHÔNG phải bug.)
> - **C** simplify: metamorphic eval-equality (cse/dce/constant_fold/algebraic on/off == nhau) + verifier-after-pass. SẠCH.
> - **D** fusion: metamorphic across strategy off/xla/dominator/epilogue vs eager. SẠCH.
> - **E** decompose/canon: canonicalize idempotent (run 2×==1×) + decomposition composite-op==eager. SẠCH.
> - **F** lowering: TensorIR verifier + LIR verifier invariant trên random graph. SẠCH.
> - **G** layout: layout-opt on/off numerically invariant (conv/pool). SẠCH.
> - **H** memory: inplace/remat/alignment on/off invariant. SẠCH.
> - **I** partition: mọi op assigned + partitioned-compile==eager. SẠCH.
> - **J** quant: quant/dequant round-trip error ≤ scale/2 + monotonic. SẠCH.
> - **K** schedule: scheduling on/off vs eager → ra 3 BUG WASM (xem đợt 14 dưới). Fix xong + unit test
>   `tests/backend/wasm/wasm-parallel.test.js`.
> - **L** analysis: use_def topo + opUsers vs brute-force, dominance postDom vs idom-chain, shape/dtype-infer==eager. SẠCH.
> - **M** autotune: workload_key deterministic + target-sensitive, tuning_db serialize round-trip, autotuned-compile==eager. SẠCH.
> - **N** backend codegen: SIMD vs scalar metamorphic (WASM simd on/off) + dtype marshalling round-trip
>   (f32/f64/i32/i16/i8/ui8/f16/bf16 cpu+wasm). SẠCH.
> - **O** runtime/pipeline: opt-level differential O0/O1/O2 == eager (cpu+wasm). SẠCH.
> → Đã quét tới hết O. Tổng: 1 batch bug tầng K (3 fix WASM scheduling), các tầng khác đã vững từ trước.

### Bug đã fix đợt 14 (2026-06-11) — tầng K (schedule) trên WASM: 3 bug (fuzz scheduling on/off vs eager)
> Fuzz `scheduling:{enabled:true}` vs eager (300 prog × cpu+wasm). CPU LUÔN đúng; WASM ra 39/300 sai → 3 bug độc lập
> trong đường SIMD/parallel scheduling (off-by-default, experimental). Fix xong còn 2/300 (limitation ghi dưới).
> Unit test pin fix: `tests/backend/wasm/wasm-parallel.test.js` (reduce axis-0 sync+async, max-reduce, matmul sync+async, oracle tính tay). Full regression pass.
- **BUG 1 — worker-pool partition sai cho reduction/matmul** (`io/node/wasm_pool.js` + `backend/wasm/codegen.js` +
  `runtime.js`): pool chia MỌI buffer theo `bufLen/extent` contiguous, giả định mọi buffer partition đều theo parallel
  loop. Sai khi input đọc strided (reduce axis≠cuối: input cột stride; matmul: B đọc full). Fix: (a) codegen thêm
  `poolSafe` = mọi store nằm TRONG parallel loop + đúng 1 parallel loop top-level (loại reduce/matmul có init/reshape
  ngoài loop); (b) runtime `isAsync`/`runAsync` chỉ dùng pool khi `poolSafe`, còn lại chạy SYNC full-extent (đã đúng);
  (c) pool gửi MỌI buffer FULL tại base offset (kernel địa chỉ tuyệt đối) + stitch chỉ partition-range của output.
- **BUG 2 — `_visitVectorizedFor` vectorize loop reduction (store độc lập vec-var) sai** (`backend/wasm/codegen.js`):
  schedule vectorize loop có store ghi cùng 1 ô mọi lane (reduction/broadcast-write) → SIMD lane-parallel sai. Fix:
  guard `_vecStoresLaneIndexed` + `_vecLoadsContiguous` (store/load phải index theo lane var ở vị trí cuối) → nếu không,
  scalarize (`_emitForLoop`, luôn đúng).
- **BUG 3 — `_visitVecAccumulator` strided operand sai** (`backend/wasm/codegen.js`): matmul contraction vectorized,
  operand `buf_B[k, j]` (k=vec var ở index KHÔNG cuối) → vec-load contiguous SAI (cần gather). Fix: guard
  `_vecAccumOperandsUnitStride` — chỉ đi SIMD-accumulator khi mọi load unit-stride theo vec var; còn lại scalar
  accumulator (đúng).
- **LIMITATION còn lại (2/300, KHÔNG fix — ghi rõ):** fused matmul-sau-vectorized-elementwise/reduce với extent
  KHÔNG chia hết vector-width (vd extent 5) trên WASM SIMD: intermediate SIMD-partial-vector + consumer matmul lệch
  ~1-10%. SIMD-off đúng. Đây là edge sâu của partial-vector intermediate layout, trong feature off-by-default. CPU
  luôn đúng. Test section K scope vào reduce/elementwise-all-axes + standalone-matmul (đều pass); ghi nhận edge này.

### A. Frontend — tracing / dispatcher / IR builder
- [~] Round-trip IR: build graph → print → parse → print — CHƯA có IR parser trong repo (chỉ có printer). Đã verify
  print-determinism (printModule 2 lần == nhau). Round-trip thật cần build parser mới → để lại (ngoài bug-hunt).
- [x] Tracing vs eager: mở rộng op-set qua block "section A" trong `fuzz-differential.test.js` (400 prog scalar-arg
  ops: reduce/argreduce/transpose/softmax, neg-dim+keepdim+multi-dim, cpu+wasm).
- [x] `_BUILDER_METHOD_MAP` / `_SCALAR_ARG_SPEC`: fuzz scalar-arg (dim âm, keepdim, multi-dim) — QUÉT XONG, ra 2 BUG argmax/argmin (dưới).
- [x] Invariant: verifier PASS sau khi trace mọi graph hợp lệ — section-A fuzzer chạy `verifyModule` trên mọi prog trace (sạch).

### Bug đã fix đợt 13 (2026-06-11) — fuzz section A (frontend scalar-arg): argmax/argmin neg-dim + keepdim
> Fuzz `_SCALAR_ARG_SPEC` scalar-arg space (dim âm/keepdim/multi-dim). Differential eager-vs-compiled + verifier
> invariant + INDEPENDENT oracle (vì eager cũng sai ở keepdim → differential bỏ sót).
- **BUG 1 — argmax/argmin dim ÂM không reduce (sai SHAPE):** `builder.argmax/argmin` (`ir/graph/builder.js`)
  KHÔNG normalize axis âm trước khi build (khác convention các method khanh — softmax/concat ở dòng 353/358/392
  đã normalize). `inferArgReduceTypes` (`ops/reduction.js`) so `i === axis` với axis=-1 → không khớp dim nào →
  KHÔNG giảm rank → ra tensor cùng rank input (vd `argmax([2,2,3],-1)` ra `[2,2,3]i32` thay vì `[2,2]`). Eager
  ĐÚNG (oracle) → differential bắt qua shape mismatch. Fix: `const dim = axis<0 ? input.type.rank+axis : axis;`
  trong cả argmax + argmin.
- **BUG 2 — argmax/argmin keepdim GHI SAI Ô (sai VALUE), cả eager+cpu+wasm:** lowering `registerArgReduce`
  (`passes/lowering/rules/reduction.js`) index `outBuf` bằng `accNest.indices` = CHỈ spatial-dims, nhưng khi
  keepdim outBuf có rank = input rank (chèn 1 ở reduceDim) → thiếu 1 index → offset lệch khi reduceDim KHÔNG phải
  trailing dim (2D dim-cuối "may" đúng vì dim size-1 ở cuối → test cũ `argmax_keepdim` bỏ sót). Fix: helper
  `outIndicesFor(nest)` — khi keepDims chèn `IntImmNode(0)` ở vị trí reduceDim, spatial ivs ở các vị trí còn lại;
  non-keepdim giữ nguyên. Áp cho cả init-store + acc load/store.
- Test: `differential-nn.test.js` block "argmax/argmin: negative dim + keepdim vs independent oracle" (eager+cpu+
  wasm vs `refArg` thuần JS, 4 shape × mọi dim±× keepdim) + `fuzz-differential.test.js` block "frontend (section A)".
  Revert builder fix → 48 RED + section-A SHAPE RED; revert lowering fix → 60 RED (keepdim). Full
  e2e+compiler+tensor+nn+dispatcher+lightning+autograd+backend **3375 pass** (trừ 2 = webgpu GPU segfault, blocker sẵn).

### B. AD — autodiff (`compiler/ad`, `ad/vjp_rules/*`)
- [ ] VJP từng op vs numerical gradient (finite-diff) cho mọi op có rule: arithmetic/unary/reduction/linalg/shape/composite.
- [ ] BackwardGraphBuilder: forward+backward graph → verifier PASS, không dangling saved-value.
- [ ] RematPolicy: bật/tắt remat phải cho cùng gradient (metamorphic).
- [ ] compileWithBackward vs eager autograd trên random graph (saved-value mapping by-source).
- [ ] Double-backward / gradient của gradient.

### C. Passes — simplify (`simplify/*`)
- [ ] Metamorphic: chạy graph CÓ vs KHÔNG mỗi pass (constant_fold / algebraic / cse / dce) → kết quả eval phải bằng nhau.
- [ ] constant_fold vs eager eval của chính subgraph đó (oracle độc lập, không cần eager toàn graph).
- [ ] CSE: số op sau CSE ≤ trước; eval không đổi; không gộp nhầm op có side-effect/khác dtype.
- [ ] DCE: không xoá op có result escape (đã từng có bug class này — `project_dangling_operand_refs`).
- [ ] algebraic: từng rewrite rule là property (x+0==x, x*1==x, x*0==0, double-neg...) — fuzz input ngẫu nhiên check đẳng thức.

### D. Passes — fusion (`fusion/*`)
- [ ] Metamorphic: CÓ vs KHÔNG fusion → eval bằng nhau (dominator / epilogue / multi_output / merger).
- [ ] Invariant: sau fusion verifier PASS, không escaping-use bị nuốt (hasEscapingUse guard).
- [ ] fusion_cost / fusion_groups: fuzz graph để cost model không chọn group sinh cycle (dep cycle check).
- [ ] Số kernel sau fusion ≤ trước; output set không đổi.

### E. Passes — decompose / canonicalize / rewrite
- [ ] Decomposition: op phức (softmax/layernorm/gelu...) bung ra primitive → eval == op gốc (metamorphic).
- [ ] Canonicalize idempotent: chạy 2 lần == chạy 1 lần (fixed-point).
- [ ] Rewrite pattern (`rewrite/pattern.js`): match→replace bảo toàn ngữ nghĩa, không vòng lặp vô hạn.

### F. Passes — lowering (graph→tensor→LIR)
- [ ] Mỗi lowering rule (elementwise/linalg/pooling/reduction/shape/resize/quantization/control_flow/layout) vs eager op.
- [ ] graph_to_tensor + tensor_to_lir: round-trip shape/stride, LIR verifier PASS (`ir/lir/verifier.js`).
- [ ] Scanner/flatten (`ir/lir/*`): fuzz nested loop/scope không mất biến, không sai scope.

### G. Passes — layout
- [ ] layout_transform: transform rồi inverse phải == gốc (round-trip).
- [ ] Metamorphic: layout policy NCHW vs NHWC → cùng kết quả số.
- [ ] layout_analysis: layout gán không mâu thuẫn giữa producer/consumer.

### H. Passes — memory (`memory/*`)
- [ ] buffer_liveness / buffer_assignment: hai buffer overlap thời gian sống KHÔNG được chia ô nhớ (invariant) — fuzz graph rồi assert no-alias-while-live.
- [ ] inplace_analysis: chỉ inplace khi input không còn dùng sau đó; differential phải == không-inplace.
- [ ] rematerialization: bật/tắt remat → cùng kết quả (metamorphic), peak-mem giảm.
- [ ] memory_planning: tổng mem ≤ tổng nếu không reuse; không ghi đè buffer còn live.

### I. Passes — partition
- [ ] partition_pass: ghép các partition lại phải == graph gốc (metamorphic); không cắt giữa op có data-dep sai chiều.
- [ ] partitioner: fuzz để không sinh partition rỗng / cycle giữa partitions.

### J. Passes — quantization
- [ ] calibration / observer → quantize → dequantize: sai số trong ngưỡng (property).
- [ ] quantization_pass vs reference quant eager; quantization_sensitivity không chọn nhầm layer.
- [ ] quantization_patterns matching: fuzz pattern conv-bn-relu... đảm bảo match đúng cấu hình.

### K. Schedule (`compiler/schedule/*`)
- [ ] schedule rules (split/reorder/fuse/tile/bind): mỗi primitive bảo toàn ngữ nghĩa (metamorphic vs unscheduled).
- [ ] validator (`schedule/validator.js`): mọi schedule fuzz ra phải hợp lệ hoặc bị reject sạch (không crash).
- [ ] dep_analysis: reorder không vi phạm data-dependency (invariant).
- [ ] trace replay (`schedule/trace.js`): replay trace → cùng schedule state (round-trip).

### L. Analysis (`compiler/analysis/*`)
- [ ] shape_analysis / sym_int: shape suy ra == shape runtime thực tế trên nhiều concrete shape.
- [ ] dtype_analysis: dtype suy ra == dtype eager.
- [ ] alias_analysis / use_def / liveness / dominance: so với brute-force reference trên graph nhỏ ngẫu nhiên (oracle độc lập).
- [ ] memory_effect: phân loại pure/side-effect đúng.

### M. Autotune (`compiler/autotune/*`)
- [ ] search_space: mọi config sinh ra phải compile+chạy đúng (không config sinh kernel sai số).
- [ ] cost_model vs benchmark thực: ranking không đảo ngược hoàn toàn (property loose).
- [ ] workload_key: cùng workload → cùng key, khác workload → khác key (không collision/false-share trong tuning_db).
- [ ] tuning_db: ghi/đọc round-trip; cache hit trả đúng kernel.

### N. Backend codegen (`backend/{cpu,wasm,gpu,webgpu}`)
- [ ] Cross-backend differential: CPU vs WASM (đang có) — mở thêm GPU/WebGPU khi có máy.
- [ ] Static kernel lint (đã có `tests/_utils/kernel_lint.js`): mở rộng rule (uninit read, OOB index, lane mismatch SIMD, dtype mismatch f32/i32, accumulator sai).
- [ ] WASM: SIMD vectorize vs scalar phải == nhau (metamorphic, bật/tắt `_emitVecExpr`).
- [ ] Marshalling host (`runtime.js runWasmKernel`): mọi dtype/typed-array vào-ra đúng (round-trip).

### O. Runtime / pipeline
- [ ] Opt-level differential: O0 vs O1/O2 (số pass khác nhau) → cùng kết quả số.
- [ ] Pipeline đầy đủ: cùng 1 graph, mọi target → cùng kết quả (đến tol).
- [ ] Cache: jit_cache cùng key trả cùng entry; khác shape/dtype → key khác (đã từng có bug i32 reduce-init xuyên qua đây).

### Oracle tổng (áp cho mọi tầng)
1. **Differential** — eager vs compiled, backend vs backend, opt-level vs opt-level.
2. **Metamorphic** — bật/tắt 1 pass phải bảo toàn ngữ nghĩa.
3. **Round-trip** — print/parse, transform/inverse, serialize/deserialize.
4. **Invariant/verifier** — sau MỖI pass: verifier PASS, SSA hợp lệ, no dangling, no live-overlap.
5. **Property-based** — đẳng thức đại số, bound sai số, monotonic cost.
6. **Crash-only** — chỉ cần không throw/không hang trên input hợp lệ (bắt assert/infinite-loop).

### Blocker production (không phải bug đếm được)
- [ ] GPU / WebGPU backend chưa verify được (không có máy GPU). Đây là blocker lớn nhất.
- [ ] Reference (eager) không độc lập — eager từng có bug, differential có thể bỏ sót khi cả 2 cùng sai.
  → Cần oracle độc lập (numerical-diff cho grad, brute-force cho analysis, metamorphic cho pass).

## Ghi chú vận hành
- Coding rules: ko comment, ko hardcode, ko O(n²); fix xong viết test, đặt vào file test có sẵn, ko có mới tạo mới.
- KHÔNG `git checkout` file đang có uncommitted work (đã mất việc 2 lần) — backup `cp` trước.
- Revert-test mỗi fix: bỏ fix phải thấy test RED, apply lại GREEN.
