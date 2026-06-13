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
> - **K** schedule: scheduling on/off vs eager → ra 3 BUG WASM (đợt 14) + 1 BUG fused matmul→reduce mismatched
>   parallel-extent ghi tràn buffer (đợt 29, xem dưới). Fix xong + unit test `tests/backend/wasm/wasm-parallel.test.js`.
> - **L** analysis: use_def topo + opUsers vs brute-force, dominance postDom vs idom-chain, shape/dtype-infer==eager. SẠCH.
> - **M** autotune: workload_key deterministic + target-sensitive, tuning_db serialize round-trip, autotuned-compile==eager. SẠCH.
> - **N** backend codegen: SIMD vs scalar metamorphic (WASM simd on/off) + dtype marshalling round-trip
>   (f32/f64/i32/i16/i8/ui8/f16/bf16 cpu+wasm). SẠCH.
> - **O** runtime/pipeline: opt-level differential O0/O1/O2 == eager (cpu+wasm) 1680 case + multi-output/kernel-reuse/
>   run-vs-async/modules/dynamic-shapes/cache-key (đợt 30). SẠCH.
> → Đã quét HẾT A–O. Tổng: tầng K = 4 bug WASM scheduling (đợt 14 ×3 + đợt 29 ×1), các tầng khác vững từ trước.

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
### Bug đã fix đợt 29 (2026-06-12) — tầng K (schedule) WASM: fused matmul→reduce ghi tràn buffer khi 2 parallel-loop khác extent
- TRIGGER chính xác: `reduce(matmul(A[M,K], B[K,N]), axis=0)` → out[N], với scheduling+SIMD bật. Lệch CHỈ khi
  matmul-row-extent M == vector-width (4) VÀ M != reduce-output-extent N (vd M=4, N=5). Kết quả = ref + hằng số đều
  mọi cột (vd +7 = matmul_out[0,0]). M=3 (dưới ngưỡng parallel) OK; M=5=N (extent khớp) OK.
- ROOT: codegen WASM bind MỌI `ForKind.PARALLEL` loop vào CÙNG `_par_start/_par_end`, nhưng runtime chỉ phân hoạch
  ĐÚNG MỘT trục extent = `_parallelExtent`. Scheduler đánh dấu CẢ matmul-row-loop (extent M) LẪN reduce-output-loop
  (extent N) là PARALLEL. `_isParallelSafe` thấy >1 parallel → `poolSafe=false` → chạy single-worker với
  parEnd=`_parallelExtent`(=N=5). Loop có extent THẬT < parEnd (matmul M=4) lặp dư 1 vòng → ghi `matmul_out[4,0]`
  = offset ngay sau buffer [M,N] = buffer hằng-số-init (scalarConstant(0)) liền kề → init reduce bị +matmul_out[0,0].
  Chẩn đoán quyết định: đặt init=1000 → kết quả = ref+1007 (init đọc đúng, +7 là tràn riêng).
- FIX: `_visitFor` (codegen.js) — PARALLEL loop chỉ tiêu thụ `_par_start/_par_end` khi `extent === _parallelExtent`;
  loop PARALLEL có extent tĩnh KHÁC → phát serial loop `_emitForLoop(0..extent)`. poolSafe-path (đúng 1 parallel,
  extent luôn khớp) KHÔNG đổi. Revert-test: bỏ guard → simd0=83 (sai) vs ref=76.
- TEST: tests/backend/wasm/wasm-parallel.test.js — describe "fused matmul->reduce with mismatched parallel extents"
  (repro [4,2]@[2,5] + grid M×N×K). LIMITATION cũ đợt 14 (matmul→reduce extent != vec-width) ĐÃ GIẢI QUYẾT tại đây.

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
- [x] VJP từng op vs numerical gradient — fuzz rộng (unary đầy đủ + binary div/maximum/minimum + softmax/log_softmax + reduce + view) × {compiled-cpu/wasm, remat on/off, joint}. Ra 2 BUG default-path (rsqrt, log_softmax+reduce fusion) + 2 BUG joint (reduce-grad + fusion-cycle). TẤT CẢ ĐÃ FIX. Fuzz lại 561 prog × 4 oracle = 0 fail.
- [x] RematPolicy bật/tắt metamorphic — quét sạch (remat on/off cùng grad, trừ NaN/Inf blowup do data, không phải bug).
- [x] compileWithBackward joint vs separate (mode metamorphic) — XONG, joint giờ khớp separate+numerical (fix đợt 15+16).
- [ ] BackwardGraphBuilder verifier no-dangling-saved-value — chưa dựng verifier riêng (differential + numerical đã bắt giá trị).
- [ ] compileWithBackward vs eager autograd random graph (saved-value by-source).
- [ ] Double-backward / gradient của gradient (feature `createGraph` chưa có).

### Bug đã fix đợt 15 (fuzz section B — AD numerical+metamorphic) — 2 bug DEFAULT-PATH
> Fuzz 500 prog × {numerical finite-diff, remat on/off, joint, wask} oracle. Tới đâu fix tới đó (ko comment/hardcode/O(n²)).
- **rsqrt KHÔNG có VJP rule** (`compiler/ad/vjp_rules/unary.js`): compiled backward trả grad RỖNG `[]` (eager cũng ko có autograd node → throw "does not require grad"). rsqrt là op phổ biến (RMSNorm). Fix: thêm `registerVJPRule('rsqrt')` — dy/dx = -0.5·y³ (y=result), chỉ cần result (remat-friendly). cpu+wasm khớp numerical.
- **broadcast_in_dim giữ-VIEW khi user là `fusion` nhưng TRONG fusion body value bị `reduce` đọc → reduce bỏ qua broadcast dim, ra sai SHAPE+VALUE** (`compiler/passes/lowering/graph_to_tensor.js`): `BROADCAST_VIEW_SAFE` chứa `'fusion'` → broadcast feeding 1 fusion luôn giữ view, NHƯNG check chỉ nhìn op fusion ngoài, ko nhìn op TRONG thân fusion. log_softmax VJP làm `reduce(grad,[axis])` với grad = broadcast của upstream-reduce-grad; khi op sau log_softmax là `reduce` trên TRỤC KHÁC (vd `sum(log_softmax(x,1),2)`), backward sinh kHorizontal-fusion chứa reduce đọc broadcast-view → reduce hạ theo shape VẬT LÝ [2,3,1]→[2,1] thay vì LOGICAL [2,3,2]→[2,2] → consumer đọc OOB → số hạng correction mất → grad = passthrough (comp=1 thay vì đúng). softmax VJP ko dính (reduce 1 product tươi, ko reduce grad trực tiếp). Fix: helper `broadcastViewSafeForUser(value,user)` đệ quy — khi user là fusion, map operand→block-arg rồi check MỌI inner-user của block-arg cũng broadcast-safe (reduce → unsafe → materialize). Ảnh hưởng MỌI VJP reduce-broadcasted-grad (log_softmax/softmax-family) ở default path + wasm.
- Test: `differential-backward.test.js` block "rsqrt VJP + reduce-after-log_softmax fusion vs numerical" (AD_BWD, 7 case × cpu+wasm, oracle numerical độc lập, data positive-domain cho rsqrt/log). Revert rsqrt → 6 RED; revert broadcast-view → 6 RED.

### Bug đã fix đợt 16 (joint mode = forward+backward fused trong 1 graph) — 2 BUG + 1 fusion-pass bug
> Joint mode (`compileWithBackward(...,{mode:'joint'})`) sai ~166/500 fuzz so với separate+numerical. Fix cả AD-layer lẫn fusion-pass.
- **BUG 1 — joint THIẾU `reduceGradToOperandShape`** (`compiler/ad/joint_builder.js`, cả `build` + `_buildCheckpointed`): grad của operand bị broadcast ko sum về shape gốc (separate `backward_builder.js` có, joint quên). Vd `mul(add(x[2,3], y[3]), x)` → grad_y joint ghi 6 phần tử vào buffer 3-ô. Fix: export `reduceGradToOperandShape` từ backward_builder, joint dùng chung (giống separate).
- **BUG 2 (fusion-pass, catalog D) — horizontal fusion gom op tạo CYCLE giữa 2 group** (`compiler/passes/fusion/fusion_groups.js`): khi vá BUG 1, lộ ra. `buildHorizontalGroups` gom 2 op cùng shape nếu ko phụ thuộc TRỰC TIẾP — nhưng (a) bỏ sót phụ thuộc GIÁN TIẾP (transitive), (b) ko thấy cycle XUYÊN 2 group. Vd joint `mul(mean(x,0)→scalar, y)`: group_A={mean(fwd), reduce(bwd-grad)} + group_B={mul(fwd), mul(bwd)} — mul-fwd(B)←mean(A) và reduce(A)←mul-bwd(B) → CYCLE → IR có `%?` dangling → forward joint ra **0**. Fix 2 phần:
  1. `_anyGroupMemberDependent` + `_transitivelyDependent` (tái dùng `_dependsOnOps` transitive reachability) thay check `_hasDependency` trực tiếp → ko gom op có path gián tiếp (chống vi phạm thứ tự TRONG group).
  2. `_condensedHasCycle(func, opToRep)` — condense mỗi group thành super-node, DFS 3-màu phát hiện cycle O(V+E); `buildAllGroups` thử thêm từng horizontal group, nếu tạo cycle giữa các group thì BỎ group đó (horizontal fusion là optional, bỏ luôn an toàn). Chống cycle XUYÊN group.
- Test: `differential-backward.test.js` block "joint-mode backward ... (fusion-cycle guard)" (JOINT_BWD 6 case: broadcast-row/col, maximum-bcast, scalar-reduce mean→scalar, chain-scalar-reduce, logsoftmax→sum; joint+separate đều vs numerical) + `tests/compiler/fusion/groups-fusion.test.js` "rejects ... TRANSITIVELY dependent". Revert: joint-reduce-grad → 3 RED; cycle-guard → 2 RED; transitive-check → 1 RED.
- Fuzz lại 561 prog × {numerical, remat, joint, wasm} = **0 fail**. Full e2e+compiler+autograd+backend+nn+dispatcher+tensor+tracing pass (trừ 2 webgpu GPU-segfault + 2 transformer Tera "Unknown name transpose" — CẢ HAI blocker pre-existing, ko liên quan: stash changes của tao vẫn fail y hệt).

### C. Passes — simplify (`simplify/*`)
- [x] Metamorphic differential (800 prog identity-rich eager-vs-compiled cpu+wasm) + property special-value sweep. Ra 1 BUG-class: 5 algebraic rewrite UNSOUND cho float (đợt 17). constant_fold/cse/dce/các algebraic khác: SẠCH.
- [x] CSE: quét sạch (op-count giảm, eval==eager, hash gồm dtype+attrs nên ko gộp nhầm).
- [x] DCE: quét sạch (ko xoá escaping-use; differential identity-rich + verifier sau pass).
- [x] algebraic property: x+0/x*1/x-0/x/1/double-neg/add-neg→sub... SẠCH; x-x/x÷x/x*0/exp∘log/log∘exp UNSOUND → fix (đợt 17).
- [ ] constant_fold vs eager subgraph oracle độc lập — chưa dựng riêng (differential pipeline đã cover value-correctness).

### Bug đã fix đợt 17 (fuzz section C — simplify) — 5 algebraic rewrite UNSOUND cho float
> Differential identity-rich (eager vs compiled) sạch trên giá trị thường; property special-value (NaN/Inf/0) ra divergence.
> Vi phạm bất biến eager==compiled (cùng class P2 NaN/Inf). User chọn: fix cho IEEE-sound.
- **Root**: 5 rewrite áp dụng vô điều kiện cho float:
  - `SubSelf` x−x→0: SAI khi x=±Inf/NaN (Inf−Inf=NaN, ko phải 0).
  - `DivSelf` x/x→1: SAI khi x=0 (0/0=NaN), x=±Inf (Inf/Inf=NaN), x=NaN. (Cả int cũng sai: x=0 → div-by-zero→0≠1.)
  - `MulZero` x*0→0: SAI khi x=Inf/NaN (Inf*0=NaN). (latent — chỉ fire khi operand là constant-0 thật.)
  - `ExpLog` exp(log x)→x: SAI khi x<0 (log(x)=NaN→exp=NaN, ko phải x).
  - `LogExp` log(exp x)→x: SAI khi exp overflow (log(Inf)=Inf, ko phải x).
- **Fix** (`patterns.js` + `algebraic.js` + `ops/unary.js`):
  - `SubSelf`/`MulZero`: guard `match` bằng `isDtypeInt(result.dtype)` → CHỈ fire cho integer (x−x=0, x*0=0 luôn đúng, ko NaN/Inf). Vẫn tối ưu cho int.
  - `DivSelf`/`ExpLog`/`LogExp`: bỏ đăng ký (unsound cho float, ko guard an toàn được; DivSelf unsound cả int x=0). Gỡ khỏi `_algebraicPatterns` + gỡ `getCanonicalizationPatterns` của op `exp` (đăng ký ExpLog ở op-level — đây là chỗ canonicalize tái áp dụng dù đã gỡ khỏi algebraic). LogExp/DivSelf chỉ ở algebraic.
  - Sound rewrite GIỮ NGUYÊN: AddZero/SubZero/MulOne/DivOne/DoubleNeg/MulNegNeg/AddNegToSub/SubNegToAdd/Transpose²/Reshape²/DoubleConvert (đều đúng IEEE).
- **Test**: `differential.test.js` block "algebraic ... IEEE-sound: special values" (7 case × cpu+wasm, eager==compiled cho div/sub-self 0/Inf/NaN + exp(log neg) + mul*0-Inf, so bit-NaN). Cập nhật unit test cũ assert hành vi MỚI: `arithmetic-simplify`/`inverse-simplify`/`shape-simplify`/`arithmetic-canonicalize`/`inverse-canonicalize`/`compare-pad-slice-canonicalize` — float→KHÔNG simplify, mechanics (broadcast/scalar/multi-consumer) chuyển sang i32 (vẫn fire, vẫn đúng). Revert fix → 13 RED. Full compiler+e2e+backend+autograd+tensor+dispatcher **2535 pass** (trừ 2 transformer Tera = blocker src/cli pre-existing).

### D. Passes — fusion (`fusion/*`)
- [x] Metamorphic across strategy {off, xla, dominator, +epilogue} vs eager (700 prog elementwise+reduce+broadcast+matmul+transpose × cpu+wasm). xla/dominator SẠCH. epilogue → ra 1 BUG (đợt 18).
- [x] fusion_cost / fusion_groups cycle — ĐÃ fix ở đợt 16 (`buildHorizontalGroups` transitive-dep + `_condensedHasCycle` condensed-graph check).
- [x] hasEscapingUse guard (epilogue): còn nguyên, ko escaping-use bị nuốt (quét sạch).
- [ ] (ghi chú) epilogue fusion mặc định CHỈ bật cho GPU target (`enableEpilogueFusion`); CPU/WASM gate off. Path GPU ko verify được (blocker no-GPU).

### Bug đã fix đợt 18 (fuzz section D — fusion) — LIR accumulator hoist chỉ số phụ thuộc loop-var
> Metamorphic fusion strategy sạch cho xla/dominator. Ép epilogue fusion ON trên CPU (`CPUTarget({enableEpilogueFusion:true})` + `fusion:{epilogue:true}`) để test `fused_dot_epilogue` lowering → ra BUG codegen.
- **Repro**: `add(matmul(x,w), bias)` ép epilogue → kernel sinh `let _acc_1 = buf_7[ep0*5 + ep1_20]` TRƯỚC vòng lặp `for ep1_20` → `ReferenceError: ep1_20 is not defined`. (Epilogue lowering + TensorIR ĐÚNG; bug ở scheduler→LIR.)
- **Root**: `detectAccumulator` (`passes/lowering/tensor_to_lir.js`) kiểm tra bất biến loop-var SAI: so `forNode.loopVar.name` (vd `ep1_20`) với `store.indices` dùng tên BLOCK ITER VAR (`epv1_21`, bind `epv1_21 = ep1_20` qua BlockRealize). Vì tên khác nhau, `storeKey.includes('$ep1_20')` = false → tưởng nhầm vòng epilogue (in-place `buf[i,j] = buf[i,j] + bias[j]`) là REDUCTION → tạo `LIRAccumulatorNode` hoist init-load `buf[i,j]` ra ngoài vòng `j` → `j` chưa khai báo. (Codegen `_detectReductionAcc` resolve alias trước nên ĐÚNG; chỉ scheduler-LIR sai.)
- **Fix**: tính `outerIndices` (resolve block iter-var→binding) TRƯỚC, check bất biến trên `resolvedKey = indicesKey(outerIndices)` thay vì raw `storeKey`. Reduction thật (matmul: store index = ls/rs bound to outer loops, ko chứa loop-var c0) vẫn fire đúng; epilogue (index chứa loop-var sau resolve) → bail → hạ thành vòng lồng thường (đúng). Bug chung cho MỌI in-place `+=` có index phụ thuộc loop-var qua BlockRealize, ko riêng epilogue.
- **Test**: `differential.test.js` block "epilogue fusion (forced on) matches eager" (6 pattern: matmul+bias/+relu/+tanh/+scale/+exp+neg, ép epilogue, cpu+wasm vs eager). Revert (check raw `store.indices`) → matmul_bias RED. Full compiler+e2e+backend+autograd+tensor+nn+dispatcher **2658 pass, 0 fail**.

### E. Passes — decompose / canonicalize / rewrite — QUÉT SẠCH (đợt 19), KHÔNG bug
- [x] Decomposition vs independent reference (15 activation variant incl. custom alpha/slope × 80 seed × cpu+wasm): SẠCH. Composite op (softmax/gelu/silu/sigmoid/elu/celu/selu/mish/hardswish/hardsigmoid/leaky_relu/layer_norm...) KHÔNG có direct lowering → LUÔN decompose; oracle = reference formula thuần JS (đa số op này ko có trên eager nên formula là oracle độc lập đúng). Hạ đúng giá trị.
- [x] Canonicalize idempotent: 200 graph canonicalize-rich, chạy 2× == 1× (run thứ 2 luôn UNCHANGED) — SẠCH, không oscillation/non-convergence. 69/200 thực sự CHANGED ở run đầu (test ko rỗng).
- [x] Rewrite framework (`applyPatterns` maxIterations=10): fixed-point đạt, không vòng lặp vô hạn (idempotency test cover).
- **Coverage thêm (ko phải bug fix):** test decompose activation TRƯỚC CHỈ check op-structure, KHÔNG check số. Thêm block "activation decomposition: end-to-end numerical correctness vs reference" vào `tests/compiler/decompose/activation-decompose.test.js` (12 activation × cpu+wasm, compileGraph+run vs reference formula) — lấp lỗ hổng số học cho elu/celu/selu/mish/hardswish/hardsigmoid/leaky_relu (ko có trên eager nên differential ko cover). 48/48 pass.

### Bug đã fix đợt 22 (control_flow lowering — `while` HOÀN TOÀN HỎNG) — đào sâu sau khi user hỏi "nãy giờ ko bug à"
> E/F/G ban đầu báo sạch vì fuzz nhẹ tay + vùng test kỹ. Đào lại các path CHƯA đụng: control_flow (if/while), resize bilinear, gather/scatter.
> resize bilinear: khớp reference. `if`: đúng cpu+wasm. **`while`: HỎNG hoàn toàn — vòng lặp vô tận + giá trị sai, chưa từng có e2e test chạy nó.**
- **Repro**: `whileOp([acc=0, i=0], cond: i<K, body: acc+=x, i+=1)` → kernel sinh `while (_wcond)` (test OBJECT typed-array luôn truthy → vô tận), `buf[0]=(0<3)` (so hằng khởi tạo, ko phải counter), `acc=x` (ko cộng dồn). out=[0,0,0,0].
- **3 bug**:
  1. **Lowering** (`passes/lowering/rules/control_flow.js`): loop-state dùng THẲNG buffer hằng khởi tạo (`loopBufs=inputs`). Constant-buffer opt (a) inline đọc thành hằng init (`0<3`, `add(0,x)`), (b) BỎ QUA store vào constant buffer (codegen `_visitBufferStoreNode` skip nếu `_constantBuffers.has`) → copy-back state thành no-op → vô tận. Fix: dùng `outputs[i]` (buffer kết quả, mutable, nối return) làm loop-state + copy init-value vào trước vòng (`initStmts` + `SeqNode`).
  2. **CPU codegen** (`backend/cpu/codegen.js:326`): `while (${condVar.name})` → test mảng (luôn truthy). Fix `[0]`.
  3. **WASM codegen** (`backend/wasm/codegen.js:875`): `(local.get $condVar)` nhưng condVar là MEMORY buffer ko phải local → "invalid local index". Fix: `_emitAddr(condVar,[]) + _emitLoadOp(dtype)` (load từ memory như BufferLoadNode).
- **Test**: `tests/compiler/lowering/control-flow-lowering.test.js` block "control flow end-to-end execution vs reference" (while K=0/1/3/5 acc=K*x + if then/else, cpu+wasm). Trước đây control-flow test CHỈ structural (check WhileNode trong IR, KHÔNG chạy) → bug sống sót. Full **2769 pass, 0 fail**.
- GHI CHÚ: `while` ko có frontend-op user thật (latent), nhưng là bug đúng nghĩa (hoàn toàn ko chạy được). gpu/webgpu codegen có thể còn bug condVar tương tự (ko verify được, no-GPU).

### F. Passes — lowering (graph→tensor→LIR) — QUÉT (đợt 20) + bug control_flow (đợt 22)
- [x] Mỗi lowering rule vs reference/eager: 105 graph parameterized (elementwise/reduce(5 type×axis×keepdim)/argmax-argmin/shape(transpose/reshape/slice-step/pad/concat/iota/broadcast)/matmul/pool2d(max-avg×k×s×pad×count_include_pad)/resize(nearest-bilinear up&down)/conv(stride×pad×dilation×groups)) × cpu+wasm → compile crash-only SẠCH; pool2d value vs JS reference SẠCH.
- [x] graph_to_tensor + tensor_to_lir + **LIR verifier**: lower 105 graph → `verifyLIR` = 0 error MỌI graph. **Phát hiện: pipeline KHÔNG bao giờ chạy `verifyLIR`** (chỉ TensorVerifier trên PrimFunc, ko verifyLIR trên LIR) — verifyLIR sẽ bắt được bug đợt 18 (unbound `ep1`). Dùng làm invariant oracle.
- [x] Scanner/flatten (`scanMetadata` chạy trong lowerToLIR): exercise qua verifyLIR + compile, ko mất biến/sai scope.
- **Coverage thêm (ko phải bug):** (1) `tests/compiler/ir/lir/verifier.test.js` — block "verifyLIR on real lowered graphs across lowering-rule categories" (15 case đại diện mọi rule, lower thật → verifyLIR clean; trước đây verifier chỉ test trên LIR dựng tay). (2) `tests/compiler/lowering/pooling-lowering.test.js` — block "pool2d end-to-end value vs reference" (18 config max/avg×k×s×pad×count_include_pad × cpu+wasm). 252/252 pass.

### G. Passes — layout — QUÉT SẠCH (đợt 21), KHÔNG bug
- [x] Metamorphic layout ON vs OFF (`{optimization:{layout:true/false}}`) trên matmul/double-matmul/matmul-bias-relu/conv/conv-groups/conv-relu-pool/pool/reduce/chain × cpu+wasm → kết quả số TRÙNG KHÍT (relErr<1e-5). Layout opt là semantics-preserving.
- [x] layout_transform + consumer: NON-trivial (matmul chèn 1, conv chèn 1, double_matmul chèn 2 `layout_transform`) NHƯNG vẫn đúng số → tổ hợp transform+dot/conv sound. (Explorer nghi lowering chỉ copy ko permute; empiric: end-to-end đúng nên ko phải bug — hoặc layout là no-op metadata consumer bỏ qua, hoặc indexing tự khớp.)
- [x] layout_analysis producer/consumer: round-trip (compose+identity canonicalize patterns) + metamorphic cover; ko mâu thuẫn (numerics bảo toàn).
- **Coverage thêm (ko phải bug):** `tests/compiler/layout/transform-layout.test.js` block "layout optimization is semantics-preserving: layout ON == layout OFF" (6 graph × cpu+wasm). Trước đây layout test chỉ check STRUCTURAL insertion ở pass-level, ko check end-to-end numerics on/off. 45/45 pass.

### H. Passes — memory (`memory/*`) — 1 BUG latent (đợt 23)
- [x] buffer_liveness / buffer_assignment no-alias-while-live invariant: fuzz 10 graph × align{64,128,256} × inplace{on,off}, inspect memory plan → **RA BUG: inplace dst chia ô nhớ với buffer interfering** (đợt 23).
- [x] inplace metamorphic (inplaceReuse on/off) + remat metamorphic (rematerialization on, budget=64) + alignment{64,256}: numerics TRÙNG KHÍT vs ref (inplace=off) cpu+wasm, SẠCH. Memory opt bảo toàn giá trị end-to-end.
- [x] rematerialization on/off: cùng kết quả (đã có ở B cho AD-remat; ở đây remat pass graph-level cũng sạch).

### Bug đã fix đợt 23 (memory — BufferAssignment inplace lifetime) — LATENT (no-alias-while-live invariant)
> Oracle độc lập = no-alias invariant trên memory plan: hai buffer interfering (live range chồng STRICT) KHÔNG được chia offset. Fuzz 10 graph → 18 vi phạm (`softmax buf_18[3,6]&buf_23[4,6]` cùng off0; `planner.interfere()=true`).
- **Root** (`passes/memory/buffer_assignment.js`): nhánh inplace (dst aliases src offset) `continue` mà KHÔNG thêm dst vào `active`/KHÔNG kéo dài lifetime ô nhớ. Inplace yêu cầu `src.lastUse <= dst.firstUse` (src chết trước dst), nên pool release ô nhớ của src theo lifetime NGẮN của src → buffer cấp sau (vd buf_23) tái dùng offset đó TRONG KHI dst inplace (buf_18) còn live → 2 buffer interfering cùng ô nhớ = corruption tiềm ẩn.
- **Fix**: precompute `effLastUse` — kéo dài lifetime hiệu dụng của src lên `dst.lastUse` (truyền qua chain inplace a→b→c). Release check dùng `effLastUse` thay `interval.lastUse` → ô nhớ của src giữ đến khi dst inplace chết → ko bị tái dùng sớm.
- **LATENT**: KHÔNG repro được corruption end-to-end (pipeline chuẩn chạy CSE/DCE/fusion trước _planMemory → cấu trúc buffer khác, né được; eager-differential softmax/long_chain + fusion on/off đều khớp). Nhưng là vi phạm invariant THẬT (chứng minh qua `interfere()` + no-alias check). Fix phòng ngừa corruption tương lai.
- **Test**: `tests/compiler/memory/assignment-memory.test.js` block "inplace destination extends aliased storage lifetime (no-alias-while-live)" (2 case: src→dst inplace + buffer cấp sau ko collide; chain a→b→c giữ ô nhớ tới dst cuối). Revert (gỡ effLastUse extension) → 2 RED + fuzz invariant 18 fails. Full **2769 pass, 0 regression**. Memory tests 67/67.

### I. Passes — partition — 1 BUG (đợt 24)
- [x] Metamorphic partition ON (CPU/CPU split qua opTargetOverrides) vs OFF: numerics phải khớp. Ra **BUG materialization mất regions** (đợt 24).
- [x] Structural invariants (no cycle partition-DAG / no empty / full coverage exactly-once): fuzz 6 graph × 6 split config → SẠCH.

### Bug đã fix đợt 24 (partition — materialization mất op regions) — off-by-default path
> Partition opt-in (cần ≥2 target). Metamorphic ON vs OFF (2 CPU target tên khác nhau → executable trên CPU, force split qua opTargetOverrides).
- **Repro**: graph có `reduce`/`fusion` (vd `sub(x, bcast(reduce(x,sum)))`, softmax, layernorm) + partition → "Graph verification failed: op 'reduce' expects 1 regions, got 0; op 'fusion' expects 1 regions, got 0".
- **Root** (`passes/partition/partition_pass.js:232` `PartitionMaterializationPass._materializePartitions`): khi clone op vào sub-function, `new Operation(opName, operands, resultTypes, attrs)` — THIẾU đối số thứ 5 (regions). Op có region (reduce combiner, fusion body, if/while) bị clone ra 0 region → IR invalid → compile fail. (Structural invariant SẠCH vì assignment đúng; chỉ materialization drop region.)
- **Fix**: `clonedRegions = op.regions.length>0 ? op.regions.map(r=>cloneRegion(r)) : null` rồi truyền vào `new Operation(...,clonedRegions)` — đúng pattern đã có ở `dominator_fusion.js`/`fusion_merger.js`. `cloneRegion` (operation.js) deep-clone block+remap block-args.
- **Test**: `tests/compiler/partition/pass-partition.test.js` block "partition materialization end-to-end numerics ... region-bearing ops keep their regions" (reduce_sum/reduce_bcast_sub/softmax/ew_chain, partition ON==OFF). Trước đây partition test chỉ structural (pass-level), ko execute partitioned graph có region → bug sống sót. Revert (regions=null) → 3 region-test RED (ew_chain pass). Full **2775 pass, 0 regression**.

### J. Passes — quantization — 1 BUG (đợt 25)
- [x] quantize/dequantize round-trip COMPILED vs reference math (`QuantizationParams.quantize/dequantize`): build trực tiếp qua `_buildOp`, compile+run cpu+wasm × {sym/asym, i8/ui8, nhiều range}. Lowering KHỚP CHÍNH XÁC reference (vsRef≈0). Out-of-range → clamp đúng. Ra **BUG canonicalize unsound** (đợt 25).
- [x] quantization_pass end-to-end (matmul quant on vs off): áp dụng quant đúng (maxerr~0.2, ko bị eliminate). quantized_dot/quantize/dequantize executable cpu+wasm.

### Bug đã fix đợt 25 (quantization — fake-quant bị canonicalize unsound xóa) — off-by-default path
> Quant opt-in. Oracle độc lập = `QuantizationParams.quantize/dequantize` (JS math). Build `quantize(x)→dequantize` trực tiếp, compile+run → ra IDENTITY (out=x chính xác) thay vì quantized round-trip (lossy).
- **Root** (`ir/graph/ops/quantization.js` dequantize op `getCanonicalizationPatterns`): đăng ký `DequantizeQuantizeIdentity` — match `dequantize(quantize(x)) → x` (xóa cả cặp). NHƯNG `dequantize(quantize(x)) = (round(x/s+zp)-zp)*s` ≠ x cho float x (LOSSY, làm tròn về grid int). Xóa nó = bỏ sai số lượng tử → **fake-quant (QAT/PTQ simulation) hỏng**. Đối xứng nhầm: `QuantizeDequantizeIdentity` (`quantize(dequantize(q))→q`) ĐÚNG (q đã là int, round-trip exact) nhưng `DequantizeQuantizeIdentity` (chiều ngược) SAI — copy-paste oversight.
- **Fix**: gỡ đăng ký `DequantizeQuantizeIdentity` khỏi op `dequantize` (giữ `QuantizeDequantizeIdentity` đúng trên op `quantize`, giữ `DequantizeFoldIntoDot`/`ConstantQuantize`). KHÔNG test nào phụ thuộc (49 quant test pass khi disable). matmul quant pass thật ko cần nó (vẫn áp quant đúng).
- **Test**: `pass-quantization.test.js` block "fake-quant (dequantize∘quantize) is LOSSY — must not be removed as identity" (compiled `dequantize(quantize(x))` khớp reference math + có sai số làm tròn > 0 = ko bị eliminate, cpu+wasm). Revert (re-add pattern) → 2 RED. Full **2777 pass, 0 regression**.
- GHI CHÚ: đây là quyết định soundness (giống đợt 17 algebraic). Bằng chứng mạnh là oversight: chỉ 1 chiều unsound, ko test phụ thuộc, lowering+refmath là spec mà canonicalize mâu thuẫn. Fix khớp spec.

### K. Schedule (`compiler/schedule/*`)
- [ ] schedule rules (split/reorder/fuse/tile/bind): mỗi primitive bảo toàn ngữ nghĩa (metamorphic vs unscheduled).
- [ ] validator (`schedule/validator.js`): mọi schedule fuzz ra phải hợp lệ hoặc bị reject sạch (không crash).
- [ ] dep_analysis: reorder không vi phạm data-dependency (invariant).
- [ ] trace replay (`schedule/trace.js`): replay trace → cùng schedule state (round-trip).

### L. Analysis (`compiler/analysis/*`) — QUÉT SẠCH (đợt 26), KHÔNG bug
- [x] use_def / post-dominance / liveness / memory_effect / alias vs **brute-force reference độc lập** trên 300 graph ngẫu nhiên (elementwise+view+reshape+transpose+reduce(region)+broadcast). SẠCH:
  - use_def: topo hợp lệ (op sau operand) + opUsers == brute scan toàn graph.
  - post-dominance: `postDominates(a,b)` == "a nằm trên MỌI path b→return" (brute enumerate path + intersect).
  - liveness: interval == [topo-idx def, max topo-idx use] (brute).
  - memory_effect: mọi op pure (elementwise/view/reduce) → hasSideEffect=false.
  - alias: 2 value share base qua VIEW-chain → mayAlias=true (sound, ko miss alias).
- [x] shape/dtype: inferResultTypes đã validate gián tiếp qua MỌI compile (B-K differential) + buffer sizing. (shape inference sai → compile sai shape, đã bắt từ trước.)
- **Coverage thêm (ko phải bug):** `tests/compiler/analysis/memory-effect.test.js` block "graph analyses vs independent brute-force" (60 seeded real graph, use_def/post-dom/liveness/memory_effect). Trước đây use_def/dominance/liveness KHÔNG có test trực tiếp (chỉ mock + dùng gián tiếp qua checkpoint). Brute-force oracle = regression net thật. (Bug trong oracle test của tôi lúc đầu: filter `return` ko nhất quán → 300 false fail; sửa → 0.)

### M. Autotune (`compiler/autotune/*`) — QUÉT SẠCH (đợt 27), KHÔNG bug
- [x] Metamorphic autotune ON vs OFF (`{scheduling:{enabled:true,autotune:true}}`) — 54 graph (matmul/ew/reduce/mm+reduce) × {random,evolutionary} × nhiều seed × cpu+wasm, **shape non-power-of-2 (5,7,13,11,3)** để ép tiling/vectorization remainder. SẠCH (autotuned == baseline). Ép 80 run trên edge K-limitation (mm→reduce vectorize wasm non-divisible) × 20 seed → 0 fail (autotune ko kích bug K, validator loại config xấu / sketch ko sinh config đó).
- [x] workload_key: same workload→same key, khác shape/target→khác key (ko collision). (đã có test + xác nhận lại.)
- [x] tuning_db serialize/deserialize round-trip + best-score lookup. (gap: chưa có test round-trip → thêm.)
- **Coverage thêm (ko phải bug):** `tests/compiler/autotune/autotuner.test.js` — block "autotune end-to-end ... non-power-of-2 shapes (tiling remainders)" (4 case × cpu+wasm × 2 strategy, autotuned==baseline) + "TuningDatabase serialize/deserialize round-trip". Stress test cũ chỉ power-of-2; tuning_db round-trip trước đây ko có test. 16/16 autotune pass.
- GHI CHÚ: autotune drive cùng Schedule API như tầng K (đã fix 3 bug WASM scheduling đợt 14). Metamorphic on/off ở đây exercise lại path đó qua nhiều config → vẫn sạch sau fix K.

### N. Backend codegen — 1 BUG (đợt 28: WASM SIMD i32 dtype với guarded loop)
- [x] WASM SIMD on vs off metamorphic (`WasmTarget({simd:false})` vs default) — ew/reduce/compare-select/i32 × non-power-of-2 extent (3,4,5,7,8,13,16,17). **QUAN TRỌNG: SIMD CHỈ phát ra khi `scheduling:{enabled:true}`** (off-by-default). Ra BUG i32 SIMD dtype.
- [x] dtype marshalling round-trip (f32/f64/i32/i16/i8/ui8/bool identity in==out) — SẠCH.
- [x] CPU vs WASM differential — đã cover xuyên suốt B-M.

### Bug đã fix đợt 28 (WASM SIMD: i32 dùng f32x4 ops khi vec-loop có bounds-guard)
> SIMD chỉ active khi scheduling on (off-by-default). Metamorphic SIMD on/off trên i32 + extent non-divisible → ra bug.
- **Repro**: `add(mul(x,y),x)` dtype **i32**, extent rem≠0 cần guard (vd n=9,13,17) → SIMD ra `f32x4.mul`/`f32x4.add` (SAI dtype lane) thay vì `i32x4` → output = giá trị rác (≈ input). scalar (simd off) đúng. f32 + extent chia hết → đúng (nên ko lộ trước đây).
- **Root** (`backend/wasm/codegen.js` `_inferBodyDtype`): suy dtype của vec-loop body để chọn SIMD op-set, nhưng KHÔNG đệ quy vào nhánh `IfThenElseNode` (`.thenBody`/`.elseBody`). Khi extent ko chia hết vector-width, scheduler bọc vec-store trong `if (idx<n){store}` (bounds guard) → `_inferBodyDtype` ko tìm thấy store/load → trả null → fallback `_defaultDtype='f32'` → `_emitVecMathOp` dùng `wasmVecOp('f32',...)` = f32x4 cho data i32. (f32-default "may mắn" đúng cho f32 + extent chia hết, che bug.)
- **Fix**: thêm `if (n.thenBody) stack.push(n.thenBody); if (n.elseBody) stack.push(n.elseBody);` vào `_inferBodyDtype` traversal → tìm được store/load i32 trong nhánh guard → dtype đúng → i32x4 ops.
- **Test**: `tests/backend/wasm/wasm-simd.test.js` block "numeric equivalence: SIMD on == scalar ... incl. non-power-of-2 remainders" (f32 ew/reduce/select + i32 add+mul/max-reduce × extent {3..17}, SIMD on vs simd:false). Trước đây 37 SIMD test chỉ STRUCTURAL (check WAT có f32x4), ko execute numeric. Revert (git stash) → i32 n=13 WRONG (simd0=-11 vs ref0=77). Full **3489 pass, 0 regression**.

### O. Runtime / pipeline — QUÉT SẠCH (đợt 30), KHÔNG bug
- [x] Opt-level differential: O0/O1/O1dom/O1epi/O2 × {cpu,wasm} × 15 program × 8 seed = 1680 case → eager == compiled. SẠCH.
- [x] Pipeline đầy đủ thêm: multi-output, kernel-reuse (compile 1 lần chạy nhiều input), run-vs-runAsync (pool path),
  recompile-isolation, conv2d/pool/layernorm modules, dynamic-shapes, scalar-0d output, where/clamp/cat/stack → SẠCH.
- [x] Quantization-enabled pipeline: cpu-quant == wasm-quant tới 6 chữ số (chênh eager full-precision là lossy mong đợi,
  KHÔNG phải bug).
- [x] Cache: `_cacheKey` = opName|shape:dtype|scalarArgs|target.name. Verify same→same entry; khác dtype/shape/dim/target
  → entry khác (pin lại fix i32 reduce-init cũ qua cache). Collision same-name-khác-config (vd WasmTarget simd:false vs
  true) CÓ tồn tại NHƯNG vô hại: (a) JIT path không bật scheduling nên config không đổi kernel; (b) eager dispatch
  `_TARGET_FOR_KEY` luôn dùng factory mặc định → không bao giờ truyền config khác cùng tên. KHÔNG fix (không reachable wrong-result).
- TEST: opt-level → tests/e2e/differential.test.js (describe "opt-level differential"); cache-key →
  tests/dispatcher/dispatcher.test.js (describe "jit cache key distinguishes ...").

### Oracle tổng (áp cho mọi tầng)
1. **Differential** — eager vs compiled, backend vs backend, opt-level vs opt-level.
2. **Metamorphic** — bật/tắt 1 pass phải bảo toàn ngữ nghĩa.
3. **Round-trip** — print/parse, transform/inverse, serialize/deserialize.
4. **Invariant/verifier** — sau MỖI pass: verifier PASS, SSA hợp lệ, no dangling, no live-overlap.
5. **Property-based** — đẳng thức đại số, bound sai số, monotonic cost.
6. **Crash-only** — chỉ cần không throw/không hang trên input hợp lệ (bắt assert/infinite-loop).

### Blocker production (không phải bug đếm được)
- [x] GPU / WebGPU backend đã verify được. Oracle ĐÚNG = **real Chrome WebGPU qua Puppeteer** (Dawn+DXC, ổn
  định), KHÔNG dùng `webgpu` npm (Dawn+FXC, segfault flaky → false fail). Fuzz differential CPU-vs-WebGPU
  trong-browser 200+ program SẠCH (fuzz = scratch throwaway, ĐÃ XÓA; chỉ commit unit/e2e xác định cho bug
  thật). `npm run test:webgpu`. Xem đợt 31 dưới.
- [ ] Reference (eager) không độc lập — eager từng có bug, differential có thể bỏ sót khi cả 2 cùng sai.
  → Cần oracle độc lập (numerical-diff cho grad, brute-force cho analysis, metamorphic cho pass).

### Bug đã fix đợt 31 (2026-06-12) — WebGPU fuzz (GPU-vs-CPU differential, máy có GPU)
> Fuzz random program (unary/binary/reduce/softmax/transpose/matmul/layernorm × f32 × 2D/3D) compile WebGPU
> vs CPU (oracle). Dawn (D3D12) segfault in-process sau ~2 pipeline → chạy 1 program/subprocess. Native
> segfault (0xC0000005) NHIỀU nhưng FLAKY (mỗi lần khác program) = Dawn-on-Windows instability, KHÔNG phải
> codegen bug (xác nhận: program crash trong batch chạy lại standalone OK).
- **BUG binding-layout pruning (sai value / validation fail)** — ĐÃ FIX. Khi forward bỏ qua 1 vài input (vd
  `(a,b,c)=>f(a)`), codegen vẫn khai báo `@binding(i)` cho MỌI buffer trong bufferMap, nhưng shader ko tham
  chiếu binding của input thừa → WebGPU `layout:'auto'` LƯỢC BỎ binding ko dùng khỏi bind group layout →
  runtime (cấp 1 buffer/binding đã khai báo) set binding index ko có trong layout → `CreateBindGroup` fail
  validation → output buffer giữ nguyên 0 (sai). Lộ qua program reduce-only/elementwise-only ăn input thừa.
  - Root: `webgpu_runtime.js createPipeline` dùng `layout:'auto'` + `pipeline.getBindGroupLayout(0)` (auto
    prune binding ko dùng trong shader).
  - Fix (ko comment/hardcode): dựng explicit `GPUBindGroupLayout` từ `kernel.metadata.bindings` (mỗi binding
    → type: `_shapes`→uniform / read_write→storage / read→read-only-storage), `createPipelineLayout` tường
    minh, `runWebGPUKernel` dùng layout đó thay `getBindGroupLayout`. Capture `GPUShaderStage` ở ensureDevice.
  - **Spec-level bug, KHÔNG phải Dawn**: xác nhận trên real Chrome — revert fix → `gpu=[0,0]` (sai), apply →
    `gpu=[4,5]` (đúng). (WebGPU spec: `layout:'auto'` prune binding ko reference trong shader.)
  - Test: `webgpu-chrome.test.js` ("forward ignoring extra inputs" → [4,5]) + `webgpu-gpu.test.js` (Dawn-node,
    opt-in). Revert → RED; apply → GREEN. Full default suite **3998/3998**.

- **"Deep-kernel segfault" HÓA RA KHÔNG phải bug** (chốt qua Chrome): mọi kernel segfault trên `webgpu` npm
  (q43=ln+sm+sigmoid+sqrt+abs, ln+sm×3, 8-layer ln-chain, mm-chain...) chạy **ĐÚNG trên real Chrome** (maxErr
  ~1e-8). Segfault + "parser recursive depth" CHỈ là Dawn-node (FXC) lởm dưới tải / kernel single-invocation
  lớn — KHÔNG phải lỗi codegen. → Bài học: test WebGPU phải dùng Chrome (Puppeteer), đừng tin `webgpu` npm.
- **Tối ưu kèm theo (ko bắt buộc nhưng giữ): buffer-slot reuse cho private array** (`backend/webgpu/codegen.js`).
  Trước: mỗi intermediate là 1 `var<private> array` riêng (deep graph → 30-40 array → ép giới hạn private/
  register + làm Dawn-node crash). Giờ: liveness-based slot allocation — buffer ko chồng live-range share 1 slot
  (`_assignLocalSlots` + `_computeBufferLiveness` loop-backedge-safe + `MinHeap` O(n log n), ko O(n²)). ln+sm×3:
  22 array → 7 slot; lnsm3 từ crash→pass cả trên Dawn-node. Verify ĐÚNG trên Chrome (200 fuzz + targeted).
  Test: `webgpu/codegen.test.js` (+"reuses one slot for non-overlapping locals"=2 slot, +"keeps distinct slots
  for simultaneously-live"=3 slot). Revert reuse → tên buffer gốc, slot test RED.
- HẠ TẦNG test mới: `tests/backend/webgpu/webgpu-chrome.test.js` (esbuild bundle browser → http server → Puppeteer
  headless Chrome → CPU-vs-WebGPU in-page; self-skip nếu thiếu Chrome/puppeteer-core). CHỈ test XÁC ĐỊNH:
  unused-binding (pin bug binding) + 2 e2e deep kernel (ln/sm chain + mm chain) verify slot-reuse đúng trên GPU
  thật. KHÔNG có vòng fuzz random (fuzz chỉ để săn bug, throwaway). `npm run test:webgpu`. Excluded khỏi
  `npm test`. devDep: `puppeteer-core` + Chrome hệ thống.

### Bug đã fix đợt 32 (2026-06-12) — WebGPU dtype marshalling (probe Chrome ra)
> Probe rộng op/dtype trên Chrome: f32 ổn, NHƯNG mọi dtype ≠ f32 ra rác. Root = runtime marshal hardcode
> `Float32Array`.
- **BUG: WebGPU runtime marshal MỌI buffer bằng Float32Array** (`webgpu_runtime.js`) → i32/i16/i8/ui8/bool/
  f16/bf16/i64/f64 sai bit + buffer size sai (narrow-int cấp numel×1 byte thay vì ×4). Kéo theo MỌI op dùng
  index tensor i32 (index_select/gather/scatter/embedding/one_hot, argmax/topk/argsort trả index) sai vì index
  bị đọc như f32. Xác nhận node+Chrome: `sum(i32)` → `[-1,-28]` (đúng `[6,15]`); `index_select` lấy nhầm hàng.
  - Fix: marshal theo WGSL element type. Codegen gắn `dtype` vào mỗi binding (+ packed entry); guard packing chỉ
    khi MỌI buffer cùng wgslType. Runtime `wgslViewCtor` (i32→Int32, u32→Uint32, f16→Uint16, else Float32) +
    `packTensorInto`/`unpackTensorFrom` (bf16 decode/encode qua half.js, i64 BigInt↔i32 truncate, còn lại
    .set numeric-convert/wrap), size theo `wgslBytes` + `align4` (f16 2-byte). Bật device feature `shader-f16`.
  - Test: `webgpu-chrome.test.js` (i32 reduce/elementwise, i16/i8/ui8 wrap, bool where, index_select/gather i32,
    f16/bf16 bit-exact, i64, f64). Revert (Float32Array-only) → 5 RED; apply → 10/10. Full suite **3998/3998**.
  - GIỚI HẠN: i64 dùng WGSL i32 (no native 64-bit) → đúng <2^31, lớn hơn truncate; f16 cần adapter có
    `shader-f16` (đa số desktop có). bf16/f64 compute ở f32 (như CPU path).

### Đợt 33 (2026-06-12) — test compiler OPTIONS trên WebGPU (Chrome) — FIX scheduling GPU-reduction race
> Bật từng option compiler, differential WebGPU-vs-CPU-default (oracle) trên Chrome, 8 prog (elem/reduce/softmax/
> matmul/layernorm/conv2d...). Ma trận 10 config × 8 prog.
- SẠCH (WebGPU đúng): default, `fusion.strategy=dominator`, `fusion.epilogue`, `optimization.rematerialization`,
  `optimization.layout`, `optimization.fastMath`, `memory.inplaceReuse=false`. → 7 config OK.
- **BUG: `scheduling.enabled` (+`autotune`) sinh kernel SAI trên GPU** (off-by-default nên compile() thường ko dính):
  - `scheduling:{enabled:true}` → conv2d sai. `+autotune` → reduce ra **0**, softmax/layernorm/conv2d sai.
  - **Root**: scheduler FUSE trục REDUCTION (serial) vào chiều THREAD song song. Vd `sum(relu(x),1)` [16,16]:
    `workgroup_size(256)`, `sav0=lid.x/16` (output row), `rv0=lid.x%16` (reduce col), rồi
    `buf_3[sav0] = buf_3[sav0] + buf_10[...]` → **16 thread đua nhau read-modify-write cùng buf_3[sav0]** = data
    race (cần atomic hoặc tree/shared-mem reduction). ĐÚNG trên CPU/WASM (1 core, vòng lặp tuần tự), SAI trên GPU
    (256 thread). Autotune chọn config fuse reduce-into-thread → kích race ở nhiều op.
  - **ĐÃ FIX (GPU-safe reduction)**: trục reduce giữ là vòng lặp TUẦN TỰ trong mỗi thread; chỉ trục spatial (output)
    bind vào thread. Sau fix kernel reduce: `if (lid.x < 16) { var acc = buf_3[sa0]; for r0 in 0..16 { acc +=
    buf_10[sa0*16+r0]; } buf_3[sa0] = acc; }` — KO race.
    - Fix 1 (`schedule/rules.js`): thay heuristic vị trí (`estimateSpatialDims=loopCount-1`, sai cho multi-axis
      reduce như conv) bằng **classifier cấu trúc** `computeReductionLoopVars`: loop là reduction iff iterVar của nó
      KO xuất hiện trong index của buffer WRITE (chỉ index read → bị contract). Xử lý đúng conv (3 trục reduce) +
      keepdim. `isReductionLoop` dùng set này. Export `classifyBlock`/`isReductionLoop`.
    - Fix 2 (`autotune/search_space.js`): thêm `createReductionGPUSketch` (chỉ parallel spatial, reduce tuần tự),
      và `getSketchesForBlock` chọn nó cho GPU+reduction (trước rơi về elementwise-sketch fuse-tất-cả). Dùng
      `classifyBlock` (robust) thay `classifyBlockForSketch` (cũ chỉ xem initBody → reduce_acc ko có initBody → miss).
    - Test: `webgpu-chrome.test.js` (scheduled reduce/conv2d/softmax/layernorm vs CPU, `scheduling` + `autotune`),
      `autotuner.test.js` (sketch selection: CPU→reduction_cpu, GPU→reduction_gpu). Revert classifier → 3 RED;
      apply → 13/13 Chrome + autotuner 11/11. Ma trận option full lại trên Chrome: **40/40 OK** (default/scheduling/
      autotune/ALL × reduce/conv/softmax/layernorm/matmul...). Full suite **3999/3999**.
  - GHI CHÚ: `compile()` mặc định vẫn scheduling=OFF (single-invocation), giờ scheduling=ON cũng đúng trên WebGPU.

### Đợt 34 (2026-06-12) — fuzz kỹ MỌI option trên Chrome (chain × 9 config × 50 seed)
> Differential: gpu(config) vs cpu(SAME config) = bug WebGPU; cpu(config) vs cpu(default) = bug pass (ko phải WebGPU).
- **FIX BUG WebGPU: matmul có elementwise-prefix dưới scheduling/autotune sai** (vd `matmul(tanh(x),y)`): intermediate
  tanh là `var<private>` per-thread; thread k chỉ ghi `_lt0[k]` nhưng matmul đọc CẢ HÀNG `_lt0[i*3+c]` (cross-thread)
  → đọc rác thread khác. Phải để workgroup memory + barrier.
  - Root: `_analyzeSharing` (`backend/webgpu/codegen.js`) CHỈ promote+barrier khi thread-extent KHÁC nhau; matmul-tanh
    mọi stage cùng extent → ko trigger. Fix: thêm `_findCrossThreadBuffers` — phát hiện local buffer ĐỌC trong vòng
    lặp với loop-var trong index (range-read = cross-thread) trong khi GHI per-thread; promote chúng lên `var<workgroup>`
    + barrier. Gate: chỉ khi có thread bindings (đường single-invocation ko cần). Precise: single Linear (đọc cùng index)
    KO bị promote → ko thừa barrier.
  - Test: `webgpu-exec.test.js` (sẵn: single-layer no-barrier vẫn pass). Fuzz lại Chrome: sched/autotune/ALL **CLEAN**.
    Full suite **3999/3999**.
- **KHÔNG phải bug WebGPU (đã xác minh, ghi để khỏi đào lại):**
  - `fusion.strategy='dominator'`: SAI cả trên **CPU** (NaN/Inf cho chuỗi softmax/reduce). Bug pass dominator chung,
    ko phải WebGPU (WebGPU chỉ render trung thực graph đã hỏng). → cần fix riêng DominatorFusionPass nếu muốn dùng.
  - conv+**max_pool2d** eager ra NaN trong **browser bundle** (esbuild) nhưng ĐÚNG trên node → artifact bundling của
    harness, ko phải WebGPU codegen (conv+maxpool default chạy đúng trên node WebGPU).
### Đợt 37 (2026-06-12) — FIX conv+maxpool autotune (cross-workgroup shared intermediate)
> Trước ghi "còn tồn"; đào tiếp ra root + fix.
- **Root**: kernel fused multi-stage có intermediate cross-thread (đợt 34) được promote lên `var<workgroup>` + barrier.
  NHƯNG khi data > maxThreadsPerBlock (vd conv output 300 > 256), scheduler split stage ra NHIỀU workgroup
  (`blockIdx.x`/`_wid.x`, dispatch>1). `var<workgroup>` + `workgroupBarrier` chỉ phạm vi 1 workgroup → workgroup 1
  KO thấy data workgroup 0 ghi → maxpool đọc rác/0. (Cũng dính Linear(4,64)→ReLU→Linear(64,2): exec test cũ chỉ check
  CẤU TRÚC nên lọt, thực ra GPU ra SAI — xác minh revert maxErr 0.19.)
- **Fix** (`backend/webgpu/codegen.js`): khi có cross-thread-shared intermediate VÀ dispatch>1 (đa-workgroup, ko biểu
  diễn được bằng workgroup mem) → `_serializeThreads`: emit MỌI thread-binding loop thành vòng lặp TUẦN TỰ
  (workgroup_size 1, single invocation, `_gid`, ko barrier). Đúng tuyệt đối (1 thread tuần tự); mất parallel cho case
  này nhưng ĐÚNG. Case fit-1-workgroup (matmul-tanh nhỏ) vẫn parallel như cũ.
  - Phụ: `_findCrossThreadBuffers` follow LET/iterVar ALIAS của loop-var (maxpool index qua `let pv4=pkh` chứ ko dùng
    pkh trực tiếp → trước miss). Thêm LIRBindings/BlockNode-iterVar/LetStmt alias-tracking.
- **Test**: `webgpu-gpu.test.js` (node subprocess, maxpool chạy đúng ở node) "conv2d+relu+maxpool autotune == CPU";
  `webgpu-chrome.test.js` "wide bottleneck matmul serialized == CPU" (Chrome); `webgpu-exec.test.js` "wide bottleneck
  serializes to one invocation" (structural). Revert serialize → exec RED + maxErr 0.19. Verify node WebGPU 5 autotune
  seed + evolutionary + non-zero output: maxErr ~1e-7. Full suite **4013/4013** + chrome 15/15.
- GHI CHÚ còn lại: browser-bundle hỏng maxpool eager (artifact esbuild, node đúng) — chưa truy, ko ảnh hưởng codegen;
  conv dynamic-batch N (P2.3 gap). dominator fusion (bug pass chung, ko phải WebGPU).

## Ghi chú vận hành
- Coding rules: ko comment, ko hardcode, ko O(n²); fix xong viết test, đặt vào file test có sẵn, ko có mới tạo mới.
- KHÔNG `git checkout` file đang có uncommitted work (đã mất việc 2 lần) — backup `cp` trước.
- Revert-test mỗi fix: bỏ fix phải thấy test RED, apply lại GREEN.
