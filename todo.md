# TODO — bug hunt compiler

## Trạng thái hiện tại (2026-06-10)
- Fuzzer `tests/e2e/fuzz-differential.test.js` (seeded, 200 prog × cpu+wasm) đang xanh.
- Đợt vừa rồi fix: WASM i32 `abs` (emit f32.abs), i32 max/min reduce-init (±Infinity → INT_MIN/MAX theo dtype).
  - Code: `src/backend/dtype_map.js` (`reduceInitValue` + `INT_RANGE`), `src/tracing/tracer.js`, `src/dispatcher/jit_cache.js`.
  - Test: `tests/e2e/differential-nn.test.js` (+max_i32_neg, +min_i32_pos), fuzz-differential.test.js.
- **Chưa commit git.** Full e2e + AD: 413 passed.

## Fuzzer mới quét được (phạm vi hẹp)
- Op: unary/binary(+broadcast row/col)/matmul-2D/reduce(sum,mean,max,min).
- Shape: chỉ 2D, dim 2–6, chain 2–6 op. Dtype: f32 + i32. Backend: CPU + WASM.

## Việc tiếp theo — ưu tiên cao → thấp

### P0 — mở rộng generator (mật độ bug lịch sử cao nhất ở đây)
- [ ] Rank ≠ 2: 0D scalar, 1D, 3D/4D+, batched matmul, broadcast nhiều chiều (cả unsqueeze ngầm).
- [ ] View-ops trong chain: reshape / transpose / permute / expand / slice (có step) / squeeze / unsqueeze / narrow / select / flatten. Trộn view + compute để bắt lỗi stride/contiguous.
- [ ] Backward/autodiff: fuzz random graph rồi so gradient eager vs compiled (giờ mới chỉ fuzz forward). Gồm cả double-backward (grad of grad).

### P1 — dtype & op còn trống
- [ ] f64 (mới test lẻ), f16/bf16 (lỗi rounding/denormal hay ẩn ở đây).
- [ ] Quantized i8/u8 — cả quant path (observer→quantize→dequant) chưa fuzz lần nào.
- [ ] bool/mask, index dtype (gather/scatter index).
- [ ] conv/pool (đủ stride/pad/dilation/groups), layer_norm/batch_norm/group_norm/softmax/log_softmax, embedding, scatter/gather/index_select, comparison/select/where, cumsum/argmax/argmin, concat/split/stack/pad.

### P2 — edge values & shape biên
- [ ] NaN / Inf / -0 input, empty tensor (dim=0), dim=1 (broadcast biên), overflow i32, div-by-0, reduce trên axis rỗng.
- [ ] Shape động (DYNAMIC / sym_int): kích thước symbolic, so giữa nhiều giá trị concrete khác nhau.
- [ ] Số rất lớn / rất nhỏ để ép sai lệch tích luỹ (reduce trên n lớn, matmul K lớn).

---

## CATALOG FUZZ TOÀN COMPILER — theo tầng + oracle bắt bug

> Khác biệt cốt lõi: differential (eager vs compiled) chỉ là 1 oracle. Compiler còn cần
> **metamorphic** (pass phải bảo toàn ngữ nghĩa), **round-trip**, **invariant/verifier**,
> **property-based**. Mỗi mục dưới ghi rõ oracle dùng.

### A. Frontend — tracing / dispatcher / IR builder
- [ ] Round-trip IR: build graph → print → parse → print, hai bản print phải bằng nhau (`ir/graph/printer.js`).
- [ ] Tracing vs eager: mọi op trace ra IR rồi eval phải == eager (đang có, mở rộng op set).
- [ ] `_BUILDER_METHOD_MAP` / `_SCALAR_ARG_SPEC`: fuzz đủ scalar-arg (dim âm, keepdim, multi-dim) cho mọi op map.
- [ ] Invariant: verifier (`ir/graph/verifier.js`) phải PASS sau khi build mọi graph hợp lệ — không dangling operand, SSA đúng, dtype/shape khớp.

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
