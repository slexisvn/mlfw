# feature.md — Lộ trình production cho mlfw compiler

> Tài liệu sống. Mỗi mục có: **Why** (vì sao cần), **What** (làm gì), **Done when** (tiêu chí xong), **Refs** (file liên quan).
> Ưu tiên theo thứ tự P0 → P3. P0 = chặn production. Đánh dấu `[x]` khi xong, ghi đợt/PR.
> Bối cảnh: rút ra từ đợt bug-hunt A–O (15 tầng) + review kiến trúc. Compiler đã đúng concept TVM/XLA;
> việc còn lại chủ yếu là **kỷ luật bất biến (invariant discipline)**, **soundness**, **scale**, và **GPU**.

---

## P0 — Chặn production (correctness & soundness)

### P0.1 — Hợp thức hoá hợp đồng Schedule ↔ Codegen  ✅ (đợt 31)
- **Why:** Codegen WASM giả định **đúng một** trục parallel (`_parallelExtent` lấy từ PARALLEL node đầu tiên),
  nhưng `SchedulePolicy` có thể đánh dấu nhiều loop PARALLEL khác extent → ghi tràn buffer (bug đợt 29:
  fused matmul→reduce). Đây là nợ kiến trúc, không phải 1 bug lẻ.
- **What:**
  - [x] Thêm invariant vào `ScheduleValidator._checkPartitionConsistency`: thu thập mọi PARALLEL loop có
    constant extent; ≥2 extent KHÁC nhau → error "Ambiguous parallel partition". Một lần duyệt O(n).
  - [x] Codegen đã chọn tường minh loop khớp `_parallelExtent`, loop khác extent → serial (đợt 29). Validator
    nay enforce contract tường minh (autotuner reject candidate vi phạm; default path đã an toàn nhờ codegen).
  - [x] Test: `tests/compiler/schedule/primitives-schedule.test.js` (mismatched→reject, same-extent/single→pass).
- **Done when:** ✅ Validator từ chối schedule vi phạm; không path nào suy luận ngầm số trục parallel.
- **Refs:** `src/compiler/schedule/validator.js`, `src/backend/wasm/codegen.js:_visitFor/_scanParallel`.

### P0.2 — Tách cờ soundness (IEEE) khỏi optimization mặc định  ✅ (đợt 31)
- **Why:** Simplifier từng bật mặc định các rewrite **không IEEE-sound** (`x−x→0`, `x/x→1`, `exp∘log→id`)
  — sai trên NaN/Inf/±0. Đã gỡ thủ công; cần cơ chế ngăn tái phát. TVM/XLA chỉ làm dưới fast-math.
- **What:**
  - [x] Thêm `CompilerConfig.optimization.fastMath` (default `false`). `AlgebraicSimplificationPass({fastMath})`
    chọn pattern set sound vs fast-math.
  - [x] Pattern soundness-sensitive nhận cờ `fastMath`: `SubSelf`/`MulZero` (int luôn, float chỉ khi fastMath);
    `DivSelf`/`ExpLog`/`LogExp` chỉ khi fastMath.
  - [x] Test: `tests/compiler/algebraic/inverse-simplify.test.js` (default giữ nguyên; fastMath fold).
    Block IEEE-sound differential (đợt cũ) vẫn là gate eager==compiled.
- **Done when:** ✅ `fastMath:false` không đổi giá trị IEEE; `fastMath:true` mới cho phép rewrite không-sound.
- **Refs:** `src/compiler/passes/simplify/algebraic.js`, `src/compiler/ir/graph/patterns.js`,
  `src/compiler/pipeline/compiler.js` (CompilerConfig.optimization.fastMath).

### P0.3 — Verifier là nguồn chân lý sau MỖI pass (debug mode)  ✅ (đợt 31)
- **Why:** Có verifier 3 tầng nhưng lưới thủng (SSA dangling-operand check từng bị tắt → bug partition/fusion lọt
  tới codegen). "pipeline never runs verifyLIR" (ghi rõ trong test cũ). Bug giá rẻ khi bắt sớm.
- **What:**
  - [x] `PassManager.setVerifyHook` chạy verifier sau MỖI graph pass (CHANGED) → quy lỗi đúng pass gây ra.
  - [x] Chế độ `verify:'full'`: wire hook graph (`verifyModule`/`verifyFunction`) + `TensorVerifier` sau
    scheduling + `verifyLIR` sau LIR lowering (đóng lỗ "never runs verifyLIR").
  - [x] Test: `tests/compiler/passes/pass-manager.test.js` (hook attribute lỗi cho pass, strict/resilient) +
    e2e `tests/e2e/differential.test.js` (verify:'full' chấp nhận IR hợp lệ, eager==compiled).
- **Done when:** ✅ Pass phá invariant là đỏ ngay, quy đúng pass. LIR được verify trong pipeline.
- **Refs:** `src/compiler/passes/pass_manager.js`, `src/compiler/pipeline/compiler.js`.

### P0.4 — Differential CI gate (eager vs compiled, mọi opt-level × target)
- **Why:** Oracle mạnh nhất đã tìm ra phần lớn bug. Cần đóng băng thành gate chống regression.
- **What:**
  - [x] Test coverage đã có: opt-level differential (O0/O1/O2 × cpu/wasm), verify:'full' e2e, backward/AD
    differential, cache-key. (đợt 30–31)
  - [ ] CI config: bật các test trên thành **required check** chặn merge (infra, ngoài code).
  - [ ] Nightly fuzz seed ngẫu nhiên (crash-only + numeric).
  - [ ] Oracle độc lập cho grad (numerical finite-diff) thay vì chỉ eager-vs-compiled (eager từng có bug).
- **Done when:** Không merge được nếu differential đỏ; nightly fuzz báo cáo coverage.
- **Refs:** `tests/e2e/differential*.test.js`, `tests/dispatcher/dispatcher.test.js` (cache-key).

---

## P1 — Cần cho production thực tế

### P1.1 — GPU / WebGPU backend verify được (BLOCKER lớn nhất)
- **Why:** WebGPU backend chưa verify được (không có máy GPU); GPU segfault loại khỏi pass/fail. Không thể
  production-claim GPU khi chưa có oracle.
- **What:**
  - [ ] CI có runner GPU (hoặc software rasterizer/SwiftShader/Dawn headless) chạy WebGPU.
  - [ ] Differential cpu-vs-webgpu cho elementwise/reduce/matmul/conv/softmax.
  - [ ] Tách rõ "GPU experimental" trong docs đến khi xanh.
- **Done when:** WebGPU vào differential gate như cpu/wasm.
- **Refs:** `src/backend/wasm/...`, `src/compiler/runtime/webgpu_runtime.js`, `tests/stress`, `tests/webgpu`.

### P1.2 — Autotune đo thật (measurement-based), không chỉ cost-model
- **Why:** Hiện cost-model analytical mặc định, đo thật chỉ refine top-K. AutoTVM/Ansor đo mọi candidate vì
  cost-model analytical lệch thực tế. Production cần số đo thật + cache bền.
- **What:**
  - [ ] Bật benchmark-loop ổn định (warmup/repeat/median) cho top-K rộng hơn; chống nhiễu (lock affinity nếu được).
  - [ ] `LearnedCostModel`: nâng từ linear regression → mô hình mạnh hơn (gradient-boosted / small MLP) HOẶC ghi rõ
    giới hạn.
  - [ ] `TuningDatabase` persist ra đĩa + versioning theo target/compiler-version; invalidate khi đổi codegen.
  - [ ] Time-budget thực thi (`timeBudgetMs` hiện không enforce).
- **Done when:** Tuned kernel ≥ baseline trên benchmark thật, tái lập qua phiên, tôn trọng time budget.
- **Refs:** `src/compiler/autotune/{autotuner,cost_model,tuning_db,benchmark}.js`.

### P1.3 — Scale: bỏ O(n²) trên graph lớn  ✅ phần fusion (đợt 32)
- **Why:** Đo thực tế: `FusionPass` blow-up **O(n³)** — graph 400 op fuse mất **29 GIÂY**, 3200 op mất **138 GIÂY**.
  Hotspot = `buildHorizontalGroups`: `_anyGroupMemberDependent` chạy BFS độc lập **mỗi member × mỗi cặp candidate**
  trong shape-bucket → O(B²·group·traversal). Phụ: `_condensedHasCycle` gọi lại 1 lần / horizontal-group → O(H·n).
- **What:**
  - [x] `buildHorizontalGroups` viết lại bằng **forward-taint tuyến tính**: duyệt topo-window 1 lần / seed, lan
    "depends-on-group" qua operand (operand đã xử lý trước → O(1)/op). Tận dụng bucket đã ở topo order nên candidate
    chỉ có thể *phụ thuộc* member, không thể bị phụ thuộc → bỏ vòng per-member. Window = `maxFusionSize` (config,
    KHÔNG hardcode). Kết quả: **O(n³) → O(n)**, 3200 op: 138s → **0.9s** (153×), per-op phẳng ~0.3ms.
  - [x] `buildAllGroups`: thay vì check cycle / horizontal-group (O(H·n)), **áp tất cả candidate rồi check cycle
    MỘT lần** (common-case O(n)); chỉ fallback incremental khi thật sự có cycle (hiếm). Monotonic-acyclic đảm bảo
    đúng.
  - [x] Test: `tests/compiler/fusion/groups-fusion.test.js` (cross-shape transitive-dep + scaling guard 800 op < 5s).
  - [x] **Post-dominance O(n²) → O(n log n)** (đợt 33): đo thấy pattern residual/skip-connection ("broom") là O(n²)
    thật — `intersect` walk idom-chain O(height)/node, n=3200 = 178ms, n→4× thì time 17.5× (≈n²). Vì value-graph là
    **DAG single-sink**, viết lại `compute()`: 1 pass reverse-topo (successors-trước-node → idom final ngay, bỏ
    `while(changed)`) + **binary-lifting LCA** thay chain-walk. `exits.includes` → Set luôn. Kết quả: n=12800
    178ms→44ms, per-op phẳng (linear). Revert-test: guard 12k op cũ = 3138ms (>2s) đỏ, mới ~320ms xanh.
- **Done when:** ✅ Fusion + post-dominance tuyến tính theo số op, có scaling-guard test cho cả hai.
- **Refs:** `src/compiler/passes/fusion/fusion_groups.js`, `src/compiler/analysis/dominance.js`.

### P1.4 — Quantization end-to-end production-grade  ✅ (đợt 34–35)
- **Why:** Pipeline quant chạy đúng & cross-target nhất quán (đợt 30). Khảo sát kỹ: phần lớn đã có sẵn.
- **What:**
  - [x] Calibration minmax/**entropy (KL)**/percentile + histogram: ĐÃ CÓ SẴN (`calibration.js`). Xác nhận.
  - [x] Int8 matmul/conv kernel path **THẬT** (integer accumulate, không phải dequant→f32): ĐÃ CÓ SẴN —
    `quantized_dot`/`quantized_conv` lower thành int8×int8→i32 accumulate (trừ zero-point) rồi dequant. Xác nhận.
  - [x] **Per-channel weight scale** (đợt 34): `QuantizationParams.fromConstantArrayPerChannel` +
    `quantizeArrayPerChannel`/`dequantizeArrayPerChannel` (O(n), map flat-index→channel theo stride/axis). Trước đó
    chỉ có stub (`quantizeArray` throw). Numerical harness: per-channel error **>1.5×** thấp hơn per-tensor.
  - [x] **BUG FIX: array-constant lowering** (đợt 35): `lowerConstant` từng hạ ARRAY constant thành **0** (chỉ
    handle scalar). Fix: array → chuỗi indexed store với literal từng phần tử (codegen có sẵn handle, không đổi
    codegen). Xác minh f32 + int8 array constant đúng trên cpu+wasm. Test:
    `tests/compiler/lowering/elementwise-lowering.test.js`.
  - [x] **Per-channel int8 matmul END-TO-END trong COMPILED pipeline** (đợt 35): nhờ fix array-constant, dựng được
    quantize(act) + int8-weight-constant + `quantized_dot` + per-channel dequant (convert + mul broadcast scaleVec).
    Compiled accuracy harness: per-channel error **>1.5×** thấp hơn per-tensor + **cpu == wasm bit-khớp**. Test:
    `tests/compiler/quantization/pass-quantization.test.js`.
  - [ ] (mở rộng) Quant PASS tự động phát hiện per-channel weight (weight là captured-param khi trace, cần
    per-channel quantize codegen hoặc fold param→const) + accuracy top-1 trên model thật. Capability codegen đã đủ.
- **Done when:** ✅ Per-channel params + int8 kernel thật + array-const bug fix + per-channel compiled harness
  (cpu==wasm). Auto-pass-detection cho trace-param-weight là mở rộng (capability đã chứng minh).
- **Refs:** `src/compiler/ir/graph/quantization_types.js`, `src/compiler/passes/lowering/lowering_registry.js`
  (`lowerConstant`), `tests/compiler/quantization/`, `tests/compiler/lowering/elementwise-lowering.test.js`.

### P1.5 — Pass pipeline hội tụ (fixed-point) thay vì tuần tự cứng
- **Why:** PassManager tuần tự, phase-ordering hand-tuned → giòn (đổi thứ tự canonicalize/simplify/fold dễ vỡ).
  XLA dùng `HloPassPipeline` lặp tới fixed-point.
- **What:**
  - [ ] Cho phép nhóm pass lặp tới khi `PassResult.UNCHANGED` (có max-iter guard).
  - [ ] Pass khai báo `preservedAnalyses` đầy đủ để tránh recompute thừa.
- **Done when:** Thứ tự pass trong nhóm canonicalize không đổi kết quả cuối; có max-iter chống loop.
- **Refs:** `src/compiler/passes/pass_manager.js`, `src/compiler/pipeline/compiler.js:_runGraphPasses`.

---

## P2 — Độ chín & vận hành

### P2.1 — Observability / debug compiler
- **What:**
  - [ ] IR dump ổn định mỗi phase (đã có trace) + diff giữa các phase.
  - [ ] "Why-not-fused" / "why-this-schedule" explain log cho từng block.
  - [ ] Crash repro tự động: dump graph + config + inputs khi compile/run throw.
- **Refs:** `src/compiler/pipeline/trace.js`, `src/compiler/ir/*/printer.js`.

### P2.2 — API stability & error UX
- **What:**
  - [ ] Chốt public API (`compile`, `CompilerConfig`, target factories); semver.
  - [ ] Thông báo lỗi có ngữ cảnh (op, shape, dtype, phase) thay vì generic throw.
  - [ ] `resilient` mode trả về danh sách func fail + lý do, không nuốt lỗi.
- **Refs:** `src/tracing/compile.js`, `src/compiler/pipeline/compiler.js` (CompilationError).

### P2.3 — Dynamic shapes diện rộng
- **Why:** sym_int chạy end-to-end (verify đợt P2 edge), nhưng cần phủ rộng hơn cho production serving.
- **What:**
  - [ ] Bucketing/guard cho dynamic dim; specialization cache theo shape-class.
  - [ ] Differential dynamic-shape cho conv/matmul/attention với nhiều batch/seq-len.
- **Refs:** `src/tracing/compile.js` (dynamicShapes), `src/compiler/analysis/sym_int.js`.

### P2.4 — Memory planning nâng cấp
- **What:**
  - [ ] Đo fragmentation thực; cân nhắc best-fit / interference-graph khi linear-scan kém.
  - [ ] Remat: kết hợp với scheduling (hiện chạy tách); chính sách theo memory budget thật của target.
- **Refs:** `src/compiler/passes/memory/{buffer_assignment,rematerialization}.js`.

---

## P3 — Mở rộng / dài hạn

### P3.1 — Schedule trace-authoritative (tái thiết kế)
- **Why:** Schedule đang mutate IR in-place + rebuild `SRefTree` như index → cả lớp bug kiểu K. TensorIR coi
  schedule-state là chân lý, validate từng primitive.
- **What:**
  - [ ] Chuyển sang: schedule = chuỗi decision áp lên state đã-validate; IR là sản phẩm sinh ra.
  - [ ] Mỗi primitive kiểm tra legality với `ScheduleState`/`DependencyAnalysis` trước khi áp.
- **Done when:** Không primitive nào tạo được IR vi phạm invariant codegen.
- **Refs:** `src/compiler/schedule/{schedule,sref,schedule_state,trace}.js`.

### P3.2 — Ansor-style sketch auto-generation
- **Why:** Sketch hiện hand-written (AutoTVM template). Ansor auto-generate phủ pattern rộng hơn.
- **Refs:** `src/compiler/autotune/search_space.js`.

### P3.3 — Backend mở rộng
- [ ] LLVM/native (qua WASM→native hoặc binding) cho server.
- [ ] Library-op offload (BLAS/cuBLAS/oneDNN) thay vì codegen thuần cho matmul/conv lớn.

### P3.4 — Training path production
- [ ] AD đã có (VJP/joint/backward builder); cần optimizer fusion, grad checkpoint policy, mixed-precision loop.
- **Refs:** `src/compiler/ad/`.

---

## Định nghĩa "production-ready" (exit criteria tổng)
1. P0 xanh toàn bộ: soundness IEEE, verifier-as-truth, schedule↔codegen invariant, differential gate.
2. ≥1 backend (cpu **và** wasm) qua differential + nightly fuzz, 0 known-wrong-result.
3. GPU/WebGPU hoặc xanh-trong-CI hoặc gắn nhãn experimental rõ ràng.
4. Autotune đo thật + tuning DB bền; tuned ≥ baseline.
5. Compile-time có guard scale; API + error UX chốt; docs phủ public surface.

---

## Nhật ký
- 2026-06-12: Khởi tạo từ review kiến trúc + đợt hunt A–O. Bug tầng K (4) đã fix; A/C/D/E/F/G/L/M/O sạch;
  B/H/I/J/N mỗi tầng 1 fix. Xem `todo.md` để biết chi tiết từng đợt.
- 2026-06-12 (đợt 31): Hoàn thành **P0.1/P0.2/P0.3** (P0.4 còn phần CI infra). +25 test, full regression
  3832 pass / 0 fail. Mỗi mục revert-test xác nhận load-bearing (revert → đúng assertion mới đỏ).
- 2026-06-12 (đợt 32): **P1.3 fusion** — fix O(n³) `buildHorizontalGroups` → O(n) (forward-taint) + `buildAllGroups`
  cycle-check O(H·n) → O(n) common-case. Đo: 3200 op 138s → 0.9s. +2 test (cross-shape transitive + scaling guard).
  Full regression 3834 pass / 0 fail.
- 2026-06-12 (đợt 33): **P1.3 post-dominance** — fix O(n²) (residual/skip "broom" pattern) → O(n log n) bằng
  single-pass reverse-topo + binary-lifting LCA (value-graph là DAG). +2 test (residual correctness + 12k scaling
  guard). Revert-test: cũ 3138ms đỏ, mới ~320ms. Full regression **3857 pass / 0 fail**. P1.3 xong.
- 2026-06-12 (đợt 34): **P1.4 quantization** — khảo sát: calibration modes + int8 kernel thật ĐÃ CÓ. Thêm
  per-channel weight scale ở tầng số học (`fromConstantArrayPerChannel` + quantize/dequantizeArrayPerChannel) +
  accuracy harness (per-channel >1.5× chính xác hơn per-tensor). +4 test. Phát hiện `lowerConstant` hạ array-const
  thành 0 → chặn wiring per-channel vào codegen (ghi rõ). Full regression **3861 pass / 0 fail**.
- 2026-06-12 (đợt 35): **P1.4 xong** — fix BUG array-constant lowering (array→0, fix bằng indexed-store literals,
  không đổi codegen, đúng cpu+wasm). Nhờ đó dựng per-channel int8 matmul END-TO-END trong compiled pipeline:
  per-channel >1.5× chính xác hơn per-tensor + cpu==wasm. +7 test (4 array-const + 3 per-channel compiled).
  Revert-test cả hai load-bearing. Full regression **3868 pass / 0 fail**.
