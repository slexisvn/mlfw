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

### P1.2 — Autotune đo thật (measurement-based), không chỉ cost-model  ✅ (đợt 43)
- **Why:** Hiện cost-model analytical mặc định, đo thật chỉ refine top-K. AutoTVM/Ansor đo mọi candidate vì
  cost-model analytical lệch thực tế. Production cần số đo thật + cache bền.
- **What:**
  - [x] **Benchmark-loop ổn định + chống nhiễu** (đợt 43): `robustStats` (median/min/trimmed-mean 10% + coefficient
    of variation); `BenchmarkRunner` re-measure 1 vòng khi `cv > benchmarkMaxCv` (bounded, không O(n²)); top-K rộng
    qua `topKForBenchmark`. Giới hạn: V8/JS không set được CPU affinity → dùng trimmed-stat + re-measure thay cho pin.
  - [x] **`LearnedCostModel` → MLP** (đợt 43): thay linear-regression bằng MLP 1 hidden-layer (tanh) + chuẩn-hoá
    feature/target + seeded init (deterministic). Fit được tương tác phi tuyến (XOR: MSE→0; linear floor 0.25).
  - [x] **`TuningDatabase` persist đĩa + versioning** (đợt 43): `CODEGEN_VERSION`; `serialize` nhúng version,
    `deserialize` **invalidate** khi `codegenVersion` lệch; `saveToFile/loadFromFile` (DI `fs` — browser-safe, không
    import `node:fs`). Inject DB qua `scheduling.tuningDB` → tái dùng cross-session. FIX kèm: `_applyBestSchedule`
    trước **bỏ qua** kết quả `fromCache` (cache chỉ tiết kiệm search, KHÔNG re-apply schedule → run-2 rớt về baseline)
    → giờ apply cached sketch+params → **tái lập qua phiên** (run-2 sinh source y hệt run-1).
  - [x] **Time-budget enforce** (đợt 43): `Deadline(timeBudgetMs, clock)` chia sẻ toàn `tune()`; Random/Evolutionary
    search + refine-loop break khi hết hạn. Clock injectable (test deterministic).
- **Done when:** Tuned kernel ≥ baseline trên benchmark thật, tái lập qua phiên, tôn trọng time budget. ✅
- **Refs:** `src/compiler/autotune/{autotuner,cost_model,tuning_db,benchmark,search,budget}.js`.

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
  - [x] **(mở rộng) Quant PASS tự auto-detect per-channel weight + fold param→const + top-1 model thật** (đợt 44):
    (1) `QuantizationPass` nay tự nhận diện weight per-channel: khi `scheme=PER_CHANNEL` và rhs của `dot` là constant
    array → sinh `quantized_dot` (rhs int8 đã quantize per-channel) + dequant per-channel (`convert`→`broadcast_in_dim`
    scaleVec→`mul`), thay vì âm thầm rớt về per-tensor như trước. (2) **fold param→const**: `foldWeightParams` hạ
    captured-weight-param (block-arg khi trace) thành `constant` baked-in + prune khỏi `capturedParams`/inputTypes
    (`Block.removeArgument`); wire qua `compile(model, x, { foldWeights:true })`. (3) **transpose constant-fold** mới
    (`transpose(const)`→`const`) để `Linear` (`matmul(x, transpose(W))`) sau khi fold W lộ rhs constant cho pass.
    (4) Top-1 model thật: classifier traced (`Sequential(Linear)`) quantize per-channel int8 → top-1 == float == nhãn
    (100%/40 mẫu). Per-channel error <per-tensor **2.18×** (weight lệch + activation chặt), cpu==wasm bit-khớp.
    +9 test. 4 revert-test load-bearing.
- **Done when:** ✅ Per-channel params + int8 kernel thật + array-const bug fix + per-channel compiled harness
  (cpu==wasm). Auto-pass-detection cho trace-param-weight là mở rộng (capability đã chứng minh).
- **Refs:** `src/compiler/ir/graph/quantization_types.js`, `src/compiler/passes/lowering/lowering_registry.js`
  (`lowerConstant`), `tests/compiler/quantization/`, `tests/compiler/lowering/elementwise-lowering.test.js`.

### P1.5 — Pass pipeline hội tụ (fixed-point) thay vì tuần tự cứng  ✅ (đợt 36)
- **Why:** PassManager tuần tự, phase-ordering hand-tuned → giòn (đổi thứ tự canonicalize/simplify/fold dễ vỡ).
  XLA dùng `HloPassPipeline` lặp tới fixed-point.
- **What:**
  - [x] `FixedPointGroup(name, passes, maxIterations)` trong `pass_manager.js`: lặp nhóm pass tới khi MỘT lượt
    không pass nào CHANGED (hội tụ) hoặc chạm `maxIterations` (guard chống loop). Refactor per-pass logic thành
    `_applyPass` trả `{changed, fatal}` để dùng chung cho path thường + group (không lặp code, giữ nguyên hành vi
    verify-hook/error/resilient — 7 test cũ vẫn xanh).
  - [x] Wire cluster canonicalize `[Canonicalize, AlgebraicSimplify, ConstantFold, CSE, DCE]` thành FixedPointGroup
    trong `_runGraphPasses`. `maxSimplifyIterations` (config, default 8, KHÔNG hardcode). Các pass đã report
    CHANGED/UNCHANGED chính xác (applyPatterns hội tụ nội bộ) → group dừng sớm ở fixed-point (2 vòng), không blow-up.
  - [x] `preservedAnalyses`: cơ chế sẵn (`invalidate(func, pass.preservedAnalyses)`); default bảo toàn (rỗng = invalidate
    hết) là SOUND. Khai báo preserved per-pass là tối ưu perf (analyses lazy) — để mở.
  - [x] Test: `tests/compiler/passes/pass-manager.test.js` (FixedPointGroup: lặp tới settle = 3 applies; max-iter cap;
    hội tụ với one-shot change). Full regression **4003 pass / 0 fail**, thời gian không đổi (~12s).
- **Done when:** ✅ Nhóm canonicalize lặp tới fixed-point, max-iter chống loop; toàn bộ differential/correctness cũ
  vẫn xanh (bằng chứng order-độc-lập: cluster giờ chạy tới điểm bất động, kết quả không đổi).
- **Refs:** `src/compiler/passes/pass_manager.js`, `src/compiler/pipeline/compiler.js:_runGraphPasses`.

---

## P2 — Độ chín & vận hành

### P2.1 — Observability / debug compiler  ✅ (đợt 40, explain log xong)
- **What:**
  - [x] IR dump mỗi phase: ĐÃ CÓ (`trace.irDump`/`shouldSnapshot` afterGraphPasses/afterLowering/afterScheduling).
  - [x] **"Why-not-fused" / "why-this-schedule" explain log** (đợt 40): `TraceLog.explain(category, subject,
    decision, reason, data)` ở DEBUG level (no-op mặc định). Wire:
    - Fusion (XLA `FusionPass` + `DominatorFusionPass`): mỗi group → `fused`/`not-fused` + lý do (cost-model
      reason / cycle / reduction-limit / không inline-fusable). Vd: `div+reduce+exp+sub -> not-fused: fusing
      would create a dependency cycle`.
    - Schedule (`SchedulePolicy.applyToBlock`): mỗi block → rule áp + lý do (vd `reduce_acc_5 -> reduction_cpu`),
      hoặc `none: no rule matched; runs sequentially`.
    - Autotune (`_scheduleAll`): mỗi block → sketch chọn + score (`reduce_acc -> reduction_cpu: autotuned, score
      3.758`).
    - Test: `pipeline-config.test.js` (fusion explain shape; cyclic-dominator not-fused reason; schedule rule
      per block; autotune sketch per block). Gated DEBUG → 0 perf mặc định.
  - [x] **Crash repro tự động** (đợt 36): `compile()` đính `error.repro` khi compile/run throw — gồm `phase`
    (compile/run), `target`, `inputs` (shape+dtype mỗi input), `config` (fusion/scheduling/optimization/quant/
    dynamicShapes). Best-effort, không che lỗi gốc. Wrap `_compile` + `_execute` (sync+async). Test:
    `tests/tracing/compile.test.js` ("compile failures carry a reproduction context").
- **Done when:** ✅ Mọi quyết định fuse/schedule có explain (subject+decision+reason) ở DEBUG; crash repro; IR dump.
- **Refs:** `src/compiler/pipeline/trace.js` (`explain`), `passes/fusion/{fusion_pass,dominator_fusion}.js`,
  `schedule/rules.js` (SchedulePolicy), `pipeline/compiler.js` (autotune explain), `src/tracing/compile.js`.

### P2.2 — API stability & error UX  ◑ (đợt 36, error UX xong)
- **What:**
  - [ ] Chốt public API (`compile`, `CompilerConfig`, target factories); semver. (quyết định docs, ngoài code)
  - [x] **Thông báo lỗi có ngữ cảnh** (đợt 36): `_inferAndBuild` lỗi giờ kèm op-name + lý do (op chưa đăng ký /
    inferResultTypes trả rỗng / thiếu) + operand `[shape]:dtype` (`describeType`). Crash-repro (P2.1) thêm phase/
    target/config. Test: `tests/compiler/ir/graph/operation.test.js` ("inference errors carry operand context").
  - [x] `resilient` mode trả func fail + lý do: ĐÃ CÓ (`CompilationResult.errors` + getter `failedFunctions`;
    `PassManager` resilient gom lỗi/func-fail, không nuốt). Xác nhận.
- **Refs:** `src/compiler/ir/graph/builder.js` (`describeType`), `src/tracing/compile.js`, `src/compiler/pipeline/compiler.js` (CompilationError/CompilationResult).

### P2.3 — Dynamic shapes diện rộng  ✅ (đợt 36 + 39 + 41)
- **Why:** sym_int chạy end-to-end (verify đợt P2 edge), nhưng cần phủ rộng hơn cho production serving.
- **What:**
  - [x] Specialization cache theo shape-class: ĐÃ CÓ (`compile.js` `_cacheEntries` + shape-guard `evaluateGuards`,
    compile-once-run-many). Xác nhận.
  - [x] **Differential dynamic-shape rộng hơn** (đợt 36): thêm **matmul-chain** (dyn M), **attention-like**
    (softmax→matmul, dyn rows), **layernorm** (dyn batch) vào `differential-nn.test.js` block dynamic (vs eager,
    cpu+wasm). Cùng matmul dyn M/K, reduce, transpose, two-dynamic-dims sẵn có.
  - [x] **conv2d/conv1d dynamic batch N** (đợt 39): FIX — loop accumulate dùng `IntImmNode(batch=-1)` thay
    `ctx.extentNode(...)` → `for cn < -1` không chạy → ra 0. Thay mọi extent buffer-shape bằng `ctx.extentNode`
    (`linalg.js` + `quantization.js`). Test: differential-nn dynamic (+conv2d/conv1d/pad-stride × cpu+wasm).
  - [x] **Bucketing/guard nâng cao cho dynamic dim** (đợt 41): `compile()` thêm `shapeBuckets` — pre-compile
    kernel TĨNH (tối ưu hơn) cho các shape "nóng" khai báo trước, đặt TRƯỚC entry dynamic trong cache. Runtime
    `_findCachedEntry` ưu tiên bucket khớp chính xác (static kernel nhanh), shape khác rơi về dynamic fallback.
    Không padding/mask (đúng tuyệt đối). Test: `tests/tracing/compile.test.js` (bucket shape dùng static entry,
    shape ngoài bucket dùng dynamic; cả hai đúng giá trị).
- **Refs:** `src/tracing/compile.js` (dynamicShapes/shapeBuckets), `src/compiler/analysis/sym_int.js`, `tests/e2e/differential-nn.test.js`.

### P2.4 — Memory planning nâng cấp  ✅ (đợt 36 + 42)
- **What:**
  - [x] **Best-fit + đo fragmentation** (đợt 36): `MemoryPool` thêm `strategy` ('best-fit' default | 'first-fit')
    — best-fit chọn gap KHÍT NHẤT (giảm phân mảnh khi size khác nhau) thay vì gap thấp nhất. Viết lại `_findFreeOffset`
    gap-based O(L log L) (trước O(L²)/allocate). Thêm `fragmentation()` = phần peak không bị live-block chiếm.
  - [x] **Interference-graph allocation + FIX BUG allocator default** (đợt 42): implement allocation theo interference
    chính xác (gán offset né MỌI buffer interfering = interval chồng nhau, không chỉ "active set"). **Phát hiện bug
    thật**: `BufferAssignment` cũ (active-set + sort size-desc) FREE buffer theo thứ-tự-xử-lý, một buffer đã free có
    thể bị buffer xử-lý-sau (interval chồng) tái-overlap → 2 buffer SỐNG-CÙNG-LÚC dùng CHUNG offset (sai kết quả
    tiềm ẩn). Fuzz 2000 interval-set: best-fit cũ **335 vi phạm**, interference **0**. Fix: cả 2 strategy dùng
    `_interferenceOffset` (xét toàn bộ interfering placed; 'best-fit'=gap khít nhất, 'interference'/'first-fit'=gap
    thấp nhất) → đều conflict-free. Interference cũng pack chặt hơn (559 win vs 195 trên fuzz). Config
    `memory.allocStrategy`. Test: `assignment-memory.test.js` (seed-12 regression + 200 random conflict-free) +
    `pipeline-config.test.js` (e2e interference==best-fit). Revert (HEAD allocator) → seed-12 vi phạm 1. Full
    suite **4031/4031**.
  - [x] **Remat dùng memory-budget thật của target** (đợt 42): trước remat budget mặc định `Infinity` → no-op luôn.
    Thêm `target.memoryBudgetBytes`; compiler suy `rematConfig.memoryBudget` từ target nếu chưa set → remat THỰC SỰ
    chạy khi target có giới hạn bộ nhớ. Test: `remat-memory.test.js` (target budget → pass dùng budget đó; không
    budget → Infinity/no-op). (Remat↔scheduling co-design sâu = P3 redesign, ngoài phạm vi.)
- **Refs:** `src/compiler/passes/memory/{buffer_assignment,memory_planning,rematerialization}.js`, `src/backend/target.js`.

---

## P3 — Mở rộng / dài hạn

### P3.1 — Schedule trace-authoritative (tái thiết kế)  ✅ legality-per-primitive (đợt 48)
- **Why:** Schedule đang mutate IR in-place + rebuild `SRefTree` như index → cả lớp bug kiểu K. TensorIR coi
  schedule-state là chân lý, validate từng primitive.
- **What:**
  - [ ] Chuyển sang: schedule = chuỗi decision áp lên state đã-validate; IR là sản phẩm sinh ra. (còn — tái thiết kế lớn)
  - [x] **Mỗi primitive kiểm tra legality TRƯỚC khi áp** (đợt 48, `schedule/legality.js`): `parallelize`/`vectorize`
    nay **reject reduction loop** (loop-carried dep: biến loop *đọc* nhưng không có trong write-index → các vòng song
    song đua trên accumulator = bug kiểu K, vd race conv/reduction đợt 37). Check chạy TRƯỚC mutation → IR không bao
    giờ vào trạng thái invalid; trace chỉ chứa decision đã-validate (trace-authoritative). Backstop: `ScheduleValidator`
    cũng flag PARALLEL/VECTORIZED reduction loop (theo-dõi stack par-loop, O(n·depth), KHÔNG O(n²)). Spatial loop
    (biến ∈ write-index) vẫn parallelize/vectorize bình thường → không false-positive (full regression xác nhận).
- **Done when:** Không primitive nào tạo được IR vi phạm invariant codegen. ✅ cho lớp reduction-race (parallelize/
  vectorize); các invariant khác đã có sẵn trong validator.
- **Refs:** `src/compiler/schedule/{schedule,validator,legality,sref,schedule_state,trace}.js`.

### P3.2 — Ansor-style sketch auto-generation  ✅ (đợt 45)
- **Why:** Sketch hiện hand-written (AutoTVM template). Ansor auto-generate phủ pattern rộng hơn.
- **What (đợt 45):** `getSketchesForBlock` viết lại thành **rule-based derivation** (Ansor-style) thay vì chọn template
  bằng `blockName.includes('matmul')` (string-match hardcode, cũ). Phân loại **cấu trúc** từ IR (`analyzeBlockStructure`:
  spatial/reduction loop count + read-buffer count + hasReduction), rồi áp tập **derivation rules** (`CPU_SKETCH_RULES`/
  `GPU_SKETCH_RULES`) → sinh **nhiều** sketch/block (Ansor diversity). Rule compute-intensive (reduction + spatial≥1 +
  ≥2 read) → `mlt_cpu` (multi-level tiling MỚI: tile spatial0/spatial1/reduction + parallelize + vectorize) **+**
  `reduction_cpu`. GPU matmul rule gated `spatial===2` (matmul thật, không áp template matmul cho conv). Reduction-only
  → `reduction_*`; spatial-only → `elementwise_*`. Backward-compat: pure-reduction block vẫn `reduction_cpu/gpu` đầu tiên.
  Bỏ hoàn toàn string-match. +3 test (matmul-struct→mlt; matmul-NAME-nhưng-elementwise→elementwise [chứng minh hết
  hardcode]; mlt sinh schedule hợp lệ toàn param-space). 2 revert-test load-bearing. Không O(n²) (O(loops)+O(rules)).
- **Refs:** `src/compiler/autotune/search_space.js`.

### P3.3 — Backend mở rộng
- [ ] LLVM/native (qua WASM→native hoặc binding) cho server.
- [ ] Library-op offload (BLAS/cuBLAS/oneDNN) thay vì codegen thuần cho matmul/conv lớn.

### P3.4 — Training path production  ✅ (đợt 46–47)
- AD đã có (VJP/joint/backward builder); cần optimizer fusion, grad checkpoint policy, mixed-precision loop.
- **What (đợt 46):**
  - [x] **Optimizer fusion** (`src/optim/fused.js`): `FusedSGD`/`FusedAdam` build bước update thành **đồ thị
    elementwise** rồi `compileGraph` → **1 kernel fused** (SGD `w-lr·g`, Adam `m/v/w` multi-output → **1 loop, 0 temp**)
    thay vì nhiều vòng eager. Update **in-place** (alias buffer in=out), cache kernel theo `numel` (O(số-shape)), khớp
    eager tới ~1e-7 (SGD plain/mom/wd/nesterov; Adam/wd/amsgrad). Scalar step-varying (lr, stepSize, bc2sqrt) là input
    rank-0; hằng cấu trúc (betas/eps/momentum/wd) baked.
  - [x] **Grad checkpoint policy** — ĐÃ CÓ SẴN (`ad/checkpoint_policy.js`: EveryK/Sqrt/MemoryBudget/Explicit +
    boundary; `ad/remat_policy.js`). Xác nhận.
  - [x] **Mixed-precision loop — loss scaling** (`src/optim/grad_scaler.js`): `GradScaler` dynamic loss-scale
    (`scale`/`unscale_`/`step`/`update`): phát hiện grad inf/nan → **skip step** + **backoff** scale; chuỗi clean ≥
    `growthInterval` → **grow**. `enabled:false` = pass-through. Default PyTorch (initScale 2¹⁶, growth 2000).
  - [x] **f16-autocast graph pass** (đợt 47, `src/compiler/passes/precision/mixed_precision.js`): `applyAutocast`/
    `MixedPrecisionPass` bọc op trong allow-list (default `dot`/`conv`) thành `convert→f16 → op(f16) → convert→f32`
    (không cần dtype-propagation; convert thừa giữa 2 allow-op kề nhau tự triệt tiêu qua simplifier). Xác nhận f16
    codegen CPU/wasm THẬT (helper `__mlfw_f32_to_f16`/`f16_to_f32`; quanh compute matmul ra sai số **~3e-3** đúng độ lớn
    f16; op block-list như `reduce` giữ f32 chính xác). +5 test, 1 revert-test load-bearing.
- **Refs:** `src/compiler/ad/`, `src/optim/{fused,grad_scaler}.js`, `src/compiler/passes/precision/mixed_precision.js`.

---

## Định nghĩa "production-ready" (exit criteria tổng)
1. P0 xanh toàn bộ: soundness IEEE, verifier-as-truth, schedule↔codegen invariant, differential gate.
2. ≥1 backend (cpu **và** wasm) qua differential + nightly fuzz, 0 known-wrong-result.
3. GPU/WebGPU hoặc xanh-trong-CI hoặc gắn nhãn experimental rõ ràng.
4. Autotune đo thật + tuning DB bền; tuned ≥ baseline.  ✅ (đợt 43)
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
- 2026-06-12 (đợt 31b–35b): **WebGPU production** (xem todo.md đợt 31–34) — máy có GPU, oracle = real Chrome qua
  Puppeteer (Dawn npm flaky). Fix: binding-layout pruning, dtype marshalling (mọi dtype + index ops), slot-reuse,
  reduction GPU-safe (race), matmul cross-thread (workgroup+barrier). `webgpu-chrome.test.js` (Chrome differential)
  + unit/e2e cho từng fix. P1.1 (WebGPU verify) coi như đạt cho default + scheduling (conv+pool+autotune còn gap).
- 2026-06-12 (đợt 36): **P1.5 + P2 (P2.1/P2.2 xong, P2.3/P2.4 phần lớn)**.
  - P1.5: `FixedPointGroup` lặp cluster canonicalize tới fixed-point (max-iter guard, config `maxSimplifyIterations`);
    refactor `_applyPass` dùng chung. +3 test.
  - P2.1: crash-repro `error.repro` (phase/target/inputs/config) khi compile/run throw. +1 test.
  - P2.2: lỗi infer-result-types kèm op + lý do + operand `[shape]:dtype`. +1 test. (resilient-mode đã có)
  - P2.3: differential dynamic-shape +matmul-chain/attention/layernorm. +cases. (conv-dyn batch là gap, ghi rõ)
  - P2.4: best-fit + `fragmentation()`, `_findFreeOffset` O(L²)→O(L log L). +2 test.
  - KHÔNG comment/hardcode/O(n²). Full regression **4013 pass / 0 fail** + webgpu-chrome 14/14.
- 2026-06-12 (đợt 37–39): bug-hunt tiếp (xem todo.md) — conv+maxpool autotune (cross-workgroup serialize),
  dominator fusion cycle (NaN/Inf), conv dynamic-batch (-1 extent), browser-bundle pooling (node-type tag esbuild
  rename). Full regression **4021/4021**.
- 2026-06-12 (đợt 40): **P2.1 xong** — explain log "why-not-fused / why-this-schedule" cho từng block ở DEBUG
  (fusion cost/cycle/reduction-limit; schedule rule per block; autotune sketch+score). `TraceLog.explain`. +4 test.
  No-op mặc định (gated DEBUG). Full regression **4025/4025**.
- 2026-06-12 (đợt 42): **P2.4 xong** — interference-graph allocation (phát hiện+fix BUG allocator default: 2 buffer
  sống-cùng-lúc dùng chung offset, 335/2000 fuzz vi phạm → 0) + remat dùng `target.memoryBudgetBytes`. +6 test.
  Revert-test load-bearing. Full regression **4031/4031**.
- 2026-06-13 (đợt 48): **P3.1 legality-per-primitive xong** — `parallelize`/`vectorize` reject reduction loop TRƯỚC
  khi mutate (`schedule/legality.js`: `loopCarriesReduction` = biến loop đọc nhưng ∉ write-index của block đóng);
  IR không bao giờ invalid → trace-authoritative. Backstop trong `ScheduleValidator` (stack par-loop, O(n·depth) KHÔNG
  O(n²)). +6 test, 2 revert-test load-bearing (primitive 3 RED + validator 1 RED). Spatial loop không bị false-positive.
  Full regression **4158/4159** (1 = ResNet vitest-timeout flake; standalone 2/2 pass). Còn phần tái-thiết-kế lớn
  (IR-as-generated-product) — chưa.
- 2026-06-13 (đợt 47): **P3.4 xong nốt** — f16-autocast graph pass (`applyAutocast`/`MixedPrecisionPass`): bọc
  allow-op (`dot`/`conv`) thành `convert→f16 → op → convert→f32`. Xác minh f16 codegen CPU thật (sai số ~3e-3 quanh
  matmul, `reduce` giữ f32). +5 test, 1 revert-test load-bearing (no-op→3 RED). Full regression **4153/4153**.
- 2026-06-13 (đợt 46): **P3.4 phần lớn xong** — training path production. (1) **Optimizer fusion**: `FusedSGD`/
  `FusedAdam` compile update thành đồ thị elementwise → 1 kernel fused (1 loop, 0 temp; Adam m/v/w multi-output),
  in-place, cache theo numel, == eager ~1e-7. (2) **Grad checkpoint**: đã có sẵn (checkpoint_policy/remat_policy),
  xác nhận. (3) **Mixed-precision loop**: `GradScaler` dynamic loss-scaling (skip-step+backoff khi inf/nan, grow sau
  growthInterval clean). +19 test (12 fused + 7 scaler), 2 revert-test load-bearing (tắt fusion→nhiều loop RED; bỏ
  overflow-guard→step-không-skip RED). Còn f16-autocast graph pass (hoãn — rủi ro f16 codegen). No comment/hardcode/
  O(n²). Full regression **4148/4148**.
- 2026-06-13 (đợt 45): **P3.2 xong** + **audit O(n²)/hardcode feature cũ**. (A) P3.2 Ansor-style sketch
  auto-generation: `getSketchesForBlock` rule-based derivation theo CẤU TRÚC (bỏ `blockName.includes('matmul')`
  string-match), sinh nhiều sketch/block; thêm `mlt_cpu` (multi-level tiling) + `createMultiLevelTileCPUSketch`; GPU
  matmul gated `spatial===2` (không áp lên conv). +3 test, 2 revert-test load-bearing. (B) Audit feature cũ (đợt 32–42)
  bằng 5 sub-agent + verify: **fix 4 vi phạm thật** — `quantization_patterns.js` hardcode `?8:8`→`scalarBytes*8`;
  `partition_pass.js` O(N²) (order-index/find-insert rebuild mỗi edge)→precompute 1-lần O(N×devices); `tracer.js`
  `captureConstant` O(P²) (rebuild inputTypes mỗi param)→mutable+freeze-1-lần O(P); `codegen.js` O(B²) `.some()`→Set.
  (C) Xoá dead-code `dep_analysis.allDeps` (O(blocks²), 0 caller). Báo cáo phần intrinsic/bounded/defensible không sửa.
  Full regression **4128/4129** (1 fail = Dawn-node flake concurrency; conv-GPU test standalone 11/11).
- 2026-06-13 (đợt 44): **P1.4 mở rộng xong** — Quant PASS tự auto-detect per-channel weight + fold param→const +
  top-1 model thật. (1) `QuantizationPass` per-channel cho `dot` constant-weight: int8 const + dequant per-channel
  (convert→broadcast→mul); trước đó `scheme=PER_CHANNEL` âm thầm rớt về per-tensor-asymmetric. (2) `foldWeightParams`
  hạ captured-weight-param→constant (+`Block.removeArgument`, prune capturedParams/inputTypes), wire qua
  `compile(...,{foldWeights:true})`. (3) `transpose(const)`→`const` constant-fold (cho `Linear` lộ rhs constant).
  (4) Top-1 classifier traced per-channel int8 == float == nhãn (100%); per-channel err <per-tensor 2.18×; cpu==wasm.
  +9 test (transpose-fold×2, per-channel pass×2, foldWeightParams×2, fold-wiring×1, top-1×1, + numerics). 4 revert-test
  load-bearing. Full regression **4125/4126** (1 fail = flake có sẵn: ResNet-stress dùng `Math.random` init không seed
  → output đôi khi non-finite; ngoài path đụng tới).
- 2026-06-13 (đợt 43): **P1.2 xong** — autotune đo thật + cost-model mạnh hơn + DB bền + time-budget. (1) `robustStats`
  median/min/trimmed-mean/cv + re-measure khi nhiễu cao (bounded); (2) `LearnedCostModel` linear→**MLP** 1-hidden-tanh
  chuẩn-hoá + seeded (XOR MSE→0 vs linear floor 0.25); (3) `TuningDatabase` versioning (`CODEGEN_VERSION` invalidate)
  + `saveToFile/loadFromFile` (DI fs, browser-safe) + inject qua `scheduling.tuningDB`; FIX `_applyBestSchedule` bỏ
  qua `fromCache` → giờ re-apply (tái lập cross-session: run-2 source ≡ run-1); (4) `Deadline` enforce `timeBudgetMs`
  ở search + refine. +15 test. 4 revert-test load-bearing (budget×2, MLP, DB-version, cache-apply). No comment/
  hardcode/O(n²). Browser-bundle OK. Full regression **4119/4119**.
- 2026-06-12 (đợt 41): **P2.3 xong** — conv dynamic-batch đã fix (đợt 39), cập nhật feature.md. Thêm `shapeBuckets`:
  pre-compile kernel TĨNH cho shape "nóng" khai báo + dynamic fallback (entry static đặt trước, runtime ưu tiên
  khớp chính xác). Không padding/mask → đúng tuyệt đối. +1 test (bucket dùng static kernel `!/_ds/`, off-bucket
  dùng dynamic, đều đúng). Revert → RED. Full regression **4026/4026**.
