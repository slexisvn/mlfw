# Documentation and implementation issues found while reviewing `docs/`

Phạm vi: nội dung đã viết từ Part 0 đến Part VIII (Chapter 1–48).

Nhãn mức độ:

- **P0**: có thể làm thay đổi kết quả chương trình hoặc phá vỡ correctness contract công khai.
- **P1**: khẳng định kỹ thuật sai/quá rộng, invariant không được implementation bảo đảm, hoặc ví dụ có thể dẫn người đọc tới kết luận sai.
- **P2**: vấn đề diễn đạt, thuật ngữ, cấu trúc hoặc benchmark methodology.

Tài liệu này chỉ liệt kê issue, không bao gồm phương án sửa.

## Cross-part issues

- [ ] **[P1][Completeness]** Tài liệu/mục lục tạo cảm giác bộ sách đã có đủ 67 chương và các appendix, nhưng repository hiện chỉ có nội dung hoàn chỉnh cho Chapter 1–48; Part IX–XII và một số appendix vẫn chỉ là kế hoạch.
- [ ] **[P1][Formal claims]** Các nhãn “Theorem”, “Proposition” và “Corollary” đang trộn lẫn kết quả kinh điển, invariant mong muốn, quan sát thực nghiệm và claim về implementation; nhiều claim không có đủ giả thiết hoặc không đúng với code hiện tại.
- [ ] **[P1][Numerical semantics]** Không có một numerical-semantics contract xuyên suốt để phân biệt exact equivalence, tolerance-based equivalence, fast-math, reassociation và reduction-order dependence.
- [ ] **[P2][Audience]** Luồng nhập môn compiler và luồng forensic audit dành cho maintainer được trộn trong cùng narrative, làm độ khó tăng đột ngột từ các Part giữa trở đi.
- [ ] **[P2][Terminology]** Các khái niệm graph op, loop, generated CPU function, kernel và device launch đôi lúc được dùng thay thế cho nhau dù không tương đương.
- [ ] **[P2][Benchmarking]** Nhiều lab dùng giá trị nhỏ nhất của nhiều lần chạy làm đại diện và gọi đó là phép đo ổn định/robust; kết quả dễ chịu ảnh hưởng của noise và không thể hiện dispersion.
- [ ] **[P2][Link]** Có internal link hỏng tại `docs/OUTLINE.md:165`.

## Part 0 — Orientation

### Part-level

- [ ] **[P1][Expectation]** `docs/part0/README.md` nói mỗi chapter kết thúc bằng lab, nhưng Chapter 1 không có lab.
- [ ] **[P2][Prerequisites]** Phần prerequisite đánh giá thấp lượng calculus, linear algebra, floating-point semantics và program analysis được sử dụng ở các Part sau.

### Chapter 1 — What this book is, and how to read it

- [ ] **[P1][Theorem 1.2]** Điều kiện “side-effect-free operations” mâu thuẫn với phần mô tả các operation có thể ghi vào cùng một storage location (`docs/part0/ch01-what-this-book-is/README.md:121`).
- [ ] **[P1][Overclaim]** Câu mô tả thiết kế LLVM/MLIR/TVM/XLA là “correct” mang tính tuyệt đối và không xác định correctness contract nào đang được nói tới.
- [ ] **[P1][Completeness]** Chapter giới thiệu appendix/generated appendix như nội dung sẵn có dù các phần này chưa tồn tại đầy đủ trong repository.
- [ ] **[P2][Terminology]** Cụm “machine code, or rather JavaScript” gọi JavaScript target source là machine code.
- [ ] **[P2][Evidence]** Cụm “tests contain executable proof” đồng nhất test evidence với proof.

### Chapter 2 — Setting up

- [ ] **[P2][Onboarding]** Ước lượng “about ten minutes” không phản ánh khối lượng setup, đọc và chạy lab đối với người hoàn toàn mới.
- [ ] **[P1][Contradiction]** Mô tả CUDA/WebGPU tests vừa nói môi trường thiếu backend sẽ skip, vừa nói chạy không có browser sẽ fail; expected behavior không nhất quán.

### Chapter 3 — A map of the codebase

- [ ] **[P1][TIR model]** Câu TIR “knows nothing about layout” quá tuyệt đối; TIR đã chứa buffer, index expression, scope và các thông tin liên quan trực tiếp đến layout.
- [ ] **[P1][Eager path]** Eager execution được mô tả là gọi “hand-written kernels”, nhưng một số arithmetic operation thực tế sử dụng cached single-op JIT path.
- [ ] **[P1][Pass contract]** Câu một pass “reports no change” vẫn có thể thực hiện công việc quyết định mâu thuẫn với change/invalidation contract được dùng ở Part III (`docs/part0/ch03-map-of-the-codebase/README.md:227`).
- [ ] **[P2][Tracing]** Sơ đồ codebase không giải thích đầy đủ nested trace phases và việc phase nào đang sở hữu active trace state.

## Part I — Why machine learning needs a compiler

### Part-level

- [ ] **[P1][Performance model]** Các mô hình cost được trình bày như định lượng tổng quát nhưng nhiều giả thiết về cache, allocation, dispatch, reuse và backend không được nêu rõ.

### Chapter 4 — Eager execution, and where it hurts

- [ ] **[P1][Lab math]** Lab không thực sự fit mô hình `T(n) = alpha + beta*n`; nó suy ra đại lượng từ `T(1)` và `T(N)/N` thay vì giải đường thẳng từ hai hay nhiều điểm đo.
- [ ] **[P1][Memory traffic]** `add` và `tanh` được so sánh như có cùng traffic; với f32, binary add đọc hai input và ghi một output, còn unary tanh đọc một input và ghi một output.
- [ ] **[P1][Implementation claim]** Khẳng định `Math.tanh` được thực hiện bằng một polynomial cụ thể không được JavaScript specification bảo đảm.
- [ ] **[P1][Matmul model]** Công thức traffic `12n^2` cho matmul giả định ideal reuse/lower-bound traffic nhưng giả thiết này không được ghi trong claim.
- [ ] **[P1][Amdahl analysis]** Toàn bộ eager `tanh` time bị xem là phần không thể tối ưu, dù compiler có thể loại dispatch, allocation và intermediate memory traffic; ceiling `1.49x` không được suy ra từ phép đo đã trình bày.
- [ ] **[P2][Benchmarking]** Kết quả performance dựa vào best/minimum timing thay vì phân phối timing.

### Chapter 5 — From a sequence of calls to a program

- [ ] **[P0][Compilation guard]** Compiled artifact được trace từ f32 có thể được gọi với i32 cùng shape mà không recompile; `div(2)` cho kết quả eager i32 khác compiled-from-f32.
- [ ] **[P0][Compilation guard]** Public cache/signature contract không bao phủ đầy đủ dtype và device dù chúng có thể thay đổi semantics/lowering.
- [ ] **[P1][Theorem 5.3]** Claim về captured state không phân biệt user inputs, lifted parameter tensors, host scalars và folded parameters; parameter tensor mutation sau compile vẫn được artifact quan sát trong trường hợp đã kiểm tra.
- [ ] **[P1][Shape contract]** Exact-shape theorem không mô tả đúng các path có dynamic shape/abstract signature và guard.
- [ ] **[P1][Overclaim]** Câu interpreter “can never” thực hiện một số optimization là tuyệt đối; interpreter/JIT hybrid có thể cache, specialize hoặc fuse ở runtime.

### Chapter 6 — The pipeline in one picture

- [ ] **[P1][Phase-order theorem]** Proof chỉ so sánh `A -> B` với `B -> A`, không loại trừ `A -> B -> A`, repeated application hoặc các thứ tự dài hơn.
- [ ] **[P1][Fixed point]** Mô tả fixed-point convergence thiếu giả thiết về determinism, truthful change reporting, hidden state và oscillation.
- [ ] **[P2][Lab coverage]** Narrative nói lab đi qua Graph IR, TIR và LIR nhưng lab không hiển thị LIR tương ứng.

### Chapter 7 — Vocabulary

- [ ] **[P2][Learning load]** Chapter đưa toàn bộ glossary như tuyến đọc chính thay vì reference, tạo lượng thuật ngữ phải ghi nhớ trước khi người mới có đủ ví dụ trực quan.
- [ ] **[P2][Counting]** Số lượng scheduling primitives được nhắc tới không nhất quán với danh sách/code hiện tại ở các Part sau.

## Part II — Representing programs

### Part-level

- [ ] **[P1][Type theory]** Part README gọi shape compatibility là partial order, trong khi Chapter 10 đưa ra compatibility relation không transitive (`docs/part2/README.md:25`).

### Chapter 8 — SSA and dataflow

- [ ] **[P1][Graph direction]** Use-def edge được định nghĩa theo hướng consumer -> producer, nhưng theorem dùng “topological order” với producer đứng trước consumer; với hướng edge đã định nghĩa phải là reverse topological order (`docs/part2/ch08-ssa-and-dataflow/README.md:39`).
- [ ] **[P1][Operation order]** Ví dụ đảo operation order cũng di chuyển return/terminator lên đầu; parser có thể tạo representation này nhưng verifier bác bỏ, nên claim về reorder operation thiếu điều kiện terminator.

### Chapter 9 — Value, Operation, Block, Region, Function, Module

- [ ] **[P0][Mutation/versioning]** Claim “every edit bumps version” sai: `Operation.setAttr` và `removeAttr` không bump module version (`src/compiler/ir/graph/operation.ts:84`).
- [ ] **[P1][Encapsulation]** Claim mọi mutation đi qua một đường kiểm soát không đúng vì nhiều mutable container vẫn public và có thể bị chỉnh trực tiếp.

### Chapter 10 — The type system

- [ ] **[P1][Compatibility relation]** Shape compatibility được mô tả bằng partial-order terminology dù relation được trình bày là không transitive (`docs/part2/ch10-type-system/README.md:187`).
- [ ] **[P1][Dynamic broadcast]** Các claim về dynamic-shape broadcasting rộng hơn tập trường hợp implementation thực sự chứng minh/xử lý; giới hạn của unknown dimensions không được phản ánh đầy đủ.

### Chapter 11 — Ops as a dialect

- [ ] **[P0][Float traits]** Arithmetic ops khai báo `ASSOCIATIVE` vô điều kiện, bao gồm floating-point add/mul (`src/compiler/ir/graph/ops/helpers.ts:69`); trait này được optimization sử dụng như semantic truth.
- [ ] **[P1][Trait validation]** Traits là declaration không được verifier chứng minh; một op có thể khai báo law sai mà IR vẫn được coi là hợp lệ.

### Chapter 12 — What “valid IR” means

- [ ] **[P1][Validity scope]** “Valid IR” chủ yếu bao phủ structural/type invariants nhưng narrative có lúc hàm ý semantic correctness; verifier không chứng minh trait laws, numerical equivalence hoặc mutation/version consistency.
- [ ] **[P1][Verification timing]** Invalid state có thể tồn tại sau public mutation và chỉ bị phát hiện nếu một verification path cụ thể được chạy.

### Chapter 13 — IR as text

- [ ] **[P1][Round-trip claim]** Textual round trip thể hiện structural/printed equivalence, không bảo đảm identity hoặc lossless preservation của mọi runtime metadata.
- [ ] **[P1][Parser validity]** Parser có thể dựng operation ordering mà verifier sau đó bác bỏ; parsing thành công không đồng nghĩa IR hợp lệ.

## Part III — The transformation infrastructure

### Part-level

- [ ] **[P1][Uniform rewrite model]** Part-level narrative tạo cảm giác optimization passes dùng chung một rewrite engine, nhưng constant folding, CSE và DCE có traversal/worklist riêng; chỉ một phần canonicalization/algebraic rewriting dùng `PatternApplicator`.

### Chapter 14 — What a pass is

- [ ] **[P1][UNCHANGED definition]** `UNCHANGED => P(m) = m` không có ý nghĩa khi pass nhận và mutate cùng object; object identity vẫn giữ nguyên dù IR content đã đổi.
- [ ] **[P1][Change reporting]** False `UNCHANGED` có thể làm analysis cache không invalidated và khiến verifier quy lỗi cho pass chạy sau.

### Chapter 15 — The pass manager

- [ ] **[P1][Verification contract]** “Verify after every pass” không đúng với implementation; verifier chỉ chạy khi pass báo `CHANGED` (`src/compiler/passes/pass_manager.ts:134`).
- [ ] **[P1][Fixed point]** Pass manager có iteration cap nhưng không chứng minh convergence và không phân biệt convergence với oscillation/hitting the cap.
- [ ] **[P2][Benchmarking]** Lab gọi minimum-of-40 là robust measurement dù minimum là thống kê nhạy với noise theo hướng lạc quan.

### Chapter 16 — Analyses and the invalidation problem

- [ ] **[P0][Cache soundness]** Claim versioned analysis cache “sound by construction” sai vì attribute mutation không bump version; cached result có thể khác recomputed result.
- [ ] **[P1][Preservation definition]** Công thức `A(P(m)) = A(m)` quá mạnh và không khớp cache có thể giữ entry cho object đã bị xóa; implementation cần observational validity nhưng chapter phát biểu equality toàn phần.

### Chapter 17 — Pattern rewriting

- [ ] **[P1][Architecture claim]** Constant folding, CSE và DCE không đều được biểu diễn bằng pattern applicator như narrative gợi ý.
- [ ] **[P1][Normal-form evidence]** Lab chỉ cho thấy bốn input text hội tụ tới cùng output; nó không kiểm tra idempotence hoặc việc không còn applicable pattern.
- [ ] **[P0][Float rewriting]** Pattern rewriting sử dụng float `ASSOCIATIVE` trait để reassociate biểu thức dù fast-math tắt.

### Chapter 18 — Watching the compiler work

- [ ] **[P0][Resilient compilation]** Module pass mutate IR rồi throw không được rollback; resilient mode có thể tiếp tục compile IR đã bị thay đổi một phần (`src/compiler/passes/pass_manager.ts:120`).
- [ ] **[P1][Transactional claim]** Cụm “transactional resilient compilation” không đúng với module-level mutation behavior hiện tại.
- [ ] **[P2][Tracing overhead]** Claim disabled tracing chỉ tốn “one integer comparison” không phản ánh object/method dispatch vẫn xảy ra nếu call site không được pre-gate.

## Part IV — Graph-level optimization

### Part-level

- [ ] **[P1][Rule count]** Part IV nói có 32 lowering rules trong khi Part VI mô tả 66 ruled operations; cùng một khái niệm count không nhất quán (`docs/part4/README.md:26`, `docs/part6/README.md:11`).

### Chapter 19 — Constant folding, CSE, and dead code elimination

- [ ] **[P0][f32 constant folding]** Constant folder dùng JavaScript f64 intermediate cho f32 operations; `16777216f + 1f - 16777216f` cho folded result khác stepwise f32 execution (`src/compiler/passes/simplify/constant_fold.ts:62`).
- [ ] **[P1][CSE semantics]** CSE correctness phụ thuộc traits/effect metadata là đúng, nhưng verifier không chứng minh metadata đó phản ánh semantics thực tế.

### Chapter 20 — Algebraic simplification meets IEEE 754

- [ ] **[P0][Reassociation]** `fastMath=false` vẫn cho phép float reassociation qua unconditional `ASSOCIATIVE` trait và canonicalization; đã quan sát eager result `0` nhưng compiled result `1`.
- [ ] **[P1][Documentation contradiction]** Chapter nói compiler không reassociate float khi fast-math tắt, trái với `src/compiler/passes/canonicalize/canonicalize.ts:19` và `src/compiler/ir/graph/patterns.ts:493`.

### Chapter 21 — Decomposition

- [ ] **[P1][Kernel count]** “One emitted kernel” trên CPU chỉ chứng minh có một generated entry function; không chứng minh chỉ có một loop, không có temporary buffer hoặc chỉ có một device launch.
- [ ] **[P2][Lab observability]** Lab không hiển thị LIR/loop structure đủ để hỗ trợ kết luận về số kernel/loop sau decomposition.

### Chapter 22 — Fusion I: why it is the single most valuable optimization

- [ ] **[P1][Kernel terminology]** “Four ops to one kernel” không đúng cho CPU backend: fusion-off vẫn có thể sinh một function chứa nhiều loop và temporary buffers; fusion-on là một function với loop/temp structure khác.
- [ ] **[P1][Cost model]** `launchSaved = 5us` cho mỗi graph op không mô hình hóa CPU execution path nơi không có device launch tương ứng.
- [ ] **[P2][Benchmarking]** Lab dùng minimum-of-20 và không báo variance/percentiles.

### Chapter 23 — Fusion II: legality

- [ ] **[P1][Algorithm attribution]** Implementation Kahn-sorts một rank window nhưng được gán cho Pearce–Kelly; thuật toán và bound được mô tả không khớp paper được viện dẫn.
- [ ] **[P1][Complexity claim]** Bound `O(m^(3/2))` được gán cho Pearce–Kelly dù paper dùng affected-set analysis và thảo luận bound này trong ngữ cảnh các thuật toán khác.

### Chapter 24 — Fusion III: the three strategies

- [ ] **[P1][Unit mismatch]** Benefit score cộng bytes với microseconds trực tiếp, tạo đại lượng không có unit nhất quán.
- [ ] **[P1][Complexity claim]** Claim fusion partitioning là NP-hard không kèm problem definition/reduction đủ cụ thể để xác định bài toán nào đang được nói tới.
- [ ] **[P2][Stale result]** Con số speedup `2.4x` không khớp kết quả hiện tại và không ghi source snapshot/machine/date.

### Chapter 25 — Layout

- [ ] **[P1][Feature status]** Layout-aware ops/default preference path chưa hoạt động đầy đủ (`layoutAwareOps` rỗng), trong khi narrative có lúc mô tả như một pipeline có hiệu lực chung.
- [ ] **[P1][Conversion count]** Claim reshape “forces two conversions” không khớp algorithm hiện tại; reshape reset row-major preference và conversion phụ thuộc consumer path.
- [ ] **[P2][Performance claim]** Forced layout conversion example có thể chậm hơn baseline nhưng chapter chưa tách rõ demonstration of mechanism khỏi demonstrated optimization benefit.

### Chapter 26 — Three optional pipelines

- [ ] **[P1][Quantization formula]** Công thức quantization trong tài liệu bỏ phép clamp dù implementation có clamp (`src/compiler/passes/quantization/quant_math.ts:16`).
- [ ] **[P1][Default calibration]** Default range `[-6, 6]` có thể tạo sai số lớn trên workload ngoài range; chapter example cho thấy chênh lệch đáng kể nhưng public expectation chưa được giới hạn rõ.
- [ ] **[P1][Rematerialization theorem]** Checkpointing theorem cho chain/backprop được áp sang clone-pure-multiuse rematerialization pass có assumptions và objective khác.
- [ ] **[P2][Lab coverage]** Rematerialization lab đã kiểm tra không thực sự chạm memory budget, nên không chứng minh budget-driven behavior.

## Part V — Automatic differentiation

### Part-level

- [ ] **[P1][Complexity]** Dense-Jacobian time/memory claims được phát biểu quá phổ quát; complexity phụ thuộc representation, sparsity, operation set và cost model.
- [ ] **[P2][Benchmarking]** Finite-difference cost chủ yếu được ước lượng thay vì đo trực tiếp, và timing dùng minimum.

### Chapter 27 — Differentiating programs

- [ ] **[P1][Linear algebra notation]** `Jv` không tự động là một column của Jacobian và `w^T J` không tự động là một row; điều này chỉ đúng khi `v`/`w` là basis vectors phù hợp.
- [ ] **[P1][Memory claim]** Forward mode được mô tả như tránh memory burden nói chung, dù tangent propagation vẫn có memory cost phụ thuộc số direction và intermediate state.

### Chapter 28 — Writing a VJP rule

- [ ] **[P1][Saved values]** Narrative gợi ý fine-grained dependency saving, nhưng backward builder hiện resolve/save mọi operand và result trước khi gọi VJP (`src/compiler/ad/backward_builder.ts:73`).
- [ ] **[P2][Code-size claim]** Câu tất cả VJP rules dài không quá 15 dòng không đúng với source hiện tại.

### Chapter 29 — Building the backward graph

- [ ] **[P1][Joint graph memory]** Joint forward/backward graph vẫn có thể giữ intermediate values; chapter dễ tạo ấn tượng joint compilation tự động giải quyết retention cost.
- [ ] **[P1][Constant representation]** “Dense constant in artifact” nói quá rộng; scalar attributes có thể được materialize dưới dạng typed tensor/value thay vì một dense constant theo nghĩa thông thường.
- [ ] **[P0][Deferred input semantics]** Joint compiled execution có thể đọc backing storage sau thời điểm API call; input mutation trước materialization làm output/gradient phản ánh state mới (`src/tracing/compile_backward.ts:374`).

### Chapter 30 — Trading memory for recomputation

- [ ] **[P1][Feature status]** `scanCheckpoint`, `maxRematDepth` và một số policy knobs được mô tả nhưng chưa được nối vào quyết định thực tế.
- [ ] **[P1][Budget metric]** “Bytes” trong một số accounting path thực tế là element count; produced-memory metric không phải peak live memory.
- [ ] **[P1][Purity assumption]** Rematerialization correctness phụ thuộc operation pure/deterministic nhưng contract này không được enforcement đầy đủ.
- [ ] **[P1][Theorem scope]** Chain checkpointing bound được phát biểu như áp trực tiếp cho current general graph rematerialization implementation.

### Chapter 31 — Differentiating control flow

- [ ] **[P1][Barrier classification]** `one_hot` tạo f32 output nhưng được xếp cùng nhóm nondifferentiable/control barriers; dtype của output không khớp cách phân loại được trình bày.
- [ ] **[P1][Sweep behavior]** Compare/control ops được tạo trong backward graph không được differentiation sweep quay lại xử lý; barrier explanation không mô tả asymmetry này.
- [ ] **[P0][Call-time semantics]** Separate/joint compiled AD không có contract rõ về việc mutation input giữa call, output materialization và backward là hợp lệ hay undefined; đã quan sát output/gradient không cùng phản ánh call-time value.

## Part VI — Lowering to loops: TIR

### Part-level

- [ ] **[P1][Coverage count]** Tổng số lowering rules/covered ops không nhất quán với Part IV; bảng 96 cases gồm 66 ruled, 21 decomposed và 9 structural nhưng các Part dùng count khác nhau mà không phân biệt category.

### Chapter 32 — From tensor algebra to loop nests

- [ ] **[P1][Non-injective claim]** “Non-injective mapping cannot be recovered” quá tuyệt đối; không thể uniquely recover từ mapping/TIR alone không đồng nghĩa không analysis hay extra metadata nào có thể suy ra intent.

### Chapter 33 — Buffers, blocks, iteration variables

- [ ] **[P1][Declared properties]** Block/iteration properties được mô tả như semantic facts, nhưng một phần completeness, affine binding và region information là declaration chưa được verifier chứng minh.

### Chapter 34 — Lowering rules

- [ ] **[P1][Counting taxonomy]** Các con số 96, 66, 21, 9 và 32 xuất hiện ở nhiều Part mà taxonomy “rule/decomposition/structural/covered op” không được giữ nhất quán.
- [ ] **[P1][Coverage implication]** Có lowering rule không tự động chứng minh rule đúng cho mọi dtype, dynamic shape, layout và backend combination mà op registry chấp nhận.

### Chapter 35 — Index arithmetic

- [ ] **[P1][Performance generalization]** Câu integer division là operation chậm nhất trên mọi target quá rộng; cost phụ thuộc architecture, constant divisor lowering, vectorization và surrounding instructions.
- [ ] **[P2][Timing portability]** Các timing được trình bày không gắn machine/runtime/source snapshot, nên không có tính portable.

### Chapter 36 — Dependence analysis

- [ ] **[P1][Definition]** Định nghĩa dependence yêu cầu `I < J`, nhưng phần loop-independent dependence lại cần trường hợp `I = J`; hai phần dùng execution-order relation không nhất quán.
- [ ] **[P0][Direction orientation]** Direction vector được tính từ static source/destination access order thay vì orient theo dynamic earlier/later instance; mixed directions có thể bị đảo (`src/compiler/analysis/dependence.ts:106`, `:175`).
- [ ] **[P0][Reorder legality]** Có loop nest với real WAR direction `(<, >)` mà analysis trả hướng ngược; `Schedule.reorder` được chấp nhận và làm thay đổi output.
- [ ] **[P1][Test oracle]** Existing brute-force dependence tests dùng cùng orientation assumption với implementation, nên không phát hiện lỗi direction trên.

### Chapter 37 — Proving things about indices

- [ ] **[P0][Runtime bounds]** Computed indices như `embedding(i.add(1))` không được host/runtime guard bao phủ; out-of-range access có thể trả `NaN` thay vì throw (`src/compiler/analysis/index_bounds.ts:53`).
- [ ] **[P1][Documentation contradiction]** Main chapter understates computed-index gap trong khi lab đã cảnh báo trường hợp này.
- [ ] **[P1][Precision claim]** Câu affine analysis là exact mâu thuẫn với phần MIV/GCD analysis thừa nhận false positives/imprecision (`docs/part6/ch37-proving-things-about-indices/README.md:330`).
- [ ] **[P1][Reduction semantics]** `CommReduce`/commutative reduction contract không nêu giới hạn do f32 non-associativity và order-sensitive results.

## Part VII — Scheduling

### Part-level

- [ ] **[P0][Soundness claim]** Claim tất cả 22 scheduling primitives đều semantics-preserving/sound sai với current `reorder` và `rfactor` behavior.
- [ ] **[P1][Validation gap]** Schedule rules có path không chạy legality validator trước khi mutation được áp dụng.

### Chapter 38 — Separating what from how

- [ ] **[P0][Core contract]** Schedule được định nghĩa là chỉ thay đổi “how” không đổi “what”, nhưng public `reorder` và `rfactor` hiện có phản ví dụ làm đổi observable result.

### Chapter 39 — The sref tree and block scopes

- [ ] **[P1][Complexity claim]** Claim mutation update cost `O(k * depth)` không phản ánh `replace` subtree traversal và whole-analysis invalidation hiện tại (`docs/part7/ch39-sref-tree-and-block-scopes/README.md:5`).
- [ ] **[P1][Incrementality]** Narrative mô tả local/incremental maintenance rộng hơn mức implementation thực sự duy trì.

### Chapter 40 — Loop primitives

- [ ] **[P0][Split lower bound]** `split` không bảo toàn đúng semantics cho loop có non-zero `min`; generated quotient/remainder mapping giả định lower bound bằng 0.
- [ ] **[P1][Thread binding]** Split copies thread tag sang cả outer và inner loops, có thể tạo binding structure không phản ánh original execution contract.

### Chapter 41 — Memory and reduction primitives

- [ ] **[P0][rfactor product identity]** Product reduction được chấp nhận nhưng factor buffer dùng identity fallback `0`; product `[2,3,4,5]` có thể đổi từ `120` thành `0` (`src/compiler/schedule/schedule.ts:656`).
- [ ] **[P0][rfactor validation]** Check hiện tại chỉ tìm load từ cùng buffer; không chứng minh cùng subscript hoặc update expression độc lập đúng cách với accumulator.
- [ ] **[P1][Iff theorem]** Claim `rfactor` hợp lệ “iff associative and commutative” quá mạnh; đây không phải điều kiện cần/đủ đầy đủ cho mọi program và implementation của transform.
- [ ] **[P1][Reachability claim]** Tài liệu nói multiplication path không reachable, trái với public scheduling API/reproduction đã chạy.

### Chapter 42 — Legality

- [ ] **[P0][Conservative-mask claim]** Proposition nói dependence analysis chỉ có thể bảo thủ và bỏ lỡ optimization, nhưng direction-orientation bug cho false-safe reorder (`docs/part7/ch42-legality/README.md:53`).
- [ ] **[P1][Equivalence theorem]** Claim preserving dependence edges là điều kiện “iff” cho semantic equivalence quá mạnh; dependence preservation là sufficient framework condition nhưng không cần thiết cho mọi observationally equivalent program.

### Chapter 43 — Scheduling for GPUs

- [ ] **[P1][Barrier theorem]** Proposition về barrier repair thiếu giả thiết unique writers, absence/handling of WAW, convergent participation và phase separation (`docs/part7/ch43-scheduling-for-gpus/README.md:47`).
- [ ] **[P1][Dynamic grid]** Dynamic grid limitation được mô tả như hardware fact dù đây chủ yếu là giới hạn của current compile-time scheduling/lowering implementation.
- [ ] **[P1][Race validation]** Barrier insertion analysis không cấu thành general proof rằng generated GPU program race-free.

## Part VIII — Autotuning

### Part-level

- [ ] **[P0][Inherited unsoundness]** Part lặp lại claim 22 scheduling primitives đều sound dù tuner có thể chọn/replay unsound `rfactor` và `reorder` (`docs/part8/README.md:3`).
- [ ] **[P0][Internal contradiction]** Part nói schedule candidates không tạo wrong kernel trong khi chính các limitation về `rfactor`/float reduction cho phép result thay đổi (`docs/part8/README.md:60`).
- [ ] **[P0][End-to-end autotune]** Forced database path đã cache/replay `rfactor` cho reduction nhạy thứ tự và đổi serial result từ `3` thành tuned result `6`.
- [ ] **[P1][Stress test]** Expanded autotune/stress run có 1 deterministic failure tại `tests/stress/autotune.test.js:106`; baseline và autotune cùng giữ `ADD_ZERO`, nên test expectation và attribution không nhất quán.

### Chapter 44 — How big is the search space

- [ ] **[P1][Proposition 44.6]** Claim blind heuristic necessarily pays the full gap between workload-specific optima là sai; có candidate set nơi heuristic có regret nhỏ trên cả hai workload dù hai optimum khác nhau (`docs/part8/ch44-how-big-is-the-search-space/README.md:70`).
- [ ] **[P2][Dynamic extent wording]** “Dynamic extent has no divisors” nhầm compile-time unknown divisors với việc số nguyên runtime không có divisor.
- [ ] **[P1][Search-space count]** Product-space counts giả định decisions độc lập dù legality và earlier schedule choices làm later choices conditional.

### Chapter 45 — Sketches

- [ ] **[P1][Proposition 45.5]** Split mapping theorem thiếu giả thiết loop minimum bằng 0; implementation `split` bỏ qua non-zero `min` (`docs/part8/ch45-sketches/README.md:49`).
- [ ] **[P1][Theorem 45.7 premise]** Conditional soundness theorem có premise hợp lý nhưng current implementation không thỏa premise vì sketch có thể chứa unsound primitives.
- [ ] **[P1][Sketch validity]** Một sketch được dựng thành công không đồng nghĩa candidate đã qua đầy đủ semantic legality validation.

### Chapter 46 — Cost models

- [ ] **[P1][Corollary 46.4]** Claim arbitrarily bad ranking with arbitrarily small MSE quá rộng; exact zero MSE fixes ranking và small MSE plus positive margins có thể constrain ranking (`docs/part8/ch46-cost-models/README.md:46`).
- [ ] **[P1][Backend mismatch]** Example nói model nên rank 64 parallel chunks tốt hơn, nhưng shipped CPU backend không thực thi parallel annotation như implied.
- [ ] **[P1][Training labels]** Whole-function measurement được ghép với mini-feature assumption rằng contribution của các block khác là additive constant; cache/JIT/inter-block effects không bảo đảm giả thiết này.
- [ ] **[P1][Degenerate model]** Current default CPU feature/cost path có thể cho gần-flat scores, làm model không phân biệt candidates dù narrative mô tả meaningful ranking.

### Chapter 47 — Search and measurement

- [ ] **[P1][Theorem 47.8 application]** Theorem giả định unbiased estimator nhưng chapter dùng median và gọi nó unbiased/good mà không chứng minh unbiasedness (`docs/part8/ch47-search-and-measurement/README.md:59`).
- [ ] **[P1][Population definition]** Population được định nghĩa như subset size `N` dù evolutionary population có thể chứa duplicate candidates/individuals.
- [ ] **[P1][Search degeneration]** Một số seeded/default paths collapse về cùng candidate hoặc `Infinity`, nên search behavior thực tế hẹp hơn narrative.
- [ ] **[P1][Measurement status]** Default CPU tuning path có thể không thực hiện hardware measurement dù chapter đặt measurement ở trung tâm correctness/performance decision.

### Chapter 48 — Reproducibility

- [ ] **[P1][Proposition 48.4]** Claim mọi unrecorded mutation đều làm replay diverge là sai nếu mutation bị later recorded step overwrite/eliminate; theorem thiếu persistence assumption (`docs/part8/ch48-reproducibility/README.md:52`).
- [ ] **[P0][Stale cache semantics]** Claim stale cache chỉ “degrades to no tuning, not wrong results” sai khi cùng sketch name/parameters vẫn parse nhưng transform semantics đã đổi (`docs/part8/ch48-reproducibility/README.md:172`).
- [ ] **[P1][Trace surface]** Dynamic dispatch recorder có thể nhìn thấy inherited `Object` methods chứ không chỉ public `Schedule` API; trace surface rộng hơn mô tả.
- [ ] **[P1][Replay evidence]** Record/replay lab kiểm tra một case thành công nhưng được dùng để hỗ trợ soundness claim rộng cho toàn bộ primitive set.
- [ ] **[P0][Cache identity]** Workload/cache key không encode đầy đủ implementation semantics, target behavior và numerical mode; cache hit có thể replay schedule không còn tương đương với context tạo record.
