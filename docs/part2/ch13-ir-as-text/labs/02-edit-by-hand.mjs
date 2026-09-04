import {
  tensor, Linear, ReLU, Sequential, trace, printModule, parseModule, IRParseError, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const original = printModule(await trace((t) => model.forward(t), [tensor([[0.5, -1.5], [1.0, 2.0]])]));

console.log('=== as traced ===');
console.log(original);

const relu = original.replace(
  '%10 = tera.maximum %7, %9 : tensor<2x8xf32>',
  '%10 = "tera.tanh"(%7) : (tensor<2x8xf32>) -> tensor<2x8xf32>',
);
console.log('\n=== after changing the activation, in a text editor ===');
const edited = parseModule(relu);
console.log(printModule(edited));

const func = [...edited.functions()][0];
console.log(`operations now: ${func.opsArray().map((op) => op.opName).join(', ')}`);
console.log(`the broadcast constant is still there, with ${func.findOp((op) => op.opName === 'broadcast_in_dim').getResult(0).useCount} users`);

console.log('\n=== a module written by hand, never traced ===');
const byHand = `module @handwritten {
  func.func @handwritten(%0: tensor<3x4xf32>, %1: tensor<4x2xf32>) -> (tensor<3x2xf32>) {
    %2 = tera.dot %0, %1 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>, rhs_batch = array<i64>, rhs_contracting = array<i64: 0>} : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf32>
    %3 = "tera.tanh"(%2) : (tensor<3x2xf32>) -> tensor<3x2xf32>
    return %3 : tensor<3x2xf32>
  }
}`;
console.log(byHand);
const hand = parseModule(byHand);
const handFunc = [...hand.functions()][0];
console.log(`parsed: ${handFunc.numOps()} operations, ${handFunc.inputTypes.length} inputs, ${handFunc.outputTypes.length} output`);
console.log(`the dot contracts lhs dim ${handFunc.findOp((op) => op.opName === 'dot').getAttr('lhs_contracting')} against rhs dim ${handFunc.findOp((op) => op.opName === 'dot').getAttr('rhs_contracting')}`);
console.log(`round-trips: ${printModule(hand) === byHand}`);

console.log('\n=== an edit that does not typecheck as text ===');
try {
  parseModule(original.replace('%10 = tera.maximum %7, %9', '%10 = tera.maximum %7, %99'));
} catch (e) {
  console.log(`${e instanceof IRParseError ? 'IRParseError' : 'Error'}: ${e.message}`);
}

console.log('\n=== an op the dialect does not define, spelt in its own form ===');
try {
  parseModule(original.replace('%10 = tera.maximum %7, %9 : tensor<2x8xf32>', '%10 = tera.tanh %7 : tensor<2x8xf32>'));
} catch (e) {
  console.log(`${e instanceof IRParseError ? 'IRParseError' : 'Error'}: ${e.message}`);
}
