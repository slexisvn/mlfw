import { parseModule, printModule } from '../../../../dist/index.node.js';

const GOOD = `module @m {
  func @m(%0: tensor<2x2xf32>, %1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    %2 = add(%0, %1) : tensor<2x2xf32>
    %3 = mul(%2, %0) : tensor<2x2xf32>
    return(%3)
  }
}`;

console.log('the valid module round-trips:', printModule(parseModule(GOOD)) === GOOD);

const CYCLE = `module @m {
  func @m(%0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    %1 = add(%0, %2) : tensor<2x2xf32>
    %2 = add(%0, %1) : tensor<2x2xf32>
    return(%2)
  }
}`;

const cases = [
  ['a value nobody defines', GOOD.replace('add(%0, %1)', 'add(%0, %9)')],
  ['one name, two definitions', GOOD.replace('%3 = mul', '%2 = mul').replace('return(%3)', 'return(%2)')],
  ['a dependency cycle', CYCLE],
  ['an operation nobody registered', GOOD.replace('add(', 'frobnicate(')],
  ['the wrong number of operands', GOOD.replace('add(%0, %1)', 'add(%0)')],
  ['a result type that does not follow', GOOD.replace('%2 = add(%0, %1) : tensor<2x2xf32>', '%2 = add(%0, %1) : tensor<4x4xf32>')],
  ['a return that does not match the signature', GOOD.replace('return(%3)', 'return(%3, %2)')],
];

for (const [label, text] of cases) {
  console.log(`\n${label}`);
  let module = null;
  try {
    module = parseModule(text);
    console.log('  parser        : accepted');
  } catch (e) {
    console.log(`  parser        : rejected -- ${e.message}`);
    continue;
  }
  const errors = module.verify();
  console.log(`  module.verify(): ${errors.length === 0 ? 'no complaints' : errors.join(' | ')}`);
}
