import { describe, it, expect } from 'vitest';
import { buildFunction, IRBuilder } from '../../../src/compiler/ir/graph/builder.js';
import { Operation } from '../../../src/compiler/ir/graph/operation.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { verifyFunction } from '../../../src/compiler/ir/graph/verifier.js';
import { printModule } from '../../../src/compiler/ir/graph/printer.js';
import { parseModule } from '../../../src/compiler/ir/graph/parser.js';
import { fileLocation, nameLocation, formatLocation, locationSites, locationNames, primarySite } from '../../../src/compiler/ir/location.js';
import { setDefaultLocationSource } from '../../../src/compiler/ir/loc_source.js';
import { CanonicalizePass } from '../../../src/compiler/passes/canonicalize/canonicalize.js';
import { DecompositionPass } from '../../../src/compiler/passes/decompose/decomposition_pass.js';
import { FusionPass } from '../../../src/compiler/passes/fusion/fusion_pass.js';
import { BackwardGraphBuilder } from '../../../src/compiler/ad/index.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { walk } from '../../../src/compiler/ir/ir_visitor.js';
import { CPUTarget } from '../../../src/backend/target.js';

const f32 = (shape) => new TensorType(shape, ScalarType.F32);

function opsOf(func, opName) {
  const found = [];
  for (const op of func.opsRecursive()) {
    if (op.opName === opName) found.push(op);
  }
  return found;
}

function lineOf(op) {
  const site = primarySite(op.loc);
  return site === null ? null : site.line;
}

function blocksWithSource(primFunc) {
  const blocks = [];
  walk(primFunc.body, (node) => {
    if (node.type === 'BlockNode' && node.sourceOp) blocks.push(node);
  });
  return blocks;
}

describe('IRBuilder location', () => {
  it('stamps the current location onto every op it inserts', () => {
    const loc = fileLocation('model.js', 11, 2);
    const func = buildFunction('f', [f32([4]), f32([4])], [f32([4])], (b, args) => {
      b.location = loc;
      const sum = b.add(args[0], args[1]);
      b.returnOp([sum.getResult(0)]);
    });
    expect(opsOf(func, 'add')[0].loc).toBe(loc);
  });

  it('leaves ops unlocated when no source and no location are set', () => {
    const func = buildFunction('f', [f32([4]), f32([4])], [f32([4])], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    expect(opsOf(func, 'add')[0].loc).toBeNull();
  });

  it('restores the previous location after withLocation, including on a throw', () => {
    const outer = fileLocation('model.js', 1, 0);
    const inner = fileLocation('model.js', 2, 0);
    const func = buildFunction('f', [f32([4]), f32([4])], [f32([4])], (b, args) => {
      b.location = outer;
      const sum = b.withLocation(inner, () => b.add(args[0], args[1]));
      expect(b.location).toBe(outer);
      expect(() => b.withLocation(inner, () => { throw new Error('rule failed'); })).toThrow('rule failed');
      expect(b.location).toBe(outer);
      const negated = b.neg(sum.getResult(0));
      b.returnOp([negated.getResult(0)]);
    });
    expect(opsOf(func, 'add')[0].loc).toBe(inner);
    expect(opsOf(func, 'neg')[0].loc).toBe(outer);
  });

  it('falls back to the default location source when the builder has none', () => {
    const loc = fileLocation('traced.js', 42, 3);
    const previous = setDefaultLocationSource(() => loc);
    try {
      const func = buildFunction('f', [f32([4]), f32([4])], [f32([4])], (b, args) => {
        b.returnOp([b.add(args[0], args[1]).getResult(0)]);
      });
      expect(opsOf(func, 'add')[0].loc).toBe(loc);
    } finally {
      setDefaultLocationSource(previous);
    }
  });
});

describe('Operation.clone', () => {
  it('carries the location to the clone', () => {
    const loc = fileLocation('model.js', 8, 1);
    const func = buildFunction('f', [f32([4]), f32([4])], [f32([4])], (b, args) => {
      b.location = loc;
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const original = opsOf(func, 'add')[0];
    expect(original.clone().loc).toBe(loc);
  });
});

describe('pattern rewriting', () => {
  it('gives the op a canonicalization builds the location of the matched root', () => {
    const inner = fileLocation('model.js', 4, 0);
    const outer = fileLocation('model.js', 5, 0);
    const func = buildFunction('f', [f32([2, 6])], [f32([12])], (b, args) => {
      const first = b.withLocation(inner, () => b.reshape(args[0], [3, 4]));
      const second = b.withLocation(outer, () => b.reshape(first.getResult(0), [12]));
      b.returnOp([second.getResult(0)]);
    });

    new CanonicalizePass().run(func);

    const returned = func.getReturnOp().getOperand(0).definingOp;
    expect(returned.opName).toBe('reshape');
    expect(returned.getOperand(0)).toBe(func.args[0]);
    expect(returned.loc).toBe(outer);
  });
});

describe('DecompositionPass', () => {
  it('locates every primitive it emits at the op it replaced', () => {
    const loc = fileLocation('model.js', 17, 4);
    const func = buildFunction('f', [f32([2, 5])], [f32([2, 5])], (b, args) => {
      const soft = b.withLocation(loc, () => b.softmax(args[0], 1));
      b.returnOp([soft.getResult(0)]);
    });

    new DecompositionPass(null).run(func);

    expect(opsOf(func, 'softmax').length).toBe(0);
    const emitted = [...func.opsRecursive()].filter(op => op.opName !== 'return');
    expect(emitted.length).toBeGreaterThan(1);
    for (const op of emitted) expect(op.loc).toBe(loc);
  });
});

describe('FusionPass', () => {
  it('fuses the locations of the ops it grouped', () => {
    const addLoc = fileLocation('model.js', 20, 2);
    const negLoc = fileLocation('model.js', 21, 2);
    const func = buildFunction('f', [f32([64, 64]), f32([64, 64])], [f32([64, 64])], (b, args) => {
      const sum = b.withLocation(addLoc, () => b.add(args[0], args[1]));
      const negated = b.withLocation(negLoc, () => b.neg(sum.getResult(0)));
      b.returnOp([negated.getResult(0)]);
    });

    new FusionPass({}).run(func);

    const fusion = opsOf(func, 'fusion')[0];
    expect(locationSites(fusion.loc)).toEqual([addLoc, negLoc]);
    expect(formatLocation(fusion.loc)).toBe('fused["model.js":20:2, "model.js":21:2]');
  });

  it('keeps the original location on each op inside the fusion body', () => {
    const addLoc = fileLocation('model.js', 30, 0);
    const negLoc = fileLocation('model.js', 31, 0);
    const func = buildFunction('f', [f32([64, 64]), f32([64, 64])], [f32([64, 64])], (b, args) => {
      const sum = b.withLocation(addLoc, () => b.add(args[0], args[1]));
      b.returnOp([b.withLocation(negLoc, () => b.neg(sum.getResult(0))).getResult(0)]);
    });

    new FusionPass({}).run(func);

    expect(opsOf(func, 'add')[0].loc).toBe(addLoc);
    expect(opsOf(func, 'neg')[0].loc).toBe(negLoc);
  });
});

describe('BackwardGraphBuilder', () => {
  it('marks each gradient op with the forward location it came from', () => {
    const mulLoc = fileLocation('model.js', 50, 6);
    const func = buildFunction('f', [f32([4]), f32([4])], [f32([4])], (b, args) => {
      const product = b.withLocation(mulLoc, () => b.mul(args[0], args[1]));
      b.returnOp([product.getResult(0)]);
    });

    const { backwardFunc } = new BackwardGraphBuilder().build(func);

    const located = [...backwardFunc.opsRecursive()].filter(op => op.loc !== null);
    expect(located.length).toBeGreaterThan(0);
    for (const op of located) {
      expect(locationNames(op.loc)).toContain('grad');
      expect(primarySite(op.loc)).toBe(mulLoc);
    }
  });
});

describe('lowerGraphToPrimFunc', () => {
  it('records the graph op location on the TIR block it produced', () => {
    const addLoc = fileLocation('model.js', 60, 1);
    const func = buildFunction('f', [f32([8]), f32([8])], [f32([8])], (b, args) => {
      b.returnOp([b.withLocation(addLoc, () => b.add(args[0], args[1])).getResult(0)]);
    });

    const blocks = blocksWithSource(lowerGraphToPrimFunc(func, CPUTarget()));
    const added = blocks.filter(block => block.sourceOp.name === 'add');
    expect(added.length).toBeGreaterThan(0);
    for (const block of added) expect(block.sourceOp.loc).toBe(addLoc);
  });
});

describe('printModule with locations', () => {
  const moduleWithLocations = () => {
    const module = new GraphModule('m');
    module.addFunction(buildFunction('f', [f32([4]), f32([4])], [f32([4])], (b, args) => {
      const sum = b.withLocation(nameLocation('forward', fileLocation('model.js', 70, 2)), () => b.add(args[0], args[1]));
      b.returnOp([b.withLocation(fileLocation('model.js', 71, 2), () => b.neg(sum.getResult(0))).getResult(0)]);
    }));
    return module;
  };

  it('omits locations unless asked', () => {
    expect(printModule(moduleWithLocations())).not.toContain('loc(');
  });

  it('prints one loc per located op', () => {
    const text = printModule(moduleWithLocations(), { locations: true });
    expect(text).toContain('loc("forward"("model.js":70:2))');
    expect(text).toContain('loc("model.js":71:2)');
  });

  it('round-trips locations through the parser', () => {
    const printed = printModule(moduleWithLocations(), { locations: true });
    const reparsed = parseModule(printed);
    expect(printModule(reparsed, { locations: true })).toBe(printed);

    const parsedFunc = [...reparsed][0];
    expect(lineOf(opsOf(parsedFunc, 'add')[0])).toBe(70);
    expect(locationNames(opsOf(parsedFunc, 'add')[0].loc)).toEqual(['forward']);
    expect(lineOf(opsOf(parsedFunc, 'neg')[0])).toBe(71);
  });

  it('round-trips a fused location produced by fusion', () => {
    const module = new GraphModule('m');
    module.addFunction(buildFunction('f', [f32([64, 64]), f32([64, 64])], [f32([64, 64])], (b, args) => {
      const sum = b.withLocation(fileLocation('model.js', 80, 0), () => b.add(args[0], args[1]));
      b.returnOp([b.withLocation(fileLocation('model.js', 81, 0), () => b.neg(sum.getResult(0))).getResult(0)]);
    }));
    new FusionPass({}).run([...module][0]);

    const printed = printModule(module, { locations: true });
    expect(printed).toContain('loc(fused["model.js":80:0, "model.js":81:0])');
    expect(printModule(parseModule(printed), { locations: true })).toBe(printed);
  });
});

describe('VerificationError', () => {
  it('names the location of the offending op', () => {
    const loc = fileLocation('model.js', 90, 5);
    const func = buildFunction('f', [f32([4]), f32([4])], [f32([4])], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const builder = new IRBuilder(func);
    const broken = new Operation('add', [func.args[0]], [f32([4])]);
    broken.loc = loc;
    builder.setInsertionPoint(func.getReturnOp());
    builder._insert(broken);

    const messages = verifyFunction(func).map(String);
    expect(messages.some(text => text.includes('expects 2 operands') && text.includes('at "model.js":90:5'))).toBe(true);
  });

  it('leaves the message unadorned when the op has no location', () => {
    const func = buildFunction('f', [f32([4]), f32([4])], [f32([4])], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const builder = new IRBuilder(func);
    builder.setInsertionPoint(func.getReturnOp());
    builder._insert(new Operation('add', [func.args[0]], [f32([4])]));

    const messages = verifyFunction(func).map(String);
    expect(messages.some(text => text.includes('expects 2 operands'))).toBe(true);
    expect(messages.every(text => !text.includes(' at "'))).toBe(true);
  });
});
