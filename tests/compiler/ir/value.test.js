import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Value, UseLink, BlockArgument } from '../../../src/compiler/ir/graph/value.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';

const f32_2x3 = new TensorType([2, 3], ScalarType.F32);

describe('Value', () => {
  it('creates with type', () => {
    const v = new Value(f32_2x3);
    assert.ok(v.type.equals(f32_2x3));
    assert.equal(v.definingOp, null);
    assert.equal(v.hasUses, false);
    assert.equal(v.useCount, 0);
  });

  it('unique ids', () => {
    const a = new Value(f32_2x3);
    const b = new Value(f32_2x3);
    assert.notEqual(a.id, b.id);
  });
});

describe('UseLink chain', () => {
  it('add and remove uses', () => {
    const v = new Value(f32_2x3);
    const fakeOp1 = { operands: [v] };
    const fakeOp2 = { operands: [v] };
    const link1 = new UseLink(fakeOp1, 0);
    const link2 = new UseLink(fakeOp2, 0);
    v.addUse(link1);
    v.addUse(link2);
    assert.equal(v.useCount, 2);
    assert.ok(v.hasUses);

    v.removeUse(link1);
    assert.equal(v.useCount, 1);

    v.removeUse(link2);
    assert.equal(v.useCount, 0);
    assert.ok(!v.hasUses);
  });

  it('iterate uses', () => {
    const v = new Value(f32_2x3);
    const fakeOps = [];
    const links = [];
    for (let i = 0; i < 5; i++) {
      const op = { operands: [v] };
      fakeOps.push(op);
      const link = new UseLink(op, 0);
      links.push(link);
      v.addUse(link);
    }
    const users = v.getUsers();
    assert.equal(users.length, 5);
    for (let i = 0; i < 5; i++) {
      assert.equal(users[i], fakeOps[i]);
    }
  });

  it('remove from middle', () => {
    const v = new Value(f32_2x3);
    const ops = [{ operands: [v] }, { operands: [v] }, { operands: [v] }];
    const links = ops.map((op, i) => {
      const link = new UseLink(op, 0);
      v.addUse(link);
      return link;
    });
    v.removeUse(links[1]);
    assert.equal(v.useCount, 2);
    const users = v.getUsers();
    assert.equal(users[0], ops[0]);
    assert.equal(users[1], ops[2]);
  });
});

describe('replaceAllUsesWith', () => {
  it('transfers all uses', () => {
    const old = new Value(f32_2x3);
    const replacement = new Value(f32_2x3);

    const op1 = { operands: [old, null] };
    const op2 = { operands: [null, old] };
    const link1 = new UseLink(op1, 0);
    const link2 = new UseLink(op2, 1);
    old.addUse(link1);
    old.addUse(link2);

    old.replaceAllUsesWith(replacement);

    assert.equal(old.useCount, 0);
    assert.equal(old.hasUses, false);
    assert.equal(replacement.useCount, 2);
    assert.equal(op1.operands[0], replacement);
    assert.equal(op2.operands[1], replacement);
  });

  it('self-replacement is no-op', () => {
    const v = new Value(f32_2x3);
    const op = { operands: [v] };
    const link = new UseLink(op, 0);
    v.addUse(link);
    v.replaceAllUsesWith(v);
    assert.equal(v.useCount, 1);
    assert.equal(op.operands[0], v);
  });

  it('appends to existing uses', () => {
    const old = new Value(f32_2x3);
    const replacement = new Value(f32_2x3);

    const existingOp = { operands: [replacement] };
    const existingLink = new UseLink(existingOp, 0);
    replacement.addUse(existingLink);

    const op = { operands: [old] };
    const link = new UseLink(op, 0);
    old.addUse(link);

    old.replaceAllUsesWith(replacement);
    assert.equal(replacement.useCount, 2);
    assert.equal(old.useCount, 0);
  });
});

describe('BlockArgument', () => {
  it('is a Value subclass', () => {
    const fakeBlock = {};
    const arg = new BlockArgument(f32_2x3, fakeBlock, 0);
    assert.ok(arg instanceof Value);
    assert.ok(arg.isBlockArgument());
    assert.equal(arg.ownerBlock, fakeBlock);
    assert.equal(arg.argIndex, 0);
  });
});
