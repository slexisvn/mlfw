import { ForKind } from '../ir/tensor/nodes.js';
import { reductionLoopVars, classifyBufferIndex } from './legality.js';
import { Analyzer } from '../analysis/analyzer.js';
import { irBound } from '../analysis/ir_arith.js';

export class ScheduleValidator {
  static validate(primFunc) {
    const errors = [];
    const ctx = {
      boundVars: new Set(),
      threadBindings: new Map(),
      parallelExtents: new Map(),
      parLoops: [],
      innermostLoopVar: null,
      loopStack: [],
      analyzer: new Analyzer(),
      condDepth: 0,
      errors
    };

    for (const param of primFunc.params) {
      ctx.boundVars.add(param.name);
    }

    ScheduleValidator._visitNode(primFunc.body, ctx);
    ScheduleValidator._checkPartitionConsistency(ctx);
    return errors;
  }

  static _checkPartitionConsistency(ctx) {
    if (ctx.parallelExtents.size <= 1) return;
    const entries = [...ctx.parallelExtents.entries()]
      .map(([extent, varName]) => `'${varName}'(extent ${extent})`);
    ctx.errors.push(
      `Ambiguous parallel partition: ${ctx.parallelExtents.size} distinct parallel extents ` +
      `[${entries.join(', ')}] — runtime partitions a single axis, mismatched-extent ` +
      `parallel loops corrupt buffers`
    );
  }

  static _visitNode(node, ctx) {
    if (!node) return;

    switch (node.type) {
      case 'ForNode':
        ScheduleValidator._visitFor(node, ctx);
        break;
      case 'BlockNode':
        ScheduleValidator._visitBlock(node, ctx);
        break;
      case 'SeqNode':
        for (const s of node.stmts) ScheduleValidator._visitNode(s, ctx);
        break;
      case 'IfThenElseNode':
        ScheduleValidator._visitExpr(node.condition, ctx);
        ctx.condDepth++;
        ScheduleValidator._visitNode(node.thenBody, ctx);
        if (node.elseBody) ScheduleValidator._visitNode(node.elseBody, ctx);
        ctx.condDepth--;
        break;
      case 'AllocateNode':
        ScheduleValidator._visitNode(node.body, ctx);
        break;
      case 'LetStmtNode':
        ctx.boundVars.add(node.variable.name);
        ScheduleValidator._visitNode(node.body, ctx);
        ctx.boundVars.delete(node.variable.name);
        break;
      case 'BufferStoreNode':
        ScheduleValidator._validateBufferAccess(node, ctx);
        break;
      case 'EvaluateNode':
        break;
      default:
        break;
    }
  }

  static _visitFor(node, ctx) {
    const varName = node.loopVar.name;

    if (ctx.boundVars.has(varName)) {
      ctx.errors.push(`Duplicate loop variable: ${varName}`);
    }

    if (node.kind === ForKind.THREAD_BINDING) {
      if (!node.threadTag) {
        ctx.errors.push(`Thread-bound loop '${varName}' missing threadTag`);
      } else {
        if (ctx.threadBindings.has(node.threadTag)) {
          ctx.errors.push(
            `Duplicate thread binding '${node.threadTag}': ` +
            `already bound to '${ctx.threadBindings.get(node.threadTag)}'`
          );
        }
        ctx.threadBindings.set(node.threadTag, varName);
      }
    }

    if (node.kind === ForKind.VECTORIZED) {
      const extent = node.extent;
      if (extent.type === 'IntImmNode') {
        if (extent.value <= 0) {
          ctx.errors.push(`Vectorized loop '${varName}' has non-positive extent ${extent.value}`);
        }
      }
    }

    if (node.kind === ForKind.PARALLEL) {
      const extent = node.extent;
      if (extent && extent.type === 'IntImmNode' && !ctx.parallelExtents.has(extent.value)) {
        ctx.parallelExtents.set(extent.value, varName);
      }
    }

    if (node.kind === ForKind.PARALLEL || node.kind === ForKind.THREAD_BINDING) {
      ScheduleValidator._checkNoNestedParallel(node.body, varName, ctx);
    }

    const tracksReduction = node.kind === ForKind.PARALLEL || node.kind === ForKind.VECTORIZED;
    if (tracksReduction) ctx.parLoops.push({ varName, kind: node.kind });

    ctx.boundVars.add(varName);
    const prevBound = ctx.analyzer.getVarBound(varName);
    if (node.extent && node.extent.type === 'IntImmNode' && node.extent.value > 0) {
      ctx.analyzer.bind(varName, 0, node.extent.value - 1);
    } else {
      ctx.analyzer.setVarBound(varName, null);
    }
    const prevInnermost = ctx.innermostLoopVar;
    ctx.innermostLoopVar = varName;
    ctx.loopStack.push(varName);
    ScheduleValidator._visitNode(node.body, ctx);
    ctx.loopStack.pop();
    ctx.innermostLoopVar = prevInnermost;
    ctx.boundVars.delete(varName);
    ctx.analyzer.setVarBound(varName, prevBound);

    if (tracksReduction) ctx.parLoops.pop();

    if (node.kind === ForKind.THREAD_BINDING && node.threadTag) {
      ctx.threadBindings.delete(node.threadTag);
    }
  }

  static _checkNoNestedParallel(node, outerVarName, ctx) {
    if (!node) return;
    if (node.type === 'ForNode') {
      if (node.kind === ForKind.PARALLEL) {
        ctx.errors.push(
          `Parallel loop '${node.loopVar.name}' nested inside parallel/thread-bound loop '${outerVarName}'`
        );
      }
    }
  }

  static _visitBlock(node, ctx) {
    const savedBounds = [];
    for (const r of node.iterVars) {
      if (r.iterVar) {
        ctx.boundVars.add(r.iterVar.name);
        savedBounds.push([r.iterVar.name, ctx.analyzer.getVarBound(r.iterVar.name)]);
        ctx.analyzer.setVarBound(r.iterVar.name, r.binding ? irBound(ctx.analyzer, r.binding) : null);
      }
    }
    const reductionVars = reductionLoopVars(node);
    if (node.initBody && ctx.innermostLoopVar) {
      let reductionEnclosing = 0;
      for (const v of ctx.loopStack) {
        if (reductionVars.has(v)) reductionEnclosing++;
      }
      if (!reductionVars.has(ctx.innermostLoopVar) || reductionEnclosing > 1) {
        ctx.errors.push(
          `Reduction block '${node.name}' violates the init contract: codegen zeroes the accumulator ` +
          `at the single innermost loop, so an init-bearing block needs exactly one enclosing reduction ` +
          `loop and it must be innermost (innermost '${ctx.innermostLoopVar}', ${reductionEnclosing} enclosing reduction loops)`
        );
      }
    }
    if (ctx.parLoops.length > 0) {
      for (const pl of ctx.parLoops) {
        if (reductionVars.has(pl.varName)) {
          const kind = pl.kind === ForKind.VECTORIZED ? 'Vectorized' : 'Parallel';
          ctx.errors.push(
            `${kind} loop '${pl.varName}' carries a reduction in block '${node.name}': ` +
            `the loop variable is read but never written, so parallel iterations race on the accumulator`
          );
        }
      }
    }
    ScheduleValidator._visitNode(node.body, ctx);
    if (node.initBody) ScheduleValidator._visitNode(node.initBody, ctx);
    for (const r of node.iterVars) {
      if (r.iterVar) ctx.boundVars.delete(r.iterVar.name);
    }
    for (const [name, b] of savedBounds) ctx.analyzer.setVarBound(name, b);
  }

  static _validateBufferAccess(node, ctx) {
    if (!node.buffer) {
      ctx.errors.push('BufferStore with null buffer');
    } else {
      ScheduleValidator._checkRank(node, ctx);
      ScheduleValidator._checkBounds(node, ctx);
    }
    if (node.indices) {
      for (const idx of node.indices) ScheduleValidator._visitExpr(idx, ctx);
    }
    if (node.value) ScheduleValidator._visitExpr(node.value, ctx);
  }

  static _checkRank(node, ctx) {
    if (node.indices && node.buffer.shape) {
      if (node.indices.length !== node.buffer.shape.length) {
        ctx.errors.push(
          `Buffer '${node.buffer.name}' rank mismatch: ` +
          `${node.indices.length} indices for rank-${node.buffer.shape.length} buffer`
        );
      }
    }
  }

  static _checkBounds(node, ctx) {
    if (ctx.condDepth > 0) return;
    if (!node.buffer || !node.buffer.shape || !node.indices) return;
    if (node.indices.length !== node.buffer.shape.length) return;
    for (let i = 0; i < node.indices.length; i++) {
      const dim = node.buffer.shape[i];
      if (classifyBufferIndex(ctx.analyzer, node.indices[i], dim) === 'oob') {
        ctx.errors.push(
          `Buffer '${node.buffer.name}' access is out of bounds on axis ${i}: ` +
          `index is provably outside [0, ${dim - 1}]`
        );
      }
    }
  }

  static _visitExpr(node, ctx) {
    if (!node || typeof node !== 'object') return;
    switch (node.type) {
      case 'BufferLoadNode':
        if (!node.buffer) {
          ctx.errors.push('BufferLoad with null buffer');
        } else {
          ScheduleValidator._checkRank(node, ctx);
          ScheduleValidator._checkBounds(node, ctx);
        }
        if (node.indices) {
          for (const idx of node.indices) ScheduleValidator._visitExpr(idx, ctx);
        }
        break;
      case 'BufferStoreNode':
        ScheduleValidator._validateBufferAccess(node, ctx);
        break;
      case 'MathOpNode':
      case 'CompareNode':
        ScheduleValidator._visitExpr(node.a, ctx);
        ScheduleValidator._visitExpr(node.b, ctx);
        break;
      case 'CastNode':
        ScheduleValidator._visitExpr(node.expr, ctx);
        break;
      case 'CallExternNode':
        for (const a of node.args) ScheduleValidator._visitExpr(a, ctx);
        break;
      case 'IfThenElseNode':
        ScheduleValidator._visitExpr(node.condition, ctx);
        ctx.condDepth++;
        ScheduleValidator._visitNode(node.thenBody, ctx);
        if (node.elseBody) ScheduleValidator._visitNode(node.elseBody, ctx);
        ctx.condDepth--;
        break;
    }
  }
}
