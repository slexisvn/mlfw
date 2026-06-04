import { ForKind } from '../ir/tensor/nodes.js';

export class ScheduleValidator {
  static validate(primFunc) {
    const errors = [];
    const ctx = {
      boundVars: new Set(),
      threadBindings: new Map(),
      errors
    };

    for (const param of primFunc.params) {
      ctx.boundVars.add(param.name);
    }

    ScheduleValidator._visitNode(primFunc.body, ctx);
    return errors;
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
        ScheduleValidator._visitNode(node.thenBody, ctx);
        if (node.elseBody) ScheduleValidator._visitNode(node.elseBody, ctx);
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

    if (node.kind === ForKind.PARALLEL || node.kind === ForKind.THREAD_BINDING) {
      ScheduleValidator._checkNoNestedParallel(node.body, varName, ctx);
    }

    ctx.boundVars.add(varName);
    ScheduleValidator._visitNode(node.body, ctx);
    ctx.boundVars.delete(varName);

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
    for (const r of node.iterVars) {
      if (r.iterVar) ctx.boundVars.add(r.iterVar.name);
    }
    ScheduleValidator._visitNode(node.body, ctx);
    if (node.initBody) ScheduleValidator._visitNode(node.initBody, ctx);
    for (const r of node.iterVars) {
      if (r.iterVar) ctx.boundVars.delete(r.iterVar.name);
    }
  }

  static _validateBufferAccess(node, ctx) {
    if (!node.buffer) {
      ctx.errors.push('BufferStore with null buffer');
      return;
    }
    if (node.indices && node.buffer.shape) {
      if (node.indices.length !== node.buffer.shape.length) {
        ctx.errors.push(
          `Buffer '${node.buffer.name}' rank mismatch: ` +
          `${node.indices.length} indices for rank-${node.buffer.shape.length} buffer`
        );
      }
    }
  }
}
