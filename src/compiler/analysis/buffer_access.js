import { walk } from '../ir/ir_visitor.js';
import { toLinearForm, coverRangeOfForm, composeForm, LinearForm } from './iter_map.js';

export const AccessKind = Object.freeze({ READ: 'read', WRITE: 'write' });

export class BufferAccess {
  constructor(node, buffer, kind, position, conditional, regions, selfReferential, forms, iterSpace, block, indexBuffers) {
    this.node = node;
    this.buffer = buffer;
    this.kind = kind;
    this.position = position;
    this.conditional = conditional;
    this.regions = regions;
    this.selfReferential = selfReferential;
    this.forms = forms;
    this.iterSpace = iterSpace;
    this.block = block;
    this.indexBuffers = indexBuffers;
  }

  get isWrite() { return this.kind === AccessKind.WRITE; }
  get isRead() { return this.kind === AccessKind.READ; }
}

export class BlockAccessInfo {
  constructor(block, iterSpace, bindings, affineBinding, typedIterVars, directBodyVars) {
    this.block = block;
    this.iterSpace = iterSpace;
    this.bindings = bindings;
    this.affineBinding = affineBinding;
    this.typedIterVars = typedIterVars;
    this.directBodyVars = directBodyVars;
    this.accesses = [];
  }

  iterKindsOfLoopVar(name) {
    if (!this.affineBinding || !this.typedIterVars || this.directBodyVars.has(name)) return null;
    const kinds = new Set();
    for (const binding of this.bindings) {
      if (binding.form && binding.form.terms.has(name)) kinds.add(binding.kind);
    }
    return kinds.size === 0 ? null : kinds;
  }
}

function directBodyVarsOf(block, own) {
  const vars = new Set();
  const visit = (n) => { if (n.type === 'VariableNode' && !own.has(n.name)) vars.add(n.name); };
  walk(block.body, visit);
  if (block.initBody) walk(block.initBody, visit);
  return vars;
}

function constValue(node) {
  return node && node.type === 'IntImmNode' && Number.isInteger(node.value) ? node.value : null;
}

export function loadedBuffers(exprs) {
  const found = new Set();
  for (const expr of exprs) {
    walk(expr, (n) => { if (n.type === 'BufferLoadNode' && n.buffer) found.add(n.buffer); });
  }
  return found;
}

export function collectBufferAccesses(root, env = null) {
  const order = [];
  const byBuffer = new Map();
  const byBlock = new Map();
  const indexLoaded = new Set();
  const loopRanges = new Map();
  const varForms = new Map();
  const scopes = [];
  const loopStack = [];
  const blockStack = [];
  let conditionalDepth = 0;
  let position = 0;

  if (env) {
    for (const [name, range] of env.loopRanges) loopRanges.set(name, range);
    for (const [name, form] of env.varForms) varForms.set(name, form);
    for (const level of env.iterSpace) loopStack.push(level);
  }

  const bind = (name, range, form) => {
    const frame = scopes[scopes.length - 1];
    frame.push([name, loopRanges.get(name), varForms.get(name)]);
    if (range === null) loopRanges.delete(name);
    else loopRanges.set(name, range);
    if (form === null) varForms.delete(name);
    else varForms.set(name, form);
  };

  const openScope = () => scopes.push([]);
  const closeScope = () => {
    for (const [name, prevRange, prevForm] of scopes.pop()) {
      if (prevRange === undefined) loopRanges.delete(name);
      else loopRanges.set(name, prevRange);
      if (prevForm === undefined) varForms.delete(name);
      else varForms.set(name, prevForm);
    }
  };

  const record = (node, buffer, kind, indices, valueLoads) => {
    const regions = [];
    const forms = [];
    for (const idx of indices) {
      const raw = idx ? toLinearForm(idx) : null;
      regions.push(coverRangeOfForm(raw, loopRanges));
      forms.push(composeForm(raw, varForms));
    }
    const indexBuffers = loadedBuffers(indices);
    for (const b of indexBuffers) indexLoaded.add(b);
    const selfReferential = indexBuffers.has(buffer) || (valueLoads !== null && valueLoads.has(buffer));
    const block = blockStack.length > 0 ? blockStack[blockStack.length - 1] : null;
    const access = new BufferAccess(
      node, buffer, kind, position++, conditionalDepth > 0, regions, selfReferential,
      forms, loopStack.slice(), block, indexBuffers
    );
    order.push(access);
    let list = byBuffer.get(buffer);
    if (!list) { list = []; byBuffer.set(buffer, list); }
    list.push(access);
    if (block) byBlock.get(block).accesses.push(access);
    return access;
  };

  walk(root, {
    pre(node) {
      switch (node.type) {
        case 'ForNode': {
          openScope();
          const min = constValue(node.min);
          const extent = constValue(node.extent);
          const known = min !== null && extent !== null;
          if (node.loopVar) {
            bind(node.loopVar.name, known ? [min, extent] : null, known ? LinearForm.variable(node.loopVar.name) : null);
            if (known) loopStack.push({ name: node.loopVar.name, min, extent, node });
            else loopStack.push(null);
          } else {
            loopStack.push(null);
          }
          break;
        }
        case 'BlockNode': {
          openScope();
          blockStack.push(node);
          const bindings = [];
          let affine = true;
          let typed = true;
          for (const iv of node.iterVars || []) {
            if (!iv || !iv.iterVar) continue;
            const raw = iv.binding ? toLinearForm(iv.binding) : null;
            const composed = composeForm(raw, varForms);
            if (composed === null) affine = false;
            if (iv.kind === undefined) typed = false;
            bindings.push({ name: iv.iterVar.name, form: composed, kind: iv.kind });
            bind(iv.iterVar.name, coverRangeOfForm(raw, loopRanges), composed);
          }
          byBlock.set(node, new BlockAccessInfo(
            node, loopStack.slice(), bindings, affine, typed,
            directBodyVarsOf(node, new Set(bindings.map((b) => b.name)))
          ));
          break;
        }
        case 'IfThenElseNode':
        case 'WhileNode':
          conditionalDepth++;
          break;
        case 'BufferStoreNode':
          record(node, node.buffer, AccessKind.WRITE, node.indices, loadedBuffers([node.value]));
          break;
        case 'BufferLoadNode':
          record(node, node.buffer, AccessKind.READ, node.indices, null);
          break;
      }
    },
    post(node) {
      switch (node.type) {
        case 'ForNode':
          loopStack.pop();
          closeScope();
          break;
        case 'BlockNode':
          blockStack.pop();
          closeScope();
          break;
        case 'IfThenElseNode':
        case 'WhileNode':
          conditionalDepth--;
          break;
      }
    },
  });

  return { order, byBuffer, byBlock, indexLoaded };
}

export function coversWholeBuffer(access) {
  const shape = access.buffer.shape;
  if (access.regions.length !== shape.length) return false;
  for (let d = 0; d < shape.length; d++) {
    const extent = shape[d];
    if (typeof extent !== 'number') return false;
    const range = access.regions[d];
    if (!range || range[0] !== 0 || range[1] !== extent) return false;
  }
  return true;
}
