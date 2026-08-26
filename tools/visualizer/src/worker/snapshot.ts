import { IRPrinter, formatAttrValue } from 'mlfw/compiler/ir/graph/printer.js';
import { typeToString } from 'mlfw/compiler/ir/graph/types.js';
import { printLIR, LIRPrinter } from 'mlfw/compiler/ir/lir/printer.js';
import { opOfBlockName } from 'mlfw/compiler/ir/tensor/block_name.js';
import { ForKind, IterVarKind } from 'mlfw/compiler/ir/tensor/nodes.js';
import { walk } from 'mlfw/compiler/ir/ir_visitor.js';
import { functionCost, sumCosts } from './cost.js';
import type { Block } from 'mlfw/compiler/ir/graph/block.js';
import type { GraphFunction } from 'mlfw/compiler/ir/graph/function.js';
import type { GraphModule } from 'mlfw/compiler/ir/graph/module.js';
import type { Value } from 'mlfw/compiler/ir/graph/value.js';
import type { Cost } from './cost.js';
import type { Dag, DagNode, DagValue, IRLevelName, NestKind, NestNode, Snapshot } from '../protocol.js';

const ATTR_TEXT_LIMIT = 96;
const DENSE_ELEMENT_LIMIT = 8;
const RETURN_OP = 'return';

const EMPTY: Snapshot = { text: '', ops: 0, bytes: 0, flops: 0, dags: [], nests: [] };

function attrText(value: unknown): string {
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    const view = value as unknown as { length: number; constructor: { name: string } };
    if (view.length > DENSE_ELEMENT_LIMIT) {
      return `dense<${view.constructor.name}>[${view.length} elements]`;
    }
  }
  const text = formatAttrValue(value as never);
  return text.length > ATTR_TEXT_LIMIT ? `${text.slice(0, ATTR_TEXT_LIMIT)}…` : text;
}

function dagForFunction(func: GraphFunction, names: ReadonlyMap<Value, string>): Dag {
  const args: DagValue[] = [];
  const values: DagValue[] = [];
  const nodes: DagNode[] = [];
  const returns: number[] = [];

  const describe = (value: Value): DagValue => ({
    id: value.id,
    name: names.get(value) ?? '%?',
    type: typeToString(value.type),
    producer: value.definingOp ? value.definingOp.id : null,
  });

  const visitBlock = (block: Block, into: DagNode[]): number[] => {
    const blockArgs: number[] = [];
    for (const arg of block.arguments) {
      values.push(describe(arg));
      blockArgs.push(arg.id);
    }
    for (const op of block) {
      if (op.opName === RETURN_OP) {
        for (const operand of op.operands) returns.push(operand.id);
        continue;
      }
      for (const result of op.results) values.push(describe(result));
      const regions: DagNode[][] = [];
      const regionArgs: number[][] = [];
      for (const region of op.regions) {
        const inner: DagNode[] = [];
        const args: number[] = [];
        for (const inner_block of region) args.push(...visitBlock(inner_block, inner));
        regions.push(inner);
        regionArgs.push(args);
      }
      into.push({
        id: op.id,
        opName: op.opName,
        operands: op.operands.map(v => v.id),
        results: op.results.map(v => v.id),
        resultTypes: op.results.map(v => typeToString(v.type)),
        attrs: [...op.attributes].map(([key, value]) => [key, attrText(value)] as [string, string]),
        regions,
        regionArgs,
      });
    }
    return blockArgs;
  };

  for (const arg of func.args) args.push(describe(arg));
  for (const block of func.body) visitBlock(block, nodes);

  return { func: func.name, args, values, nodes, returns };
}

function dagOf(func: GraphFunction): Dag {
  const printer = new IRPrinter();
  printer.printFunction(func, []);
  return dagForFunction(func, printer.valueNames);
}

function graphSnapshot(module: GraphModule): Snapshot {
  const text = new IRPrinter().printModule(module);
  const dags: Dag[] = [];
  const costs: Cost[] = [];
  let ops = 0;

  for (const func of module) {
    ops += func.numOps();
    dags.push(dagOf(func));
    costs.push(functionCost(func));
  }

  const cost = sumCosts(costs);
  return { text, ops, bytes: cost.bytes, flops: cost.flops, dags, nests: [] };
}

function functionSnapshot(func: GraphFunction): Snapshot {
  const printer = new IRPrinter();
  const lines: string[] = [];
  printer.printFunction(func, lines);
  const cost = functionCost(func);
  return {
    text: lines.join('\n'),
    ops: func.numOps(),
    bytes: cost.bytes,
    flops: cost.flops,
    dags: [dagForFunction(func, printer.valueNames)],
    nests: [],
  };
}

function countNodes(root: object): number {
  let total = 0;
  walk(root as never, () => { total++; });
  return total;
}

function expr(node: unknown): string {
  if (!node || typeof node !== 'object') return String(node ?? '');
  return new LIRPrinter().print(node as never).replace(/\s+/g, ' ').trim();
}

type TirLike = Record<string, unknown> & { type: string };

const STORE_LABEL_LIMIT = 88;

function store(target: string, value: string, path: string): NestNode[] {
  const full = value === '' ? target : `${target} = ${value}`;
  const label = full.length > STORE_LABEL_LIMIT ? `${full.slice(0, STORE_LABEL_LIMIT)}…` : full;
  return [{ id: `store:${path}`, kind: 'store', label, detail: label === full ? '' : full, op: null, opId: null, children: [] }];
}

function nestFor(node: unknown, path: string): NestNode[] {
  if (!node || typeof node !== 'object') return [];
  const n = node as TirLike;

  const wrap = (
    kind: NestKind, label: string, detail: string, children: NestNode[] = [],
    id?: string, op: string | null = null, opId: number | null = null,
  ): NestNode[] => [{ id: id ?? `${kind}:${path}`, kind, label, detail, op, opId, children }];

  switch (n.type) {
    case 'SeqNode':
      return (n.stmts as unknown[]).flatMap((child, i) => nestFor(child, `${path}.${i}`));

    case 'ForNode': {
      const loopVar = n.loopVar as { name: string };
      const kind = String(n.kind);
      const tag = n.threadTag ? ` @${String(n.threadTag)}` : kind === ForKind.SERIAL ? '' : ` ${kind}`;
      return wrap('for', `for ${loopVar.name} < ${expr(n.extent)}${tag}`, kind,
        nestFor(n.body, `${path}.b`), `for:${loopVar.name}`);
    }

    case 'BlockNode': {
      const name = String(n.name);
      const source = n.sourceOp as { name: string; id: number } | undefined;
      const iterVars = (n.iterVars as { kind: string }[]) ?? [];
      const reduce = iterVars.filter(iv => iv.kind !== IterVarKind.DATA_PAR).length;
      const detail = `${iterVars.length} iter var${iterVars.length === 1 ? '' : 's'}`
        + (reduce > 0 ? `, ${reduce} reduction` : '');
      const children = [...nestFor(n.initBody, `${path}.i`), ...nestFor(n.body, `${path}.b`)];
      return wrap('block', name, detail, children, `block:${name}`,
        source ? source.name : opOfBlockName(name), source ? source.id : null);
    }

    case 'AllocateNode': {
      const buffer = n.buffer as { name?: string };
      return wrap('alloc', `alloc ${buffer.name ?? '?'} @${String(n.scope)}`, '', nestFor(n.body, `${path}.b`));
    }

    case 'LetStmtNode': {
      const variable = n.variable as { name: string };
      return wrap('let', `let ${variable.name} = ${expr(n.value)}`, '', nestFor(n.body, `${path}.b`));
    }

    case 'IfThenElseNode':
      return wrap('if', `if ${expr(n.condition)}`, '',
        [...nestFor(n.thenBody, `${path}.t`), ...nestFor(n.elseBody, `${path}.e`)]);

    case 'WhileNode':
      return wrap('while', 'while', '', nestFor(n.loopBody, `${path}.b`));

    case 'BufferStoreNode': {
      const buffer = n.buffer as { name?: string };
      const indices = (n.indices as unknown[]).map(expr).join(', ');
      return store(`${buffer.name ?? '?'}[${indices}]`, expr(n.value), path);
    }

    case 'LIRFlatStoreNode': {
      const buffer = n.buffer as { name?: string };
      return store(`${buffer.name ?? '?'}[${expr(n.offsetExpr)}]`, expr(n.value), path);
    }

    case 'LIRAccumulatorNode': {
      const loopVar = n.loopVar as { name: string };
      return wrap('accumulator',
        `${String(n.localName)} ${String(n.op)}= over ${loopVar.name} < ${expr(n.extent)}`,
        String(n.dtype), nestFor(n.body, `${path}.b`), `acc:${String(n.localName)}`);
    }

    case 'LIRBindingsNode': {
      const bindings = (n.bindings as { name: string }[]) ?? [];
      return wrap('bindings', bindings.map(b => b.name).join(', ') || 'bindings', '', nestFor(n.body, `${path}.b`));
    }

    case 'EvaluateNode':
      return wrap('stmt', expr(n.value), '');

    default:
      return wrap('stmt', String(n.type).replace(/Node$/, ''), '');
  }
}

function nestForFunc(func: { name: string; body?: unknown }): NestNode {
  return {
    id: `func:${func.name}`,
    kind: 'func',
    label: func.name,
    detail: '',
    op: null,
    opId: null,
    children: nestFor(func.body, 'r'),
  };
}

function nestedSnapshot(funcs: Iterable<{ name: string }>): Snapshot {
  const chunks: string[] = [];
  const nests: NestNode[] = [];
  let ops = 0;

  for (const func of funcs) {
    chunks.push(printLIR(func as never));
    ops += countNodes(func);
    nests.push(nestForFunc(func as { name: string; body?: unknown }));
  }

  return { text: chunks.join('\n\n'), ops, bytes: 0, flops: 0, dags: [], nests };
}

export function takeSnapshot(target: unknown, level: IRLevelName): Snapshot {
  if (!target) return EMPTY;
  if (level === 'graph-module') return graphSnapshot(target as GraphModule);
  if (level === 'graph-func') return functionSnapshot(target as GraphFunction);
  return nestedSnapshot(target as Iterable<{ name: string }>);
}
