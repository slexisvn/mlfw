import { ScheduleSketch, SearchVariable } from './sketch.js';
import { enumerateFactorizations } from './factorization.js';
import { levelCounts } from './tile_structure.js';

const THREAD_AXES = ['x', 'y', 'z'];

function staticExtent(loop) {
  return loop.extent && loop.extent.type === 'IntImmNode' ? loop.extent.value : null;
}

function multiLevelSplit(schedule, blockName, loopVarName, factors) {
  const levelNames = [];
  let curName = loopVarName;
  for (let i = 0; i < factors.length - 1; i++) {
    const cur = schedule.getLoops(blockName).find(l => l.loopVar.name === curName);
    if (!cur) return null;
    let inner = 1;
    for (let j = i + 1; j < factors.length; j++) inner *= factors[j];
    const [outer, innerLoop] = schedule.split(cur, inner);
    levelNames.push(outer.loopVar.name);
    curName = innerLoop.loopVar.name;
  }
  levelNames.push(curName);
  return levelNames;
}

function applyRoles(schedule, blockName, structure, spatialLevelNames, reductionLevelNames) {
  const find = (name) => schedule.getLoops(blockName).find(l => l.loopVar.name === name);
  for (const [kind, level] of structure.order) {
    const role = structure.roles[kind + level];
    if (!role) continue;
    const axes = kind === 'S' ? spatialLevelNames : reductionLevelNames;
    if (role === 'parallelize') {
      const loop = axes[0] && level < axes[0].length ? find(axes[0][level]) : null;
      if (loop) schedule.parallelize(loop);
    } else if (role === 'vectorize') {
      const lastAxis = axes[axes.length - 1];
      const loop = lastAxis && level < lastAxis.length ? find(lastAxis[level]) : null;
      if (loop) schedule.vectorize(loop);
    } else if (role === 'unroll') {
      for (const axisNames of axes) {
        const loop = level < axisNames.length ? find(axisNames[level]) : null;
        if (loop) schedule.unroll(loop);
      }
    } else if (role === 'blockIdx' || role === 'threadIdx') {
      axes.forEach((axisNames, ai) => {
        if (ai >= THREAD_AXES.length) return;
        const loop = level < axisNames.length ? find(axisNames[level]) : null;
        if (loop) schedule.bindThread(loop, `${role}.${THREAD_AXES[ai]}`);
      });
    }
  }
}

function tileBlock(schedule, blockName, structure, spatialNames, reductionNames, params) {
  const spatialLevelNames = [];
  for (let i = 0; i < spatialNames.length; i++) {
    const names = multiLevelSplit(schedule, blockName, spatialNames[i], params[`s${i}`]);
    if (!names) return;
    spatialLevelNames.push(names);
  }
  const reductionLevelNames = [];
  for (let j = 0; j < reductionNames.length; j++) {
    const names = multiLevelSplit(schedule, blockName, reductionNames[j], params[`r${j}`]);
    if (!names) return;
    reductionLevelNames.push(names);
  }

  const orderedNames = [];
  for (const [kind, level] of structure.order) {
    const axes = kind === 'S' ? spatialLevelNames : reductionLevelNames;
    for (const axisNames of axes) {
      if (level < axisNames.length) orderedNames.push(axisNames[level]);
    }
  }
  const byName = new Map(schedule.getLoops(blockName).map(l => [l.loopVar.name, l]));
  const orderedLoops = orderedNames.map(n => byName.get(n)).filter(Boolean);
  if (orderedLoops.length >= 2) schedule.reorder(...orderedLoops);

  applyRoles(schedule, blockName, structure, spatialLevelNames, reductionLevelNames);
}

function splitLoops(blockInfo) {
  const spatialLoops = blockInfo.loops.filter(l => !blockInfo.reductionLoopVars.has(l.loopVar.name));
  const reductionLoops = blockInfo.loops.filter(l => blockInfo.reductionLoopVars.has(l.loopVar.name));
  return { spatialLoops, reductionLoops };
}

function tilingVariables(spatialLoops, reductionLoops, spatialLevels, reductionLevels) {
  const variables = [];
  spatialLoops.forEach((l, i) => variables.push(new SearchVariable(`s${i}`, enumerateFactorizations(staticExtent(l), spatialLevels))));
  reductionLoops.forEach((l, j) => variables.push(new SearchVariable(`r${j}`, enumerateFactorizations(staticExtent(l), reductionLevels))));
  return variables;
}

export function createMultiLevelTilingSketch(blockInfo, structure) {
  const { spatialLevels, reductionLevels } = levelCounts(structure);
  const { spatialLoops, reductionLoops } = splitLoops(blockInfo);
  if (spatialLoops.length === 0) return null;
  for (const l of [...spatialLoops, ...reductionLoops]) {
    if (staticExtent(l) === null) return null;
  }

  const variables = tilingVariables(spatialLoops, reductionLoops, spatialLevels, reductionLevels);
  const spatialNames = spatialLoops.map(l => l.loopVar.name);
  const reductionNames = reductionLoops.map(l => l.loopVar.name);

  return new ScheduleSketch(structure.name, variables, (schedule, blockName, tgt, params) => {
    tileBlock(schedule, blockName, structure, spatialNames, reductionNames, params);
  });
}

export function createSSRSRSTilingSketch(blockInfo, structure) {
  const { spatialLevels, reductionLevels } = levelCounts(structure);
  const { spatialLoops, reductionLoops } = splitLoops(blockInfo);
  if (spatialLoops.length === 0 || reductionLoops.length === 0) return null;
  for (const l of [...spatialLoops, ...reductionLoops]) {
    if (staticExtent(l) === null) return null;
  }

  const variables = tilingVariables(spatialLoops, reductionLoops, spatialLevels, reductionLevels);
  const spatialNames = spatialLoops.map(l => l.loopVar.name);
  const reductionNames = reductionLoops.map(l => l.loopVar.name);

  return new ScheduleSketch(structure.name, variables, (schedule, blockName, tgt, params) => {
    schedule.decomposeReduction(blockName);
    tileBlock(schedule, `${blockName}_upd`, structure, spatialNames, reductionNames, params);
  });
}
