export const STORE_NODE_TYPES: ReadonlySet<string> = new Set(['BufferStoreNode', 'LIRFlatStoreNode']);
export const LOAD_NODE_TYPES: ReadonlySet<string> = new Set(['BufferLoadNode', 'LIRFlatLoadNode']);
export const ACCESS_NODE_TYPES: ReadonlySet<string> = new Set([...STORE_NODE_TYPES, ...LOAD_NODE_TYPES]);
export const SEQ_LOOP_NODE_TYPES: ReadonlySet<string> = new Set(['WhileNode', 'LIRAccumulatorNode']);
export const LOOP_NODE_TYPES: ReadonlySet<string> = new Set(['ForNode', ...SEQ_LOOP_NODE_TYPES]);
