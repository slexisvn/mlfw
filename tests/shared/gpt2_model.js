export function buildGpt2(nn, tensor, dims) {
  const { V, D, H, FF, L, SEQ } = dims;
  const tok = new nn.Embedding(V, D);
  const posEmb = new nn.Embedding(SEQ, D);
  const blocks = Array.from({ length: L }, () => new nn.TransformerEncoderLayer(D, H, FF, 0.0, 'gelu', 1e-5, true, true));
  blocks.forEach((m) => m.eval());
  const lnf = new nn.LayerNorm(D, 1e-5);
  const head = new nn.Linear(D, V);
  const positions = tensor(Array.from({ length: SEQ }, (_, i) => i));
  const fwd = (ids) => {
    let h = tok.forward(ids).add(posEmb.forward(positions));
    for (const blk of blocks) h = blk.forward(h, null, null, true);
    return head.forward(lnf.forward(h));
  };
  const ids = tensor([Array.from({ length: SEQ }, (_, i) => (i * 7 + 3) % V)]);
  return { fwd, ids };
}
