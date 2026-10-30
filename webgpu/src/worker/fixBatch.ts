// Rewrites the batch axis of an ONNX model from symbolic ("batch") to a fixed size.
//
// onnxruntime-web's WebGL backend cannot run a model whose input has a symbolic dimension, and it ignores
// the freeDimensionOverrides session option, so the shape has to be fixed in the file itself. The network
// has no intermediate shape information (no value_info), so only the graph's input and output declarations
// carry the batch axis. This walks the protobuf, replaces that one dimension in each of them and copies
// every other byte (the 47 MiB of weights included) untouched.
//
// Protobuf layout used (onnx.proto):
//   ModelProto.graph = 7
//   GraphProto.input = 11, output = 12, value_info = 13
//   ValueInfoProto.type = 2 -> TypeProto.tensor_type = 1 -> Tensor.shape = 2 -> TensorShapeProto.dim = 1
//   Dimension: dim_value = 1 (varint), dim_param = 2 (string)

type Parts = Uint8Array[];

const F_MODEL_GRAPH = 7;
const F_GRAPH_VALUE_INFOS = [11, 12, 13];
const F_VALUE_INFO_TYPE = 2;
const F_TYPE_TENSOR = 1;
const F_TENSOR_SHAPE = 2;
const F_SHAPE_DIM = 1;
const F_DIM_VALUE = 1;
const F_DIM_PARAM = 2;

function readVarint(buf: Uint8Array, pos: number): [value: number, next: number] {
  let value = 0;
  let scale = 1;
  for (;;) {
    if (pos >= buf.length) throw new Error('truncated protobuf varint');
    const byte = buf[pos++]!;
    value += (byte & 0x7f) * scale;
    if (byte < 0x80) return [value, pos];
    scale *= 128;
  }
}

function varint(value: number): Uint8Array {
  const out: number[] = [];
  let v = value;
  while (v >= 0x80) {
    out.push((v % 0x80) | 0x80);
    v = Math.floor(v / 0x80);
  }
  out.push(v);
  return Uint8Array.from(out);
}

const sizeOf = (parts: Parts) => parts.reduce((n, p) => n + p.length, 0);

/**
 * Copies a message field by field. A field whose number has a handler and is length-delimited is replaced by
 * the handler's output (re-framed with the new length); every other field is copied as it is.
 */
function rewriteFields(buf: Uint8Array, handlers: Record<number, (payload: Uint8Array, index: number) => Parts>): Parts {
  const out: Parts = [];
  const seen: Record<number, number> = {};
  let pos = 0;
  while (pos < buf.length) {
    const start = pos;
    const [tag, afterTag] = readVarint(buf, pos);
    const field = Math.floor(tag / 8);
    const wire = tag % 8;
    let end: number;
    let payloadStart = afterTag;
    if (wire === 0) end = readVarint(buf, afterTag)[1];
    else if (wire === 1) end = afterTag + 8;
    else if (wire === 5) end = afterTag + 4;
    else if (wire === 2) {
      const [len, next] = readVarint(buf, afterTag);
      payloadStart = next;
      end = next + len;
    } else throw new Error(`unsupported protobuf wire type ${wire}`);
    if (end > buf.length) throw new Error('truncated protobuf field');

    const handler = wire === 2 ? handlers[field] : undefined;
    if (handler) {
      const index = (seen[field] = (seen[field] ?? -1) + 1);
      const inner = handler(buf.subarray(payloadStart, end), index);
      out.push(buf.subarray(start, afterTag), varint(sizeOf(inner)), ...inner);
    } else {
      out.push(buf.subarray(start, end));
    }
    pos = end;
  }
  return out;
}

const flatten = (parts: Parts): Uint8Array => {
  const out = new Uint8Array(sizeOf(parts));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
};

/** The first dimension (the batch axis) of one input / output declaration becomes `batch`. */
function fixValueInfo(valueInfo: Uint8Array, batch: number): Parts {
  return rewriteFields(valueInfo, {
    [F_VALUE_INFO_TYPE]: (type) =>
      rewriteFields(type, {
        [F_TYPE_TENSOR]: (tensor) =>
          rewriteFields(tensor, {
            [F_TENSOR_SHAPE]: (shape) =>
              rewriteFields(shape, {
                [F_SHAPE_DIM]: (dim, index) => {
                  if (index !== 0) return [dim];
                  return fixDimension(dim, batch);
                },
              }),
          }),
      }),
  });
}

/** A dimension with dim_param (or an old dim_value) replaced by dim_value = batch; other fields are kept. */
function fixDimension(dim: Uint8Array, batch: number): Parts {
  const out: Parts = [];
  let pos = 0;
  let wroteValue = false;
  while (pos < dim.length) {
    const start = pos;
    const [tag, afterTag] = readVarint(dim, pos);
    const field = Math.floor(tag / 8);
    const wire = tag % 8;
    let end: number;
    if (wire === 0) end = readVarint(dim, afterTag)[1];
    else if (wire === 2) {
      const [len, next] = readVarint(dim, afterTag);
      end = next + len;
    } else throw new Error(`unexpected wire type ${wire} in a tensor dimension`);
    if (field === F_DIM_PARAM) {
      // symbolic name -> fixed value
      if (!wroteValue) out.push(Uint8Array.of(F_DIM_VALUE * 8), varint(batch));
      wroteValue = true;
    } else if (field === F_DIM_VALUE) {
      if (!wroteValue) out.push(Uint8Array.of(F_DIM_VALUE * 8), varint(batch));
      wroteValue = true;
    } else {
      out.push(dim.subarray(start, end));
    }
    pos = end;
  }
  if (!wroteValue) out.push(Uint8Array.of(F_DIM_VALUE * 8), varint(batch));
  return out;
}

/**
 * Returns a copy of an ONNX model whose batch axis is `batch` instead of symbolic, for every graph input and
 * output. Throws if the bytes are not a protobuf with a graph in it.
 */
export function withFixedBatch(model: Uint8Array, batch: number): Uint8Array {
  if (!Number.isInteger(batch) || batch < 1) throw new Error(`invalid batch size ${batch}`);
  let found = false;
  const parts = rewriteFields(model, {
    [F_MODEL_GRAPH]: (graph) => {
      found = true;
      const handlers: Record<number, (payload: Uint8Array) => Parts> = {};
      for (const f of F_GRAPH_VALUE_INFOS) handlers[f] = (vi) => fixValueInfo(vi, batch);
      return rewriteFields(graph, handlers);
    },
  });
  if (!found) throw new Error('not an ONNX model: no graph found');
  return flatten(parts);
}
