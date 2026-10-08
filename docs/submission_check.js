"use strict";
// Browser version of test_submission.py. The submission page runs it in a Web Worker on the chosen file and only
// uploads files that pass, so a broken model doesn't use up one of the participant's daily attempts.
//
// Keep the checks in step with test_submission.py and evaluate.py (load_model, embed), which stay the reference. It
// adds a first check that the file is ONNX at all; the throughput estimate runs only locally.
//
// Protocol: the page posts {file, max_bytes, base_url}; this script posts
//   {type: "steps", steps: [title, ...], command}       once, first
//   {type: "step", index, status: "running"|"ok"|"fail", note?}
//   {type: "warn", index, message}
//   {type: "done", ok, message?}                        the file passed (ok) or must be fixed (message says how)
//   {type: "unavailable", message}                      the check itself could not run; the page doesn't block

const ORT_VERSION = "1.22.0";  // closest onnxruntime-web release to the evaluator's onnxruntime 1.22.1
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const COMMAND = "uv run test_submission.py model.onnx";
const BATCH = 64, CHANNELS = 1, CONTEXT = 1024, MAX_DIM = 2048, OPSET_MIN = 17;
// The evaluator's onnxruntime 1.22.1 reads ONNX IR versions up to 10 and opsets up to 22. They are checked before
// loading because onnxruntime-web 1.22 reports load errors without a message.
const MAX_IR = 10, MAX_OPSET = 22;
const EXPORT_HINT = "export() in example_train.py exports an encoder in the right format.";

class Failed extends Error {}  // a problem with the model; the message says how to fix it
class Unavailable extends Error {}  // the check itself could not run in this browser
class SubmissionError extends Error {}  // raised by loadModel and embed, like evaluate.SubmissionError

const post = message => self.postMessage(message);

function warn(ctx, message) {
  ctx.warnings.push(message);
  post({type: "warn", index: ctx.step, message});
}

const pyShape = dims => "(" + dims.join(", ") + (dims.length === 1 ? ",)" : ")");
const pyList = dims => "[" + dims.map(d => typeof d === "string" ? `'${d}'` : String(d)).join(", ") + "]";
const mib = bytes => (bytes / 1024 ** 2).toLocaleString("en-US", {minimumFractionDigits: 1, maximumFractionDigits: 1});
const num = x => x.toLocaleString("en-US", {maximumSignificantDigits: 6});

// --------------------------------------------------------------------------------------------- ONNX protobuf

// Reads just the parts of an ONNX ModelProto the checks need, skipping weight data, so a 20 MiB file parses quickly.
// Field numbers are from onnx.proto.
class Proto {
  constructor(bytes, start = 0, end = bytes.length) { this.b = bytes; this.pos = start; this.end = end; }

  varint() {
    let result = 0n, shift = 0n;
    for (;;) {
      if (this.pos >= this.end || shift > 63n) throw new Error("truncated or malformed varint");
      const byte = this.b[this.pos++];
      result |= BigInt(byte & 0x7f) << shift;
      if (!(byte & 0x80)) return BigInt.asIntN(64, result);
      shift += 7n;
    }
  }

  // Yields [field number, wire type, value], where value is a number for varints and a sub-reader for
  // length-delimited fields. Other wire types are skipped.
  *fields() {
    while (this.pos < this.end) {
      const key = Number(this.varint()), field = key >>> 3, wire = key & 7;
      if (wire === 0) yield [field, wire, Number(this.varint())];
      else if (wire === 2) {
        const length = Number(this.varint());
        if (length < 0 || this.pos + length > this.end) throw new Error("truncated length-delimited field");
        yield [field, wire, new Proto(this.b, this.pos, this.pos + length)];
        this.pos += length;
      } else if (wire === 1 || wire === 5) {
        this.pos += wire === 1 ? 8 : 4;
        if (this.pos > this.end) throw new Error("truncated fixed-width field");
      } else throw new Error(`unsupported wire type ${wire}`);
    }
  }

  string() { return new TextDecoder().decode(this.b.subarray(this.pos, this.end)); }
}

function parseModel(bytes) {
  const model = {ir_version: null, opsets: {}, graph: null};
  for (const [field, wire, value] of new Proto(bytes).fields()) {
    if (field === 1 && wire === 0) model.ir_version = value;
    else if (field === 8 && wire === 2) {
      let domain = "", version = null;
      for (const [f, w, v] of value.fields()) {
        if (f === 1 && w === 2) domain = v.string();
        else if (f === 2 && w === 0) version = v;
      }
      model.opsets[domain || "ai.onnx"] = version;
    } else if (field === 7 && wire === 2) model.graph = parseGraph(value);
  }
  return model;
}

function parseGraph(proto) {
  const graph = {nodes: 0, inputs: [], outputs: [], initializers: []};
  for (const [field, wire, value] of proto.fields()) {
    if (wire !== 2) continue;
    if (field === 1) graph.nodes++;
    else if (field === 5) {
      let name = "", external = false;
      for (const [f, w, v] of value.fields()) {
        if (f === 8 && w === 2) name = v.string();
        else if (f === 14 && w === 0) external = v === 1;  // data_location == EXTERNAL
      }
      graph.initializers.push({name, external});
    } else if (field === 11 || field === 12) (field === 11 ? graph.inputs : graph.outputs).push(parseValueInfo(value));
  }
  return graph;
}

function parseValueInfo(proto) {
  const info = {name: "", tensor: false, dtype: null, dims: null};
  for (const [field, wire, value] of proto.fields()) {
    if (field === 1 && wire === 2) info.name = value.string();
    else if (field === 2 && wire === 2) {
      for (const [f, w, tensor] of value.fields()) {
        if (f !== 1 || w !== 2) continue;  // TypeProto.tensor_type
        info.tensor = true;
        for (const [tf, tw, tv] of tensor.fields()) {
          if (tf === 1 && tw === 0) info.dtype = DTYPES[tv] || `TYPE_${tv}`;
          else if (tf === 2 && tw === 2) {
            info.dims = [];
            for (const [sf, sw, dim] of tv.fields()) {
              if (sf !== 1 || sw !== 2) continue;
              let d = null;
              for (const [df, dw, dv] of dim.fields()) {
                if (df === 1 && dw === 0) d = dv;
                else if (df === 2 && dw === 2 && d === null) d = dv.string() || null;
              }
              info.dims.push(d);
            }
          }
        }
        info.dtype ??= "UNDEFINED";
      }
    }
  }
  return info;
}

const DTYPES = ["UNDEFINED", "FLOAT", "UINT8", "INT8", "UINT16", "INT16", "INT32", "INT64", "STRING", "BOOL",
  "FLOAT16", "DOUBLE", "UINT32", "UINT64", "COMPLEX64", "COMPLEX128", "BFLOAT16", "FLOAT8E4M3FN", "FLOAT8E4M3FNUZ",
  "FLOAT8E5M2", "FLOAT8E5M2FNUZ", "UINT4", "INT4", "FLOAT4E2M1", "FLOAT8E8M0", "UINT2", "INT2"];

// --------------------------------------------------------------------------------------------- 1. file

function zipNames(bytes) {
  // The end-of-central-directory record is in the last 64 KiB; it points at the list of file names.
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 65557); i--) {
    if (view.getUint32(i, true) !== 0x06054b50) continue;
    const names = [], count = view.getUint16(i + 10, true);
    let p = view.getUint32(i + 16, true);
    for (let k = 0; k < count; k++) {
      if (p + 46 > bytes.length || view.getUint32(p, true) !== 0x02014b50) return null;
      const n = view.getUint16(p + 28, true), extra = view.getUint16(p + 30, true), comment = view.getUint16(p + 32, true);
      names.push(new TextDecoder().decode(bytes.subarray(p + 46, p + 46 + n)));
      p += 46 + n + extra + comment;
    }
    return names;
  }
  return null;
}

function sniff(bytes, head) {
  const starts = prefix => typeof prefix === "string"
    ? [...prefix].every((c, i) => head[i] === c.charCodeAt(0)) : prefix.every((b, i) => head[i] === b);
  const pytorch = "This looks like a PyTorch checkpoint (torch.save / TorchScript), not an ONNX model. Load your " +
    "encoder in PyTorch and export it to ONNX, as export() in example_train.py does.";
  if (starts("version https://git-lfs"))
    return "This is a Git LFS pointer file, not the model itself. Run `git lfs pull` (or download the file " +
      "directly) to get the real model.";
  const firstNonSpace = head.find(b => ![0x20, 0x09, 0x0a, 0x0b, 0x0c, 0x0d].includes(b));
  if (firstNonSpace === 0x3c)
    return "This is an HTML/XML page, not a model. This usually happens when downloading from a share link " +
      "(Google Drive, Colab, Dropbox, ...) saves the web page instead of the file. Download the file " +
      "itself, for example from the browser's download button.";
  if (starts("PK")) {
    const names = zipNames(bytes);
    if (names === null) return "This is a damaged zip archive, not an ONNX model.";
    const onnx = names.filter(n => n.toLowerCase().endsWith(".onnx"));
    if (onnx.length)
      return `This is a zip archive containing '${onnx[0]}'. Unzip it and check/submit the .onnx file itself.`;
    if (names.some(n => ["data.pkl", "constants.pkl", "version"].some(s => n.endsWith(s)))) return pytorch;
    return "This is a zip archive, not an ONNX model. Submit the .onnx file itself.";
  }
  if (starts([0x1f, 0x8b])) return "This is a gzip-compressed file. Decompress it (gunzip) and submit the .onnx file itself.";
  if (head[0] === 0x80 && [2, 3, 4, 5].includes(head[1]))
    return pytorch.replace("(torch.save / TorchScript)", "or another Python pickle");
  if (starts([0x89, 0x48, 0x44, 0x46]))
    return "This is an HDF5 file (for example a Keras .h5 model), not ONNX. Convert it with tf2onnx, or export " +
      "your PyTorch encoder as export() in example_train.py does.";
  if (starts([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]))  // \x93NUMPY
    return "This is a NumPy array file (.npy), not an ONNX model.";
  if (head.length >= 10 && head[8] === 0x7b && head[9] === 0x22)
    return "This looks like a safetensors file, which holds only weights. Load the weights into your model " +
      "in PyTorch and export it as export() in example_train.py does.";
  try {
    const text = new TextDecoder("utf-8", {fatal: true}).decode(head);
    if (/^(?:\P{C}|\s)*$/u.test(text))
      return `This is a text file, not an ONNX model. It starts with: '${text.slice(0, 80)}'`;
  } catch { /* not UTF-8 text */ }
  return null;
}

async function checkFile(ctx) {
  const {file} = ctx;
  if (file.size === 0)
    throw new Failed("The file is empty (0 bytes). The export or the copy did not finish; export the model again.");
  ctx.bytes = new Uint8Array(await file.arrayBuffer());
  const problem = sniff(ctx.bytes, ctx.bytes.subarray(0, 512));
  if (problem) throw new Failed(problem);
  if (file.size > ctx.maxBytes)
    throw new Failed(`The file is ${mib(file.size)} MiB, over the ${num(ctx.maxBytes / 1024 ** 2)} MiB upload ` +
      `limit, so the submission page will reject it. Use a smaller encoder (about ` +
      `${num(ctx.maxBytes / 1024 ** 2 / 4)} million float32 parameters at most).`);
  const dot = file.name.lastIndexOf(".");
  const suffix = dot > 0 ? file.name.slice(dot) : "";
  if (suffix.toLowerCase() !== ".onnx")
    warn(ctx, `The file name ends in '${suffix || "(no extension)"}' rather than '.onnx'. That's fine for ` +
      `evaluation, but double-check that this is the file you meant.`);
  return `${file.name}, ${mib(file.size)} MiB`;
}

// --------------------------------------------------------------------------------------------- 2. graph

function checkGraph(ctx) {
  let model;
  try {
    model = parseModel(ctx.bytes);
  } catch (e) {
    throw new Failed(`The file is not a readable ONNX model (${e.message}). It may be truncated by an interrupted ` +
      `export, copy or download, or not be ONNX at all. Export it again.`);
  }
  const graph = model.graph || {nodes: 0, inputs: [], outputs: [], initializers: []};
  if (!graph.nodes && !graph.outputs.length)
    throw new Failed("The file parses but contains no model graph, so it is probably not an ONNX model. Export it " +
      "again. " + EXPORT_HINT);
  const external = graph.initializers.filter(t => t.external).map(t => t.name);
  if (external.length)
    throw new Failed(`The weights (${external.length} tensors, e.g. '${external[0]}') are stored in separate files ` +
      `next to the .onnx (ONNX 'external data'), but only the .onnx file is uploaded. Merge them into one file ` +
      `with:\n    import onnx; onnx.save(onnx.load('${ctx.file.name}'), 'merged.onnx')\n` +
      `This only works for models under 2 GB.`);
  // test_submission.py checks the opset after running the model; it needs no runtime, so check it first here.
  const opset = model.opsets["ai.onnx"] ?? null;
  ctx.opset = opset;
  if (opset !== null && opset < OPSET_MIN)
    throw new Failed(`The model uses ONNX opset ${opset}, but the minimum is ${OPSET_MIN}. Export with ` +
      `opset_version=${OPSET_MIN} or newer. ` + EXPORT_HINT);
  if (opset !== null && opset > MAX_OPSET)
    throw new Failed(`The model uses ONNX opset ${opset}, newer than the evaluator's onnxruntime supports (at most ` +
      `${MAX_OPSET}). Export with opset_version=${OPSET_MIN}.`);
  if (model.ir_version !== null && model.ir_version > MAX_IR)
    throw new Failed(`The model file format (ONNX IR version ${model.ir_version}) is newer than the evaluator's ` +
      `onnxruntime reads (at most ${MAX_IR}), usually because a newer onnx/torch wrote it. Lower it with:\n` +
      `    import onnx; m = onnx.load('${ctx.file.name}'); m.ir_version = ${MAX_IR}; onnx.save(m, '${ctx.file.name}')`);
  return `${graph.nodes} nodes, opset ${opset ?? "unknown"} (minimum ${OPSET_MIN})`;
}

// --------------------------------------------------------------------------------------------- 3. load

// onnxruntime-web wraps errors in its C API call and the C++ source location; keep only the message itself.
const ortMessage = e => String(e?.message || e).trim()
  .replace(/^(?:Can't create a session\.|failed to call OrtRun\(\)\.)\s*ERROR_CODE: \d+, ERROR_MESSAGE: /, "")
  .replace(/^\S+\.(?:h|cc|cpp):\d+ [^(]*\([^)]*\)(?: const)? /, "");

// A one-node Identity model, loaded first so a runtime that can't start is never blamed on the participant's model.
const TINY_MODEL = "CAgSADo3ChAKAXgSAXkiCElkZW50aXR5EgF0Wg8KAXgSCgoICAESBAoCCAFiDwoBeRIKCggIARIECgIIAUIECgAQEQ==";

let runtime = null;
function startRuntime() {
  runtime ??= (async () => {
    importScripts(ORT_URL + "ort.wasm.min.js");
    ort.env.wasm.wasmPaths = ORT_URL;
    ort.env.wasm.numThreads = 1;  // threads need cross-origin isolation, which the submission page doesn't have
    ort.env.logLevel = "error";
    const tiny = Uint8Array.from(atob(TINY_MODEL), c => c.charCodeAt(0));
    await (await ort.InferenceSession.create(tiny)).release();
  })();
  return runtime;
}

// Same rules as evaluate.load_model.
async function checkLoad(ctx) {
  try {
    await startRuntime();
  } catch (e) {
    throw new Unavailable(`onnxruntime-web could not start in this browser (${e?.message || e}).`);
  }
  let sess;
  try {
    sess = await ort.InferenceSession.create(ctx.bytes, {executionProviders: ["wasm"]});
  } catch (e) {
    const msg = ortMessage(e), lower = msg.toLowerCase();
    let hint;
    if (/^\d*$/.test(msg))  // onnxruntime-web 1.22 sometimes throws only an internal error code
      throw new Failed(`onnxruntime could not load the model, and gave no details in the browser. Run ${COMMAND} on ` +
        `your computer to see the full error.`);
    if (msg.includes("IR version"))
      hint = `The model file format (ONNX IR version) is newer than the evaluator's onnxruntime reads, usually ` +
        `because a newer onnx/torch wrote it. Lower it with:\n    import onnx; m = onnx.load('${ctx.file.name}'); ` +
        `m.ir_version = 10; onnx.save(m, '${ctx.file.name}')`;
    else if (lower.includes("opset") && (lower.includes("support") || lower.includes("version")))
      hint = `The model uses ONNX opset ${ctx.opset ?? "unknown"}, newer than the evaluator's onnxruntime ` +
        `supports. Export with opset_version=${OPSET_MIN}.`;
    else if (msg.includes("No Op registered") || lower.includes("not a registered") || lower.includes("custom") ||
             lower.includes("domain"))
      hint = "The model uses an operator that onnxruntime doesn't have (a custom op or a non-standard domain). " +
        "Evaluation uses plain onnxruntime, so the model must use standard ONNX ops; replace the custom layer " +
        "(e.g. a fused CUDA kernel) with plain PyTorch ops before export.";
    else hint = "Re-export the model, or simplify the layer named in the error above.";
    throw new Failed(`Could not load the ONNX model:\n    ${msg}\n${hint}`);
  }
  const ins = sess.inputMetadata, outs = sess.outputMetadata;
  if (ins.length !== 1 || outs.length !== 1) {
    const names = list => list.map(v => `'${v.name}'`).join(", ") || "none";
    throw new Failed(`ONNX model must have exactly one input and one output, but this one has ${ins.length} ` +
      `input(s) (${names(ins)}) and ${outs.length} output(s) (${names(outs)}). Export only the encoder, with a ` +
      `single window tensor in and a single embedding tensor out. ` + EXPORT_HINT);
  }
  const input = ins[0];
  if (!input.isTensor || input.type !== "float32")
    throw new Failed(`Model input must be float32, not ${input.isTensor ? input.type : "a non-tensor value"}. ` +
      `Export the model in float32 (no model.half()) with a float32 example input.`);
  const shape = input.shape, want = [BATCH, CHANNELS, CONTEXT];
  if (shape.length !== 3 || shape.some((g, i) => typeof g === "number" && g !== want[i]))
    throw new Failed(`Model input must accept shape ${pyShape(want)}; it declares ${pyList(shape)}. Export with ` +
      `an example input torch.randn(${BATCH}, ${CHANNELS}, ${CONTEXT}): a batch of ${BATCH} windows, 1 channel, ` +
      `${CONTEXT} values. ` + EXPORT_HINT);
  ctx.session = sess;
  return `one float32 input accepting ${pyShape(want)}; onnxruntime-web ${ORT_VERSION} in this browser`;
}

// --------------------------------------------------------------------------------------------- 4-7. running

function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1, exp = (h >> 10) & 0x1f, frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

function floats(tensor) {
  if (tensor.type === "float16" && tensor.data instanceof Uint16Array) return Float64Array.from(tensor.data, halfToFloat);
  return Float64Array.from(tensor.data, Number);
}


function explainSubmissionError(e) {
  const msg = e.message;
  if (msg.startsWith("Embedding shape"))
    return `${msg}. The model must return one embedding per window, [${BATCH}, D]: pool or flatten any token or ` +
      `channel dimensions inside the model.`;
  if (msg.startsWith("Embedding width"))
    return `${msg}. Make the last layer narrower, or project the embedding to at most ${MAX_DIM} features.`;
  if (msg.includes("NaN or infinite"))
    return `${msg}. Look for divisions by zero, log(0), normalising by a zero standard deviation (add a small ` +
      `epsilon), or float16 overflow.`;
  if (msg.startsWith("Model failed"))
    return `${msg}\nThe model crashed while running. Check that it accepts the input shape above; export with an ` +
      `example input of exactly that shape.`;
  return msg;
}

// Same as evaluate.embed for a [n <= BATCH, CONTEXT] input: one zero-padded batch, trimmed back to n rows.
async function embed(sess, windows) {
  const shape = [BATCH, CHANNELS, CONTEXT], x = new Float32Array(BATCH * CONTEXT), n = windows.length;
  windows.forEach((w, i) => x.set(w, i * CONTEXT));
  let z;
  try {
    const result = await sess.run({[sess.inputNames[0]]: new ort.Tensor("float32", x, shape)});
    z = result[sess.outputNames[0]];
  } catch (e) {
    throw new SubmissionError(`Model failed on float32 input of shape ${pyShape(shape)}: ${ortMessage(e)}`);
  }
  if (z.dims.length !== 2 || z.dims[0] !== BATCH)
    throw new SubmissionError(`Embedding shape ${pyShape(z.dims)}, expected (${BATCH}, D)`);
  const dim = z.dims[1];
  if (!(dim >= 1 && dim <= MAX_DIM)) throw new SubmissionError(`Embedding width ${dim}, expected between 1 and ${MAX_DIM}`);
  const values = floats(z).subarray(0, n * dim);
  if (!values.every(Number.isFinite)) throw new SubmissionError("Embeddings contain NaN or infinite values");
  return Array.from({length: n}, (_, i) => values.subarray(i * dim, (i + 1) * dim));
}

async function embedOrFail(ctx, windows) {
  try {
    return await embed(ctx.session, windows);
  } catch (e) {
    if (e instanceof SubmissionError) throw new Failed(explainSubmissionError(e));
    throw e;
  }
}

// Standard normal windows from a fixed seed (xorshift32 + Box-Muller).
function randomWindows(n, seed = 1) {
  const uniform = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return ((seed >>> 0) + 0.5) / 2 ** 32; };
  return Array.from({length: n}, () => Float32Array.from({length: CONTEXT},
    () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform())));
}

async function checkEmbed(ctx) {
  ctx.x = randomWindows(BATCH);
  ctx.z = await embedOrFail(ctx, ctx.x);
  return `embeddings (${BATCH}, ${ctx.z[0].length}), finite`;
}

async function checkDeterministic(ctx) {
  const again = await embedOrFail(ctx, ctx.x);
  if (!ctx.z.every((row, i) => row.every((v, j) => v === again[i][j])))
    throw new Failed("The same input gave different embeddings. The model probably contains randomness such as " +
      "dropout; call model.eval() before exporting.");
  return "the same input gives the same embedding";
}

async function checkIndependent(ctx) {
  const perm = ctx.x.map((_, i) => i).reverse();
  const z = await embedOrFail(ctx, perm.map(i => ctx.x[i]));
  const close = z.every((row, k) => row.every((v, j) => {
    const ref = ctx.z[perm[k]][j];
    return Math.abs(v - ref) <= 1e-4 + 1e-3 * Math.abs(ref);
  }));
  if (!close)
    throw new Failed("A window's embedding depends on the other windows in its batch. This usually means batch " +
      "norm was exported in training mode; call model.eval() before exporting.");
  return "each window is encoded independently of its batch";
}

async function checkScales(ctx) {
  for (const [label, f] of [["scaled by 1e-3", v => v * 1e-3], ["scaled by 1e3", v => v * 1e3],
                            ["offset by 1000", v => v + 1000]]) {
    try {
      await embed(ctx.session, ctx.x.map(w => w.map(f)));
    } catch (e) {
      if (!(e instanceof SubmissionError)) throw e;
      throw new Failed(`With inputs ${label}: ${e.message}. The evaluator passes raw windows without normalizing ` +
        `them, and real series have very different scales and offsets. Normalize each window inside the model, ` +
        `e.g. subtract its mean and divide by its standard deviation plus a small epsilon, as SeriesViT in ` +
        `example_train.py does.`);
    }
  }
  return "finite embeddings for inputs scaled by 1e-3 and 1e3 and offset by 1000";
}

// --------------------------------------------------------------------------------------------- main

const STEPS = [
  ["File is an ONNX model", checkFile],
  ["ONNX graph is valid", checkGraph],
  ["Loads, with one float32 input accepting (64, 1, 1024)", checkLoad],
  ["Returns finite (64, D) embeddings, 1 ≤ D ≤ 2048", checkEmbed],
  ["Deterministic", checkDeterministic],
  ["Each window encoded independently of its batch", checkIndependent],
  ["Finite embeddings for inputs at very different scales", checkScales],
];

async function run({file, max_bytes, base_url}) {
  post({type: "steps", steps: STEPS.map(([title]) => title), command: COMMAND});
  startRuntime().catch(() => {});  // download onnxruntime while the first checks run; failures surface in step 3
  const ctx = {file, maxBytes: max_bytes, baseUrl: base_url, warnings: [], step: 0};
  for (const [index, [, check]] of STEPS.entries()) {
    ctx.step = index;
    post({type: "step", index, status: "running"});
    try {
      const note = await check(ctx);
      post({type: "step", index, status: "ok", note});
    } catch (e) {
      if (e instanceof Unavailable) { post({type: "unavailable", message: e.message}); return; }
      post({type: "step", index, status: "fail"});
      post({type: "done", ok: false, message: e instanceof Failed ? e.message
        : `Unexpected error: ${e?.message || e}\nThis most likely comes from the model. If you think it's a bug ` +
          `in the check, run ${COMMAND} and report its output to the organizers.`});
      return;
    }
  }
  post({type: "done", ok: true});
}

self.onmessage = event => { run(event.data); };
