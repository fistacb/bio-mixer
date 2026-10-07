// 生物ミキサー：2軸パッド（年齢×性格など）
// 四隅の4種を双線形に混ぜる。C：事前生成の格子画像を重み付きで重ねる／A：4つの latent を混ぜて TAESDXL でデコード
const ORT_VER = "1.30.0";
const params = new URLSearchParams(location.search);
const CONF = Object.assign({ root: "../", data: "../data/persona/" }, window.BIO_MIXER || {});
const DECODERS = { taesd: `${CONF.root}models/taesd_decoder.onnx`, taesdxl: `${CONF.root}models/taesdxl_decoder.onnx`, taesdxl768: `${CONF.root}models/taesdxl_decoder_768.onnx` };

const st = { index: null, style: null, mode: "c", u: 0.5, v: 0.5, latents: new Map(), sessions: new Map(), busy: false, pending: false, backend: "" };
const $ = (id) => document.getElementById(id);
const els = { style: $("style"), pad: $("pad"), dot: $("dot"), canvas: $("view"), status: $("status"), dev: $("dev"), devinfo: $("devinfo"), weights: $("weights") };
const ctx = els.canvas.getContext("2d");
const imgCache = new Map();
const base = () => `${CONF.data}styles/${st.style.id}`;

async function init() {
  st.index = await (await fetch(`${CONF.data}index.json`, { cache: "no-cache" })).json();
  const grids = st.index.styles.filter((s) => s.mode === "grid2d");
  for (const s of grids) els.style.add(new Option(s.name, s.id));
  st.style = grids.find((s) => s.id === params.get("style")) ?? grids.find((s) => s.id === "manga_xl") ?? grids[0];
  els.style.value = st.style.id;
  els.style.onchange = () => { st.style = grids.find((s) => s.id === els.style.value); renderCorners(); update(); };
  document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  if (params.has("dev")) els.dev.hidden = false;

  const move = (e) => {
    const r = els.pad.getBoundingClientRect();
    st.u = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    st.v = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    update();
  };
  els.pad.addEventListener("pointerdown", (e) => { els.pad.setPointerCapture(e.pointerId); move(e); });
  els.pad.addEventListener("pointermove", (e) => { if (e.buttons) move(e); });
  els.pad.addEventListener("keydown", (e) => {
    const d = 0.05, k = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, -d], ArrowDown: [0, d] }[e.key];
    if (!k) return;
    e.preventDefault();
    st.u = Math.min(1, Math.max(0, st.u + k[0])); st.v = Math.min(1, Math.max(0, st.v + k[1])); update();
  });
  renderCorners();
  setMode(params.get("mode") === "a" ? "a" : "c");
}

function renderCorners() {
  const ax = st.style.axes || { x: ["", ""], y: ["", ""] };
  $("axL").textContent = ax.x[0]; $("axR").textContent = ax.x[1];
  $("axT").textContent = ax.y[0]; $("axB").textContent = ax.y[1];
  const [c00, c10, c01, c11] = st.style.order;
  for (const [id, cid] of [["k00", c00], ["k10", c10], ["k01", c01], ["k11", c11]]) {
    const name = st.index.creatures.find((c) => c.id === cid)?.name ?? cid;
    $(id).innerHTML = `<img src="${base()}/catalog/${cid}.webp" alt=""><span>${name}</span>`;
  }
}

function setMode(m) {
  st.mode = m;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === m)));
  update();
}

function weights() {
  const { u, v } = st;
  return [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v];
}

function update() {
  els.dot.style.left = `${st.u * 100}%`; els.dot.style.top = `${st.v * 100}%`;
  els.pad.setAttribute("aria-valuetext", `横 ${Math.round(st.u * 100)}%、縦 ${Math.round(st.v * 100)}%`);
  const names = st.style.order.map((cid) => st.index.creatures.find((c) => c.id === cid)?.name ?? cid);
  els.weights.textContent = weights().map((w, i) => `${names[i]} ${Math.round(w * 100)}%`).join(" ／ ");
  if (st.mode === "c") updateC(); else updateA();
}

// ---------- C：格子画像を双線形の重みで重ねる ----------
function loadImg(src) {
  if (!imgCache.has(src)) imgCache.set(src, new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; }));
  return imgCache.get(src);
}
let cToken = 0;
async function updateC() {
  const N = st.style.grid, x = st.u * (N - 1), y = st.v * (N - 1);
  const x0 = Math.min(Math.floor(x), N - 2), y0 = Math.min(Math.floor(y), N - 2), fx = x - x0, fy = y - y0;
  const cells = [[x0, y0, (1 - fx) * (1 - fy)], [x0 + 1, y0, fx * (1 - fy)], [x0, y0 + 1, (1 - fx) * fy], [x0 + 1, y0 + 1, fx * fy]];
  const my = ++cToken;
  const imgs = await Promise.all(cells.map(([i, j]) => loadImg(`${base()}/frames/grid/${String(i).padStart(2, "0")}_${String(j).padStart(2, "0")}.webp`)));
  if (my !== cToken) return;
  if (els.canvas.width !== 512) { els.canvas.width = 512; els.canvas.height = 512; }
  // 重み付き平均：順に「これまでの合計に対する割合」で重ねる
  let acc = 0;
  ctx.globalAlpha = 1; ctx.clearRect(0, 0, 512, 512);
  cells.forEach(([, , w], k) => {
    if (w <= 0) return;
    acc += w;
    ctx.globalAlpha = w / acc;
    ctx.drawImage(imgs[k], 0, 0, 512, 512);
  });
  ctx.globalAlpha = 1;
  setStatus("");
}

// ---------- A：4つの latent を混ぜてデコード ----------
function getSession() {
  const name = st.style.decoder ?? "taesdxl";
  if (!st.sessions.has(name)) {
    st.sessions.set(name, (async () => {
      ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
      const order = params.get("ep") === "wasm" ? ["wasm"] : ("gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"]);
      let last;
      for (const ep of order) {
        try {
          setStatus(`デコーダを読み込み中（${ep}）…`);
          const s = await ort.InferenceSession.create(DECODERS[name], { executionProviders: [ep] });
          st.backend = ep; return s;
        } catch (e) { last = e; console.warn(ep, e); }
      }
      st.sessions.delete(name); throw last;
    })());
  }
  return st.sessions.get(name);
}
async function getLatent(cid) {
  const key = `${st.style.id}/${cid}`;
  if (!st.latents.has(key)) st.latents.set(key, new Float32Array(await (await fetch(`${base()}/latents/${cid}.f32`)).arrayBuffer()));
  return st.latents.get(key);
}
let imgData = null;
async function updateA() {
  if (st.busy) { st.pending = true; return; }
  st.busy = true;
  try {
    do {
      st.pending = false;
      const session = await getSession();
      const zs = await Promise.all(st.style.order.map(getLatent));
      const w = weights(), n = zs[0].length, z = new Float32Array(n);
      for (let k = 0; k < 4; k++) { const zk = zs[k], wk = w[k]; if (wk) for (let i = 0; i < n; i++) z[i] += wk * zk[i]; }
      const t0 = performance.now();
      const out = await session.run({ latent: new ort.Tensor("float32", z, st.style.latent_shape) });
      const ms = performance.now() - t0;
      const W = out.image.dims[3], H = out.image.dims[2], chw = out.image.data;
      if (els.canvas.width !== W || !imgData) { els.canvas.width = W; els.canvas.height = H; imgData = ctx.createImageData(W, H); }
      const d = imgData.data, m = W * H;
      for (let i = 0; i < m; i++) { d[i * 4] = chw[i] * 255; d[i * 4 + 1] = chw[m + i] * 255; d[i * 4 + 2] = chw[2 * m + i] * 255; d[i * 4 + 3] = 255; }
      ctx.putImageData(imgData, 0, 0);
      setStatus("");
      els.devinfo.textContent = `backend: ${st.backend} / decode: ${ms.toFixed(0)} ms`;
    } while (st.pending && st.mode === "a");
  } catch (e) {
    console.error(e); setStatus("デコードに失敗しました（Cタブをお使いください）");
  } finally { st.busy = false; }
}

function setStatus(s) { els.status.textContent = s; }
init().catch((e) => { console.error(e); setStatus("読み込みに失敗しました"); });
