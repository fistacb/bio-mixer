// 生物ミキサー
// C: 事前生成フレームを切り替え（隣接2枚をクロスフェード）
// A: 事前計算した UNet latent を補間し、TAESD デコーダ(ONNX)でブラウザ上でデコード
const ORT_VER = "1.30.0";
const params = new URLSearchParams(location.search);
// スタイルごとのデコーダ（SD系=TAESD、SDXL系=TAESDXL）
// ページごとの置き場所（別ページからは window.BIO_MIXER で上書き）
const CONF = Object.assign({ root: "", data: "data/" }, window.BIO_MIXER || {});
const DECODERS = { taesd: `${CONF.root}models/taesd_decoder.onnx`, taesdxl: `${CONF.root}models/taesdxl_decoder.onnx` };
const latentShape = () => st.style.latent_shape ?? [1, 4, 64, 64];
const latentSize = () => latentShape().reduce((x, y) => x * y, 1);

const st = {
  index: null, style: null, mode: "c", a: "dog", b: "octopus",
  t: 0.5, interp: "lerp",
  latents: new Map(),      // `${style}/${id}` -> Float32Array
  sessions: new Map(),     // decoder名 -> Promise<InferenceSession>
  backend: "",
  busy: false, pending: false,
};

const $ = (id) => document.getElementById(id);
const els = {
  style: $("style"), pickA: $("pickA"), pickB: $("pickB"), t: $("t"),
  nameA: $("nameA"), nameB: $("nameB"), frameC: $("frameC"), img0: $("imgC0"), img1: $("imgC1"),
  canvas: $("frameA"), status: $("status"), dev: $("dev"), devinfo: $("devinfo"),
};
const ctx = els.canvas.getContext("2d");
let imgData = null;

const base = () => `${CONF.data}styles/${st.style.id}`;
const nameOf = (id) => st.index.creatures.find((c) => c.id === id)?.name ?? id;

async function init() {
  // スタイル一覧は更新されるので、毎回サーバに確認する（GitHub Pages のキャッシュ対策）
  st.index = await (await fetch(`${CONF.data}index.json`, { cache: "no-cache" })).json();
  // 初期の組み合わせ：ページの生物セットにない場合は最初と最後の生物にする
  const ids = st.index.creatures.map((c) => c.id);
  if (!ids.includes(st.a)) st.a = ids[0];
  if (!ids.includes(st.b)) st.b = ids[ids.length - 1];
  for (const s of st.index.styles) els.style.add(new Option(s.name, s.id));
  const pref = st.index.styles.find((s) => s.id === params.get("style"))
    ?? st.index.styles.find((s) => s.id === "anime_xl") ?? st.index.styles[0];
  els.style.value = pref.id;
  st.style = pref;

  els.style.onchange = () => { st.style = st.index.styles.find((s) => s.id === els.style.value); renderPickers(); update(); };
  els.t.oninput = () => { st.t = els.t.value / 1000; update(); };
  document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  document.querySelectorAll("input[name=interp]").forEach((r) => (r.onchange = () => { st.interp = r.value; update(); }));
  if (params.has("dev")) els.dev.hidden = false;

  renderPickers();
  setMode(params.get("mode") === "a" ? "a" : "c");
}

function renderPickers() {
  for (const [el, key] of [[els.pickA, "a"], [els.pickB, "b"]]) {
    el.innerHTML = "";
    for (const id of st.style.creatures) {
      const btn = document.createElement("button");
      btn.setAttribute("aria-pressed", String(st[key] === id));
      btn.innerHTML = `<img src="${base()}/catalog/${id}.webp" alt=""><span>${nameOf(id)}</span>`;
      btn.onclick = () => { st[key] = id; renderPickers(); update(); };
      el.appendChild(btn);
    }
  }
  els.nameA.textContent = nameOf(st.a);
  els.nameB.textContent = nameOf(st.b);
}

function setMode(mode) {
  st.mode = mode;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  els.frameC.hidden = mode !== "c";
  els.canvas.hidden = mode !== "a";
  update();
}

function update() {
  if (st.mode === "c") updateC(); else updateA();
}

// ---------- C: 事前生成 ----------
function framePath(a, b, i) {
  return `${base()}/frames/${a}__${b}/${String(i).padStart(2, "0")}.webp`;
}
function updateC() {
  setStatus("");
  const { a, b } = st;
  if (a === b) {
    show(els.img0, `${base()}/catalog/${a}.webp`, 1); els.img1.style.opacity = 0; return;
  }
  // フレームは辞書順 (x < y) で保存されている
  const [x, y, t] = a < b ? [a, b, st.t] : [b, a, 1 - st.t];
  const pos = t * (st.style.frames - 1);
  const i0 = Math.floor(pos), i1 = Math.min(i0 + 1, st.style.frames - 1), f = pos - i0;
  show(els.img0, framePath(x, y, i0), 1);
  show(els.img1, framePath(x, y, i1), f);
  preloadPair(x, y);
}
function show(img, src, opacity) {
  if (!img.src.endsWith(src)) img.src = src;
  img.style.opacity = opacity;
}
const preloaded = new Set();
function preloadPair(x, y) {
  const key = `${st.style.id}/${x}__${y}`;
  if (preloaded.has(key)) return;
  preloaded.add(key);
  for (let i = 0; i < st.style.frames; i++) new Image().src = framePath(x, y, i);
}

// ---------- A: ブラウザでデコード ----------
function getSession() {
  const name = st.style.decoder ?? "taesd";
  if (!st.sessions.has(name)) {
    st.sessions.set(name, (async () => {
      ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
      const order = params.get("ep") === "wasm" ? ["wasm"] : ("gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"]);
      let lastErr;
      for (const ep of order) {
        try {
          setStatus(`デコーダを読み込み中（${ep}）…`);
          const s = await ort.InferenceSession.create(DECODERS[name], { executionProviders: [ep] });
          st.backend = ep;
          return s;
        } catch (e) { lastErr = e; console.warn(ep, e); }
      }
      st.sessions.delete(name);
      throw lastErr;
    })());
  }
  return st.sessions.get(name);
}

async function getLatent(id) {
  const key = `${st.style.id}/${id}`;
  if (!st.latents.has(key)) {
    const buf = await (await fetch(`${base()}/latents/${id}.f32`)).arrayBuffer();
    st.latents.set(key, new Float32Array(buf));
  }
  return st.latents.get(key);
}

function mix(za, zb, t, mode) {
  const LATENT_SIZE = za.length;
  const out = new Float32Array(LATENT_SIZE);
  if (mode === "slerp") {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < LATENT_SIZE; i++) { dot += za[i] * zb[i]; na += za[i] * za[i]; nb += zb[i] * zb[i]; }
    const cos = Math.min(1, Math.max(-1, dot / Math.sqrt(na * nb)));
    const th = Math.acos(cos);
    if (th > 1e-4) {
      const s = Math.sin(th), wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
      for (let i = 0; i < LATENT_SIZE; i++) out[i] = wa * za[i] + wb * zb[i];
      return out;
    }
  }
  for (let i = 0; i < LATENT_SIZE; i++) out[i] = (1 - t) * za[i] + t * zb[i];
  return out;
}

// 最新のスライダー値だけを処理する（デコード中に来た要求はまとめる）
async function updateA() {
  if (st.busy) { st.pending = true; return; }
  st.busy = true;
  try {
    do {
      st.pending = false;
      const session = await getSession();
      const [za, zb] = await Promise.all([getLatent(st.a), getLatent(st.b)]);
      const z = mix(za, zb, st.t, st.interp);
      const t0 = performance.now();
      if (z.length !== latentSize()) throw new Error(`latent size mismatch: ${z.length} != ${latentSize()}`);
      const out = await session.run({ latent: new ort.Tensor("float32", z, latentShape()) });
      const ms = performance.now() - t0;
      draw(out.image.data, out.image.dims[3], out.image.dims[2]);
      setStatus("");
      els.devinfo.textContent = `backend: ${st.backend} / decode: ${ms.toFixed(0)} ms / interp: ${st.interp}`;
    } while (st.pending && st.mode === "a");
  } catch (e) {
    console.error(e);
    setStatus("デコードに失敗しました（Cタブをお使いください）");
  } finally {
    st.busy = false;
  }
}

function draw(chw, w, h) {
  if (els.canvas.width !== w || els.canvas.height !== h || !imgData) {
    els.canvas.width = w; els.canvas.height = h;
    imgData = ctx.createImageData(w, h);
  }
  const n = w * h, d = imgData.data;
  for (let i = 0; i < n; i++) {
    d[i * 4] = chw[i] * 255;
    d[i * 4 + 1] = chw[n + i] * 255;
    d[i * 4 + 2] = chw[2 * n + i] * 255;
    d[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}


function setStatus(s) { els.status.textContent = s; }

init().catch((e) => { console.error(e); setStatus("読み込みに失敗しました"); });
