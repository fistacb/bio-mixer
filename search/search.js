// 「さがす」：事前計算したベクトル（SigLIP、長さ1に正規化済み）を、ブラウザで内積だけ計算して並べ替える
const DATA = "../data/";
const TOPK = 12;
const st = { meta: null, vec: null, dim: 0, n: 0, filter: "all", query: null };
const $ = (id) => document.getElementById(id);

function half(u) { // float16 → float32
  const s = u & 0x8000 ? -1 : 1, e = (u >> 10) & 0x1f, f = u & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

async function load() {
  const [meta, buf] = await Promise.all([
    fetch(DATA + "search/items.json", { cache: "no-cache" }).then((r) => r.json()),
    fetch(DATA + "search/vectors.f16", { cache: "no-cache" }).then((r) => r.arrayBuffer()),
  ]);
  const u = new Uint16Array(buf), v = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) v[i] = half(u[i]);
  Object.assign(st, { meta, vec: v, dim: meta.dim, n: meta.images.length });
  $("count").textContent = `${st.n}枚の画像と${meta.queries.length}個の検索語`;
  buildFilters(); buildGallery(); buildChips();
  pickQuery({ kind: "text", index: 0 });
}

const row = (k) => st.vec.subarray(k * st.dim, (k + 1) * st.dim);
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
const label = (it) => `${it.page}／${it.style}`;
const src = (it) => DATA + it.src;

function buildFilters() {
  const pages = [...new Set(st.meta.images.map((it) => it.page))];
  const sel = $("filter");
  sel.add(new Option("すべてのページ", "all"));
  pages.forEach((p) => sel.add(new Option(p, p)));
  sel.onchange = () => { st.filter = sel.value; buildGallery(); };
}

function buildGallery() {
  const g = $("gallery"); g.textContent = "";
  st.meta.images.forEach((it, k) => {
    if (st.filter !== "all" && it.page !== st.filter) return;
    const b = document.createElement("button");
    b.type = "button"; b.title = label(it);
    b.innerHTML = `<img loading="lazy" src="${src(it)}" alt="">`;
    b.onclick = () => { pickQuery({ kind: "image", index: k }); $("result").scrollIntoView({ behavior: "smooth", block: "start" }); };
    g.appendChild(b);
  });
}

function buildChips() {
  const c = $("chips");
  st.meta.queries.forEach((q, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = q.label; b.dataset.i = i;
    b.onclick = () => pickQuery({ kind: "text", index: i });
    c.appendChild(b);
  });
}

function pickQuery(q) {
  st.query = q;
  document.querySelectorAll("#chips button").forEach((b) => b.setAttribute("aria-pressed", q.kind === "text" && +b.dataset.i === q.index));
  const qv = q.kind === "text" ? row(st.n + q.index) : row(q.index);
  const scores = new Float32Array(st.n);
  for (let k = 0; k < st.n; k++) scores[k] = dot(qv, row(k));
  const order = [...scores.keys()].filter((k) => !(q.kind === "image" && k === q.index)).sort((a, b) => scores[b] - scores[a]);

  const qbox = $("qbox");
  if (q.kind === "text") {
    const t = st.meta.queries[q.index];
    qbox.innerHTML = `<div class="qword">「${t.label}」</div><p class="note">モデルには英語で「${t.text}」として渡しています</p>`;
  } else {
    const it = st.meta.images[q.index];
    qbox.innerHTML = `<img src="${src(it)}" alt=""><p class="note">${label(it)}</p>`;
  }

  const top = order.slice(0, TOPK);
  const best = scores[top[0]];
  // 「ほぼ同点」は、全体のスコアの幅（最高〜最低）の 5% 以内とする。言葉と画像ではスコアの桁が違うため、固定値にしない
  const worst = scores[order[order.length - 1]], tol = 0.05 * (best - worst);
  const close = order.filter((k) => best - scores[k] <= tol).length;
  $("summary").innerHTML = `いちばん似ている画像のスコアは <b>${best.toFixed(3)}</b>（全体の最低は ${worst.toFixed(3)}）。1位とほぼ同点（差が全体の幅の5%以内）の画像が <b>${close}枚</b> あります。`;

  const list = $("hits"); list.textContent = "";
  top.forEach((k, r) => {
    const it = st.meta.images[k];
    const li = document.createElement("li");
    li.innerHTML = `<img loading="lazy" src="${src(it)}" alt=""><span class="rank">${r + 1}</span><span class="score">${scores[k].toFixed(3)}</span><span class="where">${label(it)}</span>`;
    li.onclick = () => pickQuery({ kind: "image", index: k });
    list.appendChild(li);
  });
  drawHist(order.map((k) => scores[k]), scores[top[top.length - 1]]);
}

function drawHist(vals, cut) {
  const W = 640, H = 150, P = 28, bins = 40;
  const lo = Math.min(...vals), hi = Math.max(...vals), w = (hi - lo) / bins || 1;
  const cnt = new Array(bins).fill(0);
  vals.forEach((v) => cnt[Math.min(bins - 1, Math.floor((v - lo) / w))]++);
  const mx = Math.max(...cnt), bw = (W - 2 * P) / bins;
  const x = (v) => P + ((v - lo) / (hi - lo || 1)) * (W - 2 * P);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="全画像のスコアの分布">`;
  cnt.forEach((c, i) => {
    const h = (c / mx) * (H - 2 * P), x0 = P + i * bw, hot = lo + (i + 1) * w > cut;
    s += `<rect x="${x0 + 1}" y="${H - P - h}" width="${bw - 2}" height="${h}" class="${hot ? "bar hot" : "bar"}"><title>${c}枚</title></rect>`;
  });
  s += `<line x1="${x(cut)}" x2="${x(cut)}" y1="${P - 10}" y2="${H - P}" class="cut"/>`;
  s += `<text x="${x(cut)}" y="${P - 14}" text-anchor="middle" class="lab">上位${TOPK}件の境目</text>`;
  s += `<text x="${P}" y="${H - 8}" class="lab">${lo.toFixed(2)}</text><text x="${W - P}" y="${H - 8}" text-anchor="end" class="lab">${hi.toFixed(2)}</text>`;
  s += `<text x="${W / 2}" y="${H - 8}" text-anchor="middle" class="lab">← 似ていない　スコア　似ている →</text></svg>`;
  $("hist").innerHTML = s;
}

load().catch((e) => { $("count").textContent = "読み込みに失敗しました：" + e.message; });
