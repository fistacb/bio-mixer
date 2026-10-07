// 「ことばのベクトル」：事前計算した結果（data/text/demo.json）を表示するだけ。ブラウザではモデルを動かさない
const $ = (id) => document.getElementById(id);
const pct = (p) => (p * 100).toFixed(p >= 0.1 ? 0 : 1) + "%";
let D, eMode = "natural", eState = [0, null];

function chips(el, items, onPick) {
  el.textContent = "";
  items.forEach((label, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = label;
    b.onclick = () => { el.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b)); onPick(i); };
    el.appendChild(b);
  });
  el.querySelector("button").click();
}

/* 1. 文章をさがす */
function searchBy(vec, selfIndex) {
  const rows = D.corpus.map((t, k) => ({ t, k, s: vec.reduce((a, v, i) => a + v * D.corpus_vec[k][i], 0) }))
    .filter((r) => r.k !== selfIndex).sort((a, b) => b.s - a.s);
  const ol = $("t-hits"); ol.textContent = "";
  rows.forEach((r, n) => {
    const li = document.createElement("li");
    li.className = n < 5 ? "top" : "";
    li.innerHTML = `<span class="score">${r.s.toFixed(3)}</span><span class="bar" style="width:${Math.max(0, r.s) * 100}%"></span><span class="txt">${r.t}</span>`;
    li.onclick = () => { $("t-query").textContent = `「${r.t}」`; $("t-chips").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", false)); searchBy(D.corpus_vec[r.k], r.k); };
    ol.appendChild(li);
  });
}

/* 2. BERT と 3. ELECTRA の文表示 */
function sentenceView(box, s, onWord, cls) {
  box.textContent = "";
  s.words.forEach((w, i) => {
    const b = document.createElement(s.clickable[i] ? "button" : "span");
    b.className = "w " + (cls ? cls(i) : "");
    b.textContent = w;
    if (s.clickable[i]) { b.type = "button"; b.onclick = () => onWord(i); }
    box.appendChild(b);
  });
}

function bert(si, wi) {
  const s = D.sentences[si];
  $("b-gloss").textContent = s.gloss;
  if (wi == null) wi = s.clickable.findIndex((c, i) => c && i > 0);
  sentenceView($("b-sent"), { ...s, words: s.words.map((w, i) => (i === wi ? "[MASK]" : w)) }, (i) => bert(si, i), (i) => (i === wi ? "mask" : ""));
  const pred = s.bert[wi] || [];
  const orig = s.words[wi];
  $("b-pred").innerHTML = pred.map(([t, p]) => `<li class="${t === orig ? "hit" : ""}"><span class="tok">${t}</span><span class="bar" style="width:${p * 100}%"></span><span class="p">${pct(p)}</span></li>`).join("");
  const rank = pred.findIndex(([t]) => t === orig);
  $("b-note").textContent = rank >= 0 ? `元の単語「${orig}」は ${rank + 1} 位でした。` : `元の単語「${orig}」は上位5件に入りませんでした。文としては他の候補も自然です。`;
}

function electra(si, wi) {
  eState = [si, wi];
  const s = D.sentences[si];
  $("e-gloss").textContent = s.gloss;
  const sw = wi == null ? null : s.electra.swaps[wi][eMode];
  const words = s.words.map((w, i) => (i === wi ? sw.word : w));
  const judge = sw ? sw.judge : s.electra.original;
  sentenceView($("e-sent"), { ...s, words }, (i) => electra(si, i), (i) => (i === wi ? "swapped" : ""));
  const mx = Math.max(...judge);
  [...$("e-sent").children].forEach((el, i) => {
    const p = judge[i];
    el.style.setProperty("--fake", (p / mx).toFixed(3)); // 文の中でいちばん怪しい単語を最も赤くする
    if (p === mx) el.classList.add("sus");
    el.title = `すり替えの確率 ${pct(p)}`;
    el.insertAdjacentHTML("beforeend", `<small>${pct(p)}</small>`);
  });
  if (!sw) { $("e-note").textContent = "すり替えていない元の文です。単語を押すと、生成役がその単語を別の単語にすり替えます。"; return; }
  const p = judge[wi], hit = p === mx;
  const sus = s.words.map((w, i) => (i === wi ? sw.word : w))[judge.indexOf(mx)];
  $("e-note").textContent = `「${s.words[wi]}」→「${sw.word}」にすり替えました。` +
    (hit ? `判定役は「${sw.word}」を文の中でいちばん怪しい（${pct(p)}）と判定しました。見抜けています。`
         : `判定役がいちばん怪しいとしたのは「${sus}」で、すり替えた「${sw.word}」は ${pct(p)} でした。見逃しています。`);
}

async function main() {
  D = await fetch("../../data/text/demo.json", { cache: "no-cache" }).then((r) => r.json());
  chips($("t-chips"), D.queries, (i) => { $("t-query").textContent = `「${D.queries[i]}」`; searchBy(D.query_vec[i], -1); });
  const labels = D.sentences.map((s) => s.gloss.replace(/。$/, ""));
  chips($("b-chips"), labels, (i) => bert(i));
  chips($("e-chips"), labels, (i) => electra(i, null));
  chips($("e-mode"), ["自然なすり替え", "不自然なすり替え"], (i) => { eMode = i ? "odd" : "natural"; electra(eState[0], eState[1]); });
  $("cmp").innerHTML = D.compare.map((c) => `<tr><td>${c.a}<br>${c.b}</td><td>${c.note}</td><td>${c.bert_raw.toFixed(3)}</td><td>${c.sentence_model.toFixed(3)}</td></tr>`).join("");
}
main().catch((e) => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<p class="note">読み込みに失敗しました：${e.message}</p>`); });
