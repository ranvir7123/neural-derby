// The stadium renders what the server tells it. It never trains anything.
(function () {
  const S = window.NeuralDerbySprites, A = window.NeuralDerbyAudio;
  const $ = (id) => document.getElementById(id);
  let LAPS_TOTAL = 20;     // replaced by the server's real lap count when a state message arrives
  let state = { act: "LOBBY" }, cards = [], cardByKey = {}, horses = [];
  let lanes = {};          // key -> {x, target, frame, colour, name}
  let lapNow = 0, verdictPage = 0, verdictData = null, factsData = null, muted = false;
  let toastTimer = null, previewRun = 0;

  // ---------- helpers ----------
  const pct = (f) => Math.round(f * 100) + "%";
  const fmt = (n) => n.toLocaleString("en-US");
  // Text typed on a phone (nicknames) is shown as plain text, never run as HTML.
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  function show(act) { document.querySelectorAll(".screen").forEach(s => s.classList.toggle("on", s.id === "s-" + act)); $("act").textContent = act; }
  function toast(msg, ms) {
    const t = $("toast"); t.textContent = msg; t.style.display = "block";
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.style.display = "none", ms || 2500);
  }
  // Never throws. A server that is down, or a reply that is not JSON (a crash page), comes back as
  // {error: ...}, so every operator key can show what went wrong instead of doing nothing.
  async function post(url, body) {
    let r;
    try { r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) }); }
    catch (e) { return { error: "CANNOT REACH THE SERVER ON THE LAPTOP." }; }
    let j;
    try { j = await r.json(); } catch (e) { return { error: `THE SERVER SENT A BROKEN REPLY (STATUS ${r.status}).` }; }
    if (!r.ok && !(j && j.error)) return { error: `THE SERVER REFUSED (STATUS ${r.status}).` };
    return j;
  }
  function say(text) { if (muted || !window.speechSynthesis) return; const u = new SpeechSynthesisUtterance(text); u.rate = 1.05; u.pitch = 1.1; speechSynthesis.speak(u); }
  function spriteCanvas(colour, scale) { const c = document.createElement("canvas"); c.width = S.W * scale; c.height = S.H * scale; S.drawHorse(c.getContext("2d"), 0, 0, scale, colour, 0); return c; }

  // ---------- state ----------
  function onState(m) {
    state = m; show(m.act);
    if (m.laps > 0) { LAPS_TOTAL = m.laps; $("lap").textContent = `LAP ${lapNow} / ${LAPS_TOTAL}`; }
    $("c-phones").textContent = m.phones + " PHONES"; $("c-photos").textContent = m.photos + " PHOTOS"; $("c-bets").textContent = m.bets + " BETS";
    $("photo-count").textContent = m.photos + " / 120";
    if (m.photos > 0) $("paddock-hint").textContent = m.photos_open ? "KEEP THEM COMING" : "GATES CLOSED";
    $("bets-banner").textContent = m.bets_open ? "PLACE YOUR BETS" : (m.data_loaded ? "GATES CLOSED. PRESS SPACE TO RACE" : "GATES CLOSED. LOADING PHOTOS...");
    if (m.act === "SHIP" && m.champion && cardByKey[m.champion]) $("ship-champ").textContent = cardByKey[m.champion].name;
  }
  // Like post(): a server that is down or a broken reply shows a toast, never a silent blank address.
  fetch("/api/join-url").then(r => r.json()).then(j => { if (!j || !j.url) throw new Error("no url"); $("join-url").textContent = j.url; $("join-url2").textContent = j.url; })
    .catch(() => toast("COULD NOT GET THE JOIN ADDRESS FROM THE SERVER. RELOAD THE PAGE.", 8000));
  $("qr").src = "/api/qr.png"; $("qr2").src = "/api/qr.png";

  // ---------- paddock ----------
  function onPhoto(m) {
    const wall = $("wall");
    const t = document.createElement("div"); t.className = "tile";
    t.innerHTML = `<img src="data:image/jpeg;base64,${m.thumb}"><div class="tag"><b>${m.class.toUpperCase()}</b>${esc(m.nick)}</div>`;
    wall.prepend(t); while (wall.children.length > 48) wall.lastChild.remove();   // 48 = what fits on the wall
    A.chime();
  }

  // ---------- multiply ----------
  function onPreviews(m) {
    const run = ++previewRun;   // a second copy of this message restarts the show; old timers then do nothing
    const wall = $("aug-wall"); wall.innerHTML = "";
    m.pairs.forEach((p, i) => {
      const t = document.createElement("div"); t.className = "tile";
      t.innerHTML = `<img src="data:image/png;base64,${p[0]}">`; wall.appendChild(t);
      setTimeout(() => { if (run !== previewRun) return; t.classList.add("flip"); t.querySelector("img").src = "data:image/png;base64," + p[1]; }, 600 + i * 120);
    });
    // Only the server's own numbers go on screen. n_before 0 means no phone photos yet: the wall shows the backup set.
    const caption = m.n_before > 0 ? `${fmt(m.n_before)} PHOTOS JUST BECAME ${fmt(m.n_after)}`
      : (m.pairs.length ? "NO PHONE PHOTOS YET: SHOWING THE BACKUP SET" : "NO PHOTOS TO SHOW YET");
    $("aug-caption").textContent = "MULTIPLYING...";
    setTimeout(() => { if (run !== previewRun) return; $("aug-caption").textContent = caption; A.jingle(); }, 600 + m.pairs.length * 120);
  }

  // ---------- horses ----------
  function onHorses(m) {
    cards = m.cards; cardByKey = {}; horses = cards.map(c => c.key);
    const maxP = Math.max(...cards.map(c => c.params)), maxM = Math.max(...cards.map(c => c.macs)), maxD = Math.max(...cards.map(c => c.depth));
    const el = $("cards"); el.innerHTML = "";
    cards.forEach(c => {
      cardByKey[c.key] = c;
      const d = document.createElement("div"); d.className = "card"; d.style.setProperty("--c", c.colour);
      const form = c.form.races ? `WINS ${c.form.race_wins}/${c.form.races}` : "NO FORM";
      d.innerHTML = `<div class="odds">${form}</div><div class="name">${c.name}</div><div class="block">${c.block.toUpperCase()}</div>
        <div class="tag">${c.tagline}</div>
        <div class="stat"><span>WEIGHT</span><div class="bar"><i style="width:${100 * c.params / maxP}%"></i></div></div>
        <div class="stat"><span>WORK</span><div class="bar"><i style="width:${100 * c.macs / maxM}%"></i></div></div>
        <div class="stat"><span>DEPTH</span><div class="bar"><i style="width:${100 * c.depth / maxD}%"></i></div></div>
        <div class="nums">${fmt(c.params)} PARAMS<br>${fmt(c.macs)} MULTIPLIES<br>${c.depth} LAYERS</div>`;
      d.insertBefore(spriteCanvas(c.colour, 5), d.querySelector(".block"));
      el.appendChild(d);
    });
    lanes = {}; cards.forEach(c => lanes[c.key] = { x: 0, target: 0, frame: 0, colour: c.colour, name: c.name });
    lapNow = 0; $("lap").textContent = `LAP 0 / ${LAPS_TOTAL}`; $("tote").innerHTML = ""; clearTicker();
  }

  // ---------- ticker ----------
  // One lap can bring several commentary lines at once. They wait their turn, so each one stays up
  // long enough to read. At most two wait; if more pile up, the oldest is dropped so the ticker keeps up.
  let tickerQueue = [], tickerTimer = null;
  function tick(line, speak) {
    tickerQueue.push({ line, speak }); if (tickerQueue.length > 2) tickerQueue.shift();
    if (!tickerTimer) nextLine();
  }
  function nextLine() {
    const next = tickerQueue.shift();
    if (!next) { tickerTimer = null; return; }
    $("ticker").textContent = next.line; if (next.speak) say(next.line);
    tickerTimer = setTimeout(nextLine, 2500);
  }
  function clearTicker() { tickerQueue = []; clearTimeout(tickerTimer); tickerTimer = null; $("ticker").textContent = ""; }

  // ---------- derby ----------
  const canvas = $("track"), ctx = canvas.getContext("2d");
  function drawTrack() {
    const W = canvas.width, H = canvas.height, top = 30, n = Math.max(horses.length, 1), laneH = Math.floor((H - top - 10) / n);
    const left = 340, right = W - 70, trackW = right - left, trackH = n * laneH;   // the names sit left of the start
    ctx.fillStyle = "#0d2a16"; ctx.fillRect(0, 0, W, H);
    ctx.textBaseline = "middle";
    // Lanes first, then the markers and the finish line on top, then the horses.
    horses.forEach((k, i) => { ctx.fillStyle = i % 2 ? "#0f3119" : "#0d2a16"; ctx.fillRect(0, top + i * laneH, W, laneH); });
    ctx.font = "12px PS2P";
    [0.25, 0.5, 0.75, 1.0].forEach(f => { const x = left + f * trackW; if (f < 1) { ctx.fillStyle = "#2f7a45"; ctx.fillRect(x - 1, top, 3, trackH); } ctx.fillStyle = "#8fa0ff"; ctx.fillText(pct(f), x - 18, top / 2); });
    for (let i = 0; i * 12 < trackH; i++) {   // checkered finish line, two squares wide
      const h = Math.min(12, trackH - i * 12);
      ctx.fillStyle = i % 2 ? "#000" : "#fff"; ctx.fillRect(right, top + i * 12, 6, h);
      ctx.fillStyle = i % 2 ? "#fff" : "#000"; ctx.fillRect(right + 6, top + i * 12, 6, h);
    }
    horses.forEach((k, i) => {
      const L = lanes[k]; if (!L) return;
      const mid = top + i * laneH + laneH / 2;
      ctx.font = "16px PS2P"; ctx.fillStyle = L.colour; ctx.fillText(L.name, 16, mid);
      L.x += (L.target - L.x) * 0.08;
      const scale = Math.max(3, Math.floor(laneH / 20)), nose = left + L.x * trackW;
      S.drawHorse(ctx, nose - S.W * scale, mid - S.H * scale / 2, scale, L.colour, L.frame);
      ctx.font = "14px PS2P"; ctx.fillStyle = "#fff"; ctx.fillText(pct(L.target), nose + 10, mid);
    });
  }
  setInterval(() => { Object.values(lanes).forEach(L => { if (Math.abs(L.target - L.x) > 0.002) L.frame = (L.frame + 1) % S.FRAMES; }); }, 150);
  (function loop() { if (state.act === "DERBY") drawTrack(); requestAnimationFrame(loop); })();

  function renderTote(standings) {
    const tote = $("tote"); tote.innerHTML = "";
    standings.forEach((s, i) => { const li = document.createElement("li"); if (i === 0) li.className = "lead";
      li.innerHTML = `<span>${s.rank}</span><span class="sw" style="background:${lanes[s.horse] ? lanes[s.horse].colour : "#fff"}"></span><span>${lanes[s.horse] ? lanes[s.horse].name : s.horse}</span><span class="acc">${pct(s.final_acc)}</span>`; tote.appendChild(li); });
  }
  function onLap(m) {
    const e = m.event, L = lanes[e.horse]; if (L && e.test_acc != null) L.target = e.test_acc;
    if (e.lap !== lapNow) { lapNow = e.lap; $("lap").textContent = `LAP ${e.lap} / ${LAPS_TOTAL}`; if (e.lap === 1) A.gun(); else A.chime(); }
    renderTote(m.standings);
    m.lines.forEach(line => tick(line, true));
  }
  function onRaceDone(m) {
    // After a reload the laps are not sent again, so draw the finish from the final standings.
    const st = m.standings || [];
    st.forEach(s => { const L = lanes[s.horse]; if (L && s.final_acc != null) L.target = s.final_acc; });
    renderTote(st);
    if (lapNow === 0) { lapNow = LAPS_TOTAL; $("lap").textContent = `LAP ${LAPS_TOTAL} / ${LAPS_TOTAL}`; }
    if (state.act === "DERBY") A.roar();
    const w = st[0], name = w ? (lanes[w.horse] ? lanes[w.horse].name : w.horse) : null;
    tick(name ? `CHECKERED FLAG. ${name} WINS AT ${pct(w.final_acc)}. PRESS SPACE FOR THE SPRINT.` : "CHECKERED FLAG. PRESS SPACE FOR THE SPRINT.", false);
  }

  // ---------- sprint ----------
  function onSprint(m) {
    $("sprint-wait").style.display = "none";
    const bars = $("bars"); bars.innerHTML = ""; const maxMs = Math.max(...m.sprint.map(s => s.ms));
    m.sprint.forEach(s => { const c = cardByKey[s.horse] || { name: s.horse, colour: "#fff" };
      const row = document.createElement("div"); row.className = "brow";
      // under 10 ms one decimal, so a fast horse never shows as "0 ms"
      row.innerHTML = `<div class="nm" style="color:${c.colour}">${c.name}</div><div class="bb"><i style="width:0;background:${c.colour}"></i></div><div class="ms">${s.ms.toFixed(s.ms < 10 ? 1 : 0)} ms</div>`;
      bars.appendChild(row); });
    void bars.offsetWidth;   // lay the bars out at zero width first, so the growth below animates
    [...bars.querySelectorAll(".brow i")].forEach((bar, k) => { bar.style.width = (100 * m.sprint[k].ms / maxMs) + "%"; });
    const pod = $("podium"); pod.innerHTML = "";
    [1, 0, 2].forEach(i => { const c = m.champion[i]; if (!c) return; const card = cardByKey[c.horse] || { name: c.horse, colour: "#fff" };
      const d = document.createElement("div"); d.className = "pod p" + (i + 1); d.style.setProperty("--c", card.colour);
      d.innerHTML = `<div class="pname">${card.name}</div><div class="box"><b>${i === 0 ? "CHAMPION" : (i + 1) + (i === 1 ? "ND" : "RD")}</b>RACE ${c.race_rank} +<br>SPRINT ${c.sprint_rank}</div>`;
      pod.appendChild(d); });
    const w = m.winners;
    // at most 12 names, so a big class still fits on the screen
    const names = (list) => !list.length ? "NOBODY" : list.slice(0, 12).map(esc).join(", ") + (list.length > 12 ? ` +${list.length - 12} MORE` : "");
    $("winners").innerHTML = `TIMED ON: <b>${esc(m.device.toUpperCase())}</b><br>PICKED THE CHAMPION: <b>${names(w.champion)}</b><br>PICKED THE RACE WINNER: <b>${names(w.race)}</b>`;
    if (state.act === "SPRINT") { A.jingle(); say(`The champion is ${(cardByKey[m.champion[0].horse] || {}).name || m.champion[0].horse}.`); }
  }

  // ---------- verdict ----------
  function renderVerdict() {
    if (!verdictData) return;
    const keys = horses.length ? horses : Object.keys(verdictData);
    const el = $("verdict"); el.innerHTML = "";
    [keys[verdictPage * 2], keys[verdictPage * 2 + 1]].forEach(k => {
      if (!k) return; const c = cardByKey[k] || { name: k, colour: "#fff", table: [] };
      const d = document.createElement("div"); d.className = "vcard"; d.style.setProperty("--c", c.colour);
      const table = c.table || [];
      const rows = table.slice(0, 10).map(r => `<tr><td>${r.layer}</td><td>${r.type}</td><td>${r.out_shape.join("x")}</td><td>${fmt(r.params)}</td></tr>`).join("") +
        (table.length > 10 ? `<tr class="more"><td colspan="4">+ ${table.length - 10} MORE LAYERS</td></tr>` : "");   // say the table is cut short
      d.innerHTML = `<div class="name">${c.name}</div>` + (verdictData[k] || []).map(s => `<p>${s}</p>`).join("") +
        `<table class="vtable"><tr><th>LAYER</th><th>TYPE</th><th>OUT</th><th>PARAMS</th></tr>${rows}</table>`;
      el.appendChild(d);
    });
    const pages = Math.max(1, Math.ceil(keys.length / 2));
    $("verdict-hint").textContent = `PAGE ${verdictPage + 1} / ${pages}. LEFT / RIGHT TO PAGE THROUGH THE HORSES`;
  }
  function onVerdict(m) { verdictData = m.verdicts; factsData = m.facts; verdictPage = 0; renderVerdict(); }

  // ---------- ship ----------
  function onShip(m) { if (m.champion && cardByKey[m.champion]) $("ship-champ").textContent = cardByKey[m.champion].name; if (state.act === "SHIP") A.jingle(); }
  function onVeteran(m) { $("veteran").innerHTML = `THE VETERAN (PRETRAINED MOBILENET): <b>${pct(m.acc)}</b> AFTER ONE LAP<br>BUT <b>${m.ms.toFixed(0)} ms</b> PER PHOTO ON ${esc(m.device.toUpperCase())} VS THE CHAMPION'S <b>${m.champion_ms.toFixed(0)} ms</b><br>IT HAD SEEN A MILLION PHOTOS BEFORE TODAY. THAT IS TRANSFER LEARNING. TOO HEAVY FOR THE PHONE: THAT IS EDGE AI.`; }

  // ---------- errors ----------
  // Something broke on the laptop (loading the photos, or the race). The operator has to act, so it stays up longer.
  // A race failure happens in DERBY, where R replays. A failed photo load cannot be recovered without a server restart.
  function onError(m) {
    toast(m.where === "race" ? `THE RACE FAILED: ${m.message}. PRESS R TO REPLAY THE LAST RACE.`
                             : `LOADING THE PHOTOS FAILED: ${m.message}. RESTART THE SERVER.`, 8000);
  }

  // ---------- socket ----------
  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?role=stadium`);
    ws.onmessage = (ev) => { const m = JSON.parse(ev.data);
      ({ state: onState, photo: onPhoto, previews: onPreviews, horses: onHorses, lap: onLap, race_done: onRaceDone, sprint: onSprint, verdict: onVerdict, ship: onShip, veteran: onVeteran, error: onError }[m.type] || (() => {}))(m); };
    ws.onclose = () => setTimeout(connect, 3000);
  }
  connect();

  // ---------- keys ----------
  document.addEventListener("keydown", async (e) => {
    // A held key must not fire again and skip acts. Browser shortcuts (Ctrl+R reload, Ctrl+L address bar,
    // Alt+Left back...) belong to the browser, never to the show.
    if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    A.unlock();
    const send = async (url) => { const r = await post(url); if (r.error) toast(r.error); };
    if (e.code === "Space") { e.preventDefault(); await send("/api/advance"); }
    else if (e.key === "g" || e.key === "G") await send("/api/gate/close");
    else if (e.key === "l" || e.key === "L") await send("/api/sprint/laptop");
    // A replay wipes the Sprint, verdict and ship results, so a stray R in a later act must not reach the server.
    else if (e.key === "r" || e.key === "R") { if (state.act === "DERBY") await send("/api/replay"); else toast("R REPLAYS THE RACE. IT ONLY WORKS DURING THE DERBY."); }
    else if (e.key === "m" || e.key === "M") { muted = A.toggleMute(); toast(muted ? "MUTED" : "SOUND ON"); }
    else if (e.key === "ArrowRight" && state.act === "VERDICT") { verdictPage = Math.min(verdictPage + 1, Math.ceil(horses.length / 2) - 1); renderVerdict(); }
    else if (e.key === "ArrowLeft" && state.act === "VERDICT") { verdictPage = Math.max(verdictPage - 1, 0); renderVerdict(); }
  });
})();
