// The phone page: join, shoot photos, bet, follow the race, then run the champion on the phone itself.
(function () {
  const $ = (id) => document.getElementById(id);
  // The team phone link carries a secret token: /phone/?team=<token>. No token means an ordinary crowd phone.
  const TEAM_TOKEN = new URLSearchParams(location.search).get("team") || "";
  const TEAM = TEAM_TOKEN !== "";
  const CLASSES = ["bottle", "pen", "book", "phone", "backpack", "shoe"];
  const device = localStorage.getItem("derby_device") || (() => { const d = Array.from(crypto.getRandomValues(new Uint8Array(8)), b => b.toString(16).padStart(2, "0")).join(""); localStorage.setItem("derby_device", d); return d; })();
  let nick = localStorage.getItem("derby_nick") || "";
  let state = { act: "LOBBY", laps: 20 }, cards = [], myBet = localStorage.getItem("derby_bet") || null, sent = 0, chosenClass = null, shotFile = null, session = null;
  let shipped = null, liveStarted = false, timingRun = 0;

  // Text that came from the server is shown as plain text, never run as HTML.
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  function show(id) { document.querySelectorAll(".screen").forEach(s => s.classList.toggle("on", s.id === id)); }
  function banner(text, ok) { const b = $("banner"); b.textContent = text; b.className = "banner" + (ok ? " ok" : ""); }
  // Always resolves to an object: a dropped connection or an answer that is not JSON becomes {error}.
  function post(url, body) {
    return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then(r => r.json()).catch(() => ({ error: "No connection. Try again." }));
  }

  // ---------- routing by act ----------
  function route() {
    if (!nick) { show("p-join"); return; }
    const a = state.act;
    if (a === "PADDOCK" || a === "MULTIPLY") forgetBet();   // a bet kept from an earlier show does not count
    if (a === "LOBBY") { newShow(); $("wait-title").textContent = "HOLD ON"; $("wait-text").textContent = "The paddock opens soon. Look at the big screen."; show("p-wait"); }
    else if (a === "PADDOCK") { if (state.photos_open && sent < 3) show("p-capture"); else { $("wait-title").textContent = sent >= 3 ? "THANK YOU" : "GATES CLOSED"; $("wait-text").textContent = "Your photos are in. Betting opens next."; show("p-wait"); } }
    else if (a === "MULTIPLY") { $("wait-title").textContent = "AUGMENTING"; $("wait-text").textContent = "Your photos are being spun, darkened, and multiplied."; show("p-wait"); }
    else if (a === "HORSES") { if (state.bets_open) show("p-bet"); else { $("wait-title").textContent = myBet ? "BET LOCKED" : "GATES CLOSED"; $("wait-text").textContent = myBet ? "You picked " + nameOf(myBet) + ". Watch the big screen." : "No bet this time. Watch the race!"; show("p-wait"); } }
    else if (a === "DERBY") show("p-race");
    else if (a === "SPRINT" || a === "VERDICT") { if (!$("result-title").textContent.trim() || $("result-title").textContent === "...") { $("result-title").textContent = "TIMING ON THE PHONE"; $("result-text").textContent = "The sprint decides the champion."; } show("p-result"); }
    else if (a === "SHIP") show("p-ship");
  }
  function nameOf(key) { if (!key) return "THE CHAMPION"; const c = cards.find(c => c.key === key); return c ? c.name : String(key).toUpperCase(); }
  function forgetBet() { if (myBet) { myBet = null; localStorage.removeItem("derby_bet"); } }
  // A page left open across a server restart must not carry the last show's photos, result or model.
  function newShow() { forgetBet(); sent = 0; $("sent-count").textContent = "0 of 3 sent"; $("result-title").textContent = "..."; shipped = null; }

  // ---------- join ----------
  $("nick").value = nick;
  if (nick) show("p-wait");   // a returning phone waits for the show instead of flashing the name screen
  $("join-btn").onclick = async () => {
    const typed = $("nick").value.trim().slice(0, 12) || "anon";
    const r = await post("/api/join", { device, nick: typed });
    if (r.error) { banner(r.error); return; }
    nick = typed; localStorage.setItem("derby_nick", nick); route();
  };

  // ---------- capture ----------
  const grid = $("class-grid");
  CLASSES.forEach(c => { const b = document.createElement("button"); b.textContent = c.toUpperCase(); b.onclick = () => { chosenClass = c; grid.querySelectorAll("button").forEach(x => x.classList.toggle("sel", x === b)); $("shot").click(); }; grid.appendChild(b); });
  $("shot").onchange = () => { shotFile = $("shot").files[0]; if (!shotFile) return; $("preview").src = URL.createObjectURL(shotFile); $("preview-box").hidden = false; };
  // On a short screen RETAKE and SEND sit below the fold once the photo has its height: bring them into view.
  $("preview").onload = () => $("preview-box").scrollIntoView({ block: "end", behavior: "smooth" });
  $("retake").onclick = () => { $("preview-box").hidden = true; $("shot").value = ""; $("shot").click(); };
  $("send").onclick = async () => {
    if (!shotFile || !chosenClass) return;
    const fd = new FormData(); fd.append("device", device); fd.append("class_name", chosenClass); fd.append("file", shotFile, "photo.jpg");
    $("send").disabled = true;
    const r = await fetch("/api/photo", { method: "POST", body: fd }).then(r => r.json()).catch(() => ({ error: "No connection. Try again." }));
    $("send").disabled = false;
    if (r.error) { banner(r.error); return; }
    sent = r.count; $("sent-count").textContent = `${sent} of 3 sent`; banner("SENT! " + Math.max(0, 3 - sent) + " left", true);
    $("preview-box").hidden = true; $("shot").value = ""; shotFile = null;
    if (sent >= 3) route();
  };

  // ---------- bet ----------
  function renderBets() {
    const el = $("bet-list"); el.innerHTML = "";
    cards.forEach(c => { const d = document.createElement("div"); d.className = "bet" + (myBet === c.key ? " sel" : ""); d.style.setProperty("--c", c.colour);
      const form = c.form.races ? "wins " + c.form.race_wins + "/" + c.form.races : "no form";
      d.innerHTML = `<div><div class="nm">${esc(c.name)}</div><div class="tg">${esc(c.tagline)}</div><div class="tg">${c.params.toLocaleString("en-US")} params · ${form}</div></div>`;
      d.onclick = async () => { const r = await post("/api/bet", { device, horse: c.key }); if (r.error) { banner(r.error); return; } myBet = c.key; localStorage.setItem("derby_bet", myBet); banner("BET LOCKED ON " + c.name, true); renderBets(); };
      el.appendChild(d); });
  }

  // ---------- race ----------
  // Laps are not sent again after a reload, so the race screen is always redrawn from the whole standings.
  function renderStandings(lap, standings) {
    $("race-title").textContent = `LAP ${lap} / ${state.laps || "?"}`;
    const ol = $("mini"); ol.innerHTML = "";
    standings.forEach(s => { const li = document.createElement("li");
      const nm = document.createElement("span"); nm.style.color = (cards.find(c => c.key === s.horse) || {}).colour || "#fff"; nm.textContent = `${s.rank}. ${nameOf(s.horse)}`;
      const acc = document.createElement("span"); acc.textContent = s.final_acc == null ? "--" : Math.round(s.final_acc * 100) + "%";
      li.append(nm, acc); ol.appendChild(li); });
    $("your-horse").textContent = "";
    if (myBet) { const s = standings.find(s => s.horse === myBet); if (s) $("your-horse").textContent = `YOUR HORSE ${nameOf(myBet)} IS ${s.rank}${["st", "nd", "rd"][s.rank - 1] || "th"}`; }
  }
  function onLap(m) { renderStandings(m.event.lap, m.standings); }
  function onRaceDone(m) { renderStandings(state.laps || "?", m.standings || []); }
  function onSprint(m) {
    timingRun++;   // the Sprint is decided: a timing loop still running on the team phone stops
    const mine = m.bet_results ? m.bet_results[device] : undefined;   // the server's record of this phone's bet
    $("result-title").textContent = mine === undefined ? "THE CHAMPION" : (mine ? "YOU WON!" : "BETTER LUCK NEXT RACE");
    $("result-text").textContent = `Champion: ${nameOf(m.champion[0] && m.champion[0].horse)}. Timed on the ${m.device}.` + (myBet ? ` You picked ${nameOf(myBet)}.` : "");
    if (state.act === "SPRINT" || state.act === "VERDICT") show("p-result");
  }

  // ---------- inference on the phone ----------
  const Infer = {
    async load(url) { ort.env.wasm.wasmPaths = "/vendor/onnxruntime-web/"; ort.env.wasm.numThreads = 1; return ort.InferenceSession.create(url, { executionProviders: ["wasm"] }); },
    tensorFromCanvas(canvas) {
      const { data } = canvas.getContext("2d").getImageData(0, 0, 96, 96);
      const f = new Float32Array(3 * 96 * 96);
      for (let i = 0; i < 96 * 96; i++) { for (let c = 0; c < 3; c++) f[c * 96 * 96 + i] = (data[i * 4 + c] / 255 - 0.5) / 0.5; }
      return new ort.Tensor("float32", f, [1, 3, 96, 96]);
    },
    drawSquare(source, sw, sh, canvas) { const s = Math.min(sw, sh); canvas.getContext("2d").drawImage(source, (sw - s) / 2, (sh - s) / 2, s, s, 0, 0, 96, 96); },
    async classify(sess, canvas) {
      const t0 = performance.now(); const out = await sess.run({ image: Infer.tensorFromCanvas(canvas) }); const ms = performance.now() - t0;
      const l = Array.from(out.logits.data); const m = Math.max(...l); const e = l.map(v => Math.exp(v - m)); const z = e.reduce((a, b) => a + b, 0);
      return { probs: CLASSES.map((label, i) => ({ label, p: e[i] / z })).sort((a, b) => b.p - a.p), ms };
    },
    async timeModel(url, n) {
      const sess = await Infer.load(url); const t = new ort.Tensor("float32", new Float32Array(3 * 96 * 96), [1, 3, 96, 96]);
      try {
        for (let i = 0; i < 3; i++) await sess.run({ image: t });
        const times = []; for (let i = 0; i < n; i++) { const a = performance.now(); await sess.run({ image: t }); times.push(performance.now() - a); }
        times.sort((a, b) => a - b); return times[Math.floor(times.length / 2)];
      } finally { await sess.release(); }   // seven models in a row would otherwise pile up in the phone's memory
    },
  };
  window.NeuralDerbyInfer = Infer;

  function showProbs(r) {
    const el = $("bars"); el.innerHTML = "";
    r.probs.slice(0, 3).forEach(p => { const d = document.createElement("div"); d.className = "b"; d.innerHTML = `<div class="lab"><span>${p.label.toUpperCase()}</span><span>${Math.round(p.p * 100)}%</span></div><div class="bar"><i style="width:${p.p * 100}%"></i></div>`; el.appendChild(d); });
    $("infer-ms").textContent = `${r.ms.toFixed(0)} ms on this phone, no laptop involved`;
  }
  async function classifyCanvas() {
    try { showProbs(await Infer.classify(session, $("work"))); }
    catch (e) { $("ship-status").textContent = "could not run the model: " + e; }
  }
  async function onShip(m) {
    show("p-ship");
    const which = m.url + "#" + m.champion;
    if (session && shipped === which) return;   // the same message again after a reconnect: the model is already here
    $("ship-status").textContent = "downloading " + nameOf(m.champion) + "...";
    try { session = await Infer.load(m.url); } catch (e) { $("ship-status").textContent = "could not load the model: " + e; return; }
    shipped = which;
    $("ship-status").textContent = nameOf(m.champion) + " is on your phone."; $("classify-btn").disabled = false;
    $("classify-btn").onclick = () => $("shot2").click();
    if (TEAM && navigator.mediaDevices && location.protocol === "https:" && !liveStarted) startLive();
  }
  $("shot2").onchange = () => {
    const f = $("shot2").files[0]; if (!f || !session) return;
    const img = new Image(); img.onload = async () => { Infer.drawSquare(img, img.naturalWidth, img.naturalHeight, $("work")); await classifyCanvas(); URL.revokeObjectURL(img.src); $("shot2").value = ""; };
    img.src = URL.createObjectURL(f);
  };
  async function startLive() {
    liveStarted = true;
    const v = $("cam");
    try { v.srcObject = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: 640 } }); await v.play(); }
    catch (e) { liveStarted = false; $("ship-status").textContent = "No camera (" + e.name + "). Use the button to take a photo."; return; }
    v.hidden = false; $("classify-btn").hidden = true;
    (async function tick() { if (!session) return; Infer.drawSquare(v, v.videoWidth, v.videoHeight, $("work")); await classifyCanvas(); setTimeout(tick, 150); })();
  }

  // ---------- team phone: sprint timing ----------
  async function onSprintRequest(m) {
    const run = ++timingRun;   // a second copy of this message restarts the timing; the old loop then stops
    $("result-title").textContent = "TIMING THE HORSES"; show("p-result");
    for (const model of m.models) {
      if (run !== timingRun) return;
      $("result-text").textContent = "timing " + nameOf(model.key) + "...";
      try {
        const ms = await Infer.timeModel(model.url, 30);
        if (run !== timingRun) return;
        const r = await post("/api/sprint-result", { horse: model.key, ms, device: "phone", token: TEAM_TOKEN });
        if (r.error) { $("result-text").textContent = r.error; return; }
      } catch (e) { $("result-text").textContent = "failed on " + nameOf(model.key) + ": " + e; return; }
    }
    if (run === timingRun) $("result-text").textContent = "all horses timed.";
  }

  // ---------- socket ----------
  const handlers = {
    state: (m) => { state = m; route(); },
    horses: (m) => { cards = m.cards; renderBets(); route(); },   // on connect, state comes before horses: redraw with the real names
    lap: onLap, race_done: onRaceDone, sprint: onSprint, ship: onShip,
    sprint_request: (m) => { if (TEAM) onSprintRequest(m); },
  };
  function connect() {
    const role = TEAM ? "team&token=" + encodeURIComponent(TEAM_TOKEN) : "phone";
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?role=${role}`);
    ws.onopen = () => banner(TEAM ? "TEAM PHONE CONNECTED" : "CONNECTED", true);
    ws.onmessage = (ev) => { const m = JSON.parse(ev.data); (handlers[m.type] || (() => {}))(m); };
    ws.onclose = () => { banner("RECONNECTING..."); setTimeout(connect, 3000); };
  }
  connect();
})();
