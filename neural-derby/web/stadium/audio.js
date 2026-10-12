// 8-bit sounds from WebAudio. Nothing is loaded from disk.
(function () {
  let ctx = null;
  let muted = false;
  function unlock() { if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)(); if (ctx.state === "suspended") ctx.resume(); }
  function tone(freq, dur, type, gain) {
    if (!ctx || muted) return;
    const o = ctx.createOscillator(); const g = ctx.createGain();
    o.type = type || "square"; o.frequency.value = freq;
    g.gain.setValueAtTime(gain || 0.08, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + dur);
    o.connect(g).connect(ctx.destination); o.start(); o.stop(ctx.currentTime + dur);
  }
  function noise(dur, gain, lowpass) {
    if (!ctx || muted) return;
    const buf = ctx.createBuffer(1, ctx.sampleRate * dur, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / d.length);
    const s = ctx.createBufferSource(); s.buffer = buf;
    const f = ctx.createBiquadFilter(); f.type = "lowpass"; f.frequency.value = lowpass || 2000;
    const g = ctx.createGain(); g.gain.value = gain || 0.2;
    s.connect(f).connect(g).connect(ctx.destination); s.start();
  }
  function gun() { noise(0.25, 0.4, 1200); }
  function chime() { tone(880, 0.08); setTimeout(() => tone(1320, 0.1), 70); }
  function roar() { noise(1.8, 0.25, 900); }
  function jingle() { [523, 659, 784, 1047].forEach((f, i) => setTimeout(() => tone(f, 0.12), i * 90)); }
  function toggleMute() { muted = !muted; return muted; }
  window.NeuralDerbyAudio = { unlock, gun, chime, roar, jingle, toggleMute };
})();
