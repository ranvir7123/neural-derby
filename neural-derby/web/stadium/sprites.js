// Pixel-art horses drawn from strings, so there are no image files to lose.
// '#' body colour, 'e' eye (black), '.' empty. 16 wide x 12 tall, two gallop frames.
(function () {
  const A = [
    "..........###...",
    ".........#e###..",
    ".........####...",
    "....#########...",
    "...###########..",
    "..############..",
    "..############..",
    "..##..###..##...",
    ".##...##...##...",
    "##....#.....##..",
    "......#......#..",
    "................",
  ];
  const B = [
    "..........###...",
    ".........#e###..",
    ".........####...",
    "....#########...",
    "...###########..",
    "..############..",
    "..############..",
    "...##.###.##....",
    "...##.##...##...",
    "...#...#....##..",
    "..#....#.....#..",
    "................",
  ];
  const FRAMES = [A, B];
  function drawHorse(ctx, x, y, scale, colour, frame) {
    const map = FRAMES[frame % FRAMES.length];
    for (let r = 0; r < map.length; r++) {
      for (let c = 0; c < map[r].length; c++) {
        const ch = map[r][c];
        if (ch === ".") continue;
        ctx.fillStyle = ch === "e" ? "#000" : colour;
        ctx.fillRect(x + c * scale, y + r * scale, scale, scale);
      }
    }
  }
  window.NeuralDerbySprites = { drawHorse, FRAMES: FRAMES.length, W: 16, H: 12 };
})();
