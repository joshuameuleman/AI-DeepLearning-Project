const BORDER = 1;
const CELL_PX = 6;
const MAX_DISPLAY_PX = 768;
const DEFAULT_WAITING_CELLS = 64;

const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const titleEl = document.getElementById("title");
const metricLabelEl = document.getElementById("metricLabel");
const hintEl = document.getElementById("hint");
const highscoreEl = document.getElementById("highscore");
const toggleBtn = document.getElementById("toggle");
const resetBtn = document.getElementById("reset");
const speedInput = document.getElementById("speed");
const gameSelect = document.getElementById("gameSelect");

const textures = {
  head: loadImage("/Games/Snake/textures/Snake_head.png"),
  body: loadImage("/Games/Snake/textures/Snake_body.png"),
  tail: loadImage("/Games/Snake/textures/Snake_tail.png"),
  apple: loadImage("/Games/Snake/textures/Snake_apple.png"),
  wall: loadImage("/Games/Snake/textures/Snake_Wall.png"),
};

const flappyTextures = {
  birdUp: loadImage("/Games/Flappy%20Bird/textures/yellowbird-upflap.png"),
  birdMid: loadImage("/Games/Flappy%20Bird/textures/yellowbird-midflap.png"),
  birdDown: loadImage("/Games/Flappy%20Bird/textures/yellowbird-downflap.png"),
  pipe: loadImage("/Games/Flappy%20Bird/textures/pipe-green.png"),
};

let running = true;
// Keep network polling at a safe fixed interval; the slider controls only visual smoothing.
const FALLBACK_POLL_MS = 220;
let pollMs = FALLBACK_POLL_MS;
let uiSpeed = Number(speedInput.value) > 0 ? Number(speedInput.value) : 30;
let poller = null;
let feed = null;
let displayFeed = null;
let bestScore = 0;
const EVENT_PATHS = ["events", "./events", "/events", "/web/events"];
let lastFeedSignature = "";
let eventSource = null;
let activeEventPath = null;
let animationFrameId = null;
let transitionFromFeed = null;
let transitionToFeed = null;
let transitionStartMs = 0;
let transitionDurationMs = 120;
const MAX_INTERPOLATE_HEAD_DISTANCE = 12.0;
const MAX_INTERPOLATE_STEP_GAP = 14;
let selectedGame = gameSelect.value || "snake";

function syncCanvasSize(gridWidth, gridHeight) {
  const viewCells = Math.max(Number(gridWidth || 10), Number(gridHeight || 10));
  const boardCells = viewCells + BORDER * 2;
  const intrinsicSize = Math.max(1, boardCells * CELL_PX);
  const displaySize = Math.min(MAX_DISPLAY_PX, intrinsicSize);

  if (canvas.width !== intrinsicSize || canvas.height !== intrinsicSize) {
    canvas.width = intrinsicSize;
    canvas.height = intrinsicSize;
  }

  canvas.style.width = `${displaySize}px`;
  canvas.style.height = `${displaySize}px`;
}

function syncCanvasPixels(widthPx, heightPx) {
  const width = Math.max(64, Number(widthPx || 288));
  const height = Math.max(64, Number(heightPx || 512));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const displayScale = Math.min(MAX_DISPLAY_PX / width, MAX_DISPLAY_PX / height, 2.0);
  canvas.style.width = `${Math.floor(width * displayScale)}px`;
  canvas.style.height = `${Math.floor(height * displayScale)}px`;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function computeTransitionDurationMs() {
  // Faster slider setting => shorter interpolation window => visibly snappier movement.
  const clamped = Math.max(10, Math.min(60, Number(uiSpeed) || 30));
  const normalized = (clamped - 10) / 50;
  return Math.round(220 - normalized * 120);
}

function interpolatePoint(a, b, t) {
  if (!a || !b) {
    return b || a;
  }
  return {
    x: lerp(Number(a.x || 0), Number(b.x || 0), t),
    y: lerp(Number(a.y || 0), Number(b.y || 0), t),
  };
}

function pointDistance(a, b) {
  if (!a || !b) {
    return Number.POSITIVE_INFINITY;
  }
  const dx = Number(a.x || 0) - Number(b.x || 0);
  const dy = Number(a.y || 0) - Number(b.y || 0);
  return Math.hypot(dx, dy);
}

function canSmoothTransition(fromFeed, toFeed) {
  if (!fromFeed || !toFeed) {
    return false;
  }
  if (fromFeed.game !== "snake" || toFeed.game !== "snake") {
    return false;
  }
  if (Number(fromFeed.gridWidth || 0) !== Number(toFeed.gridWidth || 0)) {
    return false;
  }
  if (Number(fromFeed.gridHeight || 0) !== Number(toFeed.gridHeight || 0)) {
    return false;
  }

  const fromEpisode = Number(fromFeed.episode || 0);
  const toEpisode = Number(toFeed.episode || 0);
  const fromStep = Number(fromFeed.step || 0);
  const toStep = Number(toFeed.step || 0);
  if (toEpisode !== fromEpisode) {
    return false;
  }
  if (toStep <= fromStep || toStep > fromStep + MAX_INTERPOLATE_STEP_GAP) {
    return false;
  }

  const fromHead = Array.isArray(fromFeed.snake) ? fromFeed.snake[0] : null;
  const toHead = Array.isArray(toFeed.snake) ? toFeed.snake[0] : null;
  return pointDistance(fromHead, toHead) <= MAX_INTERPOLATE_HEAD_DISTANCE;
}

function interpolateFeed(fromFeed, toFeed, t) {
  if (!fromFeed || !toFeed) {
    return toFeed || fromFeed;
  }

  const fromSnake = Array.isArray(fromFeed.snake) ? fromFeed.snake : null;
  const toSnake = Array.isArray(toFeed.snake) ? toFeed.snake : null;
  if (!fromSnake || !toSnake || toSnake.length === 0) {
    return toFeed;
  }

  // Interpolate only the head and keep body from the target frame to avoid body warping/teleport artifacts.
  const snake = toSnake.map((segment) => ({ x: Number(segment.x || 0), y: Number(segment.y || 0) }));
  snake[0] = interpolatePoint(fromSnake[0], toSnake[0], t);

  const food = interpolatePoint(fromFeed.food, toFeed.food, t);
  return {
    ...toFeed,
    snake,
    food,
  };
}

function stopAnimationLoop() {
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
}

function animationTick(now) {
  if (!transitionToFeed) {
    animationFrameId = null;
    return;
  }

  const duration = Math.max(1, transitionDurationMs);
  const t = Math.min(1, (now - transitionStartMs) / duration);
  displayFeed = interpolateFeed(transitionFromFeed, transitionToFeed, t);
  render();

  if (t < 1) {
    animationFrameId = requestAnimationFrame(animationTick);
    return;
  }

  displayFeed = transitionToFeed;
  animationFrameId = null;
}

function loadImage(src) {
  const img = new Image();
  img.src = src;
  img.onerror = () => {
    img.__failed = true;
  };
  return img;
}

function drawMessage(message) {
  ctx.fillStyle = "#1a1a1a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#95a6b9";
  ctx.font = "bold 18px Trebuchet MS";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

function drawSnakeFeed(data) {
  const snake = data.snake;
  const foods = Array.isArray(data.foods) && data.foods.length > 0
    ? data.foods
    : (data.food ? [data.food] : []);
  const gridWidth = Number(data.gridWidth || 10);
  const gridHeight = Number(data.gridHeight || 10);
  syncCanvasSize(gridWidth, gridHeight);
  const viewCells = Math.max(gridWidth, gridHeight);
  const tile = Math.max(4, Math.floor(canvas.width / (viewCells + BORDER * 2)));
  const boardPx = (viewCells + BORDER * 2) * tile;
  const offsetX = Math.floor((canvas.width - boardPx) / 2);
  const offsetY = Math.floor((canvas.height - boardPx) / 2);
  if (!Array.isArray(snake)) {
    drawMessage("Wachten op geldige snake feed...");
    return;
  }

  const camX = 0;
  const camY = 0;

  const toScreen = (x, y) => ({
    x: offsetX + (x - camX + BORDER) * tile,
    y: offsetY + (y - camY + BORDER) * tile,
  });

  const inView = (x, y) => x >= camX && x < camX + viewCells && y >= camY && y < camY + viewCells;

  ctx.fillStyle = "#1a1a1a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const widthWithBorder = viewCells + BORDER * 2;
  const heightWithBorder = viewCells + BORDER * 2;
  for (let x = 0; x < widthWithBorder; x++) {
    for (let y = 0; y < heightWithBorder; y++) {
      const border = x === 0 || y === 0 || x === widthWithBorder - 1 || y === heightWithBorder - 1;
      if (!border) {
        continue;
      }
      const px = offsetX + x * tile;
      const py = offsetY + y * tile;
      if (!textures.wall.__failed && textures.wall.complete) {
        ctx.drawImage(textures.wall, px, py, tile, tile);
      }
    }
  }

  for (const food of foods) {
    if (inView(food.x, food.y)) {
      const foodPos = toScreen(food.x, food.y);
      // Always draw a visible base shape first.
      ctx.fillStyle = "#ff4a2a";
      ctx.beginPath();
      ctx.arc(foodPos.x + tile / 2, foodPos.y + tile / 2, Math.max(2, tile * 0.32), 0, Math.PI * 2);
      ctx.fill();

      if (!textures.apple.__failed && textures.apple.complete) {
        ctx.drawImage(textures.apple, foodPos.x, foodPos.y, tile, tile);
      }
    }
  }

  for (let i = snake.length - 1; i >= 0; i--) {
    const p = snake[i];
    if (!inView(p.x, p.y)) {
      continue;
    }
    const pos = toScreen(p.x, p.y);
    const px = pos.x;
    const py = pos.y;

    // Always draw a visible base body/head/tail block first.
    if (i === 0) {
      ctx.fillStyle = "#8cf38f";
    } else if (i === snake.length - 1) {
      ctx.fillStyle = "#63bb65";
    } else {
      ctx.fillStyle = "#52d273";
    }
    ctx.fillRect(px + Math.max(1, tile * 0.08), py + Math.max(1, tile * 0.08), tile - Math.max(2, tile * 0.16), tile - Math.max(2, tile * 0.16));

    if (i === 0 && !textures.head.__failed && textures.head.complete) {
      ctx.drawImage(textures.head, px, py, tile, tile);
    } else if (i === snake.length - 1 && !textures.tail.__failed && textures.tail.complete) {
      ctx.drawImage(textures.tail, px, py, tile, tile);
    } else if (!textures.body.__failed && textures.body.complete) {
      ctx.drawImage(textures.body, px, py, tile, tile);
    }
  }
}

function drawFlappyFeed(data) {
  const screenWidth = Number(data.screenWidth || 288);
  const screenHeight = Number(data.screenHeight || 512);
  const bird = data.bird || null;
  const pipes = Array.isArray(data.pipes) ? data.pipes : [];
  const pipeGap = Number(data.pipeGap || 120);
  const pipeWidth = Number(data.pipeWidth || 52);

  syncCanvasPixels(screenWidth, screenHeight);

  if (!bird) {
    drawMessage("Wachten op geldige flappy feed...");
    return;
  }

  const bgGradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
  bgGradient.addColorStop(0, "#6ec6ff");
  bgGradient.addColorStop(1, "#9ee0ff");
  ctx.fillStyle = bgGradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (const pipe of pipes) {
    const x = Number(pipe.x || 0);
    const gapY = Number(pipe.gapY || screenHeight / 2);
    const topHeight = Math.max(0, Math.floor(gapY - pipeGap / 2));
    const bottomY = Math.min(canvas.height, Math.floor(gapY + pipeGap / 2));
    const bottomHeight = Math.max(0, canvas.height - bottomY);

    if (!flappyTextures.pipe.__failed && flappyTextures.pipe.complete) {
      if (topHeight > 0) {
        const topPipe = flappyTextures.pipe;
        const flipped = document.createElement("canvas");
        flipped.width = Math.max(1, Math.floor(pipeWidth));
        flipped.height = topHeight;
        const fctx = flipped.getContext("2d");
        fctx.translate(0, topHeight);
        fctx.scale(1, -1);
        fctx.drawImage(topPipe, 0, 0, flipped.width, topHeight);
        ctx.drawImage(flipped, Math.floor(x), 0);
      }
      if (bottomHeight > 0) {
        ctx.drawImage(flappyTextures.pipe, Math.floor(x), bottomY, Math.floor(pipeWidth), bottomHeight);
      }
    } else {
      ctx.fillStyle = "#49b749";
      if (topHeight > 0) {
        ctx.fillRect(Math.floor(x), 0, Math.floor(pipeWidth), topHeight);
      }
      if (bottomHeight > 0) {
        ctx.fillRect(Math.floor(x), bottomY, Math.floor(pipeWidth), bottomHeight);
      }
    }
  }

  const birdX = Number(bird.x || 56);
  const birdY = Number(bird.y || screenHeight / 2);
  const birdRadius = Math.max(6, Number(bird.radius || 12));
  const velocity = Number(bird.velocity || 0);

  let birdSprite = flappyTextures.birdMid;
  if (velocity < -1.0) {
    birdSprite = flappyTextures.birdUp;
  } else if (velocity > 1.5) {
    birdSprite = flappyTextures.birdDown;
  }

  if (!birdSprite.__failed && birdSprite.complete) {
    const tilt = Math.max(-30, Math.min(30, -velocity * 4));
    ctx.save();
    ctx.translate(birdX, birdY);
    ctx.rotate((tilt * Math.PI) / 180);
    ctx.drawImage(birdSprite, -17, -12, 34, 24);
    ctx.restore();
  } else {
    ctx.fillStyle = "#ffe248";
    ctx.beginPath();
    ctx.arc(birdX, birdY, birdRadius, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#d8c888";
  ctx.fillRect(0, canvas.height - 6, canvas.width, 6);
}

function applyFeed(nextFeed) {
  const nextSignature = [
    String(nextFeed.episode || ""),
    String(nextFeed.step || ""),
    String(nextFeed.score || ""),
    String(nextFeed.episodeReward || ""),
  ].join("|");
  if (nextSignature === lastFeedSignature) {
    return false;
  }

  const previousFeed = feed;
  feed = nextFeed;
  lastFeedSignature = nextSignature;

  if (canSmoothTransition(previousFeed, nextFeed)) {
    transitionFromFeed = displayFeed || previousFeed || nextFeed;
    transitionToFeed = nextFeed;
    transitionStartMs = performance.now();
    transitionDurationMs = computeTransitionDurationMs();
    if (animationFrameId === null) {
      animationFrameId = requestAnimationFrame(animationTick);
    }
    return true;
  }

  transitionFromFeed = null;
  transitionToFeed = null;
  stopAnimationLoop();
  displayFeed = nextFeed;
  render();
  return true;
}

function stopPolling() {}

function closeEventStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

function startEventStream() {
  if (!running || typeof EventSource === "undefined" || eventSource) {
    return;
  }

  const candidates = activeEventPath
    ? [activeEventPath, ...EVENT_PATHS.filter((path) => path !== activeEventPath)]
    : EVENT_PATHS;

  let index = 0;
  const tryNext = () => {
    if (!running) {
      return;
    }
    if (index >= candidates.length) {
      restartPolling();
      return;
    }

    const path = candidates[index++];
    const es = new EventSource(path);
    let opened = false;
    let timeoutId = setTimeout(() => {
      if (!opened) {
        es.close();
        tryNext();
      }
    }, 1500);

    es.onopen = () => {
      opened = true;
      activeEventPath = path;
      eventSource = es;
      clearTimeout(timeoutId);
      stopPolling();
    };

    const handleFeedEvent = (event) => {
      try {
        const nextFeed = JSON.parse(event.data);
        applyFeed(nextFeed);
      } catch (_) {
        // Ignore malformed events and continue listening.
      }
    };

    es.addEventListener("state", handleFeedEvent);
    es.onmessage = handleFeedEvent;

    es.onerror = () => {
      clearTimeout(timeoutId);
      es.close();
      if (!opened) {
        tryNext();
        return;
      }
      if (eventSource === es) {
        eventSource = null;
      }
      restartPolling();
      setTimeout(() => {
        if (running) {
          startEventStream();
        }
      }, 1200);
    };
  };

  tryNext();
}

async function pollFeed() {
}

function render() {
  const visualFeed = displayFeed || feed;
  const statsFeed = feed || visualFeed;
  const activeGame = (statsFeed && statsFeed.game) || selectedGame;
  titleEl.textContent = `${String(activeGame || "game").toUpperCase()} Meekijken`;

  if (statsFeed && statsFeed.game && gameSelect.value !== statsFeed.game) {
    gameSelect.value = statsFeed.game;
    selectedGame = statsFeed.game;
  }

  if (activeGame === "flappy") {
    syncCanvasPixels(
      Number((statsFeed && statsFeed.screenWidth) || 288),
      Number((statsFeed && statsFeed.screenHeight) || 512)
    );
  } else {
    syncCanvasSize(
      Number((statsFeed && statsFeed.gridWidth) || DEFAULT_WAITING_CELLS),
      Number((statsFeed && statsFeed.gridHeight) || DEFAULT_WAITING_CELLS)
    );
  }

  if (!visualFeed || (visualFeed.game !== "snake" && visualFeed.game !== "flappy")) {
    metricLabelEl.innerHTML = "Score: <strong id=\"score\">0</strong>";
    hintEl.textContent = `Geen live ${selectedGame}-feed gevonden. Start ${selectedGame} training/simulatie in main.py.`;
    drawMessage("Wachten op live training...");
    return;
  }

  if (visualFeed.game === "snake") {
    drawSnakeFeed(visualFeed);
  } else {
    drawFlappyFeed(visualFeed);
  }

  bestScore = Math.max(bestScore, Number((statsFeed && statsFeed.score) || 0));
  metricLabelEl.innerHTML = `Score: <strong id="score">${Number((statsFeed && statsFeed.score) || 0)}</strong>`;
  highscoreEl.textContent = String(bestScore);
  const status = statsFeed && statsFeed.training ? "training" : "klaar";
  if (visualFeed.game === "snake") {
    hintEl.textContent = `Status ${status} | Episode ${(statsFeed && statsFeed.episode) || 0}/${(statsFeed && statsFeed.totalEpisodes) || 0} | Step ${(statsFeed && statsFeed.step) || 0} | EpReward ${Number((statsFeed && statsFeed.episodeReward) || 0).toFixed(2)} | Epsilon ${Number((statsFeed && statsFeed.epsilon) || 0).toFixed(4)} | BoardFilled ${(statsFeed && statsFeed.boardFilledCount) || 0} | Wall ${(statsFeed && statsFeed.wallCollisionCount) || 0} | Self ${(statsFeed && statsFeed.selfCollisionCount) || 0} | Fruits ${Number((statsFeed && statsFeed.foodCount) || 0)}/${Number((statsFeed && statsFeed.targetFoodCount) || 0)} | Grid ${(statsFeed && statsFeed.gridWidth) || 10}x${(statsFeed && statsFeed.gridHeight) || 10} | ViewSpeed ${uiSpeed}`;
  } else {
    hintEl.textContent = `Status ${status} | Episode ${(statsFeed && statsFeed.episode) || 0}/${(statsFeed && statsFeed.totalEpisodes) || 0} | Step ${(statsFeed && statsFeed.step) || 0} | EpReward ${Number((statsFeed && statsFeed.episodeReward) || 0).toFixed(2)} | Epsilon ${Number((statsFeed && statsFeed.epsilon) || 0).toFixed(4)} | Pipes ${Array.isArray((statsFeed && statsFeed.pipes)) ? statsFeed.pipes.length : 0} | BirdY ${Number((statsFeed && statsFeed.bird && statsFeed.bird.y) || 0).toFixed(1)} | ViewSpeed ${uiSpeed}`;
  }
}

function restartPolling() {
  if (running) {
    startEventStream();
  }
}

toggleBtn.addEventListener("click", () => {
  running = !running;
  toggleBtn.textContent = running ? "Pauze" : "Verder";
  if (running) {
    startEventStream();
    restartPolling();
    if (feed && animationFrameId === null) {
      transitionFromFeed = displayFeed || feed;
      transitionToFeed = feed;
      transitionStartMs = performance.now();
      transitionDurationMs = 80;
      animationFrameId = requestAnimationFrame(animationTick);
    }
    return;
  }
  closeEventStream();
  stopAnimationLoop();
});

resetBtn.addEventListener("click", () => {
  bestScore = 0;
  render();
});

speedInput.addEventListener("input", (e) => {
  const value = Number(e.target.value);
  uiSpeed = value > 0 ? value : 30;
  pollMs = FALLBACK_POLL_MS;
  transitionDurationMs = computeTransitionDurationMs();
  if (!eventSource) {
    restartPolling();
  }
  render();
});

gameSelect.addEventListener("change", () => {
  selectedGame = gameSelect.value;
  render();
});

gameSelect.value = selectedGame;
render();
startEventStream();
