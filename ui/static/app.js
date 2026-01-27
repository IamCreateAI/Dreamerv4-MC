/* static/app.js - full file
 * Z: enter pointer lock (mouse hidden, controls enabled)
 * X: exit pointer lock (mouse visible, controls disabled)
 * Disable context menu while locked
 * Send RAW dx/dy pixels; display dx/dy from backend meta
 * Key hold: send keys at fixed rate while held
 */

const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const hud = document.getElementById("hud");

let ws = null;
let seq = 0;

// FPS (from incoming frames)
let lastFrameT = performance.now();
let fpsEMA = 0;

// Controls enabled only when pointer locked
let captureEnabled = false;

// Backend meta (authoritative)
let serverDx = 0;
let serverDy = 0;
let frameId = 0;
let clientSeq = 0;

// ---- key hold state ----
const keysDown = new Set(); // e.code
let keyHoldTimer = null;
const KEY_HOLD_HZ = 30;
const KEY_HOLD_INTERVAL_MS = Math.floor(1000 / KEY_HOLD_HZ);

// ---- mousemove merge (one per rAF) ----
let pendingMouse = null;
let sendScheduled = false;

function buildWsUrl() {
  const u = new URL("/ws", window.location.href);
  u.protocol = (u.protocol === "https:") ? "wss:" : "ws:";
  return u.toString();
}

function setHud(extra = "") {
  const buf = ws ? ws.bufferedAmount : 0;
  const dxdy = `dx=${serverDx} dy=${serverDy}`;
  const seqTxt = `frame_id=${frameId} client_seq=${clientSeq}`;
  const keysTxt = `keys=[${[...keysDown].slice(0, 8).join(",")}${keysDown.size > 8 ? "..." : ""}]`;
  const base = `fps~${fpsEMA.toFixed(1)} | locked=${captureEnabled} | ${dxdy} | ${seqTxt} | ${keysTxt} | wsbuf=${buf}`;
  hud.textContent = extra ? `${extra}\n${base}` : base;
}

function sendEvent(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  obj.type = "event";
  obj.seq = ++seq;
  ws.send(JSON.stringify(obj));
}

function scheduleMouseSend() {
  if (sendScheduled) return;
  sendScheduled = true;
  requestAnimationFrame(() => {
    sendScheduled = false;
    if (!pendingMouse) return;
    sendEvent(pendingMouse);
    pendingMouse = null;
  });
}

function startKeyHoldLoop() {
  if (keyHoldTimer) return;
  keyHoldTimer = setInterval(() => {
    if (!captureEnabled) return;
    if (keysDown.size === 0) return;
    sendEvent({ event: "key_hold", keys: Array.from(keysDown) });
  }, KEY_HOLD_INTERVAL_MS);
}

function stopKeyHoldLoop() {
  if (!keyHoldTimer) return;
  clearInterval(keyHoldTimer);
  keyHoldTimer = null;
}

function clearKeys() {
  keysDown.clear();
}

// ---- Pointer Lock control ----
function isLocked() {
  return document.pointerLockElement === cv;
}

function enterPointerLock() {
  cv.focus();
  cv.requestPointerLock();
}

function exitPointerLock() {
  if (document.pointerLockElement) {
    document.exitPointerLock();
  }
}

function updateCaptureFromPointerLock() {
  captureEnabled = isLocked();
  if (!captureEnabled) {
    clearKeys(); // 防止卡键
  }
  setHud(captureEnabled ? "LOCKED (X to unlock)" : "UNLOCKED (Z to lock)");
}

document.addEventListener("pointerlockchange", updateCaptureFromPointerLock);
document.addEventListener("pointerlockerror", () => {
  captureEnabled = false;
  clearKeys();
  setHud("pointer lock error");
});

// ---- Disable context menu while locked ----
window.addEventListener("contextmenu", (e) => {
  if (isLocked()) e.preventDefault();
});

// ---- WebSocket ----
function connect() {
  const t0 = performance.now();
  const url = buildWsUrl();
  console.log("[ws] url =", url);

  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  let firstBinary = true;
  setHud("connecting...");

  ws.onopen = () => {
    console.log("[ws] open in", (performance.now() - t0).toFixed(1), "ms");
    setHud("connected (Z lock, X unlock)");
    startKeyHoldLoop();
  };

  ws.onerror = (e) => console.log("[ws] error", e);

  ws.onclose = (e) => {
    console.log("[ws] closed:", e.code, e.reason);
    stopKeyHoldLoop();
    captureEnabled = false;
    clearKeys();
    setHud(`disconnected (code=${e.code}) retrying...`);
    setTimeout(connect, 500);
  };

  ws.onmessage = async (ev) => {
    // meta text (dx/dy from backend)
    if (typeof ev.data === "string") {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "meta") {
          serverDx = (msg.dx ?? 0) | 0;
          serverDy = (msg.dy ?? 0) | 0;
          frameId = Number(msg.frame_id ?? msg.frameId ?? 0) || 0;
          clientSeq = Number(msg.client_seq ?? msg.clientSeq ?? 0) || 0;
          setHud();
        }
      } catch {
        // ignore
      }
      return;
    }

    // binary JPEG
    if (firstBinary) {
      firstBinary = false;
      console.log("[ws] first binary frame in", (performance.now() - t0).toFixed(1), "ms");
    }

    const t = performance.now();
    const dt = t - lastFrameT;
    lastFrameT = t;
    const inst = 1000 / Math.max(1, dt);
    fpsEMA = fpsEMA ? (0.9 * fpsEMA + 0.1 * inst) : inst;

    const blob = new Blob([ev.data], { type: "image/jpeg" });
    const bmp = await createImageBitmap(blob);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(bmp, 0, 0, cv.width, cv.height);

    setHud();
  };
}

// ---- Focus basics ----
cv.tabIndex = 0;
cv.style.outline = "none";
cv.addEventListener("click", () => cv.focus());

// ---- Global key handlers (reliable under pointer lock) ----
window.addEventListener("keydown", (e) => {
  // Z: lock & enable controls
  if (e.code === "KeyZ") {
    enterPointerLock();
    e.preventDefault();
    return;
  }
  // X: unlock & disable controls
  if (e.code === "KeyX") {
    exitPointerLock();
    e.preventDefault();
    return;
  }

  // Only process controls when locked
  if (!captureEnabled) return;

  keysDown.add(e.code);
  sendEvent({ event: "keydown", key: e.code });

  e.preventDefault();
}, { passive: false });

window.addEventListener("keyup", (e) => {
  if (!captureEnabled) return;

  keysDown.delete(e.code);
  sendEvent({ event: "keyup", key: e.code });

  e.preventDefault();
}, { passive: false });



let accumDx = 0;
let accumDy = 0;
let mouseSendScheduled = false;

function scheduleMouseSend() {
  if (mouseSendScheduled) return;
  mouseSendScheduled = true;

  requestAnimationFrame(() => {
    mouseSendScheduled = false;

    if (!captureEnabled) {
      accumDx = 0; accumDy = 0;
      return;
    }

    // ✅ 一帧累计的总 dx/dy（raw）
    const dx = accumDx | 0;
    const dy = accumDy | 0;
    accumDx = 0;
    accumDy = 0;

    // 即使 dx=dy=0，你要不要发取决于你；一般不发更省
    // 但你后端会自己每帧发 0/0 meta + 图像，所以这里可以不发 0/0
    if (dx !== 0 || dy !== 0) {
      sendEvent({ event: "mousemove", dx, dy });
    }
  });
}



// ---- Mouse handlers (prefer window under pointer lock) ----
window.addEventListener("mousemove", (e) => {
  if (!captureEnabled) return;

  // movementX/Y 是 CSS 像素；如果你想要“物理像素”，乘 devicePixelRatio
  const scale = window.devicePixelRatio || 1;

  accumDx += (e.movementX ?? 0) * scale;
  accumDy += (e.movementY ?? 0) * scale;

  scheduleMouseSend();
});

window.addEventListener("mousedown", (e) => {
  if (!captureEnabled) return;

  // 禁用右键默认行为
  if (e.button === 2) e.preventDefault();

  sendEvent({ event: "mousedown", button: e.button });
}, { passive: false });

window.addEventListener("mouseup", (e) => {
  if (!captureEnabled) return;
  sendEvent({ event: "mouseup", button: e.button });
}, { passive: false });

window.addEventListener("wheel", (e) => {
  if (!captureEnabled) return;
  sendEvent({ event: "wheel", dy: e.deltaY });
  e.preventDefault();
}, { passive: false });

// start
connect();
setHud("UNLOCKED (Z to lock)");
