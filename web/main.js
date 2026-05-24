/**
 * main.js — Gesture AV Controller
 *
 * Uses MediaPipe Tasks Vision GestureRecognizer (runs entirely in the browser).
 * No server, no custom model file — Google's model is fetched from CDN.
 *
 * Gestures:
 *   ✌️  Victory     → capture photo + download modal
 *   👍  Thumb_Up    → audio volume up
 *   👎  Thumb_Down  → audio volume down
 */

import {
  GestureRecognizer,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js";

// ── DOM refs ──────────────────────────────────────────────────────────────
const videoEl        = document.getElementById("webcam");
const canvasEl       = document.getElementById("overlay");
const ctx            = canvasEl.getContext("2d");
const statusBadge    = document.getElementById("status-badge");
const gestureIcon    = document.getElementById("gesture-icon");
const gestureLabel   = document.getElementById("gesture-label");
const confidenceBar  = document.getElementById("confidence-bar");
const confidenceText = document.getElementById("confidence-text");
const avActionEl     = document.getElementById("av-action");
const volumeBarEl    = document.getElementById("volume-bar");
const volumeValueEl  = document.getElementById("volume-value");
const audioEl        = document.getElementById("demo-audio");
const photoModal     = document.getElementById("photo-modal");
const photoPreview   = document.getElementById("photo-preview");
const photoDownload  = document.getElementById("photo-download");
const modalClose     = document.getElementById("modal-close");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingStep    = document.getElementById("loading-step");
const loadingHint    = document.getElementById("loading-hint");
const holdCountdown  = document.getElementById("hold-countdown");
const holdRingArc    = document.getElementById("hold-ring-arc");
const holdSeconds    = document.getElementById("hold-seconds");

// ── Config ────────────────────────────────────────────────────────────────
const VOLUME_STEP         = 0.05;   // 5% per gesture trigger
const GESTURE_DEBOUNCE_MS = 700;    // minimum ms between consecutive actions
const HOLD_DURATION_MS    = 3000;   // ms to hold V-sign before photo
const RING_CIRCUMFERENCE  = 2 * Math.PI * 40; // SVG circle r=40 → ≈251.3
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

// ── State ─────────────────────────────────────────────────────────────────
let gestureRecognizer = null;
let drawingUtils      = null;
let lastActionTime    = 0;
let modalOpen         = false;
let holdStartTime     = null;   // timestamp when V-sign hold started

// ── Gesture display info (UI only) ────────────────────────────────────────
const GESTURE_DISPLAY = {
  Victory:    { icon: "✌️", label: "Hold for photo…" },
  Thumb_Up:   { icon: "👍", label: "Thumbs Up"       },
  Thumb_Down: { icon: "👎", label: "Thumbs Down"     },
};

// ── Gesture → action map (non-Victory gestures) ───────────────────────────
const GESTURE_ACTIONS = {
  Thumb_Up:   volumeUp,
  Thumb_Down: volumeDown,
};

// ── Audio & volume ─────────────────────────────────────────────────────────
function volumeUp() {
  audioEl.volume = Math.min(1, audioEl.volume + VOLUME_STEP);
  syncVolumeUI();
  setAvAction(`Volume ▲  ${Math.round(audioEl.volume * 100)}%`);
}

function volumeDown() {
  audioEl.volume = Math.max(0, audioEl.volume - VOLUME_STEP);
  syncVolumeUI();
  setAvAction(`Volume ▼  ${Math.round(audioEl.volume * 100)}%`);
}

function syncVolumeUI() {
  const pct = Math.round(audioEl.volume * 100);
  volumeBarEl.style.width   = `${pct}%`;
  volumeValueEl.textContent = pct;
}

// Keep volume bar in sync if user drags the native audio control
audioEl.addEventListener("volumechange", syncVolumeUI);

// ── V-sign hold countdown ─────────────────────────────────────────────────
function startHold() {
  holdStartTime = performance.now();
  holdRingArc.style.strokeDashoffset = RING_CIRCUMFERENCE;
  holdSeconds.textContent = "3";
  holdCountdown.classList.add("visible");
}

function updateHold() {
  const elapsed  = performance.now() - holdStartTime;
  const progress = Math.min(elapsed / HOLD_DURATION_MS, 1);
  holdRingArc.style.strokeDashoffset = RING_CIRCUMFERENCE * (1 - progress);
  holdSeconds.textContent = Math.ceil((HOLD_DURATION_MS - elapsed) / 1000);

  if (progress >= 1) {
    cancelHold();
    capturePhoto();
  }
}

function cancelHold() {
  holdStartTime = null;
  holdCountdown.classList.remove("visible");
}

// ── Photo capture ─────────────────────────────────────────────────────────
function capturePhoto() {
  if (modalOpen) return; // don't stack captures

  // Draw the clean video frame (no landmark overlay) into a temporary canvas
  const snap = document.createElement("canvas");
  snap.width  = videoEl.videoWidth;
  snap.height = videoEl.videoHeight;
  snap.getContext("2d").drawImage(videoEl, 0, 0);

  const dataUrl = snap.toDataURL("image/png");
  photoPreview.src        = dataUrl;
  photoDownload.href      = dataUrl;
  photoDownload.download  = `gesture-snap-${Date.now()}.png`;

  photoModal.classList.add("visible");
  modalOpen = true;
  setAvAction("📸 Foto capturada!");
}

function closeModal() {
  photoModal.classList.remove("visible");
  modalOpen = false;
}

modalClose.addEventListener("click", closeModal);
photoModal.addEventListener("click", (e) => {
  if (e.target === photoModal) closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});

// ── UI helpers ────────────────────────────────────────────────────────────
function setAvAction(text) {
  avActionEl.textContent = text;
  avActionEl.classList.remove("flash");
  void avActionEl.offsetWidth; // force reflow to restart animation
  avActionEl.classList.add("flash");
}

function clearGestureUI() {
  gestureIcon.textContent    = "—";
  gestureLabel.textContent   = "No gesture";
  confidenceBar.style.width  = "0%";
  confidenceText.textContent = "0%";
}

// ── Gesture handler ───────────────────────────────────────────────────────
function handleGesture(name, score) {
  const display = GESTURE_DISPLAY[name];
  if (!display) {
    // Unknown gesture — cancel any hold and clear UI
    if (holdStartTime !== null) cancelHold();
    clearGestureUI();
    return;
  }

  // Update gesture display UI
  gestureIcon.textContent    = display.icon;
  gestureLabel.textContent   = display.label;
  confidenceBar.style.width  = `${Math.round(score * 100)}%`;
  confidenceText.textContent = `${Math.round(score * 100)}%`;
  statusBadge.textContent    = "Gesture detected";
  statusBadge.className      = "badge badge--active";

  if (name === "Victory") {
    // V-sign: hold 3 s to capture photo
    if (modalOpen) return;
    if (holdStartTime === null) startHold();
    else updateHold();
  } else {
    // Any non-Victory gesture cancels an ongoing hold
    if (holdStartTime !== null) cancelHold();

    // Trigger action with debounce
    const action = GESTURE_ACTIONS[name];
    const now = Date.now();
    if (action && now - lastActionTime >= GESTURE_DEBOUNCE_MS) {
      action();
      lastActionTime = now;
    }
  }
}

// ── Detection loop (requestAnimationFrame) ────────────────────────────────
function detect() {
  // Wait until video has actual frame data
  if (videoEl.readyState >= 2) {
    // Keep canvas in sync with video dimensions
    if (canvasEl.width  !== videoEl.videoWidth)  canvasEl.width  = videoEl.videoWidth;
    if (canvasEl.height !== videoEl.videoHeight) canvasEl.height = videoEl.videoHeight;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    const results = gestureRecognizer.recognizeForVideo(videoEl, performance.now());

    // Draw hand landmarks on the overlay canvas
    if (results.landmarks?.length) {
      for (const landmarks of results.landmarks) {
        drawingUtils.drawConnectors(
          landmarks,
          GestureRecognizer.HAND_CONNECTIONS,
          { color: "#00e5ff", lineWidth: 2 }
        );
        drawingUtils.drawLandmarks(landmarks, {
          color: "#7c3aed",
          lineWidth: 1,
          radius: 4,
        });
      }
    }

    // Process top gesture (ignore "None" — no recognisable gesture)
    const topGesture = results.gestures?.[0]?.[0];
    if (topGesture && topGesture.categoryName !== "None") {
      handleGesture(topGesture.categoryName, topGesture.score);
    } else {
      if (holdStartTime !== null) cancelHold();
      clearGestureUI();
      statusBadge.textContent = "Listening…";
      statusBadge.className   = "badge badge--ready";
    }
  }

  requestAnimationFrame(detect);
}

// ── Loading overlay helpers ───────────────────────────────────────────────
function setLoading(step, hint = "") {
  loadingStep.textContent = step;
  loadingHint.textContent = hint;
}

function hideLoading() {
  loadingOverlay.classList.add("hidden");
}

// ── Boot ──────────────────────────────────────────────────────────────────
async function init() {
  statusBadge.textContent = "Loading model…";
  statusBadge.className   = "badge badge--idle";

  try {
    // 1. Resolve MediaPipe WASM binaries from CDN
    setLoading("Downloading AI model…", "First load may take ~20 s — the model is ~10 MB");
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    // 2. Create GestureRecognizer — model is fetched from Google's CDN
    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU",
      },
      runningMode:                  "VIDEO",
      numHands:                     1,
      minHandDetectionConfidence:   0.7,
      minHandPresenceConfidence:    0.5,
      minTrackingConfidence:        0.5,
    });

    drawingUtils = new DrawingUtils(ctx);

    // 3. Request camera access
    setLoading("Requesting camera…", "Allow camera access when your browser asks");
    statusBadge.textContent = "Requesting camera…";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });

    videoEl.srcObject = stream;

    // 4. Start detection loop once the first frame is ready
    videoEl.addEventListener("loadeddata", () => {
      hideLoading();
      statusBadge.textContent = "Listening…";
      statusBadge.className   = "badge badge--ready";
      detect();
    }, { once: true });

  } catch (err) {
    console.error("Init error:", err);
    const msg = err.name === "NotAllowedError"
      ? "Camera permission denied"
      : "Failed to load";
    setLoading("⚠️ " + msg, err.name === "NotAllowedError"
      ? "Click the camera icon in your browser's address bar to allow access"
      : "Try refreshing the page");
    statusBadge.textContent = msg;
    statusBadge.className = "badge badge--idle";
  }
}

// ── Initialise volume UI and boot ─────────────────────────────────────────
audioEl.volume = 0.5;
syncVolumeUI();
init();
