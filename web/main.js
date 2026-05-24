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

// ── Config ────────────────────────────────────────────────────────────────
const VOLUME_STEP         = 0.05;   // 5% per gesture trigger
const GESTURE_DEBOUNCE_MS = 700;    // minimum ms between consecutive actions
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

// ── State ─────────────────────────────────────────────────────────────────
let gestureRecognizer = null;
let drawingUtils      = null;
let lastActionTime    = 0;
let modalOpen         = false;

// ── Gesture → action map ──────────────────────────────────────────────────
const GESTURE_MAP = {
  Victory:    { icon: "✌️", label: "V-Sign",      action: capturePhoto },
  Thumb_Up:   { icon: "👍", label: "Thumbs Up",   action: volumeUp    },
  Thumb_Down: { icon: "👎", label: "Thumbs Down", action: volumeDown  },
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
  const info = GESTURE_MAP[name];
  if (!info) {
    clearGestureUI();
    statusBadge.textContent = "Listening…";
    statusBadge.className   = "badge badge--ready";
    return;
  }

  // Update gesture display
  gestureIcon.textContent    = info.icon;
  gestureLabel.textContent   = info.label;
  confidenceBar.style.width  = `${Math.round(score * 100)}%`;
  confidenceText.textContent = `${Math.round(score * 100)}%`;
  statusBadge.textContent    = "Gesture detected";
  statusBadge.className      = "badge badge--active";

  // Trigger action with debounce
  const now = Date.now();
  if (now - lastActionTime >= GESTURE_DEBOUNCE_MS) {
    info.action();
    lastActionTime = now;
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
      clearGestureUI();
      statusBadge.textContent = "Listening…";
      statusBadge.className   = "badge badge--ready";
    }
  }

  requestAnimationFrame(detect);
}

// ── Boot ──────────────────────────────────────────────────────────────────
async function init() {
  statusBadge.textContent = "Loading model…";
  statusBadge.className   = "badge badge--idle";

  try {
    // 1. Resolve MediaPipe WASM binaries from CDN
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
    statusBadge.textContent = "Requesting camera…";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });

    videoEl.srcObject = stream;

    // 4. Start detection loop once the first frame is ready
    videoEl.addEventListener("loadeddata", () => {
      statusBadge.textContent = "Listening…";
      statusBadge.className   = "badge badge--ready";
      detect();
    }, { once: true });

  } catch (err) {
    console.error("Init error:", err);
    statusBadge.textContent = err.name === "NotAllowedError"
      ? "Camera permission denied"
      : "Failed to load";
    statusBadge.className = "badge badge--idle";
  }
}

// ── Initialise volume UI and boot ─────────────────────────────────────────
audioEl.volume = 0.5;
syncVolumeUI();
init();
