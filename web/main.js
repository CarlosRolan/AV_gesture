/**
 * main.js — Gesture AV Controller (client-side inference)
 *
 * Pipeline:
 *  1. MediaPipe Hands → 21 landmarks x,y,z
 *  2. Normalise landmarks (same as Python capture_dataset.py)
 *  3. TF.js model → gesture class + confidence
 *  4. Map gesture → AV action (volume control)
 */

// ── Config ────────────────────────────────────────────────────────────────
const MODEL_URL         = "../model/tfjs_model/model.json";
const LABEL_MAP_URL     = "../model/label_map.json";
const CONFIDENCE_THRESH = 0.80;   // below this → "no gesture"
const VOLUME_STEP       = 5;      // percent per thumbs gesture trigger
const GESTURE_DEBOUNCE_MS = 600;  // ms between consecutive AV actions

// ── AV action map ─────────────────────────────────────────────────────────
const AV_ACTIONS = {
  v_sign:      { label: "Volume Control Mode",  icon: "✌️",  action: activateVolumeMode },
  thumbs_up:   { label: "Volume ▲",            icon: "👍",  action: volumeUp           },
  thumbs_down: { label: "Volume ▼",            icon: "👎",  action: volumeDown         },
};

// ── State ─────────────────────────────────────────────────────────────────
let model       = null;
let labelMap    = {};
let volume      = 50;
let volumeMode  = false;
let lastActionTime = 0;

// ── DOM refs ──────────────────────────────────────────────────────────────
const videoEl        = document.getElementById("webcam");
const canvasEl       = document.getElementById("overlay");
const ctx            = canvasEl.getContext("2d");
const statusBadge    = document.getElementById("status-badge");
const gestureIcon    = document.getElementById("gesture-icon");
const gestureLabel   = document.getElementById("gesture-label");
const confidenceBar  = document.getElementById("confidence-bar");
const confidenceText = document.getElementById("confidence-text");
const avAction       = document.getElementById("av-action");
const volumeBar      = document.getElementById("volume-bar");
const volumeValue    = document.getElementById("volume-value");

// ── Normalisation (mirrors Python capture_dataset.py) ─────────────────────
function normalizeLandmarks(landmarks) {
  // landmarks: [{x, y, z}, ...] (21 points, normalised to [0,1] by MediaPipe)
  const ox = landmarks[0].x;
  const oy = landmarks[0].y;
  const oz = landmarks[0].z;

  const pts = landmarks.map(p => [p.x - ox, p.y - oy, p.z - oz]);

  const maxDist = Math.max(...pts.slice(1).map(
    ([x, y, z]) => Math.sqrt(x * x + y * y + z * z)
  )) || 1;

  return pts.flatMap(([x, y, z]) => [x / maxDist, y / maxDist, z / maxDist]);
}

// ── AV Actions ────────────────────────────────────────────────────────────
function activateVolumeMode() {
  volumeMode = true;
  setAvAction("Volume control mode ACTIVE");
}

function volumeUp() {
  if (!volumeMode) return;
  volume = Math.min(100, volume + VOLUME_STEP);
  setAvAction(`Volume ▲  ${volume}`);
  updateVolumeUI();
}

function volumeDown() {
  if (!volumeMode) return;
  volume = Math.max(0, volume - VOLUME_STEP);
  setAvAction(`Volume ▼  ${volume}`);
  updateVolumeUI();
}

function setAvAction(text) {
  avAction.textContent = text;
  avAction.classList.add("flash");
  setTimeout(() => avAction.classList.remove("flash"), 400);
}

function updateVolumeUI() {
  volumeBar.style.width  = `${volume}%`;
  volumeValue.textContent = volume;
}

// ── Gesture handling ──────────────────────────────────────────────────────
function handleGesture(gestureKey, confidence) {
  const info = AV_ACTIONS[gestureKey];
  if (!info) return;

  gestureIcon.textContent  = info.icon;
  gestureLabel.textContent = info.label;

  const pct = Math.round(confidence * 100);
  confidenceBar.style.width  = `${pct}%`;
  confidenceText.textContent = `${pct}%`;

  const now = Date.now();
  if (now - lastActionTime >= GESTURE_DEBOUNCE_MS) {
    info.action();
    lastActionTime = now;
  }
}

function clearGestureUI() {
  gestureIcon.textContent  = "—";
  gestureLabel.textContent = "No gesture";
  confidenceBar.style.width  = "0%";
  confidenceText.textContent = "0%";
}

// ── Inference ─────────────────────────────────────────────────────────────
async function runInference(landmarks) {
  if (!model) return;

  const input  = normalizeLandmarks(landmarks);
  const tensor = tf.tensor2d([input]);

  const predictions = await model.predict(tensor).data();
  tensor.dispose();

  const maxIdx    = predictions.indexOf(Math.max(...predictions));
  const maxConf   = predictions[maxIdx];
  const label     = labelMap[maxIdx];

  if (maxConf >= CONFIDENCE_THRESH && label) {
    handleGesture(label, maxConf);
    statusBadge.textContent  = "Gesture detected";
    statusBadge.className    = "badge badge--active";
  } else {
    clearGestureUI();
    statusBadge.textContent = "Listening…";
    statusBadge.className   = "badge badge--ready";
  }
}

// ── MediaPipe Hands setup ─────────────────────────────────────────────────
function setupMediaPipe() {
  const hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  hands.setOptions({
    maxNumHands:            1,
    modelComplexity:        1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence:  0.5,
  });

  hands.onResults((results) => {
    // Sync canvas size to video
    canvasEl.width  = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    if (!results.multiHandLandmarks?.length) {
      clearGestureUI();
      if (model) {
        statusBadge.textContent = "Listening…";
        statusBadge.className   = "badge badge--ready";
      }
      return;
    }

    const landmarks = results.multiHandLandmarks[0];

    // Draw landmarks
    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: "#00e5ff", lineWidth: 2 });
    drawLandmarks(ctx,  landmarks, { color: "#7c3aed", lineWidth: 1, radius: 4 });

    runInference(landmarks);
  });

  const camera = new Camera(videoEl, {
    onFrame: async () => { await hands.send({ image: videoEl }); },
    width: 1280,
    height: 720,
  });

  camera.start().then(() => {
    if (model) {
      statusBadge.textContent = "Ready";
      statusBadge.className   = "badge badge--ready";
    }
  });
}

// ── Boot ──────────────────────────────────────────────────────────────────
async function init() {
  statusBadge.textContent = "Loading model…";

  try {
    [model, labelMap] = await Promise.all([
      tf.loadLayersModel(MODEL_URL),
      fetch(LABEL_MAP_URL).then(r => r.json()),
    ]);
    console.log("Model loaded. Labels:", labelMap);
  } catch (err) {
    console.warn("Model not found — running in demo mode (no inference).", err);
    statusBadge.textContent = "Demo mode (no model)";
  }

  setupMediaPipe();
}

document.addEventListener("DOMContentLoaded", init);
