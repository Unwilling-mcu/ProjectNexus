"""
Project Nexus — RefLens ML Pipeline
====================================
Foul / Dive / Handball / Offside Detection with DANN Domain Adaptation.
Author  : Sanchayan (Unwilling-mcu)
GitHub  : github.com/Unwilling-mcu/ProjectNexus
License : MIT

Architecture
------------
1. VideoIngestor      – pulls frames from stadium camera feeds
2. PoseExtractor      – YOLOv8-pose skeleton estimation (17 keypoints, 50fps)
3. TemporalEncoder    – 1D-TCN over 2-second pose windows → event embedding
4. EventClassifier    – 5-class head: offside / foul / dive / handball / clean
5. DANNAdaptor        – domain-adversarial training to generalise across stadiums
6. AlertDispatcher    – sub-100ms earpiece + VAR alert with confidence gate
"""

import time
from pathlib import Path
import json
import asyncio
import threading
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ── Optional heavy imports (graceful fallback for environments without GPU) ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[RefLens] PyTorch not found. Running in stub/demo mode.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[RefLens] Ultralytics not found. Pose extraction will be stubbed.")

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────

EVENT_CLASSES = {
    0: "clean",
    1: "offside",
    2: "foul",
    3: "dive",
    4: "handball",
}

CONFIDENCE_THRESHOLDS = {
    "offside":  0.82,   # High – auto-alert to assistant referee
    "foul":     0.75,   # Medium – alert + VAR clip
    "dive":     0.70,   # Medium – VAR review flag
    "handball": 0.78,   # High – ball-IMU corroboration required
    "clean":    0.00,   # No alert needed
}

@dataclass
class PoseFrame:
    """17 COCO keypoints × 3 (x, y, confidence) per frame."""
    timestamp: float
    player_id: int
    keypoints: np.ndarray          # shape (17, 3)
    camera_id: int = 0
    ball_position: Optional[np.ndarray] = None   # (x, y, z) from SAOT feed

@dataclass
class EventPrediction:
    event_type: str
    confidence: float
    player_ids: list
    timestamp: float
    clip_start: float
    domain_shift_score: float = 0.0   # 0 = same domain as training, 1 = full shift
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# 2. POSE EXTRACTOR (YOLOv8-pose wrapper)
# ─────────────────────────────────────────────────────────────────

class PoseExtractor:
    """
    Wraps YOLOv8-pose to extract player skeletons from each video frame.
    Falls back to synthetic stub data when ultralytics is not installed.
    """

    MODEL_PATH = "yolov8m-pose.pt"   # medium model balances speed + accuracy
    KEYPOINT_COUNT = 17               # COCO skeleton

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self._model = None
        if YOLO_AVAILABLE:
            try:
                self._model = YOLO(self.MODEL_PATH)
                print(f"[PoseExtractor] Loaded YOLOv8-pose for camera {camera_id}")
            except Exception as e:
                print(f"[PoseExtractor] Model load failed ({e}). Using stubs.")

    def extract(self, frame: np.ndarray, timestamp: float) -> list[PoseFrame]:
        """
        Returns a list of PoseFrame objects, one per detected player in frame.
        """
        if self._model is None:
            return self._stub_frames(timestamp)

        results = self._model(frame, verbose=False)[0]
        frames = []
        for i, kps in enumerate(results.keypoints.data):
            kp_array = kps.cpu().numpy()   # (17, 3)
            frames.append(PoseFrame(
                timestamp=timestamp,
                player_id=i,
                keypoints=kp_array,
                camera_id=self.camera_id,
            ))
        return frames

    def _stub_frames(self, timestamp: float) -> list[PoseFrame]:
        """Synthetic random keypoints for demo / unit testing."""
        n_players = np.random.randint(8, 14)
        frames = []
        for i in range(n_players):
            kp = np.random.rand(self.KEYPOINT_COUNT, 3).astype(np.float32)
            kp[:, 2] = np.clip(kp[:, 2] + 0.4, 0.0, 1.0)   # confidence bias
            frames.append(PoseFrame(
                timestamp=timestamp,
                player_id=i,
                keypoints=kp,
                camera_id=self.camera_id,
            ))
        return frames


# ─────────────────────────────────────────────────────────────────
# 3. TEMPORAL CONVOLUTIONAL NETWORK (TCN)
# ─────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class TCNBlock(nn.Module):
        """
        Dilated causal convolution block with residual connection.
        Stacking these with increasing dilation covers the 2-second
        window without expensive attention mechanisms.
        """
        def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
            super().__init__()
            pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                                  padding=pad, dilation=dilation)
            self.chomp = lambda x: x[:, :, :-pad] if pad > 0 else x
            self.norm  = nn.BatchNorm1d(out_ch)
            self.act   = nn.GELU()
            self.drop  = nn.Dropout(0.15)
            self.res   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x):
            h = self.drop(self.act(self.norm(self.chomp(self.conv(x)))))
            return h + self.res(x)


    class TemporalEncoder(nn.Module):
        """
        Input  : (batch, input_dim, seq_len)  — flattened keypoints over time
        Output : (batch, embed_dim)           — fixed-size event embedding
        """
        INPUT_DIM  = 17 * 3       # 17 keypoints × (x, y, conf)
        SEQ_LEN    = 100          # 2 seconds @ 50fps
        EMBED_DIM  = 256

        def __init__(self):
            super().__init__()
            channels = [64, 128, 256, self.EMBED_DIM]
            dilations = [1, 2, 4, 8]
            layers = []
            in_ch = self.INPUT_DIM
            for out_ch, dil in zip(channels, dilations):
                layers.append(TCNBlock(in_ch, out_ch, dilation=dil))
                in_ch = out_ch
            self.tcn  = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)

        def forward(self, x):
            h = self.tcn(x)
            return self.pool(h).squeeze(-1)   # (batch, EMBED_DIM)


    # ─────────────────────────────────────────────────────────────
    # 4. DANN DOMAIN ADAPTOR
    # ─────────────────────────────────────────────────────────────

    class GradientReversal(torch.autograd.Function):
        """
        Reverses gradients during backward pass.
        Forces the feature encoder to learn domain-invariant representations —
        the core trick of Domain-Adversarial Neural Networks (DANN).
        λ controls how strongly domain information is suppressed.
        """
        @staticmethod
        def forward(ctx, x, lam):
            ctx.save_for_backward(torch.tensor(lam))
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            lam = ctx.saved_tensors[0].item()
            return -lam * grad, None


    class DomainClassifier(nn.Module):
        """
        Tries to predict which stadium camera the input came from.
        Trained adversarially against the TemporalEncoder so the encoder
        learns to fool it — making features stadium-agnostic.
        """
        def __init__(self, embed_dim: int, n_domains: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, n_domains),
            )

        def forward(self, features, lam: float = 1.0):
            rev = GradientReversal.apply(features, lam)
            return self.net(rev)


    # ─────────────────────────────────────────────────────────────
    # 5. FULL REFLENSS MODEL
    # ─────────────────────────────────────────────────────────────

    class RefLensModel(nn.Module):
        """
        Combined model for training.
        During inference only the encoder + event_head are used.

        Loss = λ_task × CrossEntropy(event) + λ_domain × CrossEntropy(domain)
        λ_domain is annealed from 0 → 1 over training to stabilise early epochs.
        """
        N_CLASSES = 5
        N_DOMAINS = 8   # support up to 8 different stadium camera setups

        def __init__(self):
            super().__init__()
            self.encoder        = TemporalEncoder()
            self.event_head     = nn.Sequential(
                nn.Linear(TemporalEncoder.EMBED_DIM, 128),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(128, self.N_CLASSES),
            )
            self.domain_adaptor = DomainClassifier(TemporalEncoder.EMBED_DIM,
                                                   self.N_DOMAINS)

        def forward(self, x, lam: float = 1.0):
            feats  = self.encoder(x)
            events = self.event_head(feats)
            domains = self.domain_adaptor(feats, lam)
            return events, domains, feats

        def predict(self, x):
            """Inference-only — returns (class_idx, confidence)."""
            self.eval()
            with torch.no_grad():
                logits, _, _ = self.forward(x, lam=0.0)
                probs = F.softmax(logits, dim=-1)
                conf, cls = probs.max(dim=-1)
            return cls.item(), conf.item()

        @staticmethod
        def training_loss(event_logits, domain_logits,
                          event_labels, domain_labels,
                          lam_domain: float = 0.3):
            task_loss   = F.cross_entropy(event_logits, event_labels)
            domain_loss = F.cross_entropy(domain_logits, domain_labels)
            return task_loss + lam_domain * domain_loss


# ─────────────────────────────────────────────────────────────────
# 6. POSE WINDOW BUFFER
# ─────────────────────────────────────────────────────────────────

class PoseWindowBuffer:
    """
    Maintains a rolling 2-second window of pose frames per player.
    When the window is full, emits a (player_id, tensor) pair for inference.
    """
    WINDOW_FRAMES = 100   # 2s × 50fps

    def __init__(self):
        self._buffers: dict[int, list] = {}

    def push(self, pose: PoseFrame):
        pid = pose.player_id
        if pid not in self._buffers:
            self._buffers[pid] = []
        self._buffers[pid].append(pose.keypoints.flatten())   # (51,)
        if len(self._buffers[pid]) > self.WINDOW_FRAMES:
            self._buffers[pid].pop(0)

    def get_tensor(self, player_id: int):
        """Returns (1, 51, 100) tensor ready for TemporalEncoder, or None."""
        buf = self._buffers.get(player_id, [])
        if len(buf) < self.WINDOW_FRAMES:
            return None
        arr = np.stack(buf[-self.WINDOW_FRAMES:], axis=1)   # (51, 100)
        if TORCH_AVAILABLE:
            return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        return arr


# ─────────────────────────────────────────────────────────────────
# 7. ALERT DISPATCHER
# ─────────────────────────────────────────────────────────────────

class AlertDispatcher:
    """
    Routes EventPrediction objects to the appropriate outputs:
      - High confidence → immediate earpiece alert (simulated here as print)
      - Medium confidence → VAR queue
      - All events → broadcast WebSocket (for TouchField device)
    """

    def __init__(self, ws_port: int = 8765):
        self.ws_port = ws_port
        self._var_queue: list[EventPrediction] = []
        self._alert_log: list[dict] = []

    def dispatch(self, pred: EventPrediction):
        threshold = CONFIDENCE_THRESHOLDS.get(pred.event_type, 0.99)

        if pred.confidence >= threshold and pred.event_type != "clean":
            self._send_earpiece_alert(pred)
        elif pred.confidence >= threshold * 0.85 and pred.event_type != "clean":
            self._queue_var(pred)

        self._broadcast_event(pred)

    def _send_earpiece_alert(self, pred: EventPrediction):
        msg = (f"[EARPIECE ALERT] {pred.event_type.upper()} "
               f"| conf={pred.confidence:.2f} "
               f"| players={pred.player_ids} "
               f"| t={pred.timestamp:.2f}s")
        print(msg)
        self._alert_log.append({
            "type": "earpiece",
            "event": pred.event_type,
            "confidence": round(pred.confidence, 3),
            "timestamp": pred.timestamp,
        })

    def _queue_var(self, pred: EventPrediction):
        self._var_queue.append(pred)
        print(f"[VAR QUEUE]     {pred.event_type.upper()} queued for review "
              f"| conf={pred.confidence:.2f}")

    def _broadcast_event(self, pred: EventPrediction):
        """In production this publishes to WebSocket for TouchField device."""
        pass   # WebSocket server runs in separate thread; push to shared queue

    def var_queue(self) -> list[EventPrediction]:
        q, self._var_queue = self._var_queue, []
        return q

    def export_match_log(self, path: str = "match_events.json"):
        with open(path, "w") as f:
            json.dump(self._alert_log, f, indent=2)
        print(f"[AlertDispatcher] Match log saved → {path}")


# ─────────────────────────────────────────────────────────────────
# 8. FULL INFERENCE PIPELINE (orchestrates everything above)
# ─────────────────────────────────────────────────────────────────

class RefLensPipeline:
    """
    Top-level orchestrator. In production, feed_frame() is called
    at 50fps from the stadium camera capture loop.
    """

    def __init__(self, model_path: Optional[str] = None,
                 camera_id: int = 0):
        self.pose_extractor = PoseExtractor(camera_id)
        self.window_buffer  = PoseWindowBuffer()
        self.dispatcher     = AlertDispatcher()

        self._model = None
        if TORCH_AVAILABLE:
            self._model = RefLensModel()
            if model_path:
                state = torch.load(model_path, map_location="cpu")
                self._model.load_state_dict(state)
                print(f"[RefLens] Loaded weights from {model_path}")
            self._model.eval()

    def feed_frame(self, frame: np.ndarray, timestamp: float):
        """
        Main entry point — call once per camera frame.
        Returns list of EventPrediction (may be empty for clean play).
        """
        poses = self.pose_extractor.extract(frame, timestamp)
        predictions = []

        for pose in poses:
            self.window_buffer.push(pose)
            tensor = self.window_buffer.get_tensor(pose.player_id)

            if tensor is None:
                continue   # still filling the 2s window

            if self._model is not None:
                cls_idx, confidence = self._model.predict(tensor)
                event_name = EVENT_CLASSES[cls_idx]
            else:
                # Demo mode: random prediction weighted toward clean
                weights = [0.85, 0.04, 0.04, 0.04, 0.03]
                cls_idx = int(np.random.choice(5, p=weights))
                confidence = float(np.random.uniform(0.6, 0.99))
                event_name = EVENT_CLASSES[cls_idx]

            pred = EventPrediction(
                event_type=event_name,
                confidence=confidence,
                player_ids=[pose.player_id],
                timestamp=timestamp,
                clip_start=max(0.0, timestamp - 2.0),
            )
            self.dispatcher.dispatch(pred)
            if event_name != "clean":
                predictions.append(pred)

        return predictions


# ─────────────────────────────────────────────────────────────────
# 9. TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────

def train_epoch(model, optimizer, source_loader, target_loader,
                epoch: int, total_epochs: int):
    """
    DANN training: alternate between source (labelled) and target (unlabelled).
    λ annealing schedule from Ganin et al. (2016): gradually increase domain weight.
    """
    if not TORCH_AVAILABLE:
        print("[Train] PyTorch unavailable. Skipping training loop.")
        return 0.0

    model.train()
    p = epoch / total_epochs
    lam = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0   # sigmoid ramp 0 → 1

    total_loss = 0.0
    n_batches = 0

    target_iter = iter(target_loader)

    for x_src, y_event, y_domain_src in source_loader:
        try:
            x_tgt, _, y_domain_tgt = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            x_tgt, _, y_domain_tgt = next(target_iter)

        optimizer.zero_grad()

        # Source forward
        e_logits, d_logits_src, _ = model(x_src, lam)
        # Target forward (no event labels — domain only)
        _, d_logits_tgt, _ = model(x_tgt, lam)

        loss = RefLensModel.training_loss(
            e_logits,
            torch.cat([d_logits_src, d_logits_tgt]),
            y_event,
            torch.cat([y_domain_src, y_domain_tgt]),
            lam_domain=lam * 0.3,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    print(f"[Train] Epoch {epoch+1}/{total_epochs} | loss={avg_loss:.4f} | λ={lam:.3f}")
    return avg_loss


# ─────────────────────────────────────────────────────────────────
# 10. DEMO / SMOKE TEST
# ─────────────────────────────────────────────────────────────────

def run_demo(n_frames: int = 300, fps: float = 50.0):
    """
    Simulates 300 frames (~6 seconds) of match footage.
    Prints any flagged events to stdout.
    """
    print("=" * 60)
    print("  Project Nexus — RefLens Demo")
    print(f"  Simulating {n_frames} frames @ {fps}fps")
    print("=" * 60)

    pipeline = RefLensPipeline(camera_id=0)
    frame_shape = (1080, 1920, 3)   # 1080p feed

    t_start = time.time()
    for i in range(n_frames):
        fake_frame = np.zeros(frame_shape, dtype=np.uint8)
        timestamp = i / fps
        pipeline.feed_frame(fake_frame, timestamp)

    elapsed = time.time() - t_start
    pipeline.dispatcher.export_match_log(str(Path(__file__).parent / "demo_match_log.json"))

    print(f"\n[RefLens] {n_frames} frames processed in {elapsed:.2f}s "
          f"({n_frames / elapsed:.1f} fps effective)")
    print("[RefLens] Demo complete. Check demo_match_log.json for all flagged events.")


if __name__ == "__main__":
    run_demo()