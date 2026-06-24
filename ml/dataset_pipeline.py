"""
Project Nexus — RefLens Dataset Pipeline
==========================================
Handles everything from raw video → annotated pose sequences → PyTorch DataLoaders.

Pipeline stages
---------------
1. VideoSampler      – extracts frames at 50fps from match footage
2. PoseAnnotator     – runs YOLOv8-pose and saves keypoints per frame
3. EventLabeler      – maps referee annotation CSVs onto pose sequences
4. AugmentationPipe  – spatial jitter, mirror flip, speed warp, occlusion dropout
5. NexusDataset      – PyTorch Dataset returning (window_tensor, event_label, domain_label)
6. DataModule        – wraps source + target loaders for DANN training

Annotation CSV format (produce with LabelStudio or any video annotation tool)
--------------------------------------------------------------------
timestamp_s, event_type, player_ids, camera_id, stadium_id
12.34,       foul,       "3,7",      0,          "wembley"
45.01,       offside,    "11",       1,          "wembley"
...

Author  : Sanchayan (Unwilling-mcu)
GitHub  : github.com/Unwilling-mcu/ProjectNexus
"""

import os
import csv
import json
import time
import random
import hashlib
import logging
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import numpy as np

log = logging.getLogger("NexusDataset")

# ── Optional heavy imports ────────────────────────────────────────
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    log.warning("PyTorch not available — Dataset class will be stubbed.")

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False


# ─────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────────────

EVENT_LABEL_MAP = {
    "clean":    0,
    "offside":  1,
    "foul":     2,
    "dive":     3,
    "handball": 4,
}
LABEL_EVENT_MAP = {v: k for k, v in EVENT_LABEL_MAP.items()}

KEYPOINTS     = 17
KP_DIM        = 3        # x, y, confidence
WINDOW_FRAMES = 100      # 2 seconds @ 50fps
INPUT_DIM     = KEYPOINTS * KP_DIM   # 51

# Pre-clip window: how many frames BEFORE the event we include
PRE_EVENT_FRAMES  = 75   # 1.5s before
POST_EVENT_FRAMES = 25   # 0.5s after


# ─────────────────────────────────────────────────────────────────
# 2. DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────

@dataclass
class EventAnnotation:
    timestamp_s:  float
    event_type:   str
    player_ids:   list[int]
    camera_id:    int
    stadium_id:   str
    clip_path:    str = ""   # path to extracted pose-sequence .npy file

@dataclass
class PoseSequence:
    """A WINDOW_FRAMES × INPUT_DIM array for one player around one event."""
    data:        np.ndarray       # shape (INPUT_DIM, WINDOW_FRAMES)
    event_label: int
    domain_label: int             # integer stadium index
    stadium_id:  str
    camera_id:   int
    player_id:   int
    timestamp_s: float


# ─────────────────────────────────────────────────────────────────
# 3. VIDEO SAMPLER
# ─────────────────────────────────────────────────────────────────

class VideoSampler:
    """
    Extracts frames from a match video at the target FPS.
    Saves frames as JPEGs into an output directory.

    Usage:
        sampler = VideoSampler("match.mp4", "frames/", target_fps=50)
        sampler.extract()
    """
    TARGET_FPS = 50.0

    def __init__(self, video_path: str, out_dir: str,
                 target_fps: float = TARGET_FPS):
        self.video_path = video_path
        self.out_dir    = Path(out_dir)
        self.target_fps = target_fps
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def extract(self) -> list[str]:
        """Returns list of saved frame paths."""
        if not CV2_OK:
            log.error("OpenCV (cv2) not installed. pip install opencv-python")
            return []
        if not Path(self.video_path).exists():
            log.error(f"Video not found: {self.video_path}")
            return []

        cap = cv2.VideoCapture(self.video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step    = max(1, round(src_fps / self.target_fps))

        saved = []
        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                ts = frame_idx / src_fps
                out_path = self.out_dir / f"frame_{saved_idx:06d}_{ts:.3f}.jpg"
                cv2.imwrite(str(out_path), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved.append(str(out_path))
                saved_idx += 1
            frame_idx += 1

        cap.release()
        log.info(f"[VideoSampler] Extracted {saved_idx} frames from {self.video_path}")
        return saved


# ─────────────────────────────────────────────────────────────────
# 4. POSE ANNOTATOR
# ─────────────────────────────────────────────────────────────────

class PoseAnnotator:
    """
    Runs YOLOv8-pose on a directory of frames.
    Saves per-frame keypoint data as .npy files.

    Output file format: dict saved as .npy with allow_pickle=True
    {
        "timestamp_s": float,
        "players": [
            {"id": int, "keypoints": np.ndarray(17, 3)},
            ...
        ]
    }
    """
    MODEL_PATH = "yolov8m-pose.pt"

    def __init__(self, frames_dir: str, poses_dir: str):
        self.frames_dir = Path(frames_dir)
        self.poses_dir  = Path(poses_dir)
        self.poses_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        if YOLO_OK:
            try:
                self._model = YOLO(self.MODEL_PATH)
            except Exception as e:
                log.warning(f"YOLO load failed: {e}. Will generate stub poses.")

    def annotate_all(self, max_frames: int = 0) -> int:
        """Process all frames in frames_dir. Returns number processed."""
        frame_paths = sorted(self.frames_dir.glob("frame_*.jpg"))
        if max_frames:
            frame_paths = frame_paths[:max_frames]

        count = 0
        for fp in frame_paths:
            # Parse timestamp from filename: frame_000001_12.340.jpg
            try:
                ts = float(fp.stem.split("_")[-1])
            except ValueError:
                ts = count / 50.0

            pose_path = self.poses_dir / fp.with_suffix(".npy").name
            if pose_path.exists():
                count += 1
                continue

            players = self._extract_poses(str(fp), ts)
            np.save(str(pose_path), {"timestamp_s": ts, "players": players},
                    allow_pickle=True)
            count += 1

            if count % 500 == 0:
                log.info(f"[PoseAnnotator] {count}/{len(frame_paths)} frames done")

        return count

    def _extract_poses(self, frame_path: str, ts: float) -> list[dict]:
        """Returns list of {id, keypoints} dicts."""
        if self._model is not None and CV2_OK:
            frame = cv2.imread(frame_path)
            results = self._model(frame, verbose=False)[0]
            players = []
            for i, kps in enumerate(results.keypoints.data):
                players.append({
                    "id": i,
                    "keypoints": kps.cpu().numpy().astype(np.float32),
                })
            return players
        else:
            # Stub: generate plausible random poses
            n = random.randint(8, 14)
            return [
                {"id": i,
                 "keypoints": (np.random.rand(KEYPOINTS, KP_DIM) *
                               np.array([1920, 1080, 1])).astype(np.float32)}
                for i in range(n)
            ]


# ─────────────────────────────────────────────────────────────────
# 5. EVENT LABELER
# ─────────────────────────────────────────────────────────────────

class EventLabeler:
    """
    Reads a referee annotation CSV and a directory of pose .npy files.
    Extracts WINDOW_FRAMES-length pose sequences centred around each event,
    and saves them as individual .npy sequence files.

    One sequence file per (event, player) pair.
    """

    def __init__(self, annotation_csv: str, poses_dir: str,
                 sequences_dir: str, stadium_id: str,
                 camera_id: int, domain_label: int):
        self.annotation_csv  = annotation_csv
        self.poses_dir       = Path(poses_dir)
        self.sequences_dir   = Path(sequences_dir)
        self.stadium_id      = stadium_id
        self.camera_id       = camera_id
        self.domain_label    = domain_label
        self.sequences_dir.mkdir(parents=True, exist_ok=True)

    def process(self) -> list[str]:
        """Returns list of saved sequence file paths."""
        annotations = self._load_annotations()
        pose_index  = self._index_poses()

        saved = []
        for ann in annotations:
            seqs = self._extract_sequences(ann, pose_index)
            for seq in seqs:
                path = self._save_sequence(seq)
                if path:
                    saved.append(path)

        log.info(f"[EventLabeler] {len(saved)} sequences saved "
                 f"for stadium '{self.stadium_id}'")
        return saved

    def _load_annotations(self) -> list[EventAnnotation]:
        anns = []
        if not Path(self.annotation_csv).exists():
            log.warning(f"Annotation CSV not found: {self.annotation_csv}")
            return anns
        with open(self.annotation_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pids = [int(x) for x in row["player_ids"].strip('"').split(",")]
                    anns.append(EventAnnotation(
                        timestamp_s = float(row["timestamp_s"]),
                        event_type  = row["event_type"].strip().lower(),
                        player_ids  = pids,
                        camera_id   = int(row.get("camera_id", self.camera_id)),
                        stadium_id  = row.get("stadium_id", self.stadium_id),
                    ))
                except (KeyError, ValueError) as e:
                    log.debug(f"Skipping annotation row: {e}")
        return anns

    def _index_poses(self) -> dict[float, dict]:
        """Returns {timestamp_s: pose_data} dict."""
        index = {}
        for npy_path in self.poses_dir.glob("frame_*.npy"):
            try:
                ts = float(npy_path.stem.split("_")[-1])
                data = np.load(str(npy_path), allow_pickle=True).item()
                index[round(ts, 3)] = data
            except Exception:
                pass
        return index

    def _find_frames_in_window(self, pose_index: dict,
                                center_ts: float) -> list[dict]:
        """Find all pose frames in the PRE+POST event window."""
        t_start = center_ts - PRE_EVENT_FRAMES / 50.0
        t_end   = center_ts + POST_EVENT_FRAMES / 50.0
        frames  = [
            data for ts, data in sorted(pose_index.items())
            if t_start <= ts <= t_end
        ]
        return frames

    def _extract_sequences(self, ann: EventAnnotation,
                            pose_index: dict) -> list[PoseSequence]:
        frames = self._find_frames_in_window(pose_index, ann.timestamp_s)
        if len(frames) < WINDOW_FRAMES // 2:
            return []   # not enough frames around this event

        event_label = EVENT_LABEL_MAP.get(ann.event_type, 0)
        sequences   = []

        for pid in ann.player_ids:
            # Build (INPUT_DIM, WINDOW_FRAMES) array for this player
            kp_seq = []
            for frame in frames:
                players = frame.get("players", [])
                # Find this player by ID (nearest by index if not exact)
                kp = None
                for p in players:
                    if p["id"] == pid:
                        kp = p["keypoints"].flatten()
                        break
                if kp is None and players:
                    # Fall back to closest available player
                    kp = players[min(pid, len(players)-1)]["keypoints"].flatten()
                if kp is None:
                    kp = np.zeros(INPUT_DIM, dtype=np.float32)
                kp_seq.append(kp[:INPUT_DIM])

            # Pad or trim to exact WINDOW_FRAMES
            if len(kp_seq) < WINDOW_FRAMES:
                pad = [np.zeros(INPUT_DIM, dtype=np.float32)] * (WINDOW_FRAMES - len(kp_seq))
                kp_seq = pad + kp_seq   # pre-pad
            kp_seq = kp_seq[-WINDOW_FRAMES:]

            arr = np.stack(kp_seq, axis=1).astype(np.float32)  # (51, 100)

            sequences.append(PoseSequence(
                data         = arr,
                event_label  = event_label,
                domain_label = self.domain_label,
                stadium_id   = ann.stadium_id,
                camera_id    = ann.camera_id,
                player_id    = pid,
                timestamp_s  = ann.timestamp_s,
            ))
        return sequences

    def _save_sequence(self, seq: PoseSequence) -> Optional[str]:
        uid = hashlib.md5(
            f"{seq.stadium_id}{seq.timestamp_s}{seq.player_id}".encode()
        ).hexdigest()[:12]
        fname = f"{seq.event_label}_{seq.domain_label}_{uid}.npy"
        path  = self.sequences_dir / fname
        np.save(str(path), {
            "data":         seq.data,
            "event_label":  seq.event_label,
            "domain_label": seq.domain_label,
            "stadium_id":   seq.stadium_id,
            "camera_id":    seq.camera_id,
            "player_id":    seq.player_id,
            "timestamp_s":  seq.timestamp_s,
        }, allow_pickle=True)
        return str(path)


# ─────────────────────────────────────────────────────────────────
# 6. AUGMENTATION PIPELINE
# ─────────────────────────────────────────────────────────────────

class AugmentationPipe:
    """
    Applies stochastic augmentations to a pose sequence tensor.
    All augmentations are spatial or temporal — no label corruption.

    Applied independently per sample during training (not at dataset build time).
    """

    def __init__(self, p_flip=0.5, p_jitter=0.8, p_speed=0.5,
                 p_occlude=0.3, noise_std=0.02):
        self.p_flip    = p_flip
        self.p_jitter  = p_jitter
        self.p_speed   = p_speed
        self.p_occlude = p_occlude
        self.noise_std = noise_std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (INPUT_DIM, WINDOW_FRAMES)"""
        x = x.copy()
        if random.random() < self.p_flip:
            x = self._horizontal_flip(x)
        if random.random() < self.p_jitter:
            x = self._spatial_jitter(x)
        if random.random() < self.p_speed:
            x = self._speed_warp(x)
        if random.random() < self.p_occlude:
            x = self._keypoint_dropout(x)
        x += np.random.normal(0, self.noise_std, x.shape).astype(np.float32)
        return x

    def _horizontal_flip(self, x: np.ndarray) -> np.ndarray:
        """Mirror all x-coordinates (index 0, 3, 6, ... in each frame)."""
        x = x.copy()
        for k in range(KEYPOINTS):
            x_idx = k * KP_DIM      # x coordinate index
            x[x_idx, :] = 1.0 - x[x_idx, :]
        return x

    def _spatial_jitter(self, x: np.ndarray,
                         scale: float = 0.05) -> np.ndarray:
        """Add small random offset to all keypoint positions."""
        jitter = np.random.uniform(-scale, scale, x.shape).astype(np.float32)
        # Only jitter x,y channels, not confidence
        for k in range(KEYPOINTS):
            conf_idx = k * KP_DIM + 2
            jitter[conf_idx, :] = 0.0
        return np.clip(x + jitter, 0.0, 1.0)

    def _speed_warp(self, x: np.ndarray) -> np.ndarray:
        """Resample sequence at random speed (0.75×–1.25×)."""
        speed  = random.uniform(0.75, 1.25)
        t_orig = np.linspace(0, 1, WINDOW_FRAMES)
        t_new  = np.linspace(0, 1, WINDOW_FRAMES)
        t_src  = np.clip(t_new / speed, 0, 1)
        warped = np.zeros_like(x)
        for dim in range(x.shape[0]):
            warped[dim, :] = np.interp(t_src, t_orig, x[dim, :])
        return warped

    def _keypoint_dropout(self, x: np.ndarray,
                           p_kp: float = 0.15) -> np.ndarray:
        """
        Randomly zero out individual keypoints to simulate occlusion.
        Mimics partial player obscuring by other players or camera angles.
        """
        x = x.copy()
        for k in range(KEYPOINTS):
            if random.random() < p_kp:
                start = k * KP_DIM
                x[start:start+KP_DIM, :] = 0.0
        return x


# ─────────────────────────────────────────────────────────────────
# 7. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────

if TORCH_OK:

    class NexusDataset(Dataset):
        """
        Loads pre-built .npy sequence files from a sequences directory.
        Each file contains one (INPUT_DIM, WINDOW_FRAMES) pose window
        plus event and domain labels.

        Parameters
        ----------
        sequences_dir : path to directory of .npy sequence files
        augment       : whether to apply AugmentationPipe
        domain_filter : if set, only load sequences from this domain_label
        """

        def __init__(self, sequences_dir: str, augment: bool = True,
                     domain_filter: Optional[int] = None,
                     balance: bool = True):
            self.sequences_dir = Path(sequences_dir)
            self.augment       = augment
            self.aug_pipe      = AugmentationPipe() if augment else None
            self._files        = []
            self._event_labels = []
            self._domain_labels = []
            self._load_index(domain_filter)
            if balance:
                self._balance_classes()

        def _load_index(self, domain_filter):
            for f in self.sequences_dir.glob("*.npy"):
                parts = f.stem.split("_")
                if len(parts) < 2:
                    continue
                try:
                    ev  = int(parts[0])
                    dom = int(parts[1])
                except ValueError:
                    continue
                if domain_filter is not None and dom != domain_filter:
                    continue
                self._files.append(f)
                self._event_labels.append(ev)
                self._domain_labels.append(dom)

            log.info(f"[NexusDataset] Loaded {len(self._files)} sequences "
                     f"from {self.sequences_dir}")

        def _balance_classes(self):
            """
            Duplicate minority class samples so all event classes
            appear roughly equally. Addresses foul/dive/handball scarcity.
            """
            from collections import Counter
            counts = Counter(self._event_labels)
            if not counts:
                return
            max_count = max(counts.values())
            new_files, new_ev, new_dom = [], [], []
            for label, cnt in counts.items():
                idxs = [i for i, e in enumerate(self._event_labels) if e == label]
                need = max_count - cnt
                extra = random.choices(idxs, k=need)
                for i in idxs + extra:
                    new_files.append(self._files[i])
                    new_ev.append(self._event_labels[i])
                    new_dom.append(self._domain_labels[i])
            self._files, self._event_labels, self._domain_labels = \
                new_files, new_ev, new_dom
            log.info(f"[NexusDataset] After balancing: {len(self._files)} total")

        def __len__(self):
            return len(self._files)

        def __getitem__(self, idx):
            data = np.load(str(self._files[idx]), allow_pickle=True).item()
            x    = data["data"].astype(np.float32)   # (51, 100)
            ev   = int(data["event_label"])
            dom  = int(data["domain_label"])

            if self.aug_pipe is not None:
                x = self.aug_pipe(x)

            return (
                torch.tensor(x,   dtype=torch.float32),
                torch.tensor(ev,  dtype=torch.long),
                torch.tensor(dom, dtype=torch.long),
            )

        def class_weights(self) -> torch.Tensor:
            """Returns per-class inverse-frequency weights for loss weighting."""
            from collections import Counter
            counts = Counter(self._event_labels)
            total  = len(self._event_labels)
            weights = torch.zeros(len(EVENT_LABEL_MAP))
            for label, cnt in counts.items():
                weights[label] = total / (cnt * len(counts))
            return weights


    class NexusDataModule:
        """
        Wraps source (labelled) and target (unlabelled) DataLoaders
        for DANN training.

        source_dir  : sequences from the home/training stadium
        target_dir  : sequences from the new stadium (domain labels only, no event supervision)
        val_split   : fraction of source data held out for validation
        """

        def __init__(self, source_dir: str, target_dir: str,
                     batch_size: int = 32, num_workers: int = 2,
                     val_split: float = 0.15):
            self.source_dir  = source_dir
            self.target_dir  = target_dir
            self.batch_size  = batch_size
            self.num_workers = num_workers
            self.val_split   = val_split
            self._source_ds  = None
            self._target_ds  = None
            self._val_ds     = None

        def setup(self):
            full_source = NexusDataset(self.source_dir, augment=True)
            n_val = max(1, int(len(full_source) * self.val_split))
            n_train = len(full_source) - n_val
            from torch.utils.data import random_split
            self._source_ds, self._val_ds = random_split(
                full_source, [n_train, n_val],
                generator=torch.Generator().manual_seed(42)
            )
            self._target_ds = NexusDataset(self.target_dir, augment=True)
            log.info(f"[DataModule] source_train={n_train} "
                     f"source_val={n_val} target={len(self._target_ds)}")

        def source_loader(self) -> DataLoader:
            return DataLoader(self._source_ds, batch_size=self.batch_size,
                              shuffle=True, num_workers=self.num_workers,
                              pin_memory=True, drop_last=True)

        def target_loader(self) -> DataLoader:
            return DataLoader(self._target_ds, batch_size=self.batch_size,
                              shuffle=True, num_workers=self.num_workers,
                              pin_memory=True, drop_last=True)

        def val_loader(self) -> DataLoader:
            return DataLoader(self._val_ds, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers,
                              pin_memory=True)


# ─────────────────────────────────────────────────────────────────
# 8. SYNTHETIC DATA GENERATOR (training without real footage)
# ─────────────────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Generates realistic synthetic pose sequences for each event type.
    Used to pre-train RefLens before real annotated footage is available.

    Each event type has characteristic kinematic signatures:
    - foul     : sudden deceleration spike in one player, contact frame
    - dive     : deceleration without contact (arm throw, fall)
    - offside  : lateral position relative to last defender
    - handball : arm trajectory intersecting ball path
    - clean    : normal running / walking motion
    """

    def __init__(self, out_dir: str, n_per_class: int = 500,
                 n_domains: int = 4):
        self.out_dir       = Path(out_dir)
        self.n_per_class   = n_per_class
        self.n_domains     = n_domains
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self) -> int:
        total = 0
        for label, event_name in LABEL_EVENT_MAP.items():
            for i in range(self.n_per_class):
                domain = i % self.n_domains
                seq    = self._generate(event_name, domain)
                uid    = hashlib.md5(
                    f"syn_{event_name}_{domain}_{i}".encode()
                ).hexdigest()[:12]
                fname  = f"{label}_{domain}_{uid}.npy"
                np.save(str(self.out_dir / fname), {
                    "data":         seq,
                    "event_label":  label,
                    "domain_label": domain,
                    "stadium_id":   f"synthetic_{domain}",
                    "camera_id":    domain % 3,
                    "player_id":    i % 22,
                    "timestamp_s":  float(i),
                }, allow_pickle=True)
                total += 1

        log.info(f"[SyntheticGen] Generated {total} synthetic sequences → {self.out_dir}")
        return total

    def _base_motion(self, t: np.ndarray) -> np.ndarray:
        """Generates smooth running-like keypoint trajectory."""
        seq = np.zeros((INPUT_DIM, WINDOW_FRAMES), dtype=np.float32)
        for k in range(KEYPOINTS):
            freq   = 1.0 + k * 0.3
            phase  = k * 0.4
            amp    = 0.02 + 0.01 * (k % 4)
            # x: forward motion
            seq[k*KP_DIM + 0, :] = 0.5 + 0.3 * t + amp * np.sin(freq * t * 2*np.pi + phase)
            # y: vertical bounce
            seq[k*KP_DIM + 1, :] = 0.5 + amp * 0.5 * np.abs(np.sin(freq * t * np.pi + phase))
            # confidence
            seq[k*KP_DIM + 2, :] = np.clip(0.8 + 0.1 * np.random.randn(WINDOW_FRAMES), 0.5, 1.0)
        return seq

    def _generate(self, event_name: str, domain: int) -> np.ndarray:
        t   = np.linspace(0, 1, WINDOW_FRAMES)
        seq = self._base_motion(t)

        # Domain shift: slight scale + offset variation per stadium
        scale  = 0.9 + 0.2 * (domain / self.n_domains)
        offset = 0.05 * domain / self.n_domains
        for k in range(KEYPOINTS):
            for d in range(2):
                seq[k*KP_DIM + d, :] = seq[k*KP_DIM + d, :] * scale + offset

        impact_frame = int(WINDOW_FRAMES * 0.7)   # event occurs at 70% of window

        if event_name == "foul":
            # Sudden deceleration spike (x-velocity drops sharply)
            seq[0, impact_frame:] *= 0.2
            seq[INPUT_DIM//2, impact_frame:impact_frame+5] += 0.3   # contact displacement

        elif event_name == "dive":
            # Deceleration without contact in opponent's keypoints
            seq[0, impact_frame:] *= 0.15
            # Arm throw: wrist keypoints diverge
            seq[KP_DIM*9,  impact_frame:impact_frame+8] += 0.25   # right wrist x
            seq[KP_DIM*10, impact_frame:impact_frame+8] -= 0.25   # left wrist x

        elif event_name == "offside":
            # Forward player is ahead of last defender (high x-value)
            seq[0, :] = np.clip(seq[0, :] + 0.35, 0, 1)

        elif event_name == "handball":
            # Arm raises toward ball trajectory
            seq[KP_DIM*9 + 1, impact_frame-5:impact_frame+5] -= 0.2   # right wrist y rises

        # Add domain noise
        seq += np.random.normal(0, 0.01 + 0.005*domain, seq.shape).astype(np.float32)
        return np.clip(seq, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────
# 9. DATASET STATS REPORTER
# ─────────────────────────────────────────────────────────────────

def report_dataset_stats(sequences_dir: str):
    """Prints a summary of a sequences directory."""
    from collections import Counter
    p = Path(sequences_dir)
    files = list(p.glob("*.npy"))

    if not files:
        print(f"[Stats] No sequences found in {sequences_dir}")
        return

    event_counts  = Counter()
    domain_counts = Counter()

    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            try:
                event_counts[LABEL_EVENT_MAP.get(int(parts[0]), "unknown")] += 1
                domain_counts[int(parts[1])] += 1
            except ValueError:
                pass

    print(f"\n{'='*50}")
    print(f"  Dataset: {sequences_dir}")
    print(f"  Total sequences: {len(files)}")
    print(f"\n  Event breakdown:")
    for ev, cnt in sorted(event_counts.items()):
        bar = "█" * int(cnt / max(event_counts.values()) * 25)
        print(f"    {ev:<12} {cnt:>5}  {bar}")
    print(f"\n  Domain breakdown:")
    for dom, cnt in sorted(domain_counts.items()):
        print(f"    domain {dom}    {cnt:>5} sequences")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────────────────────────
# 10. DEMO / CLI
# ─────────────────────────────────────────────────────────────────

def demo():
    """Generate synthetic dataset and print stats."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    out_dir = str(Path(__file__).parent.parent / "dataset" / "synthetic_sequences")
    print("\n[NexusDataset] Generating synthetic training data...")
    gen = SyntheticDataGenerator(out_dir, n_per_class=120, n_domains=4)
    total = gen.generate_all()
    report_dataset_stats(out_dir)

    if TORCH_OK:
        print("[NexusDataset] Creating PyTorch Dataset...")
        ds = NexusDataset(out_dir, augment=True)
        print(f"  Dataset length : {len(ds)}")
        x, ev, dom = ds[0]
        print(f"  Sample shape   : {tuple(x.shape)}")
        print(f"  Event label    : {ev.item()} ({LABEL_EVENT_MAP[ev.item()]})")
        print(f"  Domain label   : {dom.item()}")
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        batch = next(iter(loader))
        print(f"  Batch shapes   : x={tuple(batch[0].shape)} "
              f"ev={tuple(batch[1].shape)} dom={tuple(batch[2].shape)}")
    print("\n[NexusDataset] Done.\n")


if __name__ == "__main__":
    demo()