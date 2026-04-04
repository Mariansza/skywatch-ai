# SkyWatch AI — Roadmap Projet

## Vision

Image/vidéo aérienne → Détection (YOLOv8) → Tracking (ByteTrack) → Export optimisé (ONNX) → API (FastAPI) → Dashboard (React)

Avec : métriques d'évaluation (mAP, MOTA), benchmarks de latence, et ADRs documentant chaque choix.

---

## Itération 1 : Pipeline de détection — TERMINÉE

**Objectif :** Pipeline image → détection → visualisation de bout en bout.

**Livré :**
- `src/utils/logger.py` — Factory de logger standardisé
- `src/models/schema.py` — Dataclasses `Detection` + `DetectionResult` (frozen, slots)
- `src/models/detector.py` — Wrapper YOLOv8 (`predict`, `predict_batch`, `from_config`)
- `src/utils/visualization.py` — `load_image`, `annotate_detections`, `save_annotated_image`
- `configs/detect.yaml` — Configuration d'inférence
- `scripts/detect.py` — CLI : `python scripts/detect.py --source image.jpg`
- 19 tests passants (ruff + mypy + pytest)

---

## Itération 2 : Tracking multi-objets (PROCHAINE)

**Objectif :** Suivre les objets détectés à travers les frames d'une vidéo avec ByteTrack.

**Modules :**
- `src/tracking/byte_track.py` — Wrapper ByteTrack
- `src/tracking/video_pipeline.py` — Pipeline vidéo (lecture, détection, tracking, visualisation)
- `configs/track.yaml` — Paramètres de tracking
- `scripts/track.py` — CLI vidéo
- Extension de `schema.py` — `TrackedDetection` avec `tracker_id`

---

## Itération 3 : Entraînement sur données aériennes

**Objectif :** Fine-tuner YOLOv8 sur DOTA ou VisDrone.

**Modules :**
- `src/data/dataset.py` — Chargement datasets aériens
- `src/data/augmentations.py` — Augmentations spécifiques aérien
- `src/data/convert.py` — Conversion DOTA → YOLO
- `src/models/trainer.py` — Wrapper d'entraînement
- `configs/train_dota.yaml` + `scripts/train.py`

---

## Itération 4 : Export et optimisation edge

**Objectif :** Export ONNX + inférence ONNX Runtime pour déploiement edge.

**Modules :**
- `src/models/export.py` — Export PyTorch → ONNX (quantization INT8/FP16)
- Backend ONNX dans `Detector` (même interface, backend différent)
- `configs/export.yaml` + `scripts/export.py`
- Benchmarks PyTorch vs ONNX

---

## Itération 5 : Évaluation et métriques

**Objectif :** Mesurer les performances objectivement.

**Modules :**
- `src/eval/metrics.py` — mAP, MOTA/IDF1, latence
- `src/eval/benchmark.py` — Benchmarks automatisés
- `configs/eval.yaml` + `scripts/evaluate.py` + `scripts/benchmark.py`

---

## Itération 6 : API FastAPI

**Objectif :** API REST d'inférence.

**Modules :**
- `src/api/routes.py` — POST /detect, POST /track, GET /health
- `src/api/schemas.py` — Schemas Pydantic
- `src/api/app.py` — Application FastAPI

---

## Itération 7 : Frontend React

**Objectif :** Dashboard web.

**Modules :**
- `frontend/` — React + TypeScript + Vite + Tailwind
- Upload images/vidéos, visualisation détections, playback tracking, métriques

---

## Dépendances entre itérations

```
Itération 1 (Détection) ✅
    ↓
Itération 2 (Tracking)      Itération 3 (Entraînement)
    ↓                              ↓
    ↓                        Itération 4 (Export ONNX)
    ↓                              ↓
    └──────────┬───────────────────┘
               ↓
         Itération 5 (Évaluation)
               ↓
         Itération 6 (API)
               ↓
         Itération 7 (Frontend)
```

Les itérations 2 et 3 sont partiellement parallélisables.
