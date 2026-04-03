# SkyWatch AI — Détection & tracking sur imagerie aérienne

## Contexte
Projet portfolio pour candidature ingénieur IA défense.
C'est aussi un projet d'apprentissage : je veux comprendre chaque décision.

## Comment travailler avec moi
- TOUJOURS expliquer POURQUOI avant de coder (quel pattern, quelle raison technique)
- Quand tu utilises un concept ML/CV que je pourrais ne pas maîtriser, explique-le brièvement
- Propose le plan AVANT d'implémenter — attends ma validation
- Après chaque bloc de code, résume les choix non-évidents
- Ne JAMAIS faire de git commit/push/checkout — je gère git moi-même
- Quand un bloc est terminé, propose un message de commit conventionnel

## Stack
- Python 3.11, PyTorch, Ultralytics (YOLOv8), OpenCV
- ByteTrack pour le tracking
- ONNX Runtime pour l'optimisation edge
- FastAPI (API), React + TypeScript + Vite + Tailwind CSS (frontend)
- pytest, ruff, mypy

## Architecture
src/data/       → datasets, augmentations, conversion formats
src/models/     → wrappers détection, entraînement, export
src/tracking/   → intégration ByteTrack, pipeline vidéo
src/eval/       → métriques (mAP, MOTA, latence), benchmarks
src/api/        → FastAPI inference
src/utils/      → logging, visualisation, helpers
configs/        → YAML (train, eval, export)
scripts/        → CLI (train.py, evaluate.py, export.py, benchmark.py)
tests/          → pytest
frontend/       → React + TypeScript (Vite, Tailwind CSS)
docs/adr/       → Architecture Decision Records

## Commandes
ruff check src/ tests/ --fix && ruff format src/ tests/
pytest tests/ -v --tb=short
python scripts/train.py --config configs/train_dota.yaml
cd frontend && npm run dev

## Conventions
- snake_case variables/fonctions, PascalCase classes
- Type hints sur toutes les fonctions publiques
- Docstrings Google-style
- Imports : stdlib → third-party → local
- Max 88 chars/ligne (ruff)
- logging, pas print()
- pathlib.Path, pas de chemins en dur

## Règles
- JAMAIS commit de .pt/.onnx/data dans git
- JAMAIS de secrets dans le code
- Un test pour chaque fonction publique dans src/
- Lancer ruff + pytest après chaque changement

## Leçons apprises
<!-- Ajouter ici chaque correction -->