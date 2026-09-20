# DriftTrace — Syllabus Mapping (CI3203D Units I-VI) `[PITCH]`

Reproduced from the pitch "Syllabus Mapping" slide, preserving wording.

| Unit | Title | Coverage in DriftTrace | Requirements | Phase |
| --- | --- | --- | --- | --- |
| Unit I | ML lifecycle & MLOps architecture | Problem framing, lifecycle diagram, operational failure scenario | §1-§4, FR-17 | 0-1 |
| Unit II | Data, feature & experiment management | DVC datasets, validation checks, MLflow runs and model registry | FR-1, FR-2, FR-3, FR-4, FR-5 | 1 |
| Unit III | Pipeline automation | Airflow DAG, feature transformations, training/evaluation, CI tests | FR-6, FR-16 | 2 |
| Unit IV | Deployment & serving | FastAPI endpoints, Docker image, batch and real-time monitoring hooks | FR-7, FR-13 | 3 |
| Unit V | Responsible & explainable AI | SHAP/LIME for prediction explanation; fairness, privacy, governance checklist | FR-14, FR-15 | 4 |
| Unit VI | Cloud MLOps & applications | AWS SageMaker deployment/monitoring as stretch goal; finance use case | FR-20 `[STRETCH]`, FR-3 finance | 5 |

Roadmap-to-practicals (from the pitch roadmap slide):
Phase 1 Reproduce (Practicals 1-3) · Phase 2 Automate (Practicals 4 + 7) ·
Phase 3 Deploy (Practicals 5-6) · Phase 4 Operate (Practicals 8-9) ·
Phase 5 Cloud (Practical 10; stretch) · Future: automatic graph learning, temporal windows + GNNs.

Research foundation `[PITCH]`: "Graphical Causal Reasoning for Root Cause Analysis in Cloud
Networks" (causal graphs + temporal reasoning for cloud incidents). DriftTrace adapts the graph
idea to real-time ML feature pipelines. Reference cited on the pitch slide:
F. Chraim, D. Janzing, and J. Evans, arXiv:2606.13532, 2026.
Note `[DECISION]`: the citation is transcribed exactly as printed on the pitch; it was not
independently verified and is recorded here only as the stated research foundation.
