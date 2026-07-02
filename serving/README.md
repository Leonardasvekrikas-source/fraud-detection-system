# serving/ — Phase 1 (not built yet)

**Planned:** a FastAPI service that scores a transaction and returns the fraud probability
plus a SHAP-based explanation of the top contributing features, a small demo UI (paste a
transaction → see score + explanation), a Dockerfile, and deployment to Hugging Face Spaces.

It will load a persisted model via `fraud_detection.artifacts.ModelBundle` (see
`fraud-detect train`). This folder is a placeholder so the roadmap is visible; there is no
serving code here yet.
