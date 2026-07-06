# Deployment runbook

Two separate targets (this is the conventional split):

- **GitHub** — the portfolio repo. **Code only**, model gitignored.
- **Hugging Face Space** — the live demo. A Docker Space that ships the trained **fusion**
  bundle (LightGBM + LSTM, ~162 MB) via **Git LFS** and runs the full dual-XAI system.

Prerequisites you provide (I never handle credentials): a GitHub account
(`Leonardasvekrikas-source`) and a Hugging Face account. `git` is installed; `gh` and
`huggingface-cli` are not (commands below use plain git, with tool-based alternatives noted).

---

## 1. Local container check (do this first — needs Docker Desktop running)

```bash
# Full dual-XAI fusion image (LightGBM + LSTM, SHAP + LIME) — this is what goes live:
docker build -f Dockerfile.fusion -t fraud-demo-fusion .
docker run -p 7860:7860 fraud-demo-fusion
# open http://localhost:7860 and score a transaction

# (The lean, LightGBM-only image is still available via the root Dockerfile:
#   docker build -t fraud-demo . )
```

If that works, the Space will too (HF builds the same Dockerfile).

---

## 2. Push the code to GitHub (code only)

Create an **empty** repo at github.com/new named e.g. `fraud-detection-system` (no README/license
— we already have them). Then:

```bash
git branch -M main
git remote add origin https://github.com/Leonardasvekrikas-source/fraud-detection-system.git
git push -u origin main
```

Git will prompt for auth — use a **Personal Access Token** as the password
(github.com → Settings → Developer settings → Fine-grained tokens, `Contents: read/write`),
or run `gh auth login` if you install the GitHub CLI. The datasets, `.venv`, `system_*`
folders, and `artifacts/` are all gitignored, so only code is pushed.

---

## 3. Deploy the fusion demo to the Hugging Face Space (Docker + model via LFS)

This **replaces** the existing lean Space with the full dual-XAI build. HF Docker Spaces build
whatever file is named `Dockerfile`, so we ship `Dockerfile.fusion` **as** `Dockerfile`.

First train the fusion bundle (needs TensorFlow — the `[lstm]` extra):

```bash
pip install -e ".[lstm,serve]"
fraud-detect train --save artifacts/model-fusion
```

Then assemble and push the Space:

```bash
pip install huggingface_hub
huggingface-cli login                       # paste an HF write token

# The Space already exists; just clone it (it's a git repo):
git clone https://huggingface.co/spaces/Leonardasvekrikas-source/fraud-detection-demo
cd fraud-detection-demo

# Copy the app + the FUSION Dockerfile + the FUSION bundle from the project repo:
#   (adjust the source path to your checkout)
SRC="/c/Users/strei/Documents/!Random_projects/!GitPortfolio/Fraud detection system"
cp -r "$SRC/src" "$SRC/config" "$SRC/pyproject.toml" .
cp "$SRC/Dockerfile.fusion" Dockerfile          # HF reads ./Dockerfile
rm -rf artifacts && mkdir -p artifacts && cp -r "$SRC/artifacts/model-fusion" artifacts/model-fusion
cp "$SRC/deploy/space_README.md" README.md      # Space README (dual-XAI front-matter)

# Track the big model file with LFS, then push:
git lfs install
git lfs track "*.joblib" "*.keras" "*.npy"
git add .gitattributes .
git commit -m "Deploy full dual-XAI fusion demo (LightGBM + LSTM, SHAP + LIME)"
git push
```

The Space rebuilds the Dockerfile (now with TensorFlow) and goes live at
`https://huggingface.co/spaces/Leonardasvekrikas-source/fraud-detection-demo`.

> **Heads-up:** the fusion image is larger (~4 GB with TensorFlow) than the old lean image, so
> the first build takes longer and, on the free CPU tier, cold-starts after the Space sleeps are
> slower. Each score also runs LIME over the LSTM (~1 s). This is the trade-off for showing the
> complete Algorithm-1 fusion with both explanations.

---

## 4. Wire the live link + demo GIF into the README

Once the Space is live:
- Replace the `TODO(Phase 1): live demo link` marker in the root `README.md` with the Space URL.
- Record a short GIF of scoring a transaction (any screen recorder → e.g. `demo.gif`) and drop it
  in the `TODO(Phase 1): demo GIF` marker at the top of the README.

Then commit and push those README changes to GitHub.
