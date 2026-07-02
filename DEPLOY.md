# Deployment runbook

Two separate targets (this is the conventional split):

- **GitHub** — the portfolio repo. **Code only**, model gitignored.
- **Hugging Face Space** — the live demo. A Docker Space that ships the trained model
  (155 MB) via **Git LFS**.

Prerequisites you provide (I never handle credentials): a GitHub account
(`Leonardasvekrikas-source`) and a Hugging Face account. `git` is installed; `gh` and
`huggingface-cli` are not (commands below use plain git, with tool-based alternatives noted).

---

## 1. Local container check (do this first — needs Docker Desktop running)

```bash
docker build -t fraud-demo .
docker run -p 7860:7860 fraud-demo
# open http://localhost:7860 and score a transaction
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

## 3. Deploy the demo to a Hugging Face Space (Docker + model via LFS)

```bash
pip install huggingface_hub
huggingface-cli login                       # paste an HF write token

# Create a Docker Space (or make one at huggingface.co/new-space, SDK = Docker):
huggingface-cli repo create fraud-detection-demo --type space --space_sdk docker -y

# Clone it and assemble the Space contents:
git clone https://huggingface.co/spaces/Leonardasvekrikas-source/fraud-detection-demo
cd fraud-detection-demo

# Copy the app + the trained model from the project repo:
#   (adjust the source path to your checkout)
SRC="/c/Users/strei/Documents/!Random_projects/!GitPortfolio/Fraud detection system"
cp -r "$SRC/src" "$SRC/config" "$SRC/pyproject.toml" "$SRC/Dockerfile" .
mkdir -p artifacts && cp -r "$SRC/artifacts/model" artifacts/model
cp "$SRC/deploy/space_README.md" README.md      # the Space README (Docker front-matter)

# Track the big model file with LFS, then push:
git lfs install
git lfs track "*.joblib" "*.keras"
git add .gitattributes .
git commit -m "Deploy explainable fraud detection demo"
git push
```

The Space will build the Dockerfile and go live at
`https://huggingface.co/spaces/Leonardasvekrikas-source/fraud-detection-demo`.
Edit the `<REPO>` link in `README.md` to point at the GitHub repo.

> The lean image has no TensorFlow, so the demo runs in `lightgbm_only` mode (probability +
> SHAP + single-model verdict). To show the full Algorithm-1 fusion verdict, train with the
> LSTM (`fraud-detect train --save artifacts/model`) and add `tensorflow~=2.17` to the
> Dockerfile's pip install — a heavier image, but the complete decision-level fusion.

---

## 4. Wire the live link + demo GIF into the README

Once the Space is live:
- Replace the `TODO(Phase 1): live demo link` marker in the root `README.md` with the Space URL.
- Record a short GIF of scoring a transaction (any screen recorder → e.g. `demo.gif`) and drop it
  in the `TODO(Phase 1): demo GIF` marker at the top of the README.

Then commit and push those README changes to GitHub.
