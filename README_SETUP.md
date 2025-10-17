### Environment Setup (Python 3.10)

This project targets Python 3.10. It includes a local virtual environment workflow and a Docker workflow.

- **.env**: You already have one. Keep your secrets there. See `.env.example` for keys like `NEWSAPI_KEY`.

---

### 1) Local Setup

```bash
# From project root
python3.10 -m venv .venv || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Download common NLTK assets
python - <<'PY'
import nltk
for p in ["punkt","punkt_tab","stopwords","wordnet","omw-1.4","averaged_perceptron_tagger","vader_lexicon"]:
    nltk.download(p)
PY


Or run the helper script (recommended):

```bash
# Activates into the current shell; do not `bash setup_env.sh`
source ./setup_env.sh
```

Verify installs:

```bash
python -c "import sys, pandas, numpy, torch, transformers; print(sys.version); print(pandas.__version__, numpy.__version__); print(torch.__version__); print(transformers.__version__)"
python -c "import nltk, vaderSentiment, sklearn, statsmodels, xgboost; print('OK')"
```

---

### 2) Docker Setup

Build image:

```bash
docker build -t nlp-project:dev .
```

Run container (mount project for live dev):

```bash
docker run --rm -it \
  --env-file ./.env \
  -v "$(pwd)":/app \
  -w /app \
  nlp-project:dev bash
```

Quick verification inside the container:

```bash
python -c "import sys, pandas, numpy, torch, transformers; print(sys.version); print(pandas.__version__, numpy.__version__); print(torch.__version__); print(transformers.__version__)"
```

---

### Notes: Transformers CPU vs GPU

- **Default**: Requirements install CPU-only `torch`. This is reliable across most hosts and CI.
- **GPU (NVIDIA CUDA)**: Use official instructions to install a CUDA-enabled PyTorch matching your driver and CUDA toolkit. Example (adjust versions):
  ```bash
  pip uninstall -y torch torchvision torchaudio
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  ```
- Hugging Face `transformers` will automatically use GPU if `torch.cuda.is_available()` is `True`.

---

### Environment Variables

- Add secrets (e.g., `NEWSAPI_KEY`) to your existing `.env`. See `.env.example` for names.

---

### Troubleshooting

- macOS on Apple Silicon: Python 3.10 + `torch` CPU wheels are supported; if build issues occur, ensure Xcode CLT and Homebrew tools are installed.
- `newspaper3k` may require system libs; Dockerfile includes build deps. Locally, you may need `libxml2`, `libxslt`, and JPEG/zlib dev libs provided by your OS package manager.

---

### End-to-end Pipeline CLI

```bash
# Copy and edit config
cp config.yaml.example config.yaml
# Set NEWSAPI_KEY in .env if using newsapi mode

# Run pipeline
python main.py --config config.yaml
```

### API Server

```bash
# Local
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Docker build and run
docker build -t nlp-api:dev .
docker run --rm -it -p 8000:8000 --env-file ./.env nlp-api:dev uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Test with curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"AAPL","date":"2025-09-25"}'
```

Notes:
- Models are loaded from `models/`. Ensure you have trained and saved a classifier (e.g., RF) before using `/predict`.
- Reproducibility: seeds are set in `main.py` (NumPy, Python, and PyTorch if available). External sources and training randomness may still introduce variability.

---

### Extended historical news (GDELT mode)

- Set in `config.yaml`:
```yaml
news:
  mode: gdelt
  query: AAPL
  max_articles: 800  # higher for ~200 days
```
- GDELT provides historical article metadata for free. We fetch metadata and scrape full text via newspaper3k.
- Then run the pipeline:
```bash
python main.py --config config.yaml
```
