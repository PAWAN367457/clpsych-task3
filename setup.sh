#!/bin/bash
# ============================================================
# CLPsych 2026 Task 3 — New Server Setup Script
# Run once: bash setup.sh
# ============================================================

set -e  # Exit on any error

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="clpsych"
PYTHON_VERSION="3.10"

echo "============================================"
echo " CLPsych 2026 Task 3 — Server Setup"
echo " Project dir: $PROJECT_DIR"
echo "============================================"

# ── 1. Conda environment ─────────────────────────────────────
echo ""
echo "[1/5] Setting up conda environment: $ENV_NAME"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "  → Environment '$ENV_NAME' already exists, skipping creation."
else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    echo "  → Created environment '$ENV_NAME'."
fi

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "  → Activated '$ENV_NAME'."

# ── 2. Python dependencies ────────────────────────────────────
echo ""
echo "[2/5] Installing Python dependencies..."

pip install --quiet --upgrade pip
pip install --quiet \
    torch \
    transformers \
    accelerate \
    bitsandbytes \
    requests \
    tqdm \
    rouge_score \
    nltk \
    sentencepiece \
    huggingface_hub

python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

echo "  → Python packages installed."

# ── 3. Ollama ─────────────────────────────────────────────────
echo ""
echo "[3/5] Setting up Ollama..."

if command -v ollama &> /dev/null; then
    echo "  → Ollama already installed."
else
    echo "  → Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  → Ollama installed."
fi

# Pull Llama 3 model (8B for speed; change to llama3:70b for quality)
echo "  → Pulling llama3:8b model (this may take a while)..."
ollama pull llama3:8b
echo "  → Model ready."

# ── 4. Verify project structure ───────────────────────────────
echo ""
echo "[4/5] Verifying project structure..."

for f in \
    "$PROJECT_DIR/run_pipeline.py" \
    "$PROJECT_DIR/src/data_loader.py" \
    "$PROJECT_DIR/src/prompt_builder.py" \
    "$PROJECT_DIR/src/llm_generator.py" \
    "$PROJECT_DIR/src/task31_runner.py" \
    "$PROJECT_DIR/src/task32_runner.py"
do
    if [ -f "$f" ]; then
        echo "  ✓ $(basename $f)"
    else
        echo "  ✗ MISSING: $f"
    fi
done

# ── 5. Quick smoke test ───────────────────────────────────────
echo ""
echo "[5/5] Running smoke test (mock mode, no model needed)..."

cd "$PROJECT_DIR"
python run_pipeline.py --mock --evaluate && echo "  → Smoke test PASSED." || echo "  → Smoke test FAILED — check errors above."

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " NEXT STEPS:"
echo "   1. Copy your training JSON into:  data/"
echo "   2. Start Ollama in a separate terminal:  ollama serve"
echo "   3. Run Task 3.1:"
echo "      python run_pipeline.py --task 31 --data data/YOUR_TRAIN.json --backend ollama --model llama3:8b"
echo "   4. Run Task 3.2:"
echo "      python run_pipeline.py --task 32 --data data/YOUR_TRAIN.json --backend ollama --model llama3:8b"
echo "   5. Full pipeline + eval:"
echo "      python run_pipeline.py --data data/YOUR_TRAIN.json --backend ollama --model llama3:8b --evaluate"
echo "============================================"
