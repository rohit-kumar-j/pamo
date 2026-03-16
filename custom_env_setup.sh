#!/usr/bin/env bash
set -eo pipefail
# Note: intentionally no -u flag. Conda's own activate/deactivate scripts
# (e.g. deactivate-gcc_linux-64.sh) reference unbound variables and will
# fail if -u is set.

# =============================================================================
# PAMO Environment Setup Script
# Sets up a complete conda environment for https://github.com/SarahWeiii/pamo
# Tested on: Ubuntu 22.04, NVIDIA Driver 570.x, RTX 4090
# =============================================================================

# --- Defaults ----------------------------------------------------------------
ENV_NAME="pamo"
PYTHON_VERSION="3.10"
CUDA_TOOLKIT_VERSION="12.8"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
PAMO_REPO="https://github.com/SarahWeiii/pamo.git"
PAMO_DIR=""

# --- Parse CLI args ----------------------------------------------------------
for arg in "$@"; do
  case $arg in
    --env_name=*|--env-name=*)
      ENV_NAME="${arg#*=}"
      ;;
    --python=*)
      PYTHON_VERSION="${arg#*=}"
      ;;
    --pamo_dir=*|--pamo-dir=*)
      PAMO_DIR="${arg#*=}"
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --env-name=NAME    Conda environment name (default: pamo)"
      echo "  --python=VERSION   Python version (default: 3.10)"
      echo "  --pamo-dir=PATH    Path to existing pamo repo (skips git clone)"
      echo "  --help, -h         Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Run '$0 --help' for usage"
      exit 1
      ;;
  esac
done

# --- Colors ------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $1"; exit 1; }

echo ""
echo "============================================"
echo "  PAMO Environment Setup"
echo "  Env: ${ENV_NAME} | Python: ${PYTHON_VERSION}"
echo "============================================"
echo ""

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================
command -v nvidia-smi &>/dev/null || fail "nvidia-smi not found. Install NVIDIA drivers first."
command -v conda &>/dev/null      || fail "conda not found. Install Miniconda/Anaconda first."
command -v git &>/dev/null        || fail "git not found."

info "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
info "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# =============================================================================
# RESOLVE PAMO_DIR EARLY (before any env changes)
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${PAMO_DIR}" ]; then
  # Auto-detect: check if we're inside a pamo repo (has setup.sh + simp_cuda)
  if [ -f "${SCRIPT_DIR}/setup.sh" ] && [ -d "${SCRIPT_DIR}/simp_cuda" ]; then
    PAMO_DIR="${SCRIPT_DIR}"
    info "Auto-detected pamo repo at ${PAMO_DIR}"
  elif [ -f "$(pwd)/setup.sh" ] && [ -d "$(pwd)/simp_cuda" ]; then
    PAMO_DIR="$(pwd)"
    info "Auto-detected pamo repo at ${PAMO_DIR}"
  else
    PAMO_DIR="$(pwd)/pamo"
    info "Will clone pamo to ${PAMO_DIR}"
  fi
fi

# Resolve to absolute path
PAMO_DIR="$(cd "$(dirname "${PAMO_DIR}")" 2>/dev/null && pwd)/$(basename "${PAMO_DIR}")"
info "PAMO_DIR: ${PAMO_DIR}"

# =============================================================================
# 1. CREATE CONDA ENVIRONMENT
# =============================================================================
info "[1/10] Creating conda env '${ENV_NAME}'..."

if conda info --envs | grep -qE "^${ENV_NAME}\s"; then
  warn "Environment '${ENV_NAME}' already exists."
  read -p "  Remove and recreate? [y/N] " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    conda deactivate 2>/dev/null || true
    conda env remove -n "${ENV_NAME}" -y
  else
    info "Reusing existing environment..."
  fi
fi

conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
ok "Conda env '${ENV_NAME}' ready"

# Activate
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
ok "Activated '${ENV_NAME}' ($(python --version))"

# =============================================================================
# 2. INSTALL CUDA TOOLKIT
# =============================================================================
info "[2/10] Installing CUDA toolkit ${CUDA_TOOLKIT_VERSION}..."
conda install -c nvidia cuda-toolkit="${CUDA_TOOLKIT_VERSION}" -y
ok "CUDA toolkit installed"

# =============================================================================
# 3. CONFIGURE CUDA_HOME (persistent across conda activate/deactivate)
# =============================================================================
info "[3/10] Setting CUDA_HOME..."
export CUDA_HOME="${CONDA_PREFIX}"

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${CONDA_PREFIX}/etc/conda/deactivate.d"

cat > "${CONDA_PREFIX}/etc/conda/activate.d/cuda_env.sh" << 'EOF'
#!/bin/bash
export CUDA_HOME="${CONDA_PREFIX}"
export _PAMO_OLD_LD="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF

cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/cuda_env.sh" << 'EOF'
#!/bin/bash
unset CUDA_HOME
if [ -n "${_PAMO_OLD_LD:-}" ]; then
  export LD_LIBRARY_PATH="${_PAMO_OLD_LD}"
else
  unset LD_LIBRARY_PATH
fi
unset _PAMO_OLD_LD
EOF

chmod +x "${CONDA_PREFIX}/etc/conda/activate.d/cuda_env.sh"
chmod +x "${CONDA_PREFIX}/etc/conda/deactivate.d/cuda_env.sh"
source "${CONDA_PREFIX}/etc/conda/activate.d/cuda_env.sh"

ok "CUDA_HOME=${CUDA_HOME}"
ok "LD_LIBRARY_PATH configured"

# =============================================================================
# 4. SYMLINK /usr/local/cuda
#    Conda puts CUDA headers in targets/x86_64-linux/include/ instead of
#    the standard include/. Many CUDA projects hardcode /usr/local/cuda.
# =============================================================================
info "[4/10] Creating /usr/local/cuda symlinks (requires sudo)..."

CUDA_INCLUDE="${CONDA_PREFIX}/targets/x86_64-linux/include"
CUDA_LIB="${CONDA_PREFIX}/targets/x86_64-linux/lib"
CUDA_BIN="${CONDA_PREFIX}/bin"

[ -f "${CUDA_INCLUDE}/cuda_runtime.h" ] || fail "cuda_runtime.h not found. Toolkit install failed?"

sudo rm -rf /usr/local/cuda
sudo mkdir -p /usr/local/cuda
sudo ln -sf "${CUDA_INCLUDE}" /usr/local/cuda/include
sudo ln -sf "${CUDA_LIB}"     /usr/local/cuda/lib64
sudo ln -sf "${CUDA_BIN}"     /usr/local/cuda/bin

# Verify critical headers exist
for header in cuda_runtime.h cudaTypedefs.h; do
  [ -f "/usr/local/cuda/include/${header}" ] || fail "Missing /usr/local/cuda/include/${header}"
done

ok "/usr/local/cuda -> conda CUDA ($(nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,.*//'))"

# =============================================================================
# 5. INSTALL PYTORCH (CUDA 12.8)
# =============================================================================
info "[5/10] Installing PyTorch (cu128)..."
pip install --force-reinstall \
  torch torchvision torchaudio \
  --index-url "${TORCH_INDEX_URL}"

TORCH_VER=$(python -c "import torch; print(torch.__version__)")
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
ok "PyTorch ${TORCH_VER} (CUDA ${TORCH_CUDA})"

# =============================================================================
# 6. INSTALL SETUPTOOLS (pkg_resources needed by some build scripts)
# =============================================================================
info "[6/10] Installing setuptools..."
pip install setuptools
ok "setuptools installed"

# =============================================================================
# 7. INSTALL PDMC (--no-build-isolation to avoid pkg_resources error)
#    pip's isolated build env pulls newer setuptools that dropped pkg_resources.
#    --no-build-isolation uses the env's setuptools which still has it.
# =============================================================================
info "[7/10] Installing pdmc..."
pip install pdmc --no-build-isolation
ok "pdmc installed"

# =============================================================================
# 8. CLONE PAMO REPO + INIT SUBMODULES
# =============================================================================
info "[8/10] Setting up pamo repo at ${PAMO_DIR}..."

if [ -d "${PAMO_DIR}/.git" ]; then
  info "Repo already exists at ${PAMO_DIR}"
elif [ -d "${PAMO_DIR}" ] && [ -f "${PAMO_DIR}/setup.sh" ]; then
  info "Directory exists at ${PAMO_DIR} (looks like a pamo checkout)"
else
  git clone --recursive "${PAMO_REPO}" "${PAMO_DIR}"
  ok "Cloned to ${PAMO_DIR}"
fi

# Always ensure submodules are initialized (handles existing checkouts too)
info "Initializing git submodules..."
cd "${PAMO_DIR}"
git submodule update --init --recursive
ok "Submodules ready"

# =============================================================================
# 9. PATCH WARP BUILD SCRIPT
#    Conda CUDA only ships shared libs (libnvrtc.so), not static (libnvrtc_static.a).
#    Swap static references to shared in the warp linker config.
# =============================================================================
WARP_BUILD="${PAMO_DIR}/simp_cuda/safe_project/warp_/warp/build_dll.py"
info "[9/10] Patching warp build_dll.py..."

if [ -f "${WARP_BUILD}" ]; then
  if grep -q "nvrtc_static" "${WARP_BUILD}"; then
    sed -i 's/nvrtc_static/nvrtc/g'                   "${WARP_BUILD}"
    sed -i 's/nvrtc-builtins_static/nvrtc-builtins/g' "${WARP_BUILD}"
    ok "Patched: static -> shared nvrtc libs"
  else
    ok "Already patched (no nvrtc_static references found)"
  fi
else
  fail "build_dll.py not found at ${WARP_BUILD}. Git submodules may not have initialized correctly."
fi

# =============================================================================
# 10. RUN PAMO SETUP
#     pamo's setup.sh calls pip install for CUDA extensions (cumesh2sdf,
#     simp_cuda) which import torch at build time. pip's build isolation
#     creates a clean env without torch, causing "No module named 'torch'".
#     We patch setup.sh to add --no-build-isolation to all pip install lines.
# =============================================================================
info "[10/10] Running pamo setup.sh..."
cd "${PAMO_DIR}"

# Patch setup.sh: add --no-build-isolation to every 'pip install' invocation
SETUP_SH="${PAMO_DIR}/setup.sh"
if [ -f "${SETUP_SH}" ]; then
  # Only patch if not already patched
  if ! grep -q "\-\-no-build-isolation" "${SETUP_SH}"; then
    sed -i 's/pip install /pip install --no-build-isolation /g' "${SETUP_SH}"
    ok "Patched setup.sh: added --no-build-isolation to pip commands"
  else
    ok "setup.sh already has --no-build-isolation"
  fi
fi

bash setup.sh
ok "pamo setup.sh complete"

# =============================================================================
# VERIFY
# =============================================================================
echo ""
info "Verifying imports..."

python << 'PYEOF'
import sys

checks = [
    ("torch",            lambda: __import__("torch").__version__),
    ("torch.cuda",       lambda: "available" if __import__("torch").cuda.is_available() else "NOT available"),
    ("torchcumesh2sdf",  lambda: (__import__("torchcumesh2sdf"), "OK")[1]),
    ("pdmc",             lambda: (__import__("pdmc"), "OK")[1]),
    ("pamo",             lambda: (__import__("pamo"), "OK")[1]),
    ("warp",             lambda: __import__("warp").__version__),
    ("trimesh",          lambda: __import__("trimesh").__version__),
]

errors = []
for name, fn in checks:
    try:
        result = fn()
        print(f"  {name:20s} {result}")
    except Exception as e:
        errors.append(name)
        print(f"  {name:20s} FAILED: {e}")

sys.exit(1 if errors else 0)
PYEOF

if [ $? -eq 0 ]; then
  echo ""
  echo -e "${GREEN}============================================${NC}"
  echo -e "${GREEN}  Setup complete!${NC}"
  echo -e "${GREEN}============================================${NC}"
  echo ""
  echo "  Activate:  conda activate ${ENV_NAME}"
  echo "  Repo:      ${PAMO_DIR}"
  echo ""
  echo "  Note: /usr/local/cuda points to this env's CUDA."
  echo "  If you have other CUDA projects, you may need to"
  echo "  re-run this or adjust symlinks when switching envs."
  echo ""
else
  echo ""
  echo -e "${RED}============================================${NC}"
  echo -e "${RED}  Setup finished with errors (see above)${NC}"
  echo -e "${RED}============================================${NC}"
  echo -e "${RED}run pip install \"numpy<2.0\" in env${NC}"
  exit 1
fi
