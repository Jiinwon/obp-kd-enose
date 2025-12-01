#!/bin/bash
#SBATCH -J gnina-build-torch
#SBATCH -p gpu1
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o gnina_build_torch.%j.out
#SBATCH -e gnina_build_torch.%j.err

# 안정 설정: -u는 conda activate/deactivate 구간에서만 잠깐 해제
set -eo pipefail
set -u

echo "[SLURM] Node: $(hostname)"
nvidia-smi || true

############################
# 0) conda 초기화 + env 활성화
############################
safe_conda_activate() {
  set +u
  if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
  conda activate gnina_env
  set -u
}
safe_conda_deactivate() {
  set +u
  conda deactivate || true
  set -u
}
trap safe_conda_deactivate EXIT
safe_conda_activate

############################
# 1) 모듈/환경 (CUDA 12.1.1)
############################
module purge
module load gnu12/12.3.0
module load cuda/12.1.1

export CUDA_HOME=/opt/ohpc/pub/cuda/12.1.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

############################
# 2) cuDNN 9 (CUDA12.x용 tar 설치 경로)
############################
export CUDNN_ROOT="$HOME/libcudnn9/cudnn-linux-x86_64-9.0.0.312_cuda12-archive"
export CUDNN_INCLUDE_DIR="$CUDNN_ROOT/include"
export CUDNN_LIBRARY="$CUDNN_ROOT/lib/libcudnn.so.9"
export LD_LIBRARY_PATH="$CUDNN_ROOT/lib:$LD_LIBRARY_PATH"

############################
# 3) Boost = conda 1.82로 통일
############################
# conda list 결과에 따라 이미 설치되어 있음
export BOOST_ROOT="$CONDA_PREFIX"
export BOOST_INCLUDEDIR="$CONDA_PREFIX/include"
export BOOST_LIBRARYDIR="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$BOOST_LIBRARYDIR:$LD_LIBRARY_PATH"

############################
# 4) jsoncpp (필수) 확인
############################
if [ -f "${CONDA_PREFIX}/include/json/json.h" ]; then
  export JSONCPP_INCLUDE_DIR="${CONDA_PREFIX}/include"
  if [ -f "${CONDA_PREFIX}/lib/libjsoncpp.so" ]; then
    export JSONCPP_LIBRARY="${CONDA_PREFIX}/lib/libjsoncpp.so"
  fi
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH"
else
  echo "[ERROR] jsoncpp 헤더가 없습니다: ${CONDA_PREFIX}/include/json/json.h"
  echo "        conda install -c conda-forge jsoncpp --solver=classic"
  exit 1
fi

############################
# 5) LibTorch (cu121) 경로
############################
export TORCH_DIR="$HOME/deps/libtorch"
export Torch_DIR="$TORCH_DIR/share/cmake/Torch"
export LD_LIBRARY_PATH="$TORCH_DIR/lib:$LD_LIBRARY_PATH"

# 존재 확인
[ -f "$Torch_DIR/TorchConfig.cmake" ] || { echo "[ERROR] TorchConfig.cmake 없음: $Torch_DIR"; exit 1; }
[ -f "$TORCH_DIR/lib/libtorch.so" ]   || { echo "[ERROR] libtorch.so 없음: $TORCH_DIR/lib"; exit 1; }

############################
# 6) GPU 아키텍처 자동 감지 (예: RTX 3090 -> 86)
############################
detect_arch() {
  local ccap_list; ccap_list=$(nvidia-smi --query-gpu=compute_cap,name --format=csv,noheader 2>/dev/null || true)
  declare -A seen; local out="" cap name sm
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    cap=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
    name=$(echo "$line" | cut -d',' -f2- | sed 's/^ //')
    case "$cap" in
      9.0*) sm="90" ;; 8.6*) sm="86" ;; 8.0*) sm="80" ;; 7.5*) sm="75" ;; 7.0*) sm="70" ;;
      *) case "$name" in
           *H100*) sm="90" ;;
           *A6000*|*RTX\ 3*|*RTX\ 4*) sm="86" ;;
           *A100*) sm="80" ;;
           *V100*) sm="70" ;;
           *T4*)   sm="75" ;;
         esac ;;
    esac
    [ -n "${sm:-}" ] && seen["$sm"]=1
  done <<< "$ccap_list"
  for k in 70 75 80 86 90; do [[ -n "${seen[$k]:-}" ]] && out="${out:+$out;}$k"; done
  echo "${out:-86}"
}
export GNINA_ARCH_LIST="$(detect_arch)"
echo "[INFO] CUDA_ARCHITECTURES=${GNINA_ARCH_LIST}"

############################
# 7) CMake + Build (Torch ON, Boost=conda 1.82)
############################
cd "$HOME/gnina"
mkdir -p build && cd build
rm -f CMakeCache.txt
rm -rf CMakeFiles

# Torch와 conda 패키지를 CMake가 안정적으로 찾도록
export CMAKE_PREFIX_PATH="${Torch_DIR};${CONDA_PREFIX}"

cmake .. \
  -DUSE_CUDNN=ON \
  -DUSE_TORCH=ON \
  -DTorch_DIR="${Torch_DIR}" \
  -DCUDNN_INCLUDE_DIR="${CUDNN_INCLUDE_DIR}" \
  -DCUDNN_LIBRARY="${CUDNN_LIBRARY}" \
  -DJSONCPP_INCLUDE_DIR="${JSONCPP_INCLUDE_DIR}" \
  -DJSONCPP_LIBRARY="${JSONCPP_LIBRARY:-}" \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DBOOST_ROOT="${BOOST_ROOT}" \
  -DBOOST_INCLUDEDIR="${BOOST_INCLUDEDIR}" \
  -DBOOST_LIBRARYDIR="${BOOST_LIBRARYDIR}" \
  -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCUDA_ARCHITECTURES="${GNINA_ARCH_LIST}" \
  -DCUDA_PROPAGATE_HOST_FLAGS=OFF \
  -DCMAKE_CUDA_FLAGS="--expt-relaxed-constexpr -Xcompiler=-std=c++17" \
  -DCMAKE_BUILD_TYPE=Release \
  -Wno-dev

make -j"$(nproc)"

############################
# 8) 검증
############################
echo "[CHECK] ldd boost bindings"
ldd ./gnina | grep -i boost || true

echo "[CHECK] run"
./gnina --version
./gnina --gpu 0 --no_docking
