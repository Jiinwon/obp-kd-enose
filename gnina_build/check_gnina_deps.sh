#!/usr/bin/env bash
set -euo pipefail

echo "=== [A] 기본 정보 ==="
echo "CONDA_PREFIX : ${CONDA_PREFIX:-<unset>}"
echo "Torch_DIR    : ${Torch_DIR:-<unset>}"
echo "TORCH_DIR    : ${TORCH_DIR:-<unset>}"
echo "CUDA_HOME    : ${CUDA_HOME:-<unset>}"
echo "CUDNN_ROOT   : ${CUDNN_ROOT:-<unset>}"
echo "LD_LIBRARY_PATH:"
echo "${LD_LIBRARY_PATH:-<unset>}" | tr ':' '\n' | sed 's/^/  - /'

has() { command -v "$1" >/dev/null 2>&1; }

hdr() { echo; echo "=== [$1] $2 ==="; }

check_lib() {
  local libname="$1"; shift
  hdr "LIB" "$libname"
  local found=0
  for d in ${LD_LIBRARY_PATH//:/ } "/usr/lib64" "/usr/lib" "/lib64" "/lib" \
           "${CONDA_PREFIX:-}/lib" "${TORCH_DIR:-}/lib" "${CUDA_HOME:-}/targets/x86_64-linux/lib" \
           "${CUDNN_ROOT:-}/lib"
  do
    [[ -d "$d" ]] || continue
    ls "$d"/"$libname"* 2>/dev/null | sed 's/^/  - /' && found=1 || true
  done
  if [[ $found -eq 0 ]]; then echo "  ! 못찾음: $libname"; fi
}

check_header() {
  local hdrname="$1"; shift
  hdr "HDR" "$hdrname"
  local found=0
  for d in ${CPATH:-} ${CPLUS_INCLUDE_PATH:-} \
           "${CONDA_PREFIX:-}/include" "${CUDNN_ROOT:-}/include" \
           "${TORCH_DIR:-}/include" "${TORCH_DIR:-}/include/torch/csrc/api/include"
  do
    [[ -d "$d" ]] || continue
    if [[ -f "$d/$hdrname" ]]; then echo "  - $d/$hdrname"; found=1; fi
  done
  if [[ $found -eq 0 ]]; then echo "  ! 못찾음: $hdrname"; fi
}

### CUDA/NVIDIA
hdr "CUDA" "nvcc & 드라이버"
if has nvcc; then
  echo "nvcc: $(nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/\1/p')"
else
  echo "  ! nvcc 명령 없음"
fi
if has nvidia-smi; then
  echo "GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | tr '\n' '; ')"
else
  echo "  ! nvidia-smi 없음"
fi

### cuDNN
check_header "cudnn_version.h"
check_lib "libcudnn.so"
check_lib "libcudnn.so.9"

### LibTorch / Caffe2
hdr "TORCH" "LibTorch"
if [[ -n "${Torch_DIR:-}" ]]; then
  echo "Torch_DIR: $Torch_DIR"
  [[ -f "$Torch_DIR/TorchConfig.cmake" ]] && echo "  - TorchConfig.cmake OK" || echo "  ! TorchConfig.cmake 없음"
fi
if [[ -n "${TORCH_DIR:-}" ]]; then
  echo "TORCH_DIR: $TORCH_DIR"
  ls "$TORCH_DIR/lib/libtorch.so" 2>/dev/null && echo "  - libtorch.so OK" || echo "  ! libtorch.so 없음"
  check_lib "libtorch"
  check_lib "libc10"
  check_lib "libc10_cuda"
  check_lib "libtorch_cuda"
fi

### Boost (conda 1.82 우선)
hdr "BOOST" "헤더/라이브러리/중복"
check_header "boost/version.hpp"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "conda boost headers: $(ls ${CONDA_PREFIX}/include/boost/version.hpp 2>/dev/null || echo none)"
  echo "conda boost libs:"
  ls "${CONDA_PREFIX}/lib/libboost_"*.so* 2>/dev/null | sed 's/^/  - /' || echo "  (없음)"
fi
# 잠재적 구버전(1.80) 경로 탐지
for suspect in "$HOME/boost180" "$HOME/boost" "/usr/local/boost" ; do
  [[ -d "$suspect" ]] || continue
  echo "의심 경로: $suspect"
  ls "$suspect"/lib/libboost_*.so* 2>/dev/null | sed 's/^/  - /' || true
done

### OpenBabel
hdr "OPENBABEL" "라이브러리/Config"
check_lib "libopenbabel.so"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  obcfg="${CONDA_PREFIX}/lib/cmake/openbabel3/OpenBabelConfig.cmake"
  [[ -f "$obcfg" ]] && echo "  - OpenBabelConfig.cmake: $obcfg" || echo "  ! OpenBabelConfig.cmake 못찾음"
  echo "  openbabel 버전(있으면):"
  "${CONDA_PREFIX}/bin/obabel" -V 2>/dev/null || echo "  (obabel 실행파일 없음)"
fi

### JsonCpp
hdr "JSONCPP" "헤더/라이브러리"
check_header "json/json.h"
check_lib "libjsoncpp.so"

### libmolgrid (gnina가 찾음)
hdr "LIBMOLGRID" "헤더/라이브러리"
check_header "libmolgrid/libmolgrid.h"
check_lib "libmolgrid.so"

### CMake 캐시(있다면)
hdr "CMAKE" "build 디렉터리 캐시 요약"
if [[ -f "./CMakeCache.txt" ]]; then
  grep -E 'Torch_DIR|Caffe2_DIR|CUDNN|CUDA|Boost|OpenBabel|JSONCPP|CMAKE_CUDA_ARCHITECTURES' CMakeCache.txt || true
else
  echo "  현재 폴더에 CMakeCache.txt 없음 (빌드 디렉터리에서 실행하면 요약 표시됨)"
fi

echo; echo "=== [요약] 필요 파일 누락 여부는 위의 '! 못찾음' 라인 확인 ==="
