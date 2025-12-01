#!/usr/bin/env bash
set -euo pipefail

# ==== 경로/환경 ====
BASE="/home1/won0316/_Capstone/obp-kd-enose"
LIG_SDF="$BASE/docking/inputs/ligands"
PROT_PDB="$BASE/docking/inputs/proteins"
LIG_PDBQT="$BASE/docking/inputs/pdbqt/ligands"
PROT_PDBQT="$BASE/docking/inputs/pdbqt/proteins"
OUT_SDF="$BASE/docking/results/sdf"
OUT_LOG="$BASE/docking/results/logs"
OUT_CSV="$BASE/docking/results/csv"
mkdir -p "$LIG_PDBQT" "$PROT_PDBQT" "$OUT_SDF" "$OUT_LOG" "$OUT_CSV"

# libstdc++ 최신 사용(이미 모듈 로드 했다면 넘어감)
module load gnu12/12.3.0 2>/dev/null || true
export LD_LIBRARY_PATH="$(dirname $(g++ -print-file-name=libstdc++.so.6)):${LD_LIBRARY_PATH:-}"

# 도구 체크(없으면 설치 힌트만 출력)
command -v gnina >/dev/null || { echo "[!] gnina 미설치"; exit 1; }
command -v mk_prepare_ligand.py >/dev/null || { echo "[!] meeko 미설치"; exit 1; }
command -v mk_prepare_receptor.py >/dev/null || { echo "[!] meeko 미설치"; exit 1; }
command -v fpocket >/dev/null || { echo "[!] fpocket 미설치"; exit 1; }

# ==== 1) 리간드 SDF -> PDBQT ====
echo "[*] Ligand -> PDBQT"
shopt -s nullglob
for f in "$LIG_SDF"/*.sdf; do
  b="$(basename "$f" .sdf)"
  out="$LIG_PDBQT/${b}.pdbqt"
  [ -s "$out" ] || mk_prepare_ligand.py -i "$f" -o "$out"
done

# ==== 2) 수용체 PDB -> PDBQT ====
echo "[*] Receptor -> PDBQT"
for p in 1A3Y 1OBP 3FIQ; do
  in="$PROT_PDB/${p}.pdb"
  out="$PROT_PDBQT/${p}.pdbqt"
  [ -s "$out" ] || mk_prepare_receptor.py -r "$in" -o "$out"
done

# ==== 3) fpocket 포켓 탐지 & center/size 추출 ====
get_center() {
  local pdb="$1"
  local tag="$(basename "$pdb" .pdb)"
  local outdir="${tag}_out"
  [ -d "$outdir" ] || fpocket -f "$pdb" >/dev/null
  # info.txt 에서 첫 포켓(center) 파싱. 실패 시 단백질 중심으로 폴백.
  python - "$outdir" "$pdb" <<'PY'
import sys,os,re
outdir,pdb = sys.argv[1], sys.argv[2]
info = os.path.join(outdir, "pockets", "info.txt")
def print_center(cx,cy,cz):
    print(f"{cx} {cy} {cz}")
    sys.exit(0)
if os.path.exists(info):
    with open(info) as f:
        for line in f:
            m=re.search(r'[Cc]enter[^:\n]*[:]\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', line)
            if m:
                print_center(m.group(1), m.group(2), m.group(3))
# fallback: 단백질 원자 좌표 평균
xs=ys=zs=n=0.0,0.0,0.0,0
with open(pdb) as f:
    for L in f:
        if L.startswith("ATOM"):
            try:
                x=float(L[30:38]); y=float(L[38:46]); z=float(L[46:54])
            except: continue
            xs+=x; ys+=y; zs+=z; n+=1
print_center(xs/n, ys/n, zs/n)
PY
}

# ==== 4) GNINA 도킹(3 수용체 × 5 리간드) ====
echo "[*] GNINA docking"
SIZE=22  # 필요시 20~26 범위 조정
for p in 1A3Y 1OBP 3FIQ; do
  pdb="$PROT_PDB/${p}.pdb"
  center="$(get_center "$pdb")"
  CX=$(awk '{print $1}' <<<"$center"); CY=$(awk '{print $2}' <<<"$center"); CZ=$(awk '{print $3}' <<<"$center")
  rec="$PROT_PDBQT/${p}.pdbqt"
  for L in "$LIG_PDBQT"/*.pdbqt; do
    base="$(basename "$L" .pdbqt)"
    out_sdf="$OUT_SDF/${p}_${base}.sdf"
    out_log="$OUT_LOG/${p}_${base}.log"
    if [ ! -s "$out_log" ]; then
      gnina -r "$rec" -l "$L" \
            --center_x $CX --center_y $CY --center_z $CZ \
            --size_x $SIZE --size_y $SIZE --size_z $SIZE \
            --exhaustiveness 16 --num_modes 10 \
            --out "$out_sdf" --log "$out_log"
    fi
  done
done

echo "[*] Done docking. Logs in $OUT_LOG, SDF in $OUT_SDF"
