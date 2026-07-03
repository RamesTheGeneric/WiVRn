#!/bin/bash
# Conformance test for the CPU reference HEVC intra encoder.
#
# Encodes several resolutions and QPs, decodes each with ffmpeg (an independent
# reference decoder), and asserts the decoded frame is byte-identical to the
# encoder's own reconstruction over the display (conformance-window) region.
# A pass proves the encoder emits standards-compliant HEVC.
#
# Requires: g++ (C++20), ffmpeg. Run from this directory.
set -e
cd "$(dirname "$0")"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

echo "building test_encode..."
g++ -std=c++20 -I.. -O2 test_encode.cpp ../cpu_encoder.cpp ../transform.cpp \
    ../residual_coding.cpp ../cabac_reference.cpp ../param_sets.cpp ../bitwriter.cpp \
    -o "$OUT/test_encode"

cmp_py="$OUT/cmp.py"
cat > "$cmp_py" <<'PY'
import sys
w,h,recf,decf=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],sys.argv[4]
cw=((w+63)//64)*64; ch=((h+63)//64)*64
rec=open(recf,"rb").read(); dec=open(decf,"rb").read()
Y=b"".join(rec[y*cw:y*cw+w] for y in range(h))
cyo=cw*ch; Cb=b"".join(rec[cyo+y*(cw//2):cyo+y*(cw//2)+w//2] for y in range(h//2))
cro=cyo+(cw//2)*(ch//2); Cr=b"".join(rec[cro+y*(cw//2):cro+y*(cw//2)+w//2] for y in range(h//2))
recd=Y+Cb+Cr; n=w*h*3//2
m=sum(1 for a,b in zip(recd[:n],dec[:n]) if a!=b)
sys.exit(0 if (m==0 and len(dec)>=n) else 1)
PY

fail=0
for wh in 256:256 64:64 512:288 1832:1920 1920:1080 320:240 128:96 800:600; do
  w=${wh%:*}; h=${wh#*:}
  for q in 18 26 32 40 45 51; do
    "$OUT/test_encode" "$w" "$h" "$q" "$OUT/f" >/dev/null 2>&1
    ffmpeg -hide_banner -loglevel error -y -i "$OUT/f.265" -f rawvideo -pix_fmt yuv420p "$OUT/f_dec.yuv" 2>/dev/null
    if python3 "$cmp_py" "$w" "$h" "$OUT/f_recon.yuv" "$OUT/f_dec.yuv"; then
      printf "  PASS  %dx%d qp%d\n" "$w" "$h" "$q"
    else
      printf "  FAIL  %dx%d qp%d\n" "$w" "$h" "$q"; fail=1
    fi
  done
done
[ $fail -eq 0 ] && echo "ALL CONFORMANT" || { echo "CONFORMANCE FAILURES"; exit 1; }
