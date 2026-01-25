#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
OUT_DIR="$ROOT_DIR/lib/proto/qbrixproto"

cd "$ROOT_DIR/proto"

echo "Generating proto stubs..."
buf generate

echo "Fixing imports for package use..."
# Fix imports in _pb2.py files (protobuf)
for file in "$OUT_DIR"/*_pb2.py; do
    if [[ -f "$file" ]]; then
        sed -i '' 's/^import \([a-z_]*_pb2\) as /from qbrixproto import \1 as /' "$file"
    fi
done

# Fix imports in _pb2_grpc.py files (grpc)
for file in "$OUT_DIR"/*_pb2_grpc.py; do
    if [[ -f "$file" ]]; then
        sed -i '' 's/^import \([a-z_]*_pb2\) as /from qbrixproto import \1 as /' "$file"
    fi
done

echo "Done! Proto stubs generated in $OUT_DIR"
