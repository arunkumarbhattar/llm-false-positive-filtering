#!/bin bash

SRCDIR=sarif_files/
DSTDIR=sarifs_med-precision_high-severity/

PATTERN=$(cat <<EOF
-**/*
+**:cpp/uncontrolled-allocation-size
+**:cpp/certificate-not-checked
+**:cpp/certificate-result-conflation
+**:cpp/overrun-write
+**:cpp/incorrect-allocation-error-handling
+**:cpp/invalid-pointer-deref
+**:cpp/uncontrolled-process-operation
+**:cpp/cleartext-storage-database
+**:cpp/user-controlled-bypass
+**:cpp/cleartext-storage-buffer
+**:cpp/unterminated-variadic-call
+**:cpp/overrunning-write-with-float
+**:cpp/unbounded-write
+**:cpp/overrunning-write
+**:cpp/tainted-permissions-check
+**:cpp/unsafe-create-process-call
+**:cpp/allocation-too-small
+**:cpp/missing-check-scanf
+**:cpp/suspicious-allocation-size
+**:cpp/offset-use-before-range-check
+**:cpp/uninitialized-local
+**:cpp/unsafe-strcat
+**:cpp/bad-strncpy-size
+**:cpp/unsafe-strncat
EOF
)

mkdir -p "$DSTDIR"

for file in $SRCDIR/*.sarif; do
  echo $file
  filter-sarif --input "$file" --output "$DSTDIR/$(basename "$file")" \
    --split-lines -- "$PATTERN" > /dev/null
done
