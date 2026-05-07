#!/usr/bin/env bash
# End-to-end smoke test: mount rfs, exercise the FUSE ops we implemented,
# unmount. Exits non-zero on any assertion failure.
#
# Usage: ./scripts/mount_smoke.sh [--release]

set -euo pipefail

profile="debug"
cargo_flag=""
if [[ "${1:-}" == "--release" ]]; then
    profile="release"
    cargo_flag="--release"
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${repo_root}/target/${profile}/rfs"
mountpoint="$(mktemp -d -t rfs-mount-XXXXXX)"
scratch="$(mktemp -d -t rfs-scratch-XXXXXX)"

# Build if stale.
(cd "$repo_root" && cargo build $cargo_flag --quiet)

pass=0
fail=0
rfs_pid=""

cleanup() {
    local ec=$?
    set +e
    if mountpoint -q "$mountpoint" 2>/dev/null; then
        fusermount3 -u "$mountpoint" 2>/dev/null || fusermount -u "$mountpoint" 2>/dev/null
    fi
    [[ -n "$rfs_pid" ]] && kill "$rfs_pid" 2>/dev/null
    rm -rf "$mountpoint" "$scratch"
    if [[ $ec -ne 0 && $fail -eq 0 ]]; then
        echo "!! script aborted (exit $ec) with $pass passed / $fail failed before failure"
    fi
    exit $ec
}
trap cleanup EXIT

check() {
    # $1 = label, rest = command to evaluate (returns 0 = pass).
    local label="$1"; shift
    if "$@"; then
        pass=$((pass + 1))
        printf '  [PASS] %s\n' "$label"
    else
        fail=$((fail + 1))
        printf '  [FAIL] %s\n' "$label"
    fi
}

eq() {
    # $1 = label, $2 = expected, $3 = actual
    local label="$1" expected="$2" actual="$3"
    if [[ "$expected" == "$actual" ]]; then
        pass=$((pass + 1))
        printf '  [PASS] %s\n' "$label"
    else
        fail=$((fail + 1))
        printf '  [FAIL] %s\n    expected: %q\n    actual:   %q\n' "$label" "$expected" "$actual"
    fi
}

printf 'Using binary: %s\n' "$binary"
printf 'Mountpoint:   %s\n' "$mountpoint"

# Mount in background.
"$binary" "$mountpoint" &
rfs_pid=$!

# Wait up to 3s for the kernel to register the mount.
for _ in $(seq 1 30); do
    if mountpoint -q "$mountpoint"; then break; fi
    sleep 0.1
done
if ! mountpoint -q "$mountpoint"; then
    echo "!! mount did not appear within 3s; aborting" >&2
    exit 1
fi
printf 'Mounted. Running tests...\n\n'

# ---- Tests ----

# 1. Empty root.
eq "empty root has no entries"    "0" "$(ls -1A "$mountpoint" | wc -l)"
check "ls on empty root succeeds" bash -c "ls '$mountpoint' > /dev/null"

# 2. mkdir + readdir.
mkdir "$mountpoint/d"
check "mkdir creates directory"       test -d "$mountpoint/d"
eq    "readdir shows the new dir"     "d" "$(ls "$mountpoint")"

# 3. create + write + read small file.
echo "hello from rfs" > "$mountpoint/d/f"
eq "cat small file"                   "hello from rfs" "$(cat "$mountpoint/d/f")"
eq "stat size matches content"        "15" "$(stat -c%s "$mountpoint/d/f")"

# 4. getattr: uid/gid/perm sanity.
eq "file perm == 664 (umask 002)"     "664" "$(stat -c%a "$mountpoint/d/f")"
eq "file uid == current user"         "$(id -u)" "$(stat -c%u "$mountpoint/d/f")"

# 5. cd .. path resolution.
pwd_after_updir="$(cd "$mountpoint/d" && cd .. && pwd)"
eq "cd .. returns to mountpoint"      "$mountpoint" "$pwd_after_updir"

# 6. Nested directories.
mkdir "$mountpoint/d/e"
echo "deep" > "$mountpoint/d/e/g"
eq "nested file readback"             "deep" "$(cat "$mountpoint/d/e/g")"

# 7. Multi-block binary round-trip (12 KB, straddles 3 extents).
dd if=/dev/urandom of="$scratch/rand.bin" bs=4096 count=3 status=none
cp "$scratch/rand.bin" "$mountpoint/big.bin"
check "12KB random binary round-trip" cmp -s "$scratch/rand.bin" "$mountpoint/big.bin"

# 8. Text spanning multiple blocks.
seq 1 2000 > "$scratch/text.txt"  # ~15 KB
cp "$scratch/text.txt" "$mountpoint/nums.txt"
check "multi-block text round-trip"   cmp -s "$scratch/text.txt" "$mountpoint/nums.txt"

# 9. Directory listing order & contents.
listing="$(cd "$mountpoint" && ls | tr '\n' ',')"
eq "root listing"                     "big.bin,d,nums.txt," "$listing"

# 10. Partial-block overwrite (RMW): patch middle of an existing file.
echo -n "AAAAAAAAAA" > "$mountpoint/rmw"  # 10 A's
printf 'XX' | dd of="$mountpoint/rmw" bs=1 seek=4 conv=notrunc status=none
eq "partial-block RMW preserves tail" "AAAAXXAAAA" "$(cat "$mountpoint/rmw")"

# 11. EEXIST on create of existing name.
if mkdir "$mountpoint/d" 2>/dev/null; then
    fail=$((fail + 1))
    printf '  [FAIL] mkdir on existing name returned success\n'
else
    pass=$((pass + 1))
    printf '  [PASS] mkdir on existing name returns EEXIST\n'
fi

# ---- Summary ----
printf '\n%d passed, %d failed\n' "$pass" "$fail"
if [[ $fail -ne 0 ]]; then exit 1; fi
