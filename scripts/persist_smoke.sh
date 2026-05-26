#!/usr/bin/env bash
# Persistence smoke test: mount with --image, write some files, unmount,
# mount again on the same image, verify everything survived.
#
# Usage: ./scripts/persist_smoke.sh [--release]

set -euo pipefail

profile="debug"
cargo_flag=""
if [[ "${1:-}" == "--release" ]]; then
    profile="release"
    cargo_flag="--release"
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${repo_root}/target/${profile}/rfs"
mountpoint="$(mktemp -d -t rfs-pmount-XXXXXX)"
image="$(mktemp -u -t rfs-img-XXXXXX).img"

(cd "$repo_root" && cargo build $cargo_flag --quiet)

pass=0
fail=0
rfs_pid=""

cleanup() {
    local ec=$?
    set +e
    if mountpoint -q "$mountpoint" 2>/dev/null; then
        fusermount -u "$mountpoint" 2>/dev/null || \
            fusermount3 -u "$mountpoint" 2>/dev/null
    fi
    [[ -n "$rfs_pid" ]] && kill "$rfs_pid" 2>/dev/null
    rm -rf "$mountpoint"
    rm -f "$image"
    exit $ec
}
trap cleanup EXIT

eq() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$expected" == "$actual" ]]; then
        pass=$((pass + 1))
        printf '  [PASS] %s\n' "$label"
    else
        fail=$((fail + 1))
        printf '  [FAIL] %s\n    expected: %q\n    actual:   %q\n' "$label" "$expected" "$actual"
    fi
}

mount_image() {
    "$binary" "$mountpoint" --image "$image" &
    rfs_pid=$!
    for _ in $(seq 1 30); do
        if mountpoint -q "$mountpoint"; then return 0; fi
        sleep 0.1
    done
    echo "!! mount did not appear within 3s; aborting" >&2
    return 1
}

unmount() {
    fusermount -u "$mountpoint" 2>/dev/null || fusermount3 -u "$mountpoint" 2>/dev/null
    if [[ -n "$rfs_pid" ]]; then
        wait "$rfs_pid" 2>/dev/null || true
        rfs_pid=""
    fi
}

printf 'Using binary: %s\n' "$binary"
printf 'Mountpoint:   %s\n' "$mountpoint"
printf 'Image:        %s\n\n' "$image"

# ===== Round 1: create + populate =====
printf 'Round 1: creating image and writing data...\n'
mount_image

mkdir "$mountpoint/d"
echo "hello from round 1" > "$mountpoint/d/greeting.txt"
mkdir "$mountpoint/d/nested"
echo "deep" > "$mountpoint/d/nested/leaf.txt"
dd if=/dev/urandom of="$mountpoint/big.bin" bs=4096 count=3 status=none
big_sha_before="$(sha256sum "$mountpoint/big.bin" | cut -d' ' -f1)"

unmount
printf 'Unmounted.\n\n'

# Image file should still exist on disk.
if [[ ! -f "$image" ]]; then
    echo "!! image file disappeared after unmount" >&2
    exit 1
fi
size_after_round1="$(stat -c%s "$image")"
printf 'Image size after round 1: %s bytes\n\n' "$size_after_round1"

# ===== Round 2: reopen + verify =====
printf 'Round 2: remounting same image and verifying...\n'
mount_image

eq "directory d survived"               "0"   "$([[ -d "$mountpoint/d" ]] && echo 0 || echo 1)"
eq "greeting.txt content survived"      "hello from round 1" "$(cat "$mountpoint/d/greeting.txt")"
eq "nested directory survived"          "0"   "$([[ -d "$mountpoint/d/nested" ]] && echo 0 || echo 1)"
eq "nested file content survived"       "deep" "$(cat "$mountpoint/d/nested/leaf.txt")"
eq "big binary survived (sha256)"       "$big_sha_before" "$(sha256sum "$mountpoint/big.bin" | cut -d' ' -f1)"

# Add a new file in round 2 to test write-after-open.
echo "appended in round 2" > "$mountpoint/d/round2_only.txt"
unmount
printf 'Unmounted round 2.\n\n'

# ===== Round 3: verify round-2 writes also persist =====
printf 'Round 3: re-verify after another reopen cycle...\n'
mount_image

eq "round-2 added file survives 2nd reopen" "appended in round 2" "$(cat "$mountpoint/d/round2_only.txt")"
eq "round-1 greeting still there"           "hello from round 1"  "$(cat "$mountpoint/d/greeting.txt")"

# Mutating ops to test on the next round.
rm "$mountpoint/d/greeting.txt"                              # unlink
mv "$mountpoint/d/round2_only.txt" "$mountpoint/renamed.txt" # cross-dir rename
echo "first-half-AAAA-second-half" > "$mountpoint/rmw.txt"
# Partial-block overwrite: replace "AAAA" -> "ZZZZ" mid-file.
printf 'ZZZZ' | dd of="$mountpoint/rmw.txt" bs=1 seek=11 count=4 conv=notrunc status=none
mkdir "$mountpoint/empty_then_rmdir" && rmdir "$mountpoint/empty_then_rmdir"
unmount

# ===== Round 4: verify mutations from round 3 also persist =====
printf 'Round 4: verify unlink / rename / RMW / rmdir cross-mount...\n'
mount_image

eq "unlink survived: greeting.txt is gone"   "missing" \
    "$([[ -e "$mountpoint/d/greeting.txt" ]] && echo present || echo missing)"
eq "rename survived: src is gone"            "missing" \
    "$([[ -e "$mountpoint/d/round2_only.txt" ]] && echo present || echo missing)"
eq "rename survived: dst has content"        "appended in round 2" \
    "$(cat "$mountpoint/renamed.txt")"
eq "RMW within block survived reopen"        "first-half-ZZZZ-second-half" \
    "$(cat "$mountpoint/rmw.txt")"
eq "rmdir of empty dir survived"             "missing" \
    "$([[ -e "$mountpoint/empty_then_rmdir" ]] && echo present || echo missing)"
# Larger read-back to confirm the binary blob is bit-identical even
# after several mount cycles.
eq "12KB binary still bit-identical (round 4)" "$big_sha_before" \
    "$(sha256sum "$mountpoint/big.bin" | cut -d' ' -f1)"

unmount

printf '\n%d passed, %d failed\n' "$pass" "$fail"
if [[ $fail -ne 0 ]]; then exit 1; fi
