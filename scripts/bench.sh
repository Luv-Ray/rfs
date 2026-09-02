#!/usr/bin/env bash
# Run the microbenchmark suite with reduced measurement noise.
#
# Two layers of stabilization:
#
#   1. Per-process (always, no root):
#        - taskset : pin to one otherwise-idle core so the scheduler doesn't
#                    migrate the bench between cores mid-measurement.
#        - setarch -R : disable ASLR *for this process only* so memory layout
#                    (and thus cache/TLB behavior) is reproducible run to run.
#                    The global /proc/sys/kernel/randomize_va_space is untouched.
#
#   2. Frequency lock (only when run as root):
#        Pins every CPU to its base frequency (no turbo, no downclock) by
#        setting the governor to `performance` and clamping scaling_{min,max}
#        to base_frequency. This is GLOBAL machine state, so the script saves
#        the prior values and restores them on exit (including Ctrl-C / error).
#        We lock to *base*, not the turbo ceiling: the goal is a steady clock,
#        not a fast one — turbo is itself a source of run-to-run variance.
#        Without root, the script prints a warning and skips this step; the
#        per-process layer still applies.
#
# Usage:
#   scripts/bench.sh                       # whole suite
#   scripts/bench.sh read_                 # only benches matching a filter
#   sudo scripts/bench.sh write_overwrite  # + frequency lock (root)
#
# Any argument is passed through as the libtest bench filter.

set -euo pipefail

CPUFREQ_ROOT=/sys/devices/system/cpu

# ---- choose a core to pin to -------------------------------------------------
# Highest-numbered core is least likely to hold system/IRQ work; good enough
# for a single-threaded microbench on an otherwise-idle box.
NCORES=$(nproc)
PIN_CORE=$((NCORES - 1))

# ---- frequency lock / restore (root only) ------------------------------------
LOCKED=0
declare -a SAVED_GOV SAVED_MAX SAVED_MIN
LOCK_CPUS=()

restore_freq() {
    [ "$LOCKED" -eq 1 ] || return 0
    local i cpu
    for i in "${!LOCK_CPUS[@]}"; do
        cpu="${LOCK_CPUS[$i]}"
        local d="$CPUFREQ_ROOT/$cpu/cpufreq"
        # Restore min before max isn't required, but restore all three.
        [ -n "${SAVED_MAX[$i]:-}" ] && echo "${SAVED_MAX[$i]}" > "$d/scaling_max_freq" 2>/dev/null || true
        [ -n "${SAVED_MIN[$i]:-}" ] && echo "${SAVED_MIN[$i]}" > "$d/scaling_min_freq" 2>/dev/null || true
        [ -n "${SAVED_GOV[$i]:-}" ] && echo "${SAVED_GOV[$i]}" > "$d/scaling_governor" 2>/dev/null || true
    done
    echo ">>> restored CPU frequency governor and limits"
    LOCKED=0
}

lock_freq() {
    local cpu d base gov max min idx=0
    LOCK_CPUS=()
    for d in "$CPUFREQ_ROOT"/cpu[0-9]*/cpufreq; do
        [ -d "$d" ] || continue
        cpu=$(basename "$(dirname "$d")")
        base=$(cat "$d/base_frequency" 2>/dev/null || cat "$d/cpuinfo_min_freq")
        gov=$(cat "$d/scaling_governor")
        max=$(cat "$d/scaling_max_freq")
        min=$(cat "$d/scaling_min_freq")
        SAVED_GOV[$idx]="$gov"
        SAVED_MAX[$idx]="$max"
        SAVED_MIN[$idx]="$min"
        LOCK_CPUS[$idx]="$cpu"
        idx=$((idx + 1))
        # performance governor first so it doesn't fight the clamp, then pin
        # both bounds to base -> clock is welded at base_frequency.
        echo performance > "$d/scaling_governor"
        echo "$base"     > "$d/scaling_max_freq"
        echo "$base"     > "$d/scaling_min_freq"
    done
    LOCKED=1
    local base_mhz=$(( $(cat "$CPUFREQ_ROOT/cpu0/cpufreq/base_frequency" 2>/dev/null || echo 0) / 1000 ))
    echo ">>> locked all CPUs to base frequency (${base_mhz} MHz), governor=performance"
}

# cargo must run as a real user, never as root: it resolves toolchains via
# ~/.rustup and would otherwise (a) not find the nightly toolchain (root's
# HOME is /root) and (b) leave root-owned files in target/. So when we're
# root purely to lock frequency, we drop back to the invoking user (SUDO_USER)
# for the cargo step, with HOME pointed at their real home.
RUN_PREFIX=()
if [ "$(id -u)" -eq 0 ]; then
    trap restore_freq EXIT INT TERM
    lock_freq
    if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
        USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
        # sudo resets PATH to root's, which lacks ~/.cargo/bin, so cargo isn't
        # found. Prepend the user's cargo bin explicitly (HOME alone only fixes
        # toolchain resolution, not the lookup of the cargo shim itself).
        RUN_PREFIX=(sudo -u "$SUDO_USER" env "HOME=$USER_HOME" "PATH=$USER_HOME/.cargo/bin:$PATH")
        echo ">>> running cargo as $SUDO_USER (HOME=$USER_HOME)"
    else
        echo "!!! no SUDO_USER: running cargo as root — this will taint target/"
        echo "!!! and needs a nightly toolchain under /root/.rustup. Prefer:"
        echo "!!!   sudo $0 $*"
    fi
else
    echo "!!! not root: skipping CPU frequency lock (turbo/downclock will add jitter)."
    echo "!!! for steadier numbers run: sudo $0 $*"
fi

# ---- run ---------------------------------------------------------------------
# setarch -R : ASLR off for this process tree only.
# taskset    : pin to the chosen core.
echo ">>> pinning to core $PIN_CORE (ASLR off for this run)"
taskset -c "$PIN_CORE" setarch -R \
    "${RUN_PREFIX[@]}" cargo +nightly bench --bench fs_bench -- "$@"
