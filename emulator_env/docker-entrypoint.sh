#!/usr/bin/env bash
# docker-entrypoint.sh – Start Xvfb (virtual display) then launch the OpenEnv server.
set -euo pipefail

# ---- Validate ISO mount ----
if [ ! -f "${ISO_PATH:-/app/melee.iso}" ]; then
    echo "ERROR: Melee ISO not found at ${ISO_PATH:-/app/melee.iso}"
    echo "Mount your ISO when running the container:"
    echo "  docker run -v /path/to/SSBM.iso:/app/melee.iso smash-env"
    exit 1
fi
echo "OK: Melee ISO found at ${ISO_PATH:-/app/melee.iso}"

# ---- Pre-flight diagnostics ----
echo ""
echo "=== Pre-flight diagnostics ==="
echo "DOLPHIN_PATH = ${DOLPHIN_PATH:-<not set>}"
echo "ISO_PATH     = ${ISO_PATH:-<not set>}"
echo "DISPLAY      = ${DISPLAY:-<not set>}"
echo ""

# Check Dolphin directory
if [ -d "${DOLPHIN_PATH:-}" ]; then
    echo "OK: DOLPHIN_PATH directory exists"
    echo "Contents of ${DOLPHIN_PATH}:"
    ls -la "${DOLPHIN_PATH}/" | head -20
    echo ""
else
    echo "ERROR: DOLPHIN_PATH directory does not exist: ${DOLPHIN_PATH:-<not set>}"
    exit 1
fi

# Check for the expected executable / symlink
EXPECTED_EXE="${DOLPHIN_PATH}/Slippi_Online-x86_64.AppImage"
if [ -e "$EXPECTED_EXE" ]; then
    echo "OK: Found $EXPECTED_EXE"
    file "$EXPECTED_EXE" 2>/dev/null || true
    ls -la "$EXPECTED_EXE"
else
    echo "ERROR: Missing $EXPECTED_EXE"
    echo "libmelee expects this file. Contents of DOLPHIN_PATH:"
    ls -la "${DOLPHIN_PATH}/"
    exit 1
fi

# Check if dolphin-emu can find its shared libraries
echo ""
echo "Checking dolphin-emu shared library deps..."
DOLPHIN_BIN="${DOLPHIN_PATH}/dolphin-emu"
if [ -f "$DOLPHIN_BIN" ]; then
    ldd "$DOLPHIN_BIN" 2>&1 | grep "not found" || echo "OK: All shared libraries found"
else
    echo "WARNING: dolphin-emu binary not found at $DOLPHIN_BIN"
fi

echo ""
echo "=== End diagnostics ==="
echo ""

# ---- Start Xvfb (virtual framebuffer) ----
echo "Starting Xvfb on display ${DISPLAY:-:99} ..."
Xvfb "${DISPLAY:-:99}" -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Give Xvfb a moment to initialise
sleep 2

# Verify Xvfb is running
if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "ERROR: Xvfb failed to start."
    exit 1
fi
echo "Xvfb running (PID $XVFB_PID)."

# ---- Cleanup on exit ----
cleanup() {
    echo "Shutting down ..."
    kill "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- Launch the OpenEnv server ----
echo "Starting OpenEnv server on port 8000 (log-level=debug) ..."
exec uvicorn emulator_env.server.app:app --host 0.0.0.0 --port 8000 --log-level debug
