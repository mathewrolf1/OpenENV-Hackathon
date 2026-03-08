# Dockerfile for the Smash Melee RL environment (OpenEnv + libmelee + Slippi Dolphin).
#
# Build:
#   docker build -t smash-env -f server/Dockerfile .
#
# Run (mount your Melee ISO):
#   docker run -p 8000:8000 -v "/path/to/SSBM.iso:/app/melee.iso" smash-env
#
# The container:
#   1. Downloads the official Slippi Dolphin AppImage and extracts it.
#   2. Installs Xvfb to provide a virtual display (Dolphin needs X11).
#   3. Starts Xvfb, then launches the OpenEnv FastAPI server on port 8000.

# ---------------------------------------------------------------------------
# Stage 1 – builder: install Python dependencies
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3 / python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

WORKDIR /app/env
COPY . /app/env

# Install Python deps using the system Python 3.11 (--python flag ensures uv
# doesn't download its own managed Python, so the venv links to /usr/bin/python3.11
# which we also install in the runtime stage).
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable --python /usr/bin/python3.11; \
    else \
        uv sync --no-editable --python /usr/bin/python3.11; \
    fi

# ---------------------------------------------------------------------------
# Stage 2 – runtime: Slippi Dolphin + Xvfb + Python app
# ---------------------------------------------------------------------------
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies:
#   - python3.11: the venv's python symlinks to /usr/bin/python3.11
#   - xvfb: virtual framebuffer for Dolphin
#   - libgl/X11/audio: libraries Dolphin links against
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
    xvfb \
    libgl1-mesa-dri libgl1-mesa-glx libegl1-mesa \
    libx11-6 libxrandr2 libxi6 libxext6 libxcb1 libxfixes3 libxcursor1 \
    libpulse0 libasound2 \
    curl ca-certificates file libfuse2 \
    && rm -rf /var/lib/apt/lists/*

# Download and extract the official Slippi Dolphin AppImage.
ARG SLIPPI_VERSION=v3.5.2
ARG SLIPPI_APPIMAGE_URL=https://github.com/project-slippi/Ishiiruka/releases/download/${SLIPPI_VERSION}/Slippi_Online-x86_64.AppImage

RUN curl -L -o /tmp/slippi.AppImage "${SLIPPI_APPIMAGE_URL}" \
    && chmod +x /tmp/slippi.AppImage \
    && cd /opt \
    && /tmp/slippi.AppImage --appimage-extract \
    && mv squashfs-root slippi \
    && rm /tmp/slippi.AppImage \
    && chmod +x /opt/slippi/usr/bin/* \
    && ln -s /opt/slippi/usr/bin/dolphin-emu /opt/slippi/usr/bin/Slippi_Online-x86_64.AppImage

WORKDIR /app

# Copy app code (includes the venv at /app/env/.venv) from builder.
# The venv stays at the same path so shebang lines remain valid.
COPY --from=builder /app/env /app/env

# Copy the entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# ---- Environment variables ----
ENV PATH="/app/env/.venv/bin:${PATH}"

# Dolphin: directory containing the dolphin-emu binary
ENV DOLPHIN_PATH="/opt/slippi/usr/bin"

# ISO mount point — users mount their ISO here at runtime
ENV ISO_PATH="/app/melee.iso"

# Display for Xvfb
ENV DISPLAY=:99

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
