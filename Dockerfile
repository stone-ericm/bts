FROM python:3.12-slim-bookworm

# System deps for LightGBM + SSL + Tailscale + utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Install Tailscale
RUN curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.noarmor.gpg \
    | tee /usr/share/keyrings/tailscale-archive-keyring.gpg > /dev/null \
    && curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.tailscale-keyring.list \
    | tee /etc/apt/sources.list.d/tailscale.list \
    && apt-get update \
    && apt-get install -y tailscale \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml uv.lock ./
# Install deps only (skip local package — src/ not copied yet)
RUN --mount=type=cache,target=/tmp/uv-cache \
    UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model --frozen --no-dev --no-editable --no-install-project

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/fly-entrypoint.sh scripts/fly-cron-loop.sh scripts/fly-bootstrap.sh ./scripts/

# Now install the local package (deps already cached from above)
RUN --mount=type=cache,target=/tmp/uv-cache \
    UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model --frozen --no-dev --no-editable

RUN chmod +x scripts/fly-entrypoint.sh scripts/fly-cron-loop.sh scripts/fly-bootstrap.sh

RUN mkdir -p /data

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["./scripts/fly-entrypoint.sh"]
