FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    PATH="/workspace/opd-lab/.venv/bin:${PATH}"

WORKDIR /workspace/opd-lab
RUN python -m pip install "uv==0.11.2"
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra data --extra gpu --extra eval --extra tracking --no-dev

COPY . .
ENTRYPOINT ["opd"]
CMD ["doctor", "--config", "configs/main.yaml"]
