FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl zstd pciutils lshw ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN useradd -m -u 1000 appuser
USER appuser

ENV HOME=/home/appuser \
    PATH="/home/appuser/.local/bin:$PATH" \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_ORIGINS=* \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_MODELS=/home/appuser/.ollama/models

WORKDIR /home/appuser/app

COPY --chown=appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser . .
RUN chmod +x start.sh

EXPOSE 7860
CMD ["./start.sh"]
