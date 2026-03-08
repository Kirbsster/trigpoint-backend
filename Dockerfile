# Dockerfile
FROM python:3.11-slim

ARG INSTALL_YOLO=true

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-yolo.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_YOLO" = "true" ]; then pip install --no-cache-dir -r requirements-yolo.txt; fi

COPY . .

# IMPORTANT: change "app.main:app" if your module path is different
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
