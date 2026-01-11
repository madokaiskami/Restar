FROM python:3.10-slim
WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements-infer.txt /app/
RUN pip install --no-cache-dir -r requirements-infer.txt

COPY src/ /app/src/
COPY outputs/dvc_run/model/ /app/outputs/dvc_run/model/

ENTRYPOINT ["python", "-m", "restar.predict"]


