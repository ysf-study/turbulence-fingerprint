# Reproduces every number in the package without installing anything on the host.
#
#   docker build -t tf-repro .
#   docker run --rm tf-repro
#
# No GPU and no raw videos: the container works from the released tables.
FROM python:3.13-slim

LABEL org.opencontainers.image.title="turbulence-fingerprint" \
      org.opencontainers.image.description="Code and derived data for the turbulence-fingerprint manuscript" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/ysf-study/turbulence-fingerprint"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /repo

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# integrity checks, then the five-fold / LOO run and the ablation
CMD ["bash", "scripts/reproduce_all.sh"]
