# Entity Resolution pipeline -- runtime image.
#
# python:3.11-slim (Debian, glibc), not alpine: pandas/numpy/scipy/xgboost/
# rapidfuzz all ship prebuilt manylinux wheels for glibc-based images, which
# alpine's musl libc can't use -- alpine would force slow source builds (or
# outright fail for some of these) and end up bigger and slower to build
# than this despite the smaller base image.
FROM python:3.11-slim

WORKDIR /app

# xgboost's compiled extension links against OpenMP at runtime. The full
# python:3.11 image happens to pull this in as a side effect of other
# packages; the slim image does not, and its absence fails at IMPORT time
# inside the container (not at pip-install time) -- the same class of error
# already hit locally on macOS as a missing libomp.dylib; this is its
# Debian/Linux equivalent (missing libgomp.so.1).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first, from PyTorch's CPU wheel index (~200MB instead of the
# ~2GB+ CUDA build a plain `pip install torch` pulls from PyPI). Installed
# before requirements.txt so its `torch==2.3.1` pin is already satisfied and
# pip doesn't fetch the CUDA wheel over it. Needed by the optional
# --use_embeddings path (sentence-transformers).
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

COPY aads_submission/business_entity_resolution/code/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Application code only -- the dataset, model checkpoints, and outputs are
# mounted at runtime (see README "Docker" section), never baked into the
# image, so the image size doesn't grow with the ~12M-row dataset.
COPY aads_submission/ /app/aads_submission/
COPY optimize_submission.py /app/optimize_submission.py

ENV PYTHONPATH=/app/aads_submission/business_entity_resolution/code
ENV PYTHONUNBUFFERED=1
# Hugging Face model downloads (only used with --use_embeddings) are cached
# under the mounted /models volume, so the model is downloaded once and
# survives container restarts instead of being re-fetched every run.
ENV HF_HOME=/models/hf

# /data, /output, /models are plain directories in the image, not declared
# as VOLUMEs -- every documented `docker run` in the README explicitly
# bind-mounts (-v) all three anyway, and declaring them as VOLUMEs on top of
# that adds nothing except a foot-gun: if you ever forget one of those -v
# flags, Docker silently creates an empty anonymous volume there instead of
# an error, which looks like "it ran" but silently drops your output.
ENTRYPOINT ["python3", "/app/aads_submission/business_entity_resolution/code/src/pipeline.py"]
CMD ["--data_dir", "/data", "--matching_out", "/output/matching_results.tsv", "--candidate_out", "/output/candidate_pairs.tsv", "--model_path", "/models/entity_model.json"]
