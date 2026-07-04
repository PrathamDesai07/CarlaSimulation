FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    netcat-openbsd \
    git \
    wget \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY docker/client/requirements_fixed.txt /app/requirements_fixed.txt
RUN pip install --no-cache-dir -r /app/requirements_fixed.txt

# mmcv is NOT needed for evaluation — model_eval.py uses pure PyTorch/torchvision.
# For training, build mmcv from source with: TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Copy client-side scripts into the image
COPY docker/client/scripts/ /app/docker/client/scripts/
COPY docker/client/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
