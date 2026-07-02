FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

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
    && rm -rf /var/lib/apt/lists/*

COPY docker/client/requirements_fixed.txt /app/requirements_fixed.txt
RUN pip install --no-cache-dir -r /app/requirements_fixed.txt

COPY docker/client/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
