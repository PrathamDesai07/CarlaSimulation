FROM carlasim/carla:0.9.16

USER root
RUN apt-get update && apt-get install -y \
    ffmpeg \
    netcat-openbsd \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    xvfb \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA driver's Vulkan/OpenGL ICDs are injected by the NVIDIA Container Toolkit at runtime
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
USER carla

EXPOSE 2000-2002

ENTRYPOINT ["/entrypoint.sh"]
