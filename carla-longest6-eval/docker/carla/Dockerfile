FROM carlasim/carla:0.9.16

USER root
RUN apt-get update && apt-get install -y \
    ffmpeg \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
USER carla

EXPOSE 2000-2002

ENTRYPOINT ["/entrypoint.sh"]
