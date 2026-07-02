FROM carlasim/carla:0.9.16

RUN apt-get update && apt-get install -y \
    ffmpeg \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 2000-2002

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
