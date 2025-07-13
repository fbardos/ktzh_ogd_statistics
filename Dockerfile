FROM python:3.10-slim

ENV MPLCONFIGDIR=/tmp/.matplotlib

WORKDIR /ktzh_ogd_statistics

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# RUN git clone https://github.com/fbardos/ktzh_ogd_statistics.git .
COPY requirements.txt /tmp

RUN pip3 install -r /tmp/requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# ENTRYPOINT gets set in docker-compose
