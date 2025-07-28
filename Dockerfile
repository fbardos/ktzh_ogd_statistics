FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

RUN useradd -m streamlituser
USER streamlituser

ENV HOME=/usr/src/app
WORKDIR /usr/src/app

COPY . .
RUN mkdir .streamlit
RUN mkdir tmp_app tmp_matplotlib
ENV TEMPDIR=$HOME/tmp_app
ENV MPLCONFIGDIR=$HOME/tmp_matplotlib

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT [ \
  "streamlit", \
  "run", \
  "app.py", \
  "--server.port=8501", \
  "--server.address=0.0.0.0"]
