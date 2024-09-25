FROM python:3.9.10-slim

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
COPY ./requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -f https://pypi.anaconda.org/menpo/simple dlib uwsgi && \
    rm ./requirements.txt

# Setup container directories
RUN mkdir /app

# Copy local code to the container
COPY ./ /app

# Launch server with gunicorn
WORKDIR /app
EXPOSE 8080
# CMD ["gunicorn", "main:app", "--timeout=0", "--preload", \
#      "--workers=1", "--threads=4", "--bind=0.0.0.0:8080"]
# CMD ["python", "main.py", "--timeout=0", "--preload", \
# "--workers=1", "--threads=4", "--bind=0.0.0.0:8080"]

# CMD ["uwsgi", "--http", "0.0.0.0:8080", "--module", "main:app", \
#      "--master", "--processes", "4", "--threads", "2", \
#      "--logto", "/dev/stdout", "--log-date", "--log-prefix"]
CMD ["python3.9", "main.py"]