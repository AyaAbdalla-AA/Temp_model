FROM python:3.10-slim

WORKDIR /app

# system dependencies (important for numpy / tensorflow / mlflow)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy all project files
COPY . .

# expose both ports (FastAPI + Gradio)
EXPOSE 8000 7860

# run both services using a simple script
CMD ["sh", "-c", "\
uvicorn main:app  --host 0.0.0.0 --port 8000 & \
python gradio_ui.py \
"]
