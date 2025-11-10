FROM python:3.10-slim

# Instala ffmpeg y dependencias del sistema
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia requirements e instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código
COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
