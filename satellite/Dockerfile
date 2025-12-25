FROM python:3.11-slim

# Installer les dépendances système pour GDAL et rasterio
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY satellite.py .

# Shared module is mounted via docker-compose volume

# Commande de démarrage
CMD ["python", "satellite.py"]

