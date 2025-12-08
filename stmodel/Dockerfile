FROM python:3.11-slim

WORKDIR /app

# Note: matplotlib fonctionne en mode headless sans dépendances graphiques supplémentaires
# Les dépendances Python sont suffisantes

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY stmodel.py .

# Commande de démarrage
CMD ["python", "stmodel.py"]

