FROM node:18-alpine

WORKDIR /app

# Copier les fichiers de dépendances
COPY package.json .

# Installer les dépendances
RUN npm install

# Copier le code source
COPY index.js .

# Exposer un port si nécessaire (optionnel)
# EXPOSE 3000

# Commande de démarrage
CMD ["npm", "start"]

