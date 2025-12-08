FROM node:18-alpine

WORKDIR /app

# Copier les fichiers de dépendances
COPY package.json .

# Installer les dépendances
RUN npm install

# Copier le code source
COPY index.js .

# Commande de démarrage
CMD ["npm", "start"]

