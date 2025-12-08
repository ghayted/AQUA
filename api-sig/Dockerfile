FROM node:18-alpine

WORKDIR /app

# Outils nécessaires (curl pour healthcheck)
RUN apk add --no-cache curl

# Copier les fichiers de dépendances
COPY package.json .

# Installer les dépendances
RUN npm install

# Copier le code source
COPY index.js .

# Exposer le port
EXPOSE 3000

# Commande de démarrage
CMD ["npm", "start"]

