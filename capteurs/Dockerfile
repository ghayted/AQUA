FROM node:18-alpine

WORKDIR /app

# Install dos2unix to fix Windows line endings
RUN apk add --no-cache dos2unix

# Copier les fichiers de dépendances
COPY package.json .

# Installer les dépendances
RUN npm install

# Copier le code source
COPY index.js .

# Convert Windows line endings (CRLF) to Unix (LF)
RUN dos2unix index.js

# Commande de démarrage
CMD ["npm", "start"]
