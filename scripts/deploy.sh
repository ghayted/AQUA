#!/bin/bash

# Script de déploiement automatisé pour AQUA
# Ce script gère le déploiement complet de l'application

set -e

echo "========================================="
echo "🚀 Déploiement AQUA"
echo "========================================="

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
COMPOSE_FILE="docker-compose.yml"

# Fonction de logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Étape 1: Backup des données
log "📦 Sauvegarde des données..."
mkdir -p "$BACKUP_DIR"

# Backup des bases de données
if docker ps | grep -q "aquawatch-timescaledb"; then
    log "Backup TimescaleDB..."
    docker exec aquawatch-timescaledb pg_dump -U postgres aquawatch > "$BACKUP_DIR/timescaledb.sql" || warn "Backup TimescaleDB échoué"
fi

if docker ps | grep -q "aquawatch-postgres"; then
    log "Backup PostgreSQL..."
    docker exec aquawatch-postgres pg_dump -U postgres alertes > "$BACKUP_DIR/postgres.sql" || warn "Backup PostgreSQL échoué"
fi

log "✓ Sauvegarde terminée dans $BACKUP_DIR"

# Étape 2: Arrêt des services
log "🛑 Arrêt des services existants..."
docker-compose down || warn "Certains services n'ont pas pu être arrêtés"

# Étape 3: Nettoyage (optionnel)
log "🧹 Nettoyage des images non utilisées..."
docker image prune -f || warn "Nettoyage échoué"

# Étape 4: Pull des nouvelles images (si depuis un registry)
# Décommentez si vous utilisez un registry Docker
# log "📥 Récupération des nouvelles images..."
# docker-compose pull

# Étape 5: Build des nouvelles images
log "🏗️ Construction des nouvelles images..."
docker-compose build --parallel

# Étape 6: Démarrage des services
log "▶️ Démarrage des services..."
docker-compose up -d

# Étape 7: Attente du démarrage
log "⏳ Attente du démarrage des services..."
sleep 30

# Étape 8: Vérification de la santé
log "🏥 Vérification de la santé des services..."

# Vérifier les conteneurs
RUNNING_CONTAINERS=$(docker-compose ps --services --filter "status=running" | wc -l)
TOTAL_CONTAINERS=$(docker-compose ps --services | wc -l)

log "Conteneurs en cours d'exécution: $RUNNING_CONTAINERS/$TOTAL_CONTAINERS"

# Vérifier les endpoints critiques
log "Test de l'API..."
if curl -f http://localhost:3000/health > /dev/null 2>&1; then
    log "✓ API opérationnelle"
else
    warn "⚠ API non disponible"
fi

log "Test de l'interface web..."
if curl -f http://localhost:80 > /dev/null 2>&1; then
    log "✓ Interface web opérationnelle"
else
    warn "⚠ Interface web non disponible"
fi

# Étape 9: Afficher l'état
log "📊 État des services:"
docker-compose ps

# Étape 10: Afficher les logs récents
log "📋 Logs récents:"
docker-compose logs --tail=10

# Résumé
echo ""
echo "========================================="
log "✅ Déploiement terminé!"
echo "========================================="
echo ""
echo "Services disponibles:"
echo "  - Interface Web: http://localhost:80"
echo "  - API: http://localhost:3000"
echo "  - GeoServer: http://localhost:8080/geoserver"
echo "  - MinIO Console: http://localhost:9002"
echo "  - Jenkins: http://localhost:8081"
echo ""
echo "Backup sauvegardé dans: $BACKUP_DIR"
echo ""

# Code de sortie
if [ "$RUNNING_CONTAINERS" -eq "$TOTAL_CONTAINERS" ]; then
    log "Tous les services sont opérationnels!"
    exit 0
else
    warn "Certains services ne sont pas démarrés!"
    exit 1
fi
