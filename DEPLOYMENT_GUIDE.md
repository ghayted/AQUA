# Guide de Déploiement du Projet AQUA

## Prérequis Système

### Configuration Matérielle Recommandée
- **RAM**: 8 Go minimum (4 Go minimum)
- **CPU**: 2 cœurs minimum
- **Stockage**: 20 Go d'espace disque disponible
- **Système d'exploitation**: Linux (Ubuntu 20.04+), Windows 10+, macOS 10.15+

### Logiciels Requis
- **Docker Engine**: Version 20.10.0 ou supérieure
- **Docker Compose**: Version 1.29.0 ou supérieure
- **Git**: Pour cloner le dépôt (optionnel si téléchargement direct)
- **Navigateur Web**: Chrome, Firefox, Safari ou Edge (version récente)

### Vérification des Prérequis
Avant de procéder au déploiement, vérifiez que Docker est correctement installé :

```bash
# Vérifier la version de Docker
docker --version

# Vérifier la version de Docker Compose
docker-compose --version

# Vérifier que Docker fonctionne
docker run hello-world
```

## Installation

### Étape 1: Obtenir le Code Source

Clonez le dépôt ou téléchargez l'archive :

```bash
# Via Git (recommandé)
git clone https://github.com/votre-organisation/aqua.git
cd aqua

# Ou téléchargez et extrayez l'archive
wget https://github.com/votre-organisation/aqua/archive/main.zip
unzip main.zip
cd aqua-main
```

### Étape 2: Examiner la Configuration

Le déploiement est configuré via le fichier `docker-compose.yml`. Aucune configuration supplémentaire n'est requise pour le déploiement initial.

Structure du projet :
```
AQUA/
├── capteurs/           # Service de génération de données des capteurs
├── api-sig/            # Service API REST
├── alertes/            # Service de surveillance et d'alertes
├── satellite/          # Service de traitement satellitaire
├── stmodel/            # Service de modèle d'apprentissage automatique
├── web/                # Interface utilisateur web
├── timescaledb/        # Scripts d'initialisation de la base de données
├── postgres-alertes/   # Scripts d'initialisation de la base d'alertes
├── docker-compose.yml  # Configuration principale
└── .env.example        # Exemple de variables d'environnement
```

### Étape 3: Démarrer les Services

Démarrez tous les services en mode détaché :

```bash
# À partir du répertoire racine du projet
docker-compose up -d

# Vérifier que tous les conteneurs démarrent
docker-compose ps
```

Le premier démarrage peut prendre plusieurs minutes car Docker doit :
1. Télécharger les images Docker requises
2. Construire les images personnalisées
3. Initialiser les bases de données
4. Démarrer tous les services

## Vérification du Déploiement

### Surveiller les Logs

Surveillez les logs pendant le démarrage initial :

```bash
# Voir tous les logs en temps réel
docker-compose logs -f

# Voir les logs d'un service spécifique
docker-compose logs -f capteurs
```

### Vérifier l'État des Services

Une fois le démarrage terminé, vérifiez que tous les services sont opérationnels :

```bash
# Liste des conteneurs en cours d'exécution
docker-compose ps

# Vérifier les contrôles de santé
docker-compose ps | grep healthy
```

Un déploiement réussi devrait montrer tous les services dans l'état "Up" avec "(healthy)" pour ceux qui ont des contrôles de santé.

### Tester l'Accès aux Services

#### Interface Web
Ouvrez un navigateur web et accédez à :
- http://localhost

Vous devriez voir le tableau de bord AQUA avec :
- Carte interactive affichant les emplacements des capteurs
- Graphiques en temps réel des données
- Liste des alertes
- Données satellitaires

#### API REST
Testez l'endpoint de santé de l'API :
```bash
# Via curl
curl http://localhost:3000/health

# Via wget (sur Windows)
wget http://localhost:3000/health -q -O -
```

Réponse attendue :
```json
{
  "status": "ok",
  "timestamp": "2023-01-01T00:00:00.000Z",
  "services": {
    "database": "connected",
    "mqtt": "connected"
  }
}
```

### Vérifier la Génération de Données

Après quelques minutes, vérifiez que les données sont générées :

```bash
# Vérifier les données des capteurs
curl http://localhost:3000/api/capteurs | jq '.features | length'

# Vérifier les alertes
curl http://localhost:3000/api/alertes | jq '.length'
```

Vous devriez voir respectivement 15 capteurs et plusieurs alertes.

## Configuration Post-Déploiement

### Configuration des Données Satellitaires Réelles

Pour activer les données satellitaires réelles au lieu de la simulation :

1. Obtenez des identifiants SentinelHub :
   - Créez un compte sur https://apps.sentinel-hub.com/
   - Créez une nouvelle configuration d'application

2. Mettez à jour le fichier `.env` :
```bash
cp .env.example .env
```

3. Modifiez `.env` avec vos identifiants :
```env
SENTINEL_CLIENT_ID=votre_client_id
SENTINEL_CLIENT_SECRET=votre_client_secret
```

4. Redémarrez le service satellite :
```bash
docker-compose restart satellite
```

### Configuration du Courrier Électronique

Pour activer les notifications par courrier électronique :

1. Modifiez le fichier `.env` :
```env
SMTP_HOST=smtp.votrefournisseur.com
SMTP_PORT=587
SMTP_USER=votre_email@domaine.com
SMTP_PASS=votre_mot_de_passe
ALERT_EMAIL_FROM=alertes@aquasys.com
ALERT_EMAIL_TO=destinataire@domaine.com
```

2. Redémarrez le service d'alertes :
```bash
docker-compose restart alertes
```

## Opérations de Maintenance

### Sauvegarde des Données

Les données sont persistantes grâce aux volumes Docker. Pour sauvegarder manuellement :

```bash
# Créer une sauvegarde de la base TimescaleDB
docker-compose exec timescaledb pg_dump -U postgres aquawatch > backup_timescale.sql

# Créer une sauvegarde de la base d'alertes
docker-compose exec postgres-alertes pg_dump -U postgres alertes > backup_alertes.sql
```

### Restauration des Données

Pour restaurer à partir d'une sauvegarde :

```bash
# Copier les fichiers de sauvegarde dans les conteneurs
docker cp backup_timescale.sql aqua_timescaledb_1:/tmp/
docker cp backup_alertes.sql aqua_postgres-alertes_1:/tmp/

# Restaurer les bases de données
docker-compose exec timescaledb psql -U postgres -d aquawatch -f /tmp/backup_timescale.sql
docker-compose exec postgres-alertes psql -U postgres -d alertes -f /tmp/backup_alertes.sql
```

### Mise à Jour du Système

Pour mettre à jour vers la dernière version :

```bash
# Arrêter les services
docker-compose down

# Mettre à jour le code source
git pull origin main

# Reconstruire les images
docker-compose build --no-cache

# Redémarrer les services
docker-compose up -d
```

### Surveillance des Performances

Surveillez l'utilisation des ressources :

```bash
# Voir l'utilisation des ressources en temps réel
docker stats

# Voir les logs d'un service spécifique
docker-compose logs --tail=100 service_name

# Vérifier l'utilisation du disque
docker system df
```

## Dépannage

### Problèmes Fréquents

#### Les Services Ne Démarrent Pas
```bash
# Vérifier les logs détaillés
docker-compose logs

# Redémarrer un service spécifique
docker-compose restart nom_du_service

# Reconstruire un service
docker-compose build nom_du_service
```

#### Impossible d'Accéder à l'Interface Web
1. Vérifiez que tous les services sont en cours d'exécution :
   ```bash
   docker-compose ps
   ```

2. Vérifiez les logs de Nginx :
   ```bash
   docker-compose logs web
   ```

3. Testez la connectivité réseau :
   ```bash
   docker-compose exec web ping api-sig
   ```

#### Aucune Donnée Affichée
1. Vérifiez que le service capteurs fonctionne :
   ```bash
   docker-compose logs capteurs
   ```

2. Vérifiez la connectivité à la base de données :
   ```bash
   docker-compose exec capteurs node -e "
   const { Client } = require('pg');
   const client = new Client({connectionString: 'postgresql://postgres:password@timescaledb:5432/aquawatch'});
   client.connect().then(() => console.log('Connected')).catch(e => console.error(e));
   "
   ```

#### Erreurs de Base de Données
1. Vérifiez l'état des conteneurs de base de données :
   ```bash
   docker-compose ps | grep -E '(timescaledb|postgres)'
   ```

2. Essayez de vous connecter manuellement :
   ```bash
   docker-compose exec timescaledb psql -U postgres -d aquawatch
   ```

### Nettoyage

Pour nettoyer complètement le déploiement :

```bash
# Arrêter et supprimer tous les conteneurs
docker-compose down

# Supprimer les volumes (ATTENTION: cela supprime toutes les données)
docker-compose down -v

# Supprimer les images construites
docker-compose down --rmi all

# Nettoyer les réseaux et volumes orphelins
docker system prune -a
```

## Sécurité

### Considérations de Sécurité

1. **Ports Exposés**: Seuls les ports nécessaires sont exposés publiquement
2. **Variables d'Environnement**: Les secrets doivent être gérés via `.env`
3. **Mises à Jour**: Gardez les images Docker à jour
4. **Accès Root**: Évitez d'exécuter des conteneurs en tant que root

### Bonnes Pratiques

1. Utilisez toujours des mots de passe forts pour les bases de données
2. Limitez l'accès réseau aux ports exposés
3. Mettez régulièrement à jour Docker et les images
4. Sauvegardez régulièrement les données
5. Surveillez les logs pour détecter les activités suspectes

## Support

Pour obtenir de l'aide avec le déploiement :

1. Consultez les logs détaillés :
   ```bash
   docker-compose logs --since 1h
   ```

2. Vérifiez l'état du système :
   ```bash
   docker-compose ps
   docker stats
   ```

3. Contactez l'équipe de support avec :
   - Version de Docker et Docker Compose
   - Logs des erreurs
   - Description du problème rencontré

### Informations de Version

Vérifiez les versions des composants :
```bash
# Versions des services
docker-compose exec api-sig node --version
docker-compose exec capteurs node --version
docker-compose exec satellite python --version
docker-compose exec stmodel python --version

# Versions des bases de données
docker-compose exec timescaledb psql -U postgres -c "SELECT version();"
docker-compose exec postgres-alertes psql -U postgres -c "SELECT version();"
```