# Jenkins pour AQUA

Ce répertoire contient la configuration Jenkins pour l'intégration CI/CD du projet AQUA.

## 🚀 Démarrage Rapide

### 1. Lancer Jenkins

```bash
# Démarrer Jenkins avec docker-compose
docker-compose up -d jenkins

# Vérifier que Jenkins est démarré
docker-compose ps jenkins
```

### 2. Configuration Initiale

1. **Accéder à Jenkins**: http://localhost:8081

2. **Récupérer le mot de passe initial**:
```bash
docker exec aquawatch-jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

3. **Installer les plugins recommandés** lors de la première configuration

4. **Créer un utilisateur administrateur**

### 3. Créer le Pipeline

1. Dans Jenkins, cliquer sur **"New Item"**
2. Entrer le nom: `AQUA-Pipeline`
3. Sélectionner **"Pipeline"**
4. Dans la section **Pipeline**:
   - Definition: `Pipeline script from SCM`
   - SCM: `Git`
   - Repository URL: URL de votre dépôt Git
   - Branch: `*/main` (ou votre branche principale)
   - Script Path: `Jenkinsfile`
5. Sauvegarder

### 4. Lancer le Build

Cliquer sur **"Build Now"** pour lancer le premier build.

## 📋 Configuration des Credentials (Optionnel)

### Docker Hub (pour pousser les images)

1. Aller dans **Manage Jenkins** > **Manage Credentials**
2. Ajouter des credentials de type **Username with password**:
   - ID: `dockerhub`
   - Username: votre username Docker Hub
   - Password: votre token Docker Hub

### GitHub (pour les webhooks)

1. Générer un token GitHub avec les permissions `repo`
2. Ajouter dans Jenkins:
   - ID: `github`
   - Secret: votre token GitHub

## 🔧 Configuration Avancée

### Webhooks GitHub

Pour déclencher automatiquement les builds lors des push:

1. Dans votre dépôt GitHub, aller dans **Settings** > **Webhooks**
2. Ajouter un webhook:
   - Payload URL: `http://votre-serveur:8081/github-webhook/`
   - Content type: `application/json`
   - Events: `Just the push event`

### Variables d'Environnement

Vous pouvez configurer des variables d'environnement dans:
- **Manage Jenkins** > **Configure System** > **Global properties**

Variables utiles:
- `DOCKER_REGISTRY`: URL de votre registry Docker privé
- `DEPLOY_ENV`: Environnement de déploiement (dev, staging, prod)

## 📊 Pipeline Stages

Le pipeline Jenkins comprend les étapes suivantes:

1. **Checkout**: Récupération du code source
2. **Environment Setup**: Configuration de l'environnement
3. **Build Services**: Construction des images Docker
4. **Unit Tests**: Tests unitaires (Node.js et Python)
5. **Integration Tests**: Tests d'intégration
6. **Health Checks**: Vérification de la santé des services
7. **Deploy**: Déploiement (uniquement sur branche main)

## 🐛 Dépannage

### Jenkins ne démarre pas

```bash
# Vérifier les logs
docker-compose logs jenkins

# Redémarrer Jenkins
docker-compose restart jenkins
```

### Problème de permissions Docker

Si Jenkins ne peut pas accéder à Docker:

```bash
# Vérifier que le socket Docker est monté
docker exec aquawatch-jenkins ls -la /var/run/docker.sock

# Vérifier les permissions
docker exec aquawatch-jenkins groups
```

### Build échoue

1. Vérifier les logs du build dans l'interface Jenkins
2. Vérifier que tous les services sont accessibles:
```bash
docker-compose ps
```

### Nettoyer l'espace disque

```bash
# Nettoyer les images Docker non utilisées
docker system prune -a -f

# Nettoyer les volumes Jenkins
docker volume prune -f
```

## 📁 Structure des Fichiers

```
jenkins/
├── Dockerfile          # Image Jenkins personnalisée
└── README.md          # Ce fichier

scripts/
├── test-services.sh   # Script de test des services
└── deploy.sh          # Script de déploiement

Jenkinsfile            # Définition du pipeline
```

## 🔐 Sécurité

### Recommandations

1. **Changer le mot de passe admin** après la première connexion
2. **Activer HTTPS** en production
3. **Limiter l'accès** avec un reverse proxy (nginx)
4. **Sauvegarder régulièrement** le volume `jenkins_home`

### Backup

```bash
# Créer un backup du volume Jenkins
docker run --rm -v aquawatch_jenkins_home:/data -v $(pwd):/backup alpine tar czf /backup/jenkins-backup-$(date +%Y%m%d).tar.gz /data

# Restaurer un backup
docker run --rm -v aquawatch_jenkins_home:/data -v $(pwd):/backup alpine sh -c "cd /data && tar xzf /backup/jenkins-backup-YYYYMMDD.tar.gz --strip 1"
```

## 📚 Ressources

- [Documentation Jenkins](https://www.jenkins.io/doc/)
- [Pipeline Syntax](https://www.jenkins.io/doc/book/pipeline/syntax/)
- [Docker Pipeline Plugin](https://plugins.jenkins.io/docker-workflow/)
- [Blue Ocean](https://www.jenkins.io/doc/book/blueocean/)

## 🆘 Support

Pour toute question ou problème:
1. Vérifier les logs: `docker-compose logs jenkins`
2. Consulter la documentation Jenkins
3. Vérifier les issues GitHub du projet
