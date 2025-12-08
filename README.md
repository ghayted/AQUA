# AQUA - Système de Surveillance de la Qualité de l'Eau

## Aperçu du Projet

AQUA est un système complet de surveillance de la qualité de l'eau qui combine des données de capteurs IoT, des observations satellitaires et des prédictions basées sur l'apprentissage automatique. Le système est conçu pour les agences de surveillance environnementale et les autorités de gestion de l'eau.

## Résumé de l'État Actuel

| Composant | Statut | Notes |
|-----------|--------|-------|
| Services de Base | ✅ Opérationnel | Tous les microservices fonctionnent |
| Pipeline de Données | ✅ Fonctionnel | Données circulant dans tous les composants |
| Interface Web | ✅ Active | Tableau de bord affichant toutes les données |
| Couche API | ✅ Complète | Tous les endpoints répondent |
| Intégration Données Réelles | ⚠️ Simulation Uniquement | Nécessite des connexions externes |

## État Détaillé des Composants

### 1. Service Capteurs (`capteurs/`)
**Technologies**: Node.js, MQTT, TimescaleDB

#### État Actuel: ✅ Entièrement Opérationnel
- Simule 15 capteurs répartis dans 8 zones géographiques
- Génère des données de qualité d'eau réalistes (pH, turbidité, température)
- Insère des mesures dans TimescaleDB toutes les 5 secondes
- Publie des données vers le broker MQTT pour distribution en temps réel

#### Fonctionnalités Terminées:
- Simulation de zones géographiques (région Rabat-Salé)
- Variations de données réalistes avec anomalies occasionnelles
- Persistance des données avec gestion d'erreurs
- Publication de messages MQTT

#### Améliorations Futures:
- Intégration avec du matériel IoT réel
- Algorithmes de calibration améliorés

### 2. Service API (`api-sig/`)
**Technologies**: Node.js, Express.js, PostgreSQL/TimescaleDB

#### État Actuel: ✅ Entièrement Opérationnel
- API RESTful servant toutes les données du système
- Support GeoJSON pour les données spatiales
- Configuration CORS appropriée
- Pooling de connexions pour l'efficacité

#### Endpoints Disponibles:
- `GET /health` - Vérification de l'état du système
- `GET /api/capteurs` - Données des capteurs avec géolocalisation
- `GET /api/satellite` - Observations satellitaires
- `GET /api/predictions` - Prédictions de qualité
- `GET /api/alertes` - Alertes actives avec filtrage
- `GET /api/stats` - Statistiques globales du système

### 3. Service d'Alertes (`alertes/`)
**Technologies**: Node.js, PostgreSQL, Nodemailer

#### État Actuel: ✅ Entièrement Opérationnel
- Surveillance en temps réel selon les normes OMS de qualité d'eau
- Dé-duplication intelligente des alertes récentes
- Classification de sévérité multi-niveaux
- Estimation de l'exposition de la population

#### Critères de Surveillance:
- pH: 6,5-8,5 (Critique: 6,0-9,0)
- Turbidité: ≤1,0 UTN (Critique: >5,0 UTN)
- Température: ≤25°C (Critique: >30°C)
- Chlorophylle: ≤10 mg/m³ (Critique: >20 mg/m³)
- Seuils NDWI pour détection de stress hydrique

### 4. Service Satellite (`satellite/`)
**Technologies**: Python, MinIO, rasterio, sentinelhub

#### État Actuel: ✅ Opérationnel (Mode Simulation)
- Traite les images satellitaires pour les indicateurs de qualité d'eau
- Calcule NDWI, turbidité et niveaux de chlorophylle
- Stocke les données traitées dans TimescaleDB
- Archive les fichiers GeoTIFF dans le stockage MinIO

#### Limitations Actuelles:
- Fonctionne en mode simulation faute d'identifiants SentinelHub
- Utilise des données synthétiques au lieu de flux satellitaires réels

#### Conditions d'Activation:
- Configurer les variables d'environnement `SENTINEL_CLIENT_ID` et `SENTINEL_CLIENT_SECRET`
- Assurer la connectivité réseau vers les services SentinelHub

### 5. Service Machine Learning (`stmodel/`)
**Technologies**: Python, PyTorch, scikit-learn

#### État Actuel: ✅ Fonctionnel avec Modèle de Démonstration
- Implémente un réseau neuronal ConvLSTM pour la prévision spatio-temporelle
- Génère des prédictions de qualité d'eau à 24 et 72 heures
- Calcule des scores de confiance pour les prédictions
- Stocke les résultats dans TimescaleDB

#### Implémentation Actuelle:
- Modèle simplifié à des fins de démonstration
- Agrégation des données de capteurs par zone
- Ingénierie de caractéristiques basique

#### Opportunités d'Amélioration:
- Entraîner sur des données historiques pour améliorer la précision
- Implémenter des méthodes d'ensemble
- Ajouter l'intégration des prévisions météorologiques

### 6. Interface Web (`web/`)
**Technologies**: HTML5, CSS3, JavaScript, Leaflet.js, Chart.js, Nginx

#### État Actuel: ✅ Entièrement Opérationnelle
- Tableau de bord interactif avec visualisation de données en temps réel
- Cartographie géographique des emplacements de capteurs
- Liste d'alertes avec filtrage par sévérité
- Affichage des données satellitaires
- Graphiques de prédictions et tendances
- Design responsive pour tous les appareils

## État de l'Infrastructure

### Orchestration de Conteneurs
✅ Docker Compose entièrement configuré avec des contrôles de santé
✅ Politiques de redémarrage automatique des services
✅ Gestion appropriée des dépendances

### Stockage de Données
✅ TimescaleDB opérationnel pour les données temporelles
✅ PostgreSQL dédié au stockage des alertes
✅ Stockage objet MinIO pour les images satellitaires
✅ Configuration de volumes persistants

### Courtier de Messages
✅ Courtier MQTT Eclipse Mosquitto en fonctionnement
✅ Contrôles de santé implémentés
✅ Mappages de ports appropriés

### Services Auxiliaires
🟡 GeoServer présent mais non configuré
🟢 Proxy inverse Nginx opérationnel

## Problèmes et Limitations Connus

### 1. Authenticité des Données
**Problème**: Toutes les données sont actuellement simulées
**Impact**: Le système démontre ses fonctionnalités mais manque de validation du monde réel
**Résolution**: Connecter aux capteurs IoT réels et à l'API SentinelHub

### 2. Implémentation de la Sécurité
**Problème**: Aucune authentification ni autorisation
**Impact**: Non adapté pour un déploiement en production
**Résolution**: Implémenter une authentification basée sur JWT et RBAC

### 3. Couverture de Tests
**Problème**: Aucune suite de tests automatisés
**Impact**: Vérification manuelle requise pour les modifications
**Résolution**: Implémenter des tests unitaires et d'intégration

### 4. Observabilité
**Problème**: Aucune journalisation ou métrique centralisée
**Impact**: Dépannage difficile dans les environnements de production
**Résolution**: Déployer une pile de surveillance Prometheus/Grafana

## Informations d'Accès au Système

### Interfaces Principales
- **Tableau de Bord Web**: http://localhost
- **Endpoint API**: http://localhost:3000

### Connexions Base de Données
- **TimescaleDB**: localhost:5433 (Base de données: aquawatch)
- **Base d'Alertes**: localhost:5434 (Base de données: alertes)

### Ports de Service
- **Courtier MQTT**: 1883 (Courtier), 9003 (WebSocket)
- **Stockage MinIO**: 9000 (API), 9002 (Console)
- **GeoServer**: http://localhost:8080

## Instructions de Déploiement

### Prérequis
- Docker Engine 20.10+
- Docker Compose 1.29+
- 4GB RAM minimum
- 2 cœurs CPU minimum

### Démarrage Rapide
```bash
# Cloner le dépôt
git clone <url-du-dépôt>
cd AQUA

# Démarrer tous les services
docker-compose up -d

# Surveiller la progression du démarrage
docker-compose logs -f

# Accéder à l'interface web
open http://localhost
```

### Configuration de l'Environnement
Pour activer les données satellitaires réelles, définir ces variables d'environnement :
```bash
SENTINEL_CLIENT_ID=votre_id_client
SENTINEL_CLIENT_SECRET=votre_secret_client
```

## Feuille de Route de Développement

### Phase 1: Prêt pour la Production (Terminée)
- ✅ Implémentation des microservices de base
- ✅ Établissement du pipeline de données
- ✅ Développement de l'interface web
- ✅ Achèvement de la couche API

### Phase 2: Intégration de Données Réelles (En Cours)
- ⚠️ Connectivité des capteurs IoT
- ⚠️ Activation des données satellitaires
- ⚠️ Ingestion de données historiques

### Phase 3: Fonctionnalités Entreprise (En Attente)
- 🔲 Authentification et autorisation
- 🔲 Couverture de tests complète
- 🔲 Surveillance et alerting
- 🔲 Optimisation des performances

### Phase 4: Analyses Avancées (Futur)
- 🔲 Maintenance prédictive
- 🔲 Application mobile
- 🔲 Développement d'API publique
- 🔲 Intégration avec les systèmes gouvernementaux

## Contribution

Ce projet accueille favorablement les contributions dans les domaines suivants :
- Améliorations des modèles d'apprentissage automatique
- Intégration de capteurs IoT
- Améliorations de sécurité
- Automatisation des tests
- Améliorations de documentation
- Optimisation de l'expérience utilisateur

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## Support

Pour obtenir de l'aide, veuillez ouvrir un ticket dans le dépôt ou contacter l'équipe de développement.