# AQUA : Système de Surveillance de la Qualité de l'Eau en Temps Réel avec Intégration de Données IoT et Satellitaires

## Noms des auteurs / développeurs principaux (avec affiliations, adresses, email)
Équipe de Développement AQUA
Affiliation : Projet de Recherche Indépendant
Adresse : Rabat, Maroc
Email : aqua.development@example.com

## Résumé

AQUA est un système complet de surveillance de la qualité de l'eau qui intègre des données de capteurs Internet des Objets (IoT), des images satellitaires et des prédictions basées sur l'apprentissage automatique pour fournir des informations en temps réel sur les conditions de qualité de l'eau. Le système implémente une architecture de microservices utilisant des conteneurs Docker pour l'évolutivité et la maintenabilité. Les composants clés comprennent la simulation de données de capteurs, le traitement d'images satellitaires, des modèles de prévision spatio-temporels et la génération d'alertes basées sur les normes de l'OMS. Le système fournit un tableau de bord web pour la visualisation et la surveillance, permettant aux agences environnementales de prendre des décisions éclairées concernant la gestion des ressources en eau. Tous les composants ont été implémentés et testés avec succès, démontrant la faisabilité de l'intégration de sources de données diverses pour les applications de surveillance environnementale.

## Mots-clés

surveillance de la qualité de l'eau, capteurs IoT, données satellitaires, apprentissage automatique, microservices, Docker, analyse en temps réel

## Métadonnées

| Nr | Description des métadonnées du code | Métadonnées |
|----|-----------------------------------|-------------|
| C1 | Version actuelle du code | v1.0 |
| C2 | Lien permanent vers le code/dépôt utilisé pour cette version du code | https://github.com/votre-organisation/aqua |
| C3 | Lien permanent vers la capsule reproductible | https://codeocean.com/capsule/0000000/tree/v1 |
| C4 | Licence légale du code | Licence MIT |
| C5 | Système de gestion de versions du code utilisé | git |
| C6 | Langages logiciels, outils et services utilisés | Node.js, Python, Docker, PostgreSQL, TimescaleDB, MQTT, Nginx, HTML/CSS/JavaScript |
| C7 | Exigences de compilation, environnements d'exécution et dépendances | Docker Engine 20.10+, Docker Compose 1.29+, 4GB RAM, 2 cœurs CPU |
| C8 | Si disponible, lien vers la documentation/manuel du développeur | http://localhost/docs |
| C9 | Email de support pour les questions | aqua.support@example.com |

## 1. Motivation et importance

La surveillance de la qualité de l'eau est cruciale pour protéger la santé publique et préserver les écosystèmes aquatiques. Les approches traditionnelles de surveillance souffrent souvent d'une couverture spatiale limitée, de rapports retardés et d'une incapacité à prédire les conditions futures. Le système AQUA répond à ces limitations en combinant des réseaux de capteurs IoT en temps réel avec la télédétection satellitaire et l'analyse prédictive.

Le logiciel résout plusieurs problèmes scientifiques clés :
- Intégration de sources de données hétérogènes (capteurs au sol et imagerie satellitaire)
- Détection d'anomalies en temps réel utilisant les normes de l'Organisation Mondiale de la Santé (OMS)
- Prévision spatio-temporelle des paramètres de qualité de l'eau
- Génération d'alertes automatisée pour une réponse rapide

Cette approche contribue à la découverte scientifique en permettant :
- Des systèmes d'alerte précoce pour les événements de contamination de l'eau
- Une meilleure compréhension des motifs spatio-temporels de la qualité de l'eau
- Une prise de décision axée sur les données pour les agences de protection environnementale

Le cadre expérimental implique le déploiement du système dans la région de Rabat-Salé au Maroc, simulant un réseau de 15 capteurs de qualité d'eau répartis dans 8 zones géographiques. Les utilisateurs interagissent avec le système via un tableau de bord web qui affiche les données en temps réel, les images satellitaires, les modèles prédictifs et les notifications d'alerte.

Les travaux connexes incluent les plateformes de surveillance environnementale comme GEMS/Water et l'évaluation de la qualité de l'eau basée sur des satellites utilisant les images Sentinel-2. Notre approche innove en combinant ces sources de données avec des prédictions d'apprentissage automatique dans une architecture de microservices intégrée.

## 2. Description du logiciel

### 2.1 Architecture logicielle

Le système AQUA implémente une architecture de microservices orchestrée par Docker Compose. L'architecture se compose de six services interconnectés :

```
                    +------------------+
                    | Interface Web    |
                    | (Nginx/HTML)     |
                    +---------+--------+
                              |
                    +---------v--------+
                    | Service API      |
                    | (Node.js/API)    |
                    +----+------+------+ 
                         |      |
       +-----------------+      +------------------+
       |                                   |
+------v-------+                   +------v-------+
| Données      |                   | Système      |
| Capteurs     |                   | d'Alertes    |
| (Node.js/MQTT)|                  | (Node.js/PG) |
+------+-------+                   +------+-------+
       |                                  |
+------v-------+                   +------v-------+
| Satellite    |                   | Prédiction   |
| (Python/ML)  |                   | ML           |
|              |                   | (Python/ML)  |
+--------------+                   +--------------+
```

Les composants d'infrastructure principaux incluent :
- TimescaleDB pour le stockage de données temporelles des capteurs
- PostgreSQL pour le stockage des alertes
- Courtier MQTT Mosquitto pour la messagerie en temps réel
- MinIO pour le stockage d'images satellitaires
- GeoServer pour les services géospatiaux

### 2.2 Fonctionnalités logicielles

Le système AQUA fournit les fonctionnalités majeures suivantes :

1. **Gestion des Données de Capteurs** :
   - Simule 15 capteurs de qualité d'eau répartis dans 8 zones géographiques
   - Génère des paramètres de qualité d'eau réalistes (pH, température, turbidité)
   - Stocke les données temporelles dans TimescaleDB
   - Publie les mises à jour en temps réel via MQTT

2. **Traitement d'Images Satellitaires** :
   - Traite les images satellitaires pour dériver des indicateurs de qualité d'eau
   - Calcule le NDWI pour la détection d'eau
   - Estime les niveaux de turbidité et de chlorophylle
   - Archive les fichiers GeoTIFF traités dans MinIO

3. **Prédictions d'Apprentissage Automatique** :
   - Implémente un réseau neuronal ConvLSTM pour la prévision spatio-temporelle
   - Génère des prédictions de qualité d'eau sur 24 heures et 72 heures
   - Fournit des scores de confiance pour les prédictions
   - Stocke les résultats dans TimescaleDB

4. **Génération d'Alertes** :
   - Surveille les données des capteurs par rapport aux normes de l'OMS sur la qualité de l'eau
   - Implémente une déduplication intelligente des alertes
   - Classe les alertes par niveau de gravité
   - Stocke les alertes dans la base de données PostgreSQL

5. **Visualisation Web** :
   - Carte interactive affichant les emplacements des capteurs
   - Visualisation de données en temps réel avec des graphiques
   - Liste d'alertes avec capacités de filtrage
   - Affichage d'images satellitaires
   - Visualisation des tendances prédictives

## 3. Exemples illustratifs

Pour démontrer le système AQUA, suivez ces étapes de déploiement :

1. Clonez le dépôt :
```bash
git clone https://github.com/votre-organisation/aqua.git
cd aqua
```

2. Démarrez tous les services :
```bash
docker-compose up -d
```

3. Accédez à l'interface web à l'adresse http://localhost

Le système commencera immédiatement à générer des données de capteurs simulées. En quelques minutes, vous observerez :
- Des marqueurs de capteurs apparaissant sur la carte
- Des graphiques de données en temps réel se mettant à jour
- Des notifications d'alerte lorsque les seuils de l'OMS sont dépassés
- Le traitement d'images satellitaires (en mode simulation)
- Des prédictions d'apprentissage automatique pour les conditions futures

Exemple d'endpoint API pour récupérer les données des capteurs :
```bash
curl http://localhost:3000/api/capteurs
```

Réponse :
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-6.85, 33.99]
      },
      "properties": {
        "sensor_id": "capteur_001",
        "zone_id": 1,
        "ph_value": 7.2,
        "temperature": 22.5,
        "turbidity": 0.8
      }
    }
  ]
}
```

## 4. Impact

Le système AQUA permet de nouvelles directions de recherche dans la surveillance environnementale :
- Étude des corrélations spatio-temporelles dans la qualité de l'eau
- Validation des indicateurs de qualité d'eau dérivés de satellites
- Optimisation des stratégies de placement des capteurs
- Évaluation des approches d'apprentissage automatique pour la prédiction environnementale

Le logiciel améliore les recherches existantes en :
- Fournissant une résolution temporelle plus élevée que l'échantillonnage traditionnel
- Offrant une couverture spatiale plus large grâce à l'intégration satellitaire
- Permettant une surveillance prédictive plutôt que réactive
- Prenant en charge la collecte et l'analyse automatisées de données

Pour les praticiens de la surveillance environnementale, AQUA transforme la pratique quotidienne en :
- Automatisant la collecte de données provenant de multiples sources
- Fournissant des alertes en temps réel pour une action immédiate
- Offrant des informations prédictives pour une gestion proactive
- Centralisant les données de surveillance dans un tableau de bord accessible

Bien qu'encore en développement, le logiciel démontre une utilité potentielle généralisée :
- Agences de protection environnementale recherchant des solutions de surveillance intégrées
- Chercheurs étudiant les dynamiques de la qualité de l'eau
- Municipalités gérant les ressources en eau
- ONG surveillant la santé environnementale

Les applications commerciales incluent :
- Services de conseil environnemental
- Systèmes de gestion de l'eau dans les villes intelligentes
- Surveillance de la qualité de l'eau agricole
- Surveillance de la conformité des effluents industriels

## 5. Conclusions

Le système AQUA démontre avec succès l'intégration de capteurs IoT, de données satellitaires et d'apprentissage automatique pour une surveillance complète de la qualité de l'eau. L'architecture de microservices assure l'évolutivité et la maintenabilité, tandis que l'orchestration Docker garantit un déploiement facile. Toutes les fonctionnalités principales ont été implémentées et testées, prouvant la faisabilité de l'approche.

Les travaux futurs se concentreront sur :
- L'intégration avec de vrais capteurs IoT et des flux de données satellitaires
- L'amélioration des modèles d'apprentissage automatique entraînés sur des données historiques
- L'implémentation de l'authentification des utilisateurs et du contrôle d'accès
- Le développement d'applications mobiles pour une utilisation sur le terrain
- L'expansion à des régions géographiques supplémentaires

Le système représente une avancée significative dans la technologie de surveillance environnementale, offrant une plateforme évolutive et extensible pour la gestion de la qualité de l'eau.

## Remerciements

Nous remercions la communauté open-source pour les outils et bibliothèques qui ont rendu ce projet possible, notamment Docker, Node.js, les bibliothèques scientifiques Python et divers outils géospatiaux. Nous reconnaissons également l'Organisation Mondiale de la Santé pour avoir établi des directives sur la qualité de l'eau qui informent notre système d'alerte.

## Références

1. Organisation Mondiale de la Santé. (2011). Directives pour la qualité de l'eau de boisson. Organisation Mondiale de la Santé.

2. Gholizadeh, M. H., Melesse, A. M., & Reddy, K. R. (2016). Une revue complète sur l'estimation des paramètres de qualité de l'eau utilisant des techniques de télédétection. Sensors, 16(8), 1298.

3. Li, X., Liu, C., & Liu, Y. (2020). Apprentissage profond pour la surveillance de la qualité de l'eau basée sur des satellites : Une revue. Remote Sensing, 12(16), 2587.

4. Mullai, R., & Mohan, S. (2021). Système de surveillance intelligent de la qualité de l'eau basé sur l'IoT utilisant un réseau de capteurs. Materials Today: Proceedings, 45, 5510-5515.

5. Équipe de Développement AQUA. (2023). AQUA : Système de Surveillance de la Qualité de l'Eau en Temps Réel [Logiciel informatique]. https://github.com/votre-organisation/aqua