# 🌊 AquaWatch - Système Intelligent de Surveillance de la Qualité de l'Eau

Plateforme IoT + IA pour la surveillance proactive de la qualité de l'eau au Maroc.

## 🎬 Démonstration

https://github.com/user-attachments/assets/demonstration.mp4

▶️ [Voir la vidéo de démonstration](./demonstration.mp4)

---

## 📁 Structure du Projet

```
AQUA/
├── capteurs/                    # Service de simulation des capteurs IoT
│   ├── index.js                 # Code principal Node.js
│   ├── package.json            
│   └── Dockerfile              
│
├── alertes/                     # Service de détection des alertes OMS
│   ├── index.js                 # Code principal Node.js
│   ├── package.json            
│   └── Dockerfile              
│
├── stmodel/                     # Service de prédiction IA (Machine Learning)
│   ├── stmodel.py               # Code principal Python + PyTorch
│   ├── requirements.txt        
│   ├── Dockerfile              
│   └── weights/                 # Poids du modèle entraîné
│       └── trained_weights.pth
│
├── api-sig/                     # API REST centrale
│   ├── index.js                 # Serveur Express.js
│   ├── package.json            
│   └── Dockerfile              
│
├── web/                         # Interface utilisateur
│   ├── index.html               # Dashboard
│   ├── map.html                 # Carte interactive
│   ├── sensors.html             # État des capteurs
│   ├── alerts.html              # Alertes
│   ├── predictions.html         # Prédictions IA
│   ├── css/                    
│   └── js/                     
│
├── mqtt/                        # Configuration du broker MQTT
│   └── config/                 
│
├── docker-compose.yml           # Orchestration de tous les services
└── Jenkinsfile                  # Pipeline CI/CD
```

---

## 🏗️ Architecture en Couches

Le système est organisé en **4 couches** distinctes :

```
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE PRÉSENTATION                       │
│                    (web/ - Nginx Port 80)                    │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       COUCHE API                             │
│                  (api-sig/ - Port 3000)                      │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ SQL
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     COUCHE DONNÉES                           │
│         TimescaleDB (5433)  |  PostgreSQL (5434)            │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ SQL INSERT/SELECT
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     COUCHE MÉTIER                            │
│      capteurs/  |  stmodel/  |  alertes/  |  satellite/     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📡 COUCHE MÉTIER - Les Microservices Producteurs

### 1. Service Capteurs (`capteurs/index.js`)

**Rôle** : Simulation de 16 capteurs IoT mesurant la qualité de l'eau dans 10 zones géographiques.

**Ce que fait le fichier `capteurs/index.js` :**

Au démarrage, le service se connecte à TimescaleDB et au broker MQTT. Ensuite, toutes les 5 secondes, il exécute cette boucle :

1. **Génération des données** (`generateSensorData()`) : 
   - Choisit aléatoirement un capteur parmi CAPT-1 à CAPT-16
   - Génère des valeurs de pH, turbidité et température selon la zone :
     - Zones côtières : pH plus élevé (8.0), température plus basse (18°C)
     - Rivière Bouregreg : turbidité plus élevée (2.0 NTU)
     - Marrakech : 60% valeurs critiques (pour tester les alertes)
   - Dans 10% des cas, génère des valeurs anormales intentionnellement

2. **Insertion en base** (`insertSensorData()`) :
   ```sql
   INSERT INTO donnees_capteurs 
   (timestamp, capteur_id, zone, ph, turbidite, temperature, latitude, longitude)
   VALUES (...)
   ```

3. **Publication MQTT** (`publishToMQTT()`) :
   - Publie sur le topic `aquawatch/capteurs/{capteur_id}`
   - Format JSON avec toutes les mesures

**Les 10 zones couvertes :**
- Rabat-Centre, Salé-Nord, Salé-Sud, Hay-Riad, Agdal
- Côte-Océan, Bouregreg, Temara, Skhirat, Marrakech

---

### 2. Service Alertes (`alertes/index.js`)

**Rôle** : Surveiller les données des capteurs et générer des alertes quand les valeurs dépassent les seuils OMS.

**Ce que fait le fichier `alertes/index.js` :**

Le service se connecte à **deux bases différentes** :
- **TimescaleDB** : pour LIRE les données des capteurs
- **PostgreSQL** : pour ÉCRIRE les alertes

Toutes les 7 secondes, la fonction `checkOMSThresholds()` s'exécute :

1. **Lecture des mesures récentes** :
   ```sql
   SELECT capteur_id, zone, ph, turbidite, temperature 
   FROM donnees_capteurs
   WHERE timestamp > NOW() - INTERVAL '2 minutes'
   ```

2. **Comparaison aux seuils OMS** :
   
   | Paramètre | Normal | WARNING | CRITICAL |
   |-----------|--------|---------|----------|
   | pH | 6.5 - 8.5 | < 6.5 ou > 8.5 | < 6.0 ou > 9.0 |
   | Turbidité | < 1.0 NTU | > 1.0 NTU | > 5.0 NTU |
   | Température | < 25°C | > 25°C | > 30°C |

3. **Génération des alertes** : Pour chaque dépassement détecté, une alerte est créée avec :
   - Le type de problème (SEUIL_DEPASSE, SEUIL_CRITIQUE_DEPASSE)
   - La sévérité (WARNING ou CRITICAL)
   - La valeur mesurée et le seuil OMS dépassé
   - L'estimation de la population exposée

4. **Insertion dans PostgreSQL** :
   ```sql
   INSERT INTO alertes 
   (timestamp, type, severity, zone, capteur_id, parametre, valeur, seuil_oms, message)
   VALUES (...)
   ```

5. **Notification email** (simulée) : Envoi d'un email aux administrateurs

---

### 3. Service STModel (`stmodel/stmodel.py`)

**Rôle** : Prédire la qualité de l'eau pour les 24 prochaines heures en utilisant un modèle de Machine Learning ConvLSTM.

**Ce que fait le fichier `stmodel/stmodel.py` :**

**Architecture du modèle `HourlyWaterQualityPredictor` :**

```
ENTRÉE: Séquence de 12 mesures passées
        Shape: (batch, 12, 3, 4, 4)
        → 12 timestamps × 3 paramètres (pH, turb, temp) × grille 4×4 zones
                    │
                    ▼
        ┌─────────────────────┐
        │   ConvLSTM Encoder   │  ← Capture les relations spatiales entre zones
        │   Kernel 3×3         │     et les patterns temporels
        │   Hidden: 32         │
        └──────────┬──────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │   Hour Embedding     │  ← Apprend les variations jour/nuit
        │   Heure (0-23) →     │     (température plus chaude à midi, etc.)
        │   Vecteur 32D        │
        └──────────┬──────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │   Decoder MLP        │  ← Transforme en prédictions
        │   512+32 → 256 →     │
        │   128 → 30           │
        └──────────┬──────────┘
                    │
                    ▼
SORTIE: Prédictions pour 10 zones × 3 paramètres
        Shape: (batch, 10, 3)
```

**Fonctionnement toutes les 5 minutes :**

1. **Récupération des données** (`get_sensor_data_robust()`) :
   - Essaie de récupérer les données des 6 dernières heures
   - Si pas de données, étend à 24h, puis 7 jours, puis 30 jours
   - Pour les zones sans données, utilise des valeurs par défaut OMS

2. **Génération des prédictions** (`run_hourly_predictions()`) :
   - Génère 24 prédictions (00:00 à 23:00 de demain)
   - Applique des **variations horaires réalistes** :
     - Température suit le cycle jour/nuit (±4°C, pic à 12h)
     - Turbidité varie avec l'activité humaine
     - pH reste stable avec petites variations

3. **Calcul des scores** :
   - `qualite_score` (0-100) : Basé sur l'écart aux normes OMS
   - `risque_score` (0-100) : Probabilité de problèmes
   - `confidence` : Confiance basée sur la fraîcheur des données

4. **Insertion des prédictions** :
   ```sql
   INSERT INTO predictions_qualite 
   (timestamp, zone_id, ph_pred, turbidite_pred, temperature_pred,
    qualite_score, risque_score, confidence)
   VALUES (...)
   ```

---

## 🌐 COUCHE API (`api-sig/index.js`)

**Rôle** : Point d'entrée unique pour accéder à toutes les données du système via une API REST.

**Ce que fait le fichier `api-sig/index.js` :**

Au démarrage, le service crée un serveur Express.js sur le port 3000 avec Swagger pour la documentation.

**Endpoints disponibles :**

| Endpoint | Méthode | Description | Source des données |
|----------|---------|-------------|-------------------|
| `/health` | GET | Vérifier l'état du service | - |
| `/api/capteurs` | GET | Données des capteurs (GeoJSON) | TimescaleDB |
| `/api/predictions` | GET | Prédictions IA 24h | TimescaleDB |
| `/api/alertes` | GET | Alertes actives | PostgreSQL |
| `/api/satellite` | GET | Observations satellite | TimescaleDB |
| `/api/stats` | GET | Statistiques globales | Les deux bases |
| `/api-docs` | GET | Documentation Swagger | - |

**Exemple `/api/capteurs` :**
```javascript
app.get('/api/capteurs', async (req, res) => {
  const query = `
    SELECT id, timestamp, capteur_id, zone, ph, turbidite, temperature, latitude, longitude
    FROM donnees_capteurs
    ORDER BY timestamp DESC
    LIMIT $1
  `;
  const result = await dbClient.query(query, [limit]);
  
  // Transforme en GeoJSON pour la carte
  res.json({
    type: 'FeatureCollection',
    features: result.rows.map(row => ({
      type: 'Feature',
      properties: { zone: row.zone, ph: row.ph, ... },
      geometry: { type: 'Point', coordinates: [row.longitude, row.latitude] }
    }))
  });
});
```

**Exemple `/api/predictions?date=2024-12-23` :**
```javascript
app.get('/api/predictions', async (req, res) => {
  const query = `
    SELECT zone_id, timestamp, ph_pred, turbidite_pred, temperature_pred,
           qualite_score, qualite_niveau, risque_score, risque_niveau, confidence
    FROM predictions_qualite
    WHERE DATE(timestamp) = $1
    ORDER BY timestamp ASC
  `;
  const result = await dbClient.query(query, [date]);
  res.json(result.rows);
});
```

**Exemple `/api/alertes` :**
```javascript
// Connexion à PostgreSQL (base alertes, différente de TimescaleDB)
const alertesDb = new Client({
  host: 'postgres',
  port: 5432,
  database: 'alertes',
  ...
});

const query = `
  SELECT id, timestamp, severity, zone, parametre, valeur, seuil_oms, message
  FROM alertes
  WHERE status = 'ACTIVE'
  ORDER BY timestamp DESC
`;
```

---

## 💾 COUCHE DONNÉES

### TimescaleDB (Port 5433)

Extension PostgreSQL optimisée pour les **séries temporelles**.

**Tables :**

```sql
-- Mesures des capteurs (insérées par capteurs/)
CREATE TABLE donnees_capteurs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    capteur_id VARCHAR(50),
    zone VARCHAR(50),
    ph DECIMAL(5,2),
    turbidite DECIMAL(5,2),
    temperature DECIMAL(5,2),
    latitude DECIMAL(10,6),
    longitude DECIMAL(10,6)
);
-- Conversion en hypertable pour le partitionnement automatique
SELECT create_hypertable('donnees_capteurs', 'timestamp');

-- Prédictions IA (insérées par stmodel/)
CREATE TABLE predictions_qualite (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    zone_id VARCHAR(50),
    ph_pred DECIMAL(5,2),
    turbidite_pred DECIMAL(5,2),
    temperature_pred DECIMAL(5,2),
    qualite_score DECIMAL(5,2),
    qualite_niveau VARCHAR(20),
    risque_score DECIMAL(5,2),
    risque_niveau VARCHAR(20),
    confidence DECIMAL(5,2),
    PRIMARY KEY (timestamp, id)
);
```

### PostgreSQL (Port 5434)

Base séparée pour les **alertes**.

```sql
-- Alertes OMS (insérées par alertes/)
CREATE TABLE alertes (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    type VARCHAR(50),
    severity VARCHAR(20),  -- 'CRITICAL' ou 'WARNING'
    zone VARCHAR(50),
    capteur_id VARCHAR(50),
    parametre VARCHAR(50), -- 'ph', 'turbidite', 'temperature'
    valeur DECIMAL(10,2),
    seuil_oms DECIMAL(10,2),
    population_exposee INTEGER,
    message TEXT,
    status VARCHAR(20) DEFAULT 'ACTIVE'
);
```

---

## 🖥️ COUCHE PRÉSENTATION (`web/`)

Interface utilisateur servie par Nginx sur le port 80.

**Pages disponibles :**

| Page | Fichier | Description |
|------|---------|-------------|
| Dashboard | `index.html` | Vue d'ensemble avec statistiques |
| Carte | `map.html` | Visualisation géospatiale avec Leaflet |
| Capteurs | `sensors.html` | État temps réel des 16 capteurs |
| Alertes | `alerts.html` | Historique des alertes |
| Prédictions | `predictions.html` | Prévisions IA 24 heures |

**Comment le frontend communique avec l'API :**

```javascript
// Dans web/js/main.js
async function loadCapteurs() {
    const response = await fetch('http://localhost:3000/api/capteurs');
    const data = await response.json();
    // Afficher sur la carte Leaflet
    data.features.forEach(feature => {
        L.marker([feature.geometry.coordinates[1], feature.geometry.coordinates[0]])
            .addTo(map)
            .bindPopup(`Zone: ${feature.properties.zone}<br>pH: ${feature.properties.ph}`);
    });
}
```

---

## 🔄 Communication entre Services

### Flux de données complet

```
1. CAPTEURS génère des mesures
   │
   ├──→ INSERT INTO donnees_capteurs (TimescaleDB)
   │
   └──→ MQTT publish (aquawatch/capteurs/{id})
   
2. STMODEL lit les mesures et prédit
   │
   ├──→ SELECT FROM donnees_capteurs
   │
   └──→ INSERT INTO predictions_qualite
   
3. ALERTES surveille et alerte
   │
   ├──→ SELECT FROM donnees_capteurs (TimescaleDB)
   │
   └──→ INSERT INTO alertes (PostgreSQL)
   
4. API-SIG expose toutes les données
   │
   ├──→ SELECT FROM donnees_capteurs
   ├──→ SELECT FROM predictions_qualite
   └──→ SELECT FROM alertes
   
5. WEB affiche les données
   │
   └──→ fetch('http://localhost:3000/api/xxx')
```

### Protocoles utilisés

| De → Vers | Protocole | Exemple |
|-----------|-----------|---------|
| Capteurs → TimescaleDB | **SQL** | `INSERT INTO donnees_capteurs` |
| Capteurs → MQTT Broker | **MQTT** | `publish('aquawatch/capteurs/CAPT-1')` |
| STModel → TimescaleDB | **SQL** | `SELECT` puis `INSERT` |
| Alertes → PostgreSQL | **SQL** | `INSERT INTO alertes` |
| API-SIG → Bases | **SQL** | `SELECT * FROM ...` |
| Web → API-SIG | **HTTP REST** | `GET /api/capteurs` |

---

## 🚀 Démarrage

```bash
# Démarrer tous les services
docker-compose up -d

# Vérifier l'état
docker-compose ps

# Voir les logs d'un service
docker-compose logs -f capteurs
docker-compose logs -f stmodel
docker-compose logs -f alertes
```

**URLs d'accès :**

| Service | URL |
|---------|-----|
| Interface Web | http://localhost |
| API REST | http://localhost:3000 |
| Documentation Swagger | http://localhost:3000/api-docs |
| Jenkins CI/CD | http://localhost:8081 |

---

## 👥 Équipe

- **Ghayt El Idrissi Dafali**
- **Reda Bouimakliouine**
- **Souhail Azzimani**
- **Amine Ibnou Chiekh**

EMSI Marrakech - 2025-2026
