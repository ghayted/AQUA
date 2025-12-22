# 🧠 STModel - Module de Machine Learning (IA)

**Système de Prédiction de la Qualité de l'Eau basé sur Deep Learning**

---

## 📌 Vue d'Ensemble

Le module **STModel** (Spatio-Temporal Model) est le cœur intelligent du projet AquaWatch. Il utilise un réseau de neurones **ConvLSTM** (Convolutional Long Short-Term Memory) pour prédire la qualité de l'eau 24 heures à l'avance.

### Objectif Principal
Prédire les valeurs de **pH**, **turbidité** et **température** pour les 24 prochaines heures dans 10 zones géographiques, permettant une gestion proactive de la qualité de l'eau.

---

## 🏗️ Architecture du Modèle

### Réseau ConvLSTM Encoder-Decoder avec Hour Embedding

```
┌──────────────────────────────────────────────────────────────┐
│                        ENTRÉE                                 │
│  Séquence temporelle: (batch, 12, 3, 4, 4)                   │
│  → 12 pas de temps × 3 paramètres × grille spatiale 4×4      │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   CONVLSTM ENCODER                            │
│  • Kernel: 3×3                                                │
│  • Hidden dimensions: 32                                      │
│  • Capture les corrélations spatio-temporelles               │
│  • Extrait les patterns entre zones géographiques            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   HOUR EMBEDDING                              │
│  • Entrée: Heure cible (0-23)                                │
│  • Architecture: Linear(1→16) → ReLU → Linear(16→32)         │
│  • Apprend les variations cycliques jour/nuit                │
│  • Capture les patterns de température diurne                │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      DECODER MLP                              │
│  • Input: 512 (spatial) + 32 (hour) = 544 features           │
│  • Linear(544→256) → ReLU → Dropout(0.2)                     │
│  • Linear(256→128) → ReLU                                    │
│  • Linear(128→30) → Sigmoid                                  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                        SORTIE                                 │
│  Prédictions: (batch, 10, 3)                                 │
│  → 10 zones × 3 paramètres (pH, turbidité, température)      │
└──────────────────────────────────────────────────────────────┘
```

### Composants Clés

| Composant | Description | Taille |
|-----------|-------------|--------|
| **ConvLSTMCell** | Cellule récurrente avec convolutions 2D | 32 hidden units |
| **Hour Embedding** | Encodage de l'heure cible | 1 → 32 dimensions |
| **Decoder** | MLP pour générer les prédictions | 544 → 30 |
| **Total Paramètres** | - | ~500,000 |

---

## 📊 Données d'Entrée

### Format des Données Capteurs

```python
# Shape: (batch_size, sequence_length, channels, height, width)
# Example: (1, 12, 3, 4, 4)

input_data = {
    'sequence_length': 12,      # 12 derniers points temporels
    'channels': 3,              # pH, turbidité, température
    'spatial_grid': (4, 4),     # Grille 4×4 pour les 10 zones
}
```

### Zones Géographiques Couvertes

| Zone | Latitude | Longitude | Caractéristiques |
|------|----------|-----------|------------------|
| Rabat-Centre | 34.0209 | -6.8416 | Urbain |
| Salé-Nord | 34.0286 | -6.8500 | Résidentiel |
| Salé-Sud | 34.0150 | -6.8450 | Résidentiel |
| Hay-Riad | 34.0250 | -6.8350 | Urbain |
| Agdal | 34.0100 | -6.8500 | Commercial |
| Côte-Océan | 34.0350 | -6.8250 | Côtier |
| Bouregreg | 34.0180 | -6.8380 | Rivière |
| Temara | 33.9200 | -6.9100 | Suburban |
| Skhirat | 33.8500 | -7.0300 | Côtier |
| Marrakech | 31.6295 | -7.9811 | Urbain (Test) |

---

## 🔄 Pipeline de Prédiction

### Flux de Données

```
1. Collecte des données (get_sensor_data_robust)
   │
   ├── Fenêtre 6h → Données fraîches (confiance haute)
   ├── Fenêtre 24h → Données récentes
   ├── Fenêtre 7 jours → Données anciennes
   └── Fenêtre 30 jours → Imputation si nécessaire
   
2. Construction du tenseur (build_input_tensor)
   │
   └── Normalisation: pH [5.5,9.5]→[0,1], Turb [0,8]→[0,1], Temp [10,35]→[0,1]
   
3. Prédiction pour chaque heure (run_hourly_predictions)
   │
   ├── Génère 24 prédictions (00:00 à 23:00 demain)
   ├── Applique variations horaires réalistes
   └── Calcule scores qualité et risque
   
4. Stockage en base (TimescaleDB)
```

### Variations Horaires Appliquées

```python
# Cycle jour/nuit pour la température
hour_factor = sin((hour - 6) × π / 12)  # Pic à 12h

# Variations par paramètre:
pH:         ±0.2 (stable)
Turbidité:  ±1.5 NTU (activité humaine)
Température: ±4°C (cycle solaire)
```

---

## 📈 Scores et Métriques

### Score de Qualité (0-100)

```python
# Pondération des paramètres:
qualite_score = (
    0.40 × score_ph +        # 40% pour le pH
    0.35 × score_turb +      # 35% pour la turbidité
    0.25 × score_temp        # 25% pour la température
)

# Niveaux:
"Excellente" → score >= 80
"Bonne"      → score >= 60
"Moyenne"    → score >= 40
"Faible"     → score < 40
```

### Score de Risque (0-100)

| Paramètre | Warning | Critical |
|-----------|---------|----------|
| pH | <6.5 ou >8.5 (+20%) | <6.0 ou >9.0 (+40%) |
| Turbidité | >1.0 NTU (+20%) | >5.0 NTU (+40%) |
| Température | >25°C (+15%) | >30°C (+30%) |

### Confiance des Prédictions

```python
# Base: 50%
# + Bonus données fraîches (6h): +40%
# + Bonus données récentes (24h): +25%
# + Bonus quantité données: +0.1% par mesure (max 10%)
# Maximum: 95%
```

---

## 🏋️ Entraînement du Modèle

### Commande d'entraînement

```bash
docker exec stmodel python stmodel.py --train
```

### Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| Époques | 30 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | MSE (Mean Squared Error) |
| Train/Val Split | 80/20 |

### Métriques d'Évaluation

- **MSE** (Mean Squared Error): Erreur quadratique moyenne
- **MAE** (Mean Absolute Error): Erreur absolue moyenne
- **R²**: Coefficient de détermination
- **Accuracy <5%**: % prédictions avec erreur <5%
- **Accuracy <10%**: % prédictions avec erreur <10%

---

## 📁 Structure des Fichiers

```
stmodel/
├── stmodel.py              # Code principal du modèle
├── requirements.txt        # Dépendances Python
├── Dockerfile              # Image Docker
├── .env                    # Variables d'environnement
├── weights/                
│   └── trained_weights.pth # Poids du modèle entraîné
├── generate_historical_data.py  # Génération données synthétiques
└── test_model_standalone.py     # Tests unitaires
```

---

## 🐳 Déploiement Docker

### Build de l'image

```bash
docker build -t aquawatch/stmodel:latest ./stmodel
```

### Exécution

```bash
docker run -d \
  --name stmodel \
  -e TIMESCALEDB_HOST=timescaledb \
  -e TIMESCALEDB_PORT=5432 \
  -e STM_INTERVAL_SECONDS=300 \
  aquawatch/stmodel:latest
```

### Variables d'Environnement

| Variable | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `TIMESCALEDB_HOST` | Hôte de la base de données | localhost |
| `TIMESCALEDB_PORT` | Port TimescaleDB | 5433 |
| `TIMESCALEDB_DB` | Nom de la base | aquawatch |
| `TIMESCALEDB_USER` | Utilisateur | postgres |
| `TIMESCALEDB_PASSWORD` | Mot de passe | postgres |
| `STM_INTERVAL_SECONDS` | Intervalle de prédiction | 300 (5 min) |

---

## 🔌 Intégration avec l'API

### Endpoint des Prédictions

```
GET /api/predictions?date=YYYY-MM-DD
```

### Réponse JSON

```json
{
  "predictions": [
    {
      "timestamp": "2025-12-24T08:00:00+01:00",
      "zone_id": "Rabat-Centre",
      "ph_pred": 7.2,
      "turbidite_pred": 0.8,
      "temperature_pred": 22.5,
      "qualite_score": 85.3,
      "qualite_niveau": "Excellente",
      "risque_score": 10.0,
      "risque_niveau": "Faible",
      "confidence": 92.5
    }
  ]
}
```

---

## 📚 Références Scientifiques

- **ConvLSTM**: Shi, X., et al. (2015). "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
- **Normes OMS**: Organisation Mondiale de la Santé - Directives pour la qualité de l'eau de boisson

---

## 👥 Équipe

- **Ghayt El Idrissi Dafali**
- **Reda Bouimakliouine**
- **Souhail Azzimani**
- **Amine Ibnou Chiekh**

**EMSI Marrakech - 2025-2026**

---

## 📄 Licence

Projet académique - EMSI Marrakech
