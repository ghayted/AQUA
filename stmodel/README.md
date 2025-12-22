# 🧠 Module Machine Learning - AquaWatch

## Prédiction de la Qualité de l'Eau par Deep Learning

**Document technique pour le département ML**

---

## 📌 Contexte du Projet

Ce module utilise un réseau de neurones profond pour **prédire la qualité de l'eau 24h à l'avance** dans 10 zones géographiques. Le modèle apprend les patterns spatio-temporels des données capteurs pour anticiper les dépassements des seuils OMS.

---

## 1️⃣ Les Données

### 1.1 Source des Données

Les données proviennent de **16 capteurs IoT** simulés qui mesurent en continu :

| Paramètre | Unité | Plage normale (OMS) | Plage critique |
|-----------|-------|---------------------|----------------|
| **pH** | - | 6.5 - 8.5 | < 6.0 ou > 9.0 |
| **Turbidité** | NTU | < 1.0 | > 5.0 |
| **Température** | °C | < 25 | > 30 |

### 1.2 Structure de la Base de Données (TimescaleDB)

```sql
-- Table des mesures capteurs
CREATE TABLE donnees_capteurs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,      -- Horodatage
    capteur_id VARCHAR(50),              -- Ex: CAPT-1, CAPT-2...
    zone VARCHAR(50),                    -- Ex: Rabat-Centre, Salé-Nord...
    ph DECIMAL(5,2),                     -- Valeur pH
    turbidite DECIMAL(5,2),              -- Turbidité en NTU
    temperature DECIMAL(5,2),            -- Température en °C
    latitude DECIMAL(10,6),
    longitude DECIMAL(10,6)
);

-- Table des prédictions (sortie du modèle)
CREATE TABLE predictions_qualite (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,       -- Heure de la prédiction
    zone_id VARCHAR(50),                  -- Zone prédite
    ph_pred DECIMAL(5,2),                 -- pH prédit
    turbidite_pred DECIMAL(5,2),          -- Turbidité prédite
    temperature_pred DECIMAL(5,2),        -- Température prédite
    qualite_score DECIMAL(5,2),           -- Score 0-100
    risque_score DECIMAL(5,2),            -- Risque 0-100
    confidence DECIMAL(5,2),              -- Confiance 0-100
    PRIMARY KEY (timestamp, id)
);
```

### 1.3 Génération des Données d'Entraînement

Pour l'entraînement initial, un script génère des données historiques réalistes :

```python
# Distribution des données par zone:
# Zones normales (CAPT-1 à CAPT-15):
#   - 80% données bonnes (pH 6.8-7.8, turb < 1.2 NTU)
#   - 15% données warning 
#   - 5% données critiques

# Zone Marrakech (CAPT-16) - cas de test critique:
#   - 60% données critiques (pour tester les alertes)
#   - 30% données bonnes
#   - 10% données warning
```

**Variations temporelles appliquées** (cycle jour/nuit) :
```python
hour_factor = sin((heure - 6) × π / 12)  # Pic à 12h, creux à minuit
température = base_temp + hour_factor × 3  # ±3°C selon l'heure
```

---

## 2️⃣ Architecture du Modèle

### 2.1 Choix de l'Architecture : ConvLSTM

**Pourquoi ConvLSTM ?**
- Combine les **convolutions 2D** (pour les corrélations spatiales entre zones)
- Avec les **cellules LSTM** (pour les dépendances temporelles)
- Parfait pour des données spatio-temporelles comme les capteurs géolocalisés

### 2.2 Schéma de l'Architecture

```
                    ENTRÉE
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│   Séquence    │           │     Heure     │
│  temporelle   │           │    cible      │
│ (12, 3, 4, 4) │           │    (0-23)     │
└───────┬───────┘           └───────┬───────┘
        │                           │
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│   ConvLSTM    │           │     Hour      │
│   Encoder     │           │   Embedding   │
│  (32 hidden)  │           │   (1 → 32)    │
└───────┬───────┘           └───────┬───────┘
        │                           │
        │   512 features            │  32 features
        │                           │
        └───────────┬───────────────┘
                    │
                    ▼  544 features
            ┌───────────────┐
            │    Decoder    │
            │     MLP       │
            │ 544→256→128→30│
            └───────┬───────┘
                    │
                    ▼
               SORTIE
            (10 zones × 3 params)
```

### 2.3 Détail des Composants

#### A) Cellule ConvLSTM

```python
class ConvLSTMCell(nn.Module):
    """
    Cellule LSTM avec convolutions 2D au lieu de multiplications matricielles.
    Permet de capturer les corrélations spatiales entre zones voisines.
    """
    def __init__(self, input_dim=3, hidden_dim=32, kernel_size=3):
        # Convolution sur les 4 gates LSTM (input, forget, output, cell)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,  # Concat input + hidden
            out_channels=4 * hidden_dim,          # 4 gates
            kernel_size=3,
            padding=1  # Préserve la taille spatiale
        )
    
    def forward(self, x, state):
        h, c = state  # hidden state, cell state
        
        # Concaténer input et hidden state
        combined = concat([x, h], dim=1)
        
        # Convolution puis split en 4 gates
        gates = self.conv(combined)
        i, f, o, g = split(gates, 4)  # input, forget, output, cell gate
        
        # Équations LSTM classiques
        c_next = sigmoid(f) * c + sigmoid(i) * tanh(g)
        h_next = sigmoid(o) * tanh(c_next)
        
        return h_next, c_next
```

#### B) Hour Embedding

```python
# Pourquoi encoder l'heure ?
# La qualité de l'eau varie selon l'heure :
# - Température plus élevée à midi
# - Turbidité plus haute aux heures d'activité humaine

class HourEmbedding(nn.Module):
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Linear(1, 16),   # 1 entrée (heure normalisée)
            nn.ReLU(),
            nn.Linear(16, 32)   # 32 features en sortie
        )
    
    def forward(self, hour):
        # hour est normalisé : 0h → 0.0, 23h → 1.0
        return self.layers(hour)
```

#### C) Decoder MLP

```python
self.decoder = nn.Sequential(
    nn.Linear(512 + 32, 256),  # 544 entrées (spatial + heure)
    nn.ReLU(),
    nn.Dropout(0.2),           # Régularisation
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 30),        # 10 zones × 3 paramètres
    nn.Sigmoid()               # Sortie normalisée [0, 1]
)
```

---

## 3️⃣ Préparation des Données pour l'Entraînement

### 3.1 Format d'Entrée du Modèle

```python
# Tenseur d'entrée : (batch, sequence, channels, height, width)
# Exemple : (32, 12, 3, 4, 4)
#
# batch = 32          → 32 échantillons par batch
# sequence = 12       → 12 pas de temps historiques
# channels = 3        → pH, turbidité, température
# height × width = 4×4 → Grille spatiale pour 10 zones
```

### 3.2 Mapping Zones → Grille 4×4

```python
# Les 10 zones sont placées sur une grille 4×4 :
ZONES = ['Rabat-Centre', 'Salé-Nord', 'Salé-Sud', 'Hay-Riad', 
         'Agdal', 'Côte-Océan', 'Bouregreg', 'Temara', 
         'Skhirat', 'Marrakech']

# Position dans la grille (row, col) :
def zone_to_grid(zone_idx):
    row = zone_idx % 4   # 0, 1, 2, 3, 0, 1, 2, 3, 0, 1
    col = zone_idx // 4  # 0, 0, 0, 0, 1, 1, 1, 1, 2, 2
    return row, col
```

### 3.3 Normalisation des Données

```python
# Normalisation Min-Max vers [0, 1] :

# pH : plage réaliste [5.5, 9.5] → 4 unités
ph_normalized = (ph - 5.5) / 4.0

# Turbidité : plage [0, 8] NTU
turb_normalized = turbidite / 8.0

# Température : plage [10, 35]°C → 25 unités  
temp_normalized = (temperature - 10) / 25.0
```

### 3.4 Création des Séquences d'Entraînement

```python
def create_training_sequences(data):
    """
    Crée des paires (X, y) pour l'entraînement supervisé.
    X = 12 derniers pas de temps
    y = valeur à prédire (pas de temps suivant)
    """
    X_list, y_list, hour_list = [], [], []
    
    for zone in ZONES:
        zone_data = data[zone]  # Données triées par timestamp
        
        for i in range(len(zone_data) - SEQUENCE_LENGTH):
            # X : séquence de 12 observations passées
            sequence = zone_data[i : i + SEQUENCE_LENGTH]
            
            # y : observation cible (la suivante)
            target = zone_data[i + SEQUENCE_LENGTH]
            target_hour = target['hour'] / 23.0  # Normaliser heure
            
            X_list.append(sequence)
            y_list.append(target)
            hour_list.append(target_hour)
    
    return np.array(X_list), np.array(y_list), np.array(hour_list)
```

---

## 4️⃣ Entraînement du Modèle

### 4.1 Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| **Époques** | 30 | Suffisant pour convergence |
| **Batch size** | 32 | Bon compromis mémoire/généralisation |
| **Learning rate** | 0.001 | Standard pour Adam |
| **Optimizer** | Adam | Adaptatif, converge vite |
| **Loss** | MSE | Régression continue |
| **Train/Val split** | 80/20 | Standard ML |
| **Dropout** | 0.2 | Évite le surapprentissage |

### 4.2 Boucle d'Entraînement

```python
def train_model(model, X_train, y_train, hours_train, epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y, batch_h in DataLoader(dataset, batch_size=32):
            # 1. Forward pass
            predictions = model(batch_x, batch_h)
            
            # 2. Calcul de la loss
            loss = criterion(predictions, batch_y)
            
            # 3. Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # 4. Mise à jour des poids
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation à chaque époque
        val_loss = evaluate(model, X_val, y_val, hours_val)
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_loss:
            torch.save(model.state_dict(), 'weights/trained_weights.pth')
```

### 4.3 Métriques d'Évaluation

```python
def compute_metrics(y_true, y_pred):
    # MSE : Erreur quadratique moyenne
    mse = mean((y_true - y_pred)²)
    
    # MAE : Erreur absolue moyenne
    mae = mean(|y_true - y_pred|)
    
    # R² : Coefficient de détermination
    r2 = 1 - (sum((y_true - y_pred)²) / sum((y_true - mean(y_true))²))
    
    # Accuracy à 5% : % prédictions avec erreur < 5%
    acc_5 = mean(|y_true - y_pred| < 0.05) × 100
    
    # Accuracy à 10%
    acc_10 = mean(|y_true - y_pred| < 0.10) × 100
    
    return mse, mae, r2, acc_5, acc_10
```

---

## 5️⃣ Phase de Prédiction

### 5.1 Pipeline de Prédiction

```
┌─────────────────────────────────────────────────────────────┐
│  1. RÉCUPÉRATION DES DONNÉES (get_sensor_data_robust)       │
│     → Cherche données des 6h, sinon 24h, sinon 7j, 30j      │
│     → Impute valeurs manquantes avec moyennes OMS           │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  2. CONSTRUCTION DU TENSEUR (build_input_tensor)            │
│     → Normalise les valeurs                                 │
│     → Place sur grille 4×4                                  │
│     → Répète pour 12 pas de temps avec bruit               │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  3. PRÉDICTION POUR CHAQUE HEURE (0h à 23h)                 │
│     → Appelle model.forward(tensor, hour)                   │
│     → Applique variations horaires réalistes                │
│     → Dénormalise les valeurs                               │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  4. CALCUL DES SCORES                                       │
│     → Score qualité (0-100)                                 │
│     → Score risque (0-100)                                  │
│     → Score confiance (basé sur fraîcheur données)          │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  5. INSERTION EN BASE (predictions_qualite)                 │
│     → 10 zones × 24 heures = 240 prédictions               │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Variations Horaires Réalistes

```python
def apply_hourly_variations(base_values, hour):
    """
    Applique des variations physiques réalistes basées sur l'heure.
    Simule le cycle jour/nuit.
    """
    # Facteur sinusoïdal : -1 (6h) → +1 (12h) → -1 (18h)
    hour_factor = sin((hour - 6) × π / 12)
    
    # pH : très stable, légère variation
    ph = base_ph + 0.1 × hour_factor + random.normal(0, 0.05)
    
    # Turbidité : varie avec activité humaine (pic le jour)
    turb = base_turb + 0.8 × |hour_factor| + random.normal(0, 0.2)
    
    # Température : suit cycle solaire (±4°C)
    temp = base_temp + 3.0 × hour_factor + random.normal(0, 0.5)
    
    return ph, turb, temp
```

### 5.3 Calcul du Score de Qualité

```python
def compute_quality_score(ph, turb, temp):
    """
    Score de qualité globale [0-100] basé sur les normes OMS.
    """
    # Score pH (40% du total)
    # Optimal: 7.0, acceptable: 6.5-8.5
    ph_score = max(0, 100 - abs(ph - 7.0) × 30)
    
    # Score turbidité (35% du total)
    # Optimal: < 1 NTU, limite: 5 NTU
    if turb <= 1.0:
        turb_score = 100
    elif turb <= 5.0:
        turb_score = max(0, 80 - (turb - 1) × 15)
    else:
        turb_score = max(0, 20 - (turb - 5) × 5)
    
    # Score température (25% du total)
    # Optimal: < 25°C, limite: 30°C
    if temp <= 25:
        temp_score = 100
    elif temp <= 30:
        temp_score = max(0, 80 - (temp - 25) × 10)
    else:
        temp_score = max(0, 30 - (temp - 30) × 5)
    
    # Score pondéré
    total = ph_score × 0.40 + turb_score × 0.35 + temp_score × 0.25
    
    # Classification
    if total >= 80: niveau = "Excellente"
    elif total >= 60: niveau = "Bonne"
    elif total >= 40: niveau = "Moyenne"
    else: niveau = "Faible"
    
    return total, niveau
```

### 5.4 Calcul du Score de Confiance

```python
def compute_confidence(zone_data, data_quality):
    """
    Estime la fiabilité de la prédiction basée sur la qualité des données.
    """
    base = 50.0  # Base de 50%
    
    # Bonus selon la fraîcheur des données
    if zone_data['window'] == '6 hours':
        base += 40.0   # Données très récentes
    elif zone_data['window'] == '24 hours':
        base += 25.0   # Données récentes
    elif zone_data['window'] == '7 days':
        base += 10.0   # Données anciennes
    # Sinon données imputées: pas de bonus
    
    # Bonus pour quantité de données
    count_bonus = min(10.0, zone_data['count'] × 0.1)
    
    confidence = min(95.0, base + count_bonus)
    return confidence
```

---

## 6️⃣ Technologies Utilisées

| Technologie | Version | Usage |
|-------------|---------|-------|
| **Python** | 3.11+ | Langage principal |
| **PyTorch** | 2.x | Framework deep learning |
| **NumPy** | 1.24+ | Calcul numérique |
| **psycopg2** | 2.9+ | Connexion PostgreSQL |
| **TimescaleDB** | 2.x | Base de données time-series |
| **Docker** | 24+ | Conteneurisation |

---

## 7️⃣ Commandes Utiles

```bash
# Générer données historiques (15 jours)
docker exec stmodel python generate_historical_data.py

# Lancer l'entraînement
docker exec stmodel python stmodel.py --train

# Voir les logs du modèle
docker logs -f stmodel

# Vérifier les prédictions en base
docker exec -it timescaledb psql -U postgres -d aquawatch -c \
  "SELECT * FROM predictions_qualite ORDER BY timestamp DESC LIMIT 10;"
```

---

## 👥 Équipe

- **Ghayt El Idrissi Dafali**
- **Reda Bouimakliouine**
- **Souhail Azzimani**
- **Amine Ibnou Chiekh**

**EMSI Marrakech - Année Universitaire 2025-2026**
