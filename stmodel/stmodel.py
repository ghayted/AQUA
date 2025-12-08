#!/usr/bin/env python3
"""
Service stmodel pour AquaWatch
Implémente un ConvLSTM spatio-temporel qui exploite les séries temporelles
multisources pour estimer la qualité de l'eau à court terme.
"""

import os
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
import torch.nn as nn

# Configuration base
DB_CONFIG = {
    'host': os.getenv('TIMESCALEDB_HOST', 'timescaledb'),
    'port': int(os.getenv('TIMESCALEDB_PORT', '5432')),
    'database': os.getenv('TIMESCALEDB_DB', 'aquawatch'),
    'user': os.getenv('TIMESCALEDB_USER', 'postgres'),
    'password': os.getenv('TIMESCALEDB_PASSWORD', 'postgres'),
}

PREDICTION_INTERVAL = int(os.getenv('STM_INTERVAL_SECONDS', '300'))
SEQUENCE_LENGTH = int(os.getenv('STM_SEQUENCE_LENGTH', '12'))
RESET_SCHEMA = os.getenv('STM_RESET_SCHEMA', 'true').lower() == 'true'
PREDICTION_HORIZONS = [24, 72]  # heures
GRID_SHAPE = (4, 4)
CHANNELS = 3  # pH, turbidité, température

ZONES = {
    'Rabat-Centre': {'lat': 34.0209, 'lon': -6.8416},
    'Salé-Nord': {'lat': 34.0286, 'lon': -6.8500},
    'Salé-Sud': {'lat': 34.0150, 'lon': -6.8450},
    'Hay-Riad': {'lat': 34.0250, 'lon': -6.8350},
    'Agdal': {'lat': 34.0100, 'lon': -6.8500},
    'Côte-Océan': {'lat': 34.0350, 'lon': -6.8250},
    'Bouregreg': {'lat': 34.0180, 'lon': -6.8380},
    'Temara': {'lat': 33.9200, 'lon': -6.9100},
    'Skhirat': {'lat': 33.8500, 'lon': -7.0300},
}


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, state):
        h_cur, c_cur = state
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch, spatial_size, device=None):
        height, width = spatial_size
        h = torch.zeros(batch, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch, self.hidden_dim, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(num_layers):
            self.layers.append(ConvLSTMCell(dims[i], dims[i + 1], kernel_size))

    def forward(self, x):
        # x: (batch, seq_len, channels, H, W)
        batch, seq_len, _, h, w = x.shape
        cur_input = x
        device = x.device
        hidden_states = []

        for layer in self.layers:
            h_t, c_t = layer.init_hidden(batch, (h, w), device=device)
            outputs = []
            for t in range(seq_len):
                h_t, c_t = layer(cur_input[:, t], (h_t, c_t))
                outputs.append(h_t.unsqueeze(1))
            cur_input = torch.cat(outputs, dim=1)
            hidden_states.append(h_t)

        return hidden_states[-1]


class SpatioTemporalForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlstm = ConvLSTM(
            input_dim=CHANNELS,
            hidden_dims=[16, 32],
            kernel_size=3,
            num_layers=2,
        )
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        latent = self.convlstm(x)
        logits = self.head(latent)
        return torch.sigmoid(logits)


def wait_for_db(retries=30, delay=2):
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.close()
            return True
        except psycopg2.OperationalError:
            print(f'⏳ Attente TimescaleDB ({attempt}/{retries})')
            time.sleep(delay)
    return False


def init_database():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    if RESET_SCHEMA:
        cursor.execute("DROP TABLE IF EXISTS predictions_qualite CASCADE;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions_qualite (
            id BIGSERIAL,
            timestamp TIMESTAMPTZ NOT NULL,
            zone_id VARCHAR(50) NOT NULL,
            qualite_score DECIMAL(5,2),
            qualite_niveau VARCHAR(20),
            risque_score DECIMAL(5,2),
            risque_niveau VARCHAR(20),
            prediction_score DECIMAL(5,2),
            ph_pred DECIMAL(5,2),
            turbidite_pred DECIMAL(5,2),
            temperature_pred DECIMAL(5,2),
            prediction_horizon INTEGER DEFAULT 24,
            confidence DECIMAL(5,2)
        );
    """)
    cursor.execute("ALTER TABLE predictions_qualite ALTER COLUMN id SET DATA TYPE BIGINT;")
    cursor.execute("ALTER TABLE predictions_qualite ALTER COLUMN id SET NOT NULL;")
    cursor.execute("ALTER TABLE predictions_qualite ALTER COLUMN timestamp SET NOT NULL;")
    cursor.execute("ALTER TABLE predictions_qualite ADD COLUMN IF NOT EXISTS prediction_horizon INTEGER DEFAULT 24;")
    cursor.execute("ALTER TABLE predictions_qualite ADD COLUMN IF NOT EXISTS confidence DECIMAL(5,2);")
    cursor.execute("ALTER TABLE predictions_qualite DROP CONSTRAINT IF EXISTS predictions_qualite_pkey;")
    cursor.execute("ALTER TABLE predictions_qualite ADD PRIMARY KEY (timestamp, id);")
    cursor.execute("""
        SELECT create_hypertable('predictions_qualite', 'timestamp', if_not_exists => TRUE);
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_zone_id ON predictions_qualite(zone_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_horizon ON predictions_qualite(prediction_horizon);")
    conn.commit()
    cursor.close()
    print('✅ Table predictions_qualite prête')
    return conn


def fetch_sensor_data(conn):
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT timestamp, ph, turbidite, temperature, latitude, longitude
        FROM donnees_capteurs
        WHERE timestamp > NOW() - INTERVAL '3 hours'
        ORDER BY timestamp ASC
    """)
    rows = cursor.fetchall()
    cursor.close()
    return rows


def synthesize_sensor_rows():
    """Génère un jeu de données synthétique quand aucune mesure n'est disponible."""
    now = datetime.utcnow()
    rows = []
    for zone, coords in ZONES.items():
        rows.append({
            'timestamp': now,
            'ph': float(np.clip(np.random.normal(7.2, 0.4), 5.5, 9.5)),
            'turbidite': float(np.clip(np.random.normal(1.2, 0.6), 0.1, 6.0)),
            'temperature': float(np.clip(np.random.normal(21.0, 3.0), 15.0, 32.0)),
            'latitude': coords['lat'],
            'longitude': coords['lon'],
        })
    return rows


def grid_cell(lat, lon):
    lat_min, lat_max = 33.80, 34.10
    lon_min, lon_max = -7.10, -6.70
    row = int(((lat - lat_min) / (lat_max - lat_min)) * GRID_SHAPE[0])
    col = int(((lon - lon_min) / (lon_max - lon_min)) * GRID_SHAPE[1])
    row = max(0, min(GRID_SHAPE[0] - 1, row))
    col = max(0, min(GRID_SHAPE[1] - 1, col))
    return row, col


def normalize(ph, turbidity, temperature):
    ph_n = np.clip((ph - 5.5) / 4.0, 0, 1)
    turb_n = np.clip(turbidity / 8.0, 0, 1)
    temp_n = np.clip((temperature - 10) / 25.0, 0, 1)
    return ph_n, turb_n, temp_n


def build_sequences(rows):
    if not rows:
        return None, None
    sequences = []
    current = np.zeros((SEQUENCE_LENGTH, CHANNELS, *GRID_SHAPE), dtype=np.float32)
    latest_frame = np.zeros((CHANNELS, *GRID_SHAPE), dtype=np.float32)
    step = 0

    for row in rows:
        ph_n, turb_n, temp_n = normalize(row['ph'], row['turbidite'], row['temperature'])
        r, c = grid_cell(row['latitude'], row['longitude'])
        current[step, 0, r, c] = ph_n
        current[step, 1, r, c] = turb_n
        current[step, 2, r, c] = temp_n
        latest_frame[:, r, c] = [ph_n, turb_n, temp_n]
        step += 1
        if step == SEQUENCE_LENGTH:
            sequences.append(current.copy())
            current = np.zeros((SEQUENCE_LENGTH, CHANNELS, *GRID_SHAPE), dtype=np.float32)
            step = 0

    if step > 0:
        sequences.append(current)
    return np.stack(sequences), latest_frame


def load_model():
    model = SpatioTemporalForecaster()
    weights_path = os.getenv('STM_WEIGHTS_PATH')
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f'✅ Poids chargés depuis {weights_path}')
    else:
        print('ℹ️  Aucun poids fournis - utilisation des paramètres initiaux.')
    model.eval()
    return model


def resolve_zone(lat, lon):
    best_zone = None
    best_dist = float('inf')
    for zone, coords in ZONES.items():
        dist = (coords['lat'] - lat) ** 2 + (coords['lon'] - lon) ** 2
        if dist < best_dist:
            best_dist = dist
            best_zone = zone
    return best_zone or 'Zone-Inconnue'


def aggregate_zone_metrics(rows):
    zone_data = defaultdict(lambda: {'ph': [], 'turbidite': [], 'temperature': [], 'count': 0})
    for row in rows:
        zone = resolve_zone(row['latitude'], row['longitude'])
        zone_data[zone]['ph'].append(row['ph'])
        zone_data[zone]['turbidite'].append(row['turbidite'])
        zone_data[zone]['temperature'].append(row['temperature'])
        zone_data[zone]['count'] += 1

    aggregates = {}
    for zone, values in zone_data.items():
        if values['count'] == 0:
            continue
        aggregates[zone] = {
            'ph': float(np.mean(values['ph'])),
            'turbidite': float(np.mean(values['turbidite'])),
            'temperature': float(np.mean(values['temperature'])),
            'count': values['count'],
        }
    return aggregates


def level_from_score(score):
    if score >= 80:
        return 'Excellente'
    if score >= 60:
        return 'Bonne'
    if score >= 40:
        return 'Moyenne'
    return 'Faible'


def risk_from_score(score):
    if score < 30:
        return 'Faible'
    if score < 60:
        return 'Modéré'
    return 'Élevé'


def derive_scores(global_vector, zone_stats):
    g_quality, g_risk, g_score = global_vector
    base = 100 - abs(zone_stats['ph'] - 7.2) * 18 - zone_stats['turbidite'] * 9
    base -= max(0, zone_stats['temperature'] - 25) * 1.5
    base = np.clip(base, 0, 100)
    qualite = round(0.6 * base + 0.4 * g_quality * 100, 2)
    risque = round(0.5 * (100 - base) + 0.5 * g_risk * 100, 2)
    score = round(0.5 * qualite + 0.5 * g_score * 100, 2)
    confidence = round(min(100, zone_stats['count'] * 5), 2)
    return qualite, risque, score, confidence


def insert_prediction(conn, payload):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions_qualite (
            timestamp, zone_id, qualite_score, qualite_niveau,
            risque_score, risque_niveau, prediction_score,
            ph_pred, turbidite_pred, temperature_pred,
            prediction_horizon, confidence
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        payload['timestamp'],
        payload['zone_id'],
        payload['qualite_score'],
        payload['qualite_niveau'],
        payload['risque_score'],
        payload['risque_niveau'],
        payload['prediction_score'],
        payload['ph_pred'],
        payload['turbidite_pred'],
        payload['temperature_pred'],
        payload['prediction_horizon'],
        payload['confidence'],
    ))
    conn.commit()
    cursor.close()
    print(
        f"🤖 {payload['zone_id']} H+{payload['prediction_horizon']}h - "
        f"{payload['qualite_niveau']} ({payload['qualite_score']}%) | "
        f"{payload['risque_niveau']} ({payload['risque_score']}%)"
    )


def main():
    print('🚀 Démarrage du service stmodel...')
    if not wait_for_db():
        print('❌ Impossible de contacter TimescaleDB')
        return

    conn = init_database()
    model = load_model()

    try:
        while True:
            rows = fetch_sensor_data(conn)
            if not rows:
                print('ℹ️  Aucune mesure en base - génération de données synthétiques pour maintenir la simulation.')
                rows = synthesize_sensor_rows()

            sequences, latest_frame = build_sequences(rows)
            if sequences is None:
                time.sleep(PREDICTION_INTERVAL)
                continue

            tensor = torch.tensor(sequences, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(tensor).numpy()
            global_vector = outputs.mean(axis=0)

            zone_metrics = aggregate_zone_metrics(rows)
            now = datetime.utcnow()

            for zone, stats in zone_metrics.items():
                qualite, risque, score, confidence = derive_scores(global_vector, stats)
                for horizon in PREDICTION_HORIZONS:
                    insert_prediction(conn, {
                        'timestamp': now + timedelta(hours=horizon),
                        'zone_id': zone,
                        'qualite_score': qualite,
                        'qualite_niveau': level_from_score(qualite),
                        'risque_score': risque,
                        'risque_niveau': risk_from_score(risque),
                        'prediction_score': score,
                        'ph_pred': round(stats['ph'], 2),
                        'turbidite_pred': round(stats['turbidite'], 2),
                        'temperature_pred': round(stats['temperature'], 2),
                        'prediction_horizon': horizon,
                        'confidence': confidence,
                    })

            time.sleep(PREDICTION_INTERVAL)
    except KeyboardInterrupt:
        print('\n🛑 Arrêt du service stmodel...')
    except Exception as exc:
        print(f'❌ Erreur stmodel: {exc}')
    finally:
        if conn:
            conn.close()
        print('✅ Service stmodel arrêté')


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Service stmodel pour AquaWatch
Simule des prédictions de qualité d'eau toutes les 5 secondes
"""

import os
import time
import random
import psycopg2
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Backend headless pour matplotlib
import matplotlib.pyplot as plt

# Configuration de la base de données
DB_CONFIG = {
    'host': os.getenv('TIMESCALEDB_HOST', 'timescaledb'),
    'port': int(os.getenv('TIMESCALEDB_PORT', '5432')),
    'database': os.getenv('TIMESCALEDB_DB', 'aquawatch'),
    'user': os.getenv('TIMESCALEDB_USER', 'postgres'),
    'password': os.getenv('TIMESCALEDB_PASSWORD', 'postgres'),
}

# Modèle simple de réseau de neurones pour la simulation
class WaterQualityModel(nn.Module):
    """Modèle simple de prédiction de qualité d'eau"""
    def __init__(self):
        super(WaterQualityModel, self).__init__()
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)  # 3 sorties: qualité, risque, score
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def random_float(min_val, max_val):
    """Génère une valeur aléatoire dans une plage"""
    return round(random.uniform(min_val, max_val), 2)

def generate_prediction_data():
    """Génère des données de prédiction simulées"""
    # Simuler des entrées (pH, turbidité, température, chlorophylle, NDWI, etc.)
    inputs = np.array([[
        random_float(6.5, 8.5),      # pH
        random_float(0.1, 5.0),      # turbidité
        random_float(15.0, 25.0),    # température
        random_float(0.5, 15.0),     # chlorophylle
        random_float(-0.5, 0.8),     # NDWI
        random_float(0.0, 1.0),      # facteur supplémentaire
    ]])
    
    # Utiliser le modèle pour générer des prédictions
    model = WaterQualityModel()
    model.eval()
    
    with torch.no_grad():
        inputs_tensor = torch.FloatTensor(inputs)
        outputs = model(inputs_tensor)
        outputs = torch.sigmoid(outputs)  # Normaliser entre 0 et 1
    
    # Extraire les prédictions
    qualite = float(outputs[0][0].item())
    risque = float(outputs[0][1].item())
    score = float(outputs[0][2].item())
    
    # Convertir en valeurs interprétables
    qualite_niveau = "Excellente" if qualite > 0.7 else "Bonne" if qualite > 0.4 else "Moyenne" if qualite > 0.2 else "Faible"
    risque_niveau = "Faible" if risque < 0.3 else "Modéré" if risque < 0.6 else "Élevé"
    
    return {
        'timestamp': datetime.now().isoformat(),
        'zone_id': f'ZONE-{random.randint(1, 5)}',
        'qualite_score': round(qualite * 100, 2),
        'qualite_niveau': qualite_niveau,
        'risque_score': round(risque * 100, 2),
        'risque_niveau': risque_niveau,
        'prediction_score': round(score * 100, 2),
        'ph_pred': round(inputs[0][0], 2),
        'turbidite_pred': round(inputs[0][1], 2),
        'temperature_pred': round(inputs[0][2], 2),
    }

def init_database():
    """Initialise la base de données et crée les tables nécessaires"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Créer la table si elle n'existe pas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions_qualite (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                zone_id VARCHAR(50) NOT NULL,
                qualite_score DECIMAL(5,2),
                qualite_niveau VARCHAR(20),
                risque_score DECIMAL(5,2),
                risque_niveau VARCHAR(20),
                prediction_score DECIMAL(5,2),
                ph_pred DECIMAL(5,2),
                turbidite_pred DECIMAL(5,2),
                temperature_pred DECIMAL(5,2)
            );
        """)
        
        # Créer l'hypertable TimescaleDB si elle n'existe pas
        cursor.execute("""
            SELECT create_hypertable('predictions_qualite', 'timestamp', if_not_exists => TRUE);
        """)
        
        # Créer un index sur zone_id
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_zone_id ON predictions_qualite(zone_id);
        """)
        
        conn.commit()
        cursor.close()
        print('✅ Table et hypertable créées/initialisées')
        return conn
    except psycopg2.Error as e:
        print(f'❌ Erreur lors de l\'initialisation de la base de données: {e}')
        if conn:
            conn.rollback()
        raise

def insert_prediction_data(conn, data):
    """Insère des données de prédiction dans la base de données"""
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO predictions_qualite 
            (timestamp, zone_id, qualite_score, qualite_niveau, risque_score, 
             risque_niveau, prediction_score, ph_pred, turbidite_pred, temperature_pred)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            data['timestamp'],
            data['zone_id'],
            data['qualite_score'],
            data['qualite_niveau'],
            data['risque_score'],
            data['risque_niveau'],
            data['prediction_score'],
            data['ph_pred'],
            data['turbidite_pred'],
            data['temperature_pred'],
        ))
        conn.commit()
        cursor.close()
        print(f"🤖 Prédiction insérée: {data['zone_id']} - "
              f"Qualité: {data['qualite_niveau']} ({data['qualite_score']}%), "
              f"Risque: {data['risque_niveau']} ({data['risque_score']}%)")
    except psycopg2.Error as e:
        print(f'❌ Erreur lors de l\'insertion des données: {e}')
        conn.rollback()

def wait_for_db(max_retries=30, retry_delay=2):
    """Attend que la base de données soit disponible"""
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.close()
            return True
        except psycopg2.OperationalError:
            print(f'⏳ Attente de la base de données... ({i+1}/{max_retries})')
            time.sleep(retry_delay)
    return False

def main():
    """Fonction principale"""
    print('🚀 Démarrage du service stmodel...')
    
    # Attendre que la base de données soit disponible
    if not wait_for_db():
        print('❌ Impossible de se connecter à la base de données')
        return
    
    # Initialiser la base de données
    conn = init_database()
    
    print('✅ Service stmodel démarré - Génération de prédictions toutes les 5 secondes')
    
    try:
        # Générer et envoyer des prédictions toutes les 5 secondes
        while True:
            data = generate_prediction_data()
            insert_prediction_data(conn, data)
            time.sleep(5)
    except KeyboardInterrupt:
        print('\n🛑 Arrêt du service stmodel...')
    except Exception as e:
        print(f'❌ Erreur fatale: {e}')
    finally:
        if conn:
            conn.close()
        print('✅ Service stmodel arrêté')

if __name__ == '__main__':
    main()

