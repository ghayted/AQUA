#!/usr/bin/env python3
"""
Service satellite pour AquaWatch.
Récupère des observations Sentinel-2 (ou simule) puis stocke les indicateurs dans
TimescaleDB et pousse les GeoTIFF dérivés vers MinIO pour exploitation cartographique.
"""

import os
import random
import tempfile
import time
from datetime import datetime, timedelta

import numpy as np
import psycopg2
import rasterio
from minio import Minio
from rasterio.transform import from_bounds
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)

# Configuration de la base de données
DB_CONFIG = {
    'host': os.getenv('TIMESCALEDB_HOST', 'timescaledb'),
    'port': int(os.getenv('TIMESCALEDB_PORT', '5432')),
    'database': os.getenv('TIMESCALEDB_DB', 'aquawatch'),
    'user': os.getenv('TIMESCALEDB_USER', 'postgres'),
    'password': os.getenv('TIMESCALEDB_PASSWORD', 'postgres'),
}

FETCH_INTERVAL = int(os.getenv('SATELLITE_INTERVAL_SECONDS', '300'))
SIMULATION_FALLBACK = os.getenv('SATELLITE_SIMULATION_FALLBACK', 'true').lower() == 'true'
RESET_SCHEMA = os.getenv('SATELLITE_RESET_SCHEMA', 'true').lower() == 'true'

# MinIO
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio')
MINIO_PORT = os.getenv('MINIO_PORT', '9000')
MINIO_SECURE = os.getenv('MINIO_USE_SSL', 'false').lower() == 'true'
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'aquawatch-satellite')
MINIO_PUBLIC_ENDPOINT = os.getenv('MINIO_PUBLIC_ENDPOINT', 'http://localhost:9000')

# Zones d'intérêt (Rabat-Salé)
AOIS = [
    {
        'id': 'SAT-RABAT',
        'name': 'Rabat-Centre',
        'bbox': (-6.90, 33.98, -6.78, 34.08),
        'centroid': (34.03, -6.84),
    },
    {
        'id': 'SAT-BOUREGREG',
        'name': 'Bouregreg',
        'bbox': (-6.86, 33.97, -6.78, 34.05),
        'centroid': (34.00, -6.82),
    },
    {
        'id': 'SAT-COAST',
        'name': 'Côte Atlantique',
        'bbox': (-6.95, 33.95, -6.80, 34.05),
        'centroid': (34.02, -6.88),
    },
]

EVAL_SCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B03", "B04", "B08"],
      units: "REFLECTANCE"
    }],
    output: { bands: 3, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  var ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08 + 0.0000001);
  var turbidity = sample.B04 / (sample.B03 + 0.0000001);
  var chlorophyll = sample.B08 * 30.0;
  return [ndwi, turbidity, chlorophyll];
}
"""

def configure_sentinel():
    """Initialise la configuration SentinelHub à partir des variables d'environnement."""
    config = SHConfig()
    client_id = os.getenv('SENTINEL_CLIENT_ID')
    client_secret = os.getenv('SENTINEL_CLIENT_SECRET')
    if client_id and client_secret:
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret
        return config, True
    print('ℹ️  Identifiants SentinelHub absents - passage en mode simulation.')
    return config, False

def init_minio():
    """Initialise le client MinIO et crée le bucket si besoin."""
    client = Minio(
        f"{MINIO_ENDPOINT}:{MINIO_PORT}",
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)
        print(f'✅ Bucket MinIO créé: {MINIO_BUCKET}')
    return client

def random_float(min_val, max_val):
    return round(random.uniform(min_val, max_val), 2)

def wait_for_db(max_retries=30, retry_delay=2):
    """Attend que la base de données soit prête."""
    for attempt in range(1, max_retries + 1):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.close()
            return True
        except psycopg2.OperationalError:
            print(f'⏳ Attente de la base de données... ({attempt}/{max_retries})')
            time.sleep(retry_delay)
    return False

def init_database():
    """Crée la table TimescaleDB et l'hypertable."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    if RESET_SCHEMA:
        cursor.execute("DROP TABLE IF EXISTS donnees_satellite CASCADE;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS donnees_satellite (
            id BIGSERIAL,
            timestamp TIMESTAMPTZ NOT NULL,
            satellite_id VARCHAR(50) NOT NULL,
            zone VARCHAR(100),
            latitude DECIMAL(10,6),
            longitude DECIMAL(10,6),
            chlorophylle DECIMAL(8,2),
            turbidite DECIMAL(8,2),
            ndwi DECIMAL(5,2),
            storage_url TEXT,
            source VARCHAR(50)
        );
    """)

    # Aligner le schéma existant (si la table avait déjà été créée)
    cursor.execute("ALTER TABLE donnees_satellite ADD COLUMN IF NOT EXISTS zone VARCHAR(100);")
    cursor.execute("ALTER TABLE donnees_satellite ADD COLUMN IF NOT EXISTS storage_url TEXT;")
    cursor.execute("ALTER TABLE donnees_satellite ADD COLUMN IF NOT EXISTS source VARCHAR(50);")
    cursor.execute("ALTER TABLE donnees_satellite ALTER COLUMN id SET DATA TYPE BIGINT;")
    cursor.execute("ALTER TABLE donnees_satellite ALTER COLUMN id SET NOT NULL;")
    cursor.execute("ALTER TABLE donnees_satellite ALTER COLUMN timestamp SET NOT NULL;")
    cursor.execute("ALTER TABLE donnees_satellite DROP CONSTRAINT IF EXISTS donnees_satellite_pkey;")
    cursor.execute("ALTER TABLE donnees_satellite ADD PRIMARY KEY (timestamp, id);")

    cursor.execute("""
        SELECT create_hypertable('donnees_satellite', 'timestamp', if_not_exists => TRUE);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_satellite_id ON donnees_satellite(satellite_id);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_satellite_timestamp ON donnees_satellite(timestamp DESC);
    """)
    conn.commit()
    cursor.close()
    print('✅ Table donnees_satellite opérationnelle')
    return conn

def fetch_sentinel_observation(zone, config):
    """Interroge SentinelHub pour une zone donnée."""
    bbox = BBox(bbox=zone['bbox'], crs=CRS.WGS84)
    width, height = bbox_to_dimensions(bbox, resolution=20)
    request = SentinelHubRequest(
        evalscript=EVAL_SCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(datetime.utcnow() - timedelta(days=2), datetime.utcnow()),
                mosaicking_order='mostRecent',
            )
        ],
        responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
        bbox=bbox,
        size=(width, height),
        config=config,
    )
    try:
        data = request.get_data()
        tile = data[0]
        ndwi = float(np.nanmean(tile[:, :, 0]))
        turbidity = float(np.nanmean(tile[:, :, 1]) * 10.0)
        chlorophyll = float(np.nanmean(tile[:, :, 2]))
        return {
            'satellite_id': zone['id'],
            'zone': zone['name'],
            'timestamp': datetime.utcnow(),
            'latitude': zone['centroid'][0],
            'longitude': zone['centroid'][1],
            'ndwi': round(ndwi, 3),
            'turbidite': round(max(turbidity, 0), 2),
            'chlorophylle': round(max(chlorophyll, 0), 2),
            'tile': tile.astype('float32'),
            'bbox': bbox,
            'source': 'Sentinel-2',
        }
    except Exception as exc:
        print(f'⚠️  Erreur SentinelHub pour {zone["name"]}: {exc}')
        if not SIMULATION_FALLBACK:
            return None
        return generate_simulated_observation(zone, source='Sentinel-Fallback')

def generate_simulated_observation(zone, source='Simulation'):
    """Crée un raster synthétique quand SentinelHub n'est pas disponible."""
    tile = np.zeros((128, 128, 3), dtype='float32')
    ndwi_base = random_float(-0.1, 0.6)
    turb_base = random_float(0.2, 5.0)
    chloro_base = random_float(1.0, 20.0)
    tile[:, :, 0] = ndwi_base + np.random.normal(0, 0.02, (128, 128))
    tile[:, :, 1] = turb_base + np.random.normal(0, 0.1, (128, 128))
    tile[:, :, 2] = chloro_base + np.random.normal(0, 0.5, (128, 128))
    bbox = BBox(bbox=zone['bbox'], crs=CRS.WGS84)
    return {
        'satellite_id': zone['id'],
        'zone': zone['name'],
        'timestamp': datetime.utcnow(),
        'latitude': zone['centroid'][0],
        'longitude': zone['centroid'][1],
        'ndwi': round(float(np.nanmean(tile[:, :, 0])), 3),
        'turbidite': round(float(np.nanmean(tile[:, :, 1])), 2),
        'chlorophylle': round(float(np.nanmean(tile[:, :, 2])), 2),
        'tile': tile,
        'bbox': bbox,
        'source': source,
    }

def create_geotiff(observation):
    """Écrit un GeoTIFF temporaire à partir de la tuile."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
    tile = observation['tile']
    height, width, _ = tile.shape
    transform = from_bounds(
        observation['bbox'].min_x,
        observation['bbox'].min_y,
        observation['bbox'].max_x,
        observation['bbox'].max_y,
        width,
        height,
    )
    with rasterio.open(
        temp_file.name,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(tile[:, :, 0], 1)
        dst.write(tile[:, :, 1], 2)
        dst.write(tile[:, :, 2], 3)
    return temp_file.name

def upload_to_minio(minio_client, file_path, zone, timestamp):
    """Uploade le GeoTIFF dans MinIO et retourne l'URL publique."""
    safe_zone = zone.replace(' ', '_').replace('/', '-').lower()
    object_name = f"{safe_zone}/{timestamp.strftime('%Y%m%dT%H%M%S')}.tif"
    minio_client.fput_object(MINIO_BUCKET, object_name, file_path, content_type='image/tiff')
    public_url = f"{MINIO_PUBLIC_ENDPOINT.rstrip('/')}/{MINIO_BUCKET}/{object_name}"
    return public_url

def insert_satellite_data(conn, observation, storage_url):
    """Insère l'observation dans TimescaleDB."""
    cursor = conn.cursor()
    query = """
        INSERT INTO donnees_satellite (
            timestamp, satellite_id, zone, latitude, longitude,
            chlorophylle, turbidite, ndwi, storage_url, source
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(
        query,
        (
            observation['timestamp'],
            observation['satellite_id'],
            observation['zone'],
            observation['latitude'],
            observation['longitude'],
            observation['chlorophylle'],
            observation['turbidite'],
            observation['ndwi'],
            storage_url,
            observation['source'],
        ),
    )
    conn.commit()
    cursor.close()
    print(
        f"🛰️  {observation['zone']} | NDWI: {observation['ndwi']} | "
        f"Turbidité: {observation['turbidite']} NTU | Chlorophylle: {observation['chlorophylle']} mg/m³"
    )

def collect_observations(config, sentinel_enabled):
    """Collecte les observations Sentinel ou simulateur."""
    observations = []
    for zone in AOIS:
        if sentinel_enabled:
            obs = fetch_sentinel_observation(zone, config)
        else:
            obs = generate_simulated_observation(zone)
        if obs:
            observations.append(obs)
    return observations

def cleanup_temp_file(path):
    try:
        os.unlink(path)
    except OSError:
        pass

def main():
    print('🚀 Démarrage du service satellite...')
    if not wait_for_db():
        print('❌ Impossible de se connecter à la base de données')
        return

    sentinel_config, sentinel_enabled = configure_sentinel()
    conn = init_database()
    minio_client = init_minio()

    print(f'✅ Service satellite prêt - Intervalle de collecte: {FETCH_INTERVAL}s')
    
    # Register with Consul for service discovery
    try:
        import sys
        sys.path.insert(0, '/app/shared')
        from service_discovery import register_service, wait_for_consul
        if wait_for_consul(10, 2):
            register_service('satellite', 0)
    except Exception as consul_error:
        print(f'⚠️ Consul non disponible: {consul_error}')
    
    try:
        while True:
            observations = collect_observations(sentinel_config, sentinel_enabled)
            for obs in observations:
                geotiff_path = create_geotiff(obs)
                storage_url = upload_to_minio(minio_client, geotiff_path, obs['zone'], obs['timestamp'])
                cleanup_temp_file(geotiff_path)
                insert_satellite_data(conn, obs, storage_url)
            time.sleep(FETCH_INTERVAL)
    except KeyboardInterrupt:
        print('\n🛑 Arrêt du service satellite...')
    except Exception as exc:
        print(f'❌ Erreur fatale satellite: {exc}')
    finally:
        if conn:
            conn.close()
        print('✅ Service satellite arrêté')

if __name__ == '__main__':
    main()

