#!/usr/bin/env python3
"""
Script pour générer des données historiques réalistes pour tous les capteurs.
- 15 jours d'historique
- 1 mesure par heure par capteur
- Variations jour/nuit réalistes
- Marrakech (CAPT-16): 60% critique, 30% bonne, 10% warning
"""

import os
import random
from datetime import datetime, timedelta
import psycopg2
import math

DB_CONFIG = {
    'host': os.getenv('TIMESCALEDB_HOST', 'localhost'),
    'port': int(os.getenv('TIMESCALEDB_PORT', '5433')),
    'database': os.getenv('TIMESCALEDB_DB', 'aquawatch'),
    'user': os.getenv('TIMESCALEDB_USER', 'postgres'),
    'password': os.getenv('TIMESCALEDB_PASSWORD', 'postgres'),
}

# Configuration des capteurs
CAPTEURS = {
    'CAPT-1': {'zone': 'Rabat-Centre', 'lat': 34.0209, 'lon': -6.8416},
    'CAPT-2': {'zone': 'Rabat-Centre', 'lat': 34.0215, 'lon': -6.8420},
    'CAPT-3': {'zone': 'Salé-Nord', 'lat': 34.0286, 'lon': -6.8500},
    'CAPT-4': {'zone': 'Salé-Sud', 'lat': 34.0150, 'lon': -6.8450},
    'CAPT-5': {'zone': 'Hay-Riad', 'lat': 34.0250, 'lon': -6.8350},
    'CAPT-6': {'zone': 'Hay-Riad', 'lat': 34.0255, 'lon': -6.8360},
    'CAPT-7': {'zone': 'Agdal', 'lat': 34.0100, 'lon': -6.8500},
    'CAPT-8': {'zone': 'Agdal', 'lat': 34.0105, 'lon': -6.8510},
    'CAPT-9': {'zone': 'Côte-Océan', 'lat': 34.0350, 'lon': -6.8250},
    'CAPT-10': {'zone': 'Côte-Océan', 'lat': 34.0355, 'lon': -6.8260},
    'CAPT-11': {'zone': 'Bouregreg', 'lat': 34.0180, 'lon': -6.8380},
    'CAPT-12': {'zone': 'Bouregreg', 'lat': 34.0185, 'lon': -6.8390},
    'CAPT-13': {'zone': 'Temara', 'lat': 33.9200, 'lon': -6.9100},
    'CAPT-14': {'zone': 'Temara', 'lat': 33.9205, 'lon': -6.9110},
    'CAPT-15': {'zone': 'Skhirat', 'lat': 33.8500, 'lon': -7.0300},
    'CAPT-16': {'zone': 'Marrakech', 'lat': 31.6295, 'lon': -7.9811},
}

# Période de génération
DAYS = 15
HOURS_PER_DAY = 24

def random_float(min_val, max_val):
    return round(random.uniform(min_val, max_val), 2)

def get_hour_factor(hour):
    """Retourne un facteur basé sur l'heure (cycle jour/nuit)."""
    # Température plus haute vers midi, plus basse la nuit
    return math.sin((hour - 6) * math.pi / 12)

def generate_normal_data(hour):
    """Génère des données pour zones normales: 80% bonne, 15% warning, 5% critique."""
    roll = random.random()
    hour_factor = get_hour_factor(hour)
    
    if roll < 0.05:
        # 5% CRITIQUE
        ph = random.choice([random_float(5.0, 6.0), random_float(9.0, 9.5)])
        turbidite = random_float(5.5, 8.0)
        temperature = random_float(30, 35) + hour_factor * 2
    elif roll < 0.20:
        # 15% WARNING
        ph = random.choice([random_float(6.2, 6.5), random_float(8.5, 8.8)])
        turbidite = random_float(1.5, 4.5)
        temperature = random_float(25, 29) + hour_factor * 2
    else:
        # 80% BONNE
        ph = random_float(6.8, 7.8)
        turbidite = random_float(0.2, 1.2)
        temperature = random_float(18, 24) + hour_factor * 3
    
    return {
        'ph': max(4.5, min(10.5, ph)),
        'turbidite': max(0.1, min(10.0, turbidite)),
        'temperature': max(10.0, min(40.0, temperature)),
    }

def generate_marrakech_data(hour):
    """Génère des données pour Marrakech: 60% critique, 30% bonne, 10% warning."""
    roll = random.random()
    hour_factor = get_hour_factor(hour)
    
    if roll < 0.60:
        # 60% CRITIQUE
        ph = random.choice([random_float(4.8, 5.8), random_float(9.2, 10.2)])
        turbidite = random_float(6.0, 9.5)
        temperature = random_float(32, 38) + hour_factor * 2
    elif roll < 0.70:
        # 10% WARNING
        ph = random.choice([random_float(6.0, 6.4), random_float(8.6, 9.0)])
        turbidite = random_float(2.5, 5.0)
        temperature = random_float(26, 31) + hour_factor * 2
    else:
        # 30% BONNE
        ph = random_float(6.8, 7.5)
        turbidite = random_float(0.3, 1.0)
        temperature = random_float(18, 24) + hour_factor * 3
    
    return {
        'ph': max(4.5, min(10.5, ph)),
        'turbidite': max(0.1, min(10.0, turbidite)),
        'temperature': max(10.0, min(40.0, temperature)),
    }

def main():
    print("=" * 60)
    print("🌊 Génération de données historiques réalistes")
    print(f"   Période: {DAYS} jours × {HOURS_PER_DAY} heures")
    print(f"   Capteurs: {len(CAPTEURS)}")
    print(f"   Total estimé: {DAYS * HOURS_PER_DAY * len(CAPTEURS)} mesures")
    print("=" * 60)
    
    # Connexion à la base
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✅ Connecté à TimescaleDB")
    except Exception as e:
        print(f"❌ Erreur connexion: {e}")
        return
    
    # Date de début (il y a 15 jours)
    start_date = datetime.now() - timedelta(days=DAYS)
    
    inserted = 0
    total = DAYS * HOURS_PER_DAY * len(CAPTEURS)
    
    print(f"\n⏳ Génération en cours...")
    
    for day in range(DAYS):
        current_date = start_date + timedelta(days=day)
        
        for hour in range(HOURS_PER_DAY):
            timestamp = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            for capteur_id, info in CAPTEURS.items():
                # Générer données selon le type de zone
                if capteur_id == 'CAPT-16':
                    data = generate_marrakech_data(hour)
                else:
                    data = generate_normal_data(hour)
                
                # Insérer dans la base
                cursor.execute("""
                    INSERT INTO donnees_capteurs 
                    (timestamp, capteur_id, zone, ph, turbidite, temperature, latitude, longitude)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    timestamp, capteur_id, info['zone'],
                    data['ph'], data['turbidite'], data['temperature'],
                    info['lat'], info['lon']
                ))
                inserted += 1
        
        # Commit par jour
        conn.commit()
        progress = (day + 1) / DAYS * 100
        print(f"   Jour {day + 1}/{DAYS} ({progress:.0f}%) - {inserted}/{total} mesures")
    
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print(f"✅ {inserted} mesures générées avec succès!")
    print("   Distribution:")
    print("   • CAPT-1 à CAPT-15: 80% bonne, 15% warning, 5% critique")
    print("   • CAPT-16 (Marrakech): 60% critique, 30% bonne, 10% warning")
    print("=" * 60)

if __name__ == '__main__':
    main()
