const { Client } = require('pg');
const mqtt = require('mqtt');

// Configuration de la base de données
const dbConfig = {
  host: process.env.TIMESCALEDB_HOST || 'timescaledb',
  port: parseInt(process.env.TIMESCALEDB_PORT || '5432'),
  database: process.env.TIMESCALEDB_DB || 'aquawatch',
  user: process.env.TIMESCALEDB_USER || 'postgres',
  password: process.env.TIMESCALEDB_PASSWORD || 'postgres',
};

// Configuration MQTT
const mqttConfig = {
  host: process.env.MQTT_BROKER || 'mqtt-broker',
  port: parseInt(process.env.MQTT_PORT || '1883'),
};

let dbClient;
let mqttClient;

// Fonction pour générer une valeur aléatoire dans une plage
function randomFloat(min, max) {
  return Math.random() * (max - min) + min;
}

// Coordonnées simulées pour différents capteurs dans différentes zones (Région Rabat-Salé)
const capteurLocations = {
  // Zone 1: Centre-ville Rabat
  'CAPT-1': { lat: 34.0209, lon: -6.8416, zone: 'Rabat-Centre' },
  'CAPT-2': { lat: 34.0133, lon: -6.8326, zone: 'Rabat-Centre' },

  // Zone 2: Salé
  'CAPT-3': { lat: 34.0286, lon: -6.8500, zone: 'Salé-Nord' },
  'CAPT-4': { lat: 34.0150, lon: -6.8450, zone: 'Salé-Sud' },

  // Zone 3: Hay Riad
  'CAPT-5': { lat: 34.0250, lon: -6.8350, zone: 'Hay-Riad' },
  'CAPT-6': { lat: 34.0300, lon: -6.8400, zone: 'Hay-Riad' },

  // Zone 4: Agdal
  'CAPT-7': { lat: 34.0100, lon: -6.8500, zone: 'Agdal' },
  'CAPT-8': { lat: 34.0120, lon: -6.8520, zone: 'Agdal' },

  // Zone 5: Océan (côte)
  'CAPT-9': { lat: 34.0400, lon: -6.8200, zone: 'Côte-Océan' },
  'CAPT-10': { lat: 34.0350, lon: -6.8250, zone: 'Côte-Océan' },

  // Zone 6: Bouregreg (rivière)
  'CAPT-11': { lat: 34.0180, lon: -6.8380, zone: 'Bouregreg' },
  'CAPT-12': { lat: 34.0200, lon: -6.8400, zone: 'Bouregreg' },

  // Zone 7: Temara
  'CAPT-13': { lat: 33.9200, lon: -6.9100, zone: 'Temara' },
  'CAPT-14': { lat: 33.9250, lon: -6.9150, zone: 'Temara' },

  // Zone 8: Skhirat
  'CAPT-15': { lat: 33.8500, lon: -7.0300, zone: 'Skhirat' },

  // Zone 9: Marrakech (Zone critique de démonstration - 60% critique, 10% warning, 30% bonne)
  'CAPT-16': { lat: 31.6295, lon: -7.9811, zone: 'Marrakech' },
};

// Fonction pour générer des données de capteur avec variations selon la zone
function generateSensorData() {
  // Sélectionner un capteur aléatoire parmi tous les capteurs disponibles
  const capteurIds = Object.keys(capteurLocations);
  const capteurId = capteurIds[Math.floor(Math.random() * capteurIds.length)];
  const location = capteurLocations[capteurId];

  // Variations selon la zone pour rendre les données plus réalistes
  let phBase = 7.0;
  let turbiditeBase = 1.0;
  let tempBase = 20.0;

  // Ajuster selon la zone
  if (location.zone.includes('Océan') || location.zone.includes('Côte')) {
    phBase = 8.0; // pH plus élevé pour l'eau de mer
    turbiditeBase = 0.5;
    tempBase = 18.0;
  } else if (location.zone.includes('Bouregreg')) {
    phBase = 7.2;
    turbiditeBase = 2.0; // Plus de turbidité dans la rivière
    tempBase = 22.0;
  } else if (location.zone.includes('Temara') || location.zone.includes('Skhirat')) {
    phBase = 7.5;
    turbiditeBase = 1.5;
    tempBase = 21.0;
  }

  // ========= MARRAKECH: Distribution spéciale 60% critique, 10% warning, 30% bonne =========
  if (location.zone === 'Marrakech') {
    const roll = Math.random();
    let ph, turbidite, temperature;

    if (roll < 0.60) {
      // 60% CRITIQUE: valeurs très hors normes OMS
      ph = Math.random() < 0.5 ? randomFloat(4.8, 5.8) : randomFloat(9.2, 10.2);
      turbidite = randomFloat(6.0, 9.5);
      temperature = randomFloat(32, 38);
    } else if (roll < 0.70) {
      // 10% WARNING: valeurs limites OMS
      ph = Math.random() < 0.5 ? randomFloat(6.0, 6.4) : randomFloat(8.6, 9.0);
      turbidite = randomFloat(2.5, 5.0);
      temperature = randomFloat(26, 31);
    } else {
      // 30% BONNE: valeurs normales
      ph = randomFloat(6.8, 7.5);
      turbidite = randomFloat(0.3, 1.0);
      temperature = randomFloat(18, 24);
    }

    return {
      timestamp: new Date().toISOString(),
      ph: parseFloat(Math.max(4.5, Math.min(10.5, ph)).toFixed(2)),
      turbidite: parseFloat(Math.max(0.1, Math.min(10.0, turbidite)).toFixed(2)),
      temperature: parseFloat(Math.max(10.0, Math.min(40.0, temperature)).toFixed(2)),
      capteur_id: capteurId,
      latitude: location.lat,
      longitude: location.lon,
      zone: location.zone,
    };
  }
  // ========================================================================================

  // Ajouter des variations aléatoires avec parfois des valeurs hors normes OMS (10% de chance)
  // Cela permet de générer des alertes réelles
  const hasAnomaly = Math.random() < 0.10; // 10% de chance d'anomalie

  let ph, turbidite, temperature;

  if (hasAnomaly) {
    // Générer des valeurs hors normes OMS pour déclencher des alertes
    const anomalyType = Math.random();
    if (anomalyType < 0.33) {
      // pH hors norme
      ph = parseFloat((phBase + (Math.random() < 0.5 ? randomFloat(-2.0, -0.5) : randomFloat(1.5, 3.0))).toFixed(2));
      turbidite = parseFloat((turbiditeBase + randomFloat(-0.3, 0.5)).toFixed(2));
      temperature = parseFloat((tempBase + randomFloat(-2.0, 3.0)).toFixed(2));
    } else if (anomalyType < 0.66) {
      // Turbidité élevée
      ph = parseFloat((phBase + randomFloat(-0.5, 0.5)).toFixed(2));
      turbidite = parseFloat((turbiditeBase + randomFloat(1.5, 6.0)).toFixed(2)); // Dépassement du seuil
      temperature = parseFloat((tempBase + randomFloat(-2.0, 3.0)).toFixed(2));
    } else {
      // Température élevée
      ph = parseFloat((phBase + randomFloat(-0.5, 0.5)).toFixed(2));
      turbidite = parseFloat((turbiditeBase + randomFloat(-0.3, 1.0)).toFixed(2));
      temperature = parseFloat((tempBase + randomFloat(6.0, 12.0)).toFixed(2)); // Température élevée
    }
  } else {
    // Valeurs normales avec variations
    ph = parseFloat((phBase + randomFloat(-0.8, 0.8)).toFixed(2));
    turbidite = parseFloat((turbiditeBase + randomFloat(-0.3, 1.0)).toFixed(2));
    temperature = parseFloat((tempBase + randomFloat(-3.0, 5.0)).toFixed(2));
  }

  return {
    timestamp: new Date().toISOString(),
    ph: Math.max(5.5, Math.min(9.5, ph)), // Permettre des valeurs hors normes
    turbidite: Math.max(0.1, Math.min(8.0, turbidite)), // Permettre des valeurs élevées
    temperature: Math.max(12.0, Math.min(35.0, temperature)), // Permettre des températures élevées
    capteur_id: capteurId,
    latitude: location.lat,
    longitude: location.lon,
    zone: location.zone,
  };
}

// Fonction pour initialiser la base de données
async function initDatabase() {
  try {
    dbClient = new Client(dbConfig);
    await dbClient.connect();
    console.log('✅ Connecté à TimescaleDB');

    // Créer la table si elle n'existe pas avec géolocalisation (sans PostGIS)
    await dbClient.query(`
      CREATE TABLE IF NOT EXISTS donnees_capteurs (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        capteur_id VARCHAR(50) NOT NULL,
        zone VARCHAR(50),
        ph DECIMAL(5,2),
        turbidite DECIMAL(5,2),
        temperature DECIMAL(5,2),
        latitude DECIMAL(10,6),
        longitude DECIMAL(10,6)
      );
    `);

    // Ajouter la colonne zone si elle n'existe pas (pour les tables existantes)
    try {
      const checkColumn = await dbClient.query(`
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='donnees_capteurs' AND column_name='zone'
      `);
      if (checkColumn.rows.length === 0) {
        await dbClient.query(`ALTER TABLE donnees_capteurs ADD COLUMN zone VARCHAR(50);`);
        console.log('✅ Colonne zone ajoutée');
      }
    } catch (error) {
      console.log('Colonne zone déjà présente ou erreur:', error.message);
    }

    // Créer l'hypertable TimescaleDB si elle n'existe pas
    try {
      await dbClient.query(`
        SELECT create_hypertable('donnees_capteurs', 'timestamp', if_not_exists => TRUE);
      `);
    } catch (error) {
      // L'hypertable existe peut-être déjà
      console.log('Hypertable déjà créée ou en cours');
    }

    // Créer un index sur capteur_id (après la création de l'hypertable)
    try {
      await dbClient.query(`
        CREATE INDEX IF NOT EXISTS idx_capteur_id ON donnees_capteurs(capteur_id);
      `);
    } catch (error) {
      // Ignorer l'erreur si l'index existe déjà
      console.log('Index déjà créé ou en cours de création');
    }

    console.log('✅ Table et hypertable créées/initialisées');
  } catch (error) {
    console.error('❌ Erreur lors de l\'initialisation de la base de données:', error.message);
    throw error;
  }
}

// Fonction pour insérer des données dans la base
async function insertSensorData(data) {
  try {
    const query = `
      INSERT INTO donnees_capteurs (timestamp, capteur_id, zone, ph, turbidite, temperature, latitude, longitude)
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    `;
    await dbClient.query(query, [
      data.timestamp,
      data.capteur_id,
      data.zone || 'Non-définie',
      data.ph,
      data.turbidite,
      data.temperature,
      data.latitude,
      data.longitude,
    ]);
    console.log(`📊 Données insérées: ${data.capteur_id} (${data.zone}) - pH: ${data.ph}, Turbidité: ${data.turbidite}, Temp: ${data.temperature}°C`);
  } catch (error) {
    console.error('❌ Erreur lors de l\'insertion des données:', error.message);
  }
}

// Fonction pour publier sur MQTT
function publishToMQTT(data) {
  if (mqttClient && mqttClient.connected) {
    const topic = `aquawatch/capteurs/${data.capteur_id}`;
    const message = JSON.stringify(data);
    mqttClient.publish(topic, message, { qos: 1 });
    console.log(`📡 Publié sur MQTT: ${topic}`);
  }
}

// Fonction pour connecter MQTT
function connectMQTT() {
  try {
    mqttClient = mqtt.connect(`mqtt://${mqttConfig.host}:${mqttConfig.port}`, {
      reconnectPeriod: 5000,
    });

    mqttClient.on('connect', () => {
      console.log('✅ Connecté au broker MQTT');
    });

    mqttClient.on('error', (error) => {
      console.error('❌ Erreur MQTT:', error.message);
    });

    mqttClient.on('offline', () => {
      console.log('⚠️  MQTT déconnecté');
    });
  } catch (error) {
    console.error('❌ Erreur lors de la connexion MQTT:', error.message);
    // Continuer sans MQTT si le broker n'est pas disponible
  }
}

async function main() {
  console.log('🚀 Démarrage du service capteurs...');

  // Initialiser la base de données
  await initDatabase();
  console.log('🔄 Database initialized, connecting MQTT...');

  // Connecter MQTT (optionnel, continue même si MQTT n'est pas disponible)
  connectMQTT();
  console.log('🔄 MQTT connect initiated, setting up interval...');

  // Get interval from environment variable or default to 5000ms (5 seconds)
  const CAPTEURS_INTERVAL_SECONDS = parseInt(process.env.CAPTEURS_INTERVAL_SECONDS || '5');
  const intervalMs = CAPTEURS_INTERVAL_SECONDS * 1000;

  // Générer et envoyer des données selon l'intervalle configuré
  setInterval(async () => {
    const data = generateSensorData();
    await insertSensorData(data);
    publishToMQTT(data);
  }, intervalMs);

  console.log(`✅ Service capteurs démarré - ${Object.keys(capteurLocations).length} capteurs dans ${new Set(Object.values(capteurLocations).map(l => l.zone)).size} zones`);
  console.log(`✅ Génération de données toutes les ${CAPTEURS_INTERVAL_SECONDS} secondes`);

  // Register with Consul for service discovery
  console.log('🔄 Attempting Consul registration...');
  try {
    const { registerService, waitForConsul } = require('./shared/service-discovery');
    console.log('🔄 Service discovery module loaded');
    if (await waitForConsul(10, 2000)) {
      console.log('🔄 Consul is available, registering...');
      await registerService('capteurs', 0);
    } else {
      console.log('⚠️ Consul not available after waiting');
    }
  } catch (consulError) {
    console.log('⚠️ Consul registration error:', consulError.message);
  }
}

// Gestion de l'arrêt propre
process.on('SIGTERM', async () => {
  console.log('🛑 Arrêt du service capteurs...');
  if (dbClient) await dbClient.end();
  if (mqttClient) mqttClient.end();
  process.exit(0);
});

process.on('SIGINT', async () => {
  console.log('🛑 Arrêt du service capteurs...');
  if (dbClient) await dbClient.end();
  if (mqttClient) mqttClient.end();
  process.exit(0);
});

// Démarrer le service
main().catch((error) => {
  console.error('❌ Erreur fatale:', error);
  process.exit(1);
});

