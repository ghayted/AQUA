const { Client } = require('pg');
const nodemailer = require('nodemailer');

// Helpers
const toNumber = (value) => {
  if (value === null || value === undefined) return null;
  if (typeof value === 'number') return value;
  const parsed = parseFloat(value);
  return Number.isNaN(parsed) ? null : parsed;
};

// Configuration de la base de données alertes
const dbConfig = {
  host: process.env.POSTGRES_HOST || 'postgres',
  port: parseInt(process.env.POSTGRES_PORT || '5432'),
  database: process.env.POSTGRES_DB || 'alertes',
  user: process.env.POSTGRES_USER || 'postgres',
  password: process.env.POSTGRES_PASSWORD || 'postgres',
};

// Configuration TimescaleDB pour lire les données
const timescaleConfig = {
  host: process.env.TIMESCALEDB_HOST || 'timescaledb',
  port: parseInt(process.env.TIMESCALEDB_PORT || '5432'),
  database: process.env.TIMESCALEDB_DB || 'aquawatch',
  user: process.env.TIMESCALEDB_USER || 'postgres',
  password: process.env.TIMESCALEDB_PASSWORD || 'postgres',
};

let dbClient;
let timescaleClient;

// Seuils OMS pour la qualité de l'eau potable
const OMS_SEUILS = {
  ph: { min: 6.5, max: 8.5, critical: { min: 6.0, max: 9.0 }, polluant: 'Acidité/Alcalinité' },
  turbidite: { max: 1.0, critical: 5.0, polluant: 'Matières en suspension' }, // NTU
  temperature: { max: 25.0, critical: 30.0, polluant: 'Température' }, // °C
  chlorophylle: { max: 10.0, critical: 20.0, polluant: 'Eutrophisation' }, // mg/m³
};

// Estimation de population par zone (simulation)
const POPULATION_ZONES = {
  'Rabat-Centre': 50000,
  'Salé-Nord': 30000,
  'Salé-Sud': 35000,
  'Hay-Riad': 25000,
  'Agdal': 40000,
  'Côte-Océan': 15000,
  'Bouregreg': 20000,
  'Temara': 45000,
  'Skhirat': 30000,
};

const SATELLITE_SEUILS = {
  chlorophylle: { warning: 10, critical: 20, polluant: 'Eutrophisation' },
  turbidite: { warning: 4, critical: 6, polluant: 'Turbidité côtière' },
  ndwi: { min: 0.05, polluant: 'Stress hydrique' },
};

// Configuration du transporteur email (simulation)
const emailTransporter = nodemailer.createTransport({
  host: 'smtp.example.com', // Serveur SMTP simulé
  port: 587,
  secure: false,
  auth: {
    user: 'alertes@aquawatch.com',
    pass: 'password',
  },
  // En mode développement, on ne va pas vraiment envoyer d'emails
  // mais on simule l'envoi
});

// Fonction pour vérifier les seuils OMS et générer des alertes
async function checkOMSThresholds() {
  const alertes = [];

  // Récupérer les dernières données des capteurs avec zone géographique
  try {
    const capteursQuery = `
      SELECT capteur_id, zone, ph, turbidite, temperature, timestamp, latitude, longitude
      FROM donnees_capteurs
      WHERE timestamp > NOW() - INTERVAL '2 minutes'
      ORDER BY timestamp DESC
      LIMIT 20
    `;
    const capteursResult = await timescaleClient.query(capteursQuery);

    // Éviter les doublons d'alertes récentes (dans les 5 dernières minutes)
    let recentAlertsMap = new Map();
    try {
      const recentAlertsQuery = `
        SELECT capteur_id, parametre, severity
        FROM alertes
        WHERE timestamp > NOW() - INTERVAL '5 minutes'
        AND status = 'ACTIVE'
        AND capteur_id IS NOT NULL
      `;
      const recentAlerts = await dbClient.query(recentAlertsQuery);
      recentAlerts.rows.forEach(alert => {
        if (alert.capteur_id && alert.parametre) {
          const key = `${alert.capteur_id}_${alert.parametre}_${alert.severity}`;
          recentAlertsMap.set(key, true);
        }
      });
    } catch (error) {
      // Ignorer l'erreur si la table n'existe pas encore
      // Ne pas logger pour éviter le spam
    }

    for (const row of capteursResult.rows) {
      const phValue = toNumber(row.ph);
      const turbValue = toNumber(row.turbidite);
      const tempValue = toNumber(row.temperature);
      if (phValue === null && turbValue === null && tempValue === null) {
        continue;
      }

      const zoneGeo = row.zone || 'Zone inconnue';
      const population = POPULATION_ZONES[zoneGeo] || 10000;

      // Vérifier le pH
      if (phValue !== null && (phValue < OMS_SEUILS.ph.critical.min || phValue > OMS_SEUILS.ph.critical.max)) {
        const alertKey = `${row.capteur_id}_ph_CRITICAL`;
        if (!recentAlertsMap.has(alertKey)) {
          alertes.push({
            type: 'SEUIL_CRITIQUE_DEPASSE',
            severity: 'CRITICAL',
            zone: zoneGeo,
            zone_geographique: zoneGeo,
            capteur_id: row.capteur_id,
            message: `🚨 ALERTE CRITIQUE - pH dangereux détecté dans ${zoneGeo}: ${phValue.toFixed(2)} (seuil OMS: 6.5-8.5). Population exposée estimée: ${population.toLocaleString()} habitants.`,
            valeur: phValue,
            parametre: 'ph',
            seuil_oms: phValue < 6.0 ? 6.0 : 9.0,
            type_polluant: OMS_SEUILS.ph.polluant,
            population_exposee: population,
          });
        }
      } else if (phValue !== null && (phValue < OMS_SEUILS.ph.min || phValue > OMS_SEUILS.ph.max)) {
        const alertKey = `${row.capteur_id}_ph_WARNING`;
        if (!recentAlertsMap.has(alertKey)) {
          alertes.push({
            type: 'SEUIL_DEPASSE',
            severity: 'WARNING',
            zone: zoneGeo,
            zone_geographique: zoneGeo,
            capteur_id: row.capteur_id,
            message: `⚠️ pH hors norme dans ${zoneGeo}: ${phValue.toFixed(2)} (seuil OMS: 6.5-8.5). Surveillance renforcée recommandée.`,
            valeur: phValue,
            parametre: 'ph',
            seuil_oms: phValue < 6.5 ? 6.5 : 8.5,
            type_polluant: OMS_SEUILS.ph.polluant,
            population_exposee: Math.floor(population * 0.3), // 30% de la zone
          });
        }
      }

      // Vérifier la turbidité
      if (turbValue !== null && turbValue > OMS_SEUILS.turbidite.critical) {
        const alertKey = `${row.capteur_id}_turbidite_CRITICAL`;
        if (!recentAlertsMap.has(alertKey)) {
          alertes.push({
            type: 'SEUIL_CRITIQUE_DEPASSE',
            severity: 'CRITICAL',
            zone: zoneGeo,
            zone_geographique: zoneGeo,
            capteur_id: row.capteur_id,
            message: `🚨 ALERTE CRITIQUE - Turbidité excessive dans ${zoneGeo}: ${turbValue.toFixed(2)} NTU (seuil OMS: 1.0 NTU). Risque de contamination microbiologique. Population exposée: ${population.toLocaleString()} habitants.`,
            valeur: turbValue,
            parametre: 'turbidite',
            seuil_oms: 1.0,
            type_polluant: OMS_SEUILS.turbidite.polluant,
            population_exposee: population,
          });
        }
      } else if (turbValue !== null && turbValue > OMS_SEUILS.turbidite.max) {
        const alertKey = `${row.capteur_id}_turbidite_WARNING`;
        if (!recentAlertsMap.has(alertKey)) {
          alertes.push({
            type: 'QUALITE_FAIBLE',
            severity: 'WARNING',
            zone: zoneGeo,
            zone_geographique: zoneGeo,
            capteur_id: row.capteur_id,
            message: `⚠️ Turbidité élevée dans ${zoneGeo}: ${turbValue.toFixed(2)} NTU (seuil OMS: 1.0 NTU). Traitement supplémentaire recommandé.`,
            valeur: turbValue,
            parametre: 'turbidite',
            seuil_oms: 1.0,
            type_polluant: OMS_SEUILS.turbidite.polluant,
            population_exposee: Math.floor(population * 0.2),
          });
        }
      }

      // Vérifier la température
      if (tempValue !== null && tempValue > OMS_SEUILS.temperature.critical) {
        const alertKey = `${row.capteur_id}_temperature_WARNING`;
        if (!recentAlertsMap.has(alertKey)) {
          alertes.push({
            type: 'RISQUE_ELEVE',
            severity: 'WARNING',
            zone: zoneGeo,
            zone_geographique: zoneGeo,
            capteur_id: row.capteur_id,
            message: `⚠️ Température élevée dans ${zoneGeo}: ${tempValue.toFixed(2)}°C (seuil recommandé: 25°C). Risque de développement bactérien.`,
            valeur: tempValue,
            parametre: 'temperature',
            seuil_oms: 25.0,
            type_polluant: OMS_SEUILS.temperature.polluant,
            population_exposee: Math.floor(population * 0.1),
          });
        }
      }
    }

  } catch (error) {
    // Erreur sur les capteurs, continuer quand même
    console.log('⚠️ Erreur lors de la vérification des capteurs:', error.message);
    // Retourner les alertes déjà collectées même en cas d'erreur
    return alertes;
  }

  // Vérification des données satellite
  try {
    const satelliteQuery = `
      SELECT satellite_id, zone, chlorophylle, turbidite, ndwi, timestamp
      FROM donnees_satellite
      WHERE timestamp > NOW() - INTERVAL '15 minutes'
      ORDER BY timestamp DESC
      LIMIT 20
    `;
    const satelliteResult = await timescaleClient.query(satelliteQuery);
    const recentAlertsMapSat = new Map();
    try {
      const recentSatelliteAlerts = await dbClient.query(`
        SELECT capteur_id, parametre, severity
        FROM alertes
        WHERE timestamp > NOW() - INTERVAL '10 minutes'
          AND status = 'ACTIVE'
          AND capteur_id LIKE 'SAT-%'
      `);
      recentSatelliteAlerts.rows.forEach(alert => {
        const key = `${alert.capteur_id}_${alert.parametre}_${alert.severity}`;
        recentAlertsMapSat.set(key, true);
      });
    } catch (error) {
      // table alertes peut être vide au démarrage
    }

    for (const row of satelliteResult.rows) {
      const zoneSatellite = row.zone || 'Observation satellite';
      const population = POPULATION_ZONES[zoneSatellite] || 80000;
      const chlorValue = toNumber(row.chlorophylle);
      const turbSatValue = toNumber(row.turbidite);
      const ndwiValue = toNumber(row.ndwi);

      if (chlorValue !== null && chlorValue > SATELLITE_SEUILS.chlorophylle.critical) {
        const alertKey = `${row.satellite_id}_chlorophylle_CRITICAL`;
        if (!recentAlertsMapSat.has(alertKey)) {
          alertes.push({
            type: 'EUTROPHISATION_CRITIQUE',
            severity: 'CRITICAL',
            zone: zoneSatellite,
            zone_geographique: zoneSatellite,
            capteur_id: row.satellite_id,
            parametre: 'chlorophylle',
            valeur: chlorValue,
            seuil_oms: SATELLITE_SEUILS.chlorophylle.critical,
            type_polluant: SATELLITE_SEUILS.chlorophylle.polluant,
            population_exposee: population,
            message: `🚨 Floraison algale détectée par Sentinel (${chlorValue.toFixed(2)} mg/m³) sur ${zoneSatellite}. Risque de désoxygénation imminent.`,
          });
        }
      } else if (chlorValue !== null && chlorValue > SATELLITE_SEUILS.chlorophylle.warning) {
        const alertKey = `${row.satellite_id}_chlorophylle_WARNING`;
        if (!recentAlertsMapSat.has(alertKey)) {
          alertes.push({
            type: 'EUTROPHISATION',
            severity: 'WARNING',
            zone: zoneSatellite,
            zone_geographique: zoneSatellite,
            capteur_id: row.satellite_id,
            parametre: 'chlorophylle',
            valeur: chlorValue,
            seuil_oms: SATELLITE_SEUILS.chlorophylle.warning,
            type_polluant: SATELLITE_SEUILS.chlorophylle.polluant,
            population_exposee: Math.floor(population * 0.5),
            message: `⚠️ Concentration de chlorophylle élevée (${chlorValue.toFixed(2)} mg/m³) - surveillance accrue des proliférations.`,
          });
        }
      }

      if (turbSatValue !== null && turbSatValue > SATELLITE_SEUILS.turbidite.critical) {
        const alertKey = `${row.satellite_id}_turbidite_CRITICAL`;
        if (!recentAlertsMapSat.has(alertKey)) {
          alertes.push({
            type: 'TURBIDITE_COTIERE_CRITIQUE',
            severity: 'CRITICAL',
            zone: zoneSatellite,
            zone_geographique: zoneSatellite,
            capteur_id: row.satellite_id,
            parametre: 'turbidite',
            valeur: turbSatValue,
            seuil_oms: SATELLITE_SEUILS.turbidite.critical,
            type_polluant: SATELLITE_SEUILS.turbidite.polluant,
            population_exposee: population,
            message: `🚨 Panache turbide détecté (${turbSatValue.toFixed(2)} NTU) sur ${zoneSatellite}. Vérifier rejets industriels/agricoles.`,
          });
        }
      }

      if (ndwiValue !== null && ndwiValue < SATELLITE_SEUILS.ndwi.min) {
        const alertKey = `${row.satellite_id}_ndwi_WARNING`;
        if (!recentAlertsMapSat.has(alertKey)) {
          alertes.push({
            type: 'STRESS_HYDRIQUE',
            severity: 'WARNING',
            zone: zoneSatellite,
            zone_geographique: zoneSatellite,
            capteur_id: row.satellite_id,
            parametre: 'ndwi',
            valeur: ndwiValue,
            seuil_oms: SATELLITE_SEUILS.ndwi.min,
            type_polluant: SATELLITE_SEUILS.ndwi.polluant,
            population_exposee: Math.floor(population * 0.4),
            message: `⚠️ NDWI faible (${ndwiValue.toFixed(2)}) indiquant un potentiel assèchement / retrait du littoral sur ${zoneSatellite}.`,
          });
        }
      }
    }
  } catch (error) {
    // La table peut ne pas être disponible dès le démarrage
    if (error && !error.message?.includes('donnees_satellite')) {
      console.log('⚠️ Erreur lors de la lecture satellite:', error.message);
    }
  }

  return alertes;
}

// Fonction pour initialiser les bases de données
async function initDatabase() {
  try {
    // Connexion à PostgreSQL pour les alertes
    dbClient = new Client(dbConfig);
    await dbClient.connect();
    console.log('✅ Connecté à PostgreSQL (alertes)');

    // Connexion à TimescaleDB pour lire les données
    timescaleClient = new Client(timescaleConfig);
    await timescaleClient.connect();
    console.log('✅ Connecté à TimescaleDB');

    // Créer la table si elle n'existe pas avec plus de détails
    await dbClient.query(`
      CREATE TABLE IF NOT EXISTS alertes (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        type VARCHAR(50) NOT NULL,
        severity VARCHAR(20) NOT NULL,
        zone VARCHAR(50),
        zone_geographique VARCHAR(100),
        capteur_id VARCHAR(50),
        parametre VARCHAR(50),
        valeur DECIMAL(10,2),
        seuil_oms DECIMAL(10,2),
        type_polluant VARCHAR(50),
        population_exposee INTEGER,
        message TEXT NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
        email_sent BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      );
    `);

    // Ajouter les nouvelles colonnes si elles n'existent pas
    try {
      await dbClient.query(`ALTER TABLE alertes ADD COLUMN IF NOT EXISTS zone_geographique VARCHAR(100);`);
      await dbClient.query(`ALTER TABLE alertes ADD COLUMN IF NOT EXISTS capteur_id VARCHAR(50);`);
      await dbClient.query(`ALTER TABLE alertes ADD COLUMN IF NOT EXISTS parametre VARCHAR(50);`);
      await dbClient.query(`ALTER TABLE alertes ADD COLUMN IF NOT EXISTS valeur DECIMAL(10,2);`);
      await dbClient.query(`ALTER TABLE alertes ADD COLUMN IF NOT EXISTS seuil_oms DECIMAL(10,2);`);
      await dbClient.query(`ALTER TABLE alertes ADD COLUMN IF NOT EXISTS type_polluant VARCHAR(50);`);
      await dbClient.query(`ALTER TABLE alertes ADD COLUMN IF NOT EXISTS population_exposee INTEGER;`);
    } catch (error) {
      // Colonnes déjà présentes
    }

    // Créer des index
    await dbClient.query(`
      CREATE INDEX IF NOT EXISTS idx_alertes_timestamp ON alertes(timestamp);
    `);
    await dbClient.query(`
      CREATE INDEX IF NOT EXISTS idx_alertes_status ON alertes(status);
    `);
    await dbClient.query(`
      CREATE INDEX IF NOT EXISTS idx_alertes_severity ON alertes(severity);
    `);

    console.log('✅ Table alertes créée/initialisée');
  } catch (error) {
    console.error('❌ Erreur lors de l\'initialisation de la base de données:', error.message);
    throw error;
  }
}

// Fonction pour insérer une alerte dans la base
async function insertAlert(alert) {
  try {
    const query = `
      INSERT INTO alertes (
        timestamp, type, severity, zone, zone_geographique, capteur_id,
        parametre, valeur, seuil_oms, type_polluant, population_exposee,
        message, status
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
      RETURNING id
    `;
    const result = await dbClient.query(query, [
      alert.timestamp,
      alert.type,
      alert.severity,
      alert.zone,
      alert.zone_geographique || alert.zone,
      alert.capteur_id || null,
      alert.parametre || null,
      alert.valeur || null,
      alert.seuil_oms || null,
      alert.type_polluant || null,
      alert.population_exposee || null,
      alert.message,
      alert.status,
    ]);

    const alertId = result.rows[0].id;
    const severityIcon = alert.severity === 'CRITICAL' ? '🚨' : alert.severity === 'WARNING' ? '⚠️' : 'ℹ️';
    console.log(`${severityIcon} Alerte [${alertId}]: ${alert.severity} - ${alert.type} - ${alert.zone_geographique || alert.zone} (${alert.parametre}: ${alert.valeur})`);

    // Simuler l'envoi d'email
    await sendEmailNotification(alert);

    return alertId;
  } catch (error) {
    console.error('❌ Erreur lors de l\'insertion de l\'alerte:', error.message);
  }
}

// Fonction pour envoyer une notification par email (simulation)
async function sendEmailNotification(alert) {
  try {
    const mailOptions = {
      from: 'alertes@aquawatch.com',
      to: 'admin@aquawatch.com',
      subject: `[${alert.severity}] Alerte AquaWatch - ${alert.type}`,
      text: alert.message,
      html: `
        <h2>Alerte AquaWatch</h2>
        <p><strong>Type:</strong> ${alert.type}</p>
        <p><strong>Sévérité:</strong> ${alert.severity}</p>
        <p><strong>Zone:</strong> ${alert.zone}</p>
        <p><strong>Message:</strong> ${alert.message}</p>
        <p><strong>Timestamp:</strong> ${alert.timestamp}</p>
      `,
    };

    // En mode simulation, on ne va pas vraiment envoyer l'email
    // mais on simule l'envoi
    console.log(`📧 Email simulé envoyé pour l'alerte: ${alert.type} - ${alert.zone}`);

    // Marquer l'email comme envoyé dans la base de données
    // (dans un vrai système, on ferait ça après confirmation d'envoi)

  } catch (error) {
    console.error('❌ Erreur lors de l\'envoi de l\'email:', error.message);
  }
}

// Fonction principale
async function main() {
  console.log('🚀 Démarrage du service alertes...');

  // Attendre que la base de données soit disponible
  let retries = 30;
  while (retries > 0) {
    try {
      await initDatabase();
      break;
    } catch (error) {
      retries--;
      if (retries === 0) {
        console.error('❌ Impossible de se connecter à la base de données');
        process.exit(1);
      }
      console.log(`⏳ Attente de la base de données... (${30 - retries}/30)`);
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  console.log('✅ Service alertes démarré - Vérification des seuils OMS toutes les 7 secondes');

  // Register with Eureka for service discovery
  try {
    // Start a simple HTTP server for Eureka health check
    const http = require('http');
    const HEALTH_PORT = process.env.HEALTH_PORT || 3002;

    const server = http.createServer((req, res) => {
      if (req.url === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'UP' }));
      } else {
        res.writeHead(404);
        res.end();
      }
    });

    server.listen(HEALTH_PORT, async () => {
      console.log(`🏥 Health check server listening on port ${HEALTH_PORT}`);
      const { registerService, waitForEureka, setupGracefulShutdown } = require('./shared/eureka-client');
      setupGracefulShutdown();
      if (await waitForEureka(30, 2000)) {
        await registerService('alertes', HEALTH_PORT, '/health');
      }
    });
  } catch (eurekaError) {
    console.log('⚠️ Eureka setup failed:', eurekaError.message);
  }

  // Vérifier les seuils OMS et générer des alertes toutes les 7 secondes
  setInterval(async () => {
    try {
      const alertes = await checkOMSThresholds();

      // Insérer seulement les vraies alertes (pas de SYSTEM_OK automatique)
      if (alertes && alertes.length > 0) {
        console.log(`📊 ${alertes.length} alerte(s) détectée(s) - Génération des alertes...`);
        for (const alert of alertes) {
          await insertAlert({
            timestamp: new Date().toISOString(),
            type: alert.type,
            severity: alert.severity,
            zone: alert.zone,
            zone_geographique: alert.zone_geographique,
            capteur_id: alert.capteur_id,
            parametre: alert.parametre,
            valeur: alert.valeur,
            seuil_oms: alert.seuil_oms,
            type_polluant: alert.type_polluant,
            population_exposee: alert.population_exposee,
            message: alert.message,
            status: 'ACTIVE',
          });
        }
      }
      // Ne plus générer d'alertes SYSTEM_OK - seulement les vraies alertes OMS détectées
    } catch (error) {
      // Ignorer complètement les erreurs de table satellite (c'est normal qu'elle n'existe pas)
      if (error && error.message && !error.message.includes('donnees_satellite') && !error.message.includes('relation "donnees_satellite"')) {
        console.error('❌ Erreur lors de la vérification des seuils:', error.message);
      }
      // L'erreur satellite est normale, continuer silencieusement
    }
  }, 7000);
}

// Gestion de l'arrêt propre
process.on('SIGTERM', async () => {
  console.log('🛑 Arrêt du service alertes...');
  if (dbClient) await dbClient.end();
  if (timescaleClient) await timescaleClient.end();
  process.exit(0);
});

process.on('SIGINT', async () => {
  console.log('🛑 Arrêt du service alertes...');
  if (dbClient) await dbClient.end();
  if (timescaleClient) await timescaleClient.end();
  process.exit(0);
});

// Démarrer le service
main().catch((error) => {
  console.error('❌ Erreur fatale:', error);
  process.exit(1);
});

