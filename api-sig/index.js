const express = require('express');
const { Client } = require('pg');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Configuration de la base de données PostGIS
const dbConfig = {
  host: process.env.POSTGIS_HOST || 'timescaledb',
  port: parseInt(process.env.POSTGIS_PORT || '5432'),
  database: process.env.POSTGIS_DB || 'aquawatch',
  user: process.env.POSTGIS_USER || 'postgres',
  password: process.env.POSTGIS_PASSWORD || 'postgres',
};

let dbClient;

// Initialiser la connexion à la base de données
async function initDatabase() {
  try {
    dbClient = new Client(dbConfig);
    await dbClient.connect();
    console.log('✅ API-SIG connecté à PostGIS/TimescaleDB');

    // PostGIS sera géré par les requêtes si disponible
    console.log('✅ Connexion établie');
  } catch (error) {
    console.error('❌ Erreur de connexion à la base de données:', error.message);
    throw error;
  }
}

// Route de santé
app.get('/health', (req, res) => {
  res.json({ status: 'OK', service: 'API-SIG' });
});

// GET /api/capteurs - Récupérer les données des capteurs avec géolocalisation
app.get('/api/capteurs', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 100;
    const query = `
      SELECT 
        id,
        timestamp,
        capteur_id,
        zone,
        ph,
        turbidite,
        temperature,
        latitude,
        longitude
      FROM donnees_capteurs
      ORDER BY timestamp DESC
      LIMIT $1
    `;
    const result = await dbClient.query(query, [limit]);
    
    const features = result.rows.map(row => ({
      type: 'Feature',
      properties: {
        id: row.id,
        timestamp: row.timestamp,
        capteur_id: row.capteur_id,
        zone: row.zone,
        ph: row.ph,
        turbidite: row.turbidite,
        temperature: row.temperature,
      },
      geometry: row.latitude && row.longitude ? {
        type: 'Point',
        coordinates: [row.longitude, row.latitude]
      } : null,
    }));

    res.json({
      type: 'FeatureCollection',
      features: features,
    });
  } catch (error) {
    console.error('Erreur:', error);
    res.status(500).json({ error: error.message });
  }
});

// GET /api/satellite - Récupérer les données satellite
app.get('/api/satellite', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 100;
    const query = `
      SELECT 
        id,
        timestamp,
        satellite_id,
        zone,
        latitude,
        longitude,
        chlorophylle,
        turbidite,
        ndwi,
        storage_url,
        source
      FROM donnees_satellite
      ORDER BY timestamp DESC
      LIMIT $1
    `;
    const result = await dbClient.query(query, [limit]);
    
    const features = result.rows.map(row => ({
      type: 'Feature',
      properties: {
        id: row.id,
        timestamp: row.timestamp,
        satellite_id: row.satellite_id,
        zone: row.zone,
        chlorophylle: row.chlorophylle,
        turbidite: row.turbidite,
        ndwi: row.ndwi,
        storage_url: row.storage_url,
        source: row.source,
      },
      geometry: row.latitude && row.longitude ? {
        type: 'Point',
        coordinates: [row.longitude, row.latitude]
      } : null,
    }));

    res.json({
      type: 'FeatureCollection',
      features: features,
    });
  } catch (error) {
    console.error('Erreur:', error);
    res.status(500).json({ error: error.message });
  }
});

// GET /api/predictions - Récupérer les prédictions
app.get('/api/predictions', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 100;
    const query = `
      SELECT 
        id,
        timestamp,
        zone_id,
        qualite_score,
        qualite_niveau,
        risque_score,
        risque_niveau,
        prediction_score,
        ph_pred,
        turbidite_pred,
        temperature_pred,
        prediction_horizon,
        confidence
      FROM predictions_qualite
      ORDER BY timestamp DESC
      LIMIT $1
    `;
    const result = await dbClient.query(query, [limit]);
    res.json(result.rows);
  } catch (error) {
    console.error('Erreur:', error);
    res.status(500).json({ error: error.message });
  }
});

// GET /api/alertes - Récupérer les alertes
app.get('/api/alertes', async (req, res) => {
  try {
    const alertesDb = new Client({
      host: process.env.POSTGRES_HOST || 'postgres',
      port: parseInt(process.env.POSTGRES_PORT || '5432'),
      database: process.env.POSTGRES_DB || 'alertes',
      user: process.env.POSTGRES_USER || 'postgres',
      password: process.env.POSTGRES_PASSWORD || 'postgres',
    });
    await alertesDb.connect();

    const limit = parseInt(req.query.limit) || 50;
    const severity = req.query.severity; // Filtrer par sévérité si fourni
    let query = `
      SELECT 
        id,
        timestamp,
        type,
        severity,
        zone,
        zone_geographique,
        capteur_id,
        parametre,
        valeur,
        seuil_oms,
        type_polluant,
        population_exposee,
        message,
        status
      FROM alertes
    `;
    const params = [];
    
    if (severity) {
      query += ` WHERE severity = $1 AND status = 'ACTIVE'`;
      params.push(severity);
      query += ` ORDER BY timestamp DESC LIMIT $2`;
      params.push(limit);
    } else {
      query += ` WHERE status = 'ACTIVE' ORDER BY timestamp DESC LIMIT $1`;
      params.push(limit);
    }
    
    const result = await alertesDb.query(query, params);
    await alertesDb.end();
    
    res.json(result.rows);
  } catch (error) {
    console.error('Erreur:', error);
    res.status(500).json({ error: error.message });
  }
});

// GET /api/stats - Statistiques globales
app.get('/api/stats', async (req, res) => {
  try {
    const capteursResult = await dbClient.query('SELECT COUNT(DISTINCT capteur_id) as capteurs_uniques, COUNT(*) as total_mesures, AVG(ph) as avg_ph, AVG(turbidite) as avg_turb FROM donnees_capteurs').catch(() => ({ rows: [{ capteurs_uniques: '0', total_mesures: '0', avg_ph: null, avg_turb: null }] }));
    const satelliteResult = await dbClient.query('SELECT COUNT(*) as count, AVG(chlorophylle) as avg_chloro, AVG(ndwi) as avg_ndwi FROM donnees_satellite').catch(() => ({ rows: [{ count: '0', avg_chloro: null, avg_ndwi: null }] }));
    const predictionsResult = await dbClient.query('SELECT COUNT(*) as count, AVG(qualite_score) as avg_qualite FROM predictions_qualite').catch(() => ({ rows: [{ count: '0', avg_qualite: null }] }));
    const alertes = await getAlertesCount();
    
    const capteurs = capteursResult.rows[0] || { capteurs_uniques: '0', total_mesures: '0', avg_ph: null, avg_turb: null };
    const satellite = satelliteResult.rows[0] || { count: '0', avg_chloro: null };
    const predictions = predictionsResult.rows[0] || { count: '0', avg_qualite: null };

    res.json({
      capteurs: {
        capteurs_uniques: parseInt(capteurs.capteurs_uniques || 0),
        total_mesures: parseInt(capteurs.total_mesures || 0),
        ph_moyen: parseFloat(capteurs.avg_ph || 0).toFixed(2),
        turbidite_moyenne: parseFloat(capteurs.avg_turb || 0).toFixed(2),
      },
      satellite: {
        total: parseInt(satellite.count || 0),
        chlorophylle_moyenne: parseFloat(satellite.avg_chloro || 0).toFixed(2),
        ndwi_moyen: parseFloat(satellite.avg_ndwi || 0).toFixed(2),
      },
      predictions: {
        total: parseInt(predictions.count || 0),
        qualite_moyenne: parseFloat(predictions.avg_qualite || 0).toFixed(2),
      },
      alertes: {
        total: alertes || 0,
      },
    });
  } catch (error) {
    console.error('Erreur:', error);
    res.status(500).json({ error: error.message });
  }
});

async function getAlertesCount() {
  try {
    const alertesDb = new Client({
      host: process.env.POSTGRES_HOST || 'postgres',
      port: parseInt(process.env.POSTGRES_PORT || '5432'),
      database: process.env.POSTGRES_DB || 'alertes',
      user: process.env.POSTGRES_USER || 'postgres',
      password: process.env.POSTGRES_PASSWORD || 'postgres',
    });
    await alertesDb.connect();
    const result = await alertesDb.query('SELECT COUNT(*) as count FROM alertes');
    await alertesDb.end();
    return parseInt(result.rows[0].count);
  } catch (error) {
    return 0;
  }
}

// Démarrer le serveur
const PORT = process.env.PORT || 3000;

async function start() {
  try {
    await initDatabase();
    app.listen(PORT, () => {
      console.log(`🚀 API-SIG démarrée sur le port ${PORT}`);
      console.log(`📡 Endpoints disponibles:`);
      console.log(`   - GET /health`);
      console.log(`   - GET /api/capteurs`);
      console.log(`   - GET /api/satellite`);
      console.log(`   - GET /api/predictions`);
      console.log(`   - GET /api/alertes`);
      console.log(`   - GET /api/stats`);
    });
  } catch (error) {
    console.error('❌ Erreur au démarrage:', error);
    process.exit(1);
  }
}

// Gestion de l'arrêt propre
process.on('SIGTERM', async () => {
  console.log('🛑 Arrêt de l\'API-SIG...');
  if (dbClient) await dbClient.end();
  process.exit(0);
});

start();

