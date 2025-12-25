const express = require('express');
const { Client } = require('pg');
const cors = require('cors');
const swaggerUi = require('swagger-ui-express');
const swaggerJsdoc = require('swagger-jsdoc');

const app = express();
app.use(cors());
app.use(express.json());

// Configuration Swagger
const swaggerOptions = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'AquaWatch API',
      version: '1.0.0',
      description: 'API REST pour le système de surveillance de qualité de l\'eau AquaWatch. Fournit les données des capteurs, prédictions IA, alertes et statistiques.',
      contact: {
        name: 'Équipe AquaWatch',
        email: 'dafalighayt@gmail.com'
      }
    },
    servers: [
      {
        url: 'http://localhost:3000',
        description: 'Serveur de développement'
      }
    ],
    tags: [
      { name: 'Capteurs', description: 'Données des capteurs IoT' },
      { name: 'Satellite', description: 'Données satellite' },
      { name: 'Prédictions', description: 'Prédictions IA 24h' },
      { name: 'Alertes', description: 'Alertes OMS' },
      { name: 'Statistiques', description: 'Statistiques globales' },
      { name: 'Santé', description: 'État du service' }
    ]
  },
  apis: ['./index.js']
};

const swaggerSpec = swaggerJsdoc(swaggerOptions);
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec, {
  customCss: '.swagger-ui .topbar { display: none }',
  customSiteTitle: 'AquaWatch API Documentation'
}));

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

/**
 * @swagger
 * /health:
 *   get:
 *     summary: Vérifier l'état de santé de l'API
 *     tags: [Santé]
 *     responses:
 *       200:
 *         description: Service opérationnel
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: OK
 *                 service:
 *                   type: string
 *                   example: API-SIG
 */
app.get('/health', (req, res) => {
  res.json({ status: 'OK', service: 'API-SIG' });
});

/**
 * @swagger
 * /api/capteurs:
 *   get:
 *     summary: Récupérer les données des capteurs IoT
 *     tags: [Capteurs]
 *     parameters:
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 100
 *         description: Nombre maximum de résultats
 *     responses:
 *       200:
 *         description: Collection GeoJSON des mesures capteurs
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 type:
 *                   type: string
 *                   example: FeatureCollection
 *                 features:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       type:
 *                         type: string
 *                         example: Feature
 *                       properties:
 *                         type: object
 *                         properties:
 *                           capteur_id:
 *                             type: string
 *                             example: CAPT-1
 *                           zone:
 *                             type: string
 *                             example: Rabat-Centre
 *                           ph:
 *                             type: number
 *                             example: 7.2
 *                           turbidite:
 *                             type: number
 *                             example: 1.5
 *                           temperature:
 *                             type: number
 *                             example: 22.5
 *       500:
 *         description: Erreur serveur
 */
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

/**
 * @swagger
 * /api/test-discovery:
 *   get:
 *     summary: Prouver la communication entre microservices via Eureka
 *     tags: [Santé]
 *     responses:
 *       200:
 *         description: Succès de la communication
 */
app.get('/api/test-discovery', async (req, res) => {
  try {
    const axios = require('axios');
    const { discoverService } = require('./shared/eureka-client');

    // 1. Découverte via Eureka
    const service = discoverService('capteurs');
    if (!service) {
      return res.status(503).json({
        status: 'ERROR',
        message: 'Le service CAPTEURS n\'est pas visible dans Eureka'
      });
    }

    // 2. Appel du service découvert
    const targetUrl = `http://${service.host}:${service.port}/health`;
    console.log(`🔍 Test de communication vers: ${targetUrl}`);

    const response = await axios.get(targetUrl, { timeout: 2000 });

    res.json({
      status: 'SUCCESS',
      message: '✅ Communication inter-service réussie !',
      discovery_proof: {
        source: 'api-sig',
        lookup: 'capteurs',
        found_address: service.host,
        found_port: service.port,
        connection_url: targetUrl
      },
      remote_response: response.data
    });
  } catch (error) {
    res.status(500).json({
      status: 'FAILED',
      error: error.message,
      details: 'Impossible de joindre le microservice cible'
    });
  }
});


/**
 * @swagger
 * /api/satellite:
 *   get:
 *     summary: Récupérer les données satellite
 *     tags: [Satellite]
 *     parameters:
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 100
 *         description: Nombre maximum de résultats
 *     responses:
 *       200:
 *         description: Collection GeoJSON des observations satellite
 *       500:
 *         description: Erreur serveur
 */
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

/**
 * @swagger
 * /api/predictions:
 *   get:
 *     summary: Récupérer les prédictions IA de qualité de l'eau
 *     tags: [Prédictions]
 *     parameters:
 *       - in: query
 *         name: date
 *         schema:
 *           type: string
 *           format: date
 *         description: Date des prédictions (YYYY-MM-DD)
 *         example: 2024-12-16
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 100
 *         description: Nombre maximum de résultats
 *     responses:
 *       200:
 *         description: Liste des prédictions horaires par zone
 *         content:
 *           application/json:
 *             schema:
 *               type: array
 *               items:
 *                 type: object
 *                 properties:
 *                   zone_id:
 *                     type: string
 *                     example: Rabat-Centre
 *                   ph_pred:
 *                     type: number
 *                     example: 7.15
 *                   turbidite_pred:
 *                     type: number
 *                     example: 1.2
 *                   temperature_pred:
 *                     type: number
 *                     example: 23.5
 *                   qualite_score:
 *                     type: number
 *                     example: 85.0
 *                   qualite_niveau:
 *                     type: string
 *                     example: Excellente
 *                   risque_score:
 *                     type: number
 *                     example: 15.0
 *                   risque_niveau:
 *                     type: string
 *                     example: Faible
 *                   confidence:
 *                     type: number
 *                     example: 90.5
 *       500:
 *         description: Erreur serveur
 */
app.get('/api/predictions', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 100;
    const date = req.query.date; // Nouveau paramètre pour filtrer par date

    let query = `
      SELECT 
        id,
        timestamp,
        zone_id,
        ph_pred,
        turbidite_pred,
        temperature_pred,
        qualite_score,
        qualite_niveau,
        risque_score,
        risque_niveau,
        prediction_horizon,
        confidence,
        model_version
      FROM predictions_qualite
    `;


    const params = [];

    // Si une date est spécifiée, filtrer par cette date
    if (date) {
      // Convertir la date en objet Date pour validation
      const targetDate = new Date(date);
      if (isNaN(targetDate.getTime())) {
        return res.status(400).json({ error: 'Invalid date format. Use YYYY-MM-DD' });
      }

      // Ajouter la condition WHERE pour filtrer par date
      query += ` WHERE DATE(timestamp) = $1`;
      params.push(date);

      // Ajouter le tri par timestamp
      query += ` ORDER BY timestamp ASC`;
    } else {
      // Tri par défaut par timestamp décroissant
      query += ` ORDER BY timestamp DESC`;
    }

    // Ajouter la limite
    query += ` LIMIT $${params.length + 1}`;
    params.push(limit);

    const result = await dbClient.query(query, params);
    res.json(result.rows);
  } catch (error) {
    console.error('Erreur:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * @swagger
 * /api/alertes:
 *   get:
 *     summary: Récupérer les alertes de dépassement OMS
 *     tags: [Alertes]
 *     parameters:
 *       - in: query
 *         name: severity
 *         schema:
 *           type: string
 *           enum: [CRITICAL, WARNING]
 *         description: Filtrer par niveau de sévérité
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 50
 *         description: Nombre maximum de résultats
 *     responses:
 *       200:
 *         description: Liste des alertes actives
 *         content:
 *           application/json:
 *             schema:
 *               type: array
 *               items:
 *                 type: object
 *                 properties:
 *                   id:
 *                     type: integer
 *                   severity:
 *                     type: string
 *                     example: CRITICAL
 *                   zone:
 *                     type: string
 *                     example: Marrakech
 *                   parametre:
 *                     type: string
 *                     example: ph
 *                   valeur:
 *                     type: number
 *                     example: 5.5
 *                   seuil_oms:
 *                     type: number
 *                     example: 6.5
 *                   message:
 *                     type: string
 *       500:
 *         description: Erreur serveur
 */
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

/**
 * @swagger
 * /api/stats:
 *   get:
 *     summary: Récupérer les statistiques globales du système
 *     tags: [Statistiques]
 *     responses:
 *       200:
 *         description: Statistiques agrégées
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 capteurs:
 *                   type: object
 *                   properties:
 *                     capteurs_uniques:
 *                       type: integer
 *                       example: 16
 *                     total_mesures:
 *                       type: integer
 *                       example: 25000
 *                     ph_moyen:
 *                       type: string
 *                       example: "7.15"
 *                     turbidite_moyenne:
 *                       type: string
 *                       example: "1.25"
 *                 satellite:
 *                   type: object
 *                   properties:
 *                     total:
 *                       type: integer
 *                     chlorophylle_moyenne:
 *                       type: string
 *                 predictions:
 *                   type: object
 *                   properties:
 *                     total:
 *                       type: integer
 *                     qualite_moyenne:
 *                       type: string
 *                 alertes:
 *                   type: object
 *                   properties:
 *                     total:
 *                       type: integer
 *       500:
 *         description: Erreur serveur
 */
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
    app.listen(PORT, async () => {
      console.log(`🚀 API-SIG démarrée sur le port ${PORT}`);
      console.log(`📚 Documentation Swagger: http://localhost:${PORT}/api-docs`);
      console.log(`📡 Endpoints disponibles:`);
      console.log(`   - GET /health`);
      console.log(`   - GET /api/capteurs`);
      console.log(`   - GET /api/satellite`);
      console.log(`   - GET /api/predictions`);
      console.log(`   - GET /api/alertes`);
      console.log(`   - GET /api/stats`);

      // Register with Eureka for service discovery with HTTP health check
      try {
        const { registerService, waitForEureka, setupGracefulShutdown } = require('./shared/eureka-client');
        setupGracefulShutdown();
        if (await waitForEureka(30, 2000)) {
          await registerService('api-sig', PORT, '/health');
        }
      } catch (eurekaError) {
        console.log('⚠️ Eureka registration failed:', eurekaError.message);
      }
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

