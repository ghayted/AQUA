// ===================================
// AquaWatch - Shared API Functions
// ===================================

const API_BASE = '/api';

// Fetch wrapper with error handling
async function fetchAPI(endpoint, options = {}) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`, options);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (error) {
        console.error(`API Error [${endpoint}]:`, error);
        throw error;
    }
}

// Get sensors data
async function getSensors(limit = 50) {
    const data = await fetchAPI(`/capteurs?limit=${limit}`);

    // Deduplicate by capteur_id (keep latest)
    const sensorMap = new Map();
    data.features.forEach(f => {
        const id = f.properties.capteur_id;
        const existing = sensorMap.get(id);
        if (!existing || new Date(f.properties.timestamp) > new Date(existing.properties.timestamp)) {
            sensorMap.set(id, f);
        }
    });

    return Array.from(sensorMap.values());
}

// Get alerts
async function getAlerts(limit = 50) {
    return await fetchAPI(`/alertes?limit=${limit}`);
}

// Get predictions for tomorrow
async function getPredictions() {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    const dateString = tomorrow.toISOString().split('T')[0];

    const data = await fetchAPI(`/predictions?date=${dateString}&limit=100`);

    // Deduplicate by zone_id
    const zoneMap = new Map();
    data.forEach(p => zoneMap.set(p.zone_id, p));

    return Array.from(zoneMap.values());
}

// Get statistics
async function getStats() {
    try {
        const [sensors, alerts, predictions] = await Promise.all([
            getSensors(100),
            getAlerts(50),
            getPredictions()
        ]);

        // Calculate stats
        const phValues = sensors.map(s => parseFloat(s.properties.ph)).filter(v => !isNaN(v));
        const turbValues = sensors.map(s => parseFloat(s.properties.turbidite)).filter(v => !isNaN(v));

        const avgPh = phValues.length > 0
            ? (phValues.reduce((a, b) => a + b, 0) / phValues.length).toFixed(2)
            : '-';

        const avgTurb = turbValues.length > 0
            ? (turbValues.reduce((a, b) => a + b, 0) / turbValues.length).toFixed(2)
            : '-';

        const criticalAlerts = alerts.filter(a => a.severity === 'CRITICAL').length;
        const warningAlerts = alerts.filter(a => a.severity === 'WARNING').length;

        const avgQuality = predictions.length > 0
            ? (predictions.reduce((sum, p) => sum + parseFloat(p.qualite_score || 0), 0) / predictions.length).toFixed(0)
            : '-';

        return {
            sensorCount: sensors.length,
            alertCount: criticalAlerts + warningAlerts,
            criticalAlerts,
            warningAlerts,
            avgPh,
            avgTurb,
            avgQuality,
            sensors,
            alerts,
            predictions
        };
    } catch (error) {
        console.error('Stats error:', error);
        return null;
    }
}

// OMS Thresholds
const OMS_THRESHOLDS = {
    ph: { min: 6.5, max: 8.5, criticalMin: 6.0, criticalMax: 9.0 },
    turbidity: { max: 1.0, critical: 5.0 },
    temperature: { max: 25.0, critical: 30.0 }
};

// Get sensor status based on values
function getSensorStatus(ph, turbidity, temperature) {
    const phVal = parseFloat(ph);
    const turbVal = parseFloat(turbidity);
    const tempVal = parseFloat(temperature);

    // Critical
    if (phVal < OMS_THRESHOLDS.ph.criticalMin || phVal > OMS_THRESHOLDS.ph.criticalMax ||
        turbVal > OMS_THRESHOLDS.turbidity.critical) {
        return { status: 'Critique', color: '#ef4444', level: 'danger' };
    }

    // Warning
    if (phVal < OMS_THRESHOLDS.ph.min || phVal > OMS_THRESHOLDS.ph.max ||
        turbVal > OMS_THRESHOLDS.turbidity.max ||
        tempVal > OMS_THRESHOLDS.temperature.critical) {
        return { status: 'Attention', color: '#f59e0b', level: 'warning' };
    }

    // Normal
    return { status: 'Normal', color: '#10b981', level: 'success' };
}

// Format date/time
function formatDateTime(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatTime(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleTimeString('fr-FR', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Color functions
function getColorForScore(score) {
    if (score >= 70) return '#10b981';
    if (score >= 50) return '#f59e0b';
    return '#ef4444';
}

function getColorForRisk(risk) {
    if (risk <= 20) return '#10b981';
    if (risk <= 50) return '#f59e0b';
    return '#ef4444';
}
