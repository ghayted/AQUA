// Configuration
const API_BASE = '/api';
const REFRESH_INTERVAL = 10000; // 10 seconds

// State
let map;
let markerLayer;
let phChart;
let globalAlerts = [];
let globalSensors = [];
let currentAlertFilter = 'all';
let currentSensorFilter = 'all';

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initChart();
    refreshData();

    // Auto refresh
    setInterval(refreshData, REFRESH_INTERVAL);
});

// Map Initialization
function initMap() {
    map = L.map('map').setView([34.020882, -6.841650], 12); // Rabat Center

    // Dark Mode Tiles (Pro)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    markerLayer = L.layerGroup().addTo(map);
}

// Chart Initialization
function initChart() {
    const ctx = document.getElementById('phChart').getContext('2d');

    // Gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(16, 185, 129, 0.5)'); // Emerald
    gradient.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

    phChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'pH Moyen',
                data: [],
                borderColor: '#10b981', // Emerald 500
                backgroundColor: gradient,
                tension: 0.4,
                fill: true,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1f2937',
                    titleColor: '#f9fafb',
                    bodyColor: '#d1d5db',
                    borderColor: '#374151',
                    borderWidth: 1
                }
            },
            scales: {
                y: {
                    grid: { color: '#374151' },
                    ticks: { color: '#9ca3af' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#9ca3af' }
                }
            }
        }
    });
}

// Data Fetching & Refresh
async function refreshData() {
    updateStatusIndicator(true);
    try {
        await Promise.all([
            loadStats(),
            loadSensors(),
            loadTomorrowPredictions(), // J+1 Logic
            loadAlerts()
        ]);
        // Rendre les capteurs après chargement
        renderSensors();
        updateStatusIndicator(false);
    } catch (error) {
        console.error('Refresh failed:', error);
        updateStatusIndicator(false, true);
    }
}

function updateStatusIndicator(loading, error = false) {
    const el = document.getElementById('status-text');
    const dot = document.getElementById('status-dot');

    if (error) {
        el.textContent = 'Erreur Connexion';
        el.style.color = '#ef4444';
        dot.style.color = '#ef4444';
    } else if (loading) {
        el.textContent = 'Mise à jour...';
    } else {
        el.textContent = 'Système Live';
        el.style.color = '#10b981';
        dot.style.color = '#10b981';
    }
}

// 1. Load Global Stats
async function loadStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        const data = await res.json();

        document.getElementById('stat-sensors').textContent = data.capteurs.capteurs_uniques || 0;
        document.getElementById('stat-alerts').textContent = data.alertes.total || 0;
        document.getElementById('stat-ph').textContent = data.capteurs.ph_moyen || '-';
        document.getElementById('stat-turb').textContent = data.capteurs.turbidite_moyenne || '-';
    } catch (e) { console.error(e); }
}

// 2. Load Sensors on Map
async function loadSensors() {
    try {
        const res = await fetch(`${API_BASE}/capteurs?limit=50`);
        const data = await res.json();

        // Don't clear layers if we want to update smoothly, but for simplicity clear for now to avoid dupes
        markerLayer.clearLayers();

        // Seuils OMS (identiques à alertes/index.js)
        const OMS_SEUILS = {
            ph: { min: 6.5, max: 8.5, criticalMin: 6.0, criticalMax: 9.0 },
            turbidite: { max: 1.0, critical: 5.0 },
            temperature: { max: 25.0, critical: 30.0 }
        };

        // Dédupliquer les capteurs - garder seulement la dernière mesure par capteur_id
        const sensorMap = new Map();
        data.features.forEach(f => {
            const capteurId = f.properties.capteur_id;
            const existingSensor = sensorMap.get(capteurId);
            // Garder la mesure la plus récente
            if (!existingSensor || new Date(f.properties.timestamp) > new Date(existingSensor.properties.timestamp)) {
                sensorMap.set(capteurId, f);
            }
        });

        // Convertir en array de capteurs uniques
        const uniqueSensors = Array.from(sensorMap.values());

        // Réinitialiser globalSensors
        globalSensors = [];

        uniqueSensors.forEach(f => {
            const props = f.properties;
            const [lon, lat] = f.geometry.coordinates;

            // Marker Color Logic - Synchronisé avec les seuils OMS du backend alertes
            let color = '#10b981'; // Success (Emerald)
            let status = 'Normal';
            let alertLevel = 'normal';

            // Critical - pH dangereux ou turbidité excessive
            if (props.ph < OMS_SEUILS.ph.criticalMin || props.ph > OMS_SEUILS.ph.criticalMax || props.turbidite > OMS_SEUILS.turbidite.critical) {
                color = '#ef4444'; // Danger
                status = 'Critique';
                alertLevel = 'critical';
            }
            // Warning - pH hors norme, turbidité élevée ou température excessive
            else if (props.ph < OMS_SEUILS.ph.min || props.ph > OMS_SEUILS.ph.max || props.turbidite > OMS_SEUILS.turbidite.max || props.temperature > OMS_SEUILS.temperature.critical) {
                color = '#f59e0b'; // Warning
                status = 'Attention';
                alertLevel = 'warning';
            }

            // Stocker le capteur avec son état
            globalSensors.push({
                capteur_id: props.capteur_id,
                zone: props.zone,
                ph: props.ph,
                turbidite: props.turbidite,
                temperature: props.temperature,
                timestamp: props.timestamp,
                status: status,
                alertLevel: alertLevel,
                color: color,
                lat: lat,
                lon: lon,
                // Déterminer quels paramètres sont hors seuils
                phOutOfRange: props.ph < OMS_SEUILS.ph.criticalMin || props.ph > OMS_SEUILS.ph.criticalMax,
                phWarning: (props.ph < OMS_SEUILS.ph.min || props.ph > OMS_SEUILS.ph.max) && !(props.ph < OMS_SEUILS.ph.criticalMin || props.ph > OMS_SEUILS.ph.criticalMax),
                turbOutOfRange: props.turbidite > OMS_SEUILS.turbidite.critical,
                turbWarning: props.turbidite > OMS_SEUILS.turbidite.max && props.turbidite <= OMS_SEUILS.turbidite.critical,
                tempWarning: props.temperature > OMS_SEUILS.temperature.critical
            });

            // Create Icon
            const icon = L.divIcon({
                className: 'custom-dot',
                html: `<div style="background:${color};width:14px;height:14px;border-radius:50%;box-shadow:0 0 15px ${color};border:2px solid #fff;"></div>`,
                iconSize: [18, 18],
                iconAnchor: [9, 9]
            });

            // Create Marker
            const marker = L.marker([lat, lon], { icon }).addTo(markerLayer);

            // Bind Popup with Name
            marker.bindPopup(`
                <div style="min-width:200px">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                        <strong style="color:#f9fafb;font-size:1rem">${props.capteur_id}</strong>
                        <span style="background:${color}20;color:${color};padding:2px 8px;border-radius:12px;font-size:0.75rem;border:1px solid ${color}">${status}</span>
                    </div>
                     <div style="font-size:0.85rem;color:#d1d5db;line-height:1.6">
                        pH: <b style="color:#f9fafb">${props.ph}</b><br>
                        Turbidité: <b style="color:#f9fafb">${props.turbidite} NTU</b><br>
                        Temp: <b style="color:#f9fafb">${props.temperature}°C</b>
                    </div>
                </div>
            `);

            // Update chart
            const time = new Date(props.timestamp).toLocaleTimeString();
            if (phChart.data.labels.length > 20) phChart.data.labels.shift();
            phChart.data.labels.push(time);
            if (phChart.data.datasets[0].data.length > 20) phChart.data.datasets[0].data.shift();
            phChart.data.datasets[0].data.push(props.ph);
        });
        phChart.update('none');

    } catch (e) { console.error(e); }
}

// 3. Load Predictions (J+1 Logic) with Interaction
async function loadTomorrowPredictions() {
    const listEl = document.getElementById('predictions-list');
    try {
        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 1);
        const dateString = tomorrow.toISOString().split('T')[0];

        const res = await fetch(`${API_BASE}/predictions?date=${dateString}&limit=100`);
        const data = await res.json();

        if (!data || data.length === 0) {
            listEl.innerHTML = `<div class="loading">Aucune prédiction pour ${dateString}<br><small>Redémarrez le service pour 'Mode Démo'</small></div>`;
            return;
        }

        const zoneMap = new Map();
        data.forEach(p => zoneMap.set(p.zone_id, p));

        listEl.innerHTML = Array.from(zoneMap.values()).map(pred => {
            const qualite = pred.qualite_score ? parseFloat(pred.qualite_score) : 0;
            const risque = pred.risque_score ? parseFloat(pred.risque_score) : 0;
            const conf = pred.confidence ? parseFloat(pred.confidence) : 100;

            // Déterminer le statut global de l'eau
            let waterStatus, waterStatusColor, waterIcon, waterMessage;
            if (qualite >= 70) {
                waterStatus = 'Eau de bonne qualité';
                waterStatusColor = '#10b981';
                waterIcon = '✅';
                waterMessage = 'Aucune action requise. Eau conforme aux normes.';
            } else if (qualite >= 50) {
                waterStatus = 'Qualité acceptable';
                waterStatusColor = '#f59e0b';
                waterIcon = '⚠️';
                waterMessage = 'Surveillance recommandée. Paramètres légèrement hors optimal.';
            } else {
                waterStatus = 'Attention requise';
                waterStatusColor = '#ef4444';
                waterIcon = '🚨';
                waterMessage = 'Action recommandée. Vérifiez les paramètres avant utilisation.';
            }

            // Interprétation du risque
            let riskLevel, riskMessage;
            if (risque <= 10) {
                riskLevel = 'Très faible';
                riskMessage = 'Peu probable que survienne un problème.';
            } else if (risque <= 30) {
                riskLevel = 'Faible';
                riskMessage = 'Risque minime de dégradation.';
            } else if (risque <= 60) {
                riskLevel = 'Modéré';
                riskMessage = 'Possibilité de dégradation. Surveillance conseillée.';
            } else {
                riskLevel = 'Élevé';
                riskMessage = 'Forte probabilité de problème. Intervention urgente.';
            }

            // Analyser les paramètres pour donner des conseils
            const ph = parseFloat(pred.ph_pred) || 7;
            const turb = parseFloat(pred.turbidite_pred) || 0;
            const temp = parseFloat(pred.temperature_pred) || 20;

            let paramWarnings = [];
            if (ph < 6.5) paramWarnings.push('pH trop acide');
            else if (ph > 8.5) paramWarnings.push('pH trop alcalin');
            if (turb > 1.0) paramWarnings.push('Turbidité élevée');
            if (temp > 25) paramWarnings.push('Température élevée');

            const safeZone = pred.zone_id.replace(/'/g, "\\'");
            const safePred = JSON.stringify(pred).replace(/"/g, '&quot;');

            return `
                <div class="prediction-card" data-zone="${safeZone}" data-pred='${safePred}'>
                    <!-- En-tête avec zone et statut -->
                    <div class="pred-card-header">
                        <div class="pred-zone">
                            <span class="pred-zone-icon">📍</span>
                            <span class="pred-zone-name">${pred.zone_id}</span>
                        </div>
                        <div class="pred-date-badge">
                            <span>🗓️ Demain 12h</span>
                        </div>
                    </div>
                    
                    <!-- Statut principal -->
                    <div class="pred-main-status" style="background: ${waterStatusColor}15; border-left: 4px solid ${waterStatusColor}">
                        <div class="pred-status-icon">${waterIcon}</div>
                        <div class="pred-status-text">
                            <strong style="color: ${waterStatusColor}">${waterStatus}</strong>
                            <small>${waterMessage}</small>
                        </div>
                    </div>
                    
                    <!-- Jauge de qualité -->
                    <div class="pred-gauge-section">
                        <div class="pred-gauge-label">
                            <span>🌊 Qualité de l'eau prévue</span>
                            <strong style="color:${getColorForScore(qualite)}">${qualite.toFixed(0)}%</strong>
                        </div>
                        <div class="pred-gauge-bar">
                            <div class="pred-gauge-fill" style="width:${qualite}%; background:${getColorForScore(qualite)}"></div>
                        </div>
                        <div class="pred-gauge-legend">
                            <span>Mauvaise</span>
                            <span>Moyenne</span>
                            <span>Excellente</span>
                        </div>
                    </div>
                    
                    <!-- Risque -->
                    <div class="pred-risk-section">
                        <div class="pred-risk-header">
                            <span>⚡ Risque de problème : <strong>${riskLevel}</strong> (${risque.toFixed(0)}%)</span>
                        </div>
                        <small class="pred-risk-explain">${riskMessage}</small>
                    </div>
                    
                    <!-- Paramètres prévus -->
                    <div class="pred-params">
                        <div class="pred-param">
                            <span class="pred-param-icon">🧪</span>
                            <span class="pred-param-label">pH</span>
                            <span class="pred-param-value ${ph < 6.5 || ph > 8.5 ? 'warning' : ''}">${ph.toFixed(2)}</span>
                            <span class="pred-param-norm">Norme: 6.5-8.5</span>
                        </div>
                        <div class="pred-param">
                            <span class="pred-param-icon">🌫️</span>
                            <span class="pred-param-label">Turbidité</span>
                            <span class="pred-param-value ${turb > 1.0 ? 'warning' : ''}">${turb.toFixed(2)} NTU</span>
                            <span class="pred-param-norm">Norme: < 1.0</span>
                        </div>
                        <div class="pred-param">
                            <span class="pred-param-icon">🌡️</span>
                            <span class="pred-param-label">Température</span>
                            <span class="pred-param-value ${temp > 25 ? 'warning' : ''}">${temp.toFixed(1)}°C</span>
                            <span class="pred-param-norm">Optimal: < 25°C</span>
                        </div>
                    </div>
                    
                    ${paramWarnings.length > 0 ? `
                    <div class="pred-warnings">
                        <strong>⚠️ Points d'attention :</strong> ${paramWarnings.join(', ')}
                    </div>
                    ` : ''}
                    
                    <!-- Fiabilité -->
                    <div class="pred-confidence">
                        <span>🎯 Fiabilité de cette prévision : <strong>${conf.toFixed(0)}%</strong></span>
                        <small>${conf >= 80 ? 'Prédiction très fiable (beaucoup de données)' : conf >= 50 ? 'Prédiction modérément fiable' : 'Peu de données, prédiction indicative'}</small>
                    </div>
                </div>
            `;
        }).join('');

        // Ajouter les event listeners pour les clics
        listEl.querySelectorAll('.prediction-card').forEach(card => {
            card.addEventListener('click', function () {
                const zoneId = this.dataset.zone;
                const pred = JSON.parse(this.dataset.pred.replace(/&quot;/g, '"'));
                focusOnPrediction(zoneId, pred, this);
            });
        });

    } catch (e) {
        console.error(e);
        listEl.innerHTML = `<div class="loading" style="color:var(--danger)">Erreur chargement</div>`;
    }
}

// Interaction: Focus on Map
function focusOnPrediction(zoneId, pred, element) {
    // 1. Highlight UI
    document.querySelectorAll('.prediction-card').forEach(el => el.classList.remove('selected'));
    element.classList.add('selected');

    // 2. Find coordinates (Mocked for zones since we don't have a zone-centroid DB, 
    //    but we can guess based on sensor names or just use a default offset if we had real geodata.
    //    For now, we will simply show a Popup in the center of the map explaining the prediction).

    // Better interaction: Try to find a sensor in this zone from the markers layer
    // This requires us to traverse the markerLayer. 
    // For this demo, let's display a dedicated "Prediction Popup" in center map for clarity.

    const center = map.getCenter();

    L.popup()
        .setLatLng(center)
        .setContent(`
            <div style="text-align:center">
                <h3 style="margin:0 0 8px 0;color:#10b981">🤖 Prévision IA: ${zoneId}</h3>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;text-align:left;font-size:0.9rem">
                    <div>Qualité: <b>${parseFloat(pred.qualite_score).toFixed(1)}%</b></div>
                    <div>Risque: <b>${parseFloat(pred.risque_score).toFixed(1)}%</b></div>
                    <div>Confiance: <b>${pred.confidence}%</b></div>
                </div>
                <hr style="border:0;border-top:1px solid #ddd;margin:8px 0">
                <small>Prévision valable pour demain à 12:00</small>
            </div>
        `)
        .openOn(map);
}

function getColorForScore(score) {
    if (score >= 70) return '#10b981';
    if (score >= 50) return '#f59e0b';
    return '#ef4444';
}

// 4. Load Alerts with Filtering
async function loadAlerts() {
    const listEl = document.getElementById('alerts-body');
    try {
        const res = await fetch(`${API_BASE}/alertes?limit=20`);
        const data = await res.json();
        globalAlerts = data; // Store for filtering
        renderAlerts();
    } catch (e) { console.error(e); }
}

function filterAlerts(level) {
    currentAlertFilter = level;

    // Update active button state
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('onclick').includes(level)) {
            btn.classList.add('active');
        }
    });

    renderAlerts();
}

function renderAlerts() {
    const listEl = document.getElementById('alerts-body');

    const filtered = globalAlerts.filter(a => {
        if (currentAlertFilter === 'all') return true;
        return a.severity === currentAlertFilter;
    });

    if (filtered.length === 0) {
        listEl.innerHTML = `<tr><td colspan="4" style="text-align:center;color:#9ca3af;padding:2rem">Aucune alerte correspondant aux critères</td></tr>`;
        return;
    }

    listEl.innerHTML = filtered.map(alerte => {
        const severityClass = alerte.severity === 'CRITICAL' ? 'danger' : 'warning';
        return `
            <tr>
                <td><span class="badge ${severityClass}">${alerte.severity}</span></td>
                <td>
                    <strong style="color:var(--text-primary)">${alerte.zone}</strong><br>
                    <small style="color:var(--text-secondary)">${alerte.capteur_id || 'Global'}</small>
                </td>
                <td>${alerte.message}</td>
                <td style="color:var(--text-secondary);font-size:0.8rem">${new Date(alerte.timestamp).toLocaleTimeString()}</td>
            </tr>
        `;
    }).join('');
}

// 5. Sensors Status Section
function filterSensors(filter) {
    currentSensorFilter = filter;

    // Update active button state for sensors filters
    document.querySelectorAll('.sensors-status-section .filter-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('onclick').includes(filter)) {
            btn.classList.add('active');
        }
    });

    renderSensors();
}

function renderSensors() {
    const listEl = document.getElementById('sensors-list');

    if (!globalSensors || globalSensors.length === 0) {
        listEl.innerHTML = `<div class="loading">Aucun capteur disponible</div>`;
        return;
    }

    // Filtrer les capteurs
    const filtered = globalSensors.filter(sensor => {
        if (currentSensorFilter === 'all') return true;
        if (currentSensorFilter === 'alert') return sensor.alertLevel !== 'normal';
        if (currentSensorFilter === 'normal') return sensor.alertLevel === 'normal';
        return true;
    });

    if (filtered.length === 0) {
        listEl.innerHTML = `<div class="loading">Aucun capteur correspondant aux critères</div>`;
        return;
    }

    listEl.innerHTML = filtered.map(sensor => {
        // Déterminer les classes CSS
        let cardClass = 'sensor-card';
        if (sensor.alertLevel === 'critical') cardClass += ' critical';
        else if (sensor.alertLevel === 'warning') cardClass += ' has-alert';

        // Générer l'indicateur d'alerte si nécessaire
        let alertBadge = '';
        if (sensor.alertLevel === 'critical') {
            alertBadge = '<span class="alert-indicator">🚨 ALERTE</span>';
        } else if (sensor.alertLevel === 'warning') {
            alertBadge = '<span class="alert-indicator warning-alert">⚠️ ALERTE</span>';
        }

        // Classes pour les valeurs hors seuils
        const phClass = sensor.phOutOfRange ? 'out-of-range' : (sensor.phWarning ? 'warning-range' : '');
        const turbClass = sensor.turbOutOfRange ? 'out-of-range' : (sensor.turbWarning ? 'warning-range' : '');
        const tempClass = sensor.tempWarning ? 'warning-range' : '';

        return `
            <div class="${cardClass}" onclick="focusOnSensor(${sensor.lat}, ${sensor.lon})">
                <div class="sensor-header">
                    <div>
                        <div class="sensor-name">${sensor.capteur_id}</div>
                        <div class="sensor-zone">📍 ${sensor.zone}</div>
                    </div>
                    <div class="sensor-status">
                        <div class="status-dot ${sensor.alertLevel}"></div>
                        ${alertBadge}
                    </div>
                </div>
                <div class="sensor-metrics">
                    <div class="sensor-metric">
                        <div class="sensor-metric-label">pH</div>
                        <div class="sensor-metric-value ${phClass}">${sensor.ph}</div>
                    </div>
                    <div class="sensor-metric">
                        <div class="sensor-metric-label">Turbidité</div>
                        <div class="sensor-metric-value ${turbClass}">${sensor.turbidite} NTU</div>
                    </div>
                    <div class="sensor-metric">
                        <div class="sensor-metric-label">Temp</div>
                        <div class="sensor-metric-value ${tempClass}">${sensor.temperature}°C</div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// Focus sur un capteur sur la carte
function focusOnSensor(lat, lon) {
    map.setView([lat, lon], 15);
}
