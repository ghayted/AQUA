# AQUA Technical Documentation

## System Architecture Overview

The AQUA system follows a microservices architecture pattern with the following components:

```
┌─────────────────┐    ┌──────────────┐    ┌────────────────┐
│   Web Client    │    │   Nginx      │    │   API-SIG      │
│  (Dashboard)    │◄──►│  (Reverse    │◄──►│   (Express)    │
└─────────────────┘    │   Proxy)     │    └────────────────┘
                       └──────────────┘             │
                                                    ▼
┌─────────────────┐                     ┌──────────────────────┐
│   Capteurs      │                     │   TimescaleDB        │
│   (Node.js)     │◄───────────────────►│   (Time-series DB)   │
└─────────────────┘     MQTT/HTTP       └──────────────────────┘
         │                                        │
         ▼                                        ▼
┌─────────────────┐                     ┌──────────────────────┐
│   MQTT Broker   │                     │   PostgreSQL         │
│  (Mosquitto)    │                     │   (Alert Storage)    │
└─────────────────┘                     └──────────────────────┘
                                                    │
┌─────────────────┐                                  ▼
│   Satellite     │                     ┌──────────────────────┐
│   (Python)      │◄───────────────────►│   MinIO              │
└─────────────────┘    GeoTIFF Files   │   (Object Storage)   │
                                                    │
┌─────────────────┐                                  ▼
│   STModel       │                     ┌──────────────────────┐
│   (Python)      │◄───────────────────►│   Alertes            │
└─────────────────┘    Predictions     │   (Node.js)          │
                                       └──────────────────────┘
```

## 1. Capteurs Service (Sensor Data Generator)

### Technology Stack
- **Runtime**: Node.js 16+
- **Database**: TimescaleDB (PostgreSQL extension)
- **Messaging**: MQTT (Eclipse Mosquitto)
- **Dependencies**: pg, mqtt

### Core Functionality
The Capteurs service generates simulated sensor data representing water quality measurements across multiple geographic zones.

#### Data Generation Logic
```javascript
// Sensor locations in Rabat-Salé region
const capteurLocations = {
  'CAPT-1': { lat: 34.0209, lon: -6.8416, zone: 'Rabat-Centre' },
  'CAPT-2': { lat: 34.0133, lon: -6.8326, zone: 'Rabat-Centre' },
  // ... 13 more sensors across 8 zones
};

// Data generation with realistic variations
function generateSensorData() {
  // Select random sensor
  const capteurId = getRandomSensor();
  const location = capteurLocations[capteurId];
  
  // Base values per zone
  let phBase = 7.0;
  let turbiditeBase = 1.0;
  let tempBase = 20.0;
  
  // Zone-specific adjustments
  if (location.zone.includes('Océan')) {
    phBase = 8.0; // Higher pH for seawater
    turbiditeBase = 0.5;
  }
  
  // Occasional anomalies (10% chance)
  if (Math.random() < 0.10) {
    // Generate out-of-range values to trigger alerts
    ph = phBase + (Math.random() < 0.5 ? -2.0 : 3.0);
  }
  
  return {
    timestamp: new Date().toISOString(),
    ph: Math.max(5.5, Math.min(9.5, ph)),
    turbidite: Math.max(0.1, Math.min(8.0, turbidite)),
    temperature: Math.max(12.0, Math.min(35.0, temperature)),
    capteur_id: capteurId,
    latitude: location.lat,
    longitude: location.lon,
    zone: location.zone
  };
}
```

### Database Schema
```sql
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

-- Create TimescaleDB hypertable for time-series optimization
SELECT create_hypertable('donnees_capteurs', 'timestamp', if_not_exists => TRUE);

-- Index for efficient querying
CREATE INDEX IF NOT EXISTS idx_capteur_id ON donnees_capteurs(capteur_id);
```

### Configuration
Environment variables:
- `TIMESCALEDB_HOST`: Database hostname (default: timescaledb)
- `TIMESCALEDB_PORT`: Database port (default: 5432)
- `TIMESCALEDB_DB`: Database name (default: aquawatch)
- `TIMESCALEDB_USER`: Database username (default: postgres)
- `TIMESCALEDB_PASSWORD`: Database password (default: postgres)
- `MQTT_BROKER`: MQTT broker hostname (default: mqtt-broker)
- `MQTT_PORT`: MQTT broker port (default: 1883)

## 2. API-SIG Service (REST API)

### Technology Stack
- **Framework**: Express.js
- **Database**: TimescaleDB, PostgreSQL
- **Dependencies**: express, pg, cors

### API Endpoints

#### Health Check
```
GET /health
Response: { "status": "OK", "service": "API-SIG" }
```

#### Sensor Data
```
GET /api/capteurs
Query Parameters:
- limit: Number of records to return (default: 100)

Response Format: GeoJSON FeatureCollection
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "id": 12345,
        "timestamp": "2023-01-01T12:00:00Z",
        "capteur_id": "CAPT-1",
        "zone": "Rabat-Centre",
        "ph": 7.2,
        "turbidite": 0.8,
        "temperature": 22.5
      },
      "geometry": {
        "type": "Point",
        "coordinates": [-6.8416, 34.0209]
      }
    }
  ]
}
```

#### Satellite Data
```
GET /api/satellite
Query Parameters:
- limit: Number of records to return (default: 100)

Response Format: GeoJSON FeatureCollection
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "id": 67890,
        "timestamp": "2023-01-01T12:00:00Z",
        "satellite_id": "SAT-RABAT",
        "zone": "Rabat-Centre",
        "chlorophylle": 12.5,
        "turbidite": 0.6,
        "ndwi": 0.45,
        "storage_url": "http://minio:9000/aquawatch-satellite/rabat-centre/20230101T120000.tif",
        "source": "Sentinel-2"
      },
      "geometry": {
        "type": "Point",
        "coordinates": [-6.8416, 34.0209]
      }
    }
  ]
}
```

#### Predictions
```
GET /api/predictions
Query Parameters:
- limit: Number of records to return (default: 100)

Response:
[
  {
    "id": 11111,
    "timestamp": "2023-01-02T12:00:00Z",
    "zone_id": "Rabat-Centre",
    "qualite_score": 85.5,
    "qualite_niveau": "Excellente",
    "risque_score": 12.3,
    "risque_niveau": "Faible",
    "prediction_score": 78.9,
    "ph_pred": 7.3,
    "turbidite_pred": 0.7,
    "temperature_pred": 21.8,
    "prediction_horizon": 24,
    "confidence": 88.5
  }
]
```

#### Alerts
```
GET /api/alertes
Query Parameters:
- limit: Number of records to return (default: 50)
- severity: Filter by severity (CRITICAL, WARNING, INFO)

Response:
[
  {
    "id": 22222,
    "timestamp": "2023-01-01T12:05:00Z",
    "type": "SEUIL_DEPASSE",
    "severity": "WARNING",
    "zone": "Rabat-Centre",
    "zone_geographique": "Rabat-Centre",
    "capteur_id": "CAPT-3",
    "parametre": "ph",
    "valeur": 8.7,
    "seuil_oms": 8.5,
    "type_polluant": "Acidité/Alcalinité",
    "population_exposee": 12000,
    "message": "⚠️ pH hors norme dans Rabat-Centre: 8.70 (seuil OMS: 6.5-8.5). Surveillance renforcée recommandée.",
    "status": "ACTIVE"
  }
]
```

#### Statistics
```
GET /api/stats

Response:
{
  "capteurs": {
    "capteurs_uniques": 15,
    "total_mesures": 10500,
    "ph_moyen": "7.27",
    "turbidite_moyenne": "1.74"
  },
  "satellite": {
    "total": 27,
    "chlorophylle_moyenne": "11.38",
    "ndwi_moyen": "0.23"
  },
  "predictions": {
    "total": 528,
    "qualite_moyenne": "68.45"
  },
  "alertes": {
    "total": 2581
  }
}
```

## 3. Alertes Service (Alert Management)

### Technology Stack
- **Runtime**: Node.js 16+
- **Database**: PostgreSQL
- **Dependencies**: pg, nodemailer

### Alert Generation Logic
The Alertes service continuously monitors sensor data against WHO standards and generates alerts when thresholds are exceeded.

#### WHO Thresholds Implementation
```javascript
// WHO water quality standards
const OMS_SEUILS = {
  ph: { min: 6.5, max: 8.5, critical: { min: 6.0, max: 9.0 }, polluant: 'Acidité/Alcalinité' },
  turbidite: { max: 1.0, critical: 5.0, polluant: 'Matières en suspension' }, // NTU
  temperature: { max: 25.0, critical: 30.0, polluant: 'Température' }, // °C
  chlorophylle: { max: 10.0, critical: 20.0, polluant: 'Eutrophisation' }, // mg/m³
};

const SATELLITE_SEUILS = {
  chlorophylle: { warning: 10, critical: 20, polluant: 'Eutrophisation' },
  turbidite: { warning: 4, critical: 6, polluant: 'Turbidité côtière' },
  ndwi: { min: 0.05, polluant: 'Stress hydrique' },
};
```

#### Alert Deduplication
To prevent alert flooding, the system implements intelligent deduplication:
```javascript
// Avoid duplicate alerts within 5-minute windows
const recentAlertsQuery = `
  SELECT capteur_id, parametre, severity
  FROM alertes
  WHERE timestamp > NOW() - INTERVAL '5 minutes'
  AND status = 'ACTIVE'
  AND capteur_id IS NOT NULL
`;
```

### Database Schema
```sql
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

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_alertes_timestamp ON alertes(timestamp);
CREATE INDEX IF NOT EXISTS idx_alertes_status ON alertes(status);
CREATE INDEX IF NOT EXISTS idx_alertes_severity ON alertes(severity);
```

## 4. Satellite Service (Earth Observation)

### Technology Stack
- **Runtime**: Python 3.8+
- **Libraries**: numpy, rasterio, psycopg2, minio, sentinelhub
- **Storage**: MinIO (S3-compatible)

### Core Functionality
The Satellite service processes Earth observation data to derive water quality indicators.

#### SentinelHub Integration
```python
# Evaluation script for deriving water quality indices
EVAL_SCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B03", "B04", "B08"],  // Blue, Green, NIR bands
      units: "REFLECTANCE"
    }],
    output: { bands: 3, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  // NDWI calculation (McFeeters, 1996)
  var ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08 + 0.0000001);
  
  // Turbidity estimation from red/green ratio
  var turbidity = sample.B04 / (sample.B03 + 0.0000001);
  
  // Chlorophyll-a approximation
  var chlorophyll = sample.B08 * 30.0;
  
  return [ndwi, turbidity, chlorophyll];
}
"""
```

#### Data Processing Pipeline
1. Request satellite imagery from SentinelHub
2. Apply evaluation script to derive indices
3. Calculate mean values for region of interest
4. Store metadata in TimescaleDB
5. Upload processed GeoTIFF to MinIO

### Database Schema
```sql
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

-- Create TimescaleDB hypertable
SELECT create_hypertable('donnees_satellite', 'timestamp', if_not_exists => TRUE);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_satellite_id ON donnees_satellite(satellite_id);
CREATE INDEX IF NOT EXISTS idx_satellite_timestamp ON donnees_satellite(timestamp DESC);
```

## 5. STModel Service (Spatio-Temporal Modeling)

### Technology Stack
- **Runtime**: Python 3.8+
- **ML Framework**: PyTorch
- **Libraries**: numpy, pandas, scikit-learn, matplotlib

### Model Architecture
The service implements a Convolutional LSTM (ConvLSTM) neural network for spatio-temporal forecasting of water quality.

#### Neural Network Structure
```python
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

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(num_layers):
            self.layers.append(ConvLSTMCell(dims[i], dims[i + 1], kernel_size))

class SpatioTemporalForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlstm = ConvLSTM(
            input_dim=3,        # pH, turbidity, temperature
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
            nn.Linear(32, 3),   # Output: quality, risk, confidence
        )
```

#### Data Preprocessing
```python
def build_sequences(rows):
    """
    Convert time-series sensor data into spatial grids for ConvLSTM input
    """
    # Grid definition for Rabat-Salé region
    GRID_SHAPE = (4, 4)
    
    # Normalize values to [0,1] range
    def normalize(ph, turbidity, temperature):
        ph_n = np.clip((ph - 5.5) / 4.0, 0, 1)
        turb_n = np.clip(turbidity / 8.0, 0, 1)
        temp_n = np.clip((temperature - 10) / 25.0, 0, 1)
        return ph_n, turb_n, temp_n
    
    # Map geographic coordinates to grid cells
    def grid_cell(lat, lon):
        lat_min, lat_max = 33.80, 34.10
        lon_min, lon_max = -7.10, -6.70
        row = int(((lat - lat_min) / (lat_max - lat_min)) * GRID_SHAPE[0])
        col = int(((lon - lon_min) / (lon_max - lon_min)) * GRID_SHAPE[1])
        return max(0, min(GRID_SHAPE[0] - 1, row)), max(0, min(GRID_SHAPE[1] - 1, col))
```

### Database Schema
```sql
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

-- Create TimescaleDB hypertable
SELECT create_hypertable('predictions_qualite', 'timestamp', if_not_exists => TRUE);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_zone_id ON predictions_qualite(zone_id);
CREATE INDEX IF NOT EXISTS idx_prediction_horizon ON predictions_qualite(prediction_horizon);
```

## 6. Web Interface (Dashboard)

### Technology Stack
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Mapping**: Leaflet.js
- **Charts**: Chart.js
- **Proxy**: Nginx

### Core Components

#### Data Visualization
The dashboard presents data through multiple visualization components:

1. **Interactive Map**: Displays sensor locations with color-coded status
2. **Statistics Cards**: Shows key metrics at a glance
3. **Alerts Panel**: Lists recent alerts with filtering capabilities
4. **Satellite Data**: Displays satellite observations
5. **Predictions Panel**: Shows model forecasts
6. **Time Series Chart**: Plots pH evolution over time

#### API Integration
All data is fetched asynchronously from the API-SIG service:

```
// Fetch statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        // Update DOM elements
        document.getElementById('stat-capteurs').textContent = data.capteurs?.capteurs_uniques || 0;
        document.getElementById('stat-ph').textContent = data.capteurs?.ph_moyen || '-';
        // ... update other stats
    } catch (error) {
        console.error('Erreur chargement stats:', error);
    }
}

// Refresh all data periodically
setInterval(refreshData, 10000); // Every 10 seconds
```

#### Responsive Design
The interface adapts to different screen sizes:
- Desktop: Multi-column layout with full feature set
- Tablet: Adjusted grid layouts
- Mobile: Single column with collapsible sections

## Infrastructure Components

### Docker Compose Configuration
The system uses Docker Compose for orchestration with proper health checks and dependencies:

```yaml
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
  
  api-sig:
    build: ./api-sig
    depends_on:
      timescaledb:
        condition: service_healthy
      postgres:
        condition: service_healthy
```

### Nginx Configuration
The web service uses Nginx as a reverse proxy with dynamic DNS resolution:

```nginx
http {
    # Dynamic DNS resolution for container networking
    resolver 127.0.0.11 ipv6=off valid=30s;
    set $api_upstream http://api-sig:3000;
    
    server {
        listen 80;
        
        # Serve static files
        location / {
            root /usr/share/nginx/html;
            index index.html;
            try_files $uri $uri/ =404;
        }
        
        # Proxy API requests
        location /api/ {
            proxy_pass $api_upstream$request_uri;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## Data Flow and Integration Points

### 1. Sensor Data Flow
```
Capteurs Service
    ↓ (every 5s)
TimescaleDB (donnees_capteurs)
    ↓ (MQTT)
Mosquitto Broker
    ↓ (API)
API-SIG Service
    ↓ (HTTP)
Web Dashboard
    ↓ (periodic polling)
Alertes Service (threshold monitoring)
```

### 2. Satellite Data Flow
```
Satellite Service
    ↓ (SentinelHub/Simulation)
Processing Pipeline
    ↓
TimescaleDB (donnees_satellite)
    ↓
MinIO (GeoTIFF storage)
    ↓
API-SIG Service
    ↓
Web Dashboard
```

### 3. Prediction Flow
```
STModel Service
    ↓ (model inference)
TimescaleDB (predictions_qualite)
    ↓
API-SIG Service
    ↓
Web Dashboard
```

### 4. Alert Flow
```
Alertes Service
    ↓ (threshold checking)
PostgreSQL (alertes)
    ↓
API-SIG Service
    ↓
Web Dashboard
```

## Performance Considerations

### Database Optimization
1. **TimescaleDB Hypertables**: Time-series data partitioned by time
2. **Indexing Strategy**: Strategic indexes on frequently queried columns
3. **Connection Pooling**: Reused database connections in services

### Caching Strategy
1. **Browser Caching**: Static assets cached by web browsers
2. **API Response Caching**: Potential for Redis caching layer
3. **Database Query Optimization**: Efficient queries with proper WHERE clauses

### Scalability
1. **Horizontal Scaling**: Services can be scaled independently
2. **Load Balancing**: Multiple instances behind reverse proxy
3. **Database Replication**: Read replicas for heavy query loads

## Security Considerations

### Current State
- No authentication/authorization implemented
- Services communicate over internal Docker network
- Database credentials stored as environment variables

### Recommended Enhancements
1. **API Authentication**: JWT-based authentication
2. **Rate Limiting**: Prevent API abuse
3. **Input Validation**: Sanitize all API inputs
4. **HTTPS**: TLS encryption for all external communications
5. **Database Security**: Role-based access controls

## Monitoring and Observability

### Current Implementation
- Docker health checks for all services
- Basic logging in each service
- API health endpoints

### Recommended Enhancements
1. **Centralized Logging**: ELK stack or similar
2. **Metrics Collection**: Prometheus integration
3. **Application Performance Monitoring**: APM tools
4. **Alerting**: Automated notifications for service failures
5. **Tracing**: Distributed tracing for complex requests

## Backup and Recovery

### Data Protection
- **Database Backups**: Regular dumps of PostgreSQL/TimescaleDB
- **Object Storage**: MinIO data replication
- **Configuration**: Version-controlled Docker Compose files

### Recovery Procedures
1. **Database Restoration**: Restore from SQL dump files
2. **Service Restart**: Docker Compose restart procedures
3. **Data Validation**: Verify data integrity after restoration

## Testing Strategy

### Current State
- No automated test suites implemented
- Manual verification of functionality

### Recommended Implementation
1. **Unit Tests**: Test individual functions and components
2. **Integration Tests**: Verify service interactions
3. **End-to-End Tests**: Validate complete workflows
4. **Performance Tests**: Load testing for scalability
5. **Security Tests**: Vulnerability scanning and penetration testing

## Deployment Guidelines

### Production Considerations
1. **Environment Separation**: Dev/Staging/Prod environments
2. **Secrets Management**: Secure credential storage
3. **Resource Allocation**: Proper CPU/RAM allocation
4. **Network Security**: Firewall rules and network segmentation
5. **Disaster Recovery**: Backup and restore procedures

### Continuous Integration/Deployment
1. **CI Pipeline**: Automated testing on code changes
2. **CD Pipeline**: Automated deployment to staging/production
3. **Version Control**: Git workflow with branching strategy
4. **Rollback Procedures**: Quick rollback capability for failed deployments
