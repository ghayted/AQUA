pipeline {
    agent any
    
    environment {
        COMPOSE_PROJECT_NAME = "aqua_ci_${BUILD_NUMBER}"
        // CHANGEMENT CRITIQUE : On utilise UNIQUEMENT le fichier CI pour éviter toute fusion de ports indésirable
        DOCKER_COMPOSE_CMD = "docker-compose -f docker-compose.ci.yml"
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo '📥 Récupération du code...'
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: '*/main']],
                    extensions: [[$class: 'CloneOption', depth: 1, shallow: true, timeout: 30, noTags: true]],
                    userRemoteConfigs: [[url: 'https://github.com/ghayted/AQUA.git']]
                ])
            }
        }
        
        stage('Environment Setup') {
            steps {
                echo '🔧 Génération Config UTRA-ISOLÉE...'
                // On recrée TOUT le fichier docker-compose pour être sûr qu'il n'y a AUCUN lien avec tes ports locaux
                writeFile file: 'docker-compose.ci.yml', text: """
version: '3.8'
services:
  eureka:
    image: steeltoeoss/eureka-server:latest
    container_name: ci-eureka-${BUILD_NUMBER}
    ports: ["18761:8761"]
    healthcheck:
      test: ["CMD", "wget", "-q", "-O", "-", "http://localhost:8761/actuator/health"]
      interval: 10s
      timeout: 5s
  
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    container_name: ci-timescaledb-${BUILD_NUMBER}
    environment: {POSTGRES_DB: aquawatch, POSTGRES_USER: postgres, POSTGRES_PASSWORD: postgres}
    ports: ["15433:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      
  postgres:
    image: postgres:15-alpine
    container_name: ci-postgres-${BUILD_NUMBER}
    environment: {POSTGRES_DB: alertes, POSTGRES_USER: postgres, POSTGRES_PASSWORD: postgres}
    ports: ["15434:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s

  mqtt-broker:
    image: eclipse-mosquitto:2.0
    container_name: ci-mqtt-${BUILD_NUMBER}
    ports: ["11883:1883", "19003:9001"]
    volumes:
      - ./mqtt/config:/mosquitto/config
    healthcheck:
       test: ["CMD-SHELL", "exit 0"]
       interval: 30s
       
  geoserver:
    image: kartoza/geoserver:latest
    container_name: ci-geoserver-${BUILD_NUMBER}
    ports: ["18082:8080"]
    healthcheck:
      test: ["CMD-SHELL", "exit 0"]

  minio:
    image: minio/minio:latest
    container_name: ci-minio-${BUILD_NUMBER}
    ports: ["19000:9000", "19002:9001"]
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD-SHELL", "exit 0"]
  
  capteurs:
    build: ./capteurs
    container_name: ci-capteurs-${BUILD_NUMBER}
    environment: {TIMESCALEDB_HOST: timescaledb, MQTT_BROKER: mqtt-broker, EUREKA_HOST: eureka, HEALTH_PORT: 3001}
    ports: ["13001:3001"]
    depends_on: [timescaledb, mqtt-broker, eureka]
    
  alertes:
    build: ./alertes
    container_name: ci-alertes-${BUILD_NUMBER}
    environment: {POSTGRES_HOST: postgres, TIMESCALEDB_HOST: timescaledb, EUREKA_HOST: eureka, HEALTH_PORT: 3002}
    ports: ["13002:3002"]
    depends_on: [postgres, timescaledb, eureka]
    
  satellite:
    build: ./satellite
    container_name: ci-satellite-${BUILD_NUMBER}
    environment: {TIMESCALEDB_HOST: timescaledb, MINIO_ENDPOINT: minio, EUREKA_HOST: eureka}
    depends_on: [timescaledb, minio, eureka]
    
  stmodel:
    build: ./stmodel
    container_name: ci-stmodel-${BUILD_NUMBER}
    environment: {TIMESCALEDB_HOST: timescaledb, EUREKA_HOST: eureka}
    depends_on: [timescaledb, eureka]
    
  api-sig:
    build: ./api-sig
    container_name: ci-api-sig-${BUILD_NUMBER}
    environment: {POSTGIS_HOST: timescaledb, POSTGRES_HOST: postgres, EUREKA_HOST: eureka, PORT: 3000}
    ports: ["13000:3000"]
    depends_on: [timescaledb, postgres, eureka]
    
  web:
    image: nginx:alpine
    container_name: ci-web-${BUILD_NUMBER}
    ports: ["10080:80"]
    volumes:
      - ./web:/usr/share/nginx/html
    
  jenkins: { profiles: ["donotstart"] }
"""
            }
        }
        
        stage('Infrastructure') {
            steps {
                echo '🗄️ Démarrage Infrastructure...'
                sh "${DOCKER_COMPOSE_CMD} up -d timescaledb postgres mqtt-broker geoserver minio eureka"
                sh 'sleep 15'
            }
        }
        
        stage('Capteurs') { steps { sh "${DOCKER_COMPOSE_CMD} up -d --build capteurs" } }
        stage('Satellite') { steps { sh "${DOCKER_COMPOSE_CMD} up -d --build satellite" } }
        stage('STModel') { steps { sh "${DOCKER_COMPOSE_CMD} up -d --build stmodel" } }
        stage('Alertes') { steps { sh "${DOCKER_COMPOSE_CMD} up -d --build alertes" } }
        stage('API-SIG') { steps { sh "${DOCKER_COMPOSE_CMD} up -d --build api-sig" } }
        stage('Web') { steps { sh "${DOCKER_COMPOSE_CMD} up -d --build web" ; sh 'sleep 10' } }
        
        stage('Health Checks') {
            steps {
                sh 'curl -f http://host.docker.internal:13000/health || exit 1'
            }
        }
    }
    
    post {
        always {
            sh "${DOCKER_COMPOSE_CMD} down -v"
        }
    }
}
