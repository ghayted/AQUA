#!/bin/bash

# Script de test automatisé pour les services AQUA
# Ce script vérifie la santé de tous les services

set -e

echo "========================================="
echo "🧪 Tests d'intégration AQUA"
echo "========================================="

# Couleurs pour l'output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour tester un service
test_service() {
    local service_name=$1
    local container_name=$2
    
    echo -e "\n${YELLOW}Testing ${service_name}...${NC}"
    
    # Vérifier si le conteneur est en cours d'exécution
    if docker ps | grep -q "$container_name"; then
        echo -e "${GREEN}✓ Container ${container_name} is running${NC}"
        
        # Vérifier les logs pour des erreurs
        if docker logs "$container_name" --tail 50 2>&1 | grep -i "error" | grep -v "0 error"; then
            echo -e "${YELLOW}⚠ Warning: Errors found in logs${NC}"
        else
            echo -e "${GREEN}✓ No critical errors in logs${NC}"
        fi
        
        return 0
    else
        echo -e "${RED}✗ Container ${container_name} is not running${NC}"
        return 1
    fi
}

# Fonction pour tester un endpoint HTTP
test_endpoint() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}
    
    echo -e "\n${YELLOW}Testing endpoint ${name}...${NC}"
    
    response_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" || echo "000")
    
    if [ "$response_code" = "$expected_code" ]; then
        echo -e "${GREEN}✓ ${name} returned ${response_code}${NC}"
        return 0
    else
        echo -e "${RED}✗ ${name} returned ${response_code} (expected ${expected_code})${NC}"
        return 1
    fi
}

# Fonction pour tester une connexion à une base de données
test_database() {
    local db_name=$1
    local container_name=$2
    local db_user=${3:-postgres}
    
    echo -e "\n${YELLOW}Testing database ${db_name}...${NC}"
    
    if docker exec "$container_name" pg_isready -U "$db_user" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Database ${db_name} is ready${NC}"
        return 0
    else
        echo -e "${RED}✗ Database ${db_name} is not ready${NC}"
        return 1
    fi
}

# Compteurs de tests
TESTS_PASSED=0
TESTS_FAILED=0

echo -e "\n${YELLOW}=== Testing Databases ===${NC}"

# Test TimescaleDB
if test_database "TimescaleDB" "aquawatch-timescaledb"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

# Test PostgreSQL
if test_database "PostgreSQL" "aquawatch-postgres"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

echo -e "\n${YELLOW}=== Testing Infrastructure Services ===${NC}"

# Test MinIO
if test_service "MinIO" "aquawatch-minio"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

# Test MQTT Broker
if test_service "MQTT Broker" "aquawatch-mqtt"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

# Test GeoServer
if test_service "GeoServer" "aquawatch-geoserver"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

echo -e "\n${YELLOW}=== Testing Application Services ===${NC}"

# Test Capteurs
if test_service "Capteurs" "aquawatch-capteurs"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

# Test Satellite
if test_service "Satellite" "aquawatch-satellite"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

# Test STModel
if test_service "STModel" "aquawatch-stmodel"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

# Test Alertes
if test_service "Alertes" "aquawatch-alertes"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

# Test API-SIG
if test_service "API-SIG" "aquawatch-api-sig"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

# Test Web
if test_service "Web" "aquawatch-web"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

echo -e "\n${YELLOW}=== Testing HTTP Endpoints ===${NC}"

# Attendre un peu pour que les services soient prêts
sleep 5

# Test API Health
if test_endpoint "API Health" "http://localhost:3000/health"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

# Test Web Interface
if test_endpoint "Web Interface" "http://localhost:80"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

# Test MinIO Console
if test_endpoint "MinIO Console" "http://localhost:9002"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

# Test GeoServer
if test_endpoint "GeoServer" "http://localhost:8080/geoserver/web"; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

# Résumé des tests
echo -e "\n========================================="
echo -e "${YELLOW}Test Summary${NC}"
echo -e "========================================="
echo -e "${GREEN}Passed: ${TESTS_PASSED}${NC}"
echo -e "${RED}Failed: ${TESTS_FAILED}${NC}"
echo -e "Total: $((TESTS_PASSED + TESTS_FAILED))"

# Afficher les conteneurs en cours d'exécution
echo -e "\n${YELLOW}=== Running Containers ===${NC}"
docker-compose ps

# Code de sortie basé sur les résultats
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Some tests failed!${NC}"
    exit 1
fi
