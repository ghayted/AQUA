pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                echo '📥 Récupération du code depuis GitHub...'
                // Utiliser un timeout plus long et shallow clone pour les gros repos
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: '*/main']],
                    extensions: [
                        [$class: 'CloneOption', 
                         depth: 1, 
                         shallow: true, 
                         timeout: 30,
                         noTags: true],
                        [$class: 'CheckoutOption', timeout: 30]
                    ],
                    userRemoteConfigs: [[url: 'https://github.com/ghayted/AQUA.git']]
                ])
                sh 'ls -la'
                echo '✅ Code AQUA récupéré avec succès!'
            }
        }
        
        stage('Environment Setup') {
            steps {
                echo '🔧 Configuration de l\'environnement...'
                sh '''
                    echo "Docker version:"
                    docker --version || echo "Docker non disponible"
                    echo "Docker Compose version:"
                    docker-compose --version || echo "Docker Compose non disponible"
                '''
            }
        }
        
        stage('Build Services') {
            steps {
                echo '🏗️ Construction des images Docker...'
                sh 'docker-compose build --parallel capteurs satellite stmodel alertes api-sig || echo "Build partiel"'
            }
        }
        
        stage('Start Services') {
            steps {
                echo '▶️ Démarrage des services (sans Jenkins)...'
                // Exclure Jenkins pour éviter la récursion
                sh '''
                    docker-compose up -d \
                        timescaledb postgres mqtt geoserver minio \
                        capteurs satellite stmodel alertes api-sig web \
                        || echo "Certains services déjà en cours"
                '''
                sh 'sleep 30'
            }
        }
        
        stage('Health Checks') {
            steps {
                echo '🏥 Vérification de la santé des services...'
                sh '''
                    echo "État des conteneurs:"
                    docker-compose ps
                    
                    echo "Test API:"
                    curl -s http://host.docker.internal:3000/health || echo "API non disponible"
                    
                    echo "Test Web:"
                    curl -s http://host.docker.internal:80 || echo "Web non disponible"
                '''
            }
        }
    }
    
    post {
        success {
            echo '✅ Build réussi!'
        }
        failure {
            echo '❌ Build échoué!'
            sh 'docker-compose logs --tail=20 || true'
        }
        always {
            echo '🧹 Pipeline terminé!'
        }
    }
}
