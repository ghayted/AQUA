pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                echo '📥 Récupération du code depuis GitHub...'
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
        
        stage('Infrastructure') {
            steps {
                echo '🗄️ Démarrage de l\'infrastructure...'
                sh 'docker-compose up -d timescaledb postgres mqtt geoserver minio'
                sh 'sleep 10'
            }
        }
        
        stage('📡 Capteurs') {
            steps {
                echo '📡 Build & Start Capteurs...'
                sh 'docker-compose build capteurs'
                sh 'docker-compose up -d capteurs'
            }
        }
        
        stage('🛰️ Satellite') {
            steps {
                echo '🛰️ Build & Start Satellite...'
                sh 'docker-compose build satellite'
                sh 'docker-compose up -d satellite'
            }
        }
        
        stage('🤖 STModel') {
            steps {
                echo '🤖 Build & Start STModel (ML)...'
                sh 'docker-compose build stmodel'
                sh 'docker-compose up -d stmodel'
            }
        }
        
        stage('🚨 Alertes') {
            steps {
                echo '🚨 Build & Start Alertes...'
                sh 'docker-compose build alertes'
                sh 'docker-compose up -d alertes'
            }
        }
        
        stage('🗺️ API-SIG') {
            steps {
                echo '🗺️ Build & Start API-SIG...'
                sh 'docker-compose build api-sig'
                sh 'docker-compose up -d api-sig'
            }
        }
        
        stage('🌐 Web') {
            steps {
                echo '🌐 Démarrage du Frontend Web...'
                sh 'docker-compose up -d web'
                sh 'sleep 15'
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
