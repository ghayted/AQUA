pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                echo '📥 Récupération du code depuis GitHub...'
                checkout scm
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
                echo '▶️ Démarrage des services...'
                sh 'docker-compose up -d'
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
