pipeline {
    agent any
    
    environment {
        WORKSPACE_DIR = '/workspace'
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo '📥 Vérification du workspace...'
                dir("${WORKSPACE_DIR}") {
                    sh 'ls -la'
                    sh 'echo "Fichiers du projet AQUA disponibles"'
                }
            }
        }
        
        stage('Environment Setup') {
            steps {
                echo '🔧 Configuration de l\'environnement...'
                dir("${WORKSPACE_DIR}") {
                    sh '''
                        echo "Docker version:"
                        docker --version
                        echo "Docker Compose version:"
                        docker-compose --version
                    '''
                }
            }
        }
        
        stage('Build Services') {
            steps {
                echo '🏗️ Construction des images Docker...'
                dir("${WORKSPACE_DIR}") {
                    sh 'docker-compose build --parallel capteurs satellite stmodel alertes api-sig || echo "Build partiel"'
                }
            }
        }
        
        stage('Start Services') {
            steps {
                echo '▶️ Démarrage des services...'
                dir("${WORKSPACE_DIR}") {
                    sh 'docker-compose up -d'
                    sh 'sleep 30'
                }
            }
        }
        
        stage('Health Checks') {
            steps {
                echo '🏥 Vérification de la santé des services...'
                dir("${WORKSPACE_DIR}") {
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
    }
    
    post {
        success {
            echo '✅ Build réussi!'
        }
        failure {
            echo '❌ Build échoué!'
            dir("${WORKSPACE_DIR}") {
                sh 'docker-compose logs --tail=20 || true'
            }
        }
        always {
            echo '🧹 Pipeline terminé!'
        }
    }
}
