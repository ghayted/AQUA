FROM jenkins/jenkins:lts

# Passer en mode root pour installer les dépendances
USER root

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Installer Docker CLI
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt-get update && apt-get install -y docker-ce-cli && rm -rf /var/lib/apt/lists/*

# Installer Docker Compose
RUN curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
RUN chmod +x /usr/local/bin/docker-compose

# Installer Node.js (pour les tests des services Node.js)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

# Installer les outils Python pour les tests via apt
RUN apt-get update && apt-get install -y python3-pytest python3-pytest-cov && rm -rf /var/lib/apt/lists/*

# Ajouter l'utilisateur jenkins au groupe docker
RUN groupadd -f docker && usermod -aG docker jenkins

# Installer les plugins Jenkins recommandés
RUN jenkins-plugin-cli --plugins \
    git \
    workflow-aggregator \
    docker-workflow \
    pipeline-stage-view \
    blueocean \
    credentials-binding \
    timestamper \
    ws-cleanup \
    build-timeout \
    github-branch-source \
    pipeline-github-lib

# Créer les répertoires nécessaires
RUN mkdir -p /var/jenkins_home/workspace
RUN chown -R jenkins:jenkins /var/jenkins_home

# Revenir à l'utilisateur jenkins
USER jenkins

# Exposer les ports
EXPOSE 8080 50000

# Point d'entrée par défaut
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/jenkins.sh"]
