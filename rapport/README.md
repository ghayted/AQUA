# Rapport LaTeX AquaWatch

## Structure du dossier

```
rapport/
├── main.tex          # Fichier principal du rapport
├── images/           # Dossier pour les screenshots
│   ├── architecture.png
│   ├── dashboard.png
│   ├── map.png
│   ├── predictions.png
│   ├── alerts.png
│   └── convlstm_architecture.png
└── README.md         # Ce fichier
```

## Instructions

### 1. Ajouter les images

Prenez des screenshots de l'interface web et placez-les dans le dossier `images/` :

1. **dashboard.png** : Page d'accueil du dashboard
2. **map.png** : Carte interactive avec les zones
3. **predictions.png** : Page des prévisions IA
4. **alerts.png** : Page des alertes
5. **architecture.png** : Diagramme d'architecture (optionnel)

### 2. Modifier les informations

Dans `main.tex`, remplacez les placeholders :

- `[NOM DE L'ÉTABLISSEMENT]` → Votre établissement
- `[Département / Filière]` → Votre filière
- `[NOM Prénom 1]`, `[NOM Prénom 2]`, etc. → Noms des étudiants
- `[NOM DE L'ENCADRANT]` → Nom de votre encadrant

### 3. Activer les images

Décommentez les lignes `\includegraphics` dans le fichier et supprimez les `\fbox` placeholders.

Exemple :
```latex
% Avant (placeholder)
\fbox{\parbox{0.9\textwidth}{\centering\vspace{4cm}[Screenshot Dashboard]\vspace{4cm}}}

% Après (image réelle)
\includegraphics[width=0.95\textwidth]{images/dashboard.png}
```

### 4. Compiler le rapport

Avec pdfLaTeX :
```bash
pdflatex main.tex
pdflatex main.tex  # Deux fois pour la table des matières
```

Ou utilisez un éditeur LaTeX comme :
- **Overleaf** (en ligne)
- **TeXstudio** (local)
- **VS Code** avec l'extension LaTeX Workshop

## Technologies mentionnées

- Docker & Docker Compose
- TimescaleDB / PostgreSQL
- MQTT (Eclipse Mosquitto)
- Python / PyTorch (ConvLSTM)
- Node.js / Express.js
- HTML / CSS / JavaScript
- Leaflet (cartes)
