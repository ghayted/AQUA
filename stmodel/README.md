# 🧠 Module Machine Learning - AquaWatch

## Prédiction de la Qualité de l'Eau

**EMSI Marrakech - 2025-2026**

---

## 🎯 Objectif Simple

> **Le modèle ML prédit la qualité de l'eau pour les 24 prochaines heures.**

On veut savoir à l'avance si l'eau sera bonne ou mauvaise demain, pour agir avant qu'il y ait un problème.

---

## 📊 Les Données

### Ce qu'on mesure (3 paramètres)

| Paramètre | C'est quoi ? | Valeur normale |
|-----------|--------------|----------------|
| **pH** | Acidité de l'eau | Entre 6.5 et 8.5 |
| **Turbidité** | Eau claire ou trouble | < 1 NTU (très claire) |
| **Température** | Chaud ou froid | < 25°C |

### D'où viennent les données ?

- **16 capteurs IoT** placés dans 10 zones (Rabat, Salé, Marrakech...)
- Chaque capteur mesure pH, turbidité, température
- Les données sont stockées dans **TimescaleDB** (base de données)

---

## 🤖 Le Modèle : ConvLSTM

### Pourquoi ce modèle ?

On a choisi **ConvLSTM** car il combine 2 choses :

| Partie | Rôle |
|--------|------|
| **Conv** (Convolution) | Comprend les relations entre zones géographiques |
| **LSTM** (Mémoire) | Se souvient du passé pour prédire le futur |

### Comment ça marche ? (Version simple)

```
1. ENTRÉE : Les 12 dernières mesures de chaque zone
      ↓
2. LE MODÈLE : Analyse les patterns (tendances)
      ↓  
3. SORTIE : Prédiction pour chaque heure de demain
```

**En résumé** : Le modèle regarde le **passé** (12 mesures) pour prédire le **futur** (24 heures).

---

## 🏋️ L'Entraînement

### C'est quoi l'entraînement ?

C'est apprendre au modèle à faire de bonnes prédictions en lui montrant des **exemples**.

### Les étapes

```
1. On lui donne des données passées (X)
2. On lui donne la vraie valeur qui suivait (Y)
3. Le modèle essaie de prédire Y à partir de X
4. On lui dit s'il a bien ou mal prédit (erreur)
5. Il ajuste ses paramètres pour faire mieux
6. On répète 30 fois (30 époques)
```

### Paramètres d'entraînement

| Paramètre | Valeur | Explication simple |
|-----------|--------|-------------------|
| Époques | 30 | Nombre de fois qu'on répète l'apprentissage |
| Batch | 32 | Nombre d'exemples traités ensemble |
| Learning rate | 0.001 | Vitesse d'apprentissage (pas trop vite, pas trop lent) |

### Comment on sait si c'est bon ?

On mesure l'**erreur** entre ce que le modèle prédit et la vraie valeur :
- **Faible erreur** = bon modèle ✅
- **Grande erreur** = mauvais modèle ❌

---

## 🔮 La Prédiction

### Quand et comment ?

Toutes les **5 minutes**, le modèle :

1. **Récupère** les dernières données des capteurs
2. **Prédit** les valeurs pour demain (00h à 23h)
3. **Calcule** un score de qualité (0 à 100)
4. **Stocke** les prédictions en base de données

### Le score de qualité

| Score | Niveau | Signification |
|-------|--------|---------------|
| 80-100 | 🟢 Excellente | Eau parfaite |
| 60-79 | 🟡 Bonne | Eau acceptable |
| 40-59 | 🟠 Moyenne | Attention |
| 0-39 | 🔴 Faible | Problème ! |

---

## 📁 Fichiers du Module

```
stmodel/
├── stmodel.py          ← Code principal (modèle + prédiction)
├── requirements.txt    ← Librairies Python nécessaires
├── Dockerfile          ← Pour créer le conteneur Docker
└── weights/
    └── trained_weights.pth  ← Poids du modèle entraîné
```

---

## 🛠️ Technologies

| Outil | Rôle |
|-------|------|
| **Python** | Langage de programmation |
| **PyTorch** | Librairie pour le deep learning |
| **TimescaleDB** | Base de données pour stocker les mesures |
| **Docker** | Pour déployer le modèle |

---

## 🚀 Commandes

```bash
# Lancer l'entraînement
docker exec stmodel python stmodel.py --train

# Voir les logs
docker logs -f stmodel
```

---

## 📝 Résumé en 1 minute

1. **Données** : 16 capteurs mesurent pH, turbidité, température
2. **Modèle** : ConvLSTM apprend les patterns passés
3. **Entraînement** : On lui montre des exemples, il apprend
4. **Prédiction** : Il prédit la qualité pour les 24h suivantes
5. **Score** : 0-100, plus c'est haut, meilleure est l'eau

---

## 👥 Équipe

- Ghayt El Idrissi Dafali
- Reda Bouimakliouine
- Souhail Azzimani
- Amine Ibnou Chiekh

**EMSI Marrakech - 2025-2026**
