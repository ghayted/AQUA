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

## 📐 Schéma des Matrices d'Entraînement

### Structure des données d'entrée (X)

```
                    MATRICE D'ENTRÉE X
        ┌─────────────────────────────────────────┐
        │    Dimensions: (batch, 12, 3, 4, 4)     │
        └─────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
        batch=32       temps=12       paramètres=3
     (32 exemples)   (12 mesures     (pH, Turb, Temp)
                      passées)              │
                                            ▼
                                      grille 4×4
                                    (10 zones sur
                                     une grille)
```

### Visualisation d'UN exemple d'entraînement

```
ENTRÉE X : Séquence de 12 pas de temps
═══════════════════════════════════════════════════════════════════

Temps T-12        Temps T-11        Temps T-10    ...    Temps T-1
(il y a 12h)      (il y a 11h)      (il y a 10h)         (maintenant)
    │                 │                 │                    │
    ▼                 ▼                 ▼                    ▼

┌─────────┐      ┌─────────┐      ┌─────────┐         ┌─────────┐
│ Grille  │      │ Grille  │      │ Grille  │         │ Grille  │
│  4 × 4  │      │  4 × 4  │      │  4 × 4  │   ...   │  4 × 4  │
│         │      │         │      │         │         │         │
│ 3 params│      │ 3 params│      │ 3 params│         │ 3 params│
└─────────┘      └─────────┘      └─────────┘         └─────────┘

    ×3               ×3               ×3                  ×3
 (pH,Turb,         (pH,Turb,       (pH,Turb,           (pH,Turb,
   Temp)             Temp)           Temp)               Temp)
```

### La Grille 4×4 : Comment les zones sont placées

```
         Colonne 0    Colonne 1    Colonne 2
        ┌───────────┬───────────┬───────────┬───────────┐
Ligne 0 │  Rabat-   │  Salé-    │  Salé-    │  Hay-     │
        │  Centre   │  Nord     │  Sud      │  Riad     │
        ├───────────┼───────────┼───────────┼───────────┤
Ligne 1 │  Agdal    │  Côte-    │ Bouregreg │  Temara   │
        │           │  Océan    │           │           │
        ├───────────┼───────────┼───────────┼───────────┤
Ligne 2 │ Skhirat   │ Marrakech │   vide    │   vide    │
        │           │           │           │           │
        ├───────────┼───────────┼───────────┼───────────┤
Ligne 3 │   vide    │   vide    │   vide    │   vide    │
        │           │           │           │           │
        └───────────┴───────────┴───────────┴───────────┘

        → 10 zones actives + 6 cases vides = grille 4×4
```

### Exemple concret d'une matrice à UN instant T

```
PARAMÈTRE: pH (valeurs réelles)           PARAMÈTRE: Turbidité (NTU)
┌──────┬──────┬──────┬──────┐             ┌──────┬──────┬──────┬──────┐
│ 7.2  │ 7.0  │ 6.8  │ 7.1  │             │ 0.8  │ 1.2  │ 0.5  │ 0.9  │
├──────┼──────┼──────┼──────┤             ├──────┼──────┼──────┼──────┤
│ 7.3  │ 7.5  │ 6.9  │ 7.0  │             │ 0.6  │ 2.1  │ 1.5  │ 0.7  │
├──────┼──────┼──────┼──────┤             ├──────┼──────┼──────┼──────┤
│ 7.1  │ 5.8  │  0   │  0   │             │ 0.9  │ 7.2  │  0   │  0   │
├──────┼──────┼──────┼──────┤             ├──────┼──────┼──────┼──────┤
│  0   │  0   │  0   │  0   │             │  0   │  0   │  0   │  0   │
└──────┴──────┴──────┴──────┘             └──────┴──────┴──────┴──────┘
        ↑                                         ↑
   Marrakech = 5.8                         Marrakech = 7.2
   (pH critique!)                          (très trouble!)


PARAMÈTRE: Température (°C)
┌──────┬──────┬──────┬──────┐
│ 22.5 │ 21.0 │ 23.1 │ 22.0 │
├──────┼──────┼──────┼──────┤
│ 24.0 │ 19.5 │ 20.0 │ 23.5 │
├──────┼──────┼──────┼──────┤
│ 21.0 │ 32.0 │  0   │  0   │
├──────┼──────┼──────┼──────┤
│  0   │  0   │  0   │  0   │
└──────┴──────┴──────┴──────┘
        ↑
   Marrakech = 32°C
   (trop chaud!)
```

### Normalisation des Valeurs (avant d'entrer dans le modèle)

```
Valeurs RÉELLES              →              Valeurs NORMALISÉES [0-1]

pH:     5.5 ──────────────────── 9.5        0.0 ──────────────────── 1.0
        Formule: (pH - 5.5) / 4.0
        Exemple: pH=7.0 → (7.0-5.5)/4.0 = 0.375

Turb:   0 ────────────────────── 8          0.0 ──────────────────── 1.0
        Formule: turbidité / 8.0
        Exemple: turb=2.0 → 2.0/8.0 = 0.25

Temp:   10 ────────────────────── 35        0.0 ──────────────────── 1.0
        Formule: (temp - 10) / 25.0
        Exemple: temp=22°C → (22-10)/25.0 = 0.48
```

### Le Processus Complet d'Entraînement

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DONNÉES HISTORIQUES (15 jours)                   │
│            16 capteurs × 24h × 15 jours = 5760 mesures              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CRÉATION DES SÉQUENCES                           │
│                                                                     │
│   Pour chaque mesure au temps T :                                   │
│   • X = les 12 mesures précédentes (T-12 à T-1)                    │
│   • Y = la mesure à prédire (temps T)                              │
│   • H = l'heure de la prédiction (0-23)                            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DIVISION TRAIN / VALIDATION                      │
│                                                                     │
│           80% pour ENTRAÎNER    │    20% pour VALIDER               │
│              (apprendre)        │       (tester)                    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BOUCLE D'ENTRAÎNEMENT (×30 époques)              │
│                                                                     │
│   1. Prendre un batch de 32 exemples                               │
│   2. Le modèle prédit Y à partir de X                              │
│   3. Comparer avec la vraie valeur Y → calcul ERREUR               │
│   4. Ajuster les poids pour réduire l'erreur                       │
│   5. Répéter pour tous les batchs                                  │
│   6. Mesurer la performance sur validation                         │
│   7. Si meilleur → sauvegarder les poids                           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MODÈLE ENTRAÎNÉ                                  │
│                    trained_weights.pth                              │
│                    (~500,000 paramètres)                            │
└─────────────────────────────────────────────────────────────────────┘
```

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
