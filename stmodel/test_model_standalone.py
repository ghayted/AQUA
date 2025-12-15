#!/usr/bin/env python3
"""
Script standalone pour tester le modèle HourlyWaterQualityPredictor
Pas besoin de Docker ni de base de données !

Usage:
    python test_model_standalone.py
"""

import torch
import torch.nn as nn
import numpy as np

# =============================================================================
# ARCHITECTURE DU MODÈLE (copie exacte de stmodel.py)
# =============================================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=kernel_size // 2)
    
    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c_next = f * c + i * torch.tanh(g)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
    
    def init_hidden(self, batch, size, device):
        return (torch.zeros(batch, self.hidden_dim, *size, device=device),
                torch.zeros(batch, self.hidden_dim, *size, device=device))


class HourlyWaterQualityPredictor(nn.Module):
    """
    ConvLSTM avec prédiction horaire.
    
    ENTRÉES:
    - x: (Batch, 12, 3, 4, 4) = 12 pas de temps × 3 params × 4×4 grille spatiale
    - hour: (Batch, 1) = heure cible normalisée [0, 1]
    
    SORTIE:
    - (Batch, 10, 3) = 10 zones × 3 paramètres (pH, turbidité, température)
    """
    
    def __init__(self, n_zones: int = 10):
        super().__init__()
        self.n_zones = n_zones
        self.encoder = ConvLSTMCell(3, 32)
        self.hour_embedding = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32 * 4 * 4 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_zones * 3),
            nn.Sigmoid()
        )
    
    def forward(self, x, hour):
        batch, seq_len, c, h, w = x.shape
        state = self.encoder.init_hidden(batch, (h, w), x.device)
        for t in range(seq_len):
            state = self.encoder(x[:, t], state)
        spatial_features = state[0].view(batch, -1)
        hour_features = self.hour_embedding(hour)
        combined = torch.cat([spatial_features, hour_features], dim=1)
        output = self.decoder(combined)
        return output.view(batch, self.n_zones, 3)
    
    def predict(self, x, hour):
        """Prédit les valeurs dénormalisées."""
        norm = self.forward(x, hour)
        denorm = norm.clone()
        denorm[:, :, 0] = norm[:, :, 0] * 4.0 + 5.5   # pH [5.5, 9.5]
        denorm[:, :, 1] = norm[:, :, 1] * 8.0         # Turb [0, 8]
        denorm[:, :, 2] = norm[:, :, 2] * 25.0 + 10.0 # Temp [10, 35]
        return denorm


# =============================================================================
# TEST DU MODÈLE
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🧪 Test du modèle HourlyWaterQualityPredictor")
    print("=" * 60)
    
    # Créer le modèle
    model = HourlyWaterQualityPredictor(n_zones=10)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Modèle créé: {params:,} paramètres")
    
    # Afficher l'architecture
    print("\n🏗️ ARCHITECTURE:")
    print(model)
    
    # Créer des données de test (simulation)
    print("\n" + "=" * 60)
    print("📥 ENTRÉES (simulées)")
    print("=" * 60)
    
    # Entrée 1: Séquence de 12 mesures (batch=1)
    # Shape: (1, 12, 3, 4, 4) = 1 batch × 12 temps × 3 params × 4×4 grille
    x = torch.randn(1, 12, 3, 4, 4)
    print(f"\n• Données capteurs: {x.shape}")
    print(f"  - Batch size: 1")
    print(f"  - Séquence: 12 pas de temps")
    print(f"  - Paramètres: 3 (pH, turbidité, température)")
    print(f"  - Grille spatiale: 4×4")
    
    # Entrée 2: Heure cible (ex: 14h = 14/23 ≈ 0.61)
    hour = torch.tensor([[14.0 / 23.0]])  # 14:00
    print(f"\n• Heure cible: {hour.item():.3f} (= 14:00)")
    
    # Prédiction
    print("\n" + "=" * 60)
    print("📤 SORTIE (prédiction)")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        output = model.predict(x, hour)
    
    print(f"\n• Shape sortie: {output.shape} (10 zones × 3 paramètres)")
    
    zones = ['Rabat-Centre', 'Salé-Nord', 'Salé-Sud', 'Hay-Riad', 'Agdal',
             'Côte-Océan', 'Bouregreg', 'Temara', 'Skhirat', 'Marrakech']
    
    print("\n📊 Prédictions pour demain 14:00:")
    print("-" * 50)
    print(f"{'Zone':<15} {'pH':>8} {'Turb (NTU)':>12} {'Temp (°C)':>10}")
    print("-" * 50)
    
    for i, zone in enumerate(zones):
        ph = output[0, i, 0].item()
        turb = output[0, i, 1].item()
        temp = output[0, i, 2].item()
        print(f"{zone:<15} {ph:>8.2f} {turb:>12.2f} {temp:>10.1f}")
    
    print("-" * 50)
    
    # Test avec différentes heures
    print("\n" + "=" * 60)
    print("⏰ COMPARAISON PAR HEURE (zone Marrakech)")
    print("=" * 60)
    
    print(f"\n{'Heure':>8} {'pH':>8} {'Turb':>8} {'Temp':>8}")
    print("-" * 36)
    
    for h in [0, 6, 12, 18, 23]:
        hour_tensor = torch.tensor([[h / 23.0]])
        with torch.no_grad():
            pred = model.predict(x, hour_tensor)
        
        # Marrakech = index 9
        ph = pred[0, 9, 0].item()
        turb = pred[0, 9, 1].item()
        temp = pred[0, 9, 2].item()
        print(f"{h:>5}:00 {ph:>8.2f} {turb:>8.2f} {temp:>7.1f}°C")
    
    print("\n✅ Test terminé!")
    print("\n💡 Note: Les valeurs sont aléatoires car le modèle n'est pas entraîné.")
    print("   Après entraînement, les valeurs refléteront les patterns appris.")
