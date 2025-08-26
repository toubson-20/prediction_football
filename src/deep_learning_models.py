"""
🧠 DEEP LEARNING MODELS - COMPOSANTS AVANCÉS IA
Transformers, LSTM/GRU, Graph Neural Networks, CNN 1D
Architecture révolutionnaire pour prédictions football

Version: 2.0 - Phase 2 ML Transformation
Créé: 23 août 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class FootballDataset(Dataset):
    """Dataset personnalisé pour données football"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 10):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class MultiHeadAttention(nn.Module):
    """Mécanisme d'attention multi-têtes pour Transformer"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Transformation linéaire et reshape pour multi-head
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Calcul de l'attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Application attention aux valeurs
        context = torch.matmul(attention_weights, V)
        
        # Reconcat et projection finale
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(context), attention_weights

class TransformerBlock(nn.Module):
    """Bloc Transformer avec attention et feed-forward"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention avec connexion résiduelle
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward avec connexion résiduelle
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class FootballTransformer(nn.Module):
    """Transformer spécialisé pour prédictions football
    
    Utilise l'attention pour identifier les features importantes
    et leurs interactions contextuelles
    """
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 max_seq_length: int = 200,
                 n_outputs: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Projection et encodage positionnel
        x = self.input_projection(x)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        
        # Passage dans les blocs Transformer
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x)
            attention_weights.append(attn_weights)
        
        # Agrégation et prédiction finale
        x = x.mean(dim=1)  # Global average pooling
        output = self.output_projection(x)
        
        return output, attention_weights

class FootballLSTM(nn.Module):
    """LSTM bidirectionnel pour séquences temporelles football
    
    Capture les patterns temporels dans la forme des équipes,
    performances récentes, momentum
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 n_layers: int = 3,
                 n_outputs: int = 1,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            dropout=dropout, bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_outputs)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention mechanism pour pondérer les timesteps
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum selon attention
        context = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classification finale
        output = self.classifier(context)
        
        return output, attention_weights

class GraphConvLayer(nn.Module):
    """Couche de convolution de graphe pour relations équipes/joueurs"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        
    def forward(self, x, adj_matrix):
        # x: [batch_size, n_nodes, in_features]
        # adj_matrix: [batch_size, n_nodes, n_nodes]
        
        # Transformation linéaire
        x = self.linear(x)
        
        # Agrégation selon adjacence
        x = torch.bmm(adj_matrix, x)
        
        return self.activation(x)

class FootballGraphNN(nn.Module):
    """Graph Neural Network pour relations équipes/joueurs
    
    Modélise les interactions entre joueurs, équipes, et leurs
    performances relatives dans différents contextes
    """
    
    def __init__(self,
                 node_features: int,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 n_outputs: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        self.graph_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_outputs)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features, adj_matrix):
        # Embedding initial des noeuds
        x = self.node_embedding(node_features)
        x = self.dropout(x)
        
        # Propagation dans les couches de graphe
        for graph_layer in self.graph_layers:
            x = graph_layer(x, adj_matrix)
            x = self.dropout(x)
        
        # Global readout (moyenne des noeuds)
        graph_representation = x.mean(dim=1)
        
        # Prédiction finale
        output = self.readout(graph_representation)
        
        return output

class CNN1DFootball(nn.Module):
    """CNN 1D pour patterns dans séries statistiques
    
    Détecte des motifs locaux dans les séquences de performances,
    statistiques de match, tendances
    """
    
    def __init__(self,
                 input_dim: int,
                 n_outputs: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Première couche conv
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Deuxième couche conv
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Troisième couche conv
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs)
        )
        
    def forward(self, x):
        # x: [batch_size, sequence_length, features]
        x = x.transpose(1, 2)  # [batch_size, features, sequence_length]
        
        # Convolutions
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification
        output = self.classifier(x)
        
        return output

class DeepLearningEnsemble:
    """Ensemble des modèles Deep Learning pour prédictions football"""
    
    def __init__(self, input_dim: int, device: str = 'cpu'):
        self.device = device
        self.input_dim = input_dim
        
        # Initialisation des modèles
        self.models = {
            'transformer': FootballTransformer(input_dim=1, d_model=128, n_heads=4, n_layers=3),
            'lstm': FootballLSTM(input_dim=input_dim, hidden_dim=64, n_layers=2),
            'gnn': FootballGraphNN(node_features=input_dim, hidden_dim=32, n_layers=2),
            'cnn1d': CNN1DFootball(input_dim=input_dim)
        }
        
        # Déplacer modèles sur device
        for name, model in self.models.items():
            model.to(device)
        
        self.scalers = {}
        self.is_trained = {name: False for name in self.models.keys()}
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None, model_type: str = 'transformer'):
        """Prépare les données selon le type de modèle"""
        
        if model_type == 'transformer':
            # Transformer attend [batch, seq_len, features]
            if len(X.shape) == 2:
                X = X.unsqueeze(1)  # Ajoute dimension séquence
        
        elif model_type == 'lstm':
            # LSTM attend [batch, seq_len, features]
            if len(X.shape) == 2:
                batch_size, features = X.shape
                seq_len = min(10, features // 10)  # Séquence adaptative
                X = X.view(batch_size, seq_len, -1)
        
        elif model_type == 'gnn':
            # GNN attend node_features et adj_matrix
            # Création matrice adjacence simple
            batch_size = X.shape[0]
            n_nodes = min(20, X.shape[1] // 10)  # Nombre noeuds adaptatif
            
            # Redimensionner pour noeuds
            X = X[:, :n_nodes*10].view(batch_size, n_nodes, 10)
            
            # Matrice adjacence simple (tous connectés avec poids décroissants)
            adj_matrix = torch.ones(batch_size, n_nodes, n_nodes)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        adj_matrix[:, i, j] = 1.0 / (abs(i-j) + 1)
            
            adj_matrix = adj_matrix.to(self.device)
            return X.to(self.device), adj_matrix
        
        elif model_type == 'cnn1d':
            # CNN1D attend [batch, seq_len, features] 
            if len(X.shape) == 2:
                batch_size, total_features = X.shape
                seq_len = min(50, total_features // 4)  # Séquence pour CNN
                features_per_step = total_features // seq_len
                X = X[:, :seq_len*features_per_step].view(batch_size, seq_len, features_per_step)
        
        return X.to(self.device)
    
    def train_model(self, 
                   model_name: str,
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   X_val: np.ndarray = None,
                   y_val: np.ndarray = None,
                   epochs: int = 50,
                   batch_size: int = 32,
                   learning_rate: float = 0.001) -> Dict:
        """Entraîne un modèle spécifique"""
        
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non disponible")
        
        model = self.models[model_name]
        
        # Préparation des données
        if isinstance(X_train, np.ndarray):
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train)
        
        # Normalisation
        if model_name not in self.scalers:
            self.scalers[model_name] = StandardScaler()
            X_train_scaled = self.scalers[model_name].fit_transform(X_train.numpy())
        else:
            X_train_scaled = self.scalers[model_name].transform(X_train.numpy())
        
        X_train_scaled = torch.FloatTensor(X_train_scaled)
        
        # Préparation selon type de modèle
        if model_name == 'gnn':
            X_prepared, adj_matrix = self._prepare_data(X_train_scaled, model_type=model_name)
        else:
            X_prepared = self._prepare_data(X_train_scaled, model_type=model_name)
        
        # Dataset et DataLoader
        dataset = FootballDataset(X_prepared, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimiseur et loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Entraînement
        model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass selon type
                if model_name == 'gnn':
                    batch_adj = adj_matrix[:batch_X.size(0)]
                    outputs = model(batch_X, batch_adj)
                elif model_name in ['transformer', 'lstm']:
                    outputs, _ = model(batch_X)
                else:  # CNN1D
                    outputs = model(batch_X)
                
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.is_trained[model_name] = True
        
        return {
            'train_losses': train_losses,
            'final_loss': train_losses[-1],
            'epochs': epochs
        }
    
    def predict(self, X: np.ndarray, model_names: List[str] = None) -> Dict:
        """Prédictions avec modèles sélectionnés"""
        
        if model_names is None:
            model_names = [name for name, trained in self.is_trained.items() if trained]
        
        predictions = {}
        
        for model_name in model_names:
            if not self.is_trained[model_name]:
                print(f"Modèle {model_name} non entraîné, ignoré")
                continue
            
            model = self.models[model_name]
            model.eval()
            
            # Normalisation
            X_scaled = self.scalers[model_name].transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            
            # Préparation selon type
            if model_name == 'gnn':
                X_prepared, adj_matrix = self._prepare_data(X_tensor, model_type=model_name)
            else:
                X_prepared = self._prepare_data(X_tensor, model_type=model_name)
            
            with torch.no_grad():
                if model_name == 'gnn':
                    outputs = model(X_prepared, adj_matrix)
                elif model_name in ['transformer', 'lstm']:
                    outputs, attention = model(X_prepared)
                else:
                    outputs = model(X_prepared)
                
                predictions[model_name] = outputs.cpu().numpy().flatten()
        
        return predictions
    
    def get_ensemble_prediction(self, X: np.ndarray, weights: Dict[str, float] = None) -> np.ndarray:
        """Prédiction d'ensemble pondérée"""
        
        predictions = self.predict(X)
        
        if not predictions:
            raise ValueError("Aucun modèle entraîné disponible")
        
        if weights is None:
            # Poids égaux par défaut
            weights = {name: 1.0/len(predictions) for name in predictions.keys()}
        
        ensemble_pred = np.zeros(len(X))
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0.0)
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def save_models(self, save_dir: str):
        """Sauvegarde tous les modèles"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            if self.is_trained[name]:
                torch.save(model.state_dict(), f"{save_dir}/{name}_model.pth")
                
                # Sauvegarder le scaler aussi
                import joblib
                joblib.dump(self.scalers[name], f"{save_dir}/{name}_scaler.joblib")
    
    def load_models(self, save_dir: str):
        """Charge tous les modèles"""
        import os
        import joblib
        
        for name in self.models.keys():
            model_path = f"{save_dir}/{name}_model.pth"
            scaler_path = f"{save_dir}/{name}_scaler.joblib"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[name].load_state_dict(torch.load(model_path, map_location=self.device))
                self.scalers[name] = joblib.load(scaler_path)
                self.is_trained[name] = True
                print(f"Modèle {name} chargé avec succès")
    
    def get_model_info(self) -> Dict:
        """Informations sur les modèles"""
        info = {}
        
        for name, model in self.models.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            info[name] = {
                'trained': self.is_trained[name],
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_type': model.__class__.__name__
            }
        
        return info

def test_deep_learning_models():
    """Test rapide des modèles Deep Learning"""
    print("=== TEST MODELES DEEP LEARNING ===")
    
    # Données factices pour test
    n_samples = 1000
    n_features = 200
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialisation ensemble
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensemble = DeepLearningEnsemble(input_dim=n_features, device=device)
    
    print(f"Device utilisé: {device}")
    print(f"Données d'entraînement: {X_train.shape}")
    
    # Test de chaque modèle
    for model_name in ['transformer', 'lstm', 'cnn1d']:  # On évite GNN pour ce test simple
        print(f"\n--- Test {model_name.upper()} ---")
        
        try:
            # Entraînement rapide
            history = ensemble.train_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                epochs=5,  # Peu d'epochs pour test rapide
                batch_size=64
            )
            
            print(f"Entraînement terminé - Loss finale: {history['final_loss']:.6f}")
            
            # Test prédiction
            predictions = ensemble.predict(X_test[:10], [model_name])
            print(f"Prédictions (10 premiers): {predictions[model_name][:5]}")
            
        except Exception as e:
            print(f"Erreur avec {model_name}: {str(e)}")
    
    # Test ensemble
    try:
        print(f"\n--- Test ENSEMBLE ---")
        ensemble_pred = ensemble.get_ensemble_prediction(X_test[:10])
        print(f"Prédictions ensemble: {ensemble_pred[:5]}")
        
        # Infos modèles
        model_info = ensemble.get_model_info()
        for name, info in model_info.items():
            if info['trained']:
                print(f"{name}: {info['total_parameters']} paramètres")
        
    except Exception as e:
        print(f"Erreur ensemble: {str(e)}")
    
    print("\n=== TEST TERMINE ===")

if __name__ == "__main__":
    test_deep_learning_models()