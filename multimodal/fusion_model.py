"""
Multimodal Fusion Model
Combines: Text features (NLP) + Image features (CNN) + Tabular features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os


class MultimodalDataset(Dataset):
    """Dataset for multimodal learning"""
    
    def __init__(self, text_features, image_features, tabular_features, prices):
        self.text_features = torch.FloatTensor(text_features)
        self.image_features = torch.FloatTensor(image_features)
        self.tabular_features = torch.FloatTensor(tabular_features)
        self.prices = torch.FloatTensor(prices)
    
    def __len__(self):
        return len(self.prices)
    
    def __getitem__(self, idx):
        return {
            'text': self.text_features[idx],
            'image': self.image_features[idx],
            'tabular': self.tabular_features[idx],
            'price': self.prices[idx]
        }


class MultimodalFusionModel(nn.Module):
    """
    Neural network that fuses three modalities:
    - Text (NLP features)
    - Image (CNN features)
    - Tabular (numeric features like area, bedrooms)
    """
    
    def __init__(self, text_dim=500, image_dim=512, tabular_dim=10, 
                 hidden_dims=[256, 128, 64], dropout=0.3):
        super(MultimodalFusionModel, self).__init__()
        
        # Text branch
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Image branch
        self.image_branch = nn.Sequential(
            nn.Linear(image_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Tabular branch
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dims[0] // 2),
            nn.BatchNorm1d(hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        fusion_dim = hidden_dims[0] * 2 + hidden_dims[0] // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], 1)  # Price prediction
        )
        
        # Attention mechanism for modality weighting
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dims[0], num_heads=4, batch_first=True)
    
    def forward(self, text, image, tabular):
        # Extract features from each modality
        text_feat = self.text_branch(text)
        image_feat = self.image_branch(image)
        tabular_feat = self.tabular_branch(tabular)
        
        # Reshape for attention
        modality_features = torch.stack([text_feat, image_feat], dim=1)
        
        # Apply attention between text and image
        attended, attention_weights = self.attention(modality_features, modality_features, modality_features)
        
        # Concatenate attended features with tabular
        fused = torch.cat([attended[:, 0, :], attended[:, 1, :], tabular_feat], dim=1)
        
        # Final prediction
        price = self.fusion(fused)
        
        return price.squeeze()


class MultimodalTrainer:
    """Trainer for multimodal model"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = None
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            text = batch['text'].to(self.device)
            image = batch['image'].to(self.device)
            tabular = batch['tabular'].to(self.device)
            prices = batch['price'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(text, image, tabular)
            loss = self.criterion(predictions, prices)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in dataloader:
                text = batch['text'].to(self.device)
                image = batch['image'].to(self.device)
                tabular = batch['tabular'].to(self.device)
                prices = batch['price'].to(self.device)
                
                preds = self.model(text, image, tabular)
                loss = self.criterion(preds, prices)
                
                total_loss += loss.item()
                predictions.extend(preds.cpu().numpy())
                actuals.extend(prices.cpu().numpy())
        
        return total_loss / len(dataloader), np.array(predictions), np.array(actuals)
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Full training loop"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        
        print(f"\n🏋️ Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, _, _ = self.validate(val_loader)
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'models/best_multimodal_model.pt')
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")
        return self.model


# Helper functions
def create_multimodal_model(text_dim=500, image_dim=512, tabular_dim=10):
    """Create a new multimodal model instance"""
    return MultimodalFusionModel(
        text_dim=text_dim,
        image_dim=image_dim,
        tabular_dim=tabular_dim
    )


def prepare_multimodal_features(df, text_vectorizer=None, scaler=None):
    """Prepare features for multimodal model"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    
    # Text features
    if text_vectorizer is None:
        text_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        text_features = text_vectorizer.fit_transform(df['description'].fillna('')).toarray()
    else:
        text_features = text_vectorizer.transform(df['description'].fillna('')).toarray()
    
    # Image features (dummy for now)
    image_features = np.random.randn(len(df), 512)
    
    # Tabular features
    tabular_cols = ['area_sqm', 'bedrooms', 'bathrooms']
    available_cols = [c for c in tabular_cols if c in df.columns]
    
    X_tabular = df[available_cols].copy()
    
    # Add encoded features
    if 'property_type' in df.columns:
        type_mapping = {'Apartment': 0, 'Villa': 1, 'Townhouse': 2, 'Studio': 3, 'Duplex': 4}
        X_tabular['property_type_encoded'] = df['property_type'].map(type_mapping).fillna(0)
    
    if 'furnishing' in df.columns:
        furn_mapping = {'Unfurnished': 0, 'Semi-furnished': 1, 'Furnished': 2}
        X_tabular['furnishing_encoded'] = df['furnishing'].map(furn_mapping).fillna(0)
    
    X_tabular = X_tabular.fillna(0)
    
    if scaler is None:
        scaler = StandardScaler()
        tabular_features = scaler.fit_transform(X_tabular)
    else:
        tabular_features = scaler.transform(X_tabular)
    
    return text_features, image_features, tabular_features, text_vectorizer, scaler


if __name__ == "__main__":
    # Test the model
    model = create_multimodal_model()
    print(f"✅ MultimodalFusionModel created")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    text = torch.randn(batch_size, 500)
    image = torch.randn(batch_size, 512)
    tabular = torch.randn(batch_size, 10)
    
    output = model(text, image, tabular)
    print(f"   Output shape: {output.shape}")