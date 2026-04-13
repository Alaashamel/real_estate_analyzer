"""
Image Feature Extractor for Multimodal Learning
Extracts deep features from property images using pretrained CNN
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from typing import List, Union

class FeatureExtractor:
    """
    Extract features from images using pretrained CNN
    Supports multiple backbones: ResNet18, ResNet50, EfficientNet
    """
    
    def __init__(self, model_name='resnet18', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_name = model_name
        
        # Load model and remove classification head
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.feature_dim = 512
            # Remove the final fully connected layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.feature_dim = 2048
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            self.feature_dim = 1280
            # Remove classifier
            self.model.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ FeatureExtractor initialized")
        print(f"   Model: {model_name}")
        print(f"   Feature dimension: {self.feature_dim}")
        print(f"   Device: {device}")
    
    def extract_single(self, image_path: str) -> np.ndarray:
        """
        Extract features from a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Feature vector as numpy array
        """
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            return np.zeros(self.feature_dim)
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().cpu().numpy()
            
            # Ensure correct shape
            if features.ndim == 0:
                features = np.array([features])
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features from {image_path}: {e}")
            return np.zeros(self.feature_dim)
    
    def extract_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract features from multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Aggregated feature vector (mean pooling across images)
        """
        if not image_paths:
            return np.zeros(self.feature_dim)
        
        features_list = []
        for path in image_paths:
            feat = self.extract_single(path)
            features_list.append(feat)
        
        # Mean pooling across all images of the same property
        if features_list:
            return np.mean(features_list, axis=0)
        return np.zeros(self.feature_dim)
    
    def extract_with_attention(self, image_paths: List[str]) -> dict:
        """
        Extract features with attention weighting
        Gives more weight to higher quality images
        
        Returns:
            Dictionary with features and attention weights
        """
        if not image_paths:
            return {'features': np.zeros(self.feature_dim), 'weights': [], 'num_images': 0}
        
        features_list = []
        quality_scores = []
        
        for path in image_paths:
            feat = self.extract_single(path)
            quality = self._estimate_image_quality(path)
            
            features_list.append(feat)
            quality_scores.append(quality)
        
        # Normalize quality scores to attention weights
        quality_scores = np.array(quality_scores)
        if quality_scores.sum() > 0:
            weights = quality_scores / quality_scores.sum()
        else:
            weights = np.ones(len(quality_scores)) / len(quality_scores)
        
        # Weighted average of features
        weighted_features = np.zeros(self.feature_dim)
        for i, feat in enumerate(features_list):
            weighted_features += weights[i] * feat
        
        return {
            'features': weighted_features,
            'weights': weights.tolist(),
            'num_images': len(image_paths),
            'quality_scores': quality_scores.tolist()
        }
    
    def _estimate_image_quality(self, image_path: str) -> float:
        """
        Estimate image quality based on resolution and basic metrics
        
        Returns:
            Quality score between 0 and 1
        """
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            # Resolution score
            resolution_score = min(1.0, (width * height) / (1920 * 1080))
            
            # Format score (JPEG/PNG preferred)
            format_score = 1.0 if img.format in ['JPEG', 'PNG'] else 0.7
            
            # Combined score
            quality = (resolution_score + format_score) / 2
            
            return quality
        except:
            return 0.5
    
    def extract_property_features(self, property_data: dict) -> np.ndarray:
        """
        Extract image features for a property from its image paths
        
        Args:
            property_data: Dictionary containing 'images' key with list of paths
            
        Returns:
            Combined feature vector
        """
        image_paths = property_data.get('images', [])
        
        if isinstance(image_paths, str):
            try:
                import json
                image_paths = json.loads(image_paths)
            except:
                image_paths = [image_paths]
        
        return self.extract_batch(image_paths)
    
    def save(self, path: str):
        """Save model state"""
        torch.save({
            'model_state': self.model.state_dict(),
            'model_name': self.model_name,
            'feature_dim': self.feature_dim
        }, path)
        print(f"💾 FeatureExtractor saved to {path}")
    
    def load(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model_name = checkpoint['model_name']
        self.feature_dim = checkpoint['feature_dim']
        print(f"📂 FeatureExtractor loaded from {path}")


class MultiScaleFeatureExtractor:
    """
    Extract features at multiple scales for better representation
    """
    
    def __init__(self, base_extractor=None):
        self.extractor = base_extractor or FeatureExtractor()
        self.scales = [(224, 224), (112, 112), (448, 448)]
    
    def extract_multiscale(self, image_path: str) -> np.ndarray:
        """Extract features at multiple scales and concatenate"""
        if not os.path.exists(image_path):
            return np.zeros(self.extractor.feature_dim * len(self.scales))
        
        all_features = []
        
        for scale in self.scales:
            # Temporarily change transform size
            original_transform = self.extractor.transform
            self.extractor.transform = transforms.Compose([
                transforms.Resize(scale),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            features = self.extractor.extract_single(image_path)
            all_features.append(features)
            
            # Restore original transform
            self.extractor.transform = original_transform
        
        return np.concatenate(all_features)
    
    def extract_batch_multiscale(self, image_paths: List[str]) -> np.ndarray:
        """Extract multiscale features for multiple images"""
        all_features = []
        for path in image_paths:
            features = self.extract_multiscale(path)
            all_features.append(features)
        
        return np.mean(all_features, axis=0) if all_features else np.zeros(self.extractor.feature_dim * len(self.scales))


if __name__ == "__main__":
    # Test feature extractor
    extractor = FeatureExtractor()
    print(f"✅ FeatureExtractor ready")
    
    # Test with a dummy feature
    dummy_feature = np.random.randn(extractor.feature_dim)
    print(f"   Sample feature shape: {dummy_feature.shape}")
    
    # Multi-scale extractor
    multi_extractor = MultiScaleFeatureExtractor(extractor)
    print(f"✅ MultiScaleFeatureExtractor ready")
    print(f"   Multi-scale feature dimension: {multi_extractor.extractor.feature_dim * len(multi_extractor.scales)}")