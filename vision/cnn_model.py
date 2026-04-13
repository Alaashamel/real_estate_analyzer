"""
CNN Model for property image analysis
- Room type classification (kitchen, bedroom, bathroom, living room)
- Property condition assessment
- Feature extraction for multimodal fusion
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import pickle

class RoomTypeClassifier(nn.Module):
    """
    CNN classifier for room types
    Classes: kitchen, bedroom, bathroom, living_room, exterior
    """
    
    def __init__(self, num_classes=5, pretrained=True):
        super(RoomTypeClassifier, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.room_types = ['kitchen', 'bedroom', 'bathroom', 'living_room', 'exterior']
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, image_tensor):
        """Predict room type for a single image"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return {
            'class': self.room_types[predicted_class],
            'confidence': probabilities[0][predicted_class].item(),
            'all_probabilities': {
                self.room_types[i]: probabilities[0][i].item() 
                for i in range(len(self.room_types))
            }
        }


class PropertyConditionAssessor(nn.Module):
    """
    CNN for assessing property condition
    Outputs: condition score (0-1) and maintenance level
    """
    
    def __init__(self, pretrained=True):
        super(PropertyConditionAssessor, self).__init__()
        
        # Use EfficientNet for better feature extraction
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Remove classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Regression head for condition score
        self.condition_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Classification head for maintenance level
        self.maintenance_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # good, average, needs_repair
        )
    
    def forward(self, x):
        features = self.backbone(x)
        condition_score = self.condition_head(features)
        maintenance_logits = self.maintenance_head(features)
        
        return {
            'condition_score': condition_score,
            'maintenance_logits': maintenance_logits,
            'features': features
        }
    
    def predict(self, image_tensor):
        """Predict condition for a single image"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(image_tensor)
            condition = outputs['condition_score'].item()
            maintenance_probs = torch.softmax(outputs['maintenance_logits'], dim=1)
            maintenance_class = torch.argmax(maintenance_probs, dim=1).item()
            
        maintenance_labels = ['Good', 'Average', 'Needs Repair']
        
        return {
            'condition_score': condition,
            'maintenance_level': maintenance_labels[maintenance_class],
            'confidence': maintenance_probs[0][maintenance_class].item()
        }


class ImageFeatureExtractor:
    """Extract features from images for multimodal fusion"""
    
    def __init__(self, model_name='resnet18', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load pretrained model for feature extraction
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            # Remove classification head
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
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
        
        # Feature dimension
        self.feature_dim = 512 if model_name == 'resnet18' else 2048
    
    def extract_features(self, image_path):
        """Extract features from a single image"""
        if not os.path.exists(image_path):
            return np.zeros(self.feature_dim)
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().cpu().numpy()
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return np.zeros(self.feature_dim)
    
    def extract_batch_features(self, image_paths):
        """Extract features from multiple images"""
        features = []
        for path in image_paths:
            feat = self.extract_features(path)
            features.append(feat)
        
        # Aggregate multiple images per property (mean pooling)
        if features:
            return np.mean(features, axis=0)
        return np.zeros(self.feature_dim)


# Preprocessing transforms for training
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


if __name__ == "__main__":
    # Test feature extraction
    extractor = ImageFeatureExtractor()
    print(f"✅ ImageFeatureExtractor initialized with dimension: {extractor.feature_dim}")
    
    # Test room classifier
    room_classifier = RoomTypeClassifier()
    print(f"✅ RoomTypeClassifier initialized with {len(room_classifier.room_types)} classes")
    
    # Test condition assessor
    condition_assessor = PropertyConditionAssessor()
    print(f"✅ PropertyConditionAssessor initialized")