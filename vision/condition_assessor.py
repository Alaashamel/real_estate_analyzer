"""
Property Condition Assessment Module
Evaluates property condition from images using CNN
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class ConditionAssessor(nn.Module):
    """
    CNN model to assess property condition from images
    Outputs: condition score (0-1) and maintenance level
    """
    
    def __init__(self, pretrained=True):
        super(ConditionAssessor, self).__init__()
        
        # Use EfficientNet-B0 for better feature extraction
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Get feature dimension
        self.feature_dim = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Condition regression head (0 = poor, 1 = excellent)
        self.condition_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Maintenance classification head
        self.maintenance_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # 3 classes: needs_repair, average, good
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.maintenance_labels = ['Needs Repair', 'Average', 'Good']
    
    def forward(self, x):
        features = self.backbone(x)
        condition_score = self.condition_head(features)
        maintenance_logits = self.maintenance_head(features)
        
        return {
            'condition_score': condition_score,
            'maintenance_logits': maintenance_logits,
            'features': features
        }
    
    def predict_image(self, image_path):
        """Predict condition for a single image"""
        self.eval()
        
        if not os.path.exists(image_path):
            return {'condition_score': 0.5, 'maintenance_level': 'Unknown', 'confidence': 0}
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.forward(image_tensor)
                
                condition = outputs['condition_score'].item()
                maintenance_probs = torch.softmax(outputs['maintenance_logits'], dim=1)
                maintenance_class = torch.argmax(maintenance_probs, dim=1).item()
                confidence = maintenance_probs[0][maintenance_class].item()
                
            return {
                'condition_score': condition,
                'maintenance_level': self.maintenance_labels[maintenance_class],
                'confidence': confidence,
                'condition_desc': self._get_condition_desc(condition)
            }
        except Exception as e:
            print(f"Error assessing image {image_path}: {e}")
            return {'condition_score': 0.5, 'maintenance_level': 'Unknown', 'confidence': 0}
    
    def predict_batch(self, image_paths):
        """Predict conditions for multiple images"""
        results = []
        for path in image_paths:
            results.append(self.predict_image(path))
        return results
    
    def _get_condition_desc(self, score):
        """Get text description based on condition score"""
        if score >= 0.8:
            return "Excellent condition - Like new, well maintained"
        elif score >= 0.6:
            return "Good condition - Minor wear and tear"
        elif score >= 0.4:
            return "Average condition - Some maintenance needed"
        elif score >= 0.2:
            return "Fair condition - Significant repairs needed"
        else:
            return "Poor condition - Major renovation required"
    
    def extract_features(self, image_path):
        """Extract features for multimodal learning"""
        self.eval()
        
        if not os.path.exists(image_path):
            return np.zeros(self.feature_dim)
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                features = self.backbone(image_tensor)
                
            return features.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(self.feature_dim)


class MultiImageConditionAssessor:
    """Aggregate condition assessment from multiple images"""
    
    def __init__(self, assessor=None):
        self.assessor = assessor or ConditionAssessor()
    
    def assess_property(self, image_paths):
        """Assess overall property condition from multiple images"""
        if not image_paths:
            return {
                'overall_score': 0.5,
                'overall_level': 'Unknown',
                'image_scores': [],
                'recommendations': []
            }
        
        # Get assessment for each image
        assessments = []
        for path in image_paths:
            result = self.assessor.predict_image(path)
            assessments.append(result)
        
        # Aggregate scores
        scores = [a['condition_score'] for a in assessments]
        overall_score = np.mean(scores) if scores else 0.5
        
        # Determine overall maintenance level
        level_counts = {}
        for a in assessments:
            level = a['maintenance_level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        if level_counts:
            overall_level = max(level_counts, key=level_counts.get)
        else:
            overall_level = 'Unknown'
        
        # Generate recommendations
        recommendations = self._generate_recommendations(overall_score, assessments)
        
        return {
            'overall_score': overall_score,
            'overall_level': overall_level,
            'image_scores': assessments,
            'recommendations': recommendations,
            'num_images_analyzed': len(assessments)
        }
    
    def _generate_recommendations(self, score, assessments):
        """Generate maintenance recommendations based on condition"""
        recommendations = []
        
        if score < 0.3:
            recommendations.append("🔴 Major renovation recommended")
            recommendations.append("🛠️ Inspect structural integrity")
            recommendations.append("💧 Check for water damage or leaks")
        elif score < 0.6:
            recommendations.append("🟡 Moderate repairs needed")
            recommendations.append("🎨 Consider repainting and refreshing")
            recommendations.append("🔧 Update fixtures and appliances")
        else:
            recommendations.append("🟢 Property is in good condition")
            recommendations.append("✨ Regular maintenance recommended")
            recommendations.append("📅 Schedule annual inspection")
        
        # Add specific recommendations based on image analysis
        for i, img in enumerate(assessments):
            if img['condition_score'] < 0.4:
                recommendations.append(f"📸 Image {i+1} shows significant issues")
        
        return recommendations[:5]  # Limit to 5 recommendations


if __name__ == "__main__":
    # Test the condition assessor
    assessor = ConditionAssessor()
    print(f"✅ ConditionAssessor initialized")
    print(f"   Feature dimension: {assessor.feature_dim}")
