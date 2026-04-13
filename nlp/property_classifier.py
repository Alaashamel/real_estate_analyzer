"""
Property Type Classifier using NLP
Classifies property type from text description
"""

import re
from typing import Dict, List, Tuple, Optional
import numpy as np

class PropertyTypeClassifier:
    """
    Classify property type from text description
    Uses keyword matching + scoring system
    """
    
    def __init__(self):
        # Keywords for each property type
        self.type_keywords = {
            'Apartment': [
                'apartment', 'flat', 'شقة', 'penthouse', 'duplex',
                'condo', 'condominium', 'unit', 'residence'
            ],
            'Villa': [
                'villa', 'فيلا', 'palace', 'mansion', 'قصر',
                'estate', 'manor', 'chateau'
            ],
            'Townhouse': [
                'townhouse', 'town house', 'تاون هاوس', 'row house',
                'terrace house', 'cluster'
            ],
            'Studio': [
                'studio', 'ستوديو', 'efficiency', 'bachelor',
                'studio apartment', 'loft'
            ],
            'Duplex': [
                'duplex', 'دوبلكس', 'two story', 'multi-level',
                'split level', 'maisonette'
            ],
            'Commercial': [
                'commercial', 'shop', 'store', 'office', 'تجاري',
                'retail', 'business', 'showroom', 'warehouse'
            ]
        }
        
        # Price indicators
        self.price_indicators = {
            'high': ['luxury', 'premium', 'exclusive', 'فاخر', 'lucury', 'high-end', 'elite'],
            'mid': ['moderate', 'reasonable', 'متوسط', 'affordable', 'value'],
            'low': ['cheap', 'economy', 'budget', 'رخيص', 'low price', 'affordable']
        }
        
        # Quality indicators
        self.quality_indicators = {
            'excellent': ['excellent', 'perfect', 'immaculate', 'ممتاز', 'pristine', 'like new'],
            'good': ['good', 'well-maintained', 'جيد', 'clean', 'nice', 'decent'],
            'average': ['average', 'fair', 'متوسط', 'okay', 'standard'],
            'poor': ['poor', 'bad', 'سيء', 'needs work', 'fixer upper', 'renovation needed']
        }
    
    def classify(self, text: str) -> Dict:
        """
        Classify property type from text
        
        Args:
            text: Property description text
            
        Returns:
            Dictionary with classification results
        """
        if not isinstance(text, str):
            text = ""
        
        text_lower = text.lower()
        
        # Calculate scores for each type
        scores = {}
        for prop_type, keywords in self.type_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Longer keywords get higher weight
                    weight = len(keyword) / 10
                    score += weight
            scores[prop_type] = score
        
        # Get the best match
        if max(scores.values()) == 0:
            predicted_type = 'Apartment'  # Default
            confidence = 0
        else:
            predicted_type = max(scores, key=scores.get)
            confidence = min(1.0, scores[predicted_type] / 3.0)
        
        return {
            'property_type': predicted_type,
            'confidence': confidence,
            'all_scores': scores,
            'matched_keywords': self._get_matched_keywords(text_lower, predicted_type)
        }
    
    def _get_matched_keywords(self, text: str, prop_type: str) -> List[str]:
        """Get keywords that matched for the predicted type"""
        matched = []
        for keyword in self.type_keywords.get(prop_type, []):
            if keyword in text:
                matched.append(keyword)
        return matched
    
    def extract_price_indicators(self, text: str) -> Dict:
        """
        Extract price indicators from text
        
        Returns:
            Dictionary with price level and confidence
        """
        if not isinstance(text, str):
            text = ""
        
        text_lower = text.lower()
        
        scores = {'high': 0, 'mid': 0, 'low': 0}
        
        for level, keywords in self.price_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[level] += 1
        
        # Determine price level
        total = sum(scores.values())
        if total == 0:
            price_level = 'unknown'
            confidence = 0
        else:
            price_level = max(scores, key=scores.get)
            confidence = scores[price_level] / total
        
        return {
            'price_level': price_level,
            'confidence': confidence,
            'scores': scores
        }
    
    def extract_quality(self, text: str) -> Dict:
        """
        Extract quality indicators from text
        
        Returns:
            Dictionary with quality level
        """
        if not isinstance(text, str):
            text = ""
        
        text_lower = text.lower()
        
        scores = {'excellent': 0, 'good': 0, 'average': 0, 'poor': 0}
        
        for quality, keywords in self.quality_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[quality] += 1
        
        # Determine quality level
        total = sum(scores.values())
        if total == 0:
            quality = 'unknown'
            confidence = 0
        else:
            quality = max(scores, key=scores.get)
            confidence = scores[quality] / total
        
        return {
            'quality_level': quality,
            'confidence': confidence,
            'scores': scores
        }
    
    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """Classify multiple texts"""
        return [self.classify(text) for text in texts]
    
    def get_type_encoding(self, prop_type: str) -> int:
        """Get numeric encoding for property type"""
        encoding = {
            'Apartment': 0,
            'Villa': 1,
            'Townhouse': 2,
            'Studio': 3,
            'Duplex': 4,
            'Commercial': 5
        }
        return encoding.get(prop_type, 0)
    
    def get_price_level_encoding(self, price_level: str) -> int:
        """Get numeric encoding for price level"""
        encoding = {
            'low': 0,
            'mid': 1,
            'high': 2,
            'unknown': 1
        }
        return encoding.get(price_level, 1)
    
    def get_quality_encoding(self, quality: str) -> float:
        """Get numeric score for quality level"""
        scores = {
            'excellent': 1.0,
            'good': 0.75,
            'average': 0.5,
            'poor': 0.25,
            'unknown': 0.5
        }
        return scores.get(quality, 0.5)


# Convenience functions
def classify_property_type(text: str) -> str:
    """Quick function to classify property type"""
    classifier = PropertyTypeClassifier()
    return classifier.classify(text)['property_type']


def extract_price_indicators(text: str) -> Dict:
    """Quick function to extract price indicators"""
    classifier = PropertyTypeClassifier()
    return classifier.extract_price_indicators(text)


if __name__ == "__main__":
    # Test the classifier
    classifier = PropertyTypeClassifier()
    
    test_texts = [
        "Luxury apartment for sale in Cairo with 3 bedrooms and sea view",
        "Beautiful villa with private pool and garden in New Cairo",
        "Studio apartment fully furnished in downtown",
        "Townhouse for sale in Sheikh Zayed with 4 bedrooms"
    ]
    
    for text in test_texts:
        result = classifier.classify(text)
        print(f"\n📝 Text: {text[:50]}...")
        print(f"   Type: {result['property_type']} (confidence: {result['confidence']:.2f})")
        print(f"   Matched: {result['matched_keywords']}")
    
    # Test price indicators
    print("\n" + "="*50)
    print("💰 Price Indicators Test")
    print("="*50)
    
    price_text = "Luxury premium apartment with high-end finishes"
    indicators = classifier.extract_price_indicators(price_text)
    print(f"Text: {price_text}")
    print(f"Price level: {indicators['price_level']} (confidence: {indicators['confidence']:.2f})")