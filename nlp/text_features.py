"""
Natural Language Processing for property descriptions
Extracts features: furnishing, amenities, property type, condition
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class PropertyTextAnalyzer:
    """Extract structured features from property descriptions"""
    
    def __init__(self):
        # Amenity keywords
        self.amenity_keywords = {
            'elevator': ['elevator', 'lift', 'ascenseur', 'مصعد'],
            'parking': ['parking', 'garage', 'car park', 'موقف', 'جراج'],
            'security': ['security', 'guard', 'cctv', 'أمن', 'كاميرات'],
            'pool': ['swimming pool', 'pool', 'حمام سباحة', 'بركة'],
            'gym': ['gym', 'fitness', 'health club', 'جيم', 'لياقة'],
            'garden': ['garden', 'yard', 'landscape', 'حديقة'],
            'ac': ['air conditioning', 'ac', 'a/c', 'تكييف'],
            'maid': ['maid room', 'driver room', 'غرفة خادمة', 'سائق'],
            'balcony': ['balcony', 'terrace', 'شرفة', 'تراس'],
            'view': ['sea view', 'landmark view', 'panoramic', 'اطلالة'],
            'furnished': ['fully furnished', 'furnished', 'مفروش', 'جهز'],
            'new': ['new', 'modern', 'recent', 'جديد', 'مودرن'],
            'renovated': ['renovated', 'refurbished', 'مجددة', 'تجديد']
        }
        
        # Condition indicators
        self.condition_keywords = {
            'excellent': ['excellent', 'perfect', 'luxury', 'ممتاز', 'فاخر', 'لوكس'],
            'good': ['good', 'well-maintained', 'جيد', 'نظيف'],
            'average': ['average', 'decent', 'متوسط', 'مقبول'],
            'needs_renovation': ['needs renovation', 'fixer upper', 'renovation needed', 'يحتاج تجديد']
        }
    
    def extract_amenities(self, text):
        """Extract amenities present in the description"""
        if not isinstance(text, str):
            return {}
        
        text_lower = text.lower()
        amenities = {}
        
        for amenity, keywords in self.amenity_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    amenities[amenity] = 1
                    break
            else:
                amenities[amenity] = 0
        
        return amenities
    
    def extract_condition_score(self, text):
        """Extract property condition score (0-1)"""
        if not isinstance(text, str):
            return 0.5
        
        text_lower = text.lower()
        scores = {'excellent': 0, 'good': 0, 'average': 0, 'needs_renovation': 0}
        
        for condition, keywords in self.condition_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    scores[condition] += 1
        
        total = sum(scores.values())
        if total == 0:
            return 0.5
        
        # Weighted score
        weighted = (scores['excellent'] * 1.0 + 
                   scores['good'] * 0.7 + 
                   scores['average'] * 0.4 + 
                   scores['needs_renovation'] * 0.1)
        
        return weighted / max(total, 1)
    
    def extract_floor_info(self, text):
        """Extract floor number if mentioned"""
        if not isinstance(text, str):
            return None
        
        patterns = [
            r'floor\s*(\d+)',
            r'(\d+)(?:st|nd|rd|th)\s+floor',
            r'الطابق\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        
        return None
    
    def extract_all_features(self, df):
        """Extract all text features for the entire dataframe"""
        
        amenities_list = []
        condition_scores = []
        floor_numbers = []
        
        for idx, row in df.iterrows():
            text = row.get('description', '')
            if pd.isna(text):
                text = row.get('title', '')
            
            amenities = self.extract_amenities(text)
            amenities_list.append(amenities)
            condition_scores.append(self.extract_condition_score(text))
            floor_numbers.append(self.extract_floor_info(text))
        
        # Convert amenities to DataFrame
        amenities_df = pd.DataFrame(amenities_list)
        
        # Add new columns
        df['condition_score'] = condition_scores
        df['floor'] = floor_numbers
        
        # Merge amenities
        for col in amenities_df.columns:
            df[f'has_{col}'] = amenities_df[col].values
        
        return df
    
    def create_tfidf_features(self, texts, max_features=500):
        """Create TF-IDF features from descriptions"""
        if isinstance(texts, pd.Series):
            texts = texts.fillna('').tolist()
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Save vectorizer
        os.makedirs('models', exist_ok=True)
        with open('models/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        return tfidf_matrix, vectorizer


class PropertyTypeClassifier:
    """Classify property type from text description"""
    
    def __init__(self):
        self.type_keywords = {
            'Apartment': ['apartment', 'flat', 'شقة', 'penthouse', 'duplex'],
            'Villa': ['villa', 'فيلا', 'palace', 'mansion', 'قصر'],
            'Townhouse': ['townhouse', 'town house', 'تاون هاوس', 'row house'],
            'Studio': ['studio', 'ستوديو', 'efficiency'],
            'Commercial': ['commercial', 'shop', 'office', 'store', 'تجاري', 'محل']
        }
    
    def classify(self, text):
        """Classify property type from text"""
        if not isinstance(text, str):
            return 'Apartment'
        
        text_lower = text.lower()
        scores = {}
        
        for prop_type, keywords in self.type_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            scores[prop_type] = score
        
        if max(scores.values()) == 0:
            return 'Apartment'
        
        return max(scores, key=scores.get)
    
    def classify_batch(self, texts):
        """Classify multiple texts"""
        return [self.classify(text) for text in texts]


if __name__ == "__main__":
    # Test the NLP module
    analyzer = PropertyTextAnalyzer()
    
    sample_text = """
    Luxury apartment for sale in Cairo. Fully furnished with sea view.
    Building has elevator, parking, 24/7 security, swimming pool and gym.
    Excellent condition, newly renovated. 3 bedrooms, 2 bathrooms.
    """
    
    print("Sample text:", sample_text)
    print("\nAmenities:", analyzer.extract_amenities(sample_text))
    print("Condition score:", analyzer.extract_condition_score(sample_text))
    print("Floor:", analyzer.extract_floor_info(sample_text))
    
    classifier = PropertyTypeClassifier()
    print("Property type:", classifier.classify(sample_text))