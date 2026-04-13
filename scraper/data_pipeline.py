"""
Data Pipeline - Converts scraped JSON to CSV
"""

import pandas as pd
import json
import os
import numpy as np
from glob import glob

class DataPipeline:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
    
    def load_json_files(self):
        """Load all JSON files from raw directory"""
        all_properties = []
        
        json_files = glob(f"{self.raw_dir}/*.json")
        
        for filepath in json_files:
            print(f"📂 Loading: {filepath}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_properties.extend(data)
                    else:
                        all_properties.append(data)
            except Exception as e:
                print(f"   Error: {e}")
        
        print(f"✅ Loaded {len(all_properties)} properties")
        return all_properties
    
    def to_dataframe(self, properties):
        """Convert to DataFrame"""
        df = pd.DataFrame(properties)
        
        # Ensure all columns exist
        required_cols = ['title', 'price', 'description', 'bedrooms', 
                         'bathrooms', 'area_sqm', 'property_type', 'furnishing', 'location']
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        return df
    
    def clean_data(self, df):
        """Clean and prepare data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['url'], keep='first')
        
        # Remove invalid prices
        df = df[(df['price'] > 50000) & (df['price'] < 100000000)]
        
        # Fill missing values
        df['bedrooms'] = df['bedrooms'].fillna(0).astype(int)
        df['bathrooms'] = df['bathrooms'].fillna(1).astype(int)
        df['area_sqm'] = df['area_sqm'].fillna(df['area_sqm'].median() if not df['area_sqm'].isna().all() else 100)
        df['furnishing'] = df['furnishing'].fillna('Unfurnished')
        df['property_type'] = df['property_type'].fillna('Apartment')
        
        # Calculate derived features
        df['price_per_sqm'] = df['price'] / df['area_sqm'].clip(lower=1)
        df['log_price'] = np.log1p(df['price'])
        
        # Add city extraction
        df['city'] = df['location'].apply(self._extract_city)
        
        return df
    
    def _extract_city(self, location):
        """Extract city from location string"""
        if not isinstance(location, str):
            return 'Cairo'
        
        cities = ['Cairo', 'Alexandria', 'Giza', 'New Cairo', 'Sheikh Zayed', 
                  'October', 'Maadi', 'Heliopolis', 'Nasr City']
        
        for city in cities:
            if city.lower() in location.lower():
                return city
        return 'Cairo'
    
    def save_csv(self, df, filename="final_dataset.csv"):
        """Save to CSV"""
        filepath = os.path.join(self.processed_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"💾 Saved to {filepath}")
        return filepath
    
    def run(self):
        """Run full pipeline"""
        print("="*60)
        print("🔄 DATA PIPELINE")
        print("="*60)
        
        properties = self.load_json_files()
        
        if not properties:
            print("❌ No data found!")
            return None
        
        df = self.to_dataframe(properties)
        print(f"📊 Initial shape: {df.shape}")
        
        df = self.clean_data(df)
        print(f"📊 After cleaning: {df.shape}")
        
        self.save_csv(df)
        
        print(f"\n📈 Summary:")
        print(f"   Total: {len(df)} properties")
        print(f"   Price range: {df['price'].min():,.0f} - {df['price'].max():,.0f} EGP")
        print(f"   Average price: {df['price'].mean():,.0f} EGP")
        
        return df


if __name__ == "__main__":
    pipeline = DataPipeline()
    df = pipeline.run()