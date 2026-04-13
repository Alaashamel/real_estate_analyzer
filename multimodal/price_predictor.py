"""
Complete Price Prediction Pipeline - FINAL FIXED VERSION
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class PricePredictor:
    """Simplified but effective price predictor"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.text_vectorizer = None
        self.model = None
    
    def prepare_features(self, df):
        """Prepare all features for training"""
        print("📊 Preparing features...")
        
        # 1. Text features (TF-IDF)
        print("   📝 Processing text features...")
        texts = df['description'].fillna('').tolist()
        
        self.text_vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
        text_features = self.text_vectorizer.fit_transform(texts).toarray()
        
        # 2. Tabular features
        print("   📊 Processing tabular features...")
        tabular_features = self._prepare_tabular_features(df)
        
        # 3. Target (log transform for better distribution)
        prices = df['price'].values
        log_prices = np.log1p(prices)
        
        # Combine all features
        X = np.hstack([text_features, tabular_features])
        
        print(f"\n✅ Feature preparation complete!")
        print(f"   Total features: {X.shape[1]}")
        print(f"   Price range: {prices.min():,.0f} - {prices.max():,.0f} EGP")
        print(f"   Log price range: {log_prices.min():.2f} - {log_prices.max():.2f}")
        
        return X, log_prices, prices
    
    def _prepare_tabular_features(self, df):
        """Prepare tabular features"""
        features = []
        
        # Numeric features
        features.append(df['area_sqm'].fillna(df['area_sqm'].median()).values)
        features.append(df['bedrooms'].fillna(0).values)
        features.append(df['bathrooms'].fillna(1).values)
        
        # Property type (one-hot)
        type_dummies = pd.get_dummies(df['property_type'], prefix='type')
        for col in type_dummies.columns:
            features.append(type_dummies[col].values)
        
        # Furnishing (one-hot)
        furn_dummies = pd.get_dummies(df['furnishing'], prefix='furn')
        for col in furn_dummies.columns:
            features.append(furn_dummies[col].values)
        
        # Price per sqm (derived feature)
        price_per_sqm = df['price'] / df['area_sqm'].clip(lower=1)
        features.append(price_per_sqm.fillna(0).values)
        
        # Log area
        log_area = np.log1p(df['area_sqm'].fillna(100))
        features.append(log_area.values)
        
        X = np.column_stack(features)
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def train(self, df, test_size=0.2):
        """Train the model"""
        print("\n" + "="*60)
        print("🏋️ Training Random Forest Model")
        print("="*60)
        
        X, y_log, y_original = self.prepare_features(df)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_log, test_size=test_size, random_state=42
        )
        
        # Train Random Forest
        print("\n🔧 Training Random Forest Regressor...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_log = self.model.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        y_actual = np.expm1(y_val)
        
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        print(f"\n📊 Evaluation Results:")
        print(f"   MAE: {mae:,.0f} EGP")
        print(f"   RMSE: {rmse:,.0f} EGP")
        print(f"   R² Score: {r2:.4f}")
        
        # Feature importance
        self._show_feature_importance(X_train.shape[1])
        
        return self.model
    
    def _show_feature_importance(self, n_features):
        """Show top features"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_idx = np.argsort(importances)[-10:][::-1]
            
            print(f"\n📈 Top 10 Features:")
            for i, idx in enumerate(top_idx[:10]):
                print(f"   {i+1}. Feature {idx}: {importances[idx]:.4f}")
    
    def predict(self, description, area_sqm, bedrooms, bathrooms, 
                property_type="Apartment", furnishing="Unfurnished"):
        """Predict price for a single property"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Text features
        text_features = self.text_vectorizer.transform([description]).toarray()
        
        # Tabular features
        tabular_row = self._prepare_single_tabular(
            area_sqm, bedrooms, bathrooms, property_type, furnishing
        )
        
        # Combine
        X = np.hstack([text_features, tabular_row])
        
        # Predict
        price_log = self.model.predict(X)[0]
        price = np.expm1(price_log)
        
        return float(price)
    
    def _prepare_single_tabular(self, area_sqm, bedrooms, bathrooms, property_type, furnishing):
        """Prepare tabular features for a single property"""
        # Numeric features
        features = [area_sqm, bedrooms, bathrooms]
        
        # Property type (one-hot)
        type_cols = ['Apartment', 'Villa', 'Townhouse', 'Studio', 'Duplex']
        for t in type_cols:
            features.append(1 if property_type == t else 0)
        
        # Furnishing (one-hot)
        furn_cols = ['Unfurnished', 'Semi-furnished', 'Furnished']
        for f in furn_cols:
            features.append(1 if furnishing == f else 0)
        
        # Derived features
        price_per_sqm = 0  # Will be scaled
        features.append(price_per_sqm)
        
        log_area = np.log1p(area_sqm)
        features.append(log_area)
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def save(self, path='models/price_predictor.pkl'):
        """Save the model"""
        os.makedirs('models', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'text_vectorizer': self.text_vectorizer,
            }, f)
        print(f"✅ Model saved to {path}")
    
    def load(self, path='models/price_predictor.pkl'):
        """Load the model"""
        if not os.path.exists(path):
            print(f"⚠️ Model not found at {path}")
            return False
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.text_vectorizer = data['text_vectorizer']
        
        print(f"✅ Model loaded from {path}")
        return True


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🏠 REAL ESTATE PRICE PREDICTOR")
    print("="*60)
    
    data_path = 'data/processed/final_dataset.csv'
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"📂 Loaded {len(df)} properties from {data_path}")
        
        # Show data summary
        print(f"\n📊 Data Summary:")
        print(f"   Price range: {df['price'].min():,.0f} - {df['price'].max():,.0f} EGP")
        print(f"   Average price: {df['price'].mean():,.0f} EGP")
        print(f"   Median price: {df['price'].median():,.0f} EGP")
        print(f"   Property types: {df['property_type'].value_counts().to_dict()}")
        
        # Train predictor
        predictor = PricePredictor()
        predictor.train(df)
        predictor.save()
        
        # Test prediction
        test_price = predictor.predict(
            description="Luxury apartment in Cairo with sea view, fully furnished, 3 bedrooms, 150 sqm",
            area_sqm=150,
            bedrooms=3,
            bathrooms=2,
            property_type="Apartment",
            furnishing="Furnished"
        )
        print(f"\n🏠 Test prediction: {test_price:,.0f} EGP")
        
        # Test another example
        test_price2 = predictor.predict(
            description="Small studio apartment in Cairo, unfurnished, 40 sqm",
            area_sqm=40,
            bedrooms=0,
            bathrooms=1,
            property_type="Studio",
            furnishing="Unfurnished"
        )
        print(f"🏠 Test prediction (studio): {test_price2:,.0f} EGP")
        
    else:
        print("❌ No data found!")
        print("   Run: python scraper/data_pipeline.py first")