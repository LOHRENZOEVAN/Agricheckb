# train_model.py - Fixed version with proper NaN handling
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import json
from pathlib import Path
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EnhancedModelTrainer:
    """Train ML model using ALL available data sources"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = []
        self.price_data = None
        self.suitability_data = {}
        
    def load_external_data(self):
        """Load all external data sources"""
        print("Loading external data sources...")
        
        # 1. Load crop price data
        if os.path.exists('crop_prices.csv'):
            try:
                price_df = pd.read_csv('crop_prices.csv')
                print(f"✅ Loaded price data with {len(price_df)} records")
                self.price_data = self.process_price_data(price_df)
            except Exception as e:
                print(f"⚠️ Could not load price data: {e}")
                self.price_data = None
        
        # 2. Load soil suitability data for each crop
        for crop in ['maize', 'rice', 'soya']:
            suitability_file = f'suitability_data_{crop}.csv'
            if os.path.exists(suitability_file):
                try:
                    suit_df = pd.read_csv(suitability_file)
                    self.suitability_data[crop] = suit_df
                    print(f"✅ Loaded suitability data for {crop}: {len(suit_df)} points")
                except Exception as e:
                    print(f"⚠️ Could not load suitability for {crop}: {e}")
    
    def process_price_data(self, price_df):
        """Extract price features from raw price data"""
        price_features = {}
        
        try:
            # Convert date column
            if 'Date Range' in price_df.columns:
                price_df['Date'] = pd.to_datetime(price_df['Date Range'].str.split(' to ').str[0], errors='coerce')
            elif 'Date' in price_df.columns:
                price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
            
            # Calculate price statistics for each commodity
            if 'Commodity' in price_df.columns and 'Closing Price' in price_df.columns:
                for commodity in price_df['Commodity'].unique():
                    commodity_data = price_df[price_df['Commodity'] == commodity]['Closing Price']
                    
                    # Standardize commodity names
                    commodity_key = commodity.lower().replace('white ', '').replace('corn', 'maize')
                    
                    price_features[commodity_key] = {
                        'mean_price': commodity_data.mean(),
                        'price_volatility': commodity_data.std() / commodity_data.mean() if commodity_data.mean() > 0 else 0,
                        'price_trend': self.calculate_price_trend(price_df[price_df['Commodity'] == commodity]),
                        'recent_price': commodity_data.iloc[-1] if len(commodity_data) > 0 else commodity_data.mean()
                    }
            
            print(f"   Processed price features for: {list(price_features.keys())}")
            
        except Exception as e:
            print(f"   Error processing price data: {e}")
        
        return price_features
    
    def calculate_price_trend(self, commodity_df):
        """Calculate price trend"""
        if len(commodity_df) < 2:
            return 0
        
        try:
            prices = commodity_df['Closing Price'].values
            x = np.arange(len(prices))
            
            if len(prices) > 1:
                slope = np.polyfit(x, prices, 1)[0]
                mean_price = prices.mean()
                if mean_price > 0:
                    return slope / mean_price
        except:
            pass
        
        return 0
    
    def get_suitability_for_location(self, lat, lon, crop):
        """Get soil suitability for a specific location and crop"""
        if crop not in self.suitability_data or self.suitability_data[crop].empty:
            return 50  # Default medium suitability
        
        df = self.suitability_data[crop]
        
        # Calculate distances to all suitability points
        distances = np.sqrt((df['latitude'] - lat)**2 + (df['longitude'] - lon)**2)
        
        # Find nearest point
        nearest_idx = distances.idxmin()
        nearest_suitability = df.loc[nearest_idx, 'suitability']
        
        # Also get average of nearby points (within 0.5 degrees)
        nearby_mask = distances < 0.5
        if nearby_mask.any():
            nearby_avg = df.loc[nearby_mask, 'suitability'].mean()
            # Weighted average: 70% nearest, 30% nearby average
            return 0.7 * nearest_suitability + 0.3 * nearby_avg
        
        return nearest_suitability
    
    def fetch_climate_features(self, lat, lon):
        """Fetch climate data for a location (simplified for training)"""
        climate_features = {}
        
        # Zone-based climate patterns
        if lat > 9.0:  # Northern Ghana
            climate_features['avg_temp'] = 28 + np.random.normal(0, 2)
            climate_features['avg_rainfall'] = 900 + np.random.normal(0, 100)
            climate_features['drought_risk'] = 0.7
            climate_features['humidity'] = 45 + np.random.normal(0, 5)
        elif lat > 7.0:  # Middle belt
            climate_features['avg_temp'] = 26 + np.random.normal(0, 2)
            climate_features['avg_rainfall'] = 1200 + np.random.normal(0, 150)
            climate_features['drought_risk'] = 0.4
            climate_features['humidity'] = 65 + np.random.normal(0, 5)
        else:  # Southern Ghana
            climate_features['avg_temp'] = 25 + np.random.normal(0, 2)
            climate_features['avg_rainfall'] = 1500 + np.random.normal(0, 200)
            climate_features['drought_risk'] = 0.2
            climate_features['humidity'] = 75 + np.random.normal(0, 5)
        
        # Add seasonal variations
        climate_features['temp_variance'] = np.random.uniform(2, 6)
        climate_features['rainfall_variance'] = np.random.uniform(50, 200)
        climate_features['extreme_heat_days'] = np.random.poisson(5)
        climate_features['heavy_rain_days'] = np.random.poisson(8)
        
        return climate_features
    
    def prepare_features(self, df):
        """Convert raw data to comprehensive features for training"""
        features = pd.DataFrame()
        
        # === 1. LOCATION FEATURES ===
        features['latitude'] = df['latitude']
        features['longitude'] = df['longitude']
        
        # === 2. ZONE FEATURES ===
        features['is_northern'] = (df['latitude'] > 9.0).astype(int)
        features['is_middle_belt'] = ((df['latitude'] > 7.0) & (df['latitude'] <= 9.0)).astype(int)
        features['is_southern'] = (df['latitude'] <= 7.0).astype(int)
        
        # === 3. CROP TYPE ENCODING ===
        features['is_maize'] = (df['crop'] == 'maize').astype(int)
        features['is_rice'] = (df['crop'] == 'rice').astype(int)
        features['is_soya'] = (df['crop'] == 'soya').astype(int)
        
        # === 4. TEMPORAL FEATURES ===
        if 'season' in df.columns:
            features['year'] = pd.to_numeric(df['season'], errors='coerce').fillna(2021)
            features['years_from_2020'] = features['year'] - 2020
        else:
            features['year'] = 2021
            features['years_from_2020'] = 1
        
        # === 5. NDVI FEATURES (if available) ===
        if 'mean_ndvi' in df.columns:
            features['mean_ndvi'] = df['mean_ndvi'].fillna(0.5)
            features['has_ndvi'] = (~df['mean_ndvi'].isna()).astype(int)
            
            if 'max_ndvi' in df.columns:
                features['max_ndvi'] = df['max_ndvi'].fillna(0.5)
                features['min_ndvi'] = df['min_ndvi'].fillna(0.5)
                features['ndvi_range'] = features['max_ndvi'] - features['min_ndvi']
                
                if 'std_ndvi' in df.columns:
                    features['ndvi_variability'] = df['std_ndvi'].fillna(0.05)
        
        # === 6. SOIL SUITABILITY FEATURES ===
        print("Adding soil suitability features...")
        suitability_values = []
        for idx, row in df.iterrows():
            suit = self.get_suitability_for_location(
                row['latitude'], 
                row['longitude'], 
                row['crop']
            )
            suitability_values.append(suit)
        
        features['soil_suitability'] = suitability_values
        
        # FIX: Handle NaN values before creating categorical feature
        suitability_series = pd.Series(features['soil_suitability'])
        
        # Replace NaN with median suitability
        suitability_series = suitability_series.fillna(suitability_series.median())
        
        # Now create categories
        suitability_categories = pd.cut(
            suitability_series,
            bins=[0, 33, 66, 100],
            labels=[0, 1, 2],  # Low, Medium, High
            include_lowest=True
        )
        
        # Convert to numeric, filling any remaining NaN with 1 (medium)
        features['suitability_category'] = pd.to_numeric(suitability_categories, errors='coerce').fillna(1).astype(int)
        
        # === 7. PRICE & MARKET FEATURES ===
        print("Adding price features...")
        if self.price_data:
            price_features = []
            volatility_features = []
            trend_features = []
            
            for idx, row in df.iterrows():
                crop = row['crop']
                if crop in self.price_data:
                    price_features.append(self.price_data[crop]['mean_price'])
                    volatility_features.append(self.price_data[crop]['price_volatility'])
                    trend_features.append(self.price_data[crop]['price_trend'])
                else:
                    # Default values
                    price_features.append(100)
                    volatility_features.append(0.2)
                    trend_features.append(0)
            
            features['crop_price'] = price_features
            features['price_volatility'] = volatility_features
            features['price_trend'] = trend_features
            
            # Normalize prices
            if features['crop_price'].std() > 0:
                features['price_normalized'] = (features['crop_price'] - features['crop_price'].mean()) / features['crop_price'].std()
            else:
                features['price_normalized'] = 0
        
        # === 8. CLIMATE FEATURES ===
        print("Adding climate features...")
        climate_data = []
        for idx, row in df.iterrows():
            climate = self.fetch_climate_features(row['latitude'], row['longitude'])
            climate_data.append(climate)
        
        climate_df = pd.DataFrame(climate_data)
        for col in climate_df.columns:
            features[f'climate_{col}'] = climate_df[col]
        
        # === 9. INTERACTION FEATURES ===
        # Suitability × NDVI interaction
        if 'mean_ndvi' in features.columns:
            features['suitability_ndvi_interaction'] = features['soil_suitability'] * features['mean_ndvi'] / 100
        
        # Climate × Crop type interactions
        features['temp_maize_interaction'] = features.get('climate_avg_temp', 26) * features['is_maize']
        features['rainfall_rice_interaction'] = features.get('climate_avg_rainfall', 1200) * features['is_rice'] / 1000
        
        # Price × Volatility interaction
        if 'crop_price' in features.columns:
            features['price_risk'] = features['price_volatility'] * features.get('price_normalized', 0)
        
        # === 10. RISK INDICATORS ===
        # Composite environmental risk
        features['environmental_risk'] = (
            features.get('climate_drought_risk', 0.5) * 0.4 +
            (1 - features['soil_suitability']/100) * 0.3 +
            features.get('climate_temp_variance', 4) / 10 * 0.3
        )
        
        # Replace any infinite values with 0
        features = features.replace([np.inf, -np.inf], 0)
        
        # Store feature names
        self.feature_columns = features.columns.tolist()
        
        print(f"Total features created: {len(self.feature_columns)}")
        
        return features
    
    def calculate_risk_scores(self, df):
        """Calculate risk scores from yield data"""
        expected_yields = {'maize': 1.8, 'rice': 2.5, 'soya': 1.2}
        
        risk_scores = []
        for idx, row in df.iterrows():
            crop = row['crop']
            yield_val = row['yield']
            
            expected = expected_yields.get(crop, 1.5)
            
            # Convert yield to risk
            yield_ratio = yield_val / expected
            risk = 1.0 - (yield_ratio / 2.0)
            risk = np.clip(risk, 0.0, 1.0)
            
            risk_scores.append(risk)
        
        return np.array(risk_scores)
    
    def train(self, data_path='datasets/combined_training_data.csv'):
        """Train the model on prepared data"""
        print("="*60)
        print("ENHANCED MODEL TRAINING WITH ALL DATA SOURCES")
        print("="*60)
        
        # Load external data sources
        self.load_external_data()
        
        # Load training data
        print("\nLoading training data...")
        df = pd.read_csv(data_path)
        print(f"Training with {len(df)} samples")
        
        # Prepare comprehensive features
        X = self.prepare_features(df)
        
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Calculate target risk scores
        y = self.calculate_risk_scores(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Train model
        print("\nTraining Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Cross-validation
        print("Performing cross-validation...")
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        
        print(f"\n📊 Model Performance:")
        print(f"   Training R2 Score: {train_score:.3f}")
        print(f"   Test R2 Score: {test_score:.3f}")
        print(f"   CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Top 10 Important Features:")
        for idx, row in importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Save model
        self.save_model()
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': importance.to_dict('records'),
            'n_features': len(self.feature_columns)
        }
    
    def save_model(self):
        """Save trained model and metadata"""
        Path('models').mkdir(exist_ok=True)
        
        # Save model components
        joblib.dump(self.model, 'models/crop_risk_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.imputer, 'models/imputer.pkl')
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'model_type': 'RandomForestRegressor',
            'trained_date': datetime.now().isoformat(),
            'n_features': len(self.feature_columns),
            'data_sources': {
                'location': True,
                'zones': True,
                'ndvi': True,
                'soil_suitability': True,
                'price_data': self.price_data is not None,
                'climate': True
            }
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save price data for prediction use
        if self.price_data:
            # Convert numpy values to Python native types for JSON serialization
            price_data_serializable = {}
            for crop, data in self.price_data.items():
                price_data_serializable[crop] = {
                    k: float(v) if isinstance(v, np.number) else v 
                    for k, v in data.items()
                }
            
            with open('models/price_features.json', 'w') as f:
                json.dump(price_data_serializable, f, indent=2)
        
        # Save suitability data references
        suitability_refs = {crop: f"suitability_data_{crop}.csv" for crop in self.suitability_data.keys()}
        with open('models/suitability_refs.json', 'w') as f:
            json.dump(suitability_refs, f, indent=2)
        
        print("\n✅ Model and metadata saved to models/")
    
    def predict(self, latitude, longitude, crop):
        """Make a prediction for new data"""
        if self.model is None:
            # Load model and preprocessors
            self.model = joblib.load('models/crop_risk_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.imputer = joblib.load('models/imputer.pkl')
            
            with open('models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
                self.feature_columns = metadata['feature_columns']
            
            # Load price features if available
            if os.path.exists('models/price_features.json'):
                with open('models/price_features.json', 'r') as f:
                    self.price_data = json.load(f)
            
            # Load suitability data
            if os.path.exists('models/suitability_refs.json'):
                with open('models/suitability_refs.json', 'r') as f:
                    refs = json.load(f)
                    for crop_type, file_path in refs.items():
                        if os.path.exists(file_path):
                            self.suitability_data[crop_type] = pd.read_csv(file_path)
        
        # Create a DataFrame with the input including ALL NDVI columns
        # Generate realistic NDVI values based on location
        base_ndvi = 0.5  # Default
        if latitude > 9.0:  # Northern - lower NDVI
            base_ndvi = 0.4
        elif latitude > 7.0:  # Middle belt
            base_ndvi = 0.5
        else:  # Southern - higher NDVI
            base_ndvi = 0.6
            
        # Adjust for crop type
        if crop == 'rice':
            base_ndvi *= 1.1
        elif crop == 'soya':
            base_ndvi *= 0.95
            
        data = pd.DataFrame([{
            'latitude': latitude,
            'longitude': longitude,
            'crop': crop,
            'yield': 1.5,  # Dummy value for feature generation
            'season': 2024,
            'mean_ndvi': base_ndvi,
            'max_ndvi': min(base_ndvi + 0.1, 1.0),
            'min_ndvi': max(base_ndvi - 0.1, 0.0),
            'std_ndvi': 0.05
        }])
        
        # Generate all features
        features = self.prepare_features(data)
        
        # Ensure all expected features are present
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
        
        # Select and order features correctly
        features = features[self.feature_columns]
        
        # Preprocess
        features_imputed = self.imputer.transform(features)
        features_scaled = self.scaler.transform(features_imputed)
        
        # Make prediction
        risk_score = self.model.predict(features_scaled)[0]
        
        # Get prediction uncertainty
        predictions = [tree.predict(features_scaled)[0] for tree in self.model.estimators_]
        uncertainty = np.std(predictions)
        
        return {
            'risk_score': float(risk_score),
            'risk_level': 'High' if risk_score > 0.7 else 'Moderate' if risk_score > 0.4 else 'Low',
            'uncertainty': float(uncertainty),
            'confidence': float(1.0 - uncertainty),
            'features_used': len(self.feature_columns)
        }

# Usage
if __name__ == "__main__":
    trainer = EnhancedModelTrainer()
    results = trainer.train()
    
    # Test prediction
    test_prediction = trainer.predict(
        latitude=5.6,
        longitude=-0.2,
        crop='maize'
    )
    print(f"\n🔮 Test prediction for Accra, Maize:")
    print(f"   Risk Score: {test_prediction['risk_score']:.3f}")
    print(f"   Risk Level: {test_prediction['risk_level']}")
    print(f"   Confidence: {test_prediction['confidence']:.1%}")
    print(f"   Features Used: {test_prediction['features_used']}")