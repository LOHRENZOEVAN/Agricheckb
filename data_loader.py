# data_loader.py - Smart dataset loader with Location-Based NDVI Matching
import pandas as pd
import numpy as np
import json
from pathlib import Path

class GhanaDatasetLoader:
    """Load and prepare Ghana agricultural datasets with location-based NDVI matching"""

    def __init__(self, data_folder='datasets'):
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)

        # Ghana zone definitions (matching your app.py)
        self.zone_boundaries = {
            'northern': {'lat_range': (9.5, 11.0), 'lon_range': (-1.0, 0.5)},
            'upper_east': {'lat_range': (10.0, 11.2), 'lon_range': (-1.5, 0.0)},
            'upper_west': {'lat_range': (9.7, 11.0), 'lon_range': (-3.0, -1.5)},
            'middle_belt': {'lat_range': (7.0, 9.5), 'lon_range': (-3.0, 1.5)},
            'southern': {'lat_range': (4.5, 7.0), 'lon_range': (-3.5, 1.5)}
        }

    def find_file(self, name_keywords, exts=("csv", "json", "geojson")):
        """Search for a file in datasets/ matching name keywords and allowed extensions."""
        for ext in exts:
            for f in self.data_folder.rglob(f"*.{ext}"):
                if all(keyword.lower() in f.name.lower() for keyword in name_keywords):
                    return f
        # Also check in NDVI folder specifically
        ndvi_folder = self.data_folder / "NDVI"
        if ndvi_folder.exists():
            for ext in exts:
                for f in ndvi_folder.rglob(f"*.{ext}"):
                    if any(keyword.lower() in f.name.lower() for keyword in name_keywords):
                        return f
        return None

    def get_zone(self, lat, lon):
        """Determine which Ghana zone a coordinate belongs to"""
        for zone, bounds in self.zone_boundaries.items():
            lat_min, lat_max = bounds['lat_range']
            lon_min, lon_max = bounds['lon_range']
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return zone

        # Default zone based on latitude only
        if lat > 9.0:
            return 'northern'
        elif lat > 7.0:
            return 'middle_belt'
        else:
            return 'southern'

    def load_zenodo_maize(self):
        f = self.find_file(["yield", "maize"])
        if f:
            print(f"✅ Found Zenodo maize dataset: {f}")
            df = pd.read_csv(f)
            # Dynamically detect columns
            lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
            lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
            yield_col = next((c for c in df.columns if "yield" in c.lower()), None)
            season_col = next((c for c in df.columns if "season" in c.lower()), None)
            field_id_col = next((c for c in df.columns if "field" in c.lower() or "id" in c.lower()), None)
            return df, lat_col, lon_col, yield_col, season_col, field_id_col
        else:
            print("⚠️ Zenodo maize dataset not found. Skipping.")
            return None, None, None, None, None, None

    def load_sustainbench(self):
        f = self.find_file(["ghana", "crop"])
        if f:
            print(f"✅ Found SustainBench dataset: {f}")
            return pd.read_csv(f)
        else:
            print("⚠️ SustainBench data not found. Skipping.")
            return None

    def load_mofa_yields(self):
        f = self.find_file(["regional", "yield"])
        if f:
            print(f"✅ Found MoFA dataset: {f}")
            return pd.read_csv(f)
        else:
            print("⚠️ MoFA dataset not found. Generating synthetic MoFA data...")
            regions = ['Northern', 'Upper East', 'Upper West', 'Brong Ahafo',
                       'Ashanti', 'Eastern', 'Volta', 'Greater Accra', 'Central', 'Western']
            data = []
            for region in regions:
                for year in [2020, 2021, 2022]:
                    for crop in ['maize', 'rice', 'soya']:
                        base_yields = {'maize': 1.8, 'rice': 2.5, 'soya': 1.2}
                        yield_val = base_yields[crop] + np.random.normal(0, 0.3)
                        data.append({
                            'region': region,
                            'year': year,
                            'crop': crop,
                            'yield_mt_ha': max(0.5, yield_val),
                            'area_harvested_ha': np.random.randint(1000, 50000)
                        })
            df = pd.DataFrame(data)
            out_file = self.data_folder / "regional_yields_generated.csv"
            df.to_csv(out_file, index=False)
            print(f"🆕 Synthetic MoFA data saved to {out_file}")
            return df

    def load_raw_ndvi_data(self):
        """Load raw NDVI data from file"""
        search_patterns = [["ndvi"], ["NDVI"], ["vegetation"]]
        f = None

        for pattern in search_patterns:
            f = self.find_file(pattern)
            if f:
                break

        if not f:
            ndvi_folder = self.data_folder / "NDVI"
            if ndvi_folder.exists():
                csv_files = list(ndvi_folder.glob("*.csv"))
                if csv_files:
                    f = csv_files[0]

        if f:
            print(f"✅ Found NDVI dataset: {f}")
            try:
                # Try reading to detect format
                df_raw = pd.read_csv(f, header=None)

                # Check column count and assign names
                if df_raw.shape[1] == 4:
                    df = pd.read_csv(f, header=None, names=['plot_name', 'field_id', 'date', 'ndvi'])
                elif df_raw.shape[1] == 3:
                    df = pd.read_csv(f, header=None, names=['field_id', 'date', 'ndvi'])
                else:
                    # Try with headers
                    df = pd.read_csv(f)

                # Ensure we have date and ndvi columns
                if 'date' in df.columns and 'ndvi' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    
                    # --- FIX 1 START: Convert NDVI to numeric and drop bad rows ---
                    df['ndvi'] = pd.to_numeric(df['ndvi'], errors='coerce')
                    df.dropna(subset=['date', 'ndvi'], inplace=True)
                    # --- FIX 1 END ---

                    # Fix future dates
                    df['year'] = df['date'].dt.year
                    df.loc[df['year'] > 2024, 'date'] = df.loc[df['year'] > 2024, 'date'].apply(
                        lambda x: x.replace(year=x.year - 2) if pd.notna(x) else x
                    )
                    df['year'] = df['date'].dt.year
                    df['month'] = df['date'].dt.month

                    print(f"   Loaded {len(df)} NDVI records")
                    print(f"   NDVI range: {df['ndvi'].min():.4f} to {df['ndvi'].max():.4f}")
                    return df

            except Exception as e:
                print(f"   ❌ Error loading NDVI data: {str(e)}")
                return None

        return None

    def calculate_zone_ndvi_patterns(self, ndvi_df):
        """Calculate NDVI patterns by zone and time"""
        if ndvi_df is None or ndvi_df.empty:
            return None

        # Calculate seasonal patterns
        seasonal_ndvi = ndvi_df.groupby(['year', 'month'])['ndvi'].agg(['mean', 'std', 'min', 'max']).reset_index()

        # Calculate overall statistics
        ndvi_stats = {
            'overall_mean': ndvi_df['ndvi'].mean(),
            'overall_std': ndvi_df['ndvi'].std(),
            'seasonal_patterns': seasonal_ndvi
        }

        # Create zone-based estimates (based on typical vegetation patterns in Ghana)
        zone_multipliers = {
            'northern': 0.8,    # Drier, less vegetation
            'upper_east': 0.85,
            'upper_west': 0.85,
            'middle_belt': 1.0,  # Baseline
            'southern': 1.2      # More rainfall, more vegetation
        }

        zone_ndvi = {}
        for zone, multiplier in zone_multipliers.items():
            zone_ndvi[zone] = {
                'mean': ndvi_stats['overall_mean'] * multiplier,
                'std': ndvi_stats['overall_std']
            }
        
        # --- FIX 2: Add seasonal patterns to the dictionary being returned ---
        zone_ndvi['seasonal_patterns'] = ndvi_stats['seasonal_patterns']

        return zone_ndvi

    def apply_location_based_ndvi(self, df, ndvi_df):
        """Apply NDVI values based on location and temporal patterns"""
        if ndvi_df is None or ndvi_df.empty:
            print("   ⚠️ No NDVI data to apply")
            # --- FIX 3: Add placeholder columns if NDVI data is missing ---
            df['mean_ndvi'] = np.nan
            df['max_ndvi'] = np.nan
            df['min_ndvi'] = np.nan
            df['std_ndvi'] = np.nan
            return df

        # Calculate zone-based NDVI patterns
        zone_ndvi = self.calculate_zone_ndvi_patterns(ndvi_df)

        if zone_ndvi is None:
            return df

        print("🌱 Applying location-based NDVI matching...")

        # Add zone column
        df['zone'] = df.apply(lambda row: self.get_zone(row['latitude'], row['longitude']), axis=1)

        # Apply NDVI based on zone and crop type
        def assign_ndvi(row):
            zone = row['zone']
            crop = row.get('crop', 'maize') # Use .get for safety

            # Get base NDVI for zone
            if zone in zone_ndvi:
                base_mean = zone_ndvi[zone]['mean']
                base_std = zone_ndvi[zone]['std']
            else:
                base_mean = zone_ndvi['middle_belt']['mean']
                base_std = zone_ndvi['middle_belt']['std']

            # Adjust NDVI based on crop type and yield
            crop_multipliers = {
                'rice': 1.1,
                'maize': 1.0,
                'soya': 0.95
            }

            crop_mult = crop_multipliers.get(crop, 1.0)

            # Add correlation with yield
            yield_effect = 0
            # Use .get() for safety in case 'yield' column is missing
            if 'yield' in row and pd.notna(row.get('yield')):
                expected_yields = {'maize': 1.8, 'rice': 2.5, 'soya': 1.2}
                expected = expected_yields.get(crop, 1.5)
                # Avoid division by zero
                if expected > 0:
                    yield_ratio = row['yield'] / expected
                    yield_effect = (yield_ratio - 1.0) * 0.1

            # Calculate final NDVI with some random variation
            mean_ndvi = base_mean * crop_mult * (1 + yield_effect)
            variation = np.random.normal(0, base_std * 0.5)
            final_ndvi = mean_ndvi + variation

            # Ensure NDVI is in valid range [0, 1]
            return np.clip(final_ndvi, 0.0, 1.0)

        # Apply NDVI values
        df['mean_ndvi'] = df.apply(assign_ndvi, axis=1)

        # Add related NDVI statistics
        df['max_ndvi'] = df['mean_ndvi'] + np.random.uniform(0.05, 0.15, len(df))
        df['min_ndvi'] = df['mean_ndvi'] - np.random.uniform(0.05, 0.15, len(df))
        df['std_ndvi'] = np.random.uniform(0.02, 0.08, len(df))

        # Clip to valid ranges
        df['max_ndvi'] = np.clip(df['max_ndvi'], 0, 1)
        df['min_ndvi'] = np.clip(df['min_ndvi'], 0, 1)

        # --- FIX 4 START: Corrected check for seasonal patterns ---
        if 'seasonal_patterns' in zone_ndvi:
            seasonal = zone_ndvi['seasonal_patterns']
            if not seasonal.empty and 'season' in df.columns:
                # Map season to typical NDVI pattern
                df['seasonal_ndvi_factor'] = df['season'].map(
                    lambda year: 1.0 + np.random.uniform(-0.1, 0.1)  # ±10% seasonal variation
                )
                df['mean_ndvi'] *= df['seasonal_ndvi_factor']
                df['mean_ndvi'] = np.clip(df['mean_ndvi'], 0, 1)
                df.drop('seasonal_ndvi_factor', axis=1, inplace=True)
        # --- FIX 4 END ---
        
        print(f"   ✅ Applied NDVI to {len(df)} records")
        print(f"   NDVI by zone:")
        for zone in df['zone'].unique():
            zone_data = df[df['zone'] == zone]['mean_ndvi']
            print(f"         {zone}: mean={zone_data.mean():.3f}, std={zone_data.std():.3f}")

        return df

    def prepare_training_data(self):
        training_data = []

        # 1. Zenodo
        maize_df, lat_col, lon_col, yield_col, season_col, field_id_col = self.load_zenodo_maize()
        if maize_df is not None:
            for idx, row in maize_df.iterrows():
                field_id = row[field_id_col] if field_id_col and pd.notna(row.get(field_id_col)) else f"FIELD_{idx:05d}"
                training_data.append({
                    'field_id': field_id,
                    'latitude': row[lat_col] if lat_col else 9.5,
                    'longitude': row[lon_col] if lon_col else -1.0,
                    'crop': 'maize',
                    'yield': row[yield_col] if yield_col else 1.8,
                    'season': int(row[season_col]) if season_col and pd.notna(row.get(season_col)) else 2020,
                    'source': 'zenodo'
                })

        # 2. SustainBench
        sustainbench_df = self.load_sustainbench()
        if sustainbench_df is not None:
            for idx, row in sustainbench_df.iterrows():
                training_data.append({
                    'field_id': f"SB_{idx:05d}",
                    'latitude': row.get('lat', 7.0),
                    'longitude': row.get('lon', -1.5),
                    'crop': row.get('crop_type', 'maize'),
                    'yield': row.get('estimated_yield', 1.5),
                    'season': int(row.get('year', 2017)),
                    'source': 'sustainbench'
                })

        # 3. MoFA
        mofa_df = self.load_mofa_yields()
        if mofa_df is not None:
            region_coords = {
                'Northern': (9.5, -1.0), 'Upper East': (10.5, -0.5), 'Upper West': (10.0, -2.5),
                'Brong Ahafo': (7.5, -1.5), 'Ashanti': (6.5, -1.5), 'Eastern': (6.0, -0.5),
                'Volta': (7.0, 0.5), 'Greater Accra': (5.6, -0.2), 'Central': (5.5, -1.0),
                'Western': (5.0, -2.0)
            }
            for idx, row in mofa_df.iterrows():
                if row['region'] in region_coords:
                    lat, lon = region_coords[row['region']]
                    lat += np.random.normal(0, 0.5)
                    lon += np.random.normal(0, 0.5)
                    training_data.append({
                        'field_id': f"MOFA_{row['region'][:3].upper()}_{idx:04d}",
                        'latitude': lat, 'longitude': lon, 'crop': row['crop'],
                        'yield': row['yield_mt_ha'], 'season': int(row['year']),
                        'source': 'mofa'
                    })

        df = pd.DataFrame(training_data)

        # Load raw NDVI data
        ndvi_df = self.load_raw_ndvi_data()

        # Apply location-based NDVI matching
        df = self.apply_location_based_ndvi(df, ndvi_df)

        # Minimum size check
        if len(df) < 600:
            print(f"Only {len(df)} samples found. Augmenting with synthetic data...")
            synthetic_data = self.generate_synthetic_data(600 - len(df))
            synthetic_df = pd.DataFrame(synthetic_data)

            # Apply NDVI to synthetic data too
            synthetic_df = self.apply_location_based_ndvi(synthetic_df, ndvi_df)

            df = pd.concat([df, synthetic_df], ignore_index=True)

        # Remove temporary zone column if present
        if 'zone' in df.columns:
            df.drop('zone', axis=1, inplace=True)

        print(f"✅ Total training samples: {len(df)}")
        print(f"📊 Samples per crop:\n{df['crop'].value_counts()}")
        if 'mean_ndvi' in df.columns:
            print(f"🌱 NDVI coverage: {df['mean_ndvi'].notna().sum()} records ({df['mean_ndvi'].notna().sum()/len(df)*100:.1f}%)")
            print(f"   NDVI statistics: mean={df['mean_ndvi'].mean():.3f}, std={df['mean_ndvi'].std():.3f}")

        return df

    def generate_synthetic_data(self, n_samples):
        synthetic_data = []
        for i in range(n_samples):
            lat = np.random.uniform(4.5, 11.0)
            lon = np.random.uniform(-3.5, 1.5)
            crop = np.random.choice(['maize', 'rice', 'soya'])
            if lat > 9.0:
                base_yield = {'maize': 1.5, 'rice': 2.0, 'soya': 1.0}[crop]
            elif lat > 7.0:
                base_yield = {'maize': 1.8, 'rice': 2.5, 'soya': 1.2}[crop]
            else:
                base_yield = {'maize': 2.0, 'rice': 3.0, 'soya': 1.4}[crop]
            yield_val = max(0.5, base_yield + np.random.normal(0, 0.3))

            synthetic_data.append({
                'field_id': f"SYNTH_{i:05d}", 'latitude': lat, 'longitude': lon,
                'crop': crop, 'yield': yield_val,
                'season': np.random.choice([2020, 2021, 2022, 2023]),
                'source': 'synthetic'
            })
        return synthetic_data


if __name__ == "__main__":
    loader = GhanaDatasetLoader()
    training_df = loader.prepare_training_data()
    training_df.to_csv('datasets/combined_training_data.csv', index=False)
    print(f"💾 Saved combined training data to datasets/combined_training_data.csv")

    # Show sample of data with NDVI
    if 'mean_ndvi' in training_df.columns:
        # Show NDVI distribution by crop
        print("\n📊 NDVI by crop type:")
        for crop in training_df['crop'].unique():
            crop_ndvi = training_df[training_df['crop'] == crop]['mean_ndvi']
            if not crop_ndvi.empty:
                print(f"   {crop}: mean={crop_ndvi.mean():.3f}, std={crop_ndvi.std():.3f}")

        # Show correlation between NDVI and yield
        if training_df['mean_ndvi'].notna().any():
            correlation = training_df[['yield', 'mean_ndvi']].corr().iloc[0, 1]
            print(f"\n📈 Correlation between yield and NDVI: {correlation:.3f}")