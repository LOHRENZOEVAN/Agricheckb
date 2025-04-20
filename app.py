from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import requests
import datetime
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get meteoblue API key from environment variables
METEOBLUE_API_KEY = os.getenv("METEOBLUE_API_KEY")
if not METEOBLUE_API_KEY:
    logger.warning("Missing METEOBLUE_API_KEY environment variable. Set this in your .env file.")

# Helper Functions
def safe_float(value, default=None):
    """Safely convert a value to float"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def load_and_process_crop_prices(file_path):
    """
    Load and process crop price data from a CSV file with the new format:
    Symbol,Opening Price,Closing Price,Price Change (%),High,Low,Volumes Traded (MT),Date Range,Commodity
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Crop price data file not found: {file_path}")
            return pd.DataFrame(), pd.DataFrame()

        try:
            # Use on_bad_lines for pandas >= 1.3
            df = pd.read_csv(file_path, on_bad_lines='warn')
            logger.info(f"CSV columns: {df.columns.tolist()}")
            logger.info(f"Initial DataFrame shape: {df.shape}")
            
            # Check if DataFrame is empty after loading
            if df.empty:
                logger.error("CSV file loaded but contains no data")
                return pd.DataFrame(), pd.DataFrame()
                
            # Show sample of Date Range column to debug parsing issues
            logger.info(f"Sample of Date Range values: {df['Date Range'].head().tolist()}")
        except Exception as e:
            logger.error(f"Failed reading CSV: {e}")
            return pd.DataFrame(), pd.DataFrame()

        required_columns = ['Date Range', 'Commodity', 'Closing Price']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns in crop prices CSV. Required: {required_columns}")
            return pd.DataFrame(), pd.DataFrame()

        # Extract and parse dates
        df['Date'] = df['Date Range'].apply(
            lambda x: x.split(' to ')[0] if isinstance(x, str) and ' to ' in x else x
        )
        
        # Try multiple date formats
        date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
        
        def safe_date_parse(date_str):
            if pd.isna(date_str):
                return pd.NaT
            
            if not isinstance(date_str, str):
                return pd.NaT
                
            for fmt in date_formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except:
                    continue
            
            # If none of the formats work, try pandas' flexible parser
            try:
                return pd.to_datetime(date_str, errors='coerce')
            except:
                return pd.NaT
        
        df['Date'] = df['Date'].apply(safe_date_parse)
        
        # If we still have no valid dates, try again with pandas' flexible parsing
        if df['Date'].notna().sum() == 0:
            logger.warning("No valid dates found with strict parsing, trying flexible parsing")
            df['Date'] = pd.to_datetime(df['Date Range'], errors='coerce')

        # Drop rows with invalid/missing dates only if we have some valid dates
        if df['Date'].notna().sum() > 0:
            before_count = len(df)
            df = df.dropna(subset=['Date'])
            after_count = len(df)
            logger.info(f"Dropped {before_count - after_count} rows with invalid dates")

        # Ensure Closing Price is numeric
        df['Closing Price'] = pd.to_numeric(df['Closing Price'], errors='coerce')
        
        # Drop rows where Closing Price is missing only if we have some valid prices
        if df['Closing Price'].notna().sum() > 0:
            before_count = len(df)
            df = df.dropna(subset=['Closing Price'])
            after_count = len(df)
            logger.info(f"Dropped {before_count - after_count} rows with invalid Closing Prices")

        # Add Product column
        df['Product'] = df['Commodity']
        
        # If DataFrame is empty after processing, return empty
        if df.empty:
            logger.error("No valid data remains after preprocessing")
            return pd.DataFrame(), df
            
        # If we only have a single row, we can't pivot - return a manually constructed DataFrame
        if len(df) == 1:
            logger.warning("Only one row of data, creating manual pivot")
            row = df.iloc[0]
            result_df = pd.DataFrame(
                {row['Product']: [row['Closing Price']]},
                index=[row['Date']]
            )
            return result_df, df
            
        try:
            # Use pivot_table to handle potential duplicates
            pivoted_df = df.pivot_table(
                index='Date', 
                columns='Product', 
                values='Closing Price',
                aggfunc='mean'
            )
            
            if pivoted_df.empty or len(pivoted_df.columns) == 0:
                logger.warning("Pivot operation resulted in empty DataFrame")
                # Try a more direct approach if pivot_table fails
                unique_products = df['Product'].unique()
                if len(unique_products) > 0:
                    result_df = pd.DataFrame(index=df['Date'].unique())
                    
                    for product in unique_products:
                        product_data = df[df['Product'] == product].set_index('Date')['Closing Price']
                        if not product_data.empty:
                            result_df[product] = product_data
                    
                    if not result_df.empty and len(result_df.columns) > 0:
                        logger.info(f"Manual product columns created: {result_df.columns.tolist()}")
                        return result_df, df
            else:
                logger.info(f"Pivot successful. Columns: {pivoted_df.columns.tolist()}")
                return pivoted_df, df
                
        except Exception as e:
            logger.error(f"Error pivoting crop price data: {str(e)}")
            
            # Return at least something if we have data
            if not df.empty:
                try:
                    # Create simple DataFrame with first product
                    first_product = df['Product'].iloc[0]
                    simple_df = pd.DataFrame(
                        {first_product: df['Closing Price'].values},
                        index=df['Date']
                    )
                    logger.info("Created simple DataFrame with first product as fallback")
                    return simple_df, df
                except Exception as e3:
                    logger.error(f"Simple DataFrame creation failed: {str(e3)}")
            
            return pd.DataFrame(), df

    except Exception as e:
        logger.error(f"Error loading crop prices: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Weather Data Functions
def get_14day_forecast(latitude, longitude, elevation=None):
    """Get 14-day weather forecast from meteoblue API"""
    # Base URL for meteoblue API
    BASE_URL = "https://my.meteoblue.com/packages"
    
    result = {}
    
    # Validate inputs
    try:
        latitude = float(latitude)
        longitude = float(longitude)
        if elevation is not None:
            elevation = int(elevation)
    except (ValueError, TypeError):
        logger.error(f"Invalid coordinates: lat={latitude}, lon={longitude}, elevation={elevation}")
        return result
    
    if not METEOBLUE_API_KEY:
        logger.error("Missing API key, cannot fetch weather forecast")
        return result
    
    # Build URL for the meteoblue basic-day API (7 days)
    url_params = {
        "lat": latitude,
        "lon": longitude,
        "apikey": METEOBLUE_API_KEY,
        "tz": "UTC",
        "forecast_days": 7
    }
    
    if elevation is not None:
        url_params["asl"] = elevation
        
    url = f"{BASE_URL}/basic-day?{'&'.join([f'{k}={v}' for k, v in url_params.items()])}"
    
    try:
        # Make request to meteoblue API for basic-day (first 7 days)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        # Build URL for the meteoblue trend-day API (days 8-14)
        url_params = {
            "lat": latitude,
            "lon": longitude,
            "apikey": METEOBLUE_API_KEY,
            "tz": "UTC"
        }
        
        if elevation is not None:
            url_params["asl"] = elevation
            
        url = f"{BASE_URL}/trend-day?{'&'.join([f'{k}={v}' for k, v in url_params.items()])}"
        
        # Make request to meteoblue API for trend-day (days 8-14)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        result["trend"] = response.json()
        
        return result
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather forecast from meteoblue API: {str(e)}")
        return result
    except Exception as e:
        logger.error(f"Unexpected error in get_14day_forecast: {str(e)}")
        return result

def get_seasonal_forecast(latitude, longitude, elevation=None):
    """Get 6-month seasonal forecast from meteoblue API"""
    # Base URL for meteoblue API
    BASE_URL = "https://my.meteoblue.com/packages"
    
    result = {}
    
    # Validate inputs
    try:
        latitude = float(latitude)
        longitude = float(longitude)
        if elevation is not None:
            elevation = int(elevation)
    except (ValueError, TypeError):
        logger.error(f"Invalid coordinates: lat={latitude}, lon={longitude}, elevation={elevation}")
        return result
    
    if not METEOBLUE_API_KEY:
        logger.error("Missing API key, cannot fetch seasonal forecast")
        return result
    
    # Build URL for the seasonalanomaly-monthly API with the exact parameters that worked
    url_params = {
        "lat": latitude,
        "lon": longitude,
        "apikey": METEOBLUE_API_KEY,
        "format": "json",  # Add format parameter
        "tz": "GMT"        # Add timezone parameter
    }
    
    if elevation is not None:
        url_params["asl"] = elevation
        
    url = f"{BASE_URL}/seasonalanomaly-monthly?{'&'.join([f'{k}={v}' for k, v in url_params.items()])}"
    
    try:
        # Log the URL (with API key redacted for security)
        log_url = url.replace(METEOBLUE_API_KEY, "API_KEY_REDACTED")
        logger.info(f"Calling Meteoblue seasonal API: {log_url}")
        
        # Make request to meteoblue API for seasonal forecast
        response = requests.get(url, timeout=15)
        
        # Log the status code
        logger.info(f"Meteoblue seasonal API responded with status code: {response.status_code}")
        
        # Check if response is successful
        if response.status_code != 200:
            logger.error(f"Seasonal API error: {response.text}")
            return result
            
        response.raise_for_status()
        result = response.json()
        
        # Log the keys in the response
        if result:
            logger.info(f"Seasonal API response contains keys: {list(result.keys())}")
            if "data_seasonalmonthly" in result:
                logger.info(f"Found {len(result['data_seasonalmonthly'].get('time', []))} months of forecast data")
            else:
                logger.error("Seasonal response does not contain 'data_seasonalmonthly' key")
        else:
            logger.error("Seasonal API returned empty response")
            
        return result
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching seasonal forecast from meteoblue API: {str(e)}")
        return result
    except Exception as e:
        logger.error(f"Unexpected error in get_seasonal_forecast: {str(e)}")
        return result

def process_forecast_data(forecast_data):
    """Process forecast data into a DataFrame"""
    if not forecast_data:
        logger.warning("Empty forecast data provided")
        return pd.DataFrame()
    
    if "data_day" not in forecast_data:
        logger.warning("Missing 'data_day' in forecast data")
        return pd.DataFrame()
    
    daily_data = []
    
    # Process first 7 days (from basic-day)
    try:
        if "time" not in forecast_data["data_day"]:
            logger.warning("Missing 'time' in forecast data_day")
            return pd.DataFrame()
        
        for i in range(min(7, len(forecast_data["data_day"]["time"]))):
            try:
                # Get data safely with defaults if missing
                day_data = {
                    "date": forecast_data["data_day"]["time"][i],
                    "temperature_min": forecast_data["data_day"].get("temperature_min", [0]*7)[i] 
                        if len(forecast_data["data_day"].get("temperature_min", [])) > i else 0,
                    "temperature_max": forecast_data["data_day"].get("temperature_max", [0]*7)[i]
                        if len(forecast_data["data_day"].get("temperature_max", [])) > i else 0,
                    "temperature_mean": forecast_data["data_day"].get("temperature_mean", [0]*7)[i]
                        if len(forecast_data["data_day"].get("temperature_mean", [])) > i else 0,
                    "precipitation_sum": forecast_data["data_day"].get("precipitation_sum", [0]*7)[i]
                        if len(forecast_data["data_day"].get("precipitation_sum", [])) > i else 0,
                    "windspeed_mean": forecast_data["data_day"].get("windspeed_mean", [0]*7)[i] 
                        if "windspeed_mean" in forecast_data["data_day"] and len(forecast_data["data_day"]["windspeed_mean"]) > i else 0,
                    "relativehumidity_mean": forecast_data["data_day"].get("relativehumidity_mean", [0]*7)[i]
                        if "relativehumidity_mean" in forecast_data["data_day"] and len(forecast_data["data_day"]["relativehumidity_mean"]) > i else 0,
                }
                daily_data.append(day_data)
            except Exception as e:
                logger.error(f"Error processing day {i} of basic-day data: {str(e)}")
                continue
        
        # Process days 8-14 (from trend-day)
        if "trend" in forecast_data and "data_day" in forecast_data["trend"]:
            trend_data = forecast_data["trend"]["data_day"]
            
            if "time" not in trend_data:
                logger.warning("Missing 'time' in trend data_day")
            else:
                for i in range(min(7, len(trend_data["time"]))):
                    try:
                        day_data = {
                            "date": trend_data["time"][i],
                            "temperature_min": trend_data.get("temperature_min", [0]*7)[i]
                                if len(trend_data.get("temperature_min", [])) > i else 0,
                            "temperature_max": trend_data.get("temperature_max", [0]*7)[i]
                                if len(trend_data.get("temperature_max", [])) > i else 0,
                            "temperature_mean": trend_data.get("temperature_mean", [0]*7)[i]
                                if len(trend_data.get("temperature_mean", [])) > i else 0,
                            "precipitation_sum": trend_data.get("precipitation_sum", [0]*7)[i]
                                if len(trend_data.get("precipitation_sum", [])) > i else 0,
                            "windspeed_mean": 0,  # Not available in trend data
                            "relativehumidity_mean": 0  # Not available in trend data
                        }
                        daily_data.append(day_data)
                    except Exception as e:
                        logger.error(f"Error processing day {i} of trend-day data: {str(e)}")
                        continue
    
    except Exception as e:
        logger.error(f"Error processing forecast data: {str(e)}")
    
    return pd.DataFrame(daily_data)

def process_seasonal_forecast(forecast_data):
    """Process seasonal forecast data into a structured format"""
    if not forecast_data or "data_seasonalmonthly" not in forecast_data:
        logger.warning("Empty seasonal forecast data provided")
        return []
    
    monthly_data = []
    
    try:
        # Extract the necessary data from the seasonal forecast
        monthly_info = forecast_data.get("data_seasonalmonthly", {})
        
        # Get available months
        months = monthly_info.get("time", [])
        
        # Process each month's data
        for i, month in enumerate(months):
            try:
                # Create a monthly data object with available metrics
                month_data = {
                    "month": month,
                    "temperature": {
                        "mean_anomaly": monthly_info.get("temperature_meananomaly", [])[i] if "temperature_meananomaly" in monthly_info and i < len(monthly_info["temperature_meananomaly"]) else None,
                        "prob_above_avg": monthly_info.get("temperature_probability_above_average", [])[i] if "temperature_probability_above_average" in monthly_info and i < len(monthly_info["temperature_probability_above_average"]) else None,
                        "prob_below_avg": monthly_info.get("temperature_probability_below_average", [])[i] if "temperature_probability_below_average" in monthly_info and i < len(monthly_info["temperature_probability_below_average"]) else None,
                        "prob_normal": monthly_info.get("temperature_probability_near_normal", [])[i] if "temperature_probability_near_normal" in monthly_info and i < len(monthly_info["temperature_probability_near_normal"]) else None
                    },
                    "precipitation": {
                        "anomaly_pct": monthly_info.get("precipitationanomaly_percentage_from_normal", [])[i] if "precipitationanomaly_percentage_from_normal" in monthly_info and i < len(monthly_info["precipitationanomaly_percentage_from_normal"]) else None,
                        "prob_above_avg": monthly_info.get("precipitation_probability_above_average", [])[i] if "precipitation_probability_above_average" in monthly_info and i < len(monthly_info["precipitation_probability_above_average"]) else None,
                        "prob_below_avg": monthly_info.get("precipitation_probability_below_average", [])[i] if "precipitation_probability_below_average" in monthly_info and i < len(monthly_info["precipitation_probability_below_average"]) else None,
                        "prob_normal": monthly_info.get("precipitation_probability_near_normal", [])[i] if "precipitation_probability_near_normal" in monthly_info and i < len(monthly_info["precipitation_probability_near_normal"]) else None
                    }
                }
                monthly_data.append(month_data)
            except Exception as e:
                logger.error(f"Error processing month {i} of seasonal data: {str(e)}")
                continue
        
        return monthly_data
    
    except Exception as e:
        logger.error(f"Error processing seasonal forecast data: {str(e)}")
        return []

# Risk Assessment Class
class CropRiskAssessment:
    """Class to assess crop production risk based on weather, price, and suitability data"""
    def __init__(self, price_data, weather_data, crop_suitability, seasonal_data=None):
        """Initialize with required data"""
        self.price_data = price_data
        self.weather_data = weather_data
        self.crop_suitability = crop_suitability
        self.seasonal_data = seasonal_data or []
    
    def calculate_weather_risk(self):
        """Calculate weather risk based on 14-day forecast"""
        # Default risk if weather data is empty
        if self.weather_data is None or self.weather_data.empty:
            return 0.5
        
        try:
            # Check temperature variability
            temp_variation = 0
            if 'temperature_max' in self.weather_data.columns and 'temperature_min' in self.weather_data.columns:
                try:
                    daily_range = self.weather_data['temperature_max'] - self.weather_data['temperature_min']
                    temp_mean = daily_range.mean()
                    temp_variation = daily_range.std() / temp_mean if temp_mean > 0 else 0
                    # Normalize to 0-1 range (assuming std/mean > 1 is high variability)
                    temp_variation = min(temp_variation, 1.0)
                except Exception as e:
                    logger.error(f"Error calculating temperature variation: {str(e)}")
                    temp_variation = 0.5
            
            # Check precipitation pattern
            precip_risk = 0.5  # Default medium risk
            if 'precipitation_sum' in self.weather_data.columns:
                try:
                    # Calculate risk based on too much rain or drought
                    total_days = len(self.weather_data)
                    if total_days > 0:
                        # Count dry days (< 1mm precipitation)
                        dry_days = (self.weather_data['precipitation_sum'] < 1).sum()
                        
                        # Count heavy rain days (> 20mm precipitation)
                        heavy_rain_days = (self.weather_data['precipitation_sum'] > 20).sum()
                        
                        # Calculate drought risk (many consecutive dry days)
                        dry_streak = 0
                        max_dry_streak = 0
                        for _, row in self.weather_data.iterrows():
                            if row['precipitation_sum'] < 1:
                                dry_streak += 1
                                max_dry_streak = max(max_dry_streak, dry_streak)
                            else:
                                dry_streak = 0
                        
                        # Normalize to 0-1 range
                        drought_risk = min(max_dry_streak / 7, 1.0)  # 7+ consecutive dry days is high risk
                        heavy_rain_risk = min(heavy_rain_days / total_days * 3, 1.0)  # If 1/3 of days have heavy rain, high risk
                        
                        # Combine precipitation risks
                        precip_risk = max(drought_risk, heavy_rain_risk)
                except Exception as e:
                    logger.error(f"Error calculating precipitation risk: {str(e)}")
                    precip_risk = 0.5
            
            # Check temperature extremes
            temp_extreme_risk = 0.5  # Default medium risk
            if 'temperature_max' in self.weather_data.columns and 'temperature_min' in self.weather_data.columns:
                try:
                    # Risk for extreme heat (> 35°C) or cold (< 0°C)
                    extreme_heat_days = (self.weather_data['temperature_max'] > 35).sum()
                    extreme_cold_days = (self.weather_data['temperature_min'] < 0).sum()
                    
                    total_days = len(self.weather_data)
                    if total_days > 0:
                        # Normalize to 0-1 range
                        heat_risk = min(extreme_heat_days / total_days * 3, 1.0)  # If 1/3 of days are extremely hot, high risk
                        cold_risk = min(extreme_cold_days / total_days * 3, 1.0)  # If 1/3 of days are extremely cold, high risk
                        
                        # Take the maximum of heat or cold risk
                        temp_extreme_risk = max(heat_risk, cold_risk)
                except Exception as e:
                    logger.error(f"Error calculating temperature extreme risk: {str(e)}")
                    temp_extreme_risk = 0.5
            
            # Combine all weather risks
            weather_risk = (temp_variation * 0.3) + (precip_risk * 0.4) + (temp_extreme_risk * 0.3)
            
            # Ensure the risk is in the 0-1 range
            return min(max(weather_risk, 0.0), 1.0)
        
        except Exception as e:
            logger.error(f"Unexpected error in calculate_weather_risk: {str(e)}")
            return 0.5  # Default medium risk on error
    
    def calculate_seasonal_risk(self):
        """Calculate long-term seasonal risk based on 6-month forecast"""
        # Default risk if seasonal data is empty
        if not self.seasonal_data:
            return 0.5
        
        try:
            # Initialize risks
            drought_risk = 0.0
            flood_risk = 0.0
            temperature_risk = 0.0
            
            # Track how many months we have valid data for
            valid_months = 0
            
            # Process each month's data
            for month in self.seasonal_data:
                if "precipitation" not in month or "temperature" not in month:
                    continue
                
                # Get precipitation data
                precip = month.get("precipitation", {})
                precip_anomaly = safe_float(precip.get("anomaly_pct"), 0)
                precip_below = safe_float(precip.get("prob_below_avg"), 0)
                precip_above = safe_float(precip.get("prob_above_avg"), 0)
                
                # Get temperature data
                temp = month.get("temperature", {})
                temp_anomaly = safe_float(temp.get("mean_anomaly"), 0)
                temp_above = safe_float(temp.get("prob_above_avg"), 0)
                temp_below = safe_float(temp.get("prob_below_avg"), 0)
                
                # Calculate risks for this month
                # Drought risk - high if precipitation below average and temperature above average
                month_drought_risk = (precip_below / 100.0) * (temp_above / 100.0)
                
                # Flood risk - high if precipitation above average
                month_flood_risk = precip_above / 100.0
                
                # Temperature risk - extreme temperatures in either direction
                month_temp_risk = max(temp_above, temp_below) / 100.0
                
                # Weight risks by probability confidence
                drought_risk += month_drought_risk
                flood_risk += month_flood_risk
                temperature_risk += month_temp_risk
                
                valid_months += 1
            
            # Calculate average risks if we have data
            if valid_months > 0:
                drought_risk /= valid_months
                flood_risk /= valid_months
                temperature_risk /= valid_months
            
            # Combine risks with more weight on drought for crops
            seasonal_risk = (drought_risk * 0.5) + (flood_risk * 0.3) + (temperature_risk * 0.2)
            
            # Ensure the risk is in the 0-1 range
            return min(max(seasonal_risk, 0.0), 1.0)
        
        except Exception as e:
            logger.error(f"Unexpected error in calculate_seasonal_risk: {str(e)}")
            return 0.5  # Default medium risk on error
    
    def calculate_price_volatility(self, crop):
        """Calculate price volatility for a specific crop"""
        # Default medium volatility
        volatility = 0.5
        
        try:
            # Check if price_data is valid
            if self.price_data is None or self.price_data.empty:
                logger.warning(f"Empty price data when calculating volatility for {crop}")
                return volatility
                
            # Check if crop exists in price data
            if crop not in self.price_data.columns:
                logger.warning(f"Crop '{crop}' not found in price data")
                return volatility
                
            # Get price data and handle NaN values
            prices = self.price_data[crop].dropna()
            
            # If not enough data points, return medium volatility
            if len(prices) < 2:
                logger.warning(f"Insufficient price data for crop '{crop}'")
                return volatility
                
            # Calculate coefficient of variation (CV) safely
            mean_price = prices.mean()
            if mean_price <= 0:
                cv = 0.5  # Default if mean price is zero or negative
            else:
                cv = prices.std() / mean_price
            
            # Calculate recent trend safely
            trend_volatility = 0.5  # Default medium volatility
            
            if len(prices) >= 12:
                recent_prices = prices.iloc[-12:]
            else:
                recent_prices = prices
            
            if len(recent_prices) >= 2:
                first_price = recent_prices.iloc[0]
                last_price = recent_prices.iloc[-1]
                
                # Avoid division by zero
                if first_price > 0:
                    price_change = (last_price - first_price) / first_price
                    trend_volatility = min(abs(price_change), 1.0)
            
            # Normalize CV to 0-1 range (assuming CV > 0.5 is high volatility)
            cv_normalized = min(cv / 0.5, 1.0) if cv > 0 else 0
            
            # Combine measures (give more weight to recent trend)
            volatility = (cv_normalized * 0.4) + (trend_volatility * 0.6)
            
            return min(max(volatility, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating price volatility for {crop}: {str(e)}")
            return volatility  # Return medium volatility on any error
    
    def generate_risk_report(self):
        """Generate comprehensive crop production risk report"""
        # Risk assessment for each crop
        risk_assessment = {}
        
        # Calculate overall weather risk (short-term)
        try:
            weather_risk = self.calculate_weather_risk()
        except Exception as e:
            logger.error(f"Failed to calculate weather risk: {str(e)}")
            weather_risk = 0.5  # Default medium risk
        
        # Calculate seasonal risk (long-term)
        try:
            seasonal_risk = self.calculate_seasonal_risk()
        except Exception as e:
            logger.error(f"Failed to calculate seasonal risk: {str(e)}")
            seasonal_risk = 0.5  # Default medium risk
        
        # Process each crop
        if self.price_data is not None and not self.price_data.empty:
            for crop in self.price_data.columns:
                try:
                    # Normalize crop name for matching with suitability data
                    crop_lower = crop.lower()
                    
                    # Skip if crop is not in our suitability data
                    suitability_score = 0.5  # Default medium suitability
                    
                    # Get suitability if available
                    if crop_lower in self.crop_suitability:
                        suitability_score = self.crop_suitability[crop_lower] / 100.0
                    
                    # Calculate price volatility for this crop
                    price_volatility = self.calculate_price_volatility(crop)
                    
                    # Calculate risk components
                    market_risk = price_volatility
                    short_term_risk = weather_risk
                    long_term_risk = seasonal_risk
                    suitability_risk = 1.0 - suitability_score
                    
                    # Combined environmental risk (short and long term)
                    environmental_risk = (short_term_risk * 0.4) + (long_term_risk * 0.6)
                    
                    # Composite risk score calculation
                    risk_score = (
                        market_risk * 0.3 +          # Market risk (price volatility)
                        environmental_risk * 0.4 +   # Environmental risk (weather + seasonal)
                        suitability_risk * 0.3       # Land suitability risk
                    )
                    
                    # Ensure risk score is in 0-1 range
                    risk_score = min(max(risk_score, 0.0), 1.0)
                    
                    # Risk level classification
                    if risk_score < 0.2:
                        risk_level = "Very Low Risk"
                    elif risk_score < 0.4:
                        risk_level = "Low Risk"
                    elif risk_score < 0.6:
                        risk_level = "Moderate Risk"
                    elif risk_score < 0.8:
                        risk_level = "High Risk"
                    else:
                        risk_level = "Very High Risk"
                    
                    # Determine key risk factors
                    risk_factors = []
                    if price_volatility >= 0.6:
                        risk_factors.append("High price volatility")
                    if short_term_risk >= 0.6:
                        risk_factors.append("Unfavorable short-term weather")
                    if long_term_risk >= 0.6:
                        risk_factors.append("Unfavorable seasonal forecast")
                    if suitability_risk >= 0.6:
                        risk_factors.append("Low land suitability")
                    
                    if not risk_factors:
                        if risk_score >= 0.4:
                            risk_factors.append("Combined moderate factors")
                        else:
                            risk_factors.append("No significant risk factors")
                    
                    # Detailed risk assessment
                    risk_assessment[crop] = {
                        'risk_level': risk_level,
                        'risk_score': round(risk_score, 2),
                        'risk_factors': risk_factors,
                        'components': {
                            'price_volatility': round(price_volatility, 2),
                            'short_term_weather_risk': round(short_term_risk, 2),
                            'long_term_seasonal_risk': round(long_term_risk, 2),
                            'suitability_score': round(suitability_score, 2)
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Error assessing risk for crop {crop}: {str(e)}")
                    risk_assessment[crop] = {
                        'risk_level': "Assessment Failed",
                        'risk_score': 0.5,
                        'risk_factors': ["Assessment error"],
                        'error': str(e)
                    }
        else:
            logger.warning("Empty price data, cannot generate comprehensive risk report")
        
        return risk_assessment

# Flask Application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Check for required files at startup
required_files = ['crop_prices.csv']
for file in required_files:
    if not os.path.exists(file):
        logger.warning(f"Required file {file} not found. API may not work correctly.")

# Load suitability data for different crops at application startup
suitability_dfs = {}
try:
    crop_types = ['maize', 'rice', 'soya']
    for crop in crop_types:
        csv_filename = f"suitability_data_{crop}.csv"
        if os.path.exists(csv_filename):
            try:
                suitability_dfs[crop] = pd.read_csv(csv_filename)
                logger.info(f"Loaded suitability data for {crop} from {csv_filename}")
            except Exception as e:
                logger.error(f"Failed to load suitability data for {crop}: {str(e)}")
        else:
            logger.warning(f"Suitability data file {csv_filename} not found")
except Exception as e:
    logger.error(f"Error loading suitability data: {str(e)}")

# Load price data at startup
try:
    price_data_file = 'crop_prices.csv'
    if os.path.exists(price_data_file):
        global_price_data, _ = load_and_process_crop_prices(price_data_file)
        logger.info(f"Loaded price data from {price_data_file}")
        logger.info(f"Available crops in price data: {list(global_price_data.columns)}")
    else:
        global_price_data = pd.DataFrame()
        logger.warning(f"Price data file {price_data_file} not found")
except Exception as e:
    global_price_data = pd.DataFrame()
    logger.error(f"Failed to load price data: {str(e)}")

def get_suitability_for_crop(crop, lat, lon):
    """Helper function to get suitability value for a specific crop at given coordinates"""
    try:
        if crop not in suitability_dfs or suitability_dfs[crop].empty:
            logger.warning(f"Crop '{crop}' not in suitability data, using default value")
            return 50  # Default medium suitability
        
        df = suitability_dfs[crop]
        df['distance'] = ((df['latitude'] - lat) ** 2 + (df['longitude'] - lon) ** 2)
        if df.empty:
            return 50  # Default medium suitability if df is empty
            
        nearest = df.loc[df['distance'].idxmin()]
        return int(nearest['suitability'])
    except Exception as e:
        logger.error(f"Error getting suitability for crop {crop}: {str(e)}")
        return 50  # Default medium suitability on error

# API Routes
@app.route('/suitability', methods=['GET'])
def get_suitability():
    """Endpoint to get crop suitability based on coordinates and crop type"""
    try:
        # Get and validate coordinates
        lat_value = request.args.get('lat')
        lon_value = request.args.get('lon')
        
        if lat_value is None or lon_value is None:
            return jsonify({
                'status': 'error',
                'message': 'Missing required parameters: lat and lon'
            }), 400
            
        try:
            lat = float(lat_value)
            lon = float(lon_value)
        except (ValueError, TypeError):
            return jsonify({
                'status': 'error',
                'message': 'Invalid coordinates format'
            }), 400
            
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({
                'status': 'error',
                'message': 'Invalid coordinate values. Latitude must be between -90 and 90, longitude between -180 and 180.'
            }), 400
        
        # Get crop parameter (default to returning all crops if not specified)
        crop = request.args.get('crop', '').lower()
        
        # Result object
        result = {
            "status": "success",
            "location": {
                "latitude": lat,
                "longitude": lon
            }
        }
        
        # If specific crop requested, return just that crop's suitability
        if crop:
            if crop not in suitability_dfs:
                return jsonify({
                    'status': 'error',
                    'message': f'Crop {crop} not supported. Available crops: {list(suitability_dfs.keys())}'
                }), 400
                
            suitability = get_suitability_for_crop(crop, lat, lon)
            result["data"] = {
                "crop": crop,
                "suitability": suitability
            }
        # If no specific crop, return all crops' suitability
        else:
            suitability_data = {}
            for crop_name in suitability_dfs:
                suitability_data[crop_name] = get_suitability_for_crop(crop_name, lat, lon)
            
            result["data"] = suitability_data
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in get_suitability: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/weather-forecast', methods=['GET'])
def get_weather_forecast_endpoint():
    """Endpoint to get weather forecast for a location (both 14-day and 6-month)"""
    try:
        # Get and validate coordinates
        lat_value = request.args.get('lat')
        lon_value = request.args.get('lon')
        
        if lat_value is None or lon_value is None:
            return jsonify({
                'status': 'error',
                'message': 'Missing required parameters: lat and lon'
            }), 400
            
        try:
            lat = float(lat_value)
            lon = float(lon_value)
        except (ValueError, TypeError):
            return jsonify({
                'status': 'error',
                'message': 'Invalid coordinates format'
            }), 400
            
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({
                'status': 'error',
                'message': 'Invalid coordinate values. Latitude must be between -90 and 90, longitude between -180 and 180.'
            }), 400
        
        # Get and validate elevation parameter (optional)
        asl_value = request.args.get('asl')
        asl = None
        if asl_value is not None:
            try:
                asl = int(asl_value)
            except (ValueError, TypeError):
                return jsonify({
                    'status': 'error',
                    'message': f'Invalid elevation format: {asl_value}'
                }), 400
        
        # Get 14-day forecast data
        forecast_data = get_14day_forecast(lat, lon, asl)
        
        if not forecast_data or "data_day" not in forecast_data:
            return jsonify({
                'status': 'error',
                'message': 'Failed to retrieve weather forecast data'
            }), 503
        
        # Get 6-month seasonal forecast data
        seasonal_data = get_seasonal_forecast(lat, lon, asl)
        
        # Process forecast data
        daily_df = process_forecast_data(forecast_data)
        monthly_data = process_seasonal_forecast(seasonal_data)
        
        if daily_df.empty:
            logger.warning(f"Empty daily weather data for coordinates: lat={lat}, lon={lon}")
        
        if not monthly_data:
            logger.warning(f"Empty monthly weather data for coordinates: lat={lat}, lon={lon}")
        
        # Convert to dictionaries for JSON response
        daily_forecast = daily_df.to_dict(orient='records')
        

        # Return response
        return jsonify({
            'status': 'success',
            'location': {
                'latitude': lat,
                'longitude': lon,
                'elevation': asl
            },
            'weather_forecast': {
                'daily': daily_forecast,
                'monthly': monthly_data
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error in get_weather_forecast_endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/assess-crop-risk', methods=['POST'])
def assess_crop_risk():
    """Endpoint to assess crop risk based on coordinates"""
    try:
        # Parse request data safely
        try:
            data = request.get_json(silent=True)
        except Exception as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            data = None
            
        if not data or not isinstance(data, dict):
            return jsonify({
                'status': 'error',
                'message': 'Invalid JSON data format or empty request'
            }), 400
        
        # Extract and validate required parameters
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if latitude is None or longitude is None:
            return jsonify({
                'status': 'error',
                'message': 'Missing required parameters: latitude and longitude'
            }), 400
        
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (ValueError, TypeError):
            return jsonify({
                'status': 'error',
                'message': 'Invalid coordinates format. Latitude and longitude must be numbers.'
            }), 400
            
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return jsonify({
                'status': 'error',
                'message': 'Invalid coordinate values. Latitude must be between -90 and 90, longitude between -180 and 180.'
            }), 400
        
        # Extract and validate optional parameters
        elevation = data.get('elevation')
        crop_type = data.get('crop_type')
        field_location = data.get('field_location')
        crop = data.get('crop', '').lower()
        
        if elevation is not None:
            try:
                elevation = int(elevation)
            except (ValueError, TypeError):
                return jsonify({
                    'status': 'error',
                    'message': f'Invalid elevation format. Must be an integer.'
                }), 400
                
        # Check if price data is available
        if global_price_data.empty:
            return jsonify({
                'status': 'error',
                'message': 'Crop price data is not available. Please check if crop_prices.csv exists and is valid.'
            }), 500
        
        # Get weather forecast data
        forecast_data = get_14day_forecast(latitude, longitude, elevation)
        weather_data = process_forecast_data(forecast_data)
        
        # Get seasonal forecast data
        seasonal_data_response = get_seasonal_forecast(latitude, longitude, elevation)
        monthly_data = process_seasonal_forecast(seasonal_data_response)
        
        # Get suitability data for all relevant crops
        crop_suitability = {}
        
        # Determine which crops to get suitability for
        target_crops = []
        
        if crop_type and crop_type.lower() != 'other':
            # Process specific crop type
            crop_type_lower = crop_type.lower()
            
            # Map common names to our suitability data names
            if crop_type_lower == 'corn':
                crop_type_lower = 'maize'
            elif crop_type_lower in ['soybeans', 'soybean']:
                crop_type_lower = 'soya'
                
            # Check if we have this crop in suitability data
            if crop_type_lower in suitability_dfs:
                target_crops.append(crop_type_lower)
            else:
                # Try to match with price data
                matching_price_crops = []
                for price_crop in global_price_data.columns:
                    price_crop_lower = price_crop.lower()
                    if price_crop_lower.startswith(crop_type_lower):
                        matching_price_crops.append(price_crop_lower)
                
                if matching_price_crops:
                    target_crops.extend(matching_price_crops)
                else:
                    target_crops.append(crop_type_lower)  # Use as is with default suitability
        
        elif crop:
            # Handle individual crop specification
            crop_key = crop
            if crop == 'corn':
                crop_key = 'maize'
            elif crop in ['soybeans', 'soybean']:
                crop_key = 'soya'
                
            target_crops.append(crop_key)
        
        else:
            # Use all crops from price data
            for price_crop in global_price_data.columns:
                target_crops.append(price_crop.lower())
        
        # Get suitability for all target crops
        for target_crop in target_crops:
            crop_key = target_crop
            # Map to suitability data names if needed
            if target_crop == 'corn':
                crop_key = 'maize'
            elif target_crop in ['soybeans', 'soybean']:
                crop_key = 'soya'
                
            # Get suitability if we have data for this crop
            if crop_key in suitability_dfs:
                crop_suitability[target_crop] = get_suitability_for_crop(crop_key, latitude, longitude)
            else:
                crop_suitability[target_crop] = 50  # Default medium suitability
        
        # Filter price data to relevant crops
        if crop_type or crop:
            filtered_price_data = pd.DataFrame()
            
            for col in global_price_data.columns:
                col_lower = col.lower()
                # Match with any target crop
                if col_lower in target_crops or any(col_lower.startswith(tc) for tc in target_crops):
                    filtered_price_data[col] = global_price_data[col]
            
            price_data_for_risk = filtered_price_data
        else:
            price_data_for_risk = global_price_data
        
        # Create risk assessment
        risk_assessor = CropRiskAssessment(
            price_data=price_data_for_risk,
            weather_data=weather_data,
            crop_suitability=crop_suitability,
            seasonal_data=monthly_data
        )
        
        # Generate risk report
        risk_report = risk_assessor.generate_risk_report()
        
        # Prepare weather data for response
        daily_weather_data = weather_data.to_dict(orient='records')
        
        return jsonify({
            'status': 'success',
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'elevation': elevation,
                'field_location': field_location
            },
            'suitability_data': crop_suitability,
            'weather_forecast': {
                'daily': daily_weather_data,
                'monthly': monthly_data
            },
            'risk_assessment': risk_report
        }), 200
    
    except Exception as e:
        logger.error(f"Error in assess_crop_risk: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

# Add a test endpoint to directly test the seasonal forecast
@app.route('/test-seasonal', methods=['GET'])
def test_seasonal_forecast():
    """Test endpoint to verify seasonal forecast functionality"""
    try:
        lat = request.args.get('lat', '5.55602')  # Use coordinates that worked 
        lon = request.args.get('lon', '-0.1969')  # for direct API call
        
        # Use the exact URL that worked directly
        url = f"https://my.meteoblue.com/packages/seasonalanomaly-monthly?apikey={METEOBLUE_API_KEY}&lat={lat}&lon={lon}&format=json&tz=GMT"
        
        direct_response = {}
        try:
            # Test direct API call
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                direct_response = response.json()
                direct_status = "success"
            else:
                direct_status = f"error: {response.status_code}"
        except Exception as e:
            direct_status = f"exception: {str(e)}"
        
        # Test through function
        function_status = "not_attempted"
        function_response = {}
        
        try:
            function_response = get_seasonal_forecast(float(lat), float(lon))
            if function_response:
                function_status = "success"
            else:
                function_status = "empty_response"
        except Exception as e:
            function_status = f"exception: {str(e)}"
        
        # Test processing
        monthly_data = []
        processing_status = "not_attempted"
        
        if function_response:
            try:
                monthly_data = process_seasonal_forecast(function_response)
                processing_status = "success" if monthly_data else "empty_result"
            except Exception as e:
                processing_status = f"exception: {str(e)}"
        
        return jsonify({
            'status': 'success',
            'coordinates': {
                'lat': lat,
                'lon': lon
            },
            'api_key_present': bool(METEOBLUE_API_KEY),
            'direct_api_call': {
                'status': direct_status,
                'url': url.replace(METEOBLUE_API_KEY, "API_KEY_REDACTED"),
                'has_data': bool(direct_response),
                'has_monthly_data': 'data_monthly' in direct_response if direct_response else False,
                'response_keys': list(direct_response.keys()) if direct_response else []
            },
            'function_call': {
                'status': function_status,
                'has_data': bool(function_response),
                'has_monthly_data': 'data_monthly' in function_response if function_response else False,
                'response_keys': list(function_response.keys()) if function_response else []
            },
            'data_processing': {
                'status': processing_status,
                'monthly_count': len(monthly_data),
                'sample': monthly_data[0] if monthly_data else None
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error in test_seasonal_forecast: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)