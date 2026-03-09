import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Mock problematic imports or data loading before importing app
with patch('pandas.read_csv') as mock_read_csv:
    # return a small dummy dataframe for any read_csv call
    mock_read_csv.return_value = pd.DataFrame({
        'latitude': [5.6], 'longitude': [-0.2], 'suitability': [80]
    })
    
    # Mock requests to avoid API calls
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {}
        mock_get.return_value.status_code = 200

        # Add current directory to path to import app
        sys.path.append(os.getcwd())
        
        # Now import the class and function
        from app import GhanaDataDrivenRiskEngine, process_seasonal_forecast

def test_monte_carlo_and_limit():
    print("Starting verification of 4-month limit and Monte Carlo optimization...")
    
    # Test 1: Seasonal Forecast Limit
    print("\nTest 1: Seasonal Forecast Limit")
    # Create dummy data with 6 months
    seasonal_payload = {
        "data_seasonalmonthly": {
            "time": ["2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08"],
            "temperature_mean_anomaly": [[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]],
            "precipitation_mean_anomaly": [[-10, -11, -12, -13, -14, -15]]
        }
    }
    
    processed_seasonal = process_seasonal_forecast(seasonal_payload)
    print(f"Number of months returned: {len(processed_seasonal)}")
    
    # Verify limit
    assert len(processed_seasonal) == 4, f"Expected 4 months, got {len(processed_seasonal)}"
    assert processed_seasonal[0]['month'] == "2024-03"
    assert processed_seasonal[3]['month'] == "2024-06"
    print("✅ Seasonal Limit Test Passed!")

    # Test 2: Monte Carlo with Limited Data
    print("\nTest 2: Monte Carlo with Limited Data")
    engine = GhanaDataDrivenRiskEngine(latitude=5.6, longitude=-0.2)
    
    # Create dummy data
    price_data = pd.DataFrame({'maize': [100, 110, 105]})
    weather_data = pd.DataFrame({
        'temperature_max': [30, 31, 29],
        'temperature_min': [22, 23, 21],
        'precipitation_sum': [0, 5, 0]
    })
    suitability_data = {'maize': 80}
    
    # Use the limited data from Test 1
    results = engine.run_monte_carlo_analysis(
        price_data=price_data,
        weather_data=weather_data,
        suitability_data=suitability_data,
        seasonal_data=processed_seasonal,
        crop='maize',
        num_simulations=10
    )
    
    print(f"Monte Carlo Mean Risk: {results['simulation_statistics']['mean_risk']}")
    assert 'simulation_statistics' in results
    print("✅ Monte Carlo Execution Test Passed!")

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
    return True

if __name__ == "__main__":
    try:
        success = test_monte_carlo_and_limit()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Verification Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
