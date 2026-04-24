import sys
import os
import json
import datetime
from unittest.mock import MagicMock

# Mock Flask and other dependencies
mock_flask = MagicMock()
sys.modules['flask'] = mock_flask
sys.modules['flask_cors'] = MagicMock()

# Mock the logic from app.py
sys.path.append('.')
import app

def test_yield_and_stage():
    print("Testing Yield and Stage Integration...")
    
    # Initialize engine
    engine = app.GhanaDataDrivenRiskEngine(latitude=5.6, longitude=-0.2)
    
    # 1. Test Growth Stage Calculation
    planting_date = (datetime.datetime.now() - datetime.timedelta(days=70)).isoformat()
    stage_info = engine._get_growth_stage('maize', planting_date)
    print(f"Growth Stage (70 days): {stage_info['stage']}, DAP: {stage_info['days_after_planting']}")
    
    # Expected: 70 days for maize should be "Flowering/Tasseling"
    assert stage_info['stage'] == "Flowering/Tasseling"
    assert stage_info['days_after_planting'] == 70
    
    # 2. Test Yield Calculation with Hectares
    # area_hectares = 0.65
    yield_res = engine.calculate_yield_prediction(
        crop='maize', 
        area_hectares=0.65, 
        fertilizer=True, 
        seed_variety='improved',
        suitability_score=80,
        risk_score=0.2
    )
    print(f"Yield Result (0.65 Ha): {json.dumps(yield_res, indent=2)}")
    
    # check if area_hectares is 0.65
    assert yield_res['area_hectares'] == 0.65
    # potential yield for maize with fert and improved seed: 1.5 + 1.5 + 1.5 = 4.5
    # actual yield per ha: 4.5 * 0.8 * (1-0.2) = 4.5 * 0.8 * 0.8 = 2.88
    # total yield: 2.88 * 0.65 = 1.872 -> rounded 1.87
    assert yield_res['predicted_yield_mt'] == 1.87
    
    # 3. Test Yield Calculation with Acres (Backward Compatibility)
    yield_res_ac = engine.calculate_yield_prediction(
        crop='maize', 
        area_acres=1.0, 
        fertilizer=True, 
        seed_variety='improved',
        suitability_score=80,
        risk_score=0.2
    )
    print(f"Yield Result (1.0 Acre): {yield_res_ac['area_hectares']} Ha")
    assert yield_res_ac['area_hectares'] == 0.4  # 1 * 0.4047 rounded to 0.4
    
    print("✅ All logic tests passed!")

if __name__ == "__main__":
    try:
        test_yield_and_stage()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
