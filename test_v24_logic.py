import numpy as np
import logging

# Mock objects needed for testing
class GhanaZone:
    ASHANTI = "ashanti"

class MockEngine:
    def __init__(self):
        self.logger = logging.getLogger("test")
        logging.basicConfig(level=logging.INFO)

    def _compute_seasonal_risk(self, seasonal_data, crop_params):
        """
        Copy of the updated v2.4 logic
        """
        if not seasonal_data: 
            return 0.5
        
        try:
            temp_medians = []
            precip_medians = []
            
            print(f"Computing seasonal risk (v2.4) for {len(seasonal_data)} months")
            
            for month_data in seasonal_data:
                # Extract anomalies
                temp = month_data.get("temperature", {})
                precip = month_data.get("precipitation", {})
                
                t_anom = [float(x) for x in temp.get("mean_anomaly", []) if x is not None]
                if t_anom:
                    temp_medians.append(np.median(t_anom))
                
                p_anom = [float(x) for x in precip.get("mean_anomaly", []) if x is not None]
                if p_anom:
                    precip_medians.append(np.median(p_anom))
            
            # Calculate averages for the seasonal period
            avg_temp_anomaly = np.mean(temp_medians) if temp_medians else 0.0
            avg_precip_anomaly = np.mean(precip_medians) if precip_medians else 0.0
            
            # Continuous Risk Mapping (Ghana-Optimized)
            temp_risk = min(abs(avg_temp_anomaly) / 5.0, 1.0)
            precip_risk = min(abs(avg_precip_anomaly) / 80.0, 1.0)
            
            seasonal_risk = (temp_risk * 0.3) + (precip_risk * 0.7)
            
            print(f"\n{'='*60}")
            print(f"🌡️ SEASONAL RISK DEBUG (v2.4)")
            print(f"{'='*60}")
            print(f"Valid months: {len(seasonal_data)}")
            print(f"Temp medians: {[round(m, 2) for m in temp_medians]}")
            print(f"Precip medians: {[round(m, 2) for m in precip_medians]}")
            print(f"Avg temp anomaly: {avg_temp_anomaly:.2f}°C")
            print(f"Avg precip anomaly: {avg_precip_anomaly:.2f}mm")
            print(f"Temp risk component: {temp_risk:.4f}")
            print(f"Precip risk component: {precip_risk:.4f}")
            print(f"✅ FINAL SEASONAL RISK: {seasonal_risk:.4f}")
            print(f"{'='*60}\n")
            
            return seasonal_risk
            
        except Exception as e:
            print(f"Error: {e}")
            return 0.5

# Test data from user prompt
test_data = [
    {
        "month": "2026-03-01",
        "temperature": {
            "mean_anomaly": [0.09, 0.4, 0.15, -0.02, 0.18, -0.44, 0.08, 0.42]
        },
        "precipitation": {
            "mean_anomaly": [0.7, 4.6, 5.0, -5.3, -1.4, -3.3, 0.6, 4.5]
        }
    }
] * 4 # Simulate 4 months of similar data

engine = MockEngine()
risk = engine._compute_seasonal_risk(test_data, {})
print(f"Resulting Risk: {risk}")
