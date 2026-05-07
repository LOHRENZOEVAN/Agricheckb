# 🌍 AgriCheck: Precision Agricultural Risk Intelligence

**AgriCheck** is a next-generation risk assessment engine designed to bridge the gap between geospatial data, climatic volatility, and market economics. By fusing high-resolution suitability mapping for **Ghana and Cameroon** with real-time market signals and physiological growth-stage sensitivity, AgriCheck provides a multidimensional view of agricultural risk.

---

## 🚀 Core Innovations

### 1. Trans-Regional Geospatial Intelligence
AgriCheck leverages massive suitability datasets (Maize, Rice, Soya) spanning **Ghana and Cameroon**. Our engine doesn't just look at simple coordinates; it maps crop-specific soil and terrain suitability at a sub-kilometer resolution, providing the foundation for localized risk profiles.

### 2. Physiological Growth-Stage Awareness
Risk is not static. AgriCheck calculates the **Days After Planting (DAP)** and identifies the crop's current stage (Emergence, Vegetative, Flowering, or Maturation). Our engine applies **dynamic sensitivity multipliers**—recognizing that a heatwave during flowering is far more critical than during maturation.

### 3. The "Blended" Risk Engine (v2.4)
Our proprietary logic integrates three distinct data streams:
*   **Environmental Flux**: Real-time 14-day weather anomalies fused with 6-month seasonal forecasts via the Meteoblue API.
*   **Market Volatility**: Live processing of `crop_prices.csv` to calculate Coefficient of Variation (CV) and market exposure risks.
*   **ML Prediction**: A Scikit-Learn based ensemble model that validates rule-based logic against historical yield patterns.

### 4. Probabilistic Uncertainty Quantification
Using **Monte Carlo Simulations**, AgriCheck runs 1,000+ stochastic scenarios for every query. We don't just give you a number; we provide confidence intervals (P5, P50, P95) and the probability of "High Risk" events.

---

## 🛠️ Tech Stack

*   **Logic**: Python 3.11+, Flask (REST API)
*   **Intelligence**: Scikit-Learn, NumPy, Pandas
*   **Data**: Meteoblue API (Weather), High-Res CSV Suitability Matrices
*   **Simulation**: Stochastic Monte Carlo Analysis

---

## 📦 Installation & Setup

1.  **Clone & Environment**:
    ```bash
    git clone <your-repo-url>
    cd agricheckb
    python -m venv .venv
    source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
    ```

2.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration**:
    Create a `.env` file in the root directory:
    ```env
    METEOBLUE_API_KEY=your_api_key_here
    ```

4.  **Model Training**:
    Before running the API, initialize the datasets and train the risk model:
    ```bash
    python data_loader.py
    python train_model.py
    ```

5.  **Run the Engine**:
    ```bash
    python app.py
    ```

---

## 📊 API Snapshot

**Endpoint**: `POST /analyze_risk`  
**Payload**:
```json
{
  "latitude": 5.6037,
  "longitude": -0.1870,
  "crop": "maize",
  "planting_date": "2024-03-15"
}
```

**Response**: Returns a `composite_risk_score`, `blended_risk_score`, and detailed `thermal_analysis` including Growing Degree Days (GDD) adequacy.

---

Developed for the future of **Climate-Smart Agriculture** in West and Central Africa.
