# Food Delivery Intelligence Platform V2

## Business problem

This project addresses the operational decisions a food-delivery platform must make in real time:

1. What ETA should be displayed to users?
2. How much demand will arrive by restaurant and hour?
3. Which restaurants should be recommended to each customer?
4. How should those recommendations be ranked for relevance and business value?
5. Which delivery partner should receive each order while minimizing travel and delay?
6. Does personalization actually improve conversion and revenue?

The goal is not to create disconnected ML demos, but to build one end-to-end decision-support workflow around a real food-delivery business problem.

## Architecture

- Data engineering and validation with Python and Pandas
- SQLite analytics store for cleaned and aggregated order data
- ETA prediction using linear regression, random forest, and XGBoost
- Hourly demand forecasting using temporal lag and rolling features
- Hybrid restaurant recommendation based on available restaurant/order metadata
- Ranking evaluation with Precision@K, Recall@K, and NDCG@K
- OR-Tools assignment optimization for synthetic delivery-partner dispatch
- Offline A/B simulation to measure personalization lift
- FastAPI model-serving layer
- Streamlit dashboard for operational reporting

## Data source

The project uses the public Zomato delivery dataset:

https://github.com/Parth-Malik/Zomato-Delivery-Time-Prediction

The dataset is stored locally at `data/raw/Zomato Dataset.csv` and is not redistributed in this repository. If the file is not present, the project attempts to download it automatically in `src/download_data.py`.

## Methodology

The pipeline is designed as a real business workflow:

- validate raw records and standardize column names
- convert order and pickup timestamps into usable date/hour features
- engineer delivery-time features such as distance, pickup hour, day of week, prep time, and peak indicators
- train ETA models with chronological validation and a training-fitted imputing pipeline
- aggregate demand into hourly orders and generate lag/rolling features
- score recommendation candidates with a transparent hybrid relevance signal
- simulate a delivery-partner assignment problem using observed order characteristics
- run a simulated offline A/B test with personalized ranking vs popularity baseline

## Model choices

### ETA
- Linear Regression baseline
- Random Forest regressor
- XGBoost regressor

Each model uses a median imputer and is evaluated using MAE, RMSE, R², and the train/test R² gap. The chosen model is the one with the lowest MAE on a chronological holdout set.

### Demand forecasting
- Aggregated hourly demand from valid order-date and pickup-time observations
- lag features: 1, 2, 3, 24, 48, 168
- rolling 24-hour demand feature
- day-of-week and hour features
- HistGradientBoostingRegressor benchmark against the naive lag-24 baseline

### Recommendation and ranking
- Candidate generation using restaurant metadata and text similarity
- Relevance score using text similarity, rating, and popularity
- Ranking evaluation on Precision@K, Recall@K, and NDCG@K

### Optimization
- Assignment problem formulated as a cost-minimization linear program with partner capacity constraints
- Synthetic dispatch scenario derived from observed order distance and delivery-time patterns

### A/B testing
- Simulated/Offline A/B Experiment
- Control: popularity-based recommendation
- Treatment: personalized/ranked recommendation

## Evaluation metrics and business KPIs

### Delivery KPI
- ETA MAE
- ETA RMSE
- R² and train/test R² gap

### Demand KPI
- model MAE
- model RMSE
- baseline MAE
- baseline RMSE
- temporal demand observations and coverage

### Recommendation KPI
- Precision@K
- Recall@K
- NDCG@K

### Optimization KPI
- total assignment cost
- average assignment cost
- predicted delivery delay
- unassigned orders

### Experiment KPI
- CTR
- conversion rate
- AOV
- revenue per user
- absolute lift
- relative lift
- confidence interval
- p-value

## Limitations

- The public dataset does not contain trustworthy production user-item interaction history, so collaborative filtering is not fabricated.
- The optimization module uses a synthetic dispatch scenario derived from observed order characteristics, not real Swiggy/Zomato assignment state.
- The A/B test is an offline simulation intended to estimate directional business lift, not a real production experiment.
- Any forecasting configuration must be based on actual data coverage; if there are not enough valid temporal observations, the code raises a clear validation error instead of generating fake data.

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py
uvicorn app.api:app --reload
streamlit run app/dashboard.py
```

If the public dataset is not in `data/raw/`, the pipeline will tell you exactly what is missing and will not silently proceed with invalid data.
