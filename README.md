# Food Delivery Intelligence Platform V2

An end-to-end food-delivery data science platform built around a public Zomato delivery dataset. The project connects data engineering, machine learning, recommendation/ranking, delivery optimization, experimentation, API serving, and a customer-facing Streamlit interface in one workflow.

## What the platform does

The platform covers the main decisions in a food-delivery system:

1. Predict delivery ETA for an order.
2. Forecast hourly demand for capacity planning.
3. Generate and rank restaurant recommendations from available restaurant/order metadata.
4. Assign orders to delivery partners subject to capacity constraints.
5. Evaluate recommendation changes with offline ranking metrics and a simulated A/B experiment.
6. Serve the models through FastAPI.
7. Provide a customer-style Streamlit experience with login, cart, checkout, and order history.

The goal is to demonstrate one connected data science workflow rather than a collection of isolated ML notebooks.

## Project architecture

```text
Public delivery dataset
        |
        v
Python + Pandas ETL
        |
        +------> SQLite analytics store
        |
        +------> ETA model
        |
        +------> Demand forecasting
        |
        +------> Restaurant catalog
        |           |
        |           v
        |     TF-IDF + hybrid ranking
        |
        +------> Dispatch optimization
        |
        v
FastAPI backend
        |
        v
Streamlit customer interface
(Login -> Home -> Find Food -> Restaurant -> Cart -> Checkout -> Orders)
```

## Technology stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SQL / SQLite
* FastAPI
* Streamlit
* OR-Tools
* Joblib
* SciPy
* Matplotlib / Seaborn

##

## Customer-facing Streamlit application

The Streamlit application is designed as a customer-facing interface rather than an analytics dashboard.

Current flow:

```text
Login / Sign up
      |
      v
Home / Find Food
      |
      v
Restaurant
      |
      v
Add to Cart
      |
      v
Cart
      |
      v
Checkout
      |
      v
Orders
```

### Authentication

Users can create an account with:

* name
* User ID
* mobile number
* password

## Repository structure

```text
food_delievery_system/
│
├── app/
│   ├── api.py
│   ├── dashboard.py
│   └── __init__.py
│
├── src/
│   ├── etl.py
│   ├── train_models.py
│   ├── recommender.py
│   ├── optimization.py
│   ├── ab_test.py
│   ├── validate_models.py
│   └── download_data.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── food_delivery.db
│
├── models/
│   ├── eta_features.json
│   ├── demand_features.json
│   └── restaurant_catalog.csv
│
├── reports/
│   ├── eta_metrics.json
│   ├── demand_metrics.json
│   ├── ranking_metrics.json
│   ├── ab_test_results.json
│   └── optimization_validation.json
│
├── tests/
├── notebooks/
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## Reproducible run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python run_pipeline.py
```

Start the FastAPI backend:

```bash
python -m uvicorn app.api:app --reload
```

```text
```

## Limitations and honest interpretation

* The public dataset is not production Zomato or Swiggy operational data.
* Restaurant names in the generated catalog are coordinate-based proxies derived from the available public data.
* Collaborative filtering is not fabricated because trustworthy production user-item interaction history is unavailable.
* Optimization uses a synthetic dispatch scenario rather than real courier assignment state.
* The A/B test is an offline simulation, not a production causal experiment.
* Demo authentication is not production identity management or secure session management.
* Streamlit cart and order state are local demo state rather than a distributed production order database.
* The source dataset does not provide reliable item-level menu and price data, so the UI intentionally avoids fabricating them.
* The customer interface does not provide real payment processing or live courier telemetry.
* Forecasting raises validation errors when the available temporal data is insufficient rather than fabricating observations.

##

The design also makes assumptions and limitations explicit, which is important when working with public datasets that do not contain the same information as a production food-delivery platform.
