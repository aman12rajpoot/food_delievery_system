# Project Walkthrough

## 1. Start with the business problem

Food-delivery operations need better ETA accuracy, demand visibility, restaurant discovery, and delivery assignment.

## 2. Establish baselines

- ETA: simple linear model
- Demand: previous-day same-hour demand
- Recommendation: popularity/quality baseline
- Assignment: unoptimized feasible assignment

## 3. Improve models

Compare tree-based ETA models, feature-based demand forecasting, hybrid ranking, and OR-Tools optimization.

## 4. Evaluate business outcomes

Do not only report ML metrics. Connect metrics to decisions:
- lower ETA error -> better customer expectation setting
- lower forecast error -> better staffing/preparation planning
- better ranking -> more relevant restaurant discovery
- lower assignment cost -> less travel/delay
- A/B test -> evidence for whether personalization should be deployed

## 5. Interview discussion

Be ready to explain:
- leakage prevention
- temporal validation
- why public data limits collaborative filtering
- ranking metrics
- optimization objective and constraints
- statistical significance vs practical significance
- API design and model serving
