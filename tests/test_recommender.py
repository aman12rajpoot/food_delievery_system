import pandas as pd
from src.recommender import ranking_metrics
def test_ranking_metrics():
    r=pd.DataFrame({"restaurant_name":["A","B","C"]})
    m=ranking_metrics(r,["A","C"],3)
    assert m["Precision@K"] == 2/3
    assert 0 <= m["NDCG@K"] <= 1
