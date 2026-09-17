from src.ab_test import run
def test_ab_output():
    x=run(1000)
    assert "p_value" in x
    assert 0 <= x["p_value"] <= 1
