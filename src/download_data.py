from pathlib import Path
import requests

URL = "https://raw.githubusercontent.com/Parth-Malik/Zomato-Delivery-Time-Prediction/main/Zomato%20Dataset.csv"
OUT = Path("data/raw/Zomato Dataset.csv")

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"Dataset already exists: {OUT}")
        return
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    OUT.write_bytes(r.content)
    print(f"Downloaded dataset to {OUT}")

if __name__ == "__main__":
    main()
