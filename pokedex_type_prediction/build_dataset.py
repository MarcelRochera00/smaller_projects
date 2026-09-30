import time
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

N = 898
session = requests.Session()

def get_json(url, tries=3):
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  retry {attempt + 1} for {url}: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed: {url}")

def generation(i):
    for gen, limit in enumerate([151, 251, 386, 493, 649, 721, 809, 898], start=1):
        if i <= limit:
            return gen

rows = []
for i in range(1, N + 1):
    species = get_json(f"https://pokeapi.co/api/v2/pokemon-species/{i}")
    data = get_json(f"https://pokeapi.co/api/v2/pokemon/{i}")

    clean_text = ""
    for entry in species["flavor_text_entries"]:
        if entry["language"]["name"] == "en":
            clean_text = " ".join(entry["flavor_text"].split())
            break

    types = [t["type"]["name"] for t in data["types"]]
    rows.append({
        "id": i,
        "gen": generation(i),
        "name": data["name"],
        "type1": types[0],
        "type2": types[1] if len(types) > 1 else "",
        "types": "|".join(types),
        "text": clean_text,
    })
    print(i, data["name"], "/".join(types))
    time.sleep(0.1)

df = pd.DataFrame(rows)
df.to_csv(DATA / "pokemon_all.csv", index=False)
print(df.shape)