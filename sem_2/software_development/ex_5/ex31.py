import requests
import polars as pl

url = "https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-1d"
params = {"parameters": "tl_i",
          "station_ids": "30",
          "start": "2026-01-01",
          "end": "2026-11-04"}
r = requests.get(url, params=params)
jsdict = r.json()
data = jsdict['features'][0]['properties']['parameters']['tl_i']['data']
df = pl.DataFrame({"tl_i": data})
df['tl_i'].arg_max()
df['tl_i'].max()

df['tl_i'].arg_min()
df['tl_i'].min()
df.filter(pl.col("tl_i") > 30)
