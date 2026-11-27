import polars as pl

df = pl.read_csv("quakes.csv")
print(df)

df = df.sort("mag").select(["mag", "place"])
df = df.with_columns(pl.col("place").str.split(",").list.last().alias("region"))
print(df)

df = pl.read_csv("quakes.csv").select("place")
df = df.with_columns(
    pl.col("place").str.split(",").list.last().str.strip_chars()
      .str.split(" ").list.first().alias("region")
)
print(df.group_by("region").len().sort("len", descending=True))
