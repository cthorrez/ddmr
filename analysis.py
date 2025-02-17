import polars as pl

def main():
    df = pl.read_parquet('games.parquet')
    players = (
        df.unpivot(index=[], on=["opponent1", "opponent2"])
        .group_by("value")
        .len()
        .sort("len", descending=True)
    )
    print(players.head(10))

    characters = (
        df.unpivot(index=[], on=["char1", "char2"])
        .group_by("value")
        .len()
        .sort("len", descending=True)
    )
    for row in characters.to_dicts():
        print(row)

if __name__ == '__main__':
    main()