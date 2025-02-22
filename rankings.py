from functools import partial
import polars as pl
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize

def load_and_preprocess():
    df = pl.scan_parquet('matches.parquet').with_row_index(name="row_index")

    df = df.filter(
        (pl.col('date') >= '2024-01-01')
        & (pl.col('date') <= '2024-12-3W1')
    )

    competitors_df = df.unpivot(index=[], on=["competitor_1", "competitor_2"])\
        .select("value").unique("value").sort("value").rename({"value": "competitor"})

    matches_df = (
        df.select([
            pl.col('competitor_1').cast(pl.Utf8).alias('comp1'),
            pl.col('competitor_2').cast(pl.Utf8).alias('comp2'),
            'row_index'
        ])
        .join(
            competitors_df.with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias('index1')),
            left_on='comp1',
            right_on='competitor'
        )
        .join(
            competitors_df.with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias('index2')),
            left_on='comp2',
            right_on='competitor'
        )
        .sort('row_index')
        .select(['index1', 'index2'])
    ).collect()

    print(matches_df)

    matches = matches_df[['index1', 'index2',]].to_jax()
    outcomes = df.collect()['outcome'].to_jax()
    return df, matches, outcomes, len(competitors_df.collect())

@jax.jit
def loss_fn(
    ratings,
    matches,
    outcomes,
    reg,
    ):
    rating_diffs = ratings[matches[:,0]] - ratings[matches[:,1]]
    probs = jax.nn.sigmoid(rating_diffs)
    ll = (jnp.log(probs) * outcomes).sum() + (jnp.log(1.0 - probs) * (1.0 - outcomes)).sum()
    reg = reg * jnp.linalg.norm(ratings)
    loss = reg - ll
    return loss

loss_and_grad = jax.value_and_grad(
    fun=loss_fn,
    argnums=(0,),
)


def bt(matches, outcomes, n_players, reg=1.0):
    initial_ratings = jnp.ones(n_players)
    result = minimize(
        partial(loss_fn, matches=matches, outcomes=outcomes, reg=reg),
        initial_ratings,
        method='BFGS',
    )
    ratings = result.x
    return ratings

def main():
    df, matches, outcomes, n_players = load_and_preprocess()
    ratings = bt(matches, outcomes, n_players, reg=10.0)
    print(ratings)



if __name__ == '__main__':
    main()
