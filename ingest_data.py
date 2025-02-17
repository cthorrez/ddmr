import os
import json
import time
from hashlib import sha256
import requests
from diskcache import Cache
import polars as pl

def main():
    timeout = 62
    url = "https://api.liquipedia.net/api/v1/game"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'accept-encoding': 'gzip',
    }
    request_params = {
        'wiki': 'smash',
        'query': 'date, opponent1, opponent2, opponent1score, opponent2score, winner, map, extradata, type, matchid, pagename',
        'conditions': '[[walkover::!1]] AND [[walkover::!2]] AND [[mode::singles]] AND [[game::melee]] AND [[opponent1::!Bye]] AND [[opponent2::!Bye]]',
        'order': 'date ASC, matchid ASC',
        'limit': 1000,
        'apikey': os.getenv('LPDB_KEY')
    }
    cache = Cache('.cache')
    offset = 0
    results = []
    done = False
    while not done:
        print(f'making request for rows {offset} to {offset + 1000}')
        request_params['offset'] = offset
        request_key = sha256(json.dumps(request_params).encode()).hexdigest()
        if request_key in cache:
            print('getting request from cache')
            response_text = cache[request_key]
        else:
            print('making request to api')
            response = requests.post(
                url=url,
                data=request_params,
                headers=headers
            )
            response_text = response.text
            cache[request_key] = response_text
            time.sleep(timeout)
            
        result = json.loads(response_text)['result']
        results.extend(result)
        offset += 1000
        if (len(result) == 0):
            done = True


    df = pl.DataFrame(results)
    df = df.with_columns(
        pl.col('extradata').struct.field('char1').alias('char1'),
        pl.col('extradata').struct.field('char2').alias('char2')
    ).drop('extradata')
    print(df.schema)
    df.write_parquet('games.parquet')
    df.write_csv('games.csv')
    







if __name__ == '__main__':
    main()