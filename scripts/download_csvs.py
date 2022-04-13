# Create a directory called 'csvs' and download the 3 curated csvs needed
# to generate indexed FIA targets for training (03-create-gold-data.py).

import pandas as pd
import os
from datetime import datetime
import requests

    
container = 'https://usfs.blob.core.windows.net/fia'
outdir = f'{os.getcwd()}/csvs'
if not os.path.exists(outdir):
    os.makedirs(outdir)
for fn in ['fia_no_pltcn.csv', 'fia_ytrain.csv', 'labeled-non-forest.csv']:
    if not os.path.exists(f'{outdir}/{fn}'):
        url = f'{container}/{fn}'
        start = datetime.now().timestamp()
        res = requests.get(url)
        with open(f'{outdir}/{fn}', 'wb') as f:
            f.write(res.content)
        end = datetime.now().timestamp()
        print(f'Downloading csv from blob took {end - start} seconds.')
    else:
        print('CSV already downloaded.')