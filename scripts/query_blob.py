# Query a blob with output hls data from data_ingestion to form
# a tiles csv to control subsetting of predictions.

import argparse
import os
import pandas as pd

import fsspec
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--years', nargs='+', help="Subset of years in -bd to produce tiles csv for.")
    parser.add_argument('-bd', '--base-directory', default='fia/hls', \
                         help="Parent virtual directory of hls collection outputs i.e contains years as subdirs.")
    args = parser.parse_args()

    dirnames = [str(float(x)) for x in args.years]

    map = fsspec.filesystem('az', account_name=os.environ['STRG_ACCOUNT_NAME'], account_key=os.environ['STRG_ACCOUNT_KEY'])
    output = []
    for dirn in dirnames:
        zarrs = map.ls(f'{args.base_directory}/{dirn}')
        tilenames = [x.split('.zarr')[-2][-5:] for x in zarrs]
        tileyears = np.full(len(tilenames), int(dirn[:-2]))
        output.extend(list(zip(tilenames, tileyears)))
    tiledf = pd.DataFrame({'tile':np.array(output)[:,0],'year':np.array(output)[:,1]})
    suf = dirnames[0] if len(dirnames)==1 else f'{dirnames[0]-dirnames[-1]}'
    tiledf.to_csv(f'collected_hls_{suf}.csv',index=False)
