# Generate source urls for retrieving dynamically generated tiles from titiler to mj_urls.txt.
# Titiler will perform a linear scaling from 0 to the given scale values for each variable
# This script assumes you have copied the mosaicjsons generated from 07-create-mosaicjsons.py 
# It also assumes you have uploaded overview mosaicjsons to blob_mj_path after 08-create-overviews.py

import fsspec
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--years', nargs='+', default = ['2015','2019'], help="Subset of years to produce urls for.")
    parser.add_argment('-td', '--titiler-domain', default = 'https://forest-disturbance-tiler.azurewebsites.net',
                        help = 'Domain of titiler server deployed on Azure App Services. See README.')
    parser.add_argument('-mp', '--blob-mj-path', default='app/mosaicjson', \
                         help="Parent virtual directory of hls collection outputs i.e contains years as subdirs.")
    parser.add_argument('-pv', '--predicted_vars', nargs='+', default=["basal_area","bio_acre","canopy_cvr","class"], \
                        help="Variables set to predict in the parameters yml for model.")
    parser.add_argument('-s', '--scales', nargs='+', default=['350','350','100','None'], \
                        help="Maximum values for each of pv. A default discrete colormap is used for class.")
    args = parser.parse_args()

    assert len(pv) == len(scales), \
           'Please provide a scale maximum for each variable.'


    vsscls = list(zip(args.predicted_vars,arg.scales))
    ttpx = f"{args.ttdomain}/mosaicjson/tiles/{z}/{x}/{y}@2x.png?url="
    mjbpx = f"https://{os.environ['STRG_ACCOUNT_NAME']}.blob.core.windows.net/"
    # See https://developmentseed.org/titiler/examples/code/tiler_with_custom_colormap/ for colormap details
    clssuf = '&rescale=0%2C51&colormap=%7B%220%22%3A+%22%2300000000%22%2C+%225%22%3A+%22%2374a33b%22%2C+%2210%22%3A+%22%23cb6527%22%2C+%2215%22%3A+%22%23dba93d%22%2C+%2220%22%3A+%22%23373737%22%7D'
    fmap = fsspec.filesystem('az', account_name=os.environ['STRG_ACCOUNT_KEY'], account_key=os.environ['STRG_ACCOUNT_KEY'])
    mosaicjsons = fmap.ls(args.blob_mj_path)
    if os.path.exists('mj_urls.txt'):
        os.remove('mj_urls.txt')
    with open('mj_urls.txt', 'a') as file:
        for yr in years:
            for rastvar,rsc in vsscls:
                matchstrs = [x for x in mosaicjsons if ((yr in x) & (rastvar in x))]
                ovr, fr = matchstrs
                if rastvar == 'class':
                    fullovrurl = f'{ttpx}{mjbpx}{ovr}{clssuf}'
                    fullfrurl = f'{ttpx}{mjbpx}{fr}{clssuf}'
                else:
                    fullovrurl = f'{ttpx}{mjbpx}{ovr}&colormap_name=viridis&rescale=0,{rsc}'
                    fullfrurl = f'{ttpx}{mjbpx}{fr}&colormap_name=viridis&rescale=0,{rsc}'
                file.write(f'URL for {rastvar}, {yr} overview is:\n')
                file.write(fullovrurl+'\n')
                file.write(f'URL for {rastvar}, {yr} full-resolution is:\n')
                file.write(fullfrurl+'\n')
