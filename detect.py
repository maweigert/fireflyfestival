from pydoc import plain


import numpy as np
import argparse
from tqdm.auto import tqdm
import tifffile
import scipy.ndimage as ndi
from skimage.measure import regionprops, block_reduce
from skimage.feature import blob_dog
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i','--input', type=str, default='data/IMG_9420.tif')
    parser.add_argument('-o','--output', type=str, default='output.csv')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()




    if not 'img' in locals():
        print('reading image')
        img = (tifffile.imread(args.input)/255).astype(np.float32)
        img0 = np.mean(img, axis=0)
        img = np.clip(img-img0,0,1)
        threshold = 0.3
        print('labeling')
        label, _ = ndi.label(img>=threshold)
        print('regionprops')
        regs = regionprops(label, intensity_image=img)

    # rows = [] 

    # for r in tqdm(regs):
    #     t1, t2 = r.bbox[::3]
    #     z, y, x = r.centroid
    #     intens = r.intensity_mean
    #     bbox = r.bbox[1::3] + r.bbox[2::3]
    #     row = dict(id=r.label, z=z, y=y,x=x, t1=t1, t2=t2, intens=intens, bbox=bbox)
    #     rows.append(row)

    # df = pd.DataFrame.from_records(rows)

    # df.to_csv(args.output, index=False)

