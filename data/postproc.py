import numpy as np
from tqdm.auto import tqdm
from imageio import imread 
from tifffile import imwrite
from pathlib import Path
from imreg_dft import imreg
import scipy.ndimage as ndi
from skimage.measure import block_reduce
import argparse 
import cv2


def register_simple(target, moving, y, subsample):
    t = imreg.translation(target, moving)
    shift = t['tvec']*subsample
    print(shift)
    y = ndi.shift(y, shift, order=1)
    return y 

def register(target, moving, y, subsample=1):
    
    orb = cv2.ORB_create(128)
    (kpsA, descsA) = orb.detectAndCompute(moving, None)
    (kpsB, descsB) = orb.detectAndCompute(target, None)
    # match the features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descsA, descsB, None)

    matches = sorted(matches, key=lambda x:x.distance)
    # keep only the top matches
    keep = 1+int(len(matches) * .1)
    matches = matches[:keep]    

    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt    
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = target.shape[:2]
    aligned = cv2.warpPerspective(y, H, (w, h))

    return aligned
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i','--input', type=str, nargs='+')
    parser.add_argument('-s','--subsample', type=int, default=2)
    parser.add_argument('-o','--output', type=str, default='output.tif')
    parser.add_argument('-d','--downsample', type=int, default=2)
    parser.add_argument('-r','--register', action='store_true')
    args = parser.parse_args()


    def _downsample(x):
        if args.downsample>1:
            # x = ndi.zoom(x, (1/args.downsample,)*2, order=1)
            x = block_reduce(x,args.subsample, np.max)
        return x

    fs = sorted(args.input)

    x = np.stack([imread(f) for f in tqdm(fs)])

    cutoff = 10

    xs = [_downsample(x[0])]

    target = x[0, ::args.subsample, ::args.subsample]<cutoff
    target = (255*target).astype(np.uint8)

    for y in tqdm(x[1:]):

        if args.register:
            moving = y[::args.subsample, ::args.subsample]<cutoff
            moving = (255*moving).astype(np.uint8)
            y = register_simple(target, moving, y, args.subsample)


        xs.append(_downsample(y))

    xs = np.stack(xs)

    xs_subtract = np.clip(xs.astype(np.float32) - np.mean(xs, axis=0), 0,255).astype(np.uint8)
    
    if len(args.output)>0:
        print(f'writing to {args.output}')
        p = Path(args.output)
        p_sub = p.parent/f'{p.name}_subtracted.tif'
        imwrite(p, xs)
        imwrite(p_sub, xs_subtract)
