import numpy as np
from tqdm.auto import tqdm
from imageio import imread 
from tifffile import imwrite
from pathlib import Path
from imreg_dft import imreg
import scipy.ndimage as ndi
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
    parser.add_argument('-r','--register', action='store_true')
    args = parser.parse_args()


    fs = sorted(args.input)

    x = np.stack([imread(f) for f in tqdm(fs)])

    cutoff = 10

    xs = [x[0]]

    target = x[0, ::args.subsample, ::args.subsample]<cutoff
    target = (255*target).astype(np.uint8)

    for y in tqdm(x[1:]):

        if args.register:
            moving = y[::args.subsample, ::args.subsample]<cutoff
            moving = (255*moving).astype(np.uint8)
            y = register_simple(target, moving, y, args.subsample)

        xs.append(y)        

    xs = np.stack(xs)
    
    if len(args.output)>0:
        print(f'writing to {args.output}')
        imwrite(args.output, xs)
