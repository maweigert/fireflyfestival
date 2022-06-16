""" 
reads and registers all frames of a firefly movie and saves them as zarr 

python convert_movie.py -i movie.mov -o output.zarr


"""

import numpy as np
from tqdm.auto import tqdm
import imageio
import itertools
import torch
from pathlib import Path
import zarr
from imreg_dft import imreg
import scipy.ndimage as ndi
import argparse


def register_simple(target, moving, subsample=1):
    x = target[::subsample, ::subsample]
    y = moving[::subsample, ::subsample]

    t = imreg.translation(x, y)
    shift = t["tvec"] * subsample
    y = ndi.shift(moving, shift, order=1)
    return y


def to_gray_and_downsample(x, factor):
    assert x.ndim == 3 and x.shape[-1] == 3
    x = np.mean(x, axis=-1)
    if factor > 1:
        # x = block_reduce(x, factor, np.max)
        x = torch.tensor(x).unsqueeze(0).unsqueeze(0)
        x = torch.nn.functional.max_pool2d(x, factor, stride=factor)
        x = x.numpy()[0, 0]
    x = x / 255
    return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input", type=str, help="input movie")
    parser.add_argument("-o", "--output", type=str, default="output.zarr")
    parser.add_argument("-d", "--downsample", type=int, default=2)
    parser.add_argument("--register_sub", type=int, default=2)
    parser.add_argument("-n", "--n_frames", type=int, default=None)
    args = parser.parse_args()

    video = imageio.get_reader(args.input, "ffmpeg")

    x = np.stack(
        [
            to_gray_and_downsample(f, args.downsample)
            for f in tqdm(
                itertools.islice(video.iter_data(), args.n_frames),
                total=video.count_frames() if args.n_frames is None else args.n_frames,
                desc="loading frames from video",
            )
        ]
    )

    cutoff = 0.04

    xs = [x[0]]

    for moving in tqdm(x[1:], desc="registering"):
        y = register_simple(x[0], moving, args.register_sub)
        xs.append(y)

    xs = np.stack(xs)

    xs_subtract = xs.astype(np.float32) - np.mean(xs, axis=0)

    if len(args.output) > 0:

        p = Path(args.output)
        p.parent.mkdir(exist_ok=True, parents=True)
        p_sub = p.parent / f'{p.with_suffix("").name}_subtracted.zarr'
        print(f"writing {p}")
        zarr.save(p, xs)
        print(f"writing {p_sub}")
        zarr.save(p_sub, xs_subtract)
