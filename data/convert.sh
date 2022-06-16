root=day3


# for f in `ls ${root}/*mov`; do
#     out=${f/.mov/};
#     mkdir -p ${out};
#     ffmpeg -y -i ${f} -vf "scale=720:-1" -f image2 -q:v 1 -qmin 1 -qmax 1 -vf format=gray ${out}/image-%4d.png;

#     in=${f/.mov/};
#     out=${f/.mov/}.tif;
# done


for f in `ls ${root}/*mov`; do
    in=${f/.mov/};
    out=${f/.mov/}.tif;
    python postproc.py -i ${in}/image-*.png -o ${out}  --register
done



