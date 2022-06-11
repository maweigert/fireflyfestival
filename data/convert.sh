root=day2
for f in ${root}/*MOV; do
    out=${f/.MOV/};
    mkdir -p ${out};
    ffmpeg -y -i ${f} -vf "scale=720:-1" -f image2 -q:v 1 -qmin 1 -qmax 1 -vf format=gray ${out}/image-%4d.png;
done

for f in ${root}/*MOV; do
    in=${f/.MOV/};
    out=${f/.MOV/}.tif;
    echo $in $out
    python postproc.py -i ${in}/*.png -o ${out} 
done

