#!/bin/bash
C=0
theDate=$(date +%Y-%m-%d)
if [ ! -d log ];then
    mkdir log
fi
for i in $(find GAN RBM VAE -name "*.py");do
    script=$(echo $i | awk -F '/' '{print $3}')
    logfile=$PWD/log/$script.$C.$theDate.log
    dirmod=$(echo $i | awk -F '/' '{print $1}')/$(echo $i | awk -F '/' '{print $2}')
    pushd $dirmod > /dev/null
    (export CUDA_VISIBLE_DEVICE=$C;annotate-output +"%Y-%m-%d %H:%M:%S" time python3 -u $script |& tee $logfile)
    popd > /dev/null
done
