#!/bin/bash
C=0
theDate=$(date +%Y-%m-%d)
if [ ! -d log ];then
    mkdir $log
fi
for i in $(find GAN RBM VAE -name "*.py");do
    logfile=$log/$(echo $i | awk -F '/' '{print $3}').$C.$theDate.log
    (export CUDA_VISIBLE_DEVICE=$C;annotate-output +"%Y-%m-%d %H:%M:%S" time python3 $i |& tee $logfile)
done
