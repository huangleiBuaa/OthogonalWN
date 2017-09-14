#!/bin/bash
methods=(wr_OLM_L1 wr_OLM)
lrs=(0.1)
datasets=(./dataset/cifar10_original.t7 ./dataset/cifar100_original.t7)

batchSize=128
weightDecay=0.0005
dr=0.3
depth=28
widen_factor=10
nN=1

n=${#methods[@]}
m=${#lrs[@]}
f=${#datasets[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	
     for ((k=0;k<$f;++k))
      do

    	echo "methods=${methods[$i]}"
    	echo "learningRates=${lrs[$j]}"
   	echo "dataset=${datasets[$k]}"
   CUDA_VISIBLE_DEVICES=0	th exp_Conv_wr.lua -model ${methods[$i]} -learningRate ${lrs[$j]} -dataset ${datasets[$k]} -max_epoch 100 -seed 1 -dropout ${dr} -m_perGroup 64 -batchSize ${batchSize} -weightDecay ${weightDecay} -widen_factor ${widen_factor} -depth ${depth} -noNesterov ${nN} 
      done
   done
done
