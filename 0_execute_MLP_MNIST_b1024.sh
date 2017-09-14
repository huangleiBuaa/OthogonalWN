#!/bin/bash
methods=(plain QR EI_QR CI_QR CayT OLM_var OLM)

lrs=(0.5 1)
layers=(4 6 8)

n=${#methods[@]}
m=${#lrs[@]}
f=${#layers[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	
     for ((k=0;k<$f;++k))
      do

    	echo "methods=${methods[$i]}"
    	echo "learningRates=${lrs[$j]}"
   	echo "nolayerinear=${layers[$j]}"
   	th exp_MLP_MNIST_debug.lua -model ${methods[$i]} -learningRate ${lrs[$j]} -layer ${layers[$k]} -seed 1 -batchSize 1024 -max_epoch 80
      done
   done
done
