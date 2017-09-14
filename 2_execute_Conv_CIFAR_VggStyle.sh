#!/bin/bash
methods=(0debug_OLM 0debug_OLM_L2 0debug_OLM_L4 0debug_plain)

lrs=(0.05)
groups=(64)

n=${#methods[@]}
m=${#lrs[@]}
f=${#groups[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	
     for ((k=0;k<$f;++k))
      do

    	echo "methods=${methods[$i]}"
    	echo "learningRates=${lrs[$j]}"
   	echo "group=${groups[$k]}"
   CUDA_VISIBLE_DEVICES=0	th exp_Conv_VggStyle.lua -model ${methods[$i]} -learningRate ${lrs[$j]} -m_perGroup ${groups[$k]} -max_epoch 100 -seed 1 
      done
   done
done
