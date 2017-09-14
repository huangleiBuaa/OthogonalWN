#!/bin/bash
methods=(plain WN_scale OLM OLM_scale batch WN_scale_batch OLM_scale_batch)

lrs=(0.05 0.1 0.2 0.5 1 2)
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
   	echo "group=${groups[$j]}"
   	th exp_MLP_PIE.lua -model ${methods[$i]} -learningRate ${lrs[$j]} -mode_nonlinear 2 -m_perGroup_WDBN ${groups[$k]} -optimization simple -seed 1 -batchSize 256
      done
   done
done
