Orthogonal Weight Normalization
======================================
## Requirements and Dependency
* install [Torch](http://torch.ch) with CUDA GPU
* install [cudnn v5](http://torch.ch)
* install dependent lua packages optnet by run:
luarocks install optnet
* install Magga (you can follow the instruction as in the file of ['Install Magga.txt'](./Install_Magga.txt) )
	Note: Magga is used for the SVD on GPU. if you don't install Magga, you can not run the code on GPU (For all the experiment on CNNs, we run the experiment on GPU )

## Experiments in the paper

#### 1.  Reproduce the results for sovling OMSDM problem:

*	run script:: 
```Bash
 bash 0_execute_MLP_MNIST_b1024.sh
 ```
This script will download MNIST dataset automatically.
	you can try more small learning rate, and add more layer, or use different batch size based on this script.
	
#### 2. Reproduce the results on MLP architecture:
* Dataset preparations: you should download the [PIE dataset](https://www.dropbox.com/sh/5pkrtv02wemqxzp/AADlVOs3vDMOEsOpRFa20Uqha?dl=0), and put the data file in the directory: './dataset/'

* Execute:
```Bash
  bash 1_execute_MLP_PIE_sgd.sh   
	  bash 1_execute_MLP_PIE_adam.sh
 ```
-----------------------------Note that the experiment above is under MLP and run on CPU, so Magga is not necessary for above experiemnt --------------------
 
#### 3. Reproduce the results on VGG style, BN-Incption and Wide residual network over CIFAR datset: 

 *	Dataset preparations: you should download the [CIFAR-10](https://yadi.sk/d/eFmOduZyxaBrT) and [CIFAR-100](https://yadi.sk/d/ZbiXAegjxaBcM) datasets, and put the data file in the directory: './dataset/' 
 * Execute: 
```Bash
th exp_GoogleNet_dataWhitening.lua –dataPath './dataset/cifar100_whitened.t7'
 ```
  *	To reproduce the experimental results, you can run the script below, which include all the information of experimental configuration: 
```Bash
  bash 2_execute_Conv_CIFAR_VggStyle.sh  
  bash 3_execute_Conv_CIFAR_BNInception.sh 
  bash 4_execute_Conv_CIFAR_wr.sh  
 ```
 


#### 4. Run the experiment on imageNet dataset. 

 *  (1) you should clone the facebook residual network project from:https://github.com/facebook/fb.resnet.torch
 *  (2) you should download imageNet dataset and put it on: '/tmp/dataset/imageNet/' directory (you also can change the Path, and change in 'opts_imageNet.lua' respectively)
 *  (3) copy  'opts_imageNet.lua', 'exp_Conv_imageNet_expDecay.lua', 'train_expDecay.lua', 'module' and 'models' to the project's root path.
 *  (4)	Execute: 
```Bash
th exp_Conv_imageNet_expDecay.lua -model imagenet/resnet_OLM_L1
 ```
you can change respective model by pass  the parameter 'model'

## Contact
huanglei@nlsde.buaa.edu.cn, Any discussions and suggestions are welcome!

