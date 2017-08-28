--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train_expDecay'
local opts = require 'opts_imageNet'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
print('--------strat to train---------')
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
iteration=0
losses={}
train_accus={}
test_accus={}
train_accus_t5={}
test_accus_t5={}
losses_epoch={}
print(model)

results={}
for epoch = startEpoch, opt.nEpochs do
      
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

    train_accus[#train_accus+1]=100-trainTop1
    train_accus_t5[#train_accus_t5+1]=100-trainTop5
    losses_epoch[#losses_epoch+1]=trainLoss
   -- Run model on validation set

   local testTop1, testTop5 = trainer:test(epoch, valLoader)
    test_accus[#test_accus+1]=100-testTop1
    test_accus_t5[#test_accus_t5+1]=100-testTop5
   local bestModel = false
   if testTop1 < bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      print(' * Best model ', testTop1, testTop5)
   end


    if string.match(opt.model,'SVD')  then
        print('--update the SVD----------')

        for k,v in pairs(model:findModules('cudnn.Spatial_SVD_Group')) do
         v:updateWeight(true)
        end
        local  batch_inputs=torch.randn(8,3,224,224):cuda() --use a small batch data to forward， and udpate weight
         local  batch_outputs = model:forward(batch_inputs)
        for k,v in pairs(model:findModules('cudnn.Spatial_SVD_Group')) do
            v:updateWeight(false)
        end

        for k,v in pairs(model:findModules('cudnn.Spatial_SVD_OrthInit')) do
         v:updateWeight(true)
        end
        local  batch_inputs=torch.randn(8,3,224,224):cuda() --use a small batch data to forward， and udpate weight
         local  batch_outputs = model:forward(batch_inputs)
        for k,v in pairs(model:findModules('cudnn.Spatial_SVD_OrthInit')) do
            v:updateWeight(false)
        end
   end
   for k,v in pairs(model:findModules('cudnn.Spatial_PN')) do
       v:updateWeight()
    end


   print('traing evaluation: epoch='..epoch..'----train accu='..(100-trainTop1)..'------loss_epoch='..trainLoss)


	results.opt=opt
	--results.losses=losses
	results.train_accus=train_accus
	results.test_accus=test_accus
	results.train_accus_t5=train_accus_t5
	results.test_accus_t5=test_accus_t5
	results.losses_epoch=losses_epoch
	if opt.tenCrop then
  	torch.save('result_ED_'..opt.model..'_depth'..opt.depth..'_G'..opt.m_perGroup..'_LR'..opt.LR..'_'..opt.dataset..'_NE'..opt.nEpochs..'_tenCrop.dat', results)
	else
  	torch.save('result_ED_'..opt.model..'_depth'..opt.depth..'_G'..opt.m_perGroup..'_LR'..opt.LR..'_'..opt.dataset..'_NE'..opt.nEpochs..'.dat', results)

	end
 --   checkpoints.save(epoch, model, trainer.optimState, bestModel)
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
