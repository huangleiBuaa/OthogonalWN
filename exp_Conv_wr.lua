-- Code for Wide Residual Networks http://arxiv.org/abs/1605.07146
-- (c) Sergey Zagoruyko, 2016
require 'xlua'
require 'optim'
require 'image'
require 'cunn'
require 'cudnn'
local c = require 'trepl.colorize'
local json = require 'cjson'
paths.dofile'augmentation.lua'

-- for memory optimizations and graph generation
local optnet = require 'optnet'
local graphgen = require 'optnet.graphgen'
--local iterm = require 'iterm'
--require 'iterm.dot'
  -- threads
   threadNumber=2
 torch.setnumthreads(threadNumber)


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('compare the Decorelated BatchNormalizaiton method with baselines on wide-resnet architechture')
cmd:text()
cmd:text('Options')

cmd:option('-dataset','./dataset/cifar10_original.t7','')
cmd:option('-model','wr_WDBN_BN_NS_v1','')
cmd:option('-max_epoch',200,'maximum number of iterations')
cmd:option('-epoch_step',"{60,120,160}",'epoch step: no lr annealing if it is larger than the maximum')
cmd:option('-save',"log_exp_Cifar10_NoW" ,'subdirectory to save logs')
cmd:option('-batchSize',128,'the number of examples per batch')

cmd:option('-optimMethod','sgd','the methods: options:sgd,rms,adagrad,adam')
cmd:option('-learningRate',0.1,'initial learning rate')
cmd:option('-learningRateDecay',0,'initial learning rate')
cmd:option('-learningRateDecayRatio',0.2,'initial learning rate')
cmd:option('-weightDecay',0.0005,'weight Decay for regularization')
cmd:option('-dampening',0,'weight Decay for regularization')
cmd:option('-momentum',0.9,'momentum')
cmd:option('-m_perGroup',16,'the number of per group')
cmd:option('-m_perGroup_W',64,'the number of per group')
cmd:option('-eps',1e-5,'the revisation for DBN')

cmd:option('-BNScale',1,'the initial value for BN scale')
cmd:option('-scaleIdentity',0,'1 indicates scaling the Identity shortcut;0 indicates not')
cmd:option('-noNesterov',1,'1 indicates dont use nesterov momentum;0 indicates not')
cmd:option('-topK',8,'for DBN_PK method, scale the topK eigenValue')
cmd:option('-eig_epsilo',1e-2,'for DBN_PEP method, scale the eigenValue larger eig_epsilo')

cmd:option('-widen_factor',1,'')
cmd:option('-depth',8,'')
cmd:option('-hidden_number',48,'')

cmd:option('-optimMethod','sgd','')
cmd:option('-init_value',10,'')
cmd:option('-shortcutType','A','')
cmd:option('-nesterov',false,'')
cmd:option('-dropout',0,'')
cmd:option('-hflip',true,'')
cmd:option('-randomcrop',4,'')
cmd:option('-imageSize',32,'')
cmd:option('-randomcrop_type','reflection','')
cmd:option('-cudnn_fastest',true,'')
cmd:option('-cudnn_deterministic',false,'')
cmd:option('-optnet_optimize',false,'')
cmd:option('-generate_graph',false,'')
cmd:option('-multiply_input_factor',1,'')

--for debug weight norm
cmd:option('-weight_debug',0,'0 indicates not debug weight; 1 indicates debug Global weight and GradW; 2 indicates add observe per module')
cmd:option('-step_WD',1,'the step to debug the weights')
cmd:option('-seed',1,'the step to debug the weights')
cmd:option('-T',389,'the step to debug the weights')
cmd:option('-modelSave',1,'the step to debug the weights')

opt = cmd:parse(arg)

opt.rundir = cmd:string('console/exp_0_debug_Cifar/Info', opt, {dir=true})
paths.mkdir(opt.rundir)

cmd:log(opt.rundir .. '/log', opt)

cutorch.manualSeed(opt.seed)

--opt = xlua.envparams(opt)
if opt.noNesterov==1 then opt.nesterov=true end

opt.epoch_step = tonumber(opt.epoch_step) or loadstring('return '..opt.epoch_step)()
print(opt)

print(c.blue '==>' ..' loading data')
--local provider = torch.load(opt.dataset)

local meanstd = {mean = {125.3, 123.0, 113.9}, std  = {63.0,  62.1,  66.7}}

 local provider = torch.load(opt.dataset)
  opt.num_classes = provider.testData.labels:max()
    if torch.type(provider.trainData.data) == 'torch.ByteTensor' then
    for i,v in ipairs{'trainData', 'testData'} do
             provider[v].data = provider[v].data:float()--:div(256)
       for ch=1,3 do
        provider[v].data:select(2,ch):add(-meanstd.mean[ch]):div(meanstd.std[ch])
        end
    end
 end




opt.num_classes = provider.testData.labels:max()
opt.num_feature = provider.testData.data:size(2)

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
local net = dofile('models/CIFAR/'..opt.model..'.lua'):cuda()
  -- print('-------------------')
do
   local function add(flag, module) if flag then model:add(module) end end
   add(opt.hflip, nn.BatchFlip():float())
   add(opt.randomcrop > 0, nn.RandomCrop(opt.randomcrop, opt.randomcrop_type):float())
   model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
   add(opt.multiply_input_factor ~= 1, nn.MulConstant(opt.multiply_input_factor):cuda())
   model:add(net)

   cudnn.convert(net, cudnn)
   cudnn.benchmark = true
   if opt.cudnn_fastest then
      for i,v in ipairs(net:findModules'cudnn.SpatialConvolution') do v:fastest() end
   end
   if opt.cudnn_deterministic then
      model:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
   end
   print(net)
   print('Network has', #model:findModules'cudnn.SpatialConvolution', 'convolutions')

   local sample_input = torch.randn(8,3,opt.imageSize,opt.imageSize):cuda()
   if opt.generate_graph then
      iterm.dot(graphgen(net, sample_input), opt.save..'/graph.pdf')
   end
   if opt.optnet_optimize then
      --optnet.optimizeMemory(net, sample_input, {inplace = false, mode = 'training'})
   end
end

local function log(t) print('json_stats: '..json.encode(tablex.merge(t,opt,true))) end
baseString=opt.model..'_depth'..opt.depth
         ..'_h'..opt.hidden_number..'_lr'..opt.learningRate
         ..'_G'..opt.m_perGroup..'_b'..opt.batchSize..'_wf'..opt.widen_factor..'_s'..opt.scaleIdentity
         ..'_wD'..opt.weightDecay..'_mm'..opt.momentum
         ..'_nN'..opt.noNesterov..'_lD'..opt.learningRateDecay..'_dr'..opt.dropout
         ..'_seed'..opt.seed

         
log_name='Cifar10_NoW_'..baseString
opt.save=opt.save..'/'..log_name
print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

local parameters,gradParameters = model:getParameters()

opt.n_parameters = parameters:numel()
print('Network has ', parameters:numel(), 'parameters')

print(c.blue'==>' ..' setting criterion')
local criterion = nn.CrossEntropyCriterion():cuda()

-- a-la autograd
local f = function(inputs, targets)
   model:forward(inputs)
   local loss = criterion:forward(model.output, targets)
   local df_do = criterion:backward(model.output, targets)
   model:backward(inputs, df_do)
   return loss
end

print(c.blue'==>' ..' configuring optimizer')
local optimState = tablex.deepcopy(opt)


function debug_scale(name)
   for k,v in pairs(model:findModules(name)) do
      local scale= v.weight
      --local bias= v.bias
      local norm=v.weight:norm(1)/v.weight:numel()
      local mean=v.weight:mean() 
     print(name..'--scale Norm:'..norm..'--mean:'..mean)
   end
end

function debug_recordScale()
  local scales={}
  for k,v in pairs(model:findModules('nn.SpatialBatchNormalization')) do
   -- print('----------debug match--------') 
    table.insert(scales, v.weight:float())
  end
   for k,v in pairs(model:findModules('nn.Spatial_Scaling')) do
     table.insert(scales, v.weight:float())
  end
  for k,v in pairs(model:findModules('cudnn.SpatialBatchNormalization')) do
     table.insert(scales, v.weight:float())
  end

  table.insert(scale_epoch, scales)
end

function train()
  model:training()

  local confusion = optim.ConfusionMatrix(opt.num_classes)
  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all minibatches have equal size
  indices[#indices] = nil

  local loss = 0

  for t,v in ipairs(indices) do
    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))

    optim[opt.optimMethod](function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      local loss_Iter=f(inputs, targets)
      
      confusion:batchAdd(model.output, targets)
     if opt.weight_debug>=1 and (iteration % opt.step_WD ==0 ) then
         local Norm_gradP=torch.norm(gradParameters,1)/gradParameters:size(1)
         local Norm_P=torch.norm(parameters,1)/gradParameters:size(1)
         print(string.format("Iter: %6s,  N_GP=%3.12f, N_P=%3.12f, loss = %6.6f", iteration, Norm_gradP,Norm_P, loss_Iter))
        if opt.weight_debug==2 then 
          local Norm_input=torch.norm(inputs,1)/inputs:numel()
          print('Norm_input:'..Norm_input)


          local Norm_GP_table={}
          local Norm_P_table={}
          local Norm_GInput_table={}
          local Norm_Output_table={} 
         for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
          --for k,v in pairs(model:findModules('nn.SpatialMM_ForDebug')) do
            table.insert(Norm_GP_table, torch.norm(v.gradWeight,1)/v.gradWeight:numel())
            table.insert(Norm_P_table, torch.norm(v.weight,1)/v.weight:numel())
            table.insert(Norm_GInput_table, torch.norm(v.gradInput,1)/v.gradInput:numel())
            table.insert(Norm_Output_table, torch.norm(v.output,1)/v.output:numel()) 
          end

          for k,v in pairs(model:findModules('cudnn.SpatialConvolution')) do
         -- print(v.weight:size())
            table.insert(Norm_GP_table, torch.norm(v.gradWeight,1)/v.gradWeight:numel())
            table.insert(Norm_P_table, torch.norm(v.weight,1)/v.weight:numel())
            table.insert(Norm_GInput_table, torch.norm(v.gradInput,1)/v.gradInput:numel())
            table.insert(Norm_Output_table, torch.norm(v.output,1)/v.output:numel())

          end
          for k,v in pairs(model:findModules('cudnn.Spatial_Weight_CenteredBN')) do
         -- print(v.weight:size())
            table.insert(Norm_GP_table, torch.norm(v.gradWeight,1)/v.gradWeight:numel())
            table.insert(Norm_P_table, torch.norm(v.weight,1)/v.weight:numel())
            table.insert(Norm_GInput_table, torch.norm(v.gradInput,1)/v.gradInput:numel())
            table.insert(Norm_Output_table, torch.norm(v.output,1)/v.output:numel())
         end
          for k,v in pairs(model:findModules('cudnn.Spatial_Weight_CenteredBN_One')) do
         -- print(v.weight:size())
            table.insert(Norm_GP_table, torch.norm(v.gradWeight,1)/v.gradWeight:numel())
            table.insert(Norm_P_table, torch.norm(v.weight,1)/v.weight:numel())
            table.insert(Norm_GInput_table, torch.norm(v.gradInput,1)/v.gradInput:numel())
            table.insert(Norm_Output_table, torch.norm(v.output,1)/v.output:numel())
         end
          for k,v in pairs(model:findModules('cudnn.Spatial_Weight_BN')) do
         -- print(v.weight:size())
            table.insert(Norm_GP_table, torch.norm(v.gradWeight,1)/v.gradWeight:numel())
            table.insert(Norm_P_table, torch.norm(v.weight,1)/v.weight:numel())
            table.insert(Norm_GInput_table, torch.norm(v.gradInput,1)/v.gradInput:numel())
            table.insert(Norm_Output_table, torch.norm(v.output,1)/v.output:numel())
         end
          for k,v in pairs(model:findModules('cudnn.Spatial_Weight_DBN_Row')) do
         -- print(v.weight:size())
            table.insert(Norm_GP_table, torch.norm(v.gradWeight,1)/v.gradWeight:numel())
            table.insert(Norm_P_table, torch.norm(v.weight,1)/v.weight:numel())
            table.insert(Norm_GInput_table, torch.norm(v.gradInput,1)/v.gradInput:numel())
            table.insert(Norm_Output_table, torch.norm(v.output,1)/v.output:numel())
         end
          local Norm_GP_perModule=torch.FloatTensor(Norm_GP_table):reshape(1,table.getn(Norm_GP_table))
          local Norm_P_perModule=torch.FloatTensor(Norm_P_table):reshape(1,table.getn(Norm_P_table))
          local Norm_GInput_perModule=torch.FloatTensor(Norm_GInput_table):reshape(1,table.getn(Norm_GInput_table))
          local Norm_Output_perModule=torch.FloatTensor(Norm_Output_table):reshape(1,table.getn(Norm_Output_table))
               
          table.insert(Norm_GradWeight, Norm_GP_perModule:clone())
          table.insert(Norm_Weight, Norm_P_perModule:clone())
          table.insert(Norm_GradInput, Norm_GInput_perModule:clone())
          table.insert(Norm_Output, Norm_Output_perModule:clone())     
         print('Norm_GradWeight_perModule')
         print(Norm_GP_perModule)
         print('Norm_Weight_perModule')
         print(Norm_P_perModule)
          print('Norm_GradInput_perModule')
         print(Norm_GInput_perModule)
          print('Norm_Output_perModule')
          print(Norm_Output_perModule)
        local Norm_scale_table={} 
         for k,v in pairs(model:findModules('nn.Spatial_Scaling')) do
         -- print(v.weight:size())
            table.insert(Norm_scale_table, torch.norm(v.weight,1)/v.weight:numel())
         end
         local Norm_scale_perModule
         if table.getn(Norm_scale_table)>0 then
          Norm_scale_perModule=torch.FloatTensor(Norm_scale_table):reshape(1,table.getn(Norm_scale_table))
         print('Norm_scale_perModule')
         print(Norm_scale_perModule)
         end

       end         
    else
       print(string.format("Iter: %6s,  loss = %6.6f", iteration,loss_Iter))            
    end 

      losses[#losses+1]=loss_Iter
      loss = loss + loss_Iter
       iteration=iteration+1

      timeCosts=torch.toc(start_time)

    print(string.format("time Costs = %6.6f", timeCosts))

      if iteration ~=0 and iteration % opt.T==0 then

        for k,v in pairs(model:findModules('nn.Spatial_SVB')) do
          v:updateWeight(0.5)
         end
      end
      
    --  debug_scale('nn.SpatialBatchNormalization') 
    --  debug_scale('nn.Spatial_Scaling') 
    --  debug_scale('cudnn.SpatialBatchNormalization') 

      return f,gradParameters
    end, parameters, optimState)
  end
 confusion:updateValids()
  train_acc = confusion.totalValid * 100
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t '):format(train_acc))

  train_accus[#train_accus+1]=train_acc
  --confusion:zero()

  return loss / #indices
end


function test()
  model:evaluate()
  local confusion = optim.ConfusionMatrix(opt.num_classes)
  local data_split = provider.testData.data:split(opt.batchSize,1)
  local labels_split = provider.testData.labels:split(opt.batchSize,1)

  for i,v in ipairs(data_split) do
    confusion:batchAdd(model:forward(v), labels_split[i])
  end

  confusion:updateValids()

 if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    local base64im
    do
       os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/test.base64')
       if f then base64im = f:read'*all' end
     end
      local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  return confusion.totalValid * 100
end

iteration=0
losses={}
losses_epoch={}
train_accus={}
test_accus={}

train_times={}
test_times={}
scale_epoch={}
Norm_GradWeight={}
Norm_Weight={}
Norm_GradInput={}
Norm_Output={}
start_time=torch.tic()

results={}

for epoch=1,opt.max_epoch do
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  -- drop learning rate and reset momentum vector
  if torch.type(opt.epoch_step) == 'number' and epoch % opt.epoch_step == 0 or
     torch.type(opt.epoch_step) == 'table' and tablex.find(opt.epoch_step, epoch) then
    opt.learningRate = opt.learningRate * opt.learningRateDecayRatio
    optimState = tablex.deepcopy(opt)
  end

  local function t(f) local s = torch.Timer(); return f(), s:time().real end

  local loss, train_time = t(train)
  local test_acc, test_time = t(test)

  print(('Test accuracy: '..c.cyan'%.2f'..' %%\t '):format(test_acc))
  
  losses_epoch[#losses_epoch+1]=loss
  test_accus[#test_accus+1]=test_acc

  train_times[#train_times+1]=train_time
  print('train time:'..train_time)
  test_times[#test_times+1]=test_time
  print('test time:'..test_time)
  --debug_recordScale()


    for k,v in pairs(model:findModules('cudnn.Spatial_SVB')) do
       v:updateWeight(0.5)
    end

    for k,v in pairs(model:findModules('cudnn.Spatial_SVD_Group')) do
       -- print('updateWeight flag true')
       v:updateWeight(true)
    end
    local  batch_inputs=provider.trainData.data[{{1,8},{},{},{}}] --use a small batch data to forward， and udpate weight
    local  batch_outputs = model:forward(batch_inputs)

    for k,v in pairs(model:findModules('cudnn.Spatial_SVD_Group')) do
      --  print('updateWeight flag false')
        v:updateWeight(false)
    end

  log{
     loss = loss,
     epoch = epoch,
     test_acc = test_acc,
     lr = opt.learningRate,
     train_time = train_time,
     test_time = test_time,
   }






results.opt=opt
results.losses=losses
results.train_accus=train_accus
results.test_accus=test_accus
results.losses_epoch=losses_epoch
results.train_times=train_times
results.test_times=test_times
results.timeCosts=timeCosts
--results.scale_epoch=scale_epoch
--results.Norm_GradWeight=Norm_GradWeight
--results.Norm_Weight=Norm_Weight
--results.Norm_GradInput=Norm_GradInput
--results.Norm_Output=Norm_Output

torch.save('result_'..baseString..'.dat',results)
end
if opt.modelSave==1 then
-- Weight normalization method, get the weight matrix 

for k,v in pairs(model:findModules('cudnn.Spatial_Weight_CenteredBN')) do
v:endTraining()
end
for k,v in pairs(model:findModules('cudnn.Spatial_Weight_Center')) do
v:endTraining()
end
for k,v in pairs(model:findModules('cudnn.Spatial_Weight_DBN_Row')) do
v:endTraining()
end


torch.save('SModel_Cifar10_NoW_'..baseString..'.t7', net:clearState())
end
