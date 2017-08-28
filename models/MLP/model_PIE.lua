require 'nn'
require 'module/Affine_module'
require 'module/Linear_Weight_BN_Row'
require 'module/Linear_Weight_DBN_Row_Group'


function create_model(opt)
  ------------------------------------------------------------------------------

  ------------------------------------------------------------------------------

   

  local model=nn.Sequential()          
  local cfg_hidden=torch.Tensor({opt.n_hidden_number,opt.n_hidden_number,opt.n_hidden_number,opt.n_hidden_number,opt.n_hidden_number})
  local n=cfg_hidden:size(1)
  
  local nonlinear 
  if opt.mode_nonlinear==0 then  --sigmod
      nonlinear=nn.Sigmoid
  elseif opt.mode_nonlinear==1 then --tanh
      nonlinear=nn.Tanh
  elseif opt.mode_nonlinear==2 then --ReLU
     nonlinear=nn.ReLU
  elseif opt.mode_nonlinear==3 then --ReLU
     nonlinear=nn.ELU
  end 
  
  local linear=nn.Linear
  local module_BN=nn.BatchNormalization
  local module_affine=nn.Affine_module
 
  local function block_sgd(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(linear(n_input,n_output))
    return s
  end
 

  local function block_batch(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_BN(n_input)) 
    s:add(nonlinear())
    s:add(linear(n_input,n_output))
    return s
  end


 
  
  local function block_WN_Row(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(nn.Linear_Weight_BN_Row(n_input, n_output))
    return s
  end 

  local function block_WN_Row_scale(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(nn.Linear_Weight_BN_Row(n_input, n_output))
    s:add(module_affine(n_output,1,true))
    return s
  end
  
   local function block_WDBN_Row(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(nn.Linear_Weight_DBN_Row_Group(n_input, n_output,opt.m_perGroup))
    return s
  end 
 

   local function block_WDBN_Row_scale(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(nn.Linear_Weight_DBN_Row_Group(n_input, n_output,opt.m_perGroup))
    s:add(module_affine(n_output,1,true))
    return s
  end
 
  
  

  local function block_WN_Row_batch(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_BN(n_input))
    s:add(nonlinear())
    s:add(nn.Linear_Weight_BN_Row(n_input,n_output))
    return s
  end
 

  local function block_WN_Row_scale_batch(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_BN(n_input))
    s:add(nonlinear())
    s:add(nn.Linear_Weight_BN_Row(n_input,n_output))
    s:add(module_affine(n_output,1,true))
    return s
  end
 
 
 

  local function block_WDBN_Row_batch(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_BN(n_input))
    s:add(nonlinear())
    s:add(nn.Linear_Weight_DBN_Row_Group(n_input, n_output,opt.m_perGroup))
    return s
  end

 

  local function block_WDBN_Row_scale_batch(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_BN(n_input))
    s:add(nonlinear())
    s:add(nn.Linear_Weight_DBN_Row_Group(n_input, n_output,opt.m_perGroup))
    s:add(module_affine(n_output,1,true))
    return s
  end

 
 

-----------------------------------------model configure-------------------

  if opt.model=='sgd' then 
       model:add(linear(opt.n_inputs,cfg_hidden[1]))
    for i=1,n do
       if i==n then
         model:add(block_sgd(cfg_hidden[i],opt.n_outputs)) 
       else
        model:add(block_sgd(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end 
  elseif opt.model=='batch' then 
       model:add(linear(opt.n_inputs,cfg_hidden[1]))
    for i=1,n do
       if i==n then
         model:add(block_batch(cfg_hidden[i],opt.n_outputs)) 

       else
        model:add(block_batch(cfg_hidden[i],cfg_hidden[i+1])) 

       end
     end

 elseif opt.model=='WN_Row' then   
     model:add(nn.Linear_Weight_BN_Row(opt.n_inputs,cfg_hidden[1]))
     for i=1,n do
       if i==n then
         model:add(block_WN_Row(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_WN_Row(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end 
 elseif opt.model=='WN_Row_scale' then

     model:add(nn.Linear_Weight_BN_Row(opt.n_inputs,cfg_hidden[1]))
     model:add(module_affine(cfg_hidden[1],1,true))
     for i=1,n do
       if i==n then
         model:add(block_WN_Row_scale(cfg_hidden[i],opt.n_outputs))
       else
         model:add(block_WN_Row_scale(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end
   elseif opt.model=='WDBN_Row' then   

     model:add(nn.Linear_Weight_DBN_Row_Group(opt.n_inputs,cfg_hidden[1],opt.m_perGroup))
     for i=1,n do
       if i==n then
         model:add(block_WDBN_Row(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_WDBN_Row(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end 
   elseif opt.model=='WDBN_Row_scale' then

     model:add(nn.Linear_Weight_DBN_Row_Group(opt.n_inputs,cfg_hidden[1],opt.m_perGroup))
     model:add(module_affine(cfg_hidden[1],1,true))
     for i=1,n do
       if i==n then
         model:add(block_WDBN_Row_scale(cfg_hidden[i],opt.n_outputs))
       else
         model:add(block_WDBN_Row_scale(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end

      
   elseif opt.model=='WN_Row_batch' then

     model:add(nn.Linear_Weight_BN_Row(opt.n_inputs,cfg_hidden[1]))
     for i=1,n do
       if i==n then
         model:add(block_WN_Row_batch(cfg_hidden[i],opt.n_outputs))
       else
         model:add(block_WN_Row_batch(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end
 
   elseif opt.model=='WN_Row_scale_batch' then

     model:add(nn.Linear_Weight_BN_Row(opt.n_inputs,cfg_hidden[1]))
     model:add(module_affine(cfg_hidden[1],1,true))
     for i=1,n do
       if i==n then
         model:add(block_WN_Row_scale_batch(cfg_hidden[i],opt.n_outputs))
       else
         model:add(block_WN_Row_scale_batch(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end
 
   elseif opt.model=='WDBN_Row_batch' then

     model:add(nn.Linear_Weight_DBN_Row_Group(opt.n_inputs,cfg_hidden[1],opt.m_perGroup))
   --  model:add(module_affine(cfg_hidden[1],1,true))
     for i=1,n do
       if i==n then
         model:add(block_WDBN_Row_batch(cfg_hidden[i],opt.n_outputs))
       else
         model:add(block_WDBN_Row_batch(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end
 
   elseif opt.model=='WDBN_Row_scale_batch' then

     model:add(nn.Linear_Weight_DBN_Row_Group(opt.n_inputs,cfg_hidden[1],opt.m_perGroup))
     model:add(module_affine(cfg_hidden[1],1,true))
     for i=1,n do
       if i==n then
         model:add(block_WDBN_Row_scale_batch(cfg_hidden[i],opt.n_outputs))
       else
         model:add(block_WDBN_Row_scale_batch(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end
 
  

 
  end
  
  
  model:add(nn.LogSoftMax()) 
 

  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  local  criterion = nn.ClassNLLCriterion()

  return model, criterion
end

