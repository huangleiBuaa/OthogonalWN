

require 'module/Linear_Weight_QR'
require 'module/Linear_Weight_ST'
require 'module/Linear_Weight_ST_CN'
require 'module/Linear_Weight_CayT'
require 'module/Linear_Weight_PCA_Row'
require 'module/Linear_Weight_DBN_Row_Group'


function create_model(opt)
  ------------------------------------------------------------------------------

  ------------------------------------------------------------------------------

   

  local model=nn.Sequential()          
  --local cfg_hidden=torch.Tensor({128,128,128,128,128})

  config_table={}
  for i=1, opt.layer do
     table.insert(config_table, opt.n_hidden_number)
  end
  local cfg_hidden=torch.Tensor(config_table)
 -- local cfg_hidden=torch.Tensor({opt.n_hidden_number,opt.n_hidden_number,opt.n_hidden_number,opt.n_hidden_number,opt.n_hidden_number})
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
  local module_Projection_ZCA=nn.Linear_Weight_Projection_ZCA
  local module_PD_ZCA=nn.Linear_Weight_PD_ZCA
  local module_PD_PCA=nn.Linear_Weight_PD_PCA
  local module_QR=nn.Linear_Weight_QR
  local module_ST=nn.Linear_Weight_ST
  local module_ST_CN=nn.Linear_Weight_ST_CN
  local module_CayT=nn.Linear_Weight_CayT
 
  local function block_sgd(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end
 

  local function block_QR(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(module_QR(n_input,n_output,opt.learningRate))
    return s
  end
  local function block_ST(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(module_ST(n_input,n_output,opt.learningRate))
    return s
  end


  local function block_ST_CN(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(module_ST_CN(n_input,n_output,opt.learningRate))
    return s
  end

  local function block_CayT(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(module_CayT(n_input,n_output,opt.learningRate))
    return s
  end

  local function block_WPCA_Row(n_input, n_output)
      local s=nn.Sequential()
       s:add(nonlinear())
       s:add(nn.Linear_Weight_PCA_Row(n_input, n_output,opt.orth_intial))
       return s
   end

  
  local function block_WDBN_Row(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(nn.Linear_Weight_DBN_Row_Group(n_input, n_output,opt.m_perGroup_WDBN))

    return s
  end 
  
  
  local function block_WDBN_Row_scale(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(nn.Linear_Weight_DBN_Row_Group(n_input, n_output,opt.m_perGroup_WDBN))
    s:add(module_affine(n_output,opt.BNScale,true))

    return s
  end 
  
 


-----------------------------------------model configure-------------------

  if opt.model=='sgd' then 
       model:add(linear(opt.n_inputs,cfg_hidden[1],opt.orth_intial))
    for i=1,n do
       if i==n then
         model:add(block_sgd(cfg_hidden[i],opt.n_outputs)) 

       else
        model:add(block_sgd(cfg_hidden[i],cfg_hidden[i+1])) 

       end
     end 
  elseif opt.model=='QR' then
   model:add(nn.Linear_Weight_QR(opt.n_inputs,cfg_hidden[1],opt.learningRate))
     for i=1,n do
       if i==n then
         --model:add(block_QR(cfg_hidden[i],opt.n_outputs))
         model:add(block_sgd(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_QR(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end
  elseif opt.model=='ST' then
   model:add(nn.Linear_Weight_ST(opt.n_inputs,cfg_hidden[1],opt.learningRate))
     for i=1,n do
       if i==n then
         --model:add(block_ST(cfg_hidden[i],opt.n_outputs))
         model:add(block_sgd(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_ST(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end
  elseif opt.model=='ST_CN' then
   model:add(nn.Linear_Weight_ST_CN(opt.n_inputs,cfg_hidden[1],opt.learningRate))
     for i=1,n do
       if i==n then
         --model:add(block_ST_CN(cfg_hidden[i],opt.n_outputs))
         model:add(block_sgd(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_ST_CN(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end
  elseif opt.model=='CayT' then
   model:add(nn.Linear_Weight_CayT(opt.n_inputs,cfg_hidden[1],opt.learningRate))
     for i=1,n do
       if i==n then
         --model:add(block_ST(cfg_hidden[i],opt.n_outputs))
         model:add(block_sgd(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_ST(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end
elseif opt.model=='WPCA_Row' then
   model:add(nn.Linear_Weight_PCA_Row(opt.n_inputs,cfg_hidden[1],opt.orth_intial))
   for i=1,n do
    if i==n then
      --model:add(block_WPCA_Row(cfg_hidden[i],opt.n_outputs))
         model:add(block_sgd(cfg_hidden[i],opt.n_outputs)) 
    else
       model:add(block_WPCA_Row(cfg_hidden[i],cfg_hidden[i+1]))
     end
    end

  
   
    elseif opt.model=='WDBN_Row' then   

     model:add(nn.Linear_Weight_DBN_Row_Group(opt.n_inputs,cfg_hidden[1],opt.m_perGroup_WDBN))
     for i=1,n do
       if i==n then
         --model:add(block_WDBN_Row(cfg_hidden[i],opt.n_outputs)) 
         model:add(block_sgd(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_WDBN_Row(cfg_hidden[i],cfg_hidden[i+1])) 
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

