local Linear_Weight_DBN_Row_Group, parent = torch.class('nn.Linear_Weight_DBN_Row_Group', 'nn.Module')

function Linear_Weight_DBN_Row_Group:__init(inputSize,outputSize,m_perGroup,unitLength_flag)
   parent.__init(self)

   self.weight = torch.Tensor( outputSize,inputSize) --make sure the weight is symetric and semi-definite
  -- self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
  -- self.gradBias = torch.Tensor(outputSize)
     ----if use unitLength, means the F_norm(W_i)=1 (like WeightNormalization style),
      -- if not, means the var(W_i)=1(like batchNormalization), The difference is 1/n
   
  if unitLength_flag ~= nil then
      assert(type(unitLength_flag) == 'boolean', 'unitLength_flag has to be true/false')
      self.unitLength_flag = unitLength_flag
   else
      self.unitLength_flag = true
   end 

  self.eps=1e-7
  self.threshold=0
  self.debug=false

   if m_perGroup~=nil then
      self.m_perGroup = m_perGroup > inputSize and inputSize  or m_perGroup 
   else
     self.m_perGroup =  inputSize
   end 
  self:reset() 
end

function Linear_Weight_DBN_Row_Group:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
      self.weight:uniform(-stdv, stdv)
        --    self.weight:randn(self.weight:size(1),self.weight:size(2))
     -- self.bias:uniform(-stdv, stdv)
   return self
end


function Linear_Weight_DBN_Row_Group:reset_orthogonal()
    local initScale = 1.1 -- math.sqrt(2)

    local M1 = torch.randn(self.weight:size(1), self.weight:size(1))
    local M2 = torch.randn(self.weight:size(2), self.weight:size(2))

    local n_min = math.min(self.weight:size(1), self.weight:size(2))

    -- QR decomposition of random matrices ~ N(0, 1)
    local Q1, R1 = torch.qr(M1)
    local Q2, R2 = torch.qr(M2)

    self.weight:copy(Q1:narrow(2,1,n_min) * Q2:narrow(1,1,n_min)):mul(initScale)

   -- self.bias:zero()
end


function Linear_Weight_DBN_Row_Group:updateOutput(input)

   assert(input:dim() == 2, 'only mini-batch supported (2D tensor), got '
             .. input:dim() .. 'D tensor instead')

   function updateOutput_perGroup(weight_perGroup,groupId)

      local n_output=weight_perGroup:size(1)
      local n_input=weight_perGroup:size(2)

      
      local n_output=weight_perGroup:size(1)
      local n_input=weight_perGroup:size(2)
     local scale=weight_perGroup.new()
      

      local centered = weight_perGroup.new()
            
      self.W_perGroup=self.W_perGroup or input.new()
      self.W_perGroup:resizeAs(weight_perGroup)

      self.buffer:mean(weight_perGroup, 2) 
      
     -- self.buffer:fill(0)  --for test, use no centered DBN
      self.buffer_2:repeatTensor(self.buffer, 1, n_input)   
      centered:add(weight_perGroup, -1, self.buffer_2)
      
       ----------------------calcualte the projection matrix----------------------
      self.buffer_1:resize(weight_perGroup:size(1),weight_perGroup:size(1))
      
      self.buffer_1:addmm(0,self.buffer_1,1/n_input,centered,centered:t()) --buffer_1 record correlation matrix
        self.buffer_1:add(self.eps,torch.eye(self.buffer_1:size(1)))
           -----------------------matrix decomposition------------- 
    
      local rotation,eig,_=torch.svd(self.buffer_1) --reuse the buffer: 'buffer' record e, 'buffer_2' record V    

      
      if self.debug then
          print(eig)
      end 
      
      scale:resizeAs(eig)     
      scale:copy(eig)
      scale:pow(-1/2) --scale=eig^(-1/2)
      self.buffer_1:diag(scale)   --self.buffer_1 cache the scale matrix  
      self.buffer_2:resizeAs(rotation) 
      self.buffer_2:mm(self.buffer_1,rotation:t()) --U= Eighta^(-1/2)*D^T
      self.buffer_1:mm(rotation,self.buffer_2)     
     
      self.W_perGroup:mm(self.buffer_1,centered)
           
      
      
      
      if self.unitLength_flag then 
        self.W_perGroup:mul(math.sqrt(1/n_input))
      end
  
         ----------------record the results of per groupt--------------
      table.insert(self.eigs, eig)
      table.insert(self.scales, scale)
      table.insert(self.rotations, rotation)
      table.insert(self.centereds, centered)
  
      return self.W_perGroup
  end      
    
 
    
    
      
--------------------------------update main function---------------------- 

  local nframe = input:size(1)
   local nElement = self.output:nElement()
   local n_output=self.weight:size(1)
   local n_input=self.weight:size(2)  
   self.output:resize(nframe, n_output)
   if self.output:nElement() ~= nElement then
       self.output:zero()
   end
   
   self.W=self.W or input.new()
   self.W:resizeAs(self.weight)  
     -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.buffer_1 = self.buffer_1 or input.new()
   self.buffer_2 = self.buffer_2 or input.new()   
     
   local groups=torch.floor((n_output-1)/self.m_perGroup)+1  
     
    -------------- initalize the group parameters---------------
      self.eigs={}
      self.scales={}
      self.rotations={}
      self.centereds={}

      for i=1,groups do 
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,n_output)      
         self.W[{{start_index,end_index},{}}]=updateOutput_perGroup(self.weight[{{start_index,end_index},{}}],i)   
      end    
      
   if input:dim() == 2 then
      self.output:addmm(0, self.output, 1, input, self.W:t())
     -- self.output:addr(1, self.addBuffer, self.bias)
   else
      error('input must be vector or matrix')
   end
   
   return self.output
end

function Linear_Weight_DBN_Row_Group:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      
     if input:dim() == 2 then
         
         self.gradInput:addmm(0, 1, gradOutput, self.W)    
         
     else
      error('input must be vector or matrix')
     end
      
      return self.gradInput
   end
end

function Linear_Weight_DBN_Row_Group:accGradParameters(input, gradOutput, scale)
 
 ------------------calculate the K matrix---------------------
  function getK(scale)
    local K=torch.Tensor(scale:size(1),scale:size(1)):fill(0)
    local revise=0    --1e-100
    for i=1,scale:size(1) do
      for j=1,scale:size(1) do
        if (i~=j) and torch.abs(scale[i]-scale[j])> self.threshold then
          K[i][j]=1/(scale[i]-scale[j]+revise)
        end
      
      end   
    end  
  
    return K  
  end

  function getK_new(eig)

    local revised=1e-45 --used for div 0, in case of that there are tow eigenValuse is the same (It's almost impossible)
    local K=torch.Tensor(eig:size(1),eig:size(1)):fill(revised)
    local b_1=torch.Tensor(eig:size(1),eig:size(1)):repeatTensor(eig, eig:size(1), 1)
    local b_2=torch.eye(eig:size(1)):add(b_1:t()):add(-1,b_1):add(K)
    K:fill(1):cdiv(b_2):add(-1, torch.eye(eig:size(1))*(1+revised))
    return K
  end
 
 ---------------------------------------------------------
 
function updateAccGradParameters_perGroup(gradW_perGroup, groupId)

       local n_output=gradW_perGroup:size(1)
      local n_input=gradW_perGroup:size(2)
     local  eig=self.eigs[groupId]
     local  scale=self.scales[groupId]
     local  rotation=self.rotations[groupId]
     local  centered=self.centereds[groupId]

     self.gradWeight_perGroup=self.gradWeight_perGroup or gradW_perGroup.new()
     self.gradWeight_perGroup:resizeAs(gradW_perGroup)
     self.hat_x=self.hat_x or gradW_perGroup.new()
     self.S=self.S or gradW_perGroup.new()
     self.M=self.M or gradW_perGroup.new()
     self.U=self.U or gradW_perGroup.new()
     self.f=self.f or gradW_perGroup.new()
     self.FC=self.FC or gradW_perGroup.new()
     self.d_hat_x=self.d_hat_x or gradW_perGroup.new() 
       
     self.hat_x:resizeAs(centered)
     self.d_hat_x:resizeAs(centered)    
     self.U:resizeAs(rotation)
     self.M:resizeAs(rotation)
     self.S:resizeAs(rotation)
     self.FC:resizeAs(rotation)
     
     
     self.buffer:diag(scale)
     self.U:mm( self.buffer,rotation:t())
     self.hat_x:mm( self.U,centered)
     self.d_hat_x:mm(rotation:t(),gradW_perGroup)  
        
     self.FC:addmm(0, self.FC, 1/n_input, self.d_hat_x, self.hat_x:t())
     self.f:mean(self.d_hat_x, 2)
     

       local sz = (#self.FC)[1]    
     local temp_diag=torch.diag(self.FC) --get the diag element of d_EighTa
    self.M:diag(temp_diag) --matrix form
 --------------------------------calculate S-----------------------------    
   --  self.buffer:diag(self.eig)
    -- self.S:mm(self.buffer, self.FC:t())
     self.S:cmul(self.FC:t(), torch.repeatTensor(eig, sz, 1):t()) 
     self.buffer:resizeAs(eig):copy(eig):pow(1/2)   
     self.buffer_1:cmul(self.FC, torch.repeatTensor(self.buffer, sz, 1):t())
     self.buffer_2:cmul(self.buffer_1, torch.repeatTensor(self.buffer, sz, 1))     
     self.S:add(self.buffer_2)
     
     self.buffer=getK_new(eig)
     

     
     self.S:cmul(self.buffer:t())
     self.buffer:copy(self.S)
     self.S:add(self.buffer, self.buffer:t())
     
     self.S:add(-1, self.M)  -- S-M
     self.buffer_1:resizeAs(self.d_hat_x)
     self.buffer_1:mm( self.S:t(),self.hat_x) --(S-M)*self.hat_x
     
     self.buffer:repeatTensor(self.f, 1, n_input)
     self.buffer_1:add(self.d_hat_x):add(-1, self.buffer)
   
    
   -- self.S:resizeAs(self.buffer_1):mm(self.buffer_1, self.M:diag(scale)) -- for debug
    self.gradWeight_perGroup:mm( self.U:t(),self.buffer_1)   
 
       
      if self.unitLength_flag then 
        self.gradWeight_perGroup:mul(math.sqrt(1/n_input))
      end
      
      return self.gradWeight_perGroup
  end     
 
-------------------------------------main function of accGrad------------ 
    assert(input:dim() == 2, 'only mini-batch supported')
   assert(gradOutput:dim() == 2, 'only mini-batch supported') 
   local n_output=self.weight:size(1)   
   self.gradW=self.gradW or input.new()


    self.gradW:resize(gradOutput:size(2),input:size(2))
    self.gradW:mm(gradOutput:t(), input)  --dL/dW  
    
   local groups=torch.floor((n_output-1)/self.m_perGroup)+1
   
   for i=1,groups do 
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,n_output)
         self.gradWeight[{{start_index,end_index},{}}]=updateAccGradParameters_perGroup(self.gradW[{{start_index,end_index},{}}],i)   
    end

end

Linear_Weight_DBN_Row_Group.sharedAccUpdateGradParameters = Linear_Weight_DBN_Row_Group.accUpdateGradParameters


function Linear_Weight_DBN_Row_Group:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
