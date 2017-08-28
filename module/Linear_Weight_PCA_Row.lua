local Linear_Weight_PCA_Row, parent = torch.class('nn.Linear_Weight_PCA_Row', 'nn.Module')

function Linear_Weight_PCA_Row:__init(inputSize,outputSize,orth_flag,unitLength_flag)
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

  if orth_flag ~= nil then
    assert(type(orth_flag) == 'boolean', 'orth_flag has to be true/false')    
    if orth_flag then
      self:reset_orthogonal()
    else
    self:reset()
    end
    
  else
    self:reset() 
  end

end



function Linear_Weight_PCA_Row:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end


      self.weight:uniform(-stdv, stdv)
      -- self.weight:randn(self.weight:size(1),self.weight:size(2))
     -- self.bias:uniform(-stdv, stdv)
 
     
   return self
end


function Linear_Weight_PCA_Row:reset_orthogonal()
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


function Linear_Weight_PCA_Row:updateOutput(input)


      local nframe = input:size(1)
      local nElement = self.output:nElement()
      local n_output=self.weight:size(1)
      local n_input=self.weight:size(2)
      self.output:resize(nframe, n_output)
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      
      self.buffer = self.buffer or input.new()
      self.buffer_1 = self.buffer_1 or input.new()
      self.buffer_2 = self.buffer_2 or input.new()
      self.centered = self.centered or input.new()
      self.scale=self.scale or input.new()
      self.eig=self.eig or input.new()
      self.rotation = self.rotation or input.new()
            
      self.W=self.W or input.new()
      self.W:resizeAs(self.weight)

      self.buffer:mean(self.weight, 2) 
      
     -- self.buffer:fill(0)  --for test, use no centered DBN
      self.buffer_2:repeatTensor(self.buffer, 1, n_input)   
      self.centered:add(self.weight, -1, self.buffer_2)
      
       ----------------------calcualte the projection matrix----------------------
      self.buffer_1:resize(self.weight:size(1),self.weight:size(1))
      
      self.buffer_1:addmm(0,self.buffer_1,1/n_input,self.centered,self.centered:t()) --buffer_1 record correlation matrix
        self.buffer_1:add(self.eps,torch.eye(self.buffer_1:size(1)))
           -----------------------matrix decomposition------------- 
    
      self.rotation,self.eig,_=torch.svd(self.buffer_1) --reuse the buffer: 'buffer' record e, 'buffer_2' record V    
   
    --  self.eig:add(self.eps)
      
      if self.debug then
          print(self.eig)
      end 
     
      
      self.scale:resizeAs(self.eig)     
      self.scale:copy(self.eig)
      self.scale:pow(-1/2) --scale=eig^(-1/2)
      self.buffer_1:diag(self.scale)   --self.buffer_1 cache the scale matrix  
      self.buffer_2:resizeAs(self.rotation) 
      self.buffer_2:mm(self.buffer_1,self.rotation:t()) --U= Eighta^(-1/2)*D^T
     
     
      self.W:mm(self.buffer_2,self.centered)
           
      
      
      
      if self.unitLength_flag then 
        self.W:mul(math.sqrt(1/n_input))
      end
      
      
      
      
   if input:dim() == 2 then
      self.output:addmm(0, self.output, 1, input, self.W:t())
     -- self.output:addr(1, self.addBuffer, self.bias)
   else
      error('input must be vector or matrix')
   end
   
   return self.output
end

function Linear_Weight_PCA_Row:updateGradInput(input, gradOutput)
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

function Linear_Weight_PCA_Row:accGradParameters(input, gradOutput, scale)
 
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
 


      local n_output=self.weight:size(1)
      local n_input=self.weight:size(2)
      self.gradW=self.gradW or input.new()


      self.gradW:resize(gradOutput:size(2),input:size(2))
      self.gradW:mm(gradOutput:t(), input)  --dL/dW

 
     self.hat_x=self.hat_x or input.new()
     self.S=self.S or input.new()
     self.M=self.M or input.new()
     self.U=self.U or input.new()
     self.f=self.f or input.new()
     self.FC=self.FC or input.new()
     
     self.hat_x:resizeAs(self.centered)
     self.U:resizeAs(self.rotation)
     self.M:resizeAs(self.rotation)
     self.S:resizeAs(self.rotation)
     self.FC:resizeAs(self.rotation)
     
     
     self.buffer:diag(self.scale)
     self.U:mm( self.buffer,self.rotation:t())
     self.hat_x:mm( self.U,self.centered)
     
     self.FC:addmm(0, self.FC, 1/n_input, self.gradW, self.hat_x:t())
     self.f:mean(self.gradW, 2)
     

     
     local temp_diag=torch.diag(self.FC) --get the diag element of d_EighTa
    self.M:diag(temp_diag) --matrix form
 --------------------------------calculate S-----------------------------    
     self.buffer:diag(self.eig)
     self.S:mm(self.buffer, self.FC:t())
     
     self.buffer=getK_new(self.eig)
     
     if self.debug_detailInfo then
       print('----------K Matrix----------')
       print(self.buffer)
     end
     
     self.S:cmul(self.buffer:t())
     self.buffer:copy(self.S)
     self.S:add(self.buffer, self.buffer:t())
     
     self.S:add(-1, self.M)  -- S-M
     self.buffer_1:resizeAs(self.gradW)
     self.buffer_1:mm( self.S:t(),self.hat_x) --(S-M)*self.hat_x
     
     self.buffer:repeatTensor(self.f, 1, n_input)
     self.buffer_1:add(self.gradW):add(-1, self.buffer)
   
    
   -- self.S:resizeAs(self.buffer_1):mm(self.buffer_1, self.M:diag(self.scale)) -- for debug
    self.gradWeight:mm( self.U:t(),self.buffer_1)   
 
       
      if self.unitLength_flag then 
        self.gradWeight:mul(math.sqrt(1/n_input))
      end
      
 
     -- self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)

end

-- we do not need to accumulate parameters when sharing
Linear_Weight_PCA_Row.sharedAccUpdateGradParameters = Linear_Weight_PCA_Row.accUpdateGradParameters


function Linear_Weight_PCA_Row:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
