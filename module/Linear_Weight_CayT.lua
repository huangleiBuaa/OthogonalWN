local Linear_Weight_CayT, parent = torch.class('nn.Linear_Weight_CayT', 'nn.Module')

function Linear_Weight_CayT:__init(inputSize, outputSize,lr, orth_flag,isBias)
   parent.__init(self)
  
   if isBias ~= nil then
      assert(type(isBias) == 'boolean', 'isBias has to be true/false')
      self.isBias = isBias
   else
      self.isBias = true
   end

   if lr ~= nil then
      self.lr = lr
   else
      self.lr = 0.1
   end

   self.PWeight = torch.Tensor(outputSize, inputSize)
  if self.isBias then
   self.bias = torch.Tensor(outputSize)
   self.gradBias = torch.Tensor(outputSize)
  end
   self.gradPWeight = torch.Tensor(outputSize, inputSize)
   
   self.W=torch.Tensor()
   self.gradW=torch.Tensor()

   self.trans_flag=false 
       if outputSize<inputSize then
           self.trans_flag=true
           self.W:resizeAs(self.PWeight:t())
           self.gradW:resizeAs(self.gradPWeight:t())
       else

           self.W:resizeAs(self.PWeight)
           self.gradW:resizeAs(self.gradPWeight)
       end
   print('--------you should trans:-----') 
   print(self.trans_flag) 
   
   print('--------learning rate:-----:'..self.lr) 
   
   self.printDetail=false
   self.debug=false
   self.debug_detailInfo=false
   self.printInterval=1
   self.count=0
   
   
  if orth_flag ~= nil then
      assert(type(orth_flag) == 'boolean', 'orth_flag has to be true/false')
      
      
    if orth_flag then
      self:reset_orthogonal()
    else
    self:reset()
    end
  else
    --self:reset()
      self:reset_orthogonal()
  
  end
  


end

function Linear_Weight_CayT:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.PWeight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.PWeight:size(1) do
         self.PWeight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
       
        if self.isBias then 
         self.bias[i] = torch.uniform(-stdv, stdv)
        end
     end
   else
      self.PWeight:uniform(-stdv, stdv)
     
     if self.isBias then

       self.bias:uniform(-stdv, stdv)
     end
    end

   return self
end


function Linear_Weight_CayT:reset_orthogonal()
    local initScale = 1 -- math.sqrt(2)
   -- local initScale =  math.sqrt(2)
   print('----------------orthogonal initionization---------------------') 
   local M1 = torch.randn(self.PWeight:size(1), self.PWeight:size(1))
    local M2 = torch.randn(self.PWeight:size(2), self.PWeight:size(2))

    local n_min = math.min(self.PWeight:size(1), self.PWeight:size(2))

    -- QR decomposition of random matrices ~ N(0, 1)
    local Q1, R1 = torch.qr(M1)
    local Q2, R2 = torch.qr(M2)

    self.PWeight:copy(Q1:narrow(2,1,n_min) * Q2:narrow(1,1,n_min)):mul(initScale)

    self.bias:zero()
end

function Linear_Weight_CayT:updateOutput(input)
   --self.bias:fill(0)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.PWeight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.PWeight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.PWeight:t())
     

      if self.isBias then
        self.output:addr(1, self.addBuffer, self.bias)
      end

   else
      error('input must be vector or matrix')
   end
   
   return self.output
end

function Linear_Weight_CayT:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.PWeight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.PWeight)
      end
      
      
      return self.gradInput
   end
end

function Linear_Weight_CayT:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradPWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
  
       local n_out=self.gradPWeight:size(1)
       local n_in=self.gradPWeight:size(2)

       self.gradPWeight:addmm(scale, gradOutput:t(), input)
       -----------make sure, n_out >n_in for W
       if self.trans_flag then
           self.gradW:copy(self.gradPWeight:t())               
           self.W:copy(self.PWeight:t())               

        else

           self.gradW:copy(self.gradPWeight)               
           self.W:copy(self.PWeight)               
        end
---------------------update W on ST manifold by Cayley Transform---------
     local n=self.W:size(1)
     local p=self.W:size(2)
     self.buffer=self.buffer or input.new()
     self.buffer1=self.buffer1 or input.new()
     self.buffer2=self.buffer2 or input.new()
     self.A=self.A or input.new()
 --    print(self.W)
  --   print(self.W:t()*self.W)
     
     
    self.buffer2:resize(n,n):mm(self.W,self.gradW:t()) 
     self.A:resize(n,n):mm(self.gradW,self.W:t()):add(-self.buffer2) --A=GX^T-XG^T 
     self.buffer1:resize(n,n):copy(self.A):mul(-self.lr/2):add(torch.eye(n))
     self.buffer:resize(n,p):mm(self.buffer1,self.W)
     self.buffer2:resize(n,n):copy(self.A):mul(self.lr/2):add(torch.eye(n))
     self.buffer1=torch.inverse(self.buffer2) 
     self.W:mm(self.buffer1,self.buffer) 
     -- print('---------------------self.buffer-------') 
   --  print(self.buffer1)
   --  print(self.buffer2)
   --  print(self.buffer)

    --------------trans back to PW---------------
    if self.trans_flag then
       self.PWeight:copy(self.W:t())
    else
       self.PWeight:copy(self.W)

     end

   if self.isBias then
        self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end
   
   self.count=self.count+1 --the ending of all the operation in this module
end

-- we do not need to accumulate parameters when sharing
Linear_Weight_CayT.sharedAccUpdateGradParameters = Linear_Weight_CayT.accUpdateGradParameters

function Linear_Weight_CayT:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.PWeight:size(2), self.PWeight:size(1))
end
