local Linear_Weight_BN_Row, parent = torch.class('nn.Linear_Weight_BN_Row', 'nn.Module')

function Linear_Weight_BN_Row:__init(inputSize,outputSize,orth_flag,unitLength_flag)
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

    self:reset()
  
  

end



function Linear_Weight_BN_Row:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
        -- self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
     -- self.bias:uniform(-stdv, stdv)
   end

   return self
end


function Linear_Weight_BN_Row:reset_orthogonal()
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


function Linear_Weight_BN_Row:updateOutput(input)

  if input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      local n_output=self.weight:size(1)
      local n_input=self.weight:size(2)
      self.output:resize(nframe, n_output)
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      
      self.buffer = self.buffer or input.new()
      self.std=self.std or input.new()

      
      self.W=self.W or input.new()
      self.W:resizeAs(self.weight)

      if self.unitLength_flag then 
        self.std:resize(n_output,1):copy(self.weight:norm(2,2)):pow(-1)
      else     
        self.buffer:resizeAs(self.weight):copy(self.weight):cmul(self.weight)
       self.std:mean(self.buffer, 2):sqrt():pow(-1)
            
     end 
      
      
      self.W:repeatTensor(self.std,1,n_input)
      self.W:cmul(self.weight)

      self.output:addmm(0, self.output, 1, input, self.W:t())
     -- self.output:addr(1, self.addBuffer, self.bias)
   else
      error('input must be vector or matrix')
   end
   
   return self.output
end

function Linear_Weight_BN_Row:updateGradInput(input, gradOutput)
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

function Linear_Weight_BN_Row:accGradParameters(input, gradOutput, scale)
--      if self.flag_inner_lr then
 --       scale = self.scale or 1.0
 --     else
        scale =scale or 1.0
 --     end
   if input:dim() == 2 then
      local n_output=self.weight:size(1)
      local n_input=self.weight:size(2)
      self.gradW=self.gradW or input.new()


      self.gradW:resize(gradOutput:size(2),input:size(2))
      self.gradW:mm(gradOutput:t(), input)  --dL/dW

      
      self.gradWeight:cmul(self.W, self.gradW)
     if self.unitLength_flag then
         self.buffer:sum(self.gradWeight,2)   
      else 
         self.buffer:mean(self.gradWeight,2)
      end

      self.gradWeight:repeatTensor(self.buffer,1, n_input)
      self.gradWeight:cmul(self.W):mul(-1)
   

   
        self.gradWeight:add(self.gradW)
        
        self.buffer:repeatTensor(self.std,1,n_input)    
        self.gradWeight:cmul(self.buffer)

      
     -- self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
   else
      error('input must be vector or matrix')
   end
   

   

end

-- we do not need to accumulate parameters when sharing
Linear_Weight_BN_Row.sharedAccUpdateGradParameters = Linear_Weight_BN_Row.accUpdateGradParameters


function Linear_Weight_BN_Row:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
