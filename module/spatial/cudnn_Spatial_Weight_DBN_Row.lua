--[[
 --This file implements orthognoal linear module, which wraps orthogonal weight normalization
  --into the SpationConvolution of cudnn.
    --
  --The code is based on the orignial Torch implemantation of SpationConvolution for cudnn.
 -- -------------------------------------------------------------------
    --Author: Lei Huang
    --mail: huanglei@nlsde.buaa.edu.cn
]]--


local Spatial_Weight_DBN_Row, parent =
    torch.class('cudnn.Spatial_Weight_DBN_Row', 'nn.SpatialConvolution')
local ffi = require 'ffi'
local find = require 'cudnn.find'
local errcheck = cudnn.errcheck
local checkedCall = find.checkedCall

function Spatial_Weight_DBN_Row:__init(nInputPlane, nOutputPlane,m_perGroup,
                            kW, kH, dW, dH, padW, padH)
    local delayedReset = self.reset
    self.reset = function() end
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
    self.reset = delayedReset
    self.padW = padW or 0
    self.padH = padH or 0
    self.groups = 1
    assert(nInputPlane % self.groups == 0,
           'nInputPlane should be divisible by nGroups')
    assert(nOutputPlane % self.groups == 0,
           'nOutputPlane should be divisible by nGroups')
    self.weight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
   
    self.W = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self.gradW = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
  
   self.inputDim=nInputPlane*self.kH*self.kW

   if m_perGroup~=nil then
      self.m_perGroup = m_perGroup > self.inputDim and self.inputDim  or m_perGroup
   else
     self.m_perGroup =   nOutputPlane > self.inputDim and self.inputDim or nOutputPlane
   end
   print("m_perGroup:"..self.m_perGroup)


  if unitLength_flag ~= nil then
      assert(type(unitLength_flag) == 'boolean', 'unitLength_flag has to be true/false')
      self.unitLength_flag = unitLength_flag
   else
      self.unitLength_flag = true
   end


   self.eps=1e-8
    self.groups_WDBN=torch.floor((nOutputPlane-1)/self.m_perGroup)+1

    local length = self.m_perGroup
    self.eye_ngroup = torch.eye(length):cuda()
    self.initial_K = torch.CudaTensor(length, length):fill(0)

    length = nOutputPlane - (self.groups_WDBN - 1) * self.m_perGroup
    self.eye_ngroup_last = torch.eye(length):cuda()
    self.initial_K_last = torch.CudaTensor(length, length):fill(0)



    self:reset()
    -- should nil for serialization, the reset will still work
    self.reset = nil
    self.isTraining=true

end

-- if you change the configuration of the module manually, call this
function Spatial_Weight_DBN_Row:resetWeightDescriptors(desc)
    -- for compatibility
    self.groups = self.groups or 1
    assert(cudnn.typemap[torch.typename(self.W)], 'Only Cuda supported duh!')
    assert(cudnn.typemap[torch.typename(self.bias)] or not self.bias, 'Only Cuda supported duh!')

    -- create descriptor for bias
    if self.bias then
        self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1))
    end

    self.WDesc = cudnn.setFilterDescriptor(
       { dataType = cudnn.typemap[torch.typename(self.W)],
         filterDimA = desc or
            {self.nOutputPlane/self.groups,
             self.nInputPlane/self.groups,
             self.kH, self.kW}
       }
    )

    return self
end

function Spatial_Weight_DBN_Row:fastest(mode)
    if mode == nil then mode = true end
    if not self.fastest_mode or self.fastest_mode ~= mode then
       self.fastest_mode = mode
       self.iDesc = nil
    end
    return self
end

function Spatial_Weight_DBN_Row:setMode(fmode, bdmode, bwmode)
    if fmode ~= nil then
        self.fmode = fmode
    end
    if bdmode ~= nil then
        self.bdmode = bdmode
    end
    if bwmode ~= nil then
        self.bwmode = bwmode
    end
    self.iDesc = nil
    return self
end

function Spatial_Weight_DBN_Row:resetMode()
    self.fmode = nil
    self.bdmode = nil
    self.bwmode = nil
    return self
end

function Spatial_Weight_DBN_Row:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end


function Spatial_Weight_DBN_Row:checkInputChanged(input)
    assert(input:isContiguous(),
           "input to " .. torch.type(self) .. " needs to be contiguous, but is non-contiguous")
    if not self.iSize or self.iSize:size() ~= input:dim() then
       self.iSize = torch.LongStorage(input:dim()):fill(0)
    end
    self.groups = self.groups or 1
    if not self.WDesc then self:resetWeightDescriptors() end
    if not self.WDesc then error "Weights not assigned!" end

    if not self.iDesc or not self.oDesc or input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
    or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] or (input:dim()==5 and input:size(5) ~= self.iSize[5]) then
       self.iSize = input:size()
       assert(self.nInputPlane == input:size(2),
              'input has to contain: '
                 .. self.nInputPlane
                 .. ' feature maps, but received input of size: '
                 .. input:size(1) .. ' x ' .. input:size(2) .. ' x ' .. input:size(3)
                 .. (input:dim()>3 and ' x ' .. input:size(4) ..
                        (input:dim()==5 and ' x ' .. input:size(5) or '') or ''))
       return true
    end
    return false
end

function Spatial_Weight_DBN_Row:createIODescriptors(input)
   local batch = true
   if input:dim() == 3 then
      input = input:view(1, input:size(1), input:size(2), input:size(3))
      batch = false
   end
   if Spatial_Weight_DBN_Row.checkInputChanged(self, input) then
        -- create input descriptor
        local input_slice = input:narrow(2,1,self.nInputPlane/self.groups)
        self.iDesc = cudnn.toDescriptor(input_slice)
        -- create conv descriptor
        self.padH, self.padW = self.padH or 0, self.padW or 0
        -- those needed to calculate hash
        self.pad = {self.padH, self.padW}
        self.stride = {self.dH, self.dW}

        self.convDescData = { padA = self.pad,
             filterStrideA = self.stride,
             upscaleA = {1,1},
             dataType = cudnn.configmap(torch.type(self.W))
        }

        self.convDesc = cudnn.setConvolutionDescriptor(self.convDescData)

        -- get output shape, resize output
        local oSize = torch.IntTensor(4)
        errcheck('cudnnGetConvolutionNdForwardOutputDim',
                 self.convDesc[0], self.iDesc[0],
                 self.WDesc[0], 4, oSize:data())
        oSize[2] = oSize[2] * self.groups
        self.output:resize(oSize:long():storage())
        self.oSize = self.output:size()

        local output_slice = self.output:narrow(2,1,self.nOutputPlane/self.groups)
        -- create descriptor for output
        self.oDesc = cudnn.toDescriptor(output_slice)
        self.oDescForBias = cudnn.toDescriptor(self.output)

        find:prepare(self, input_slice, output_slice)

        -- create offsets for groups
        local iH, iW = input:size(3), input:size(4)
        local kH, kW = self.kH, self.kW
        local oH, oW = oSize[3], oSize[4]
        self.input_offset = self.nInputPlane / self.groups * iH * iW
        self.output_offset = self.nOutputPlane / self.groups * oH * oW
        self.W_offset = self.nInputPlane / self.groups * self.nOutputPlane / self.groups * kH * kW

        if not batch then
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4))
        end

   end
   return self
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end



-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
end

local function unviewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
end

-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewGradWeight(self)
   if self.gradWeight and self.gradWeight:dim() > 0 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   end
end

local function unviewGradWeight(self)
   if self.gradWeight and self.gradWeight:dim() > 0 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end
-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewW(self)
   self.W = self.W:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
end

local function unviewW(self)
   self.W = self.W:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
end

-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewGradW(self)
   if self.gradW and self.gradW:dim() > 0 then
      self.gradW = self.gradW:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   end
end

local function unviewGradW(self)
   if self.gradW and self.gradW:dim() > 0 then
      self.gradW = self.gradW:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end



function Spatial_Weight_DBN_Row:updateOutput(input)


----used for the group eigen composition---------------------------
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

      centered:add(weight_perGroup, -1, self.buffer:expandAs(weight_perGroup))

       ----------------------calcualte the projection matrix----------------------
      self.buffer_1:resize(weight_perGroup:size(1),weight_perGroup:size(1))

      self.buffer_1:addmm(0,self.buffer_1,1/n_input,centered,centered:t()) --buffer_1 record correlation matrix
       if groupId ~= self.groups_WDBN then
          self.buffer_1:add(self.eye_ngroup * self.eps)
        else
          self.buffer_1:add(self.eye_ngroup_last * self.eps)
        end


   -----------------------matrix decomposition-------------

      local rotation,eig,_=torch.svd(self.buffer_1) 

      if self.debug then
          print(eig)
      end
      scale:resizeAs(eig)
      scale:copy(eig)
      scale:pow(-1/2) --scale=eig^(-1/2)

      self.buffer_2:resizeAs(rotation)
      self.buffer_2:cmul(scale:view(n_output, 1):expandAs(rotation),rotation:t())
     
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

 ---------------update main function----------------------

    input = makeContiguous(self, input)
    self:createIODescriptors(input)

    if self.isTraining then 
    -----------------------------transform----------------------
        viewWeight(self)
        viewW(self)
        local n_output=self.weight:size(1)
        local n_input=self.weight:size(2)

        self.eigs={}
        self.scales={}
        self.rotations={}
        self.centereds={}

        self.buffer = self.buffer or input.new()
        self.buffer_1 = self.buffer_1 or input.new()
        self.buffer_2 = self.buffer_2 or input.new()

        self.output=self.output or input.new()
        self.W=self.W or input.new()
        self.W:resizeAs(self.weight)


        for i=1,self.groups_WDBN do
             local start_index=(i-1)*self.m_perGroup+1
            local end_index=math.min(i*self.m_perGroup,n_output)
            self.W[{{start_index,end_index},{}}]=updateOutput_perGroup(self.weight[{{start_index,end_index},{}}],i)
        end

 
         unviewW(self)
        unviewWeight(self)
    end



------------------------------------------------cudnn excute-----------------------

    local finder = find.get()
    local fwdAlgo = finder:forwardAlgorithm(self, { self.iDesc[0], self.input_slice, self.WDesc[0],
                                                    self.W, self.convDesc[0], self.oDesc[0], self.output_slice})
    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
    for g = 0, self.groups - 1 do
        checkedCall(self,'cudnnConvolutionForward', cudnn.getHandle(),
                    cudnn.scalar(input, 1),
                    self.iDesc[0], input:data() + g*self.input_offset,
                    self.WDesc[0], self.W:data() + g*self.W_offset,
                    self.convDesc[0], fwdAlgo,
                    extraBuffer, extraBufferSize,
                    cudnn.scalar(input, 0),
                    self.oDesc[0], self.output:data() + g*self.output_offset);
    end

    -- add bias
    if self.bias then
        errcheck('cudnnAddTensor', cudnn.getHandle(),
                 cudnn.scalar(input, 1), self.biasDesc[0], self.bias:data(),
                 cudnn.scalar(input, 1), self.oDescForBias[0], self.output:data())
    end

    return self.output
end

function Spatial_Weight_DBN_Row:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.gradInput:resizeAs(input)
    assert(gradOutput:dim() == input:dim()-1 or gradOutput:dim() == input:dim()
              or (gradOutput:dim()==5 and input:dim()==4), 'Wrong gradOutput dimensions');
    input, gradOutput = makeContiguous(self, input, gradOutput)
    self:createIODescriptors(input)
    local finder = find.get()
    local bwdDataAlgo = finder:backwardDataAlgorithm(self, { self.WDesc[0], self.W, self.oDesc[0],
                                                             self.output_slice, self.convDesc[0], self.iDesc[0], self.input_slice })
    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
    for g = 0,self.groups - 1 do
        checkedCall(self,'cudnnConvolutionBackwardData', cudnn.getHandle(),
                    cudnn.scalar(input, 1),
                    self.WDesc[0], self.W:data() + g*self.W_offset,
                    self.oDesc[0], gradOutput:data() + g*self.output_offset,
                    self.convDesc[0],
                    bwdDataAlgo,
                    extraBuffer, extraBufferSize,
                    cudnn.scalar(input, 0),
                    self.iDesc[0], self.gradInput:data() + g*self.input_offset)
    end
    return self.gradInput
end

function Spatial_Weight_DBN_Row:accGradParameters(input, gradOutput, scale)


  ------------------calculate the K matrix---------------------
function getK(eig, is_last_group)
    local K
    if not is_last_group then
        K = self.initial_K:clone()
        local b_1 = torch.repeatTensor(eig, eig:size(1), 1)
        local b_2 = self.eye_ngroup:clone():add(b_1:t()):add(-1, b_1):add(K)
        K:fill(1):cdiv(b_2):add(-1, self.eye_ngroup*(1))
    else
        K = self.initial_K_last:clone()
        local b_1 = torch.repeatTensor(eig, eig:size(1), 1)
        local b_2 = self.eye_ngroup_last:clone():add(b_1:t()):add(-1, b_1):add(K)
        K:fill(1):cdiv(b_2):add(-1, self.eye_ngroup_last*(1))
    end
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
      self.U:cmul(scale:view(n_output, 1):expandAs(rotation),rotation:t())

     self.hat_x:mm( self.U,centered)
     self.d_hat_x:mm(rotation:t(),gradW_perGroup)

     self.FC:addmm(0, self.FC, 1/n_input, self.d_hat_x, self.hat_x:t())
     self.f:mean(self.d_hat_x, 2)


       local sz = (#self.FC)[1]

      local temp_mask
     if groupId == self.groups_WDBN then
        temp_mask = self.eye_ngroup_last
     else
        temp_mask = self.eye_ngroup
     end
     self.M = self.FC:clone():maskedFill(torch.eq(temp_mask, 0), 0)

 --------------------------------calculate S-----------------------------
     self.S:cmul(self.FC:t(), eig:view(sz, 1):expandAs(self.FC))

     
     self.buffer:resizeAs(eig):copy(eig):pow(1/2)
     self.buffer_1:cmul(self.FC, self.buffer:view(sz, 1):expandAs(self.FC))

     self.buffer_2:cmul(self.buffer_1, self.buffer:view(1, sz):expandAs(self.buffer_1))
     self.S:add(self.buffer_2)

     self.buffer=getK(eig,groupId == self.groups_WDBN)

     self.S:cmul(self.buffer:t())
     self.buffer:copy(self.S)
     self.S:add(self.buffer, self.buffer:t())

     self.S:add(-1, self.M)  -- S-M
     self.buffer_1:resizeAs(self.d_hat_x)
     self.buffer_1:mm( self.S:t(),self.hat_x) --(S-M)*self.hat_x

     self.buffer_1:add(self.d_hat_x):add(-1, self.f:expandAs(self.d_hat_x))

    self.gradWeight_perGroup:mm( self.U:t(),self.buffer_1)


      if self.unitLength_flag then
        self.gradWeight_perGroup:mul(math.sqrt(1/n_input))
      end

      return self.gradWeight_perGroup
  end

 -------------------------------------main function of accGrad------------







    self.scaleT = self.scaleT or self.W.new(1)
    -- this line forces this member to always be on CPU (needed for cudnn)
    self.scaleT = torch.type(self.W) == 'torch.CudaDoubleTensor'
       and self.scaleT:double() or self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale
    input, gradOutput = makeContiguous(self, input, gradOutput)
    self:createIODescriptors(input)
   self.gradW:fill(0)

    local finder = find.get()
    local bwdFilterAlgo = finder:backwardFilterAlgorithm(self, { self.iDesc[0], self.input_slice, self.oDesc[0],
                                                               self.output_slice, self.convDesc[0], self.WDesc[0], self.W})

    -- gradBias
    if self.bias then
        errcheck('cudnnConvolutionBackwardBias', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.oDescForBias[0], gradOutput:data(),
                 cudnn.scalar(input, 1),
                 self.biasDesc[0], self.gradBias:data())
    end

    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
    for g = 0, self.groups - 1 do
        -- gradWeight
       checkedCall(self,'cudnnConvolutionBackwardFilter', cudnn.getHandle(),
                   self.scaleT:data(),
                   self.iDesc[0], input:data() + g*self.input_offset,
                   self.oDesc[0], gradOutput:data() + g*self.output_offset,
                   self.convDesc[0],
                   bwdFilterAlgo,
                   extraBuffer, extraBufferSize,
                   cudnn.scalar(input, 1),
                   self.WDesc[0], self.gradW:data() + g*self.W_offset);
    end




-----------------------------transform--------------------------

    
   viewWeight(self)
   viewW(self)
   viewGradWeight(self)
   viewGradW(self)
    local n_output=self.weight:size(1)

   for i=1,self.groups_WDBN do
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,n_output)
         self.gradWeight[{{start_index,end_index},{}}]=updateAccGradParameters_perGroup(self.gradW[{{start_index,end_index},{}}],i)
    end




   unviewWeight(self)
   unviewW(self)
   unviewGradWeight(self)
   unviewGradW(self)


    return self.gradOutput
end

function Spatial_Weight_DBN_Row:clearDesc()
    self.WDesc = nil
    self.biasDesc = nil
    self.convDesc = nil
    self.iDesc = nil
    self.oDesc = nil
    self.oDescForBias = nil
    self.oSize = nil
    self.scaleT = nil
    return self
end

function Spatial_Weight_DBN_Row:write(f)
    self:clearDesc()
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function Spatial_Weight_DBN_Row:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_input', '_gradOutput', 'input_slice', 'output_slice')
   return nn.Module.clearState(self)
end

function Spatial_Weight_DBN_Row:endTraining()

----used for the group eigen composition---------------------------
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
       --print(self.groups_WDBN)
       --print(self.eye_ngroup)
       --print(self.eye_ngroup_last)
       if groupId ~= self.groups_WDBN then
          self.buffer_1:add(self.eye_ngroup * self.eps)
        else
          self.buffer_1:add(self.eye_ngroup_last * self.eps)
        end


   -----------------------matrix decomposition-------------

      local rotation,eig,_=torch.svd(self.buffer_1) --reuse the buffer: 'buffer' record e, 'buffer_2' record V



      if self.debug then
          print(eig)
      end
    scale:resizeAs(eig)
      scale:copy(eig)
      scale:pow(-1/2) --scale=eig^(-1/2)

      self.buffer_2:resizeAs(rotation)
    --  self.buffer_1:diag(scale)   --self.buffer_1 cache the scale matrix
    --  self.buffer_2:mm(self.buffer_1,rotation:t()) --U= Eighta^(-1/2)*D^T
      self.buffer_2:cmul(torch.repeatTensor(scale, (#scale)[1], 1):t(),rotation:t())

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

  ------------------------main funciton-------------------
  
  viewWeight(self)
   viewW(self)
   local n_output=self.weight:size(1)
   local n_input=self.weight:size(2)

 --   self.eigs={}
 --  self.scales={}
 --  self.rotations={}
 --  self.centereds={}

   -- buffers that are reused
--   self.buffer = self.buffer or input.new()
--   self.buffer_1 = self.buffer_1 or input.new()
--   self.buffer_2 = self.buffer_2 or input.new()

 --  self.output=self.output or input.new()
 --  self.W=self.W or input.new()
 --  self.W:resizeAs(self.weight)


      for i=1,self.groups_WDBN do
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,n_output)
         self.W[{{start_index,end_index},{}}]=updateOutput_perGroup(self.weight[{{start_index,end_index},{}}],i)
      end

 
   unviewW(self)
   unviewWeight(self)
   self.isTraining=false
   ------------------clear buffer-----------
--    self.weight:set()
    self.buffer:set()
    self.buffer_1:set()
    self.buffer_2:set()
 --   self.gradWeight:set()
--    self.gradW:set()
    self.gradInput:set()
    self.W_perGroup:set()
    self.gradWeight_perGroup:set()
    self.hat_x:set()
    self.S:set()
    self.M:set()
    self.U:set()
    self.f:set()
    self.FC:set()
    self.d_hat_x:set()
  --  self.eigs=nil
  --  self.scales=nil
  --  self.rotations=nil
  --  self.centereds=nil

end
