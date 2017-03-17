-- --------------------------------------------------
-- BinarySpatialConvolution Layer
-- 
--  Written by Jiaolong Xu
--  Date: 03/12/17
--  Copyright (c) 2017
-- --------------------------------------------------
local THNN = require 'nn.THNN'
local BinarySpatialConvolution, parent = torch.class('nn.BinarySpatialConvolution', 'nn.Module')

function BinarySpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   -- alpha
   self.alpha = torch.Tensor(1)

   self:reset()
   -- real-value weight
   self.realWeight = self.weight:clone() 

   self.nElement   = self.weight[1]:nElement()
   self.weightSize = self.weight:size()
end

function BinarySpatialConvolution:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function BinarySpatialConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      if self.bias then
         self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then
         self.bias:uniform(-stdv, stdv)
      end
   end
end

local function backCompatibility(self)
   self.finput = self.finput or self.weight.new()
   self.fgradInput = self.fgradInput or self.weight.new()
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   else
      self.padW = self.padW or 0
      self.padH = self.padH or 0
   end
   if self.weight:dim() == 2 then
      self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
   if self.gradWeight and self.gradWeight:dim() == 2 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

local function meanCenterWeight(self)
    local negMean = self.weight:mean(2):mul(-1):repeatTensor(1, self.weightSize[2], 1, 1)
    self.weight:add(negMean)
end

local function clampWeight(self)
    self.weight:clamp(-1, 1)
end

local function binarizeWeight(self)
    -- A = 1/n |W|, Eq(6) in the paper 
    self.alpha = self.weight:norm(1, 4):sum(3):sum(2):div(self.nElement)
    -- W = A * B
    self.weight:sign():cmul(self.alpha:expand(self.weightSize))
end

local function updateBinaryGradWeight(self)
    -- Note: for updating the parameters, real-value weight is used
    -- A = 1/n |W|, Eq(6) in the paper 
    Ae = self.alpha:expand(self.weightSize)
    -- gradients for sign()
    Ae[self.weight:le(-1)] = 0
    Ae[self.weight:ge(1)]  = 0
    Ae:add(1/self.nElement)
    -- Ae:mul(1 - 1/s[2]):mul(n) -- this is not explained in the paper
    self.gradWeight:cmul(Ae)
end

function BinarySpatialConvolution:updateOutput(input)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')

   -- backup real-value weight
   self.realWeight:copy(self.weight)
   -- binarize weight
   binarizeWeight(self)

   backCompatibility(self)
   input.THNN.SpatialConvolutionMM_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      THNN.optionalTensor(self.bias),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
   )
   return self.output
end

function BinarySpatialConvolution:updateGradInput(input, gradOutput)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   if self.gradInput then
      backCompatibility(self)
      input.THNN.SpatialConvolutionMM_updateGradInput(
         input:cdata(),
         gradOutput:cdata(),
         self.gradInput:cdata(),
         self.weight:cdata(),
         self.finput:cdata(),
         self.fgradInput:cdata(),
         self.kW, self.kH,
         self.dW, self.dH,
         self.padW, self.padH
      )
      -- restore real-value weight
      self.weight:copy(self.realWeight)
      -- update real-value weight gradient
      updateBinaryGradWeight(self)      

      return self.gradInput
   end
end

function BinarySpatialConvolution:accGradParameters(input, gradOutput, scale)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   scale = scale or 1
   backCompatibility(self)
   input.THNN.SpatialConvolutionMM_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      THNN.optionalTensor(self.gradBias),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,
      scale
   )
end

function BinarySpatialConvolution:type(type,tensorCache)
   self.finput = self.finput and torch.Tensor()
   self.fgradInput = self.fgradInput and torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function BinarySpatialConvolution:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   if self.bias then
      return s .. ')'
   else
      return s .. ') without bias'
   end
end

function BinarySpatialConvolution:clearState()
   nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
   return parent.clearState(self)
end
