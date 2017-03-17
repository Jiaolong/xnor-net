-- --------------------------------------------------
-- Create a NN Model 
-- 
--  Written by Jiaolong Xu
--  Date: 03/11/17
--  Copyright (c) 2017
-- --------------------------------------------------
require 'nn'
require 'binarySpatialConvolution'

function randomInitWeight(layer)
    local t = torch.type(layer)
    if t == 'nn.SpatialConvolution' or t == 'nn.BinarySpatialConvolution' then
        local c = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane))
        layer.weight:copy(torch.randn(layer.weight:size()) * c)
        layer.bias:fill(0)
    end
end

-- Create Network
modelFile = paths.concat(opt.modelName .. '.lua')
paths.dofile(modelFile)
print('=> Creating model from file: ' .. modelFile)
model = createModel()

-- Initializing parameters
model:apply(randomInitWeight)

-- Get parameters and gradient pointers
parameters, gradParameters = model:getParameters()

-- Create Criterion
criterion = nn.ClassNLLCriterion()

-- Print model and criterion
print('=> Model')
print(model)

print('=> Criterion')
print(criterion)
