-- --------------------------------------------------
-- Binary Weight MNIST CNN Model  
-- 
--  Written by Jiaolong Xu
--  Date: 03/11/17
--  Copyright (c) 2017
-- --------------------------------------------------
function createModel()
    require 'nn'
    require 'binarySpatialConvolution'    
    
    local function Block(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        local block = nn.Sequential()
        block:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
        block:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
        block:add(nn.ReLU(true))
        return block
    end
    
    local function BWBlock(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        local block = nn.Sequential()
        block:add(nn.BinarySpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
        block:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
        block:add(nn.ReLU(true))
        return block
    end

    local net = nn.Sequential()
    net:add(Block(1, 32, 5, 5)) -- 28 x 28 -> 24 x 24
    net:add(nn.SpatialMaxPooling(3, 3, 3, 3)) -- 24 x 24 -> 8 x 8
    net:add(BWBlock(32, 64, 5, 5)) -- 8 x 8 -> 4 x 4
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- 4 x 4 -> 2 x 2
    net:add(BWBlock(64, 200, 2, 2)) -- 2 x 2 -> 1 x 1
    net:add(nn.SpatialConvolution(200, 10, 1, 1))
    net:add(nn.View(10))
    net:add(nn.LogSoftMax())
    local model = net
    return model
end
