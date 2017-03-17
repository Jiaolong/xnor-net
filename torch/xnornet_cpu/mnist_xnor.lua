-- --------------------------------------------------
-- MNIST XNOR-Net Model  
-- 
--  Written by Jiaolong Xu
--  Date: 03/11/17
--  Copyright (c) 2017
-- --------------------------------------------------
function createModel()
    require 'nn'    
    require 'binaryActivation'
    require 'binarySpatialConvolution'    

    -- Binary block: BN -> BinActiv -> BinConv
    local function BinBlock(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        local block = nn.Sequential()
        block:add(nn.SpatialBatchNormalization(nInputPlane, 1e-4, false))
        block:add(nn.BinaryActivation())
        block:add(nn.BinarySpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
        return block
    end

    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(1, 32, 5, 5)) -- 28 x 28 -> 24 x 24
    net:add(nn.SpatialBatchNormalization(32, 1e-5, false))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3, 3, 3, 3)) -- 24 x 24 -> 8 x 8

    net:add(BinBlock(32, 64, 5, 5)) -- 8 x 8 -> 4 x 4
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- 4 x 4 -> 2 x 2
    net:add(BinBlock(64, 200, 2, 2)) -- 2 x 2 -> 1 x 1


    net:add(nn.SpatialBatchNormalization(200, 1e-3, false))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialConvolution(200, 10, 1, 1))

    net:add(nn.View(10))
    net:add(nn.LogSoftMax())
    local model = net
    return model
end
