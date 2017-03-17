-- --------------------------------------------------
-- Train MNIST Model  
-- 
--  Written by Jiaolong Xu
--  Date: 03/11/17
--  Copyright (c) 2017
-- --------------------------------------------------
require 'optim'

-- create logger
trainLogger = optim.Logger(paths.concat(opt.cacheDir, 'train.log'))

-- setup optimization state
local optimState = {
    learningRate = opt.lr,
    learningRateDecay = 0,
    momentum = opt.momentum
}

-- train function
function train(dataset)
    -- epoch tracker
    epoch = epoch or 1
    print('Epoch # ' .. epoch)
    local shuffle = torch.randperm(dataset.size)
    local count = 0
    local loss = 0
    for t = 1, dataset.size, batchSize do
        local inputs = torch.Tensor(batchSize, 1, inputSize[1], inputSize[2])
        local labels = torch.Tensor(batchSize)
        local k = 1
        -- get next mini batch
        for i = t, math.min(t+batchSize-1, dataset.size) do
            local input = dataset.data[shuffle[i]]
            local label = dataset.label[shuffle[i]]
            inputs[k] = input
            labels[k] = label
            k = k + 1
        end
        -- convert label from [0,9] to [1, 10]
        labels:add(1)
 
        local feval = function(x)
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end
            
            -- reset gradients
            model:zeroGradParameters()
            local outputs = model:forward(inputs)
            local err = criterion:forward(outputs, labels)

            local gradOutputs = criterion:backward(outputs, labels)
            model:backward(inputs, gradOutputs)
             
            return err, gradParameters
        end

        _, fs = optim.sgd(feval, parameters, optimState)
        count = count + 1
        loss = loss + fs[1]
    
        -- display progress
        xlua.progress(t, dataset.size)
    end -- for t
     
    -- print loss
    avg_loss = loss / count
    trainLogger:add{['% loss'] = avg_loss}

    -- next epoch
    epoch = epoch + 1
    return avg_loss
end -- of train
