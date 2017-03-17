-- --------------------------------------------------
-- Test Model  
-- 
--  Written by Jiaolong Xu
--  Date: 03/12/17
--  Copyright (c) 2017
-- --------------------------------------------------
function computeScore(outputs, labels)
    local num = outputs:size(1)
    local _, preds = outputs:float():sort(2, true) -- descending
    local correct = preds:eq(labels:long():view(num, 1):expandAs(outputs))
    local acc = correct:narrow(2, 1, 1):sum() / num
    return acc
end

function eval(dataset)
    print('==> Start testing ...')
    -- set mode in evaluation mode
    model:evaluate()

    local acc = 0
    local count = 0
    for t = 1, dataset.size, batchSize do
        -- display progress
        xlua.progress(t, dataset.size)

        -- create mini batch
        local inputs = torch.Tensor(batchSize, 1, inputSize[1], inputSize[2])
        local labels = torch.Tensor(batchSize)
        local k = 1
        for i = t, math.min(t + batchSize - 1, dataset.size) do
            -- load new sample
            inputs[k] = dataset.data[i]
            labels[k] = dataset.label[i]
            k = k + 1
        end
        -- convert label from [0,9] to [1, 10]
        labels:add(1)
        -- test mini batch
        local preds = model:forward(inputs)
        local acc_t = computeScore(preds, labels)
        acc = acc + acc_t
        count = count + 1
    end
    acc = acc / count
    print("Testing accuracy = " .. (acc * 100) .. '%')
end

