-- --------------------------------------------------
-- Binary Activation Layer  
-- 
--  Written by Jiaolong Xu
--  Date: 03/12/17
--  Copyright (c) 2017
-- --------------------------------------------------

local BinaryActivation, parent = torch.class('nn.BinaryActivation', 'nn.Module')

function BinaryActivation:updateOutput(input)
    local s = input:size()
    self.output:resizeAs(input):copy(input)
    self.output = self.output:sign()
    return self.output
end

function BinaryActivation:updateGradInput(input, gradOutput)
    local s = input:size()
    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    self.gradInput[input:ge(1)] = 0
    self.gradInput[input:le(-1)] = 0
    return self.gradInput
end

