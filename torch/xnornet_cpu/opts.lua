-- --------------------------------------------------
-- Parse Options From Command Line
-- 
--  Written by Jiaolong Xu
--  Date: 03/12/17
--  Copyright (c) 2017
-- --------------------------------------------------
local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('MNIST Training Script')
    cmd:text()
    cmd:text('Options: ')
    ----------- General Options ------------
    cmd:option('-cacheDir',     './cache/', 'directory to save logs')
    cmd:option('-dataDir',      '../../data/MNIST_data/unzip/', 'directory of MNIST dataset')
    ----------- Training Options ------------
    cmd:option('-nEpochs',      5,     'number of total epochs to run')
    cmd:option('-batchSize',    100,   'mini-batch size')
    cmd:option('-lr',           0.1,   'learning rate')
    cmd:option('-momentum',     0.9,   'momentum for SGD')
    ----------- Model Options ------------
    cmd:option('-modelName',    'mnist_cnn', 'Options are: mnist_cnn | mnist_bwn | mnist_xnor')
    cmd:text()

    local opt = cmd:parse(arg or {})
    return opt
end

return M
