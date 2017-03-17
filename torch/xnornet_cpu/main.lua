-- --------------------------------------------------
-- Train and Test MNIST Model  
-- 
--  Written by Jiaolong Xu
--  Date: 03/11/17
--  Copyright (c) 2017
-- --------------------------------------------------
require 'torch'
require 'paths'
require 'xlua'
local mnist = require 'dataset-mnist'

-- fix random seed
torch.manualSeed(1234)
-- threads
torch.setnumthreads(4)
-- use floats
torch.setdefaulttensortype('torch.FloatTensor')

-- load options
local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)
-- create cache directory for log files
os.execute('mkdir -p ' .. opt.cacheDir)

paths.dofile('model.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')

print(opt)

-- load dataset
local trainset = mnist.traindataset(opt.dataDir)
local testset  = mnist.testdataset(opt.dataDir)

-- train and test
inputSize = {28, 28}
batchSize = opt.batchSize
numEpoch  = opt.nEpochs

for i=1, numEpoch do
    -- train one epoch
    local loss = train(trainset)
    print(string.format('current loss: %4f', loss))
    -- test
    eval(testset)
end
