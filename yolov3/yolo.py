from __future__ import division

import torch
import torch.nn as nn
from torch.nn import modules
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np
from util import *

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class yolov3(nn.Module):
    def __init__(self, cfgfile):
        super(yolov3, self).__init__()

        self.cfgfile = cfgfile
        self.blocks = parse_cfg(self.cfgfile)
        self.net_info = self.blocks[0]
        self.module_list = nn.ModuleList()
        prev_filters = 3
        output_filters = []
        for index, x in enumerate(self.blocks[1:]):
            module = nn.Sequential()

            if x["type"] == "convolutional":
                activation = x["activation"]
                try:
                    batch_normalize = int(x["batch_normalize"])
                    bias = False
                except:
                    batch_normalize = 0
                    bias = True

                filters = int(x["filters"])
                padding = int(x["pad"])
                kernel_size = int(x["size"])
                stride = int(x["stride"])

                if padding:
                    pad = (kernel_size - 1) // 2
                else:
                    pad = 0

                conv = nn.Conv2d(in_channels=prev_filters, out_channels=filters, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)
                module.add_module("conv_{0}".format(index), conv)

                if batch_normalize:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module("batch_norm_{0}".format(index), bn)

                if activation == "leaky":
                    activn = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module("leaky_{0}".format(index), activn)

            elif x["type"] == "upsample":
                stride = int(x["stride"])
                upsample = nn.Upsample(scale_factor=2, mode="bilinear")
                module.add_module("upsample_{0}".format(index), upsample)
            
            # 路由层输出 layers所代表索引层的输出的拼接
            elif x["type"] == "route":
                x["layers"] = x["layers"].split(',')
                start = int(x["layers"][0])
                try:
                    end = int(x["layers"][1])
                except:
                    end = 0

                if start > 0:
                    start = start - index
                if end > 0:
                    end = end - index
                
                route = EmptyLayer()
                module.add_module("route_{0}".format(index), route)
                if end < 0:
                    filters = output_filters[index + start] + output_filters[index + end]
                else:
                    filters= output_filters[index + start]
            
            elif x["type"] == "shortcut":
                shortcut = EmptyLayer()
                module.add_module("shortcut_{}".format(index), shortcut)

            elif x["type"] == "yolo":
                mask = x["mask"].split(",")
                mask = [int(x) for x in mask]

                anchors = x["anchors"].split(",")
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
                anchors = [anchors[i] for i in mask]

                detection = DetectionLayer(anchors)
                module.add_module("Detection_{}".format(index), detection)

            self.module_list.append(module)
            prev_filters = filters
            output_filters.append(filters)

    def forward(self, x):
        blocks = self.blocks[1:]
        outputs = {}

        write = 0
        for i, module in enumerate(blocks):
            if module["type"] == "convolutional" or module["type"] == "upsample":
                x = self.module_list[i](x)

            elif module["type"] == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
    
                    x = torch.cat((map1, map2), 1)

            elif module["type"] == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i + from_]

            elif module["type"] == "yolo":
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        
        return detections

    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)



def parse_cfg(cfgfile):

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0] # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#'] # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

model = yolov3("yolov3.cfg")
model.load_weights("yolov3.weights")
inp = get_test_input()
pred = model(inp)
print(pred)