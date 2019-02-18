import argparse

import PIL.Image as Image
import scipy.misc
import skimage.io as sio
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')

args = parser.parse_args()
print(args)

#===== load model =====
print('===> Loading model')

if args.cuda:
    net = torch.load(args.model)
    net = net.cuda()
else:
    net = torch.load(args.model, map_location=lambda storage, loc: storage)

#===== Load input image =====
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ]
)

import os, time

path = "./result/"

dehaze_path = './dehaze_result/'

l = os.listdir(path)

for i in l:
    # print(path + i)
    name = i.split(".")[0]
    # print(name)
    img = Image.open(path + i).convert('RGB')
    imgIn = transform(img).unsqueeze_(0)

    #===== Test procedures =====
    varIn = Variable(imgIn)
    if args.cuda:
        varIn = varIn.cuda()
    t1 = time.time()
    prediction = net(varIn)
    t2 = time.time()
    print(t2-t1)
    prediction = prediction.data.cpu().numpy().squeeze().transpose((1,2,0))
    scipy.misc.toimage(prediction).save(dehaze_path+name+'_dehaze.jpg')
    print(dehaze_path+name+'_dehaze.jpg')
