from model import Net
import torch
import cv2
from torchvision import transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image

categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'space', 'nothing', 'del']

labels = dict()

for ind, label in enumerate(categories):
    labels[ind] = label

print(labels)

state_dict = torch.load("asl_2.pt", map_location=torch.device('cpu'))
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v

net = Net()
print(net)
net.load_state_dict(new_state_dict)
total_params = sum(p.numel() for p in net.parameters())
print(total_params)
loader = transforms.Compose([transforms.Resize(150), transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


def resize_image(size, im):
    desired_size = size
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im


def image_loader(image_name):
    """load image, returns cuda tensor"""
    img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    image = resize_image(150, img)
    image = Image.fromarray(image)
    image = loader(image).float()
    print(image.shape)
    image.unsqueeze_(0)
    image = Variable(image)
    out = net(image).data.numpy().argmax()
    print("Sign is: {}".format(str(labels[out])))
    cv2.putText(img, '{}'.format(labels[out]), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("letter", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_loader("images/F6.jpg")
