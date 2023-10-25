from __future__ import print_function
import argparse
from importlib.util import set_loader
import cv2
import numpy
import os
from typing_extensions import Self
import torch
import torchvision.transforms as transforms
import numpy as np
from os.path import join
import time
from PIL import Image, ImageOps
from os import listdir
import os
from libs.models import encoder4
from libs.models import decoder4
from libs.Matrix import MulLayer_2x
import image_hsv

# Training settings
parser = argparse.ArgumentParser(description='LT-VAE multiple style transfer')
parser.add_argument('--image_dataset', type=str, default='Test')
parser.add_argument("--latent", type=int, default=256, help='length of latent vector')
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth', help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth', help='pre-trained decoder path')
parser.add_argument("--matrixPath", default='models/matrix_r41_new.pth', help='pre-trained model path')

opt = parser.parse_args()

print(opt._get_args)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

content_img = cv2.imread("Test/content/camp2.jpg", 1)

vgg = encoder4()
dec = decoder4()
matrix = MulLayer_2x(z_dim=opt.latent)
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
matrix.load_state_dict(torch.load(opt.matrixPath))

vgg.to(device)
dec.to(device)
matrix.to(device)


def eval(k1,k2,k3):
    global content_img
    matrix.eval()
    vgg.eval()
    dec.eval()

    content_path = os.path.join(opt.image_dataset, 'content')
    output_path = os.path.join(opt.image_dataset, 'result')
    ref_path = os.path.join(opt.image_dataset, 'style')

    style_tf = test_transform(size=256, crop=True)
    ref_list = []
    rn = []

    for ref_file in os.listdir(ref_path):
        rn.append(ref_file)
        ref = Image.open(os.path.join(ref_path, ref_file)).convert('RGB')
        ref_list.append(ref) 

    ref_list[0] = Image.open("Test/style/mona_lisa.jpeg").convert('RGB')
    rn[0] = "freshcold"
    ref_list[1] = Image.open("Test/style/picasso.jpg").convert('RGB')
    rn[1] = "la_muse"
    ref_list[2] = Image.open("Test/style/oversoul.jpg").convert('RGB')
    rn[2] = "z"

    # pix = numpy.asarray(ref_list[1])
    # pic = image_hsv.image_hsv(pix,0, -50, -100)
    # ref_list[1]= Image.fromarray(pic)
    # pix = numpy.asarray(ref_list[0])
    # pic = image_hsv.image_hsv(pix,0, -50, 100)
    # pic = cv2.GaussianBlur(pix, (7,7),0)
    # ref_list[0]= Image.fromarray(pic)

    for i in range(3):
        ref_list[i] = style_tf(ref_list[i]).unsqueeze(0).to(device)
    for cont_file in os.listdir(content_path):
        content = Image.open(os.path.join(content_path, cont_file)).convert('RGB')
        content = transform(content).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = chop_forward(k1, k2, k3, content, ref_list)

        prediction = prediction * 255.0
        prediction = prediction.clamp(0, 255)
        content_img = numpy.asarray(prediction)
        file_name = rn[0].split('.')[0]+str(k1)+rn[1].split('.')[0]+str(k2)+rn[2].split('.')[0]+str(k3)+'.png'
        name = os.path.join(output_path, file_name)
        print(name)
        Image.fromarray(np.uint8(prediction)).save(name)



def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform =transforms.Compose(transform_list)

    return transform


transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)


def chop_forward(k1,k2,k3,content, ref):

    with torch.no_grad():
        sF_1 = vgg(ref[0])
        sF_2 = vgg(ref[1])
        sF_3 = vgg(ref[2])
        cF = vgg(content)
        feature, _ = matrix(k1, k2, k3, cF['r41'], sF_1['r41'], sF_2['r41'], sF_3['r41'])
        transfer = dec(feature)
        transfer = transfer.data[0].cpu().permute(1, 2, 0)

    return transfer


eval(0,0.5,0.5)



# style1_val = 1
# style2_val = 0
# style3_val = 0

# def on_trackbar_style1(val):
#     global style1_val
#     global style2_val
#     global style3_val
#     style1_val = val/100
#     if style1_val+style2_val > 1:
#         style1_val = style1_val/(style1_val+style2_val)
#         style2_val = style2_val/(style1_val+style2_val)
#     style3_val = 1-style2_val-style1_val

# def on_trackbar_style2(val):
#     global style1_val
#     global style2_val
#     global style3_val
#     style2_val = val/100
#     if style1_val+style2_val > 1:
#         style1_val = style1_val/(style1_val+style2_val)
#         style2_val = style2_val/(style1_val+style2_val)
#     style3_val = 1-style2_val-style1_val

# # img = cv.imread("starrynight.png", 1)

# cv2.namedWindow("Final Image")
# cv2.createTrackbar("Style 1", "Final Image", 100, 100, on_trackbar_style1)
# cv2.createTrackbar("Style 2", "Final Image", 0, 100, on_trackbar_style2)

# while True:
#     cv2.imshow("Final Image", content_img)
#     if cv2.waitKey(1) == ord('s'):
#         print("pressed s")
#         eval(style1_val, style2_val, style3_val)
#     if cv2.waitKey(1) == ord('q'):
#         cv2.destroyAllWindows()
#         break
