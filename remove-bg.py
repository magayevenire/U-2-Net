import os
from matplotlib import image
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
import cv2
import numpy as np

def finalize(image):

    # Read image
    img = cv2.imread(f'test_data3/inputs/{image}.jpg')
    mask = cv2.imread(f'test_data3/masks/{image}.png' )
    mask2 = cv2.bitwise_not(mask)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    print(mask.dtype)
    

    final = cv2.bitwise_and(img,img, mask=mask)

    print(img.shape)
    print(mask.shape)
    # print(final.shape)
    # #make mask of where the transparent bits are
    # trans_mask = final[:,:,3] == 0

    # #replace areas of transparency with white and not transparent
    # final[trans_mask] = [255, 255, 255, 255]

    # #new image without alpha channel...
    # # new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    # rgba = cv2.cvtColor(final, cv2.COLOR_RGB2RGBA)
    # rgba[:, :, 3] = mask[:, :, 1]
    cv2.imshow('img', img)
    cv2.imshow('mask', mask)
    cv2.imshow('final', final)
    cv2.imshow('mask2', mask2)



    cv2.waitKey(0)
    cv2.destroyAllWindows()

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    print("pred",pred)
    predict = predict.squeeze()
    print("predict",predict)

    predict_np = predict.cpu().data.numpy()
    print("predict_np",predict_np)
    
    # print(type(predict_np),predict_np.shape)

    im = Image.fromarray(predict_np*255).convert('RGB')
    print("im",im)

    # print("im",im.shape)
    # im.show("ddd")
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    # print("image",image.shape)

    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    print("pb_np",pb_np.shape)


    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    # print(d_dir+imidx+'.png')
    imo.save(d_dir+imidx+'.png')
    # print("imo",imo.shape)
    finalize(imidx)



def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp
    image_name = "magaye.jpg"
    image = os.path.join(os.getcwd(), 'test_data3', 'inputs' + os.sep +image_name)
    prediction_dir = os.path.join(os.getcwd(), 'test_data3', 'masks'+ os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
   
    # img_name_list = glob.glob(image_dir + os.sep + '*')
    # print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = [image],
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    print("inferencing:",image.split(os.sep)[-1])

    for i_test, data_test in enumerate(test_salobj_dataloader):


        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(image,pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
