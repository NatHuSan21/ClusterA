from __future__ import print_function

import os
import cv2
import numpy as np
from models import *
from models.vbtv import*

import torch
import torch.optim

import kornia

from utils.denoising_utils import *
from utils.blur_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


Nprueb = '420'
denb = 'den15'

agrepRep = True
LambEsp = 0.005

##TAMBIEN CAMBIAR ARCHIVOS csv
#namesIMGE = np.array(['image_House256_greylevel.png', 'image_Lena512_greylevel.png', 'image_Peppers512_greylevel.png' , 'F16_GT_greylevel.png','image_Baboon512_greylevel.png'])
#for k in range(len(namesIMGE)):
#    nameimg =  namesIMGE[k]

    
listaRep = [18]

for k in listaRep:#for k in range(1,25):#25
    #nameimg = 'kodim18.png'#'image_Lena512rgb.png' #
    if denb[:3] == 'den':
        nameimg = 'kodim' + str(k) + '.png'
    else:
        nameimg = 'kodim' + str(k) + '_small.png'

	#Lectura de la imagen clean y degradada
	#ruta = '/content/gdrive/MyDrive/SeminariodetesisI/Codigo/DIP_VBTV/data/deblurring/' 
    ruta = './data/denoising/' 
    fname = ruta + nameimg
    img_pil = crop_image(get_image(fname,-1)[0], d=32)
    img_width,img_height = img_pil.size
    
    img_np = pil_to_np(img_pil)


    ruta = './data/noisy_15C/' 
    fname = ruta + 'noisy_15_' + nameimg
    img_noisy_pil = crop_image(get_image(fname,-1)[0], d=32)
    img_noisy_np = pil_to_np(img_noisy_pil)
    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

    #Arreglos de datos
    plotY = []
    maxPnsr = []

    #Valores Lambda de prueba
    if agrepRep:
        LAMval = [LambEsp]
    else:
        LAMval = [0,0.000001,0.00001,0.00003,0.00005,0.0001,0.00025,0.0005,0.001,0.0025,0.005]
    #LAMval = [0.005]
    
    for lam in LAMval :
        # Setup
        INPUT = 'noise'
        pad = 'reflection'
        OPT_OVER ='net'

        reg_noise_std = 1./50. ###CAMBIO DE 1/30. A 1/40.
        LR = 0.01

        Lambda = lam ########## DIP ##########

        OPTIMIZER = 'adam' 
        exp_weight = 0.99

        num_iter = 10000
        input_depth = 32


        full_net =  VectorialTotalVariation(input_depth, pad, height=img_height, width= img_width, upsample_mode='bilinear').type(dtype)
        net_input = get_noise(input_depth, INPUT, (img_height,img_width)).type(dtype).detach()

        # Compute number of parameters
        s  = sum([np.prod(list(p.size())) for p in full_net.parameters()]); 

        # Loss
        mse = torch.nn.MSELoss().type(dtype)
        mae = torch.nn.L1Loss().type(dtype)

        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        out_img_avg = img_noisy_np

        pnsrLS = []

        valorMax = 0
        imgPNSRmax_np = np.zeros(out_img_avg.shape)

        i = 0
        def closure():
                
            global i, exp_weight, out_img_avg, net_input

            #Para guardar mejor imagen
            global valorMax, imgPNSRmax_np
                
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)    

            net_output, regularizer = full_net(net_input)	

            loss_regularizer = mae(regularizer,torch.zeros(1,img_height,img_width).type(dtype))
            
            loss_dataterm = mse(net_output,img_noisy_torch)

            total_loss = loss_dataterm + Lambda*loss_regularizer
            total_loss.backward(retain_graph=True)
						
            out_img_avg = out_img_avg * exp_weight + net_output.detach().cpu().numpy()[0] * (1 - exp_weight)
			
            #print(img_np.shape)
            #print(out_img_avg.shape)
            val = compare_psnr(img_np,out_img_avg)#compare_pnsr
            pnsrLS.append(val)
                
            if valorMax < val : #Guardar imagen con mayor pnsr
                imgPNSRmax_np = out_img_avg
                valorMax = val


            i += 1

            return total_loss

        p = get_params(OPT_OVER, full_net, net_input, input_depth)
        optimize(OPTIMIZER, p, closure, LR, num_iter)

        #Save image output
        out_img_avg_pil = np_to_pil(out_img_avg)
        #ruta = './restoration/P290/denoised_P290_'
        ruta = './restoration/P'+ Nprueb +'/denoised_P'+ Nprueb +'_'
        fname = ruta + str(Lambda)  + nameimg 
        out_img_avg_pil.save(fname)

        #Save best image 
        imgPNSRmax_pil = np_to_pil(imgPNSRmax_np)
        #fname = './restoration/P290/BestPnsr_P290_' + str(Lambda) + nameimg
        fname = './restoration/P'+ Nprueb +'/BestPnsr_P'+ Nprueb +'_' + str(Lambda) + nameimg 
        #ruta2 = './restoration/P'+ Nprueb +'/BestPnsr'+ denb+'_P'+ Nprueb +'_'
        imgPNSRmax_pil.save(fname)

        #Datos para gráfica
        y = pnsrLS
        plotY.append(y)
        
        #Datos para tabla
        maxIt = np.array([Lambda,np.max(y),np.argmax(y),y[-1]])
        maxPnsr.append(maxIt)
 
    if agrepRep:
        name = nameimg.replace('.png','_2.csv')
    else:
        name = nameimg.replace('.png','.csv')

    #Guardar datos de gráfica
    Y = np.array(plotY)
    #ruta = './txt290/'
    ruta = './txt'+ Nprueb +'/'
    #name = nameimg.replace('.png','.csv')
    #fname = ruta + 'P290denPNSR' + name
    fname = ruta + 'P'+ Nprueb + denb + 'PNSR' + name
    np.savetxt(fname, Y.T, delimiter =",",fmt ='% s')
                
    #Guardar datos para tabla
    MX = np.array(maxPnsr)
    #ruta = './txt290/'
    ruta = './txt'+ Nprueb +'/'
    #name = nameimg.replace('.png','_2.csv')
    #fname = ruta  + 'P290denMAXval' + name
    fname = ruta + 'P'+ Nprueb + denb + 'MAXval' + name
    np.savetxt(fname, MX, delimiter =",",fmt ='% s')
