from __future__ import print_function

import os
import cv2
import numpy as np
from models import *
from models.vbtv import*
from models.dip_reg import*

import torch
import torch.optim

import kornia

from utils.denoising_utils import *
from utils.blur_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


##TAMBIEN CAMBIAR ARCHIVOS csv
namesIMGE = np.array(['image_House256_greylevel.png', 'image_Lena512_greylevel.png', 'image_Peppers512_greylevel.png' , 'F16_GT_greylevel.png','image_Baboon512_greylevel.png'])
for k in range(len(namesIMGE)):
    nameimg =  namesIMGE[k]

    
#for k in range(1,25):
    #nameimg = 'kodim18.png'#'image_Lena512rgb.png' #
#    nameimg = 'kodim' + str(k) + '_greylevel.png'
    
    #Lectura de la imagen clean y degradada
    #ruta = '/content/gdrive/MyDrive/SeminariodetesisI/Codigo/DIP_VBTV/data/deblurring/' 
    ruta = './data/noisyNew/clean/' 
    fname = ruta + nameimg
    img_pil = crop_image(get_image(fname,-1)[0], d=32)
    wh, ht  = img_pil.size
    
    img_np = pil_to_np(img_pil)


    ruta = './data/noisyNew/noisy/' 
    fname = ruta + 'noisy25_' + nameimg
    imgNoisy = crop_image(get_image(fname,-1)[0], d=32)
    img_noisy_np = pil_to_np(imgNoisy)
    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

    # Generta the degradation operator H
    NOISE_SIGMA = 2**.5
    BLUR_TYPE = 'gauss_blur'
    GRAY_SCALE = True
    USE_FOURIER = False

    ORIGINAL  = 'Clean'
    CORRUPTED = 'Blurred'

    
    luminance = img_noisy_torch
    #print(img_degraded_torch.shape)
    differential_luminance_x = derivative_central_x_greylevel(img_noisy_torch[0,0], dtype)
    differential_luminance_y = derivative_central_y_greylevel(img_noisy_torch[0,0], dtype)
    
    #Arreglos de datos
    plotY = []
    maxPnsr = []

    #Valores Lambda de prueba
        
    LAMval = [0,0.001,0.005,0.01,0.05]#0.000001,0.000003,0.000005,0.000007,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05]#,0.0003,0.0005,0.0007]#,0.05,0.075,0.1,0.25,0.5,0.75,1,1.5]


    for lam in LAMval:
        # Setup
        INPUT = 'noise'
        pad = 'reflection'
        OPT_OVER='net'

        reg_noise_std = 1./30.
        LR = 0.01
		#Lambda = 5 #0.05
        Lambda = lam
        beta = 3000.

        OPTIMIZER = 'adam' 
        exp_weight = 0.99

        num_iter = 5000
        input_depth = 64

        full_net = TotalVariationGreyLevelCentralDerivatives(input_depth, dtype, pad, height=ht, width=wh, upsample_mode='bilinear', n_channels=1, act_fun='LeakyReLU' ).type(dtype)

        net_input = get_noise(input_depth, INPUT, (ht, wh)).type(dtype).detach()

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

            net_output, curv = full_net(net_input)
		
            loss_regularizer = mae(curv,torch.zeros(1,1,ht,wh).type(dtype))

            loss_dataterm = mse(net_output,img_noisy_torch)
		
            total_loss = loss_dataterm + Lambda*loss_regularizer
            total_loss.backward(retain_graph=True)
						
            out_img_avg = out_img_avg * exp_weight + net_output.detach().cpu().numpy()[0] * (1 - exp_weight)
			
            img1 = pil_to_np(out_img_avg)
			
            val = compare_psnr(out_img_avg,img_np) 
            pnsrLS.append(val)

			#Guardar imagen con mayor pnsr
            if valorMax < val : 
                imgPNSRmax_np = out_img_avg
                valorMax = val

            i += 1

            return total_loss

		#Starting optimization with ADAM
        p = get_params(OPT_OVER, full_net, net_input, input_depth)
        optimize(OPTIMIZER, p, closure, LR, num_iter)

		#Save image output
        out_img_avg_pil = np_to_pil(out_img_avg)
		#ruta = '/content/gdrive/MyDrive/SeminariodetesisI/Codigo/DIP_VBTV/restoration/denoised_'
        ruta = './restoration/P202/denoised_'
        fname = ruta + str(Lambda) + nameimg 
        out_img_avg_pil.save(fname)

		#Save best image 
        imgPNSRmax_pil = np_to_pil(imgPNSRmax_np)
        fname = './restoration/P202/BestPnsr' + str(Lambda) + nameimg 
        imgPNSRmax_pil.save(fname)

		#Datos para gráfica
        y = pnsrLS
        plotY.append(y)
	
		#Datos para tabla
        maxIt = np.array([Lambda,np.max(y),np.argmax(y)])
        maxPnsr.append(maxIt)
	
	#Guardar datos de gráfica
    Y = np.array(plotY)
    ruta = './txt202/'
    name = nameimg.replace('.png','.csv')
    fname = ruta + 'P202denPNSR' + name
    np.savetxt(fname, Y.T, delimiter =",",fmt ='% s')
			
	#Guardar datos para tabla
    MX = np.array(maxPnsr)
    ruta = './txt202/'
    name = nameimg.replace('.png','.csv')
    fname = ruta  + 'P202denMAXval' + name
    np.savetxt(fname, MX, delimiter =",",fmt ='% s')