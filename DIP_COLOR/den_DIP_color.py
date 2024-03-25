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

##TAMBIEN CAMBIAR ARCHIVOS csv
#namesIMGE = np.array(['image_House256_greylevel.png', 'image_Lena512_greylevel.png', 'image_Peppers512_greylevel.png' , 'F16_GT_greylevel.png','image_Baboon512_greylevel.png'])
#for k in range(len(namesIMGE)):
#    nameimg =  namesIMGE[k]

for k in range(1,25):#25
	#nameimg = 'kodim18.png'#'image_Lena512rgb.png' #
	nameimg = 'kodim' + str(k) + '.png'

	#Lectura de la imagen clean y degradada
	#ruta = '/content/gdrive/MyDrive/SeminariodetesisI/Codigo/DIP_VBTV/data/deblurring/' 
	ruta = './data/denoising/' 
	fname = ruta + nameimg
	img_pil = crop_image(get_image(fname,-1)[0], d=32)
	wh, ht  = img_pil.size

	img_np = pil_to_np(img_pil)

	# Add synthetic noise
	sigma = 25
	img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma/255.)    
	img_noisy_pil.save('data/noisy_25/noisy_25_' + nameimg)
	img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

	#Arreglos de datos
	plotY = []
	maxPnsr = []
	Lambda =0
	# Setup
	INPUT = 'noise'
	pad = 'reflection'
	OPT_OVER='net'

	reg_noise_std = 1./30.
	LR = 0.01

	OPTIMIZER = 'adam' 
	exp_weight = 0.99

	num_iter = 5000
	input_depth = 32

	full_net = DIP(input_depth, pad, upsample_mode='bilinear' ).type(dtype)

	net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

	# Compute number of parameters
	s  = sum([np.prod(list(p.size())) for p in full_net.parameters()]); 

	# Loss
	mse = torch.nn.MSELoss().type(dtype)

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

		net_output= full_net(net_input)
	
		loss = mse(net_output,img_noisy_torch)

		loss.backward(retain_graph=True)
			
		out_img_avg = out_img_avg * exp_weight + net_output.detach().cpu().numpy()[0] * (1 - exp_weight)
		
		val = compare_psnr(out_img_avg,img_np) 
		pnsrLS.append(val)

		#Guardar imagen con mayor pnsr
		if valorMax < val : 
			imgPNSRmax_np = out_img_avg
			valorMax = val


		i += 1

		return loss

	p = get_params(OPT_OVER, full_net, net_input, input_depth)
	optimize(OPTIMIZER, p, closure, LR, num_iter)

	#Save image output
	out_img_avg_pil = np_to_pil(out_img_avg)
	#ruta = '/content/gdrive/MyDrive/SeminariodetesisI/Codigo/DIP_VBTV/restoration/denoised_'
	ruta = './restoration/PC03/denoised_'
	fname = ruta  + nameimg 
	out_img_avg_pil.save(fname)

	#Save best image 
	imgPNSRmax_pil = np_to_pil(imgPNSRmax_np)
	fname = './restoration/PC03/BestPnsr' + nameimg 
	imgPNSRmax_pil.save(fname)

	#Datos para gráfica
	y = pnsrLS
	plotY.append(y)
	
	#Datos para tabla
	maxIt = np.array([Lambda,np.max(y),np.argmax(y), y[-1]])
	maxPnsr.append(maxIt)
	
	#Guardar datos de gráfica
	Y = np.array(plotY)
	ruta = './txtC03/'
	name = nameimg.replace('.png','.csv')
	fname = ruta + 'PC03denPNSR' + name
	np.savetxt(fname, Y.T, delimiter =",",fmt ='% s')
			
	#Guardar datos para tabla
	MX = np.array(maxPnsr)
	ruta = './txtC03/'
	name = nameimg.replace('.png','.csv')
	fname = ruta  + 'PC03denMAXval' + name
	np.savetxt(fname, MX, delimiter =",",fmt ='% s')