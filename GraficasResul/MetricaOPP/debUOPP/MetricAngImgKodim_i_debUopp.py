#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
    
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import random
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


# In[2]:


def printImgs(images,lsNames,lsValPSNR, n=1, m=1, title='',k=0):
    
    images = images[:n*m]
    fig, axs = plt.subplots(
        nrows=n, ncols=m, 
        figsize=(15, 20), 
        subplot_kw={
            'xticks': [], 
            'yticks': []
        }
    )
    i=0
    for ax, image,lsName,lsval in zip(axs.flat, images,lsNames,lsValPSNR):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(lsName)
        if i>k:
            ax.set_xlabel(str("{:.3f}".format(lsval)),fontsize=14)
        i=i+1    
    plt.savefig(title)    
    plt.show()


def printAng(images,lsNames,lsValPSNR, n=1, m=1, title=''):
    
    images = images[:n*m]
    fig, axs = plt.subplots(
        nrows=n, ncols=m, 
        figsize=(15, 20), 
        subplot_kw={
            'xticks': [], 
            'yticks': []
        }
    )

    # Find the min and max of all colors for use in setting the color scale.
    vmin = np.min(images)
    vmax = np.max(images)
    normalize = colors.Normalize(vmin=vmin, vmax=vmax)
    
    #normalize = colors.Normalize(vmin=0, vmax=math.pi)
    
    for ax, image,lsName,lsval in zip(axs.flat, images,lsNames,lsValPSNR):
        im=ax.imshow(image,norm=normalize,cmap='viridis')
        ax.set_title(lsName)
        ax.set_xlabel(str("{:.3f}".format(lsval)),fontsize=14) 
    
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)

    v = np.linspace(vmin, vmax, 15, endpoint=True)
    #fig.colorbar(im, cax=cax)
    fig.colorbar(im, ax=axs, orientation='vertical', fraction=.04,ticks=v)
        
    plt.savefig(title)    
    plt.show()
    

def printGrafs(images, n=1, m=1, title=''):
    
    images = images[:n*m]
    fig, axs = plt.subplots(
        nrows=n, ncols=m, 
        figsize=(18, 20), 
        subplot_kw={
            'xticks': [], 
            'yticks': []
        }
    )
    for ax, image in zip(axs.flat, images):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.savefig(title)    
    plt.show()


# In[45]:


def calcularAng(imgIN):

    color = [(255,0,0),(0,255,0),(0,0,255)]
    pares=[[0,1]]#,[0,2],[1,2]]
    
    
    imgObj = imgIN.copy()
    derX = imgObj[:,:-1,:].astype(int) - imgObj[:,1:,:].astype(int)
    derY = imgObj[:-1,:,:].astype(int) - imgObj[1:,:,:].astype(int)
    
    
    ##Orillas, coincidir dimensiones
    derX=derX[1:-1,:-1,:]
    derY=derY[:-1,1:-1,:]
    n,m,C = np.shape(derX)
    print('shape',n,m,C)
    
    normT = np.sqrt(derX**2 + derY**2)
    
    #Calcular productos
    prodBG = derX[:,:,0]*derX[:,:,1] + derY[:,:,0]*derY[:,:,1]
    prodBR = derX[:,:,0]*derX[:,:,2] + derY[:,:,0]*derY[:,:,2]
    prodGR = derX[:,:,1]*derX[:,:,2] + derY[:,:,1]*derY[:,:,2]
    
    #Calcular normas
    normBG = normT[:,:,0]*normT[:,:,1]
    normBR = normT[:,:,0]*normT[:,:,2]
    normGR = normT[:,:,1]*normT[:,:,2]
    
    lsProd = [prodBG,prodBR,prodGR]
    lsNorm = [normBG,normBR,normGR]
    
    cout=0
    ban=0
    
    imgANG= np.zeros((n,m))
    true=0
    for i in range(n):
        for j in range(m):
            
            lsAng=[]
            promPX=0
            
            #Angulos de cada canal
            for k in range(C):
                matProd=lsProd[k]
                prod = matProd[i][j]
                matNorm = lsNorm[k]
                norm = matNorm[i][j]
                #print('norm=',norm)           
                
                if(norm!=0):
                    arg = prod/norm
                    
                    if(abs(arg)>=1):
                        ang=0
                        
                    else:
                        ang=math.acos(arg)
                        #if(ban==1):
                            #print('arg',arg)
                            #print('angAcos',ang)
                            #ban=0
                    #if(prod<0):#Angulo obtuso
                        #cout+=1
                        #print('prod ij',prod,i,j)
                        #ban=1
                        #ang=np.pi-ang
                    #print('ang',ang)
                else:
                    ang=0
                   
                
                lsAng.append(ang)
            #print('ang0',lsAng)
            promPX=np.mean(np.array(lsAng))
            #print('promPX',promPX)
            
            imgANG[i][j]=promPX
    
    #print('contador',cout)
    #print('true',true)
    
    return imgANG


def printVect(i,j,lsBestImg):
    lsImgVec = []

    color = [(255,0,0),(0,255,0),(0,0,255)]
    fact=50

    for imgParche in lsBestImg:

        imgObj = imgParche.copy()
        derX = imgObj[:,:-1,:].astype(int) - imgObj[:,1:,:].astype(int)
        derY = imgObj[:-1,:,:].astype(int) - imgObj[1:,:,:].astype(int)

        for k in range(3):
            capa = k

            (v1,v2) = (derX[i,j,capa],derY[i,j,capa])
            norm = np.sqrt(v1**2+v2**2)
            if norm!=0:
                (v1u,v2u) = (fact*v1//norm,fact*v2//norm)
            else:
                (v1u,v2u) = (fact*v1,fact*v2)
            imgOut = cv2.arrowedLine(imgObj, (j,i), (int(j+v2u),int(i+v1u)), color[k], 2,  cv2.LINE_AA)#, 0, 0.5)#cv2.LINE_4)#
            print((i, j), (int(i+v1u), int(j+v2u)))

        lsImgVec.append(imgObj)
    return lsImgVec


# In[3]:


### DIP - VTV - GradRimm - Dirac
lsNumPrue = ['361','362B','440C','350B']
lambAsLS = ['0','0.00025','0.00025','0.0001']
typeP = 'debU'

lsData=[]

for k in range(1,25):

    kodim_num_i = k

    print('Imagen ', kodim_num_i)
    # In[4]:


    lsBestImg = []
        
    ruta = './IMGkodim'+ str(kodim_num_i) +'/'

    #os.makedirs(ruta)

    ruta = './data_deblurring_thomas/deblurring/' 
    
    #original
    namefig = ruta + 'kodim'+ str(kodim_num_i) + '_small.png'
    imgOut = cv2.imread(namefig)
    lsBestImg.append(imgOut)
    #filename = ruta + 'kodim'+ str(kodim_num_i) +'_small.png'
    #cv2.imwrite(filename, imgOut)
	#denoisig25
    namefig = ruta +'kodim'+ str(kodim_num_i) + '_small_blurred_uniform.png'
    imgOut = cv2.imread(namefig)
    lsBestImg.append(imgOut)
    #filename = ruta + 'kodim'+ str(kodim_num_i) + '_small_blurred_uniform.png'
    #cv2.imwrite(filename, imgOut)

    for Nprueb,lambAs in zip(lsNumPrue,lambAsLS):
        
        ruta = './'+ Nprueb +'/BestPnsr'+ typeP + '_P'+ Nprueb +'lamb'+ lambAs +'/'
        namefig = ruta + 'BestPnsr'+ typeP +'_P'+ Nprueb +'_'+ lambAs + 'kodim'+ str(kodim_num_i) +'_small.png'
        imgOut = cv2.imread(namefig)
        lsBestImg.append(imgOut)
        
        filename = ruta + 'BestIMGP'+ Nprueb +'_'+ lambAs + 'kodim'+ str(kodim_num_i) +'.png'
        cv2.imwrite(filename, imgOut)


    #In[5]:

    lsNames = ['Downsampling','Downsampling + Blur','DIP', 'DIP-VTV',' DIP-RiemannVTV', 'DIP-DiracVTV$^{NL}$']


    # In[10]:

    lsValPSNR = []

    for i in range(6):
        val = compare_psnr(lsBestImg[0],lsBestImg[i])
        lsValPSNR.append(val)


    # In[11]:


    print('First valPSNR',lsValPSNR[1:])
    print('valMax',max(lsValPSNR[1:]),'pos',np.argmax(lsValPSNR[1:])+1)



    # In[13]:


    #round(np.array(lsValPSNR),3)
    num="{:.2f}".format(lsValPSNR[0])


    # In[14]:


    lsValPSNR[0] = 0



    titulo = './OUTIMG/ComImg'+typeP +'KODIMnum' + str(kodim_num_i) + '.png'
    printImgs(lsBestImg,lsNames,lsValPSNR, 2, 3, titulo)


    # In[17]:


    lsImgDif = []
    lsGris = []
    for i in range(6):
        imgDif = lsBestImg[0] - lsBestImg[i]
        lsImgDif.append(imgDif)
        
        ImgGris = cv2.cvtColor(imgDif, cv2.COLOR_BGR2GRAY)
        lsGris.append(ImgGris)


    # In[18]:


    titulo = './OUTIMG/ComImgDifColor'+typeP +'KODIMnum' + str(kodim_num_i) + '.png'
    printImgs(lsImgDif,lsNames,lsValPSNR , 2, 3, titulo)


    # In[19]:


    titulo = './OUTIMG/ComImgDifGris'+typeP +'KODIMnum' + str(kodim_num_i) + '.png'
    printImgs(lsGris,lsNames,lsValPSNR, 2, 3, titulo)


    # In[20]:


    lsValPSNR = []

    for i in range(6):
        val = compare_psnr(lsBestImg[0],lsBestImg[i])
        lsValPSNR.append(val)


    # In[21]:


    print(lsValPSNR[1:])
    print(max(lsValPSNR[1:]))
    print(np.argmax(lsValPSNR[1:])+1)


    # In[22]:


    (a_y,a_x) = (180,75)
    (b_y,b_x) = (250,120)
    (c_y,c_x) = (50,200)
    (d_y,d_x) = (150,350)


    # In[23]:


    (a_y,a_x) = (a_y//2,a_x//2)
    (b_y,b_x) = (b_y//2,b_x//2) 
    (c_y,c_x) = (c_y//2,c_x//2)
    (d_y,d_x) = (d_y//2,d_x//2)


    # In[24]:


    lsBestImg[0].shape


    # In[25]:


    lsImgRec =[]
    lscrtG =[]
    lscrtR =[]

    for i in range(len(lsBestImg)):
        imgRec = lsBestImg[i].copy()
        rec1 = cv2.rectangle(imgRec,(a_y,a_x),(b_y,b_x),(0,255,0),2)
        rec2 = cv2.rectangle(imgRec,(c_y,c_x),(d_y,d_x),(255,0,0),2)
        lsImgRec.append(imgRec)
        
        cort1G = imgRec[a_x+1:b_x,a_y+1:b_y,:].copy()
        lscrtG.append(cort1G)
        cort1R = imgRec[c_x+1:d_x,c_y+1:d_y,:].copy()
        lscrtR.append(cort1R)


    # In[26]:


    titulo = './OUTIMG/ComImgREC'+typeP +'KODIMnum' + str(kodim_num_i) + '.png'
    printImgs(lsImgRec,lsNames,lsValPSNR, 2, 3, titulo)


    # In[27]:


    lsValPSNR = []

    for i in range(6):
        val = compare_psnr(lscrtG[0],lscrtG[i])
        lsValPSNR.append(val)


    # In[28]:


    print(lsValPSNR[1:])
    print(max(lsValPSNR[1:]))
    print(np.argmax(lsValPSNR[1:])+1)


    # In[29]:


    titulo = './OUTIMG/ComImgRectG'+typeP +'KODIMnum' + str(kodim_num_i) + '.png'

    printImgs(lscrtG,lsNames,lsValPSNR, 2, 3, titulo)


    # In[30]:


    lsValPSNR = []

    for i in range(6):
        val = compare_psnr(lscrtR[0],lscrtR[i])
        lsValPSNR.append(val)


    # In[31]:


    print(lsValPSNR[1:])
    print(max(lsValPSNR[1:]))
    print(np.argmax(lsValPSNR[1:])+1)


    # In[32]:


    titulo = './OUTIMG/ComImgRectR'+typeP +'KODIMnum' + str(kodim_num_i) + '.png'

    printImgs(lscrtR,lsNames,lsValPSNR, 2, 3, titulo)


    # In[33]:


    imgDIP = lsBestImg[2].copy()#lscrtR[2].copy()


    # In[34]:


    imgDIP.shape


    # In[35]:


    derX = imgDIP[:,:-1,:] - imgDIP[:,1:,:]
    derY = imgDIP[:-1,:,:] - imgDIP[1:,:,:]


    # In[36]:


    plt.imshow(derX[:,:,0], 'gray')


    # In[37]:


    plt.imshow(derY[:,:,0], 'gray')

    # In[46]:

    imgANG = calcularAng(lscrtG[0])

    # In[47]:

    ax = plt.subplot()
    im = ax.imshow(imgANG)#,'gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    plt.show()

    title= './OUTIMG/parche'+ str(kodim_num_i) + '.png'
    plt.savefig(title)  

    # In[50]:


    np.mean(imgANG)


    # In[51]:

    lsImgAng=[]
    lsAngProm=[]

    for img in lsBestImg:#lscrtG:
        imgOut = calcularAng(img)
        lsImgAng.append(imgOut)
        prom=np.mean(imgOut)
        lsAngProm.append(prom)
        plt.imshow(imgOut)


    # In[52]:


    lsAngProm


    # In[53]:


    plt.imshow(lsImgAng[1])


    # In[54]:


    titulo = './outAng/ImgAng'+typeP +'KODIMnum' + str(kodim_num_i) + '.png'

    printAng(lsImgAng,lsNames,lsAngProm, 2, 3, titulo)

    # In[57]:
    # In[58]:


    lsImgVec = lsBestImg.copy()
    n,m,C = np.shape(lsImgVec[0])


    # In[59]:

    ls = []
    for i in range(3):
        x = random.randint(0, n-5)
        y = random.randint(0, m-5)
        ls.append((x,y))
    print(ls)
    for (x,y) in ls: #ls:#  
        lsImgVec = printVect(x,y,lsImgVec)


    # In[60]:


    lsValPSNR = []

    for i in range(6):
        val = compare_psnr(lsBestImg[0],lsBestImg[i])
        lsValPSNR.append(val)


    # In[61]:


    print(lsValPSNR[1:])
    print(max(lsValPSNR[1:]))
    print(np.argmax(lsValPSNR[1:])+1)


    # In[65]:
    #lsNames2 = ['-','Original','Downsampling','Downsampling + Blur','DIP', 'DIP-VTV',' DIP-RiemannVTV', 'DIP-DiracVTV$^{NL}$']
    lsNames2 = ['Downsampling','Downsampling + Blur','DIP', 'DIP-VTV',' DIP-RiemannVTV', 'DIP-DiracVTV$^{NL}$']


    # In[66]:


    titulo = './OUTIMG/ComImgVect'+typeP +'KODIMnum' + str(kodim_num_i) + '.png'

    printImgs(lsImgVec,lsNames2,lsValPSNR, 2, 3, titulo)
    #printImgs(lsImg34,lsNames2,lsValPSNR2, 2, 4, titulo,2)


    # In[67]:


    titulo = './OUTIMG/ComImgVectRedu'+typeP +'KODIMnum' + str(kodim_num_i) + '.png'

    printImgs(lsImgVec[2:],lsNames[2:],lsValPSNR[2:], 1, 3, titulo,-1)

    
    lsAngProm.insert(0,kodim_num_i)
	
    lsData.append(lsAngProm)

# In[50]:

datAng = np.array(lsData)
fila_prom = np.mean(datAng,axis=0)
datAng = np.concatenate((datAng,[fila_prom]),axis= 0)
#print(datAng.shape)
fname = 'lsAngProm'+ typeP +'.csv'
TAB= pd.DataFrame(datAng)
TAB[0] = TAB[0].astype('int64')
TAB.to_csv(fname, sep=';', decimal=',',header=None,index=False,float_format='%.5f')
fname = 'lsAngPromTotal'+ typeP +'.csv'
np.savetxt(fname, datAng, delimiter =",",fmt ='% s')



def pritnTab1(Data,k):
    #n = Data.shape[0]
    for i in range(k):
        print('kodim' + str(i+1), ' & ' ,"{:.4f}".format(Data[i,1]) , ' & ', "{:.4f}".format(Data[i,2]) ,' & ', "{:.4f}".format(Data[i,3]),' & ', "{:.4f}".format(Data[i,4]) ,' & ', "{:.4f}".format(Data[i,5]),' & ', "{:.4f}".format(Data[i,6]),' \\\\' )

pritnTab1(datAng,25)

