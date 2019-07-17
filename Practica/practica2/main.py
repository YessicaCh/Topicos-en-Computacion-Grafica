import matplotlib.pyplot as plt
import numpy as np
import math 
import cv2


    
def equalization(img):
    LevelGray = 256
    M,N = img.shape
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    plt.plot(hist, color='gray' )
    plt.xlabel('Pixeles')
    plt.ylabel('Histogram')
    plt.show()
    
    Sk = []
    size=N*M
    sum=0
    for i in range(0,len(hist)):
        sum+=(hist[i]/size)
        to_add = sum*256
        Sk.append(int(round(to_add,0)))
    plt.plot(Sk, color='gray' )
    plt.xlabel('Pixeles ')
    plt.ylabel('Histogram equalizacion')
    plt.show()

    new_img = img.copy()
    for i in range(0, M):
        for j in range(0, N):
            new_img[i][j] = Sk[img[i][j]]
    return new_img
    cv2.imshow('Img Equalizate', new_img)                                     
    cv2.waitKey(0)


#Filtros 

def media(img,mask):
    rows,cols=img.shape
    new_image = img.copy()
    dif=int(mask/2)
    for i in range(dif, rows-dif):
        for j in range(dif, cols-dif):
            cont=0
            sum=0
            for x in range(i-dif,i+dif+1):
                for y in range(j-dif,j+dif+1):
                    sum+=int(img[x][y])
                    cont+=1
            new_image[i][j] = int(sum/cont)  #(int(img[i-1][j-1]) + int(img[i][j-1]) + int(img[i+1][j-1]) + int(img[i-1][j]) + int(img[i][j]) + int(img[i+1][j]) + int(img[i-1][j+1]) + int(img[i][j+1]) + int(img[i+1][j+1]))/9
    cv2.imshow('Media', new_image)
    cv2.waitKey()
def mediaPonderada(img,mask):
    rows,cols=img.shape
    new_image = img.copy()
    dif=int(mask/2)
    for i in range(dif, rows-dif):
        for j in range(dif, cols-dif):
            cont=0
            sum=0
            for x in range(i-dif,i+dif+1):
                for y in range(j-dif,j+dif+1):
                    sum+=int(img[x][y])
                    cont+=1
            sum+=int(img[i][j])
            new_image[i][j] = int(sum/cont+1)  #(int(img[i-1][j-1]) + int(img[i][j-1]) + int(img[i+1][j-1]) + int(img[i-1][j]) + int(img[i][j]) + int(img[i+1][j]) + int(img[i-1][j+1]) + int(img[i][j+1]) + int(img[i+1][j+1]))/9
    cv2.imshow('MediaPonderada', new_image)
    cv2.waitKey()
def Mediana(img,mask):
    rows,cols=img.shape
    new_image = img.copy()
    dif=int(mask/2)
    for i in range(dif, rows-dif):
        for j in range(dif, cols-dif):
            lista=[]
            for x in range(i-dif,i+dif+1):
                for y in range(j-dif,j+dif+1):
                    lista.append(int(img[x][y]))
            lista.sort()
            new_image[i][j]=lista[int((mask*mask)/2)]
    cv2.imshow('Mediana', new_image)
    cv2.waitKey()
def Max(img,mask):
    rows,cols=img.shape
    new_image = img.copy()
    dif=int(mask/2)
    for i in range(dif, rows-dif):
        for j in range(dif, cols-dif):
            lista=[]
            for x in range(i-dif,i+dif+1):
                for y in range(j-dif,j+dif+1):
                    lista.append(int(img[x][y]))
            lista.sort()
            new_image[i][j]=lista[(mask*mask)-1]
    cv2.imshow('Max', new_image)
    cv2.waitKey()
def Min(img,mask):
    rows,cols=img.shape
    new_image = img.copy()
    dif=int(mask/2)
    for i in range(dif, rows-dif):
        for j in range(dif, cols-dif):
            lista=[]
            for x in range(i-dif,i+dif+1):
                for y in range(j-dif,j+dif+1):
                    lista.append(int(img[x][y]))
            lista.sort()
            new_image[i][j]=lista[0]
    cv2.imshow('Min', new_image)
    cv2.waitKey()
# def Max(img):
#     rows,cols=img.shape
#     new_image = img.copy()
   
#     for i in range(1, rows-1):
#         for j in range(1, cols-1):
#             lista=[]
#             lista.append(int(img[i-1][j-1]))
#             lista.append(int(img[i-1][j]))
#             lista.append(int(img[i-1][j+1]))
#             lista.append(int(img[i][j-1]))
#             lista.append(int(img[i][j]))
#             lista.append(int(img[i][j+1]))

#             lista.append(int(img[i+1][j+1]))
#             lista.append(int(img[i+1][j]))
#             lista.append(int(img[i+1][j+1]))
#             lista.sort()
#             new_image[i][j]=lista[8]
#     cv2.imshow('Max', new_image)
#     cv2.waitKey()
# def Min(img):
#     rows,cols=img.shape
#     new_image = img.copy()
   
#     for i in range(1, rows-1):
#         for j in range(1, cols-1):
#             lista=[]
#             lista.append(int(img[i-1][j-1]))
#             lista.append(int(img[i-1][j]))
#             lista.append(int(img[i-1][j+1]))
#             lista.append(int(img[i][j-1]))
#             lista.append(int(img[i][j]))
#             lista.append(int(img[i][j+1]))

#             lista.append(int(img[i+1][j+1]))
#             lista.append(int(img[i+1][j]))
#             lista.append(int(img[i+1][j+1]))
#             lista.sort()
#             new_image[i][j]=lista[0]
#     cv2.imshow('Min', new_image)
#     cv2.waitKey()
# vamos a probar 
def Gaus(img): #para borrar ruido 
    rows,cols=img.shape
    new_image = img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            new_image[i][j] = (int(img[i-1][j-1]) + 2*int(img[i-1][j]) + int(img[i-1][j+1]) + 
                               2*int(img[i][j-1]) + 4*int(img[i][j]) + 2*int(img[i][j+1]) + 
                               int(img[i+1][j+1]) + 2*int(img[i+1][j]) + int(img[i+1][j+1]))/16
    
    cv2.imshow('Gaus', new_image)
    cv2.waitKey()
    
def laplace(img):
    rows,cols=img.shape
    new_image = img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            new_image[i][j] = (0*int(img[i-1][j-1]) + 1*int(img[i-1][j]) +  0*int(img[i-1][j+1]) + 
                               1*int(img[i][j-1])   + -4*int(img[i][j])  +  1*int(img[i][j+1]) + 
                               0*int(img[i+1][j+1]) + 1*int(img[i+1][j]) +  0* int(img[i+1][j+1]))
    
    cv2.imshow('Laplace', new_image)
    cv2.waitKey()
def Roberts(img):
    rows,cols=img.shape
    new_image = img.copy()
    Gx_img=img.copy()
    Gy_img=img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            Gx_img[i][j]=(-1*int(img[i-1][j-1])+int(img[i][j]))
            Gy_img[i][j]=(1*int(img[i-1][j])+int(img[i][j-1]))
    cv2.imshow('RobertGX', Gx_img)
    cv2.waitKey()
    cv2.imshow('RobertGY', Gy_img)
    new_image = Gx_img+Gy_img
    cv2.imshow('Roberts', new_image)
    cv2.waitKey()


if __name__=='__main__':
    
    img = cv2.imread('PruebasImagenes/lenaS.png', cv2.IMREAD_GRAYSCALE)      # cargar el archivo PNG indicado
    cv2.imshow('Img Origin', img)                                     
    cv2.waitKey(0)
    

    # equalization(img)
    Gaus(img)

    # img2 = cv2.imread('PruebasImagenes/lenaG.png', cv2.IMREAD_GRAYSCALE)      # cargar el archivo PNG indicado
    # cv2.imshow('Img Origin', img2)                                     
    # cv2.waitKey(0)
    # mediaPonderada(img2,3)

    # img3 = cv2.imread('PruebasImagenes/lenaG.png', cv2.IMREAD_GRAYSCALE)      # cargar el archivo PNG indicado
    # cv2.imshow('Img Origin', img3)                                     
    # cv2.waitKey(0)
    # Mediana(img3,3)

    # Max(img,5)
    # Max(img,5)
# Min(img)
# laplace(img)
# Roberts(img)
