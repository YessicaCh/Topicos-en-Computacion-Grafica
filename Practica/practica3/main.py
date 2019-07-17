import matplotlib.pyplot as plt
import numpy as np
import math 
import cv2


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
            sum = (0*int(img[i-1][j-1]) + 1*int(img[i-1][j]) +  0*int(img[i-1][j+1]) + 
                               1*int(img[i][j-1])   + -4*int(img[i][j])  +  1*int(img[i][j+1]) + 
                               0*int(img[i+1][j+1]) + 1*int(img[i+1][j]) +  0* int(img[i+1][j+1]))
            if sum < 0:
                new_image[i][j] = 0
            elif sum > 255:
                new_image[i][j] = 255
            else:
                new_image[i][j] = sum
    cv2.imshow('Laplace', new_image)
    cv2.waitKey()

def Sobel(img):
    rows,cols=img.shape
    new_image = img.copy()
    Gx_img=img.copy()
    Gy_img=img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            x=(-1*int(img[i-1][j-1]) + 0*int(img[i-1][j]) +  1*int(img[i-1][j+1]) + 
                          -2*int(img[i][j-1])   + 0*int(img[i][j])   +  2*int(img[i][j+1]) + 
                          -1*int(img[i+1][j+1]) + 0*int(img[i+1][j]) +  1* int(img[i+1][j+1]))
            if x < 0 :
                Gx_img[i][j] = 0
            elif x > 255 :
                 Gx_img[i][j] = 255
            else:
                Gx_img[i][j] = x
            y=(-1*int(img[i-1][j-1]) + -2*int(img[i-1][j]) + -1*int(img[i-1][j+1]) + 
                           0*int(img[i][j-1])   + 0*int(img[i][j])   +  0*int(img[i][j+1]) + 
                           1*int(img[i+1][j+1]) + 2*int(img[i+1][j]) +  1* int(img[i+1][j+1]))
            if y < 0 :
                Gy_img[i][j] = 0
            elif y > 255 :
                 Gy_img[i][j] = 255
            else:
                Gy_img[i][j] = y
    # cv2.imshow('SobelGX', Gx_img)
    # cv2.waitKey()
    # cv2.imshow('SobelGY', Gy_img)
    for i in range(0, rows-1):
        for j in range(0, cols-1):
            sum=Gx_img[i][j] + Gy_img[i][j]
            if sum < 0 :
                new_image[i][j] = 0
            elif sum > 255 :
                 new_image[i][j] = 255
            else:
                new_image[i][j] = sum
    cv2.imshow('Sobel', new_image)
    cv2.waitKey()

def Prewitt(img):
    rows,cols=img.shape
    new_image = img.copy()
    Gx_img=img.copy()
    Gy_img=img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            x=(-1*int(img[i-1][j-1]) + 0*int(img[i-1][j]) +  1*int(img[i-1][j+1]) + 
                          -1*int(img[i][j-1])   + 0*int(img[i][j])   +  1*int(img[i][j+1]) + 
                          -1*int(img[i+1][j+1]) + 0*int(img[i+1][j]) +  1* int(img[i+1][j+1]))
            if x<0:
                Gx_img[i][j]=0
            elif x > 255 :
                Gx_img[i][j]= 255
            else:
                Gx_img[i][j]= x


            y=(-1*int(img[i-1][j-1]) + -1*int(img[i-1][j]) + -1*int(img[i-1][j+1]) + 
                           0*int(img[i][j-1])   + 0*int(img[i][j])   +  0*int(img[i][j+1]) + 
                           1*int(img[i+1][j+1]) + 1*int(img[i+1][j]) +  1* int(img[i+1][j+1]))
            if y<0:
                Gy_img[i][j]=0
            elif y > 255 :
                Gy_img[i][j]= 255
            else:
                Gy_img[i][j]= y

    # cv2.imshow('PrewittGX', Gx_img)
    # cv2.waitKey()
    # cv2.imshow('PrewittGY', Gy_img)

    for i in range(0, rows-1):
        for j in range(0, cols-1):
            sum=0
            sum=Gx_img[i][j]+Gy_img[i][j]
            if sum < 0 :
                new_image[i][j] = 0
            elif sum > 255 :
                new_image[i][j] = 255
            else:
                new_image[i][j] = sum
    cv2.imshow('Prewitt', new_image)
    cv2.waitKey()

def Roberts(img):
    rows,cols=img.shape
    new_image = img.copy()
    Gx_img=img.copy()
    Gy_img=img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            x=(int(img[i-1][j-1]) - int(img[i][j]))
            if x<0:
                Gx_img[i][j]=0
            elif x > 255 :
                Gx_img[i][j]= 255
            else:
                Gx_img[i][j]= x


            y=int(int(img[i-1][j]) - int(img[i][j-1]))
            if y<0:
                Gy_img[i][j]=0
            elif y > 255 :
                Gy_img[i][j]= 255
            else:
                Gy_img[i][j]= y

    # cv2.imshow('PrewittGX', Gx_img)
    # cv2.waitKey()
    # cv2.imshow('PrewittGY', Gy_img)

    for i in range(0, rows-1):
        for j in range(0, cols-1):
            sum=0
            sum=Gx_img[i][j]+Gy_img[i][j]
            if sum < 0 :
                new_image[i][j] = 0
            elif sum > 255 :
                new_image[i][j] = 255
            else:
                new_image[i][j] = sum
    cv2.imshow('Roberts', new_image)
    cv2.waitKey()

if __name__=='__main__':
    
    img = cv2.imread('PruebasImagenes/church1.jpg', cv2.IMREAD_GRAYSCALE)      # cargar el archivo PNG indicado
    cv2.imshow('Img Origin', img)                                     
    cv2.waitKey(0)
    
    #Gaus(img)
    #laplace(img)
    Roberts(img)
   # Sobel(img)
    #Prewitt(img)