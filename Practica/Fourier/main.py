# 2D Discrete Fourier Transform (DFT) and its inverse
# Warning: Computation is slow so only suitable for thumbnail size images!
# FB - 20150102
from PIL import Image
import cmath
pi2 = cmath.pi * 2.0
import matplotlib.pyplot as plt
import numpy as np
import math 
import cv2

G = [[1, 4, 7, 4, 1],
     [4, 16, 26, 16, 4],
     [7, 26, 41, 26, 7],
     [4, 16, 26, 16, 4],
     [1, 4, 7, 4, 1]]

def preprocesamiento(img):
    global M, N
    (M,N) = img.shape  # (imgx, imgy)
    new_M_=M*2
    new_N_=N*2
    copy_img = img.copy()
    new_img = np.resize(img,(int(new_M_),int(new_N_)))

    for x in range(0,new_M_):
        for y in range(0,new_N_):
            if x<N and y<M:
                new_img[x][y]=img[x][y]
            else :
                new_img[x][y]=0
    cv2.imshow('Double SIZE', new_img)                                     
    cv2.waitKey(0)
    return new_img 

def DFT2D(image):
    global M, N
    (M, N) = image.size  # (imgx, imgy)
    dft2d_gray = [[0.0 for k in range(M)] for l in range(N)]
    """dft2d_grn = [[0.0 for k in range(M)] for l in range(N)]
    dft2d_blu = [[0.0 for k in range(M)] for l in range(N)]"""
    pixels = image.load()
    #print(pixels[0,0])
    for k in range(M):
        for l in range(N):
            sum_gray = 0.0
            #sum_grn = 0.0
            #sum_blu = 0.0
            for m in range(M):
                for n in range(N):
                    #print('m', m, n)
                    try:
                        (gray) = pixels[m, n]             
                    except IndexError:
                        (gray) = 0
                    e = cmath.exp(- 1j * pi2 *
                                  (float(k * m) / M + float(l * n) / N))
                    sum_gray += e * gray
                    #sum_grn += grn * e
                    #sum_blu += blu * e
            dft2d_gray[l][k] = sum_gray / M / N
            """dft2d_grn[l][k] = sum_grn / M / N
            dft2d_blu[l][k] = sum_blu / M / N"""
    return (dft2d_gray)  # , dft2d_grn, dft2d_blu)


def IDFT2D(dft2d):
    #(dft2d_red, dft2d_grn, dft2d_blu) = dft2d
    #dft2d
    global M, N
    #sum_grn = 0.0
    image = Image.new("L", (M, N))
    pixels = image.load()
    for m in range(M):
        for n in range(N):
            sum_gray = 0.0
            #sum_grn = 0.0
            #sum_blu = 0.0
            for k in range(M):
                for l in range(N):
                    e = cmath.exp(
                        1j * pi2 * (float(k * m) / M + float(l * n) / N))
                    sum_gray += dft2d[l][k] * e
                    #sum_grn += dft2d_grn[l][k] * e
                    #sum_blu += dft2d_blu[l][k] * e
            gray = int(sum_gray.real + 0.5)
            #grn = int(sum_grn.real + 0.5)
            #blu = int(sum_blu.real + 0.5)
            pixels[m, n] = (gray)  # , grn, blu)
    image.show()
    return image

if __name__=='__main__':
    
    img = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE)      # cargar el archivo PNG indicado
    # cv2.imshow('Img Origin', img)                                     
    # cv2.waitKey(0)
    preprocesamiento(img)
#     # equalization(img)
#     Gaus(img)

# # TEST
# # Recreate input image from 2D DFT results to compare to input image
# Pimg = Image.open("cameraman.jpg").convert('L')
# preprocesamiento(Pimg)
# image = IDFT2D( DFT2D(Pimg) )
# image.save("output3.png", "PNG")