import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import math
from PIL import Image as PImage

#histograma  y funciones (negativa, logaritmica, potencial)
#convertir rgb to hsi

def histograma_Gris(img,height,width):

    color =[]
    for i in range(0,256):
        color.append(0)

    for i in range (0,height-1):
        for j in range (0,width-1):
            color[img[i][j]]+=1

    hist = color
    plt.plot(hist, color='gray' )
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.show()
    return color

def histograma_Color():
    img_ = PImage.open('imagenes/mandrill.jpg')
    img = np.asarray(img_)

    height = np.size(img, 0)
    width = np.size(img, 1)

    color =[]
    for i in range(0,256):
        color.append(0)

    cl = ['r','g','b']
    ht=[]
    for c in range(0,3):
        for i in range (0,height):
            for j in range (0,width):
                color[img[i][j][c]]+=1
        hist=color

        ht.append(hist)
        plt.plot(hist, color = cl[c])
        plt.xlim([0,256])
        plt.show()
    return color

#equalizador del histograma
def equalizador(color,height,width):
    rK=[]
    prrK=[]
    sk=[]
    colorN=[]
    sum=0
    sumT=0

    for i in range(0,256):
        rK.append(float(i)/float(255))
        colorN.append(0)
        prrK.append(float(color[i])/float((height-1)*(width-1)))
        sum+=prrK[i]
        sk.append(sum)
    #ver a cual se parece y redondear
    for i in range(0,256):
        for j in range(0,256):
            if float(sk[i])<float(rK[j]):
                colorN[j]+=color[i]
                break
    sum=0
    for i in colorN:
        sum+=i
    print (sum)
    print (float((height-1)*(width-1)))
    hist = colorN
    plt.plot(hist, color='red' )
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.show()

def rgbTohsi(r,g,b):
    #normalizado
    #r=r/(r+g+b)
    #g=g/(r+g+b)
    #b=b/(r+g+b)

    temp0=(r-g)/2+(r-b)
    #temp1=math.sqrt((r-g)*(r-g)+(r-b)*(g-b))
    temp1=((r-g)*(r-g)+(r-b)*(g-b))
    H=math.acos(temp0/temp1)
    H=180/math.pi * H

    I=(r+g+b)/3
    #S=1-(3*min(r,g,b))/(r+g+b)
    S=1-(3*min(r,g,b))/I

    if b > g :
        H=2*math.pi*(-1)*H
    print (float(H))
    print (float(S))
    print (float(I))
    print ("entra")
    return

def hsiTorgb(h,s,i):
    x=i*(1-s)
    if h<2*math.pi/3:
        y = i * (1 + (s * math.cos(h)) / (math.cos(math.pi / 3 - h)))
        z = 3*i-(x+y)
        b = x
        r = y
        g = z
        return
    if h<4*math.pi/3 :
        y = i * (1 + (s * math.cos(h - 2 * math.pi / 3)) / (math.cos(math.pi / 3 - (h  - 2 * math.pi / 3))))
        z = 3 * i - (x + y)
        r = x
        g = y
        b = z
        return
    else:
        y = i*(1+(s*math.cos(h-4*math.pi/3))/(math.cos(math.pi/3-(h-4*math.pi/3))))
        z = 3*i-(x+y)
        r = z
        g = x
        b = y
        print (r)
        print (g)
        print (b)

def logaritmica(c,r):
    if c*np.log10(1+r)<0:
        return 0
    if c*np.log10(1+r)>255:
        return 255
    return c*np.log10(1+r)

def negativa(r):
    if 255-r<0:
        return 0
    return 255-r

def potencial(r,c,n):
    if c*np.power(r,n)<0:
        return 0
    if c*np.power(r,n)>255:
        return 255
    return c*np.power(r,n)

def cmd(r,g,b):
    return

#unir imagen r + b + g
def composicion(img):
    height = np.size(img, 0)

    width = np.size(img, 1)

    N = np.zeros_like(img, np.uint8)
    L = np.zeros_like(img, np.uint8)
    P = np.zeros_like(img, np.uint8)
    C = np.zeros_like(img, np.uint8)

    ##rojo
    for i in range (0,height-1):
        for j in range (0,width-1):
            N[i][j][0]=negativa(img[i][j][0])
            N[i][j][1]=255#255-img[i][j][1]
            N[i][j][2]=255#255-img[i][j][2]
    Ni = Image.fromarray(N)
    Ni.show()
    ##verde
    for i in range (0,height-1):
        for j in range (0,width-1):
            L[i][j][0]=255#255-img[i][j][0]
            L[i][j][1]=negativa(img[i][j][1])
            L[i][j][2]=255#255-img[i][j][2]
    Li = Image.fromarray(L)
    Li.show()
    ##azul
    for i in range (0,height-1):
        for j in range (0,width-1):
            P[i][j][0]=255#255-img[i][j][0]
            P[i][j][1]=255#255-img[i][j][1]
            P[i][j][2]=negativa(img[i][j][2])
    Pi = Image.fromarray(P)
    Pi.show()
    for i in range (0,height-1):
        for j in range (0,width-1):
            C[i][j][0]=N[i][j][0]
            C[i][j][1]=L[i][j][1]
            C[i][j][2]=P[i][j][2]
    #composicion
    Ci = Image.fromarray(C)
    Ci.show()



if __name__ == '__main__':
    #rgbTohsi(239.0,127.0,26.0)
    #hsiTorgb(89.5474432584,0.801020408163,130.666666667)

    img_ = PImage.open('imagenes/lena.png').convert('L')
    img_.show()
    img = np.asarray(img_)

    height = np.size(img, 0)
    width = np.size(img, 1)
    #color=[790,1023,850,656,329,245,122,81]
    color=histograma_Gris(img,height,width)
    equalizador(color,height,width)
    #histograma_Color()

    """"
    img_ = PImage.open('imagenes/lena.png').convert('L')
    img = np.asarray(img_)
    #print img
    #funciones
    height = np.size(img, 0)
    width = np.size(img, 1)
    print height
    print width

    N = np.zeros_like(img, np.uint8)

    for i in range (0,height-1):
        for j in range (0,width-1):
            #N[i][j]=negativa(img[i][j])
            #N[i][j]=logaritmica(50,img[i][j])
            N[i][j]=potencial(img[i][j],1,2)
    print N
    Ni = Image.fromarray(N)
    Ni.show()
    """
