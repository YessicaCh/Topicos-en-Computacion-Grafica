import numpy as np
import time
from PIL import Image
import math

#horizontal y filtros como sobel

Rx = [  0,0,0,
        0,-1,0,
        0,0,1  ]
Ry = [  0,0,0,
        0,0,-1,
        0,1,0]
Sx = [  1,0,-1,
        2,0,-2,
        1,0,-1  ]
Sy = [  -1,-2,-1,
        0,0,0,
        1,2,1  ]
L = [  1,1,1,
        1,-8,1,
        1,1,1]
L = [  0,1,0,
        1,-4,1,
        0,1,0]


H = [ -1,-1,-1,
       2,2,2,
       -1,-1,-1
    ]
H45 = [ -1,-1,2,
       -1,2,-1,
       2,-1,-1
    ]
V = [ -1,2,-1,
      -1,2,-1,
      -1,2,-1
    ]
V45 = [ 2,-1,-1,
       -1,2,-1,
       -1,-1,2
    ]

G = [  1,2,1,
        2,4,2,
        1,2,1]

def sobel(img):
    tam = 3
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            list=[]
            sumSx = 0
            sumSy = 0
            for z in range(0,3):
                for y in range(0,3):
                    if(i+z<height and j+y<width):
                        sumSx += img[i+z][j+y]*Sx[z*3+y]
                        sumSy += img[i+z][j+y]*Sy[z*3+y]

            prom=math.sqrt(pow(sumSx,2)+pow(sumSy,2))
            if(i+3<height and j+3<width):
                if prom<0:
                    M[i+var][j+vary]=0
                else:
                    M[i+var][j+vary]=prom

    img1 = Image.fromarray(M)
    img1.save('resultados/sobel.png')
    img1.show()
    return img1

def sobelAbs(img):
    tam = 3
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            list=[]
            sumSx = 0
            sumSy = 0
            for z in range(0,3):
                for y in range(0,3):
                    if(i+z<height and j+y<width):
                        sumSx += img[i+z][j+y]*Sx[z*3+y]
                        sumSy += img[i+z][j+y]*Sy[z*3+y]

            prom=abs(sumSx)+abs(sumSy)
            if(i+3<height and j+3<width):
                if prom<0:
                    M[i+var][j+vary]=0
                else:
                    M[i+var][j+vary]=prom
    img1 = Image.fromarray(M)
    img1.save('resultados/sobelAbs.png')
    img1.show()
    return img1

def roberts(img):
    tam = 3
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            list=[]
            sumRx = 0
            sumRy = 0
            for z in range(0,3):
                for y in range(0,3):
                    if(i+z<height and j+y<width):
                        sumRx += img[i+z][j+y]*Rx[z*3+y]
                        sumRy += img[i+z][j+y]*Ry[z*3+y]
            prom=math.sqrt(pow(sumRx,2)+pow(sumRy,2))
            if(i+3<height and j+3<width):
                M[i+var][j+vary]=prom
    img1 = Image.fromarray(M)
    img1.save('resultados/roberts.png')
    img1.show()
    return img1


def laplace( img):
    tam=3
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            sum=0
            for z in range(0,tam):
                for y in range(0,tam):
                    if(i+z<height and j+y<width):
                        sum+=img[i+z][j+y]*L[z*3+y]
            sum=sum#/(9)
            if(i+tam<height and j+tam<width):
                if sum<0:
                    M[i+var][j+vary]=0
                else:
                    M[i+var][j+vary]=sum
    img1 = Image.fromarray(M)
    img1.save('resultados/laplace.png')
    img1.show()
    return img1




def media(tam, img):
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            sum=0
            for z in range(0,tam):
                for y in range(0,tam):
                    if(i+z<height and j+y<width):
                        sum+=img[i+z][j+y]

            sum=sum/(tam*tam)
            if(i+tam<height and j+tam<width):
                M[i+var][j+vary]=sum
    img1 = Image.fromarray(M)
    img1.save('resultados/media.png')
    img1.show()
    return img1

def mediaPonderada(tam, img):
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            sum=0
            for z in range(0,tam):
                for y in range(0,tam):
                    if(i+z<height and j+y<width):
                        temp=img[i+z][j+y]
                        if(y==z):
                            temp=temp*10
                        sum+=temp
            sum=sum/(tam*tam)
            if(i+tam<height and j+tam<width):
                M[i+var][j+vary]=sum
    img1 = Image.fromarray(M)
    img1.save('resultados/medianaPonderada.png')
    img1.show()
    return img1

def mediana(tam, img):
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=int((tam*tam)/2)
    var=int(var/tam)
    vary=int(var%tam)
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            list=[]
            for z in range(0,tam):
                for y in range(0,tam):
                    if(i+z<height and j+y<width):
                        list.append(img[i+z][j+y])
            list.sort()
            mid=int(len(list)/2)
            medio=list[mid]
            if(i+tam<height and j+tam<width):
                M[i+var][j+vary]=medio
    img1 = Image.fromarray(M)
    img1.save('resultados/mediana.png')
    img1.show()
    return img1


def suma(img,img2):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    B = np.asarray( img2 )

    for i in range(0,Ah):
        for j in range(0,Aw):
            sum=(A[i][j]+B[i][j])%255
            NI[i][j]=sum
    print (NI)
    img1 = Image.fromarray(NI)
    #img.save('my.png')
    #img1.show()
    return img1



def binarizar(img):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    for i in range(0,Ah):
        for j in range(0,Aw):
            if A[i][j]>255/2:
                NI[i][j]=255
            else:
                NI[i][j]=0
    img1 = Image.fromarray(NI)
    #img.save('my.png')
    img1.show()
    return img1

def horizontal( img1,VV,n):
    tam=3
    height = np.size(img1, 0)
    width = np.size(img1, 1)
    print(height)
    print(width)
    M = np.zeros_like(img1, np.uint8)
    img2 = np.asarray( img1 )

    for i in range(0,height):
        for j in range(0,width):
            #M[i][j]=img[i][j]
            sum=0
            for z in range(0,tam):
                for y in range(0,tam):
                    if(i+z<height and j+y<width):
                        sum+=img2[i+z][j+y]*VV[z*3+y]

            if(i+3<height and j+3<width):
                if sum<0:
                    M[i+1][j+1]=0
                else:
                    M[i+1][j+1]=sum

    img4 = Image.fromarray(M)
    img4.save("resultados/"+n+".png")
    return img4


def gauss( img):
    tam=3
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            sum=0
            for z in range(0,tam):
                for y in range(0,tam):
                    if(i+z<height and j+y<width):
                        sum+=img[i+z][j+y]*G[z*3+y]
            sum=sum/(16)
            if(i+tam<height and j+tam<width):
                if sum<0:
                    M[i+var][j+vary]=0
                else:
                    M[i+var][j+vary]=sum
    img1 = Image.fromarray(M)
    img1.save('resultados/gauss.png')
    img1.show()
    return img1

if __name__ == '__main__':
        #image1 = readImage('images/cameraman.jpg')

        img = Image.open('Images/cameraman.jpg').convert('L')
        """I=media(11,img)
        P=mediaPonderada(11,img)
        Z=mediana(11,img)
        S=sobel(img)
        A=sobelAbs(img)
        R=roberts(img)
        L=laplace(img)
        """#gauss(img)

        img=binarizar(img)
        Hi=horizontal(img,H,"1")
        H45i=horizontal(img,H45,"2")
        Vi=horizontal(img,V,"3")
        V45i=horizontal(img,V45,"4")
        Q=suma(Hi,H45i)
        W=suma(Q,Vi)
        H=suma(W,V45i)
        H.show()
        img1=binarizar(H)
        img1.show()
        
