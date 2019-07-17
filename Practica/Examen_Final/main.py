import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math 
import cv2

kernel = np.ones((3,3),np.uint8)

#morfologia: dilatacion erocion opening close con mascara (+, diamante , cuadrado )
def cuadradoMask(n):#numero impar
    cuadrado = np.zeros((n,n),np.uint8)
    for i in range(0,n):
        for j in range(0,n):
            cuadrado[i][j]=255
            #l.append(255)
        #cuadrado.append(l)
    print (cuadrado)
    return cuadrado

def diamanteMask(n):#numero impar
    cuadrado = np.zeros((n,n),np.uint8)
    for i in range(0,n):
        q=int(n/2)
        if i<=q:
            for j in range(q-i,q+i+1):
                cuadrado[i][j]=255
            #cuadrado[i][q+i]=255
        else:
            for x in range(i-q,n-(i-q)):
                cuadrado[i][x]=255
        #cuadrado[0][q]=255
        #cuadrado[n-1][q]=255

    print (cuadrado)
    return cuadrado

def cruzMasks(n):
    cuadrado = np.zeros((n,n),np.uint8)
    for i in range(0,n):
        l=[]
        q=int(n/2)
        if i == int(n/2) :
            for j in range (0,n):
                cuadrado[i][j]=255
        else:
            cuadrado[i][q]=255
    print (cuadrado)
    return cuadrado

def binaryimg(img):
    rows,cols=img.shape
    Umbral=120
    new_img=np.zeros_like(img, np.uint8)
    for i in range(0,rows):
        for j in range(0,cols):
            if img[i][j]>Umbral:#255/2:
                new_img[i][j]=255
            else:
                new_img[i][j]=0
    return new_img
def erosion(img,M,x,y):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    Mh=len(M)
    Mw=len(M[0])

    for i in range (0,Ah):
        for j in range(0,Aw):
            #if i + Mh-1<Ah and j+Mw-1<Aw:
            if i+x<Ah and j+y<Aw:
                contA=0
                contM=0
                for k in range(0,Mh):
                    for l in range(0,Mw):
                        if M[k][l]==255:
                            contM+=1
                            if i+k<Ah and j+l<Aw:
                                if A[i+k][j+l]==255:
                                    contA+=1
                if contA==contM:
                    #if NI[i+k][j+l]!=255:
                    #if NI[i+x][j+y]!=255:
                    NI[i+x][j+y]=255

    img1 = Image.fromarray(NI)
    #img1.save('erosion.png')
    #img1.show()
    #print (NI)
    return img1
    """rows,cols=binImg.shape
    rowsKernel=len(kernelp)
    colsKernel=len(kernelp[0])
    img_dilatasion= np.zeros((rows,cols))
    posx=x;
    posy=y;
    for x in range (0,rows-rowsKernel):
        for y in range(0,cols-colsKernel):
            if binImg[x+posx][y+posy]==255:
                for k in range (0,rowsKernel):
                    for l in range (0,colsKernel):
                        if kernelp[k][l]==255:
                            if img_dilatasion[x+k][y+l]!=255:
                                img_dilatasion[x+k][y+l]=255

    cv2.imshow('dilatacion mio ', img_dilatasion)
    cv2.waitKey(0)"""
    """dilatasion = cv2.dilate(binImg,kernel,iterations = 1)
    cv2.imshow('dilatacion', dilatasion)
    cv2.waitKey(0)"""
    #return img_dilatasion
def dilatasion(img,M,x,y):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = img.copy()#np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    Mh=len(M)
    Mw=len(M[0])
    for i in range (0,Ah):
        for j in range(0,Aw):
            #if i + Mh-1<Ah and j+Mw-1<Aw:
            if i+x<Ah and j+y<Aw:
                if A[i+x][j+y]==255:
                    for k in range(0,Mh):
                        for l in range(0,Mw):
                            if M[k][l]==255:
                                if i+k<Ah and j+l<Aw:
                                    if NI[i+k][j+l]!=255:
                                        NI[i+k][j+l]=255
    #img1 = Image.fromarray(NI)
    #img1.save('dilatasion.png')
    dilatasion = cv2.dilate(img,kernel,iterations = 1)
    #cv2.imshow('dilatacion', dilatasion)
    #cv2.waitKey(0)
    return dilatasion
def opening(binImg,kerneln,x,y):
    resulIMG=erosion(binImg,kerneln,x,y)
    new_img=dilatasion(resulIMG,kerneln,x,y)
    new_img.show()
    new_img.save('opening.png')

    """ opening = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kerneln)
    cv2.imshow('opening', opening)
    cv2.waitKey(0)"""
def closing(binImg,kerneln,x,y):
    new_img=dilatasion(binImg,kerneln,x,y)
    resulIMG=erosion(new_img,kerneln,x,y)
    #resulIMG.show()
    #resulIMG.save('closing.png')
    #opening = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, kerneln)
    #cv2.imshow('Closing', closing)
    #cv2.waitKey(0)
    return closing
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
            sum = int(Gx_img[i][j]) + int(Gy_img[i][j])
            if sum < 0 :
                new_image[i][j] = 0
            elif sum > 255 :
                 new_image[i][j] = 255
            else:
                new_image[i][j] = sum 

    return new_image
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
#Proceso paper
def Identification_face_region(imgGrayscale,path):
    OriginImg = cv2.imread(path)
    #new_img =equalization(imgGrayscale)
    new_img=Sobel(imgGrayscale)
    new_img=binaryimg(new_img)
    #img_ycrcb = np.zeros((height,width,3))
    #cv2.imshow('Sobel ',new_img)
    #cv2.waitKey()
    #new_img=dilatasion(new_img,kernel,0,0)
    #img_ycrcb = np.zeros((height,width,3))
    new_img=dilatasion(new_img,kernel,0,0)
    col,row=new_img.shape
    print("row:"+str(row)+"col:"+str(col))
    #cv2.imshow('dilatation',new_img)
    #cv2.waitKey()
    rowF=row/2
    colF=0
    for i in range(5,row/2):
        for j in range(5,col-10):
            if new_img[i][j]==255 and i<rowF:
                colF=j
                rowF=i;
    rowS=0
    colS=col/2
    for i in range(5,row/2):
        for j in range(5,col/2):
            if new_img[i][j]==255 and j<colS:
                colS=j
                rowS=i;
    h=abs(rowS-rowF)
    w=abs(colS-colF)
    #img_rsz=np.zeros((h,))
    img=new_img.copy()
    img_rsz=np.zeros((2*h,2*w))
    CutImg=np.zeros((2*h,2*w,3))
    #img_rsz = cv2.resize(new_img,(int(h),int(w)))
    x=0
    y=0
    #import pudb
    #pudb.set_trace()
    for i in range(rowF,rowF+2*h):
        for j in range(colS,colS+2*w):
            img_rsz[x][y]=img[i][j]
            y=y+1
        x=x+1
        y=0
    cropped = OriginImg[rowF:rowF+2*h, colS:colS+2*w]
    #cv2.imshow("cropped", cropped)
    #cv2.waitKey(0)
    cv2.imshow('recortado',img_rsz)
    cv2.waitKey()
    ListImg=[cropped,img_rsz]
    return ListImg
def Extraction_eyes(new_img):
    kerneln=diamanteMask(5)
    img=closing(new_img,kerneln,2,2)
    img=erosion(img,kerneln,2,2)
    #img=opening(img,kernel,0,0)
    img=erosion(img,kerneln,2,2)
    #dilatasion = cv2.dilate(img,kerneln,iterations = 1)
    #cv2.imshow('dilatacion', img)
    #cv2.waitKey(0)
    #img=dilatasion(img,kerneln,2,2s)
    #img1.save('erosion.png')
    img.show()
    #cv2.imshow('erotion', img)
    #cv2.waitKey(0)


if __name__=='__main__':
    path = "../IMM-Frontal Face DB SMALL/01_01.jpg"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #Preprocessing
    #img = cv2.imread('../IMM-Frontal Face DB SMALL/01_01.jpg') 
    #cv2.imshow('Img Origin', img)                                     
    #cv2.waitKey(0)

    new_img=Identification_face_region(img,path)
    Extraction_eyes(new_img[1])















    #morfologia
    #img = cv2.imread('../PruebasImagenes/coins.png',cv2.IMREAD_GRAYSCALE)      # cargar el archivo PNG indicado
    #cuadradoMask(3)
    #diamanteMask(3)
    #cruzMasks(3)
    #kerneln=cruzMasks(5)
    #binImg=binaryimg(img)
    #erosion(binImg,kerneln,1,1)
    #dilatasion(binImg,kerneln,1,1)
    #opening(binImg,kerneln,1,1)
    #closing(binImg,kerneln,1,1)


    
