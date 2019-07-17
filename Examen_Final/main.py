import matplotlib.pyplot as plt
import skimage.measure as measure
from PIL import Image
from skimage.measure import label
import numpy as np
import math 
import cv2

kernel = np.ones((5,5),np.uint8)
STREL_4 = np.array([
        [0,0,0,1,0,0,0],
        [0,1,1,1,1,1,0],
        [0,1,1,1,1,1,0],
        [1,1,1,1,1,1,1],
        [0,1,1,1,1,1,0],
        [0,1,1,1,1,1,0],
        [0,0,0,1,0,0,0]], dtype=np.uint8)

def diamanteMask(n):#numero impar
    cuadrado = np.zeros((n,n),np.uint8)
    for i in range(0,n):
        q=int(n/2)
        if i<=q:
            for j in range(q-i,q+i+1):
                cuadrado[i][j]=255
        else:
            for x in range(i-q,n-(i-q)):
                cuadrado[i][x]=255

    #print (cuadrado)
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

    #img1 = Image.fromarray(NI)
    img_erosion = cv2.erode(img, M, iterations=1) 
    #img1.save('erosion.png')
    #img1.show()
    #print (NI)
    #return img1
    return img_erosion
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
def closing(binImg,kerneln,x,y):
    new_img=dilatasion(binImg,kerneln,x,y)
    resulIMG=erosion(new_img,kerneln,x,y)
    #resulIMG.show()
    #resulIMG.save('closing.png')
    #opening = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kernel)
    #closing = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, kerneln)
    #cv2.imshow('Closing', closing)
    #cv2.waitKey(0)
    return resulIMG
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
def Identification_face_region(imgGrayscale,path): 
    #Leer la imagen de Original a color 
    OriginImg = cv2.imread(path)

    #Hallar los bordes utilizando el operador Sobel
    new_img=Sobel(imgGrayscale)
    new_img=binaryimg(new_img)
    #cv2.imshow('Op SOBEL',new_img)
    #cv2.waitKey()

    #Dilated image
    new_img=dilatasion(new_img,kernel,0,0)
    col,row=new_img.shape
    #cv2.imshow('Dilatation',new_img)
    #cv2.waitKey()

    #Geometrical operations to find the minimum row and minimum column
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
    img=new_img.copy()
    img_rsz=np.zeros((2*h,2*w))
    CutImg=np.zeros((2*h,2*w,3))
    x=0
    y=0
    for i in range(rowF,rowF+2*h):
        for j in range(colS,colS+2*w):
            img_rsz[x][y]=img[i][j]
            y=y+1
        x=x+1
        y=0


    #Cropped face region image
    cropped = OriginImg[rowF:rowF+2*h, colS:colS+2*w]
    #cv2.imshow("cropped", cropped)
    #cv2.waitKey(0)
    ListImg=[cropped,img_rsz]

    return ListImg

def Extraction_eyes(ListImg,path):
    diamond=diamanteMask(9)

    new_img=ListImg[1]  #return the dilated image as shown in figure 3(b)
    img_origin=ListImg[0]

    #Morphological closing operation with diamond structuring element
    img=closing(new_img,diamond,4,4)

    #operation erotion
    img=erosion(img,STREL_4,3,3)
    cv2.imshow('erosion', img)
    cv2.waitKey(0)

    """operation erotion Centroid of white regions in the image is 
    computed for calculation of distance, angle between white areas."""

    height,Weight = img.shape
    label_img = label(img, connectivity=img.ndim)
    props = measure.regionprops(label_img)
    cp=[]
    for pr in props:
        c=pr['centroid']
        #print("C:")
        #print(c)
        if int(c[1])>=(Weight/9) and int(c[1])<=Weight-(Weight/9):
            if int(c[0])>=(height/2):
                cp.append(c);
    if len(cp)!=2:
        #print("size:CP "+ str(len(cp)))
        EyesP=[]
        for c in range(0,len(cp)):
            for p in range(c,len(cp)):
                if cp[c]!=cp[p]:
                    #print("pares:("+str(c)+str(p) +")") 
                    x = cp[p][0] - cp[c][0]
                    y = cp[p][1] - cp[c][1]
                    angle = (math.atan2(x,y) * 180.0 )/ math.pi
                    theta = abs(angle)
                    print("centroid jjj")
                    print(cp[c])
                    print(cp[p])
                    print("theta:"+str(theta))
                    dist = math.hypot(y,x)
                    print("Distancia entre centroides" + str(dist))

                    
                    if theta>177 and theta<180  and cp[p][0]>150 and cp[c][0]>150 :
                        print("\n")
                        print("centroid ")
                        print(cp[c])
                        print(cp[p])
                        print("theta:"+str(theta))
                        EyesP.append(cp[c])
                        EyesP.append(cp[p])
                        print("Distancia centroides" + str(dist))
                    """if dist>110 and dist<118:
                        #print("centroid jjj")
                        #print(cp[c])
                        #print(cp[p])
                        #print("theta:"+str(theta))
                        EyesP.append(cp[c])
                        EyesP.append(cp[p])"""
                print("\n")
            
        if len(EyesP)>=2:
            #print("EYES P")
            h = 40
            A=EyesP[0]
            B=EyesP[1]
            print("A")
            print(A)
            print("B")
            print(B)
            cv2.rectangle (img_origin, (int(A[1])-h,int(A[0])+h), (int(A[1])+h,int(A[0])-h), (0,0,255),3)
            cv2.rectangle (img_origin, (int(B[1])-h,int(B[0])+h), (int(B[1])+h,int(B[0])-h), (0,0,255),3)
        else:
            print("No se hallaron puntos")

    cv2.imwrite('Ycrcb.png', img_origin)
    cv2.imshow('img',img_origin)
    cv2.waitKey(0)

if __name__=='__main__':

    #path = "../IMM-Frontal Face DB SMALL/01_01.jpg"
    path = "../IMM-Frontal Face DB SMALL/02_04.jpg"
    #path = "../IMM-Frontal Face DB SMALL/06_08.jpg"

    #path = "../IMM-Frontal Face DB SMALL/09_04.jpg" #RECONOCE OJO Y NARIZ JEJEJEJE
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #Preprocessing
    
    new_img=Identification_face_region(img,path)
    Extraction_eyes(new_img,path)