import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math 
import cv2

kernel = np.ones((3,3),np.uint8)


def model_RGB(img): #para borrar ruido 
    b, g, r = cv2.split(img)
    zeros = np.zeros_like(b)
    cv2.imshow('Original', img)
    cv2.imshow('blue', cv2.merge( (b, zeros, zeros) ) )
    cv2.imshow('green', cv2.merge( (zeros, g, zeros) ) )
    cv2.imshow('red', cv2.merge( (zeros, zeros, r) ) )
    cv2.waitKey()
def RGB_to_CMY(img):
    rows,cols,scal=img.shape
    new_img=img.copy()
    for x in range(0,rows):
        for y in range(0,cols):
            new_img[x,y,0] = (255 - img[x,y,0]) if 255 - img[x,y,0]>0 else 0 
            new_img[x,y,1] = (255 - img[x,y,1]) if 255 - img[x,y,1]>0 else 0 
            new_img[x,y,2] = (255- img[x,y,2]) if 255 - img[x,y,2]>0 else 0 
    c, m, y = cv2.split(new_img)
    zeros = np.zeros_like(c)
    cv2.imshow('C', cv2.merge( (c, zeros, zeros) ) )
    cv2.imshow('M', cv2.merge( (zeros, m, zeros) ) )
    cv2.imshow('Y', cv2.merge( (zeros, zeros, y) ) )
    cv2.imshow('Original CMY ', new_img)
    cv2.imshow('Original', img)
    cv2.waitKey()
def RGB_to_CMYK(img):
    rows,cols,scal=img.shape
    img_C=img.copy()
    img_M=img.copy()
    img_Y=img.copy()
    img_k = np.zeros((rows,cols,3))

    Rc=img[0,0,0]/255
    Gc=img[0,0,1]/255
    Bc=img[0,0,2]/255
    for x in range(0,rows):
        for y in range(0,cols):
            if img[x,y,0]/255 > Rc:
                Rc = img[x,y,0]/255
            if img[x,y,1]/255 > Gc:
                Gc = img[x,y,1]/255
            if img[x,y,2]/255 > Bc:
                Bc = img[x,y,2]/255

    k = 1-max([Rc,Gc,Bc])
    for x in range(0,rows):
        for y in range(0,cols):
            img_C[x,y,0] = (1 - (img[x,y,0]/255)-k) /(1-k) 
            img_M[x,y,1] = (1 - (img[x,y,1]/255)-k) /(1-k) 
            img_Y[x,y,2] = (1 - (img[x,y,2]/255)-k) /(1-k) 
            img_k[x,y,0]= k 
    cv2.imshow('C',img_C)
    cv2.imshow('M',img_M)
    cv2.imshow('Y',img_Y)
    cv2.imshow('K',img_k)
    cv2.imshow('Original', img)
    #cv2.imwrite('image_ycrcb.jpg', img_ycrcb)
    cv2.waitKey()
def RGB_to_HSV(img):
    rows,cols,scal=img.shape
    new_img=img.copy()
    for x in range(0,rows):
        for y in range(0,cols):
            r=img[x,y,0]
            g=img[x,y,1]
            b=img[x,y,2]
            mx = max([r, g, b])
            mn = min([r, g, b])
            df = mx-mn
            # Hue Calculation
            if mx == r:
                new_img[x,y,0] = (60 * ((g-b)/df))
            elif mx == g:
                new_img[x,y,0] = (60 * ((b-r)/df) + 120)
            elif mx == b:
                new_img[x,y,0] = (60 * ((r-g)/df) + 240) 
            if mx == mn:
                new_img[x,y,0] = 0

            """if df == 0:
                                                    new_img[x,y,0] = 0
                                                elif mx == r:
                                                    new_img[x,y,0] = 60 * (((g - b)/df) % 6)
                                                elif mx == g:
                                                    new_img[x,y,0] = 60 * (((b - r)/df) + 2)
                                                elif mx == b:
                                                    new_img[x,y,0] = 60 * (((r - g)/df) + 4)"""

             # Saturation Calculation
            if mx == 0:
                new_img[x,y,1] = 0
            else :
                new_img[x,y,1] = df/mx
            # Value Calculation
            new_img[x,y,2] = mx

    cv2.imwrite('hsv_img.jpg',new_img)
    cv2.imshow('HSV', new_img)
    h, s, v = cv2.split(new_img)
    zeros = np.zeros_like(h)
    cv2.imshow('H', cv2.merge( (h, zeros, zeros) ) )
    cv2.imshow('S', cv2.merge( (zeros,s, zeros) ) )
    cv2.imshow('V', cv2.merge( (zeros, zeros, v) ) )
    cv2.imshow('Original', img)
    cv2.waitKey()
def HSI(img):
    height,width,channel = img.shape
    img_hsi = np.zeros((height,width,3))
    for i in np.arange(height):
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = r/255.
            g_ = g/255.
            b_ = b/255.

            Cmin = min(r_,g_,b_)
            sum = r_ + g_ + b_

            #theta
            num = (1/2.)*( (r_ - g_) + (r_ - b_) )  # numerador
            dem = math.sqrt( (r_ - g_)**2 + (r_ - b_)*(g_ - b_) )  # denominador
            if(dem!=0): div = num/dem
            else: div = 0
            theta = math.acos(div)
            #Hue
            H = theta
            if(b_ > g_): H = 360-theta 
            #Saturation
            S = 1-(3/sum)*Cmin
            #Intensity
            I = sum/3

            img_hsi.itemset((i,j,0),int(H))
            img_hsi.itemset((i,j,1),int(S))
            img_hsi.itemset((i,j,2),int(I))
    
    cv2.imwrite('hsi_img.jpg', img_hsi)
    h,s,i = cv2.split(img_hsi)
    zeros = np.zeros_like(h, dtype = float)

    cv2.imshow('HSI', img_hsi)
    cv2.imshow('H', h )
    cv2.imshow('S', cv2.merge( (zeros, s, zeros) ) )
    cv2.imshow('I', cv2.merge( (zeros, zeros, i) ) )

    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def RGB_to_HSI(img):
    rows,cols,scal=img.shape
    img_H=img.copy()
    img_S=img.copy()
    img_I=img.copy()

    for x in range(0,rows):
        for y in range(0,cols):
            r=img[x,y,0]
            g=img[x,y,1]
            b=img[x,y,2]
            theta = (1/2*((r-g)+(r-b)))
            sub = math.sqrt(math.pow(r-g,2)+(r-b)(g-b))
            theta = math.acos(theta/sub)
            print (theta)
            if b<=g :
                img_H[x,y,0]=theta
            else:
                img_H[x,y,0]=360-theta
            img_S[x,y,1] = (1-(3/(r+g+b))*min([r,g,b]))
            img_I[x,y,2] = (1/3*(r+g+b))

    cv2.imshow('H',img_H)
    cv2.imshow('S',img_S)
    cv2.imshow('V',img_I)
    cv2.imshow('Original', img)
    cv2.waitKey()
def verify_condition(b):
    if b>255 :
        return 255
    if b<0:
        return 0
    else:
        return b
def RGB_to_YUV(img):
    rows,cols,scal=img.shape
    img_Y=np.zeros((rows,cols,3))
    img_U=np.zeros((rows,cols,3))
    img_V=np.zeros((rows,cols,3))
    matrix = np.array([[0.299,0.587,0.144],
                      [-0.147,-0.289,0.436],
                      [0.615,-0.515,-0.100]])
    new_img = img.copy()
    for x in range(0,rows):
        for y in range(0,cols):
            r=img[x,y,0]
            g=img[x,y,1]
            b=img[x,y,2]
            new_img[x,y,0] = verify_condition(matrix[0][0]*r + matrix[0][1]*g + matrix[0][2]*b) 
            new_img[x,y,1] = verify_condition(matrix[1][0]*r + matrix[1][1]*g + matrix[1][2]*b) 
            new_img[x,y,2] = verify_condition(matrix[2][0]*r + matrix[2][1]*g + matrix[2][2]*b) 

    y, u, v = cv2.split(new_img)
    zeros = np.zeros_like(y)
    cv2.imshow('Y', cv2.merge( (y, zeros, zeros) ) )
    cv2.imshow('U', cv2.merge( (zeros, u, zeros) ) )
    cv2.imshow('V', cv2.merge( (zeros, zeros, v) ) )
    cv2.imshow('YUV', new_img)
    cv2.imshow('Original', img)
    cv2.waitKey()
def RGB_to_YIQ(img):
    height,width,channel = img.shape
    img_yiq = np.zeros((height,width,3))
    for i in np.arange(height):
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = r/255.
            g_ = g/255.
            b_ = b/255.

            Y = 0.299*r_ + 0.587*g_ + 0.144*b_
            I = 0.596*r_ - 0.275*g_ - 0.321*b_
            Q = 0.212*r_ - 0.523*g_ + 0.311*b_

            img_yiq.itemset((i,j,0),int(Y*255))
            img_yiq.itemset((i,j,1),int(I*255))
            img_yiq.itemset((i,j,2),int(Q*255))
    
    # Write image
    cv2.imwrite('image_yiq.jpg', img_yiq)

    y,i,q = cv2.split(img_yiq)
    zeros = np.zeros_like(y, dtype = float)
    #View image
    cv2.imshow('YIQ', img_yiq)
    cv2.imshow('Y', y )
    cv2.imshow('I', cv2.merge( (zeros, i, zeros) ) )
    cv2.imshow('Q', cv2.merge( (zeros, zeros, q) ) )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def RGB_to_YCrCb(img):
    height,width,channel = img.shape
    img_ycrcb = np.zeros((height,width,3))
    img_Y=img.copy()
    img_Cr=img.copy()
    img_Cb=img.copy()
    for i in np.arange(height):
        for j in np.arange(width):
            R = img.item(i,j,0)
            G = img.item(i,j,1)
            B = img.item(i,j,2)

            Y =   16 +  65.738*R/256. + 129.057*G/256. +  25.064*B/256.
            Cb = 128 -  37.945*R/256. -  74.494*G/256. + 112.439*B/256.
            Cr = 128 + 112.439*R/256. -  94.154*G/256. -  18.285*B/256.
            
            img_Y[i,j,0]=Y
            img_Cr[i,j,1]=Cr
            img_Cb[i,j,2]=Cb
            img_ycrcb.itemset((i,j,0),int(Y))
            img_ycrcb.itemset((i,j,1),int(Cr))
            img_ycrcb.itemset((i,j,2),int(Cb))

    y,cr,cb = cv2.split(img_ycrcb)
    zeros = np.zeros_like(y, dtype = float)

    # Write image
    cv2.imwrite('image_ycrcb.jpg', img_ycrcb)

    cv2.imshow('YCrCb', img_ycrcb)
    cv2.imshow('Y', img_Y )
    cv2.imshow('Cr', img_Cr)
    cv2.imshow('Cb', img_Cb )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def maskHorizontal_line(img):
    rows,cols=img.shape
    new_image = img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sum = (-1*int(img[i-1][j-1]) + -1*int(img[i-1][j]) +  -1*int(img[i-1][j+1]) + 
                   2*int(img[i][j-1])   + 2*int(img[i][j])  +  2*int(img[i][j+1]) + 
                   -1*int(img[i+1][j+1]) + -1*int(img[i+1][j]) +  -1* int(img[i+1][j+1]))
            if sum < 0:
                new_image[i][j] = 0
            elif sum > 255:
                new_image[i][j] = 255
            else:
                new_image[i][j] = sum
    cv2.imshow('Horizontal_line', new_image)
    cv2.waitKey()
    return new_image
def mask45_line(img):
    rows,cols=img.shape
    new_image = img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sum = (-1*int(img[i-1][j-1]) + -1*int(img[i-1][j]) +  2*int(img[i-1][j+1]) + 
                   -1*int(img[i][j-1])   + 2*int(img[i][j])  +  -1*int(img[i][j+1]) + 
                   2*int(img[i+1][j+1]) + -1*int(img[i+1][j]) +  -1* int(img[i+1][j+1]))
            if sum < 0:
                new_image[i][j] = 0
            elif sum > 255:
                new_image[i][j] = 255
            else:
                new_image[i][j] = sum
    cv2.imshow('45_line', new_image)
    cv2.waitKey()
    return new_image
def maskVertical_line(img):
    rows,cols=img.shape
    new_image = img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sum = (-1*int(img[i-1][j-1]) + 2*int(img[i-1][j]) +  -1*int(img[i-1][j+1]) + 
                   -1*int(img[i][j-1])   + 2*int(img[i][j])  +  -1*int(img[i][j+1]) + 
                   -1*int(img[i+1][j+1]) + 2*int(img[i+1][j]) +  -1* int(img[i+1][j+1]))
            if sum < 0:
                new_image[i][j] = 0
            elif sum > 255:
                new_image[i][j] = 255
            else:
                new_image[i][j] = sum
    cv2.imshow('Vertical_line', new_image)
    cv2.waitKey()
    return new_image
def maskmenus45_line(img):
    rows,cols=img.shape
    new_image = img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sum = (2*int(img[i-1][j-1]) + -1*int(img[i-1][j]) +  -1*int(img[i-1][j+1]) + 
                   -1*int(img[i][j-1])   + 2*int(img[i][j])  +  -1*int(img[i][j+1]) + 
                   -1*int(img[i+1][j+1]) + -1*int(img[i+1][j]) +  2* int(img[i+1][j+1]))
            if sum < 0:
                new_image[i][j] = 0
            elif sum > 255:
                new_image[i][j] = 255
            else:
                new_image[i][j] = sum
    cv2.imshow('-45_line', new_image)
    cv2.waitKey()
    return new_image
def pointdeteccion(img):
    rows,cols=img.shape
    new_image = img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sum = (-1*int(img[i-1][j-1]) + -1*int(img[i-1][j]) +  -1*int(img[i-1][j+1]) + 
                   -1*int(img[i][j-1])   + 8*int(img[i][j])  +  -1*int(img[i][j+1]) + 
                   -1*int(img[i+1][j+1]) + -1*int(img[i+1][j]) +  -1* int(img[i+1][j+1]))
            if sum < 0:
                new_image[i][j] = 0
            elif sum > 255:
                new_image[i][j] = 255
            else:
                new_image[i][j] = sum
    cv2.imshow('point', new_image)
    cv2.waitKey()
def YUV(img):
    #img = cv2.imread(image_path)

    # Get the image's height, width, and channels
    height,width,channel = img.shape

    # Create balnk HSV image
    img_yuv = np.zeros((height,width,3))
    for i in np.arange(height):
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = r/255.
            g_ = g/255.
            b_ = b/255.

            Y = 0.299*r_ + 0.587*g_ + 0.144*b_
            U = -0.147*r_ -0.289*g_ + 0.436*b_
            V = 0.615*r_ - 0.515*g_ - 0.100*b_

            img_yuv.itemset((i,j,0),int(Y*255))
            img_yuv.itemset((i,j,1),int(U*255))
            img_yuv.itemset((i,j,2),int(V*255))
    
    cv2.imwrite('YUV.jpg', img_yuv)
    y,u,v = cv2.split(img_yuv)
    zeros = np.zeros_like(y, dtype = float)
    cv2.imwrite('Yuv.png', cv2.merge( (y, zeros, zeros) ))
    cv2.imshow('YUV', img_yuv)
    cv2.imshow('Y', cv2.merge( (y, zeros, zeros) ) )
    #cv2.imshow('U', cv2.merge( (zeros, u, zeros) ) )
    #cv2.imshow('V', cv2.merge( (zeros, zeros, v) ) )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def YIQ(img):
    #img = cv2.imread(image_path)

    # Get the image's height, width, and channels
    height,width,channel = img.shape

    # Create balnk HSV image
    img_yiq = np.zeros((height,width,3))

    for i in np.arange(height):
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = r/255.
            g_ = g/255.
            b_ = b/255.

            Y = 0.299*r_ + 0.587*g_ + 0.144*b_
            I = 0.596*r_ - 0.275*g_ - 0.321*b_
            Q = 0.212*r_ - 0.523*g_ + 0.311*b_

            img_yiq.itemset((i,j,0),int(Y*255))
            img_yiq.itemset((i,j,1),int(I*255))
            img_yiq.itemset((i,j,2),int(Q*255))
    
    # Write image
    cv2.imwrite('YIQ.jpg', img_yiq)

    y,i,q = cv2.split(img_yiq)
    zeros = np.zeros_like(y, dtype = float)
    cv2.imwrite('Y.png', cv2.merge( (y, zeros, zeros) ))
    #View image
    cv2.imshow('YIQ', img_yiq)
    cv2.imshow('Yiq', cv2.merge( (y, zeros, zeros) ) )
    #cv2.imshow('I', cv2.merge( (zeros, i, zeros) ) )
    #cv2.imshow('Q', cv2.merge( (zeros, zeros, q) ) )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def YCrCb(img):
    #img = cv2.imread(image_path)
    
    height,width,channel = img.shape
    img_ycrcb = np.zeros((height,width,3))
    for i in np.arange(height):
        for j in np.arange(width):
            R = img.item(i,j,0)
            G = img.item(i,j,1)
            B = img.item(i,j,2)

            Y =   16 +  65.738*R/256. + 129.057*G/256. +  25.064*B/256.
            Cb = 128 -  37.945*R/256. -  74.494*G/256. + 112.439*B/256.
            Cr = 128 + 112.439*R/256. -  94.154*G/256. -  18.285*B/256.

            # print(Y, Cb, Cr)
            
            img_ycrcb.itemset((i,j,0),int(Y))
            img_ycrcb.itemset((i,j,1),int(Cr))
            img_ycrcb.itemset((i,j,2),int(Cb))



    # img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 
    cv2.imwrite('YCrCb.jpg', img_ycrcb)

    y,cr,cb = cv2.split(img_ycrcb)
    zeros = np.zeros_like(y, dtype = float)
    cv2.imwrite('Ycrcb.png', cv2.merge( (y,zeros, zeros) ))
    cv2.imshow('YCrCb', img_ycrcb)
    cv2.imshow('Y', cv2.merge( (y,zeros, zeros) ))
    #cv2.imshow('Cr',cv2.merge( (zeros,cr, zeros)))
    #cv2.imshow('Cb', cv2.merge( (zeros,zeros, cb)) )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Model CIE,lab


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
            cuadrado[i][q-i]=255
            cuadrado[i][q+i]=255
        else:
            cuadrado[i][i-q]=255
            cuadrado[i][(n-1)-(i-q)]=255
        cuadrado[q][q]=255

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
    Umbral=127
    new_img=img.copy()
    for i in range(0,rows):
        for j in range(0,cols):
            if img[i][j]>Umbral:#255/2:
                new_img[i][j]=255
            else:
                new_img[i][j]=0
    cv2.imshow('binary', new_img)
    cv2.waitKey(0)
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
    img1.save('erosion.png')
    img1.show()
    #print (NI)
    return img1
#def erocion(binImg,kernelp,x,y):
    rows,cols=binImg.shape
    rowsKernel=len(kernelp)
    colsKernel=len(kernelp[0])
    #img_erosion=binImg.copy()
    img_erosion = np.zeros((rows,cols))
    a=np.array(binImg).tolist()
    #new_img=img.copy()
   
    posx=x;
    posy=y;
    #value=kernelp[posx][posy]
    for x in range (0,rows-rowsKernel):
        for y in range(0,cols-colsKernel):
            pertenece = True
            k=0
            l=0
            while k<rowsKernel and pertenece==True:
                while l<colsKernel and pertenece==True:
                    if binImg[x+k][y+l]!=kernelp[k][l]:
                        pertenece = False
                    l=l+1
                k=k+1
                l=0
            if pertenece==True:
                img_erosion[posx][posy]=255
                #img_erosion[posx][posy]=binImg[posx][posy]
            posy=posy+1
        posx=posx+1
        posy=0

    cv2.imshow('erosion mio', img_erosion)
    cv2.waitKey(0)
    erosion = cv2.erode(binImg,kernelp,iterations = 1)
    cv2.imshow('erosion', erosion)
    cv2.waitKey(0)
    return img_erosion
#def dilatacion(img,kernelp,x,y):
#def dilatacion(binImg,kernelp,x,y):
    rows,cols=binImg.shape
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
    cv2.waitKey(0)
    """dilatasion = cv2.dilate(binImg,kernel,iterations = 1)
    cv2.imshow('dilatacion', dilatasion)
    cv2.waitKey(0)"""
    return img_dilatasion
def dilatasion(img,M,x,y):
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
                if A[i+x][j+y]==255:
                    for k in range(0,Mh):
                        for l in range(0,Mw):
                            if M[k][l]==255:
                                if i+k<Ah and j+l<Aw:
                                    if NI[i+k][j+l]!=255:
                                        NI[i+k][j+l]=255
    img1 = Image.fromarray(NI)
    img1.save('dilatasion.png')
    #img1.show()
    #print (NI)
    return img1
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
    resulIMG.show()
    resulIMG.save('closing.png')
    #opening = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kernel)
    """closing = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, kerneln)
    cv2.imshow('Closing', closing)
    cv2.waitKey(0)"""

def segmentationSimiliradid(img):
    T = 100
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    valorMin = hist[255]
    pixel = 0
    for i in range(0,len(hist)):
         if hist[i]<valorMin and hist[i]>0:
             valorMin = hist[i]
             pixel = i
    print(pixel)
    #T=pixel
    plt.plot(hist, color='gray' )
    plt.xlabel('Pixeles')
    plt.ylabel('Histogram')
    plt.show()
    rows,cols=img.shape
    new_image = img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if(img[i][j]>T):
                new_image[i][j]=255
            if(img[i][j]<=T):
               new_image[i][j]=0

    cv2.imshow('segmentation', new_image)                                     
    cv2.waitKey(0)


if __name__=='__main__':
    
    img = cv2.imread('../PruebasImagenes/lena.jpg')      # cargar el archivo PNG indicado
    YUV(img)
    YIQ(img)
    YCrCb(img)
    #RGB_to_HSV(img)
    #model_RGB(img)
    #RGB_to_CMY(img)
    #RGB_to_CMYK(img)
    #RGB_to_HSI(img)
    #RGB_to_HSV(img)
    #RGB_to_HSV(img)
    #HSI(img)
    #RGB_to_YUV(img)
    #RGB_to_YIQ(img)
    #RGB_to_YCrCb(img)binary

    #img2 = cv2.imread('../PruebasImagenes/church1.jpg', cv2.IMREAD_GRAYSCALE)      # cargar el archivo PNG indicado
    #cv2.imshow('Img Origin', img2)                                     
    #cv2.waitKey(0)
    #pointdeteccion(img2)

    #new_img =  maskHorizontal_line(img2)
    #new_img =  mask45_line(new_img)
    #new_img =  maskVertical_line(new_img)
    #new_img =  maskmenus45_line(new_img)

    # maskVertical_line(img2)
    # maskmenus45_line(img2)

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

    #segmentation
    #img = cv2.imread('../PruebasImagenes/coins.png',cv2.IMREAD_GRAYSCALE)      # cargar el archivo PNG indicado
    #segmentationSimiliradid(img)

    
