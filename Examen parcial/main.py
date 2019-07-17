import matplotlib.pyplot as plt
import numpy as np
import math 
import cv2

def ChangeColor(img):
    date=np.array(img)
    for x in range(0,len(date)):
        for y in range(0,len(date[0])):
            img[x][y]=(256-date[x][y])
    #cv2.imwrite('resul.png',img)
    cv2.imshow('Change color ', img)
    cv2.waitKey(0)
def expo(img):
    rows,cols=img.shape
    newimg = np.zeros_like(img, np.uint8)
    for x in range(0,rows):
        for y in range(0,cols):
            newimg[x][y] = math.pow(float(img[x][y]),10)
    cv2.imshow('exp color ', newimg)
    cv2.waitKey(0)
def logg(img):
    rows,cols=img.shape
    newimg = np.zeros_like(img, np.uint8)
    for x in range(0,rows):
        for y in range(0,cols):
            if (img[x][y] == 0 or img[x][y] < 1 ):
                newimg[x][y] = 0
            else:
                newimg[x][y] = math.log(img[x][y])
    cv2.imshow('log color ', newimg)
    cv2.waitKey(0)
def translate(img,x,y):                      #Remember width = number of columns, and height = number of rows.
    rows,cols=img.shape
    newimg = cv2.resize(img,(int(rows+x),int(cols+y)))
    newimg = np.zeros_like(newimg, np.uint8)
    for row in range(0,rows):
        for col in range(0,cols):
            newimg[row+x][col+y] = img[row][col]
    cv2.imshow('imagen translate',newimg)
    cv2.waitKey(0)
def radianes(value):
    return (value*math.pi)/180
def rotation(img,angle):                      #Remember width = number of columns, and height = number of rows.
    rows,cols=img.shape
    angle = radianes(angle)
    newimg = cv2.resize(img,(int(rows),int(cols)))
    newimg = np.zeros_like(newimg, np.uint8)
    for row in range(0,rows):
        for col in range(0,cols):
            x=row*math.cos(angle) - col*math.sin(angle)
            y=row*math.sin(angle) + col*math.cos(angle)
            newimg[int(x)][int(y)] = img[row][col]
    cv2.imshow('img rotation',newimg)
    cv2.waitKey(0)
def scaling(img,s):                      #Remember width = number of columns, and height = number of rows.
    rows,cols=img.shape
    newimg = cv2.resize(img,(int(rows*s+1),int(cols*s+1)))
    newimg = np.zeros_like(newimg, np.uint8)
    for row in range(0,rows):
        for col in range(0,cols):
            x=math.floor(row*s)
            y=math.floor(col*s)
            newimg[int(x)][int(y)] = img[row][col]
    cv2.imshow('img scaling',newimg)
    cv2.waitKey(0)

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
def equalization(img):
    LevelGray = 256
    M,N = img.shape
    print (M , N)
    valorMax = 0
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # valorMax = hist[0]
    # pixel = 0
    # for i in range(0,len(hist)):
    #     if hist[i]>valorMax :
    #         valorMax = hist[i]
    #         pixel = i

    # print("Valor maximo")
    # print(valorMax )
    # print(pixel)
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

    cv2.imshow('Img Equalizate', new_img)                                     
    cv2.waitKey(0)
    Roberts(new_img)
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



if __name__=='__main__':
    
    # imgcolor = cv2.imread('yessica.jpg')      # cargar el archivo PNG indicado
    # cv2.imshow('Img Color', imgcolor)                                     
    # cv2.waitKey(0)

    img = cv2.imread('yessica.jpg', cv2.IMREAD_GRAYSCALE)      # cargar el archivo PNG indicado
    cv2.imshow('Img Origin', img)                                     
    cv2.waitKey(0)

    
    equalization(img)
    # media(img,12)
    # mediaPonderada(img,12)
    # Mediana(img,12)
    # Max(img,mask)
    # Min(img,mask)
    # Gaus(img)
    # Roberts(img)
    # laplace(img)
    #Gaus(img)
    #laplace(img)
    # Roberts(img)
   # Sobel(img)
    #Prewitt(img)