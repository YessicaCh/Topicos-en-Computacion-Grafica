import numpy as np
import math 
import cv2

img = cv2.imread('imagenesTCG/cameraman.jpg', cv2.IMREAD_GRAYSCALE)      # cargar el archivo PNG indicado

cv2.imshow('Img Origin', img)                                       # mostrar la imagen en una ventana
cv2.waitKey(0)

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
# ChangeColor(img)
logg(img)
rotation(img,90)
scaling(img,0.5)

