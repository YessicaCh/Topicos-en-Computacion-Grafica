
# from PIL import Image
# from scipy import ndimage
# import numpy as np
 
# imagen = Image.open("icono.png")
# datos = np.array(imagen)
# # /print(datos)
# imagen.show()
import cv2
import numpy as np
from math import exp
# cargar el archivo PNG indicado
img = cv2.imread('icono.png', cv2.IMREAD_GRAYSCALE)
# # mostrar la imagen en una ventana
cv2.imshow('titulo', img)
cv2.waitKey(0)
#cv2.close()

def ChangeColor(img):
    date=np.array(img)
    for x in range(0,len(date)):
        for y in range(0,len(date[0])):
            img[x][y]=(256-date[x][y])
    cv2.imwrite('resul.png',img)
    cv2.imshow('titulo2', img)
    cv2.waitKey(0)

def filterExp(img):
    date=np.array(img)
    for x in range(0,len(date)):
        for y in range(0,len(date[0])):
            img[x][y]=exp(date[x][y])
    cv2.imwrite('resul.png',img)
    cv2.imshow('titulo2', img)
    cv2.waitKey(0)


# ChangeColor(datos,img)
ChangeColor(img)
#filterExp(img)




