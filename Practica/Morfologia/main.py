import cv2
import numpy as np

kernel = np.ones((7,7),np.uint8)

def erocion(img):
    erosion = cv2.erode(img,kernel,iterations = 1)
    cv2.imshow('erosion', erosion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('erosion.png', out)

def dilatacion(img):
    erosion = cv2.dilate(img,kernel,iterations = 1)
    cv2.imshow('dilatacion', erosion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('dilatacion.png', out)


if __name__=='__main__':
    
    img = cv2.imread('../PruebasImagenes/coins.png',cv2.IMREAD_GRAYSCALE)      # cargar el archivo PNG indicado
    erocion(img)
    #dilatacion(img)
    #RGB_to_HSV(img)
    #model_RGB(img)
    #RGB_to_CMY(img)
   # RGB_to_CMYK(img)
    #RGB_to_HSV(img)
    #HSI(img)
    #RGB_to_YUV(img)
    #--RGB_to_YIQ(img)
#    RGB_to_YCrCb(img)
"""img2 = cv2.imread('../PruebasImagenes/lena.jpg', cv2.IMREAD_GRAYSCALE)      # cargar el archivo PNG indicado
                cv2.imshow('Img Origin', img2)                                     
                cv2.waitKey(0)"""

    # maskHorizontal_line(img2)
    # mask45_line(img2)
    # maskVertical_line(img2)
    # maskmenus45_line(img2)
    
