import cv2
import numpy as np

img=cv2.imread('poster_1.jpg')


#grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#노이즈 제거
def remove_noise(image):
    return cv2.medianBlur(image,5)



#임계값 지정
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


#확장성
def dilate(image):
    kernel=np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations=1)



#침식, 부식
def erode(image):
    kernel=np.ones((5,5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)



#침식 후 확장
def opening(image):
    kernel=np.ones((5,5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


#canny edge
def canny(image):
    return cv2.Canny(image, 100,200)



#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)



gray=get_grayscale(img)
thresh=thresholding(gray)
opening=opening(gray)
canny=canny(gray)
print(gray)
print(thresh)
print(opening)
print(canny)
cv2.imshow('grayzz',gray)
cv2.imshow('thresh',thresh)
cv2.imshow('opening',opening)
cv2.imshow('canny',canny)
cv2.imwrite('gray\grayposter_1.jpg', gray)
cv2.imwrite('thresh\threshposter_1.jpg', thresh)
cv2.imwrite('opening\openingposter_1.jpg', opening)
cv2.imwrite('canny\cannyposter_1.jpg', canny)
