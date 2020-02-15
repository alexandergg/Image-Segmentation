import numpy as np
import argparse
import cv2

def nothing(x):
    pass

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True)
args = vars(ap.parse_args())

img = cv2.imread(args["image"], 0)
cv2.namedWindow('Treshed')
cv2.createTrackbar('Treshold','Treshed',0,255,nothing)

while(1):
    clone = img.copy()
    r = cv2.getTrackbarPos('Treshold','Treshed')
    ret,gray_threshed = cv2.threshold(clone,r,255,cv2.THRESH_BINARY)
    bilateral_filtered_image = cv2.bilateralFilter(gray_threshed, 5, 175, 175)
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
    _,contours,_ = cv2.findContours(edge_detected_image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        contour_list.append(contour)

    cv2.drawContours(clone, contour_list,  -1, (255,0,0), 2)

    blank = np.zeros(img.shape[0:2])
    img1 = cv2.drawContours(blank.copy(), contours, 0, 1)
    img2 = cv2.drawContours(blank.copy(), contours, 1, 1)

    intersection = np.logical_and(img1, img2)
    print(intersection)
    cv2.imshow('Objects Intersection', clone)
    cv2.imshow("Treshed", gray_threshed)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()