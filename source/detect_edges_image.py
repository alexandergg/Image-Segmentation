import argparse
import cv2
import os
import numpy as np
import imutils
import math
from scipy.spatial import distance as dist

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--edge-detector", type=str, required=True)
ap.add_argument("-i", "--image", type=str, required=True)
args = vars(ap.parse_args())

class CropLayer(object):
	def __init__(self, params, blobs):
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0

	def getMemoryShapes(self, inputs):

		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])

		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H

		return [[batchSize, numChannels, H, W]]

	def forward(self, inputs):
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]

class Utils():
    @staticmethod
    def generate_mask(npimage):
        mask = np.zeros(npimage.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (10,10,300,160)
        cv2.grabCut(npimage,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        final_mask = npimage*mask2[:,:,np.newaxis]
        return final_mask

    @staticmethod
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    @staticmethod
    def loadingDetectorModels():
        print("Loading edge detector models...")
        protoPath = os.path.sep.join([args["edge_detector"], "deploy.prototxt"])
        modelPath = os.path.sep.join([args["edge_detector"], "hed_pretrained_bsds.caffemodel"])
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        cv2.dnn_registerLayer("Crop", CropLayer)
        return net

    @staticmethod
    def configEdgeNet(net):
        blob = cv2.dnn.blobFromImage(mask, scalefactor=3.0, size=(W, H),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False, crop=False)
        print("Performing edge detection...")
        net.setInput(blob)
        hed = net.forward()
        hed = cv2.resize(hed[0, 0], (W, H))
        hed = (255 * hed).astype("uint8")
        return hed

    @staticmethod
    def getAreas(contours):
        contours_area = []
        for con in contours:
            area = cv2.contourArea(con)
            if 100 < area < 100000000:
                contours_area.append(con)
        return contours_area

    @staticmethod    
    def calculationCircles(contours_area):
        contours_circles = []
        for con in contours_area:
            perimeter = cv2.arcLength(con, True)
            area = cv2.contourArea(con)
            if perimeter == 0:
                break
            circularity = 4*math.pi*(area/(perimeter*perimeter))
            if 0.5 < circularity < 1.2:
                M = cv2.moments(con)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.drawContours(clone, [con], -1, (0, 255, 0), 2)
                    cv2.circle(clone, (cX, cY), 5, (255, 255, 255), -1)
                else:
                    cX, cY = 0, 0
                contours_circles.append(con)
                Utils.calculationDiameter(con)
        return contours_circles

    @staticmethod
    def calculationDiameter(contour):
        pixelsPerMetric = None
        box = cv2.minAreaRect(contour)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        (tl, tr, br, bl) = box

        (tltrX, tltrY) = Utils.midpoint(tl, tr)
        (blbrX, blbrY) = Utils.midpoint(bl, br)

        (tlblX, tlblY) = Utils.midpoint(tl, bl)
        (trbrX, trbrY) = Utils.midpoint(tr, br)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if pixelsPerMetric is None:
            pixelsPerMetric = dB / 3.5

        dimA = round((dA / pixelsPerMetric) * 2.54, 2)
        dimB = round((dB / pixelsPerMetric) * 2.54, 2)

        print("Width:"+ str(dimA) + " cm " + "Height:"+ str(dimB) + " cm")

if __name__ == "__main__":

    net = Utils.loadingDetectorModels()
    image = cv2.imread(args["image"])
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    (H, W) = image.shape[:2]
    clone = image.copy()

    mask = Utils.generate_mask(clone)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)

    hed = Utils.configEdgeNet(net)
    _,contours,_ = cv2.findContours(hed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_area = Utils.getAreas(contours)

    contours_circles = Utils.calculationCircles(contours_area)
    print("Total Timbers:" + str(len(contours_circles)))
    cv2.drawContours(clone, contours_circles, -1, (0, 0,255), 2)
    cv2.imshow("Treshed", hed)
    cv2.imshow("Image", clone)
    cv2.waitKey(0)