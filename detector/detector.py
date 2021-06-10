# USAGE
# python find_screen.py --query queries/query_marowak.jpg

# import the necessary packages
from skimage import exposure
import numpy as np
import argparse
import imutils
import cv2
from time import time
begin = time()
from support import predictScript
import os

saveWarp = True

# construct the argument parser and parse the arguments
#imagePath ='./queries/tdms_2.jpg'

#imagePath = 'queries/sarma.jpeg'
#imagePath ='./queries/bewd_0.jpg'
#imagePath ='./queries/fhd_0.jpg'
imagePath ='./queries/tdm_1.jpg'
#imagePath ='./queries/tw_1.jpg'

assert os.path.exists(imagePath)

useCam = False

resizeHeight = 300
print('Initalize predictScript: ', time()-begin,'s')
print('Starting script')


def takeCameraPic():
    cam = cv2.VideoCapture(1) # index 1 to access our usb camera
    #cam.set(cv2.CAP_PROP_AUTOFOCUS,0)
    img_counter = 0
    i = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        
        scale_percent = 100 # percent of original size

        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        frameShow = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        cv2.imshow("Take picture of card", frameShow)


        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            cam.release()
            cv2.destroyAllWindows()
            '''
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            '''
            return frame

    cam.release()
    cv2.destroyAllWindows()
    return None





def predictCard(useCamera=True):
    # load the query image, compute the ratio of the old height
    # to the new height, clone it, and resize it
    
    image = None
    if useCamera:
        image = takeCameraPic()
    
    if image is None:
        print('Loading image from a preset path!')
        assert os.path.exists(imagePath), imagePath
        image = cv2.imread(imagePath)
    begin = time()


    ratio = image.shape[0] / 300.0
    orig = image.copy()
    image = imutils.resize(image, height = resizeHeight)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    #gray = cv2.equalizeHist(gray)
    edged = cv2.Canny(gray, 30, 200)
    #edged = cv2.Canny(gray, 0, 200)
    #edged = cv2.Canny(gray, 130, 200)
    ret,edged = cv2.threshold(edged,127,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    edged = cv2.dilate(edged,kernel,iterations = 3)
    edged = cv2.erode(edged,kernel,iterations = 3)
    '''
    cv2.imshow('d',edged)
    cv2.waitKey(0)
    '''
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    #cv2.imshow('tmp',edged)
    #cv2.imshow('gray',gray)
    #cv2.waitKey(0)

    screenCntList = []

    # loop over our contours
    for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)

            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                    print('Approx 4')
                    #screenCnt = approx
                    screenCntList.append(approx)
                    break

    print('Image preprocessing time: ', time()-begin,'s')

    for i,screenCnt in enumerate(screenCntList):
        # now that we have our screen contour, we need to determine
        # the top-left, top-right, bottom-right, and bottom-left
        # points so that we can later warp the image -- we'll start
        # by reshaping our contour to be our finals and initializing
        # our output rectangle in top-left, top-right, bottom-right,
        # and bottom-left order
        pts = screenCnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype = "float32")

        # the top-left point has the smallest sum whereas the
        # bottom-right has the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # compute the difference between the points -- the top-right
        # will have the minumum difference and the bottom-left will
        # have the maximum difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # multiply the rectangle by the original ratio
        rect *= ratio

        # now that we have our rectangle of points, let's compute
        # the width of our new image
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        # construct our destination points which will be used to
        # map the screen to a top-down, "birds eye" view
        dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype = "float32")

        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

        # convert the warped image to grayscale and then adjust
        # the intensity of the pixels to have minimum and maximum
        # values of 0 and 255, respectively
        warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        #warp = exposure.rescale_intensity(warp, out_range = (0, 255))
        #cv2.imshow('warp', warp)
        #cv2.waitKey(0)

        # the pokemon we want to identify will be in the top-right
        # corner of the warped image -- let's crop this region out
        (h, w) = warp.shape
        (dX, dY) = (int(w * 0.4), int(h * 0.45))
        crop = warp[10:dY, w - dX:w - 10]

        # save the cropped image to file
        warpSavePath = 'warp_' + str(i) +'.png'
        #cv2.imwrite("cropped.png", crop)
        #cv2.imwrite(warpSavePath,imutils.resize(warp, height = 300))

        warpImg = imutils.resize(warp, height = 300)
        # show our images
        print(warpImg.dtype)
        print(warpImg.shape)
        #print(type(warpImg))
        #inputWarpImg = warpImg.astype(np.uint8)
        inputWarpImg = warpImg
        orb = cv2.ORB_create()

        if saveWarp:
            cv2.imwrite(warpSavePath, inputWarpImg)
        #inputWarpImg = cv2.imread(warpSavePath,0)
        '''
        cv2.imshow('test',inputWarpImg)
        cv2.waitKey(0)
        '''
        #cv2.imshow('inputWarpImg',inputWarpImg)
        prediction = predictScript.predictCard(inputWarpImg ,orb)

        # Now we are just printing information on the card image
        resStr = '-'.join(prediction.split('-')[:-2])


        newHeight = 1000
        orig = imutils.resize(orig, height = newHeight)

        screenCnt = (screenCnt*(newHeight/resizeHeight)).astype(int)
        x0,y0 = screenCnt[0][0]
        x1,y1 = screenCnt[1][0]
        x2,y2 = screenCnt[2][0]
        x3,y3 = screenCnt[3][0]

       

        lineColor = (255,0,0)
        #lineColor = (0,255,0)

        cv2.line(orig, (x0,y0),(x1,y1), lineColor,thickness=8,lineType=8)
        cv2.line(orig, (x1,y1),(x2,y2), lineColor,thickness=8,lineType=8)
        cv2.line(orig, (x2,y2),(x3,y3), lineColor,thickness=8,lineType=8)
        cv2.line(orig, (x3,y3),(x0,y0), lineColor,thickness=8,lineType=8)


        res = cv2.putText(orig, resStr, (x1, y1-30) , cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0) , 2, cv2.LINE_AA) 
        #res = cv2.putText(orig, resStr, (x0, y0-30) , cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0) , 2, cv2.LINE_AA) 


        cv2.drawContours(orig,screenCnt, -1,(0,255,0),3)
        #cv2.drawContours(orig,screenCnt, -1,(0,0,255),3)
        cv2.imwrite('tmp.png',res)

        cv2.imshow("image", orig)
        #cv2.imshow("edge", edged)
        cv2.imshow("warp", warpImg)
        #cv2.imshow("crop", imutils.resize(crop, height = 300))
        cv2.waitKey(0)

    return        


predictCard(useCam)


