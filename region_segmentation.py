import numpy as np
import cv2
# from PIL import Image
import os

# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
        base_path=os.path.abspath(os.path.dirname(__file__)))
OUT_PATH = "{base_path}/out/".format(base_path=os.path.abspath(os.path.dirname(__file__)))
DOWNLOAD_PATH = "{base_path}/images".format(base_path=os.path.abspath(os.path.dirname(__file__)))


def facedetection(image):
    # convert the image to grayscale, load the face cascade detector and detect faces in the image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    drects = detector.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5,
                                       minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # construct a list of bounding boxes from the detection
    drects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in drects]

    # Ignore false faces since multiple faces are detected on certain images
    height, width = img_gray.shape
    drects.sort(key=lambda x: x[1], reverse=False)
    if len(drects) > 0:
        rects = drects[:1]
        lx, ly, rx, ry = rects[0]
        if ry > height / 2 or np.var(img_gray[ly:ry, lx:rx]) < 1:
            rects = []
    else:
        rects = []

    return rects


def findbound(img):
    height, width = img.shape

    # Find the rectangle boundary in four directions
    for i in range(width):
        if img[:, i].tolist().count(255) > 2:
            left = i
            break

    for i in range(width - 1, -1, -1):
        if img[:, i].tolist().count(255) > 2:
            right = i
            break

    for i in range(height):
        if img[i, :].tolist().count(255) > 2:
            top = i
            break

    for i in range(height - 1, -1, -1):
        if img[i, :].tolist().count(255) > 2:
            bottom = i
            break

    bound = [(top, bottom, left, right)]
    return bound


def filterboundarea(img, bvalue, svalue, rect):
    top, bottom, left, right = rect

    mask = np.zeros(img.shape[:2], np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for (y, x), value in np.ndenumerate(img_gray):
        if y > top and y < bottom and x > left and x < right:
            if img_gray[y, x] > bvalue + 20 or img_gray[y, x] < bvalue - 20:
                mask[y, x] = 1

    if not np.isnan(svalue):
        for (y, x), value in np.ndenumerate(img_gray):
            if y > top and y < bottom and x > left and x < right:
                if img_gray[y, x] < svalue + 50 and img_gray[y, x] > svalue - 50:
                    mask[y, x] = 0

    fimg = img * mask[:, :, np.newaxis]
    return fimg


def backgroundmodel(img, rect):
    top, bottom, left, right = rect
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = np.ones(img.shape[:2], np.uint8)
    mask[top:bottom, left:right] = 0
    img_gray = img_gray * mask
    bvalue = np.median(img_gray[img_gray > 0])

    return bvalue


def skinmodel(img, rect):
    startX, startY, endX, endY = rect
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crop_img = img_gray[startY:endY, startX:endX]



    hist = [0] * 256
    for (y, x), value in np.ndenumerate(crop_img):
        hist[crop_img[y, x]] += 1

    return hist.index(max(hist))


def removebackground(image, rects):
    # select left corner 15x15 patch
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    height, width = img_gray.shape

    v = np.average(img_gray[1:16])
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img_blur, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=3)

    # Find the bound area
    boundrect = findbound(dilated)

    # Get the background RGB model
    bvalue = backgroundmodel(image, boundrect[0])

    # Get the skin model
    if len(rects) > 0:
        svalue = skinmodel(image, rects[0])
    else:
        svalue = np.nan

    # Exclude face area
    if len(rects) > 0:
        lx, ly, rx, ry = rects[0]
        top, bottom, left, right = boundrect[0]
        boundrect = []
        boundrect = [(ry, bottom, left, right)]


    # Filter out background in bounded area
    rgimg = filterboundarea(image, bvalue, svalue, boundrect[0])
    return rgimg, boundrect


def extractobj(filepath):
    image = cv2.imread(filepath)
    filename = filepath.split('/')[-1]
    dotidx = filename.index('.')
    new_filename = OUT_PATH + filename[:dotidx] + "_cut" + filename[dotidx:]

    print(new_filename)

    # face detection
    rects = facedetection(image)

    # detect and remove background
    fimg, boundrect = removebackground(image, rects)

    top, bottom, left, right = boundrect[0]
    crop_img = image[top:bottom, left:right]
    # cv2.imshow('a',crop_img)

    cv2.imwrite(new_filename, crop_img)
    # print(boundrect)
    return crop_img


# img = extractobj('/home/duongbk/PycharmProjects/DPT/transfer_learning/266_875.jpg')


def CutAll():
    # Get the file list in saved Google drive folder
    img_list = []
    for fimg in os.listdir(DOWNLOAD_PATH):
        if fimg.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_list.append(os.path.join(DOWNLOAD_PATH, fimg))

    print("Total number:%d" % len(img_list))

    for img_path in img_list:
        extractobj(img_path)

    # Process the files in that folder in multi-threads
    # pool = ThreadPool(num_thread)
    # pool.map(extractobj, img_list)
    # pool.close()
    # pool.join()

    return len(img_list)

# CutAll()
