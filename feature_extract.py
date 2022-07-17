import glob
from tools import sort_humanly
import cv2
import numpy as np

masks = glob.glob('F:\opencv\RAFT-NCUP-master\output/training01/*.jpg')
masks =sort_humanly(masks)
masks_generator = (cv2.imread(i, 0) for i in masks)

imgs = glob.glob('output/training01/*.jpg')
imgs =sort_humanly(imgs)

imgs_generator = (cv2.imread(i, 0) for i in imgs)
tracker = cv2.TrackerMedianFlow_create()

# print(next(masks_generator).shape)
# for i in masks_generator:
#     print(i)


for mask, img in zip(masks_generator, imgs_generator):
    img = cv2.resize(img, (640, 480))
    mask = cv2.resize(mask, (640, 480)).astype(np.uint8)
    ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    if not len(contours):
        continue
    x, y, w, h = cv2.boundingRect(contours[0])
    tracker.init(img, (x, y, w, h ))
    break
    # grab the new bounding box coordinates of the object
for  img in  imgs_generator:
    img = cv2.resize(img, (640, 480))

    (success, box) = tracker.update(img)
    # check to see if the tracking was a success
    if success:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(img, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)
    # cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(109)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
# gray = mask * 1 * gray
# gray = np.float32(gray)
#
# # 输入图像必须是float32， 最后一个参数[0.04,0.06]
# dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# cv2.imshow('dst', dst)
# dst = cv2.dilate(dst, None)
#
# img[dst > 0.01 * dst.max()] = [0, 0, 255]
# cv2.imshow('img', img)
# cv2.imshow('dst2', dst)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
