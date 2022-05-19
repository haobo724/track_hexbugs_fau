import cv2
import numpy as np
import os
import glob
from tools import sort_humanly, delet_contours


def crop_face(image, face_rect):
    top = max(face_rect[1], 0)
    left = max(face_rect[0], 0)
    bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
    right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
    return image[top:bottom, left:right, :]


def get_bb(img, mask):
    ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓
    delete_list = []
    if len(contours) == 0:
        cv2.putText(img, 'not found object', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
    else:
        contours = list(contours)
        for i in range(len(contours)):
            l = len(contours[i])
            area = cv2.contourArea(contours[i])
            if cv2.contourArea(contours[i]) < 100:
                delete_list.append(i)
            # delet contour 是序号
        contours = delet_contours(contours, delete_list)

        for cnt in contours:
            # 外接矩形框，没有方向角
            # x, y, w, h = cv2.boundingRect(cnt)
            newregion = cv2.boundingRect(cnt)

            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            mu = cv2.moments(box, False)
            try:
                mc = [mu['m10'] / mu['m00'], mu['m01'] / mu['m00']]
                cv2.drawMarker(img, (int(mc[0]), int(mc[1])), (0, 0, 255), thickness=2)
            except:
                print('no b')
    # img_new = crop_face(img, newregion)
    return img


def merge_mask_input_as_video():
    path_to_img_dirs= 'output'
    path_to_mask_dirs= '..\RAFT-NCUP-master\output'
    if not os.path.exists(path_to_img_dirs) or not os.path.exists(path_to_mask_dirs):
        raise FileNotFoundError('no dirs')

    img_dirs = sort_humanly(os.listdir(path_to_img_dirs))
    mask_dirs = sort_humanly(os.listdir(path_to_mask_dirs))
    for img_dir, mask_dir in zip(img_dirs[52:53], mask_dirs[52:53]):
        print(img_dir,mask_dir)
        imgs = sort_humanly(glob.glob(f'{path_to_img_dirs}/{img_dir}/*.jpg'))
        masks = sort_humanly(glob.glob(f'{path_to_mask_dirs}\{mask_dir}/*.jpg'))

        codec = cv2.VideoWriter_fourcc(*'mp4v')
        frameSize_s = cv2.imread(imgs[0]).shape[:2]
        rotation_flag = False

        if frameSize_s[0] / frameSize_s[1] > 1:  # (768, 1024)
            print('rotation start')
            rotation_flag = True
            # name = os.path.split(img_dir)[-1]+ '_rotated.mp4'
        # else:
        #     print(img_dir, mask_dir, 'skiped')
        #     continue
        name = os.path.split(img_dir)[-1] + '_new.mp4'
        # input()
        out_path = os.path.join('./video_output/', name)
        print(img_dir, mask_dir)
        out = cv2.VideoWriter(out_path, codec, 5, frameSize_s[::-1])

        for i, m in zip(imgs, masks):
            img = cv2.imread(i)
            mask = cv2.imread(m, 0)

            # if rotation_flag:
                # mask = np.rot90(mask, k=-1)

            concat = get_bb(img, mask)
            out.write(concat)
        out.release()


if __name__ == '__main__':
  merge_mask_input_as_video()
