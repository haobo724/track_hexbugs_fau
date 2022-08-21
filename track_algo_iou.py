import glob
import math
from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
# from dummytest import KalmanFilter
from tools import delet_contours

# kf0 = KalmanFilter()
# kf1 = KalmanFilter()
# kf2 = KalmanFilter()
# kf3 = KalmanFilter()

class Tracing_iou():
    def __init__(self, result,video):
        self.bug_nums = 0
        self.track_pool = []
        self.id_pool = {}
        self.result_series = result
        self.result_series_mask = self.result_series[:, 0, ...]
        self.result_series_head = self.result_series[:, 1, ...]
        self.result_series_mask, self.result_series_head = map(self.norm_series,
                                                               [self.result_series_mask, self.result_series_head])
        self.real_video = video

    def norm_series(self, img):
        img = expit(img) > 0.8
        img = img * 255
        # img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img = np.array(img, dtype=np.uint8)
        # for i in img:
        #     plt.imshow(i)
        #     plt.show()
        #     ret, thresh = cv2.threshold(i, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
        #     print(ret)
        #     plt.imshow(thresh)
        #     plt.show()
        return img

    def show_mask(self, code=0):
        plt.figure()
        if code == 0:
            imgs = self.result_series_mask
        else:
            imgs = self.result_series_head
        for i in imgs:
            plt.imshow(i)
            plt.show()

    def get_bug_nums(self):
        cnt_nums = []
        kernel = np.ones((3,3))
        for img in self.result_series_mask:
            thresh = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            delete_list = []
            # ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_NONE)
            contours = list(contours)
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                if area < 100:
                    delete_list.append(i)
                # delet contour 是序号
            contours = delet_contours(contours, delete_list)

            cnt_nums.append(len(contours))

        res, counts = np.unique(cnt_nums, return_counts=True)
        most = np.argmax(counts)
        self.bug_nums = res[most]
        print('[INFO] bug_nums should be :', self.bug_nums)


    def clean_result_cnts(self):
        '''
        分割结果的post processing，目的是去除细碎小物体
        没用连通域，因为给的东西太冗余，并且背景还会被计算，很烦
        用contour一样还快


        :return:
        '''
        for frame_nr, img in enumerate(self.result_series_mask):

            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓
            delete_list = []
            if len(contours) == 0:
                cv2.putText(img, 'not found object', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
            else:
                contours = list(contours)
                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    if area < 100:
                        delete_list.append(i)
                    # delet contour 是序号
                contours = delet_contours(contours, delete_list)
            blank = np.zeros_like(img)
            for cnt in contours:
                cv2.drawContours(blank, [cnt], 0, (255, 255, 255), -1)
            self.result_series_mask[frame_nr, ...] = blank

    def track(self):
        if self.bug_nums == 1:
            self.multi_bugs_track()
        else:
            self.multi_bugs_track()

    def single_bug_track(self):
        '''
        废弃
        :return:
        '''
        kernel = np.ones((3, 3), np.uint8)

        for mask, head in zip(self.result_series_mask, self.result_series_head):
            _, head_binary = cv2.threshold(head, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值

            head_binary = cv2.morphologyEx(head_binary, cv2.MORPH_CLOSE, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(head_binary)
            pixel_area = np.array(stats)[:, -1]

            if len(pixel_area) >= self.bug_nums + 1:
                bug_idx = np.argsort(-pixel_area)[1]
            else:
                res = np.unique(mask)
                if len(res) <= self.bug_nums + 1:
                    miss_nr = self.bug_nums + 1 - len(pixel_area)
                    print(f'[INFO] {miss_nr} bug(s) miss')
                    self.track_pool.append([0, 0])
                    continue
                else:
                    # 应该根据mask继续检测，太麻烦了，先不写
                    pass
                    continue
            box = stats[bug_idx][:4]
            mc = [box[0] + box[2] // 2, box[1] + box[3] // 2]  # x,y
            self.track_pool.append(mc)

            # blank = np.zeros_like(head)
            # cv2.circle(blank, (int(mc[0]), int(mc[1])),radius=2,color=(255, 255, 255), thickness=-1)
            # plt.imshow(head_binary)
            # plt.show()

    def multi_bugs_track(self):
        '''
        根据分割结果的质心追踪，knn
        bug数取决于整个视频流分割结果，取的众数
        如果某个bug丢了那么追踪框保持上次状态，继续丢继续保持，
        计划如果中途丢了就用kf补上，但是这块没完全写完
        id建立取决于第一帧分割结果，存在隐患待完善
        无法应对中途加bug这种操作(大概)
        :return:
        '''
        for mask ,frame in zip(self.result_series_mask,self.real_video):

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_NONE)
            red = 0
            for cnt in contours:
                # 外接矩形框，没有方向角
                x, y, w, h = cv2.boundingRect(cnt)
                box = [x, y, w, h]
                red +=80
                cv2.rectangle(frame, (x, y), (x + w, y + h), (red, 255, 255), 2)
                mc = (box[0] + box[2] // 2, box[1] + box[3] // 2)  # x,y
                self.track_pool.append(mc)
            if not len(self.track_pool) :
                '''
                全丢了就下张图再见
                '''
                print('[INFO] Bugs all missed')
                continue

            if not self.id_pool:
                '''
                初始化bug池
                '''
                i = 0
                for pt in self.track_pool:
                    self.id_pool.setdefault(str(i), pt)
                    i += 1
                self.track_pool = []
                continue

            distance_matrix = []
            for id, value in self.id_pool.items():
                d = []
                for pt in self.track_pool:  # 当前frame 的几个点
                    distance = math.hypot(value[0] - pt[0], value[1] - pt[1])
                    d.append(distance)

                distance_matrix.append(d)
            distance_matrix = np.array(distance_matrix)

            indices,indices2 = linear_sum_assignment(distance_matrix)
            for id1, id2 in zip(indices, indices2):
                self.id_pool[str(id1)] = self.track_pool[id2]

            # prevFrame_pt_id = np.argmin(distance_matrix,axis=0) # axis --------------------1 横一竖0
            # curFrame_pt_id = np.argmin(distance_matrix,axis=1) # axis --------------------1 横一竖0
            # if len(curFrame_pt_id)!=len(prevFrame_pt_id):
            #     '''
            #     如果丢了bug，需要判断下丢了哪个
            #     '''
            #     curFrame_pt_id=curFrame_pt_id[prevFrame_pt_id]
            #
            #     for id1,id2 in zip(prevFrame_pt_id,curFrame_pt_id):
            #         self.id_pool[str(id1)] = self.track_pool[id2]
            # else:
            #     for id1,id2 in zip(range(self.bug_nums),curFrame_pt_id):
            #         self.id_pool[str(id1)] = self.track_pool[id2]


            # _ = kf0.predict(self.id_pool['0'])
            # un_updated_idx = (set(prevFrame_pt_id.tolist())^set(list(range(self.bug_nums))))

            # if len(un_updated_idx):
            #     for i in un_updated_idx:
            #         print(i)
            #         predicted = kf0.predict(self.id_pool[str(i)])
            #         self.id_pool[str(i)] = predicted

            for key,value in self.id_pool.items():
                cv2.putText(frame, key, value, 1, 2, (0, 0, 255), 1)
                cv2.drawMarker(frame, value, (0, 0, 255), thickness=2)

            self.track_pool = []

            plt.imshow(frame)
            plt.show()


if __name__ == '__main__':
    video_path = cv2.VideoCapture(r'Training_videos\training016.mp4')
    video=[]
    while True:
        ret,frame = video_path.read()
        if not ret:
            break
        frame = cv2.resize(frame,(224,224))
        video.append(frame)
    video = np.array(video)
    path = 'tracing_mask/'
    # files = glob.glob(path + '*.npy')
    s = np.load(r'F:\opencv\tracing\tracing_mask\training016.npy')
    t = Tracing_iou(s,video)
    # t.show_mask(code=1)

    t.get_bug_nums()
    t.clean_result_cnts()
    t.track()
