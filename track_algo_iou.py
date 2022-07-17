import glob
from scipy.special import expit
import cv2
import numpy as np
from matplotlib import pyplot as plt


class Tracing_iou():
    def __init__(self, result):
        self.bug_nums = 0
        self.bug_idex = []
        self.track_pool = []
        self.result_series = result
        self.result_series_mask = self.result_series[:, 0, ...]
        self.result_series_head = self.result_series[:, 1, ...]
        self.result_series_mask, self.result_series_head = map(self.norm_series,
                                                               [self.result_series_mask, self.result_series_head])

    def norm_series(self, img):
        img  = expit(img)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img = np.array(img, dtype=np.uint8)
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
        for img in self.result_series_mask:
            ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_NONE)
            cnt_nums.append(len(contours))
        res, counts = np.unique(cnt_nums, return_counts=True)
        most = np.argmax(counts)
        self.bug_nums = res[most]
        print('[INFO] bug_nums should be :', self.bug_nums)

    def clean_result(self):
        kernel = np.ones((5, 5), np.uint8)

        for img in self.result_series_mask:
            ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 大津阈值
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
            pixel_area = np.array(stats)[:, -1]
            if len(pixel_area) >= self.bug_nums + 1:
                bug_idx = np.argsort(-pixel_area)[1:self.bug_nums + 1]
            else:
                miss_nr = self.bug_nums + 1 - len(pixel_area)
                print(f'[INFO] {miss_nr} bug(s) miss')

    def track(self):
        if self.bug_nums == 1:
            self.single_bug_track()
        else:
            self.multi_bugs_track()

    def single_bug_track(self):
        kernel = np.ones((3, 3), np.uint8)

        for mask, head in zip(self.result_series_mask,self.result_series_head):
            # _, mask_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 大津阈值
            _, head_binary = cv2.threshold(head, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值

            # mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
            head_binary = cv2.morphologyEx(head_binary, cv2.MORPH_CLOSE, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(head_binary)
            print(stats)
            pixel_area = np.array(stats)[:, -1]

            if len(pixel_area) >= self.bug_nums + 1:
                bug_idx = np.argsort(-pixel_area)[1]
            else:
                miss_nr = self.bug_nums + 1 - len(pixel_area)
                print(f'[INFO] {miss_nr} bug(s) miss')
                continue
            blank = np.zeros_like(head)
            print(bug_idx)
            box = stats[bug_idx][:4]

            mc = [box[0]+box[2]//2,box[1]+box[3]//2]
            print(mc)
            cv2.circle(blank, (int(mc[0]), int(mc[1])),radius=2,color=(255, 255, 255), thickness=-1)
            plt.imshow(head_binary)
            plt.show()
    def multi_bugs_track(self):
        pass



if __name__ == '__main__':
    path = 'tracing_mask/'
    files = glob.glob(path + '*.npy')
    s = np.load(files[0])
    t = Tracing_iou(s)
    t.get_bug_nums()
    t.show_mask(code=1)
    t.track()
