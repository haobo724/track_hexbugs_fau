import glob
import os

import cv2
from tqdm import tqdm


def split(path, output_path, sample_rate=10):
    save_name = path.split('\\')[-1][:-4]
    output_dir = os.path.join(output_path, save_name)
    print(f'[INFO] Video is {save_name}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stream = cv2.VideoCapture(path)
    frame_nr = 0
    with tqdm(total=stream.get(cv2.CAP_PROP_FRAME_COUNT)) as bar:  # total表示预期的迭代次数
        while True:
            ret, frame = stream.read()
            if not ret:
                break
            if frame_nr % sample_rate == 0:
                cv2.imwrite(os.path.join(output_dir, '{}_{:0>5}.jpg'.format(save_name, frame_nr)), frame)
            bar.update(1)
            frame_nr += 1

        print('DONE')


if __name__ == '__main__':
    path = r'F:\semantic_segmentation_unet\video'
    output_path = 'output_test'
    print(f'[INFO] input path is {path}')
    print(f'[INFO] output path is {os.path.abspath(output_path)}')
    videos = glob.glob(os.path.join(path, '*.mp4'))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in videos:
        split(i, output_path)
