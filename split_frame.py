import cv2
import glob,os
def split(path,output_path):
    save_name = path.split('\\')[-1][:-4]
    output_dir = os.path.join(output_path,save_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stream = cv2.VideoCapture(path)
    contour = 0
    while True:
        ret,frame = stream.read()
        if ret:
            # print(os.path.join(output_dir,f'{str(contour)}.jpg'))
            cv2.imwrite(os.path.join(output_dir,f'{str(contour)}.jpg'),frame)
            contour +=1
        else:
            break
    print('DONE')

if __name__ == '__main__':
    path = 'Training_videos'
    videos = glob.glob(os.path.join(path,'*.mp4'))
    output_path='output_test'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in videos:
        split(i,output_path)
