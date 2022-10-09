import cv2
import os

if __name__ == '__main__':
    vid = r'/home/rom/Downloads/1663421258645881.webm'
    out = '/home/rom/Downloads/vidOut'
    cap = cv2.VideoCapture(vid)
    if not os.path.isdir(out):
        os.mkdir(out)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        f_path = os.path.join(out, '{}.jpg'.format(idx))
        cv2.imwrite(f_path, frame)
        idx += 1


