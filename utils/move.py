import os
import cv2
import shutil
import numpy as np


def main(root, save_root):
    classdir_list = os.listdir(root)
    for classdir in classdir_list:
        train_dir = os.path.join(root, classdir, 'train')
        test_dir = os.path.join(root, classdir, 'test')
        

if __name__ == "__main__":
    root = '/mnt/qiuzheng/data/DAGM/DAGM/DAGM2007'
    save_root = '/mnt/qiuzheng/data/DAGM/DAGM2007/all'
    main(root, save_root)