import os
import cv2
import shutil
import numpy as np

def main(root, save_root):
    classdir_list = os.listdir(root)
    for classdir in classdir_list:
        save_dir = os.path.join(save_root, classdir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for split in ['Train', 'Test']:
            save_train_dir = os.path.join(save_dir, 'train')
            save_test_dir = os.path.join(save_dir, 'test')
            if not os.path.exists(save_train_dir):
                os.mkdir(save_train_dir)
            if not os.path.exists(save_test_dir):
                os.mkdir(save_test_dir)
            img_dir = os.path.join(root, classdir, split)
            img_list = os.listdir(img_dir)
            for imgname in img_list:
                if '.PNG' in imgname:
                    lbname = imgname.replace('.PNG', '_label.PNG')
                    lb_path = os.path.join(img_dir, 'Label', lbname)
                    img_path = os.path.join(img_dir, imgname)
                    if os.path.exists(lb_path):
                        save_path = os.path.join(save_test_dir, imgname)
                        shutil.copy(img_path, save_path)
                    else:
                        save_path = os.path.join(save_train_dir, imgname)
                        shutil.copy(img_path, save_path)


if __name__ == "__main__":
    root = '/mnt/qiuzheng/data/DAGM'
    save_root = '/mnt/qiuzheng/data/DAGM/DAGM2007'
    main(root, save_root)