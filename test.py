import tensorflow as tf
import numpy as np
import glob
from time import time
from PIL import Image
import math
import os.path
def test(sess,i,CSRecon,X_input):
    CS_ratio = 50  # 1, 4, 10, 25, 30,, 40, 50
    if CS_ratio == 4:
        n_input = 43
    elif CS_ratio == 1:
        n_input = 10
    elif CS_ratio == 10:
        n_input = 109
    elif CS_ratio == 25:
        n_input = 272
    elif CS_ratio == 30:
        n_input = 327
    elif CS_ratio == 40:
        n_input = 436
    elif CS_ratio == 50:
        n_input = 545  
    Test_Img = './Set%d/'%(i)
    def imread_CS_py(imgName):
        block_size = 33
        Iorg = np.array(Image.open(imgName), dtype='float32')
        [row, col] = Iorg.shape
        row_pad = block_size - np.mod(row, block_size)
        col_pad = block_size - np.mod(col, block_size)
        Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
        Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
        [row_new, col_new] = Ipad.shape
        return [Iorg, row, col, Ipad, row_new, col_new]

    def img2col_py(Ipad, block_size):
        [row, col] = Ipad.shape
        row_block = row / block_size
        col_block = col / block_size
        block_num = int(row_block * col_block)
        img_col = np.zeros([block_size ** 2, block_num])
        count = 0
        for x in range(0, row - block_size + 1, block_size):
            for y in range(0, col - block_size + 1, block_size):
                img_col[:, count] = Ipad[x:x + block_size, y:y + block_size].reshape([-1])
                # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
                count = count + 1
        return img_col

    def col2im_CS_py(X_col, row, col, row_new, col_new):
        block_size = 33
        X0_rec = np.zeros([row_new, col_new])
        count = 0
        for x in range(0, row_new - block_size + 1, block_size):
            for y in range(0, col_new - block_size + 1, block_size):
                X0_rec[x:x + block_size, y:y + block_size] = X_col[:, count].reshape([block_size, block_size])
                count = count + 1
        X_rec = X0_rec[:row, :col]
        return X_rec

    def psnr(img1, img2):
        img1.astype(np.float32)
        img2.astype(np.float32)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    if i==68:
       filepaths = glob.glob(Test_Img + '/*.png')
    else:
        filepaths = glob.glob(Test_Img + '/*.tif')

    

    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)

    for img_no in range(ImgNum):
        imgName = filepaths[img_no]

        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(imgName)
        Icol = img2col_py(Ipad, 33).transpose() / 255.0
        Img_input = Icol
        Img_output = Icol

        start = time()
        CSRecon_value = sess.run(CSRecon, feed_dict={X_input: Img_input})
        end =time()

        X_rec = col2im_CS_py(CSRecon_value.transpose(), row, col, row_new, col_new)

        rec_PSNR = psnr(X_rec * 255, Iorg)
        PSNR_All[0, img_no] = rec_PSNR

    output_data = "Avg PSNR is %.2f dB \n" % (np.mean(PSNR_All))
    print(output_data)
    print("Reconstruction READY")
    return (np.mean(PSNR_All))
