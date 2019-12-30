from __future__ import print_function
from aip import AipOcr
from aip import AipImageCensor
import cv2
import glob
import os
import shutil
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

APP_ID = '18104709'
API_KEY = '9mzaSjALI2BOGYcXwNgN6Yru'
SECRET_KEY = 'E1TWlIfgkhyVflXaXBYvW156A8iqHrgG'


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def draw_boxes(img, image_name, boxes, scale):
    base_name = image_name.split('\\')[-1]
    boxes_path =  "data/results/" + str(base_name.split('.')[0])
    ocr_words_results = []
    if not os.path.exists(boxes_path):
        os.makedirs(boxes_path)
    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for i, box in enumerate(boxes):
            min_x_= int(min(box[0], box[2], box[4], box[6]))
            min_y_= int(min(box[1], box[3], box[5], box[7]))
            max_x_= int(max(box[0], box[2], box[4], box[6]))
            max_y_= int(max(box[1], box[3], box[5], box[7]))
               
            box_img = img[min_y_:max_y_,min_x_:max_x_,:]
            box_path = os.path.join(boxes_path, str(i)+".jpg")
            cv2.imwrite(box_path, box_img)

            # 定义参数变量
            options = {
            'detect_direction': 'true',
            'language_type': 'ENG+JPN',
            }
            ocr_result = aipOcr.basicAccurate(get_file_content(os.path.join(boxes_path, str(i)+".jpg")), options)
            try:
                ocr_words_results.append(ocr_result['words_result'])
            except Exception as identifier:
                ocr_words_results.append([])

        for i, box in enumerate(boxes):
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)])
            for content in ocr_words_results[i]:
                try:
                    line += (','+content['words'])
                except Exception as e:
                    pass
            line += '\r\n'
            f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", base_name), img)


def bad_words_filter(img_name):

    base_name = img_name.split('\\')[-1]
    text = ' '
    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'r') as f:
        # text = f.read().split('\n')
        text = ' '
        # for i in text:
        #     temp += i
        lines = f.readlines()
        for line in lines:
            str_arr = line.strip().split(',')
            if len(str_arr) > 4:
                for word in str_arr[4:]:
                    text += (word+',')


    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'a') as f:
        client = AipImageCensor(appId='18097522', apiKey='bgeRXmpblbOMocErTxvMCiOF',
                                secretKey='SyRmwoFGsKXqC3LvAAYwlDlf1BckBOwG')
        if client.antiSpam(text)['result']['spam']==0:
            f.write('无敏感信息')
        else:
            f.write('待检测的文本里面含有侮辱、色情、暴力和政治敏感词汇。\n')
            for i in client.antiSpam(text)['result']['reject']:
                if (len(i['hit']) != 0):
                    f.write(str(i['hit']))
                # for key_word in i['hit']:
                #     f.write(key_word+',')


def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    # 加入截取图片
    draw_boxes(img, image_name, boxes, scale)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

if __name__ == '__main__':
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    aipOcr  = AipOcr(APP_ID, API_KEY, SECRET_KEY)

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name)
        bad_words_filter(im_name)
