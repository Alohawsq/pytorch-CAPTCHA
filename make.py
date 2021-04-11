import random
from io import BytesIO

from PIL import Image, ImageFont, ImageFilter, ImageDraw
import numpy as np
from captcha.image import ImageCaptcha
from PIL import Image
import random
import time
import string
import os
font_path='../fonts/AdobeMingStd-Light.otf'
#生成验证码图片的宽度和高度
size = (90,35)
#生成随机颜色背景图片
def getRandomColor():
    c1 = random.randint(0,250)
    c2 = random.randint(0, 250)
    c3 = random.randint(0, 250)
    return (c1,c2,c3)
#字体颜色，默认为蓝色
fontcolor =getRandomColor()
fontsize = 34

def one_hot(text):
    vector = np.zeros(4 * 62)  # (10+26+26)*4

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * 62 + char2pos(c)
        # print(text,i,char2pos(c),idx)
        vector[idx] = 1.0
    return vector

#定义随机方法
def random_captcha():
    #做一个容器
    captcha_text = []
    for i in range(4):
        #定义验证码字符
        c = random.choice(string.digits + string.ascii_letters)
        captcha_text.append(c)
    #返回一个随机生成的字符串
    return ''.join(captcha_text)

def genImage1():
    # 定义图片对象
    # generator = ImageCaptcha(width=90, height=35, font_sizes=[fontsize])
    # 获取字符串
    captcha_text = random_captcha()
    # 生成图像
   # captcha_img = generator.create_captcha_image(captcha_text, getRandomColor(), (255, 255, 255)).convert("L")
    return captcha_text   #, captcha_img


# def genImage2():
#     # 定义图片对象
#     generator = ImageCaptcha(width=90, height=35, font_sizes=[fontsize])
#     # 获取字符串
#     captcha_text = random_captcha()
#     #生成图像
#     img = generator.create_captcha_image(captcha_text, getRandomColor(), (255, 255, 255))
#     for i in range(4):
#         img = generator.create_noise_curve(img, getRandomColor())
#     captcha_img = generator.create_noise_dots(img, getRandomColor(), 1, 14).convert("L")
#     return captcha_text, captcha_img
#
# if __name__ == '__main__':
#     f = genImage2()
#     im = f[0]
#     print(f[0])
#     print("\n")
#     print(f[1])
