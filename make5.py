from captcha.image import ImageCaptcha
from PIL import Image
import random
import time
import string
import os

def getRandomColor():
    c1 = random.randint(0,250)
    c2 = random.randint(0, 250)
    c3 = random.randint(0, 250)
    return (c1,c2,c3)

fontsize = 28
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


#生成验证码方法
def gen_captcha():
    #定义图片对象
    generator = ImageCaptcha(width=90,height=35,font_sizes=[fontsize])
    #获取字符串
    captcha_text = random_captcha()
    #生成图像
    # captcha_img = Image.open(image.generate(captcha_text))
    # return captcha_text,captcha_img
    img = generator.create_captcha_image(captcha_text, getRandomColor(),(255,255,255))
    for i in range(0):
        img = generator.create_noise_curve(img, getRandomColor())
    captcha_img = generator.create_noise_dots(img, getRandomColor(), 0,0)
    return captcha_text, captcha_img


if __name__ == "__main__":
    #定义图片个数
    count = 10
    #定义图片文件夹
    path = './img/train'
    #如果没有就创建
    if not os.path.exists(path):
        os.makedirs(path)
    #循环创建图片
    for i in range(count):
        #定义创建时间
        now = str(int(time.time()))
        # print(type(now))
        #接收字符串和图片
        text,image = gen_captcha()
        #定义图片名称
        filename = text.lower() + '.png'
        #存储图片   os.path.sep  自动获取是文件夹还是文件
        image.save(path + os.path.sep + filename)
        # print('saved %s' % filename)
