# -*- coding: utf-8 -*-
import random
from PIL import Image, ImageFont, ImageFilter, ImageDraw
import numpy as np

#字体的位置，不同版本的系统会有影响
font_path='../fonts/arial.ttf'
#生成几位数的验证码
number = 4
#生成验证码图片的宽度和高度
size = (180,60)
#生成随机颜色背景图片
def getRandomColor():
    c1 = random.randint(0,250)
    c2 = random.randint(0, 250)
    c3 = random.randint(0, 250)
    return (c1,c2,c3)
#背景颜色
bgcolor = (255,255,255)
#字体颜色，默认为蓝色
fontcolor =getRandomColor()
fontsize = 32
#干扰线颜色，默认为红色
linecolor =(225,0,0)
#是否要加入干扰线
draw_line = True
#是否要加入噪点
draw_dot = True
#加入干扰线条数的上下限
line_number = 4
#加入噪点的个数
dot_number = 8


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


#随机生成一个字符串
def gene_text():
    list = []
    for i in range(0,10):
        list.append(str(i))
    for i in range(97,123):
        list.append(chr(i))
    for i in range(65,91):
        list.append(chr(i))
    return ''.join(random.sample(list,number))
    '''source =list(string.ascii_letters)
    for index in range(0,10):
        source.append(str(index))
    return ''.join(random.sample(source,number))'''

#用来绘制干扰线
def gene_line(draw,width,height):
    for i in range(line_number):
        begin = (random.randint(0,width),random.randint(0,height))
        end = (random.randint(0,width),random.randint(0,height))
        draw.line([begin,end],fill = getRandomColor())
def gene_dot(draw,width,height):
    for i in range(dot_number):
        draw.point([random.randint(0,width),random.randint(0,height)],fill=getRandomColor())
        x = random.randint(0,width)
        y = random.randint(0,height)
        draw.arc((x,y,x+4,y+4),0,90,fill=getRandomColor())
#生成验证码
def gene_code(x):
    #宽和高
    width,height = size
    #创建图片
    image = Image.new('RGBA',(width,height),bgcolor)
    #验证码字体
    font = ImageFont.truetype(font_path,fontsize)
    #创建画笔
    draw = ImageDraw.Draw(image)
    #生成字符串
    text = gene_text()
    font_width,font_height = font.getsize(text)
    #填充字符串
    draw.text(((width - font_width)/number,(height - font_height)/number),text,font=font,fill=getRandomColor())
    if draw_line:
        gene_line(draw,width,height)
    if draw_dot:
        gene_dot(draw,width,height)
    #创建扭曲
    rot = image.rotate(random.randint(-8, 8), expand=0)
    fff = Image.new('RGBA', rot.size, (255,) * 4)
    image = Image.composite(rot, fff, rot)
    #滤镜边界加强
    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    #保存验证码图片
    image.save("./image/test/%s.png"%(text))
    one_hot(text)

if __name__ == "__main__":
    for i in range(100):
        gene_code(i)