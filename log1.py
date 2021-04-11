'''
m2mp.png
cX2s
D2VF.png
Ixpf
0633.png
J4Aw
p5vp.png
cX21
khxy.png
SCSm
pf2b.png
cii1
asaB.png
YPw4
8450.png
XA4Y
ja5f.png
X2Sr
bvuv.png
8Vuv
B603.png
Keda
2s99.png
2c2x
feBo.png
h4bd
QAVQ.png
IcQJ
OOQc.png
cbbd
p9pe.png
Y5Ga
eKDi.png
4HCJ
f98n.png
k221
6053.png
PpsX
6mzn.png
6mzn
gb2m.png
SS21
kwua.png
cS24
7024.png
XPww
Lypj.png
VMkD
G60Y.png
Q5xY
8tdq.png
XgXI
fn9m.png
k2ic
cw2x.png
cw2x
bh7q.png
bhfX
3542.png
x4kp
ed3r.png
XX2s
jkth.png
JcTd
JfZj.png
Vg2v
bv3f.png
7HA9
h6ro.png
H6Ro
4mwo.png
AmW0
a23j.png
cii0
VGAU.png
XgSp
2jts.png
C5rz
'''
# a = ['h','e','r','O']
# b = ['h','e','r','O']
# print(a == b)
import numpy as np
import torch
# a = [12, 453, 234, 643]
# b = torch.Tensor(a)
# print(b)
from PIL import Image
im = Image.open("./loadimage/2s99.png")
im = im.resize((10, 20))
im = im.convert("L")
img = np.array(im)
print(img)
