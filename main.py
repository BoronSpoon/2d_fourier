def exp(i, j):
    m = min(i, j)
    n = max(i, j)
    return math.exp(m) - 1, math.exp(n) - 1

def filter_(foutput, i, j):
    return np.where((exp(i, j)[0] < foutput) & (foutput < exp(i, j)[1]) | (-exp(i, j)[1] < foutput) & (foutput < -exp(i, j)[0]), foutput, 0)

def update(val):
    foutput = filter_(foutput_, smin.val, smax.val)
    output = ifft2(foutput)
    cv2.imshow("test", output.astype("uint8"))
    cv2.waitKey(10)

###
"""
def fft2(image):
    tf_image = tf.placeholder(shape=image.shape, dtype=tf.complex64)
    tf_fimage = tf.signal.fft2d(tf_image)
    session = tf.InteractiveSession()
    return session.run([tf_fimage,tf_image], feed_dict={tf_image:image})[0]

def ifft2(fimage):
    tf_fimage = tf.placeholder(shape=fimage.shape, dtype=tf.complex64)
    tf_image = tf.signal.ifft2d(tf_fimage)
    session = tf.InteractiveSession()
    return session.run([tf_image,tf_fimage], feed_dict={tf_fimage:fimage})[0]
"""
###

import math
import numpy as np
from numpy.fft import fft2, ifft2
#import tensorflow as tf
import cv2
from scipy import stats, ndimage
#import pylab as pl
import matplotlib
import matplotlib.pyplot as pl
from matplotlib.widgets import Slider
matplotlib.use('TkAgg')

axmin = pl.axes([0.25, 0.1, 0.65, 0.03])
axmax = pl.axes([0.25, 0.15, 0.65, 0.03])

image = cv2.imread("input.png", 0)
image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
cv2.imwrite("denoise_input.png", image)
fimage = fft2(image)
foutput_ = fimage 
fimage = np.log(np.abs(fimage) + 1)
fimage_ = fimage
fimage = fimage / np.amax(fimage) * 255
cv2.imwrite("result.png", fimage)

h = fimage_.flatten()
fit = stats.norm.pdf(h, np.mean(h), np.std(h))
fig, ax = pl.subplots()
pl.subplots_adjust(left=0.25, bottom=0.25)
pl.plot(h,fit,'-o', linestyle="None")
pl.hist(h,density=True)
ax.margins(x=0)
smin = Slider(axmin, 'min', np.min(h), np.max(h), valinit=0)
smax = Slider(axmax, 'max', np.min(h), np.max(h), valinit=0)
smin.on_changed(update)
smax.on_changed(update)
pl.show()
#pl.savefig("distribution.png")
#pl.clf()
