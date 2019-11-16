def exp(i, j):
    m = min(i, j)
    n = max(i, j)
    return math.exp(m) - 1, math.exp(n) - 1

def filter_(foutput, i, j):
    return np.where((exp(i, j)[0] < foutput) & (foutput < exp(i, j)[1]) | (-exp(i, j)[1] < foutput) & (foutput < -exp(i, j)[0]), foutput, 0)

def plot_dis(image, title):
    h = image.flatten()
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))
    pl.plot(h,fit,'-o', linestyle="None")
    pl.hist(h,density=True)
    pl.savefig(title)
    pl.clf()

import math
import numpy as np 
import cv2
import scipy.stats as stats
import pylab as pl

image = cv2.imread("input.png", 0)
fimage = np.fft.fft2(image)
foutput_ = fimage 
fimage = np.log(np.abs(fimage) + 1)
fimage_ = fimage
fimage = fimage / np.amax(fimage) * 255
cv2.imwrite("result.png", fimage)

plot_dis(fimage_, 'distribution.png')

for i in range(1000):
    foutput = filter_(foutput_, 8-i/200, 8+i/200)
    output = np.fft.ifft2(foutput).real
    plot_dis(output, 'output_distribution.png')
    #cv2.imwrite("output.png", output.astype("uint8"))
    cv2.imshow("test", output.astype("uint8"))
    cv2.waitKey(10)
