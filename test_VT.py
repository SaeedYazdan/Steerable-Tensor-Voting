import encoder
import voter
import perform_tensor_decomp as ptd
import compute_rigidity_tensor as crt

import cv2
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import copy


origimage = 'paper_con.jpg'
origimage2 = 'paper0.jpg'
origimage3 = 'paper.jpg'

#origimage = 'new_img.jpg'


def imshow(img):
    plt.imshow(img)
    plt.show()




img = cv2.imread(origimage, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(origimage2, cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(origimage2, cv2.IMREAD_GRAYSCALE)
img = img + img2
img = img / 2
imshow(img)
#img2 = copy.copy(img)
#img2[img2 < 80] = 0
#imshow(img2)


[s, o] = encoder.encode(img, file_name=False)
[saliency__, ballness, orientaion] = voter.vote(s, o, 10)

[s, o] = encoder.encode(saliency__, file_name=False)
[saliency__2, ballness, orientaion] = voter.vote(s, o, 10)

cv2.imwrite('saliency__10.jpg', 255*saliency__)
cv2.imwrite('saliency__12.jpg', 255*saliency__2)


exit()

[saliency, ballness, orientaion] = voter.vote(s, o, 15)


# Run contour to trim those short.
# or you can run the code with skeletonizer and then short onse get trimmed using the loops

exit()
im = saliency

ret, thresh1 = cv2.threshold(im, 0.2, 1, cv2.THRESH_BINARY)


import numpy as np
arr = np.uint8(thresh1)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(arr, connectivity=8)
areas = stats[1:,cv2.CC_STAT_AREA]
heig = stats[1:,cv2.CC_STAT_HEIGHT]
leng = stats[1:,cv2.CC_STAT_WIDTH]

result = np.zeros((labels.shape), np.uint8)


for i in range(nlabels - 1):
    if i % 1000 == 0:
	    print(i)
    if areas[i] >= 150 and (areas[i] / (heig[i] * leng[i])) <= 0.4:   #keep
        #prop = max(leng[i] / heig[i], heig[i] / leng[i])
        #if prop > 1.5 and areas[i] >= 30:
        result[labels == i + 1] = 255


#coordinates = peak_local_max(im, min_distance=20)

# Finding maximums
'''ws = 15

ST = crt.compute_rigidity_tensor(im)
for i1 in range(2):
    for j1 in range(2):
        ST[:, :, i1, j1] = cv2.GaussianBlur(ST[:, :, i1, j1],(3, 3), sigmaX=0.8, sigmaY=0.8)
[e10, e20, l10, l20] = ptd.perform_tensor_decomp(ST)
imshow(l10)
ret, thresh1 = cv2.threshold(l10, 0.00225, 1, cv2.THRESH_BINARY)
imshow(thresh1)'''
cv2.imwrite('output.jpg', result)



[s2, o2] = encoder.encode(result, file_name=False)

#exit()
[saliency2, ballness2, orientaion2] = voter.vote(s2, o2, 35)
ret, thresh2 = cv2.threshold(saliency2, 0.25, 1, cv2.THRESH_BINARY)
arr = np.uint8(thresh2)
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(arr, connectivity=8)
heig = stats[1:,cv2.CC_STAT_HEIGHT]
leng = stats[1:,cv2.CC_STAT_WIDTH]

result2 = np.zeros((labels.shape), np.uint8)
for i in range(nlabels - 1):
    if i % 1000 == 0:
	    print(i)
    if areas[i] >= 150 and (areas[i] / (heig[i] * leng[i])) <= 0.4:   #keep
        #prop = max(leng[i] / heig[i], heig[i] / leng[i])
        #if prop > 1.5 and areas[i] >= 30:
        result2[labels == i + 1] = 255
        
        
