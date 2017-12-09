import sys 
import numpy as np

keypoints = [
	[78.0, 0.0, 159.0],
	[46.0, 126.0, 255.0],
	[161.0, 255.0, 237.0],
	[239, 255, 177],
	[255.0, 254.0, 150.0],
	[255.0, 118.0, 69.0],
	[179.0, 0.0, 38.0],
	[0,0,0],
	[0,0,0]]

'''
keypoints = [
	[83, 83, 83],
	[66, 103, 163],
	[46, 126, 255],
	[78, 162, 250],
	[106, 193, 246],
	[135, 225, 241],
	[147, 253, 255],
	[0,0,0],
	[0,0,0]]
'''
bookmarks = []
for i in range(len(keypoints) - 3):
	bookmarks.append(i * 512/6)
	print bookmarks[i]

bookmarks.append(512)

array = np.zeros((512, 3), dtype=np.float)

k = 0
for i in range(len(keypoints) - 3):
	for j in range (bookmarks[1] - bookmarks[0]):
		array[i*bookmarks[1]-bookmarks[0] + j, 0] = keypoints[i][0] + j*((keypoints[i+1][0]-keypoints[i][0])/(bookmarks[1] - bookmarks[0]))
		array[i*bookmarks[1]-bookmarks[0] + j, 1] = keypoints[i][1] + j*((keypoints[i+1][1]-keypoints[i][1])/(bookmarks[1] - bookmarks[0]))
		array[i*bookmarks[1]-bookmarks[0] + j, 2] = keypoints[i][2] + j*((keypoints[i+1][2]-keypoints[i][2])/(bookmarks[1] - bookmarks[0]))
		if (i == 5):
			print array[i*bookmarks[1]-bookmarks[0] + j,0], 
			print array[i*bookmarks[1]-bookmarks[0] + j,1],
			print array[i*bookmarks[1]-bookmarks[0] + j,2]
		'''
		print i,
		print j,
		print k
		'''
		k = k + 1

# only gets to 510
array[510] = keypoints[6]
array[511] = keypoints[6]


'''
array = np.zeros((256, 3), dtype=np.float)
for i in range(85):
	array[i, 0] = i*(168.0/85)
	array[i, 1] = i*(40.0/85)
	array[i, 2] = i*(15.0/85)
for i in range(85):
	array[i+85, 0] = 168.0 + i*((243.0-168.0)/85.0)
	array[i+85, 1] = 40.0 + i*((194.0-40.0)/85.0)
	array[i+85, 2] = 15.0 + i*((93.0-15.0)/85.0)
for i in range(86):
	array[i+170, 0] = 243.0 + i*((255.0-243.0)/86.0)
	array[i+170, 1] = 194.0 + i*((255.0-194.0)/86.0)
	array[i+170, 2] = 93.0 + i*((255.0-93.0)/86.0)

for i in range(256):
	for j in range(3):
		array[i, j] /= 255.0
'''


for i in range (511):
	for j in range(3):
		array[i][j] /= 255.0


np.savetxt("Hot_Cold_No_Zero", array, fmt='%12.6f',delimiter=' ')


