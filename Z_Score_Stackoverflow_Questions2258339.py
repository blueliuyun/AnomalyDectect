# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:02:11 2018

Reference : https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/43512887#43512887
# Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703

@author: tianye
"""
import numpy as np
import pylab

def thresholding_algo(y, lag, threshold, influence):
    nFlag = 0x00
    signals = np.zeros(len(y)) # 返回长度为 len(y) 的数组。
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag], ddof=1) # 计算全局标准差，默认情况下，numpy 计算的是总体标准偏差，ddof = 0
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:                
                signals[i] = 1
                if nFlag == 0x00:
                    nFlag = i
                    print('nFlag = %d,  y[i] = %d' % (nFlag, y[i]))
            else:
                signals[i] = -1
                if nFlag == 0x00:
                    nFlag = i
                    print('nFlag = %d,  y[i] = %d' % (nFlag, y[i]))

            """
            @2018-11-20 对异常点的数值进行平滑，以便评估下下个点是否为异常点。
             因为不做平滑，由于当前是个异常点，对平均值、方差影响较大，若是下一个点仍是异常点，可能不会识别。
            """
            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1], ddof=1)
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1], ddof=1)

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))
    
# Data
"""
y = np.array([1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,1,1.1,0.9,1,1.1,1,1,0.9,
       1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,
       2.6,4,3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1])
"""
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 6, 7, 7, 7, 8, 8, 8, 7, 8, 9, 9, 9, 10, 10, 10, 10, 10, 11, 10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 8, 7, 8, 7, 7, 7, 7, 7, 7, 5, 5, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -2, -2, -2, -2, -2, -3, -3, -3, -3, -3, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -3, -4, -4, -4, -4, -3, -3, -3, -2, -2, -2, -2, -1, 0, -1, -1, 0, 0, 0, 0, 1, 0, 1, 2, 2, 2, 2, 3, 3, 4, 3, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 9, 9, 8, 8, 9, 9, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 3, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -2, -2, -2, -2, -3, -3, -3, -4, -4, -3, -4, -5, -5, -5, -4, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -4, -5, -4, -4, -4, -3, -3, -3, -3, -3, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 9, 8, 7, 8, 7, 7, 7, 6, 7, 6, 5, 6, 5, 5, 4, 4, 3, 3, 2, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1, -2, -1, -2, -2, -3, -3, -3, -4, -4, -5, -4, -4, -4, -5, -5, -5, -5, -5, -5, -5, -6, -5, -5, -5, -5, -5, -5, -4, -5, -5, -6, -5, -5, -4, -3, -4, -3, -3, -3, -3, -2, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 2, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 6, 7, 7, 7, 7, 8, 7, 7, 7, 8, 8, 8, 9, 9, 8, 8, 8, 8, 9, 8, 8, 8, 8, 8, 9, 9, 8, 8, 7, 7, 8, 7, 7, 7, 6, 7, 5, 5, 5, 5, 4, 4, 4, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -2, -3, -2, -4, -3, -3, -4, -4, -4, -5, -5, -5, -5, -5, -5, -6, -6, -6, -5, -6, -5, -6, -6, -5, -6, -6, -5, -5, -5, -5, -4, -4, -4, -4, -4, -3, -3, -3, -2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 5, 4, 5, 5, 6, 5, 7, 6, 7, 7, 7, 8, 8, 7, 8, 8, 9, 8, 8, 8, 8, 8, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 2, -12, -35, -56, -69, -78, -87, -100, -118, -139, -160, -179, -197, -208, -213, -217, -222, -229, -236, -240, -243, -244, -244, -242, -237, -232, -226, -219, -209, -198, -190, -179, -168, -154, -137, -119, -103, -85, -68, -49, -30, -10, 10, 31, 52, 72, 91, 113, 136, 158, 178, 198, 219, 239, 259, 277, 298, 316, 333, 348, 364, 379, 395, 408, 420, 431, 441, 450, 458, 465, 471, 475, 479, 479, 478, 478, 477, 475, 470, 463, 455, 447, 437, 425, 413, 400, 386, 369, 351, 333, 314, 296, 274, 251, 228, 205, 180, 155, 129, 102, 76, 48, 20, -7, -34, -61, -89, -118, -147, -174, -201, -229, -256, -282, -308, -333, -358, -382, -405, -428, -449, -471, -490, -508, -525, -542, -557, -571, -584, -595, -606, -615, -622, -626, -631, -634, -637, -637, -634, -631, -626, -621, -613, -604, -594, -583, -569, -554, -538, -521, -503, -484, -462, -440, -417, -393, -367, -340, -314, -286, -257, -227, -197, -166, -135, -104, -71, -38, -6, 24, 57, 89, 121, 152, 183, 214, 246, 275, 304, 333, 361, 388, 414, 439, 463, 487, 509, 530, 550, 568, 586, 600, 614, 628, 639, 649, 658, 664, 669, 673, 675, 675, 673, 671, 667, 661, 653, 644, 634, 621, 607, 592, 575, 557, 538, 517, 496, 473, 449, 422, 396, 369, 341, 313, 282, 250, 219, 188, 156, 123, 89, 56, 22, -10, -44, -76, -109, -142, -175, -207, -238, -269, -299, -328, -358, -386, -412, -438, -464, -488, -510, -531, -552, -571, -588, -604, -618, -631, -644, -654, -663, -672, -679, -681, -684, -684, -684, -681, -677, -671, -664, -656, -645, -634, -621, -605, -589, -572, -553, -533, -511, -489, -465, -440, -413, -387, -359, -330, -300, -270, -239, -207, -175, -143, -110, -76, -44, -9, 23, 56, 89, 121, 153, 186, 217, 248, 279, 309, 338, 366, 392, 418, 445, 469, 492, 513, 535, 554, 572, 589, 604, 619, 630, 642, 651, 659, 666, 670, 673, 675, 674, 673, 671, 665, 658, 650, 641, 630, 617, 603, 588, 572, 553, 533, 512, 489, 467, 443, 417, 390, 362, 334, 305, 275, 243, 212, 181, 148, 115, 83, 49, 16, -16, -49, -82, -115, -148, -180, -211, -243, -273, -303, -333, -360, -387, -415, -441, -465, -488, -510, -532, -551, -570, -587, -602, -617, -629, -642, -650, -658, -665, -670, -674, -677, -677, -675, -673, -668, -662, -654, -644, -634, -622, -607, -593, -575, -559, -538, -517, -496, -473, -448, -423, -397, -369, -341, -311, -281, -251, -220, -187, -155, -122, -88, -55, -21, 11, 44, 77, 109, 142, 175, 206, 237, 268, 298, 328, 356, 383, 409, 435, 460, 483, 505, 527, 546, 565, 582, 598, 613, 626, 636, 646, 654, 660, 664, 668, 670, 670, 668, 665, 660, 654, 645, 636, 625, 612, 598, 582, 565, 546, 527, 505, 483, 460, 435, 409, 382, 354, 326, 297, 266, 236, 205, 172, 140, 106, 74, 40, 7, -25, -59, -92, -125, -157, -189, -221, -252, -282, -312, -340, -368, -396, -422, -447, -471, -494, -516, -536, -556, -573, -590, -606, -621, -636, -651, -662, -671, -677, -678, -678, -678, -676, -673, -670, -665, -659, -650, -639, -627, -612, -597, -581, -564, -545, -525, -504, -480, -457, -431, -404, -378, -349, -321, -290, -260, -229, -198, -166, -133, -99, -67, -34, 0, 32, 64, 96, 129, 162, 193, 224, 254, 285, 314, 343, 370, 397, 424, 448, 471, 494, 516, 536, 555, 573, 588, 603, 616, 628, 639, 647, 655, 660, 665, 668, 669, 667, 666, 662, 656, 649, 642, 632, 620, 607, 592, 577, 559, 541, 521, 500, 478, 454, 430, 405, 378, 351, 323, 294, 265, 234, 203, 171, 139, 107, 74, 42, 9, -23, -55, -87, -119, -151, -183, -214, -244, -275, -304, -332, -360, -387, -413, -439, -463, -485, -506, -527, -547, -565, -581, -596, -611, -624, -635, -645, -652, -659, -664, -668, -670, -670, -669, -665, -661, -656, -649, -639, -629, -617, -603, -588, -572, -554, -535, -514, -493, -471, -447, -423, -397, -369, -342, -313, -283, -253, -222, -191, -159, -127, -94, -61, -28, 6, 38, 70, 102, 135, 166, 197, 229, 259, 289, 319, 347, 374, 400, 426, 451, 474, 496, 518, 537, 556, 573, 589, 604, 617, 629, 639, 646, 653, 658, 661, 664, 664, 664, 661, 655, 649, 642, 633, 621, 609, 596, 580, 563, 544, 525, 504, 482, 459, 435, 409, 383, 355, 327, 298, 267, 237, 206, 173, 142, 109, 75, 43, 9, -22, -56, -89, -122, -154, -186, -218, -249, -280, -309, -337, -365, -393, -418, -444, -467, -490, -512, -533, -552, -569, -586, -601, -616, -631, -645, -657, -665, -669, -672, -671, -670, -669, -667, -663, -658, -651, -642, -631, -618, -603, -588, -571, -554, -535, -514, -493, -470, -446, -420, -393, -367, -338, -310, -281, -250, -220, -188, -156, -123, -91, -57, -23, 8, 40, 73, 105, 137, 169, 200, 231, 261, 290, 320, 347, 374, 400, 425, 450, 473, 495, 515, 535, 554, 571, 586, 601, 613, 625, 634, 642, 649, 655, 659, 660, 661, 661, 659, 655, 648, 641, 632, 622, 611, 597, 583, 566, 550, 531, 512, 491, 469, 445, 420, 396, 369, 342, 313, 285, 255, 225, 195, 163, 131, 99, 68, 35, 2, -29, -61, -92, -125, -158, -188, -219, -249, -279, -307, -335, -362, -389, -415, -439, -463, -485, -506, -526, -546, -564, -579, -595, -608, -620, -632, -641, -648, -655, -659, -662, -664, -664, -662, -659, -655, -649, -640, -631, -620, -608, -595, -580, -563, -545, -526, -506, -483, -461, -437, -412, -385, -359, -330, -302, -273, -243, -212, -182, -149, -116, -84, -52, -18, 13, 45, 78, 110, 142, 173, 204, 235, 265, 295, 323, 351, 378, 404, 428, 452, 475, 498, 519, 537, 556, 572, 588, 602, 614, 626, 634, 642, 649, 654, 657, 657, 658, 657, 653, 648, 641, 633, 623, 611, 599, 585, 570, 552, 533, 514, 493, 471, 448, 424, 397, 371, 344, 315, 287, 257, 226, 194, 163, 131, 99, 67, 34, 1, -30, -63, -96, -127, -159, -191, -221, -253, -282, -311, -339, -366, -393, -418, -443, -466, -488, -508, -529, -547, -565, -581, -596, -611, -627, -639, -650, -658, -663, -665, -665, -663, -662, -661, -657, -651, -644, -635, -625, -613, -599, -584, -567, -550, -532, -512, -490, -468, -444, -420, -393, -367, -340, -312, -283, -253, -223, -192, -160, -128, -96, -64, -31, 1, 33, 65, 96, 128, 160, 191, 221, 252, 280, 309, 337, 364, 389, 416, 440, 463, 485, 506, 526, 545, 562, 578, 592, 605, 340, 140])

# Settings: lag = 30, threshold = 5, influence = 0
# lag for the smoothing, 计算平均值的间隔点数.
lag = 128
threshold = 5
# when signal: how much influence for new data? (between 0 and 1)
# 对当前节点做平滑，平滑系数是(0,1)，值越大越受当前值的影响。when 1 is normal influence, 0.5 is half。
influence = 0

# Run algo with settings from above
result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)

# Plot result
pylab.subplot(211)
pylab.plot(np.arange(1, len(y)+1), y)

pylab.plot(np.arange(1, len(y)+1),
           result["avgFilter"], color="cyan", lw=2)

pylab.plot(np.arange(1, len(y)+1),
           result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)

pylab.plot(np.arange(1, len(y)+1),
           result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)

pylab.subplot(212)
pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
pylab.ylim(-1.5, 1.5)
