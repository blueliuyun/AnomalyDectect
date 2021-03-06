# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:07:58 2018

@author: tianye
  1. 用 timestamp 找对应的值  

"""

import pandas as pd
import numpy as np
import pylab
import luminol
from luminol.anomaly_detector import AnomalyDetector
from luminol.correlator import Correlator

#ts = pd.read_csv('./SAR-device.sdb.await.csv')
#ts = np.array([3, 2, 3, 3, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 2, 3, 2, 2, 3, 3, 2, 3, 2, 2, 2, 3, 3, 2, 2, 3, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 2, 1, 37, 315, 751, 812, 717, 889, 752, 733, 710, 647, 692, 592, 512, 478, 489, 358, 357, 333, 249, 168, 133, 94, 48, -9, -67, -124, -182, -239, -254, -314, -373, -425, -465, -521, -559, -592, -629, -684, -733, -763, -789, -821, -854, -886, -917, -946, -963, -978, -996, -1016, -1032, -1046, -1053, -1055, -1055, -1058, -1062, -1059, -1051, -1041, -1026, -1012, -999, -984, -963, -937, -904, -875, -848, -819, -782, -745, -708, -668, -627, -584, -542, -499, -452, -398, -351, -304, -258, -205, -149, -95, -45, 3, 57, 111, 162, 211, 261, 313, 362, 408, 452, 502, 550, 593, 633, 672, 714, 754, 789, 822, 855, 887, 916, 943, 967, 990, 1011, 1029, 1043, 1057, 1070, 1079, 1084, 1088, 1089, 1088, 1085, 1080, 1070, 1059, 1045, 1029, 1013, 992, 970, 971, 963, 880, 873, 832, 788, 757, 715, 675, 634, 592, 541, 503, 447, 399, 359, 305, 246, 198, 152, 98, 41, -13, -62, -111, -168, -219, -269, -316, -366, -416, -465, -509, -551, -596, -641, -681, -718, -754, -792, -827, -857, -886, -915, -942, -965, -986, -1006, -1023, -1038, -1049, -1056, -1063, -1067, -1069, -1067, -1063, -1055, -1045, -1032, -1019, -1001, -980, -956, -931, -902, -870, -837, -802, -769, -733, -689, -645, -604, -561, -514, -463, -414, -366, -318, -264, -207, -154, -104, -51, 6, 61, 114, 164, 216, 271, 322, 371, 418, 468, 518, 563, 606, 648, 692, 733, 770, 805, 840, 875, 905, 932, 958, 984, 1005, 1024, 1040, 1054, 1067, 1076, 1082, 1085, 1087, 1086, 1082, 1076, 1068, 1055, 1040, 1024, 1006, 983, 968, 948, 920, 864, 850, 806, 766, 741, 688, 651, 603, 552, 518, 460, 408, 363, 318, 252, 200, 150, 99, 45, -15, -69, -120, -173, -225, -280, -331, -379, -426, -479, -527, -570, -612, -658, -702, -739, -774, -810, -846, -878, -907, -933, -957, -982, -1002, -1019, -1033, -1047, -1056, -1063, -1068, -1070, -1070, -1067, -1060, -1053, -1042, -1027, -1011, -994, -973, -949, -919, -891, -862, -830, -791, -755, -719, -680, -636, -590, -548, -504, -456, -405, -353, -305, -256, -202, -146, -94, -43, 9, 65, 119, 170, 219, 270, 324, 374, 420, 466, 515, 562, 606, 646, 687, 728, 768, 802, 835, 869, 900, 929, 953, 977, 1001, 1020, 1037, 1050, 1062, 1072, 1080, 1084, 1086, 1085, 1083, 1077, 1070, 1058, 1045, 1029, 1011, 991, 969, 986, 949, 882, 870, 825, 784, 748, 714, 670, 626, 579, 531, 497, 434, 387, 344, 291, 232, 180, 130, 78, 25, -33, -87, -137, -188, -239, -294, -344, -391, -438, -490, -537, -578, -620, -665, -708, -746, -779, -814, -850, -882, -909, -935, -959, -983, -1003, -1020, -1035, -1048, -1058, -1064, -1069, -1072, -1071, -1068, -1062, -1054, -1044, -1029, -1013, -996, -975, -951, -921, -892, -863, -831, -793, -756, -721, -681, -637, -591, -549, -505, -457, -405, -355, -307, -257, -204, -148, -94, -44, 7, 64, 119, 169, 218, 269, 323, 373, 420, 466, 514, 561, 605, 646, 687, 728, 767, 802, 834, 867, 900, 928, 953, 977, 999, 1018, 1036, 1050, 1062, 1071, 1079, 1084, 1086, 1085, 1082, 1076, 1069, 1058, 1044, 1026, 1010, 990, 966, 939, 912, 885, 853, 817, 779, 744, 706, 663, 618, 574, 532, 484, 432, 382, 336, 286, 230, 175, 124, 75, 20, -37, -91, -139, -190, -244, -298, -346, -392, -440, -491, -538, -580, -621, -665, -707, -745, -779, -813, -848, -880, -908, -933, -958, -982, -1001, -1016, -1032, -1044, -1054, -1062, -1066, -1068, -1066, -1064, -1058, -1050, -1039, -1024, -1008, -992, -970, -945, -919, -892, -864, -830, -794, -757, -722, -683, -638, -593, -550, -506, -458, -406, -356, -308, -259, -204, -148, -96, -44, 6, 63, 117, 168, 217, 269, 322, 372, 419, 465, 513, 561, 604, 645, 686, 727, 766, 801, 834, 868, 899, 927, 953, 976, 999, 1018, 1034, 1049, 1062, 1071, 1078, 1083, 1085, 1084, 1080, 1076, 1067, 1056, 1042, 1026, 1009, 989, 965, 939, 912, 884, 852, 816, 779, 743, 705, 662, 617, 573, 530, 483, 430, 381, 334, 284, 229, 173, 123, 73, 17, -38, -91, -141, -191, -246, -299, -348, -394, -443, -493, -539, -581, -623, -666, -709, -747, -780, -815, -850, -882, -909, -934, -959, -983, -1001, -1018, -1032, -1046, -1055, -1062, -1067, -1069, -1067, -1064, -1059, -1051, -1039, -1025, -1009, -992, -971, -946, -920, -893, -864, -831, -794, -758, -723, -683, -638, -594, -550, -507, -458, -406, -356, -307, -259, -205, -149, -97, -45, 6, 62, 118, 168, 217, 269, 322, 372, 419, 465, 513, 561, 604, 644, 686, 727, 766, 800, 833, 867, 898, 927, 952, 976, 998, 1018, 1034, 1048, 1061, 1070, 1078, 1082, 1084, 1084, 1080, 1074, 1067, 1056, 1042, 1025, 1008, 988, 964, 937, 910, 883, 851, 814, 778, 741, 703, 660, 615, 572, 528, 481, 428, 379, 331, 282, 227, 172, 121, 71, 16, -41, -93, -143, -194, -248, -301, -349, -395, -443, -495, -542, -582, -624, -667, -710, -748, -781, -817, -851, -883, -910, -935, -960, -984, -1002, -1019, -1033, -1047, -1056, -1063, -1067, -1070, -1069, -1065, -1060, -1052, -1040, -1025, -1009, -993, -971, -946, -920, -894, -864, -832, -794, -758, -723, -683, -639, -594, -550, -507, -459, -407, -356, -308, -259, -205, -149, -96, -45, 6, 62, 116, 168, 217, 268, 322, 372, 419, 464, 513, 560, 603, 644, 686, 727, 765, 800, 833, 866, 899, 926, 951, 975, 998, 1017, 1034, 1048, 1061, 1070, 1077, 1081, 1084, 1083, 1080, 1073, 1066, 1054, 1041, 1024, 1007, 987, 963, 937, 909, 881, 850, 813, 776, 741, 702, 659, 613, 571, 527, 480, 427, 377, 330, 281, 224, 170, 119, 69, 15, -42, -96, -144, -195, -250, -302, -351, -397, -445, -496, -542, -584, -625, -669, -712, -749, -783, -818, -853, -884, -911, -935, -961, -985, -1003, -1019, -1034, -1047, -1056, -1063, -1068, -1070, -1069, -1066, -1060, -1052, -1040, -1026, -1010, -993, -972, -947, -920, -894, -866, -832, -794, -759, -723, -684, -639, -595, -551, -508, -460, -406, -357, -309, -259, -205, -150, -97, -45, 7, 62, 117, 168, 217, 268, 322, 372, 418, 464, 513, 560, 603, 644, 685, 727, 765, 801, 833, 867, 898, 926, 951, 975, 998, 1017, 1033, 1047, 1060, 1070, 1077, 1081, 1083, 1083, 1078, 1073, 1066, 1054, 1040, 1024, 1007, 986, 962, 936, 909, 881, 848, 812, 775, 739])
asData = [0]
asTime = [0]

my_detector = AnomalyDetector(
        #time_series='./SAR-device.sdb.await__20190122_1500__负荷开关__0001__BAY01_0514_20181210_023118_850__U0.csv',
        #time_series='./SAR-device.sdb.await__New__Index580__L3 区外 L5 1000欧 B相__BAY01_0046_20181001_044812_313__U0.csv',
        #time_series='./SAR-device.sdb.await__故障回放__PDZ810_20190108__RD_IN_287__F_BAY01_0239_20181126_071233_183__U0.csv',
        #time_series='./SAR-device.sdb.await__研发中心波形_高阻接地_00025_20171025_201648_049_F__U0.csv',
        #time_series='./SAR-device.sdb.await__PDZ810__20190121_Switch__BAY01_0001_20190115_091118_218__U0.csv',
        #time_series='./SAR-device.sdb.await__2017-07-20 第四项检测L3 区外 L5 1000欧 C相__U0.csv',
        time_series='./SAR-device.sdb.await__DCU1923ZeroOrder(4_8)2017-11-07 11_56_54_491016.csv',
        #score_threshold=0.1, #1.0,
        score_threshold= 4.703433200392526,
        algorithm_name='derivative_detector')#derivative_detector'exp_avg_detector#'bitmap_detector)#, algorithm_params = {'smoothing factor': 0.2, 'lag_window_size': 64 })

score = my_detector.get_all_scores()

for timestamp, value in score.iteritems():
    asData.append(value)
    #asTime.append(pd.to_datetime(timestamp))
    asTime.append(timestamp)
    #print(timestamp, value)

# 异常点集合
asAnomal = my_detector.get_anomalies()
#for a in asAnomal:
#    print(a)

asData = asData[:1664]    
pylab.figure(figsize=(32, 16))
pylab.subplot(311)
#asData = asData[:582]
x = np.arange(1, len(asData)+1, 1)
pylab.plot(x, asData) #测值
pylab.grid(True)

"""
#####################
if asAnomal:
    time_period = asAnomal[0].get_time_window()
    correlator = Correlator(time_series_a='./SAR-device.sdb.await__研发中心波形_高阻接地_00025_20171025_201648_049_F__U0.csv', 
                            time_series_b='./SAR-device.sdb.await__消弧线圈-LINE3-高阻接地__U0.csv',
                            time_period=time_period)
    print(correlator.get_correlation_result().coefficient)
    if correlator.is_correlated(threshold=0.7):
        print('Ture Correlator')
    else:
        print('False Correlator')

"""

#####################
time_period = asAnomal[0].get_time_window()
correlator = Correlator(#time_series_a='./SAR-device.sdb.await__20190122_1500__负荷开关__0001__BAY01_0514_20181210_023118_850__U0__568-588.csv',
                        #time_series_a='./SAR-device.sdb.await__DCU1923ZeroOrder(4_8)2017-11-10 11_51_02_831016__U0_657-677.csv',
                        #time_series_b='./SAR-device.sdb.await__DCU1923ZeroOrder(4_8)2017-11-10 11_51_02_831016__U0_401-421.csv')
                        time_series_a='./SAR-device.sdb.await__20190122_1500__负荷开关__0001__BAY01_0514_20181210_023118_850__U0.csv',
                        time_series_b='./SAR-device.sdb.await__故障回放__PDZ810_20190108__RD_IN_287__F_BAY01_0239_20181126_071233_183__U0.csv')
                        #time_series_a='./SAR-device.sdb.await__故障回放__PDZ810_20190108__RD_IN_287__F_BAY01_0239_20181126_071233_183__U0.csv',
                        #time_series_b='./SAR-device.sdb.await__故障回放__PDZ810_20190108__RD_IN_287__F_BAY01_0239_20181126_071233_183__U0.csv',                        
                        #time_series_b='./SAR-device.sdb.await__DCU1923ZeroOrder(4_8)2017-11-10 11_51_02_831016__U0_913-933.csv')
                        #time_series_b='./SAR-device.sdb.await__New__Index580__L3 区外 L5 1000欧 B相__BAY01_0046_20181001_044812_313__U0_314-334.csv')
                        #time_period=time_period)
print(correlator.get_correlation_result().coefficient)
if correlator.is_correlated(threshold=0.2):
    print('Ture Correlator')
else:
    print('False Correlator')

# 1. 找到前 3 CYCLE 中最大的 MAXscore； 这个思路中 【 2倍的比例关系】 仅是根据已有数据的试验分析。
# 2. 从第 (128*3+1) 点开始与 MAXscore 比较，
# 2.1 如果 MAXscore < CurrentScore <= 2*MAXscore，则用CurrentScore更新 MAXscore；
# 2.2 如果 CurrentScore > 2*MAXscore，则认为找到了突变点。
