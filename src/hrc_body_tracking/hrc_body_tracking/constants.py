import numpy as np

## definition of the body joints 
# h36m = [
#         0:  'Pelvis', 
#         1:  'LHip',
#         2:  'LKnee',
#         3:  'LAnkle',
#         4:  'RHip', 
#         5:  'RKnee',
#         6:  'RAnkle',
#         7:  'Spine',
#         8:  'Thorax',
#         9:  'Neck',
#         10: 'Head', 
#         11: 'LShoulder',
#         12: 'LElbow',
#         13: 'LWrist',
#         14: 'RShoulder',
#         15: 'RElbow',
#         16: 'RWrist'
#         ]
# coco_plus = [
#         17: 'RAnkle', 
#         18: 'RKnee', 
#         19: 'RHip', 
#         20: 'LHip', 
#         21: 'LKnee', 
#         22: 'LAnkle', 
#         23: 'RWrist',
#         24: 'RElbow',
#         25: 'RShoulder',
#         26: 'LShoulder',
#         27: 'LElbow',
#         28: 'LWrist', 
#         29: 'Neck', 
#         30: 'Head',
#         31: 'Nose',
#         32: 'LEye',
#         33: 'REye',
#         34: 'LEar',
#         35: 'REar'
#         ]

target = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 30, 34, 35]
parent_human36 = np.array([7, 0, 1, 2, 0, 4, 5, 8, 9, 10, 10, 8, 11, 12, 8, 14, 15])
parent_human36_noself = np.array([7, 0, 1, 2, 0, 4, 5, 8, 9, 10, 9, 8, 11, 12, 8, 14, 15])
parent_coco_plus = np.array([1, 2, 3, 2, 3, 4, 7, 8, 9, 8, 9, 10, 12, 14, 14, 14, 14, 14, 14]) + 17

JOINT_RIGHT = np.array([
    4, 5, 6, 14, 15, 16, 17, 18, 19, 23, 24, 25
])

JOINT_PARENT = np.array([7, 0, 1, 2, 0, 4, 5, 8, 9, 10, 10, 8, 11, 12, 8, 14, 15,
                         18, 19, 20, 19, 20, 21, 24, 25, 26, 25, 26, 27, 29, 31, 31, 31, 31, 31, 31])

JOINT_FLIP = np.array([
    0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13,
    22, 21, 20, 19, 18, 17, 28, 27, 26, 25, 24, 23, 29, 30, 31, 33, 32, 35, 34, -1
])

q2smpl = np.array([0, 1, 2, 5, 3, 4,
                   8, 7, 6, 11, 10, 9, 14, 13, 12,
                   17, 16, 15, 20, 19, 18, 23, 22, 21,
                   26, 25, 24, 29, 28, 27, 32, 31, 30,
                   35, 34, 33, 38, 37, 36, 41, 40, 39,
                   44, 43, 42, 47, 46, 45, 50, 49, 48,
                   52, 53, 51, 55, 56, 54,
                   59, 57, 58, 62, 60, 61,
                   65, 64, 63, 68, 67, 66, 71, 70, 69, 74, 73, 72,
                   ])

smpl2q = [0, 1, 2, 4, 5, 3,
          8, 7, 6, 11, 10, 9, 14, 13, 12,
          17, 16, 15, 20, 19, 18, 23, 22, 21,
          26, 25, 24, 29, 28, 27, 32, 31, 30,
          35, 34, 33, 38, 37, 36, 41, 40, 39,
          44, 43, 42, 47, 46, 45, 50, 49, 48,
          53, 51, 52, 56, 54, 55,
          58, 59, 57, 61, 62, 60,
          65, 64, 63, 68, 67, 66, 71, 70, 69, 74, 73, 72,
          ]

smpl2q_19 = [
            0, 1, 2,      # translation

            4, 5, 3,      # pelvis (old 0)   ZXY inverse -> YZX
            8, 7, 6,      # L_hip (old 1)    ZYX → same
            11,10,9,      # R_hip
            14,13,12,     # spine1

            17,16,15,     # L_knee
            20,19,18,     # R_knee

            23,22,21,     # L_ankle
            26,25,24,     # R_ankle

            29,28,27,     # spine2

            32,31,30,     # neck
            35,34,33,     # L_collar
            38,37,36,     # R_collar
            41,40,39,     # head

            44,42,43,     # L_shoulder (YZX inverse → ZXY)
            47,45,46,     # R_shoulder

            49,50,48,     # L_elbow  (ZXY inverse → YZX)
            52,53,51,     # R_elbow

            56,55,54,     # L_wrist
            59,58,57      # R_wrist
        ]

q2smpl_19 = [
            0, 1, 2,      # translation

            5, 3, 4,      # pelvis (old 0)   perm = ZXY
            8, 7, 6,      # L_hip (old 1)    ZYX
            11,10,9,      # R_hip (old 2)    ZYX
            14,13,12,     # spine1 (old 3)   ZYX

            17,16,15,     # L_knee (old 4)   ZYX
            20,19,18,     # R_knee (old 5)   ZYX

            23,22,21,     # L_ankle (old 7)  ZYX
            26,25,24,     # R_ankle (old 8)  ZYX

            29,28,27,     # spine2 (old 6)   ZYX

            32,31,30,     # neck (old 12)
            35,34,33,     # L_collar (old 13)
            38,37,36,     # R_collar (old 14)
            41,40,39,     # head (old 15)

            43,44,42,     # L_shoulder (old 16)  YZX
            46,47,45,     # R_shoulder (old 17)  YZX

            50,48,49,     # L_elbow (old 18)     ZXY
            53,51,52,     # R_elbow (old 19)     ZXY

            56,55,54,     # L_wrist (old 20)     ZYX
            59,58,57      # R_wrist (old 21)     ZYX
        ]




G2Hpose_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26, 
              33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53, 
              54,55,56,57,58,59,60,61,62,63,64,65, 
              ]

## spring-mass model 
incontact_grf = 30
sigma_v = 12
sigma_z = 60
z_rec = np.array([0, 0, 0, 0, 0,
                  0, 0, 0, 0,
                  0, -0.03, -0.03, -0.03, -0.03,
                  0, 0, -0.05, -0.05, -0.05, -0.05,
                  0,
                  0, 0, 0, 0, 0, 0, 0.02, 0.02
                  ])

## scale 
dt = 1 / 30.0
m2mm = 1000
