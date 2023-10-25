import cv2 as cv
import numpy as np
import sys
import time

style1_val = 1
style2_val = 0
style3_val = 0

def on_trackbar_style1(val):
    global style1_val
    style1_val = val/100

def on_trackbar_style2(val):
    global style3_val
    style3_val = val/100

def on_trackbar_style3(val):
    global style2_val
    style2_val = val/100

# img = cv.imread("starrynight.png", 1)

cv.namedWindow("Output Frame")
cv.createTrackbar("Style 1", "Output Frame", 100, 100, on_trackbar_style1)
cv.createTrackbar("Style 2", "Output Frame", 0, 100, on_trackbar_style2)
cv.createTrackbar("Style 3", "Output Frame", 0, 100, on_trackbar_style3)
# while True:
#     frame = img.copy()
#     frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#     # print(np.max(frame[::,::,0]))
#     h, s, v = cv.split(frame)
#
#     h_mod = h.astype(np.uint16)
#     h_mod = h_mod + style1_val
#     np.clip(h_mod, 0, 255, out=h_mod)
#     h = h_mod.astype(np.uint8)
#
#     s_mod = s.astype(np.uint16)
#     s_mod = s_mod + style3_val
#     np.clip(s_mod, 0, 255, out=s_mod)
#     s = s_mod.astype(np.uint8)
#
#     v_mod = v.astype(np.uint16)
#     v_mod = v_mod + style2_val
#     np.clip(v_mod, 0, 255, out=v_mod)
#     v = v_mod.astype(np.uint8)
#     # print("hsv: ", style1_val, style3_val, style2_val)
#
#     frame_merged = cv.merge((h, s, v))
#     frame_merged = cv.cvtColor(frame_merged, cv.COLOR_HSV2BGR)
#     cv.imshow("Output Frame", frame_merged)
#     if cv.waitKey(1) == ord('q'):
#         break


# cv.destroyAllWindows()
