import cv2 as cv
import numpy as np
import sys
import time

def image_hsv(image_name, hue_shift, value_shift, saturation_shift):
    # img = cv.imread(image_name, 1)
    img = image_name
    frame = img.copy()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(frame)

    h_mod = h.astype(np.uint16)
    h_mod = h_mod + hue_shift
    np.clip(h_mod, 0, 255, out=h_mod)
    h = h_mod.astype(np.uint8)

    s_mod = s.astype(np.uint16)
    s_mod = s_mod + saturation_shift
    np.clip(s_mod, 0, 255, out=s_mod)
    s = s_mod.astype(np.uint8)

    v_mod = v.astype(np.uint16)
    v_mod = v_mod + value_shift
    np.clip(v_mod, 0, 255, out=v_mod)
    v = v_mod.astype(np.uint8)

    frame_merged = cv.merge((h, s, v))
    frame_merged = cv.cvtColor(frame_merged, cv.COLOR_HSV2BGR)
    return frame_merged

#     cv.imshow("Output Frame", frame_merged)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# image_hsv(0,0,0, "starrynight.png")