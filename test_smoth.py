"""Self-check: a face is found and every effect actually changes it."""
import cv2
import numpy as np

import smoth

img = cv2.imread("samples/will.jpg")
assert img is not None
assert len(smoth.faces(img)) == 1
for name in smoth.EFFECTS:
    out, n = smoth.apply(img.copy(), name)
    assert n == 1 and out.shape == img.shape, name
    assert (np.abs(out.astype(int) - img).max(2) > 10).sum() > 5000, f"{name} did nothing"
blank = np.full((400, 400, 3), 200, np.uint8)
assert smoth.apply(blank, "tiny")[1] == 0
print("ok")
