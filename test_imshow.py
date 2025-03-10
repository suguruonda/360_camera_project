import cv2
import screeninfo

screen_0 = screeninfo.get_monitors()[0]
screen_1 = screeninfo.get_monitors()[1]

fname = "projectorimage/tpx/x_8_.png"
fname2 = "projectorimage/test_pattern_x/x_0.png"
img = cv2.imread(fname)
img2 = cv2.imread(fname2)

windowname = "1"
cv2.namedWindow(windowname, cv2.WINDOW_FULLSCREEN)
cv2.moveWindow(windowname, screen_1.x, screen_1.y)
cv2.setWindowProperty(windowname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

windowname2 = "2"
cv2.namedWindow(windowname2, cv2.WINDOW_FULLSCREEN)
cv2.moveWindow(windowname2, screen_0.x, screen_0.y)
cv2.setWindowProperty(windowname2, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.imshow(windowname, img)
cv2.imshow(windowname2, img2)
cv2.waitKey(0)
cv2.waitKey(1000)

cv2.destroyAllWindows()
