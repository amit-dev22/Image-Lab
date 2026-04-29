import cv2

img = cv2.imread(r'C:\Users\DELL\OneDrive\Desktop\python\sample 2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 200)

cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
