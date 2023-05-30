import cv2

# 读取清晰图像
img = cv2.imread('./images/image000.png')

# 对图像进行高斯模糊处理，其中(5,5)参数表示核的大小，20表示标准差
blur_img = cv2.GaussianBlur(img, (9,9), 20)
# blur_img = img
# 显示模糊后的图像
cv2.imshow('Blurred Image', blur_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 保存模糊后的图像
cv2.imwrite('blurred_image_99.png', blur_img)