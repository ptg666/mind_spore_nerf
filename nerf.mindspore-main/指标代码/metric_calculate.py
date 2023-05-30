import cv2
import numpy as np
import lpips
import warnings
warnings.filterwarnings("ignore")
# from skimage.measure import structural_similarity
def psnr(img1, img2):
    # 读取两张图像
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    # 将像素值转换为浮点数
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    # 归一化
    img1 = cv2.normalize(img1, None, 0.0, 1.0, cv2.NORM_MINMAX)
    img2 = cv2.normalize(img2, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # 计算MSE
    mse = np.mean((img1 - img2) ** 2)

    # 计算PSNR
    if mse == 0:
        return 100
    else:
        PIXEL_MAX = 1
        psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
        return psnr
def ssim(img1,img2):
    # 读取图像
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    # 将像素值转换为浮点数
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1 = cv2.normalize(img1, None, 0.0, 1.0, cv2.NORM_MINMAX)
    img2 = cv2.normalize(img2, None, 0.0, 1.0, cv2.NORM_MINMAX)
    # 将图像转换为灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算SSIM
    score = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
    # (score, diff) = compare_ssim(gray1, gray2, full=True)
    return score
def LPips(img1,img2):
    # 加载LPIPS模型
    loss_fn = lpips.LPIPS(net='alex')
    # 读取图像
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    # 将图像转换为浮点数类型，范围为0到1
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    # 调整图像大小，使其与LPIPS模型输入的大小一致
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))
    # 将图像转换为PyTorch Tensor，并将其移动到GPU上（如果有的话）
    img1_tensor = lpips.im2tensor(img1)
    img2_tensor = lpips.im2tensor(img2)

    # 计算LPIPS
    distance = loss_fn(img1_tensor, img2_tensor)
    print("LPIPS distance: {}".format(distance.item()))
    return distance

iter_l = [500,1000,1500,2000]
img_ids = [0,1,2]
src_1_name = "without_ds"
src_2_name = "add_nothing"
def calculate_metric(iter_l,img_ids,src_1_name,src_2_name):
    l = len(img_ids)
    for iter in iter_l:
        psnr_1 = 0
        ssim_1 = 0
        lpips_1 = 0
        psnr_2 = 0
        ssim_2 = 0
        lpips_2 = 0
        for index in img_ids:
            if index == 0:
                image_gt = "./images/image000.png"
                image_render_nods = rf"./result/{src_1_name}\test_00{iter}\0000.png"
                image_render_ds = rf"./result/{src_2_name}\test_00{iter}\0000.png"
                if iter < 1000:
                    image_render_nods = rf"./result/{src_1_name}\test_000{iter}\0000.png"
                    image_render_ds = rf"./result/{src_2_name}\test_000{iter}\0000.png"
            elif index == 1:
                image_gt = "./images/image008.png"
                image_render_nods = rf"./result/{src_1_name}\test_00{iter}\0001.png"
                image_render_ds = rf"./result/{src_2_name}\test_00{iter}\0001.png"
                if iter < 1000:
                    image_render_nods = rf"./result/{src_1_name}\test_000{iter}\0001.png"
                    image_render_ds = rf"./result/{src_2_name}\test_000{iter}\0001.png"
            else:
                image_gt = "./images/image016.png"
                image_render_nods = rf"./result/{src_1_name}\test_00{iter}\0002.png"
                image_render_ds = rf"./result/{src_2_name}\test_00{iter}\0002.png"
                if iter < 1000:
                    image_render_nods = rf"./result/{src_1_name}\test_000{iter}\0002.png"
                    image_render_ds = rf"./result/{src_2_name}\test_000{iter}\0002.png"
            # print(f"img:{index} iter:{iter}")
            # print("PSNR:")
            psnr_1 += psnr(image_render_nods,image_gt)
            ssim_1 += ssim(image_render_nods, image_gt)
            lpips_1 += LPips(image_render_nods, image_gt)
            psnr_2 += psnr(image_render_ds,image_gt)
            ssim_2 += ssim(image_render_ds, image_gt)
            lpips_2 += LPips(image_render_ds, image_gt)
        print("------------------------------")
        print(f"iter:{iter}")
        print(f"{src_1_name}")
        print(f"PSNR:{psnr_1/l}")
        print(f"SSIM:{ssim_1/l}")
        print(f"LPIPS:{lpips_1/l}")
        print("-------")
        print(f"{src_2_name}")
        print(f"PSNR:{psnr_2/l}")
        print(f"SSIM:{ssim_2/l}")
        print(f"LPIPS:{lpips_2/l}")
        print("------------------------------")

calculate_metric(iter_l,img_ids,src_1_name,src_2_name)