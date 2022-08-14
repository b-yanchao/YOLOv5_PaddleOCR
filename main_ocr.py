from paddleocr import PaddleOCR
from tools.infer.utility import draw_ocr
from PIL import Image
import cv2
from detector import Detector
import numpy as np
import os
import shutil
import sys


# 根据预测结果裁剪图片
def caijian(img, bboxes):
    for i in range(len(bboxes)):
        box = bboxes[i]
        fx = int(box[0])
        if int(box[1]) > 20:
            fy = int(box[1]) - 20
        else:
            fy = 0
        tx = int(box[2])
        ty = int(box[3]) + 20
        # 图片裁剪 裁剪区域【Ly:Ry,Lx:Rx】
        print(fx, fy, tx, ty)
        cut = img[fy:ty, fx:tx]
        return cut


# 度数转换
def DegreeTrans(theta):
    res = theta / np.pi * 180
    return res


# 逆时针旋转图像degree角度（原尺寸）
def rotateImage(src, degree):
    # 旋转中心为图像中心
    h, w = src.shape[:2]
    # 计算二维旋转的仿射变换矩阵
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    # 仿射变换，背景色填充为白色
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))
    return rotate


# 经过霍夫变换计算角度
def CalcDegree(srcImage):
    midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    dstImage = cv2.Canny(midImage, 100, 200, 3)
    lineimage = srcImage.copy()

    # 经过霍夫变换检测直线
    # 第4个参数就是阈值，阈值越大，检测精度越高
    lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 200)
    # 因为图像不一样，阈值很差设定，由于阈值设定太高致使没法检测直线，阈值太低直线太多，速度很慢
    sum = 0
    n = 0
    # 依次画出每条线段
    if lines is None:
        pass
    else:
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                if rho > 1 and (1 < theta <= 1.5 or theta >= 1.6):  # 水平直线角度在1.5到1.6之间
                    n += 1
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(round(x0 + 1000 * (-b)))
                    y1 = int(round(y0 + 1000 * a))
                    x2 = int(round(x0 - 1000 * (-b)))
                    y2 = int(round(y0 - 1000 * a))
                    # 只选角度最小的做为旋转角度
                    sum += theta
                    cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imwrite('lineimage.png', lineimage)
            # 对全部角度求平均，这样作旋转效果会更好
    angle = -1
    if n != 0:
        average = sum / n
        angle = DegreeTrans(average) - 90
    return angle - 1


# 结构化输出函数
def print_structuring(result):
    stR = "{" + "\n"
    lastY = 0  # 上一个方框的下面Y的中间的
    lastX = 0  # 上一个方框的右边X的中间的
    islianxi = 0  # 冒号出现在最后的是否还有后续标志
    islianxiz = 0  # 冒号出现在中间的是否还有后续标志
    islianxi_jg = False  # 冒号出现在最后的是否还有间隔后续标志
    islianxiz_jg = False  # 冒号出现在中间的是否还有间隔后续标志
    x = ''
    for line in result:
        texts = line[1][0]
        newYu = (line[0][0][1] + line[0][1][1]) / 2  # 当前方框的上面Y的中间值
        newYd = (line[0][2][1] + line[0][3][1]) / 2  # 当前方框的下面Y的中间值
        newX = (line[0][1][0] + line[0][2][0]) / 2  # 当前方框的右边X的中间值
        newXL = (line[0][0][0] + line[0][3][0]) / 2  # 当前方框的左边X的中间值

        # 判断：出现在结尾的，第一句后是否还有文字
        if (islianxi >= 1 or islianxi_jg):
            # 若下一句中没有冒号，且当前方框的上面Y或者下面Y与上一个方框的下面Y的值相差小于20（or用来区分方框是和冒号在同一行还是下一行），且当前方框的右边与上一个方框左边要小于10（为了防止其他位置的文字）
            if ((texts.find('：') == -1 and texts.find(':') == -1) and (
                    newYu - lastY < 50 or newYd - lastY < 50) and newX - lastX < 20 and islianxi_jg == False) or (
                    islianxi == 1 and (texts.find('：') == -1 and texts.find(':') == -1)):
                stR = stR + x + str(texts)
                lastY = (line[0][2][1] + line[0][3][1]) / 2
                lastX = (line[0][1][0] + line[0][2][0]) / 2
                islianxi += 1
                x = ''
            elif (texts.find('：') == -1 and texts.find(':') == -1) and islianxi == 0 and islianxi_jg and (
                    newYu - lastY < 50 or newYd - lastY < 50) and newX - lastX < 20:  # 有间隔的后续文本
                stR = stR[0: int(len(stR) - 2)]
                stR = stR + str(texts) + "'" + "," + "\n"
            else:
                if islianxi_jg:
                    pass
                else:
                    stR = stR + "'" + "," + "\n"
                    islianxi = 0
                    islianxi_jg = True
        # 判断：出现在中间的，第一句后是否还有文字
        if islianxiz >= 1 or islianxiz_jg:
            if (texts.find('：') == -1 and texts.find(':') == -1) and (
                    newYu - lastY < 50 or newYd - lastY < 50) and newX - lastX < 20 and islianxiz_jg == False:  # 无间隔的后续文本
                stR = stR + x + str(texts)
                lastY = (line[0][2][1] + line[0][3][1]) / 2
                lastX = (line[0][1][0] + line[0][2][0]) / 2
                x = ''
            elif (texts.find('：') == -1 and texts.find(':') == -1) and islianxiz == 0 and islianxiz_jg and (
                    newYu - lastY < 50 or newYd - lastY < 50) and newX - lastX < 20:  # 有间隔的后续文本
                stR = stR[0: int(len(stR) - 2)]
                stR = stR + str(texts) + "'" + "," + "\n"
            else:
                if islianxiz_jg:
                    pass
                else:
                    stR = stR + "'" + "," + "\n"
                    islianxiz = 0
                    islianxiz_jg = True
        # 带有：数据前端缺少
        if ((newYd - lastY < 20) or (newXL - lastX < 20)) and (
                texts.find('：') == -1 and texts.find(':') == -1) and islianxi == 0 and islianxiz == 0:
            x = str(texts)
        # ：出现在中间位置（中文）
        if texts.find('：') != -1 and texts.find('：') != len(texts) - 1:
            islianxiz_jg = False
            islianxi_jg = False
            stR = stR + "'" + x + str(texts)[0:texts.find('：')] + "'" + ":" + "'" + str(texts)[
                                                                                    texts.find('：') + 1:len(texts)]
            lastY = (line[0][2][1] + line[0][3][1]) / 2
            lastX = (line[0][1][0] + line[0][2][0]) / 2
            islianxiz += 1
            x = ''
        # :出现在中间位置（英文）
        if texts.find(':') != -1 and texts.find(':') != len(texts) - 1:
            islianxiz_jg = False
            islianxi_jg = False
            stR = stR + "'" + x + str(texts)[0:texts.find(':')] + "'" + ":" + "'" + str(texts)[
                                                                                    texts.find(':') + 1:len(texts)]
            lastY = (line[0][2][1] + line[0][3][1]) / 2
            lastX = (line[0][1][0] + line[0][2][0]) / 2
            islianxiz += 1
            x = ''
        # ：出现在文字最后（中文）
        if (texts.find('：') == len(texts) - 1 or texts.find(':') == len(texts) - 1):
            islianxiz_jg = False
            islianxi_jg = False
            stR = stR + "'" + x + str(texts)[0:texts.find('：')] + "'" + ":" + "'"
            lastY = (line[0][2][1] + line[0][3][1]) / 2
            lastX = (line[0][1][0] + line[0][2][0]) / 2
            islianxi += 1
            x = ''
            continue
        # :出现在文字最后（英文）
        if (texts.find('：') == len(texts) - 1 or texts.find(':') == len(texts) - 1):
            islianxiz_jg = False
            islianxi_jg = False
            stR = stR + "'" + x + str(texts)[0:texts.find(':')] + "'" + ":" + "'"
            lastY = (line[0][2][1] + line[0][3][1]) / 2
            lastX = (line[0][1][0] + line[0][2][0]) / 2
            islianxi += 1
            x = ''
            continue
    if islianxi != 0 or islianxiz != 0:
        stR = stR + "'" + "," + "\n" + "}"
    else:
        stR = stR + "}"
    return stR


# 非结构化输出函数
def print_unstructured(result):
    n = 0
    stR = ''
    for line in result:
        n += 1
        texts = line[1][0]
        if n == len(result):
            stR = stR + str(texts)
        else:
            stR = stR + str(texts) + '+'
    return stR


# 清空预测文件夹txt结果文件存放地址
def RemoveDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！

    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)


# 预测流程main函数
def mainOCR(img_path):
    im = cv2.imread(img_path)  # 读取图片
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", det_model_dir="OCRdetmodel/", rec_model_dir="OCRrecmodel/")
    # 图片矫正
    degree = CalcDegree(im)
    print("调整角度：", degree)
    rotate = rotateImage(im, degree)
    cv2.imwrite('rotate.png', rotate)
    # 图片检测
    detector = Detector()
    bboxes, label = detector.detect(rotate)  # 调用yolov5识别图片中的标签区域 (在这之前就已经出现两个标签粘连的情况, box出现的问题，没有将空格分开)
    print("类别" + str(label))
    # 标签区域裁剪
    cut = caijian(rotate, bboxes)
    cv2.imwrite('cut.png', cut)
    # 图片矫正
    # 文字识别
    result = ocr.ocr(cut, cls=True)
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(cut, boxes, txts, scores)
    cv2.imwrite('result.png', im_show)
    return label, result


if __name__ == '__main__':
    leixing = str(input("请输入使用模式(单图片预测：1；多图片预测：n):"))
    if leixing == '1':
        img_path = 'images/' + input("请输入要识别的标签图片(images目录下的图片名):")
    elif leixing == 'n':
        img_path = input("请输入待检测文件夹名称:")
        jindu_len = len(os.listdir(img_path))  # 图片数量
        print("共：" + str(jindu_len))
        RemoveDir('result')
        print("result文件夹已清空")
        save_dir = 'result/'
    else:
        print("类型错误，请重新输入！！！")
        sys.exit(0)
    if leixing == '1':
        label, result = mainOCR(img_path)
        stR = ''
        if label == 'jgh':
            stR = print_structuring(result)
        elif label == 'fjgh':
            stR = print_unstructured(result)
        print(stR)
    elif leixing == 'n':
        for img in os.listdir(img_path):
            print(img)
            label, result = mainOCR(img_path + '/' + img)
            stR = ''
            if label == 'jgh':
                stR = print_structuring(result)
            elif label == 'fjgh':
                stR = print_unstructured(result)
            filename = str(save_dir) + img + '.txt'
            if not os.path.exists(filename):
                file = open(filename, 'w', encoding='utf-8')
                file.write(stR)
                file.close()
            print(stR)
        print('已将全部格式化结果以txt文件格式存放到result文件中')