from ultralytics import YOLO

# 加载YOLOv8模型和ONNX会话
model2 = YOLO('pth\yolov8n.pt')
model2.export(format='onnx')
model1 = YOLO('pth\yolov8n-pose.pt')

import onnxruntime
import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt  
#1数据预处理，处理成需要的格式 2用onnx推理预测，通过缩放比例还原图片 3数据后处理，使用非极大值抑制
# 打开视频文件并获取参数
#加载onnx模型,开启推理
ort_session = onnxruntime.InferenceSession('./pth/yolov8n.onnx')
out_path = './video/new_ball.mp4'


def count_ball(video1):
    def flowSpeed(prv_frame: np.ndarray, new_frame: np.ndarray ):  
        # 确保输入的是灰度图像  
        assert len(prv_frame.shape) == 2 and len(new_frame.shape) == 2, "Input frames must be grayscale."  
        # 计算Farneback光流  
        flow = cv2.calcOpticalFlowFarneback(prv_frame, new_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)  
        # 并计算点的光流  
        flow_at_center = flow[cy, cx]
        flow_at_wh = flow[h, w] 
        dx, dy = flow_at_center
        dw, dh = flow_at_wh
        # 绘制光流矢量（这里只是返回了矢量的信息，没有实际绘制）  
        yield (int(dx), int(dy),int(cx), int(cy),int(dw),int(dh),int(w),int(h))

    v = cv2.VideoCapture(video1)
    fps = v.get(cv2.CAP_PROP_FPS)
    width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path,fourcc,fps,(width,height))

    # 初始化变量
    count =0
    prev_a = False
    str_count ='0'
    prv_img = None
    prv_check_box = None
    check_box = []
    
    while True:
        #读取视频
        _, image = v.read()
        if image is None: 
            break
        if ord('q') == cv2.waitKey(1):
            break

        res = model1(image)

        # 将图像从BGR格式转换为RGB格式
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        # 调整图像大小为640x640 
        img1 = cv2.resize(img,(640,640)) 
        # 计算宽度的缩放比例 
        scale_x = img1.shape[1]/img.shape[1]
        # 计算高度的缩放比例
        scale_y = img1.shape[0]/img.shape[0] 
        # 将图像像素值归一化到0-1范围 
        img1 = img1/255  
        # 增加一个维度，使图像符合模型输入的要求，变成(1,640,640,3)
        img1 = np.expand_dims(img1,axis=0)
        #排序，变成(1,3,640,640)
        # 按照B,C,H,W的维度进行调节
        img1 = np.transpose(img1,(0,3,1,2))
        # yolov8 onnx默认接受float32格式的数据，将图像数据类型转换为float32
        img1 = img1.astype(np.float32) 


        # 准备输入模型的数据 
        ort_inputs = {'images':img1} 
        # 运行模型进行预测 
        ort_output = ort_session.run(None,ort_inputs)[0]
        # 调整预测结果的维度顺序  
        ort_output = np.transpose(ort_output,(0,2,1)) 
        # 将预测结果数据类型转换为float32 
        ort_output = ort_output.astype(np.float32) 
        # 提取预测类别概率部分
        #...代表纵向所有
        pred_class = ort_output[0][...,4:]  
        # 找到最大置信度
        pred_conf = np.max(pred_class, axis=-1)  
        # 将置信度放在坐标后面
        pred = np.insert(ort_output[0],4,pred_conf,axis=-1) 
        # 选择置信度大于0.3的预测框
        box = pred[pred[...,4]>0.3] 
        # 提取预测类别概率部分 
        # 记录所有类别信息
        cls_conf = box[...,5:]
        # 存储预测的类别
        cls = []
        # 遍历筛选后的类别信息  
        for i in range(len(cls_conf)):
            # 保存所属类别的下表，argmax取下标
            if np.argmax(cls_conf[i])==32:
                cls.append(int(np.argmax(cls_conf[i])))
        # 过滤类别信息，set得到所有类别集合(没有重复元素)
        total_cls = list(set(cls))
        for m in range(len(total_cls)):
            output_box = []
            # 获取当前类别
            clss = total_cls[m]
            cls_box = []
            # 创建一个临时的list，用来保存检测框的xywh\置信度\所属类别
            temp = box[:,:6]
            for j in range(len(cls)):
                # 遍历筛选出来的所有类别下标
                # 如果类别下标和当前遍历的类相同
                if(cls[j]==clss):
                    # 创建一个list，用来保存[x,y,w,h,conf,cls]
                    temp[j][5] = clss
                    temp[j][0:4:2] = temp[j][0:4:2]/scale_x
                    temp[j][1:5:2] = temp[j][1:5:2]/scale_y
                    cls_box.append(temp[j][:6])
            # 转化为ndarray的格式
            # cls_box = np.array(cls_box)
            # nms:利用IOU交并比，非极大值抑制比
            # 必须是np.array格式
            sorted_cls_box = sorted(cls_box,key = lambda x:-x[4])
            output_box.append(sorted_cls_box)
            out_arr = np.array(output_box)
            nms_res = cv2.dnn.NMSBoxes(out_arr[0][:,:4],
                                    out_arr[0][:,4],
                                    0.5,
                                    0.5)
            if(len(nms_res))!=0:
                for k in nms_res:
                    x = int(out_arr[0][k][0])
                    y = int(out_arr[0][k][1])
                    w = int(out_arr[0][k][2])
                    h = int(out_arr[0][k][3])
                    # print(x)
                    conf = round(float(out_arr[0][k][4]),2)
                    cx = (x-w//2)+(w//2)
                    cy = (y-h//2)+(h//2)

                    check_box = np.array([x,y,w,h,int(cx),int(cy),conf,clss])
                    # print(check_box)
                    # print(prv_check_box)


                    x_min = x-3*w//2
                    y_min = y-3*h//2
                    x_max = x+3*w//2
                    y_max = y+3*h//2
                    
                    cv2.circle(image, (x, y),3, (0, 0, 255), -1)
                    cv2.rectangle(image,((x-w//2),y-h//2),(x+w//2,y+h//2),(0,255,255),2)
                    cv2.rectangle(image,(x_min, y_min),(x_max, y_max),(0,255,255),2)
                    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                            
                

                    # 遍历结果列表res  
                    for r in res:  
                        # 遍历每个结果的keypoints  
                        for loc in r.keypoints:  
                            # 获取loc的xy坐标  
                            all = loc.xy  
                            # 将坐标转换为列表格式  
                            all = all.tolist()  
                            # 提取x1和y1坐标  
                            x1 = all[0][15][0]  
                            y1 = all[0][15][1]  
                            # 在图像img上以(x1, y1)为圆心绘制一个圆形，半径为3，颜色为红色（RGB值为(0, 0, 255)）  
                            cv2.circle(img, (int(x1), int(y1)), 3, (0, 0, 255), -1)  
                            # 提取x2和y2坐标  
                            x2 = all[0][16][0]  
                            y2 = all[0][16][1]  
                            # 在图像image上以(x2, y2)为圆心绘制一个圆形，半径为3，颜色为红色（RGB值为(0, 0, 255)）  
                            cv2.circle(image, (int(x2), int(y2)), 3, (0, 0, 255), -1)

                            # 判断点(x1, y1)和点(x2, y2)是否在矩形内，并赋值给变量a  
                            a = x_min <= x1 <= x_max and y_min <= y1 <= y_max or x_min <= x2 <= x_max and y_min <= y2 <= y_max  
                    
                            # 如果prv_img不为空（即前一张图像存在）  
                            if prv_img is not None :  
                                # 调用flowSpeed函数计算两张图像之间的流速，并将结果赋值给flow_res  
                                flow_res = flowSpeed(prv_img, img)  
                                # 遍历flow_res中的每一个元素（包括dx、dy、cx、cy、dw、dh、w、h）  
                                for p ,(dx,dy,cx,cy,dw,dh,w,h) in enumerate(flow_res):  
                                    # 打印dy值  
                                    print(dy)  
                                    # 在图像yimage上绘制一个从(cx, cy)到(cx, cy+dy)的箭头，颜色为绿色（RGB值为(0, 128, 128)），箭头宽度为10像素  
                                    cv2.arrowedLine(image,(int(cx), int(cy)),(int(cx),int(cy+dy)),(0,128,128),10)  
                                    # 如果a的值与prev_a的值不相等且dy大于0，则将count的值加1，并将a的值赋给prev_a  
                                    if a != prev_a and dy > 0:    
                                        count += 1    
                                        prev_a = a
                                    str_count = str(int(count//2))
                                    #画图
                                    plt_count(count)

                                      
                            # 将当前图像img赋值给prv_img，以便在下一次循环中使用前一张图像  
                            prv_img = img    
                            # 将check_box函数赋值给prv_check_box，可能是为了在下一次循环中使用前一个框的函数值  
                            prv_check_box = check_box

        # 在图像上显示计数
        cv2.putText(image,str_count,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,225),2)
        cv2.imshow('test',image)
        # cv2.imshow('prv',prv_img)
        out.write(image)

    pl = cv2.imread("D:/workspace1/img/test.jpg")
    cv2.waitKey(0)
    v.release()
    out.release()
    cv2.destroyAllWindows()
    return pl
def plt_count(count):
    categories = ['player1']  
    values = [int(count//2)]  
    # 创建条形图  
    plt.bar(categories, values,width=0.02) 
    # 添加标题和标签  
    plt.title('Count_Ball')  
    plt.xlabel('players')  
    plt.ylabel('counts')  
    # 显示图形  
    plt.show()
    plt.savefig('D:/workspace1/img/test.jpg')
def print_path(choice):
    info  = ''
    if choice =='yes':
        info += out_path.format(choice)
    return info

with gr.Blocks() as demo:
    with gr.Column():
        radio_input = gr.Radio(["yes","no"], 
        label="path", info="Do you want to get the processed video path?")
        radio_output = gr.Textbox()
    path_button = gr.Button("Chose")
    with gr.Row():
        video_input = gr.Video()
        video_output = gr.Image()
    video_button = gr.Button("Generate")
    video_button.click(count_ball, inputs=video_input, outputs=video_output)
    path_button.click(print_path, inputs=radio_input, outputs=radio_output)
demo.launch(share=True)