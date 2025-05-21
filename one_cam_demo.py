import threading
import cv2
import numpy as np
import torch
from srgpu import SuperResolutionEngine
from yolo256 import YOLOv8Inference
import time

def capture_frame(video_source: int):
    cap = cv2.VideoCapture(video_source)
    ret, frame = cap.read()
    cap.release()
    return frame

def main():
    print("#####")
    video_source = "2562.mp4"
    engine_file_sr ="m10c32c3.engine"
    engine_file_det = "edge0410_256.engine"
    device = torch.device('cuda:0')

    sr_engine = SuperResolutionEngine(engine_file_sr)
    yolo_inference = YOLOv8Inference(engine_file_det, device)
    for i in range(10):
            print("####warm-up")
            sr_engine.get4srimg(np.random.rand(192, 256,3).astype(np.uint8),(800,600))
            yolo_inference.infer(np.random.rand(192, 256,3).astype(np.uint8))
           
    cap = cv2.VideoCapture(video_source)
    
    while True:
        t=time.time()
        ret, frame = cap.read()
       
        if not ret:
            print(f"无法读取摄像头的数据")
            break
            
       
        # 线程处理
        sr_result = None  #超分结果
        bboxes= None     #推理结果
        scores= None 
        labels = None 
        reszie_sr=None   #超分resise

        def run_super_resolution(img):
            nonlocal sr_result,reszie_sr      #引用外部变量
            sr_result,_ = sr_engine.get4srimg(img,(800,600))
            print(f"超分输出={sr_result.shape}")
            # reszie_sr = cv2.resize(sr_result, (640, 512), interpolation=cv2.INTER_AREA)

        def run_detection(img):
            nonlocal bboxes, scores, labels
            #使用原始图片resize进行推理
            # ori_frame = cv2.resize(img, (640, 512), interpolation=cv2.INTER_AREA)
           
            bboxes, scores, labels = yolo_inference.infer(img)

        # 启动线程
        thread_sr = threading.Thread(target=run_super_resolution,args=(frame,))
        thread_det = threading.Thread(target=run_detection,args=(frame,))

        t1=time.time()
        thread_sr.start()
        thread_det.start()

        # 等待线程结束
        thread_det.join()
        t2 = time.time()
        print(f"检测时间={(t2 - t1) * 1000}ms")
        thread_sr.join()
        t3=time.time()
        print(f"超分时间={(t3-t2)*1000}ms")


        # 在超分图像上绘制检测框
        sr_result1 = sr_result.copy()
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = np.round(bbox).astype(int)
            cls_id = int(label)
            cls = yolo_inference.CLASSES_DET[cls_id]
            color = yolo_inference.COLORS[cls]
            text = f'{cls}:{score:.3f}'
            x1, y1, x2, y2 = bbox

            (_w, _h), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
            _y1 = min(y1 + 1, sr_result.shape[0])

            cv2.rectangle(sr_result1, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(sr_result1, (x1, _y1), (x1 + _w, _y1 + _h + _bl), (0, 0, 255), -1)
            cv2.putText(sr_result1, text, (x1, _y1 + _h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # 显示结果
        cv2.imshow("Super Resolution with Detection", sr_result1)
        # cv2.imshow("ori", frame)
        # cv2.imshow("Super Resolution ", sr_result)
        e=time.time()
        print(f"#####一帧图片的处理时间={(e-t)*1000}ms")

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
