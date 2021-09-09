import os

import cv2
import numpy as np
from utils import plot_one_box, cal_iou, updata_trace_list, draw_trace

"使用最大IOU作为判别条件"

initial_target_box = [729, 238, 764, 339]  # 目标初始bouding box

if __name__ == "__main__":
    # 读取视频与标签
    video_path = "./data/testvideo1.mp4"
    label_path = "./data/labels"
    file_name = "testvideo1"
    last_frame_box = initial_target_box  # 上一帧的框初始化
    cap = cv2.VideoCapture(video_path)
    frame_counter = 1
    cv2.namedWindow("track", cv2.WINDOW_NORMAL)
    trace_list = []  # 用于保存目标box的轨迹

    SAVE_VIDEO = False
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('box output.avi', fourcc, 20,(768,576))

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # plot_one_box(last_frame_box, frame, color=(200, 0, 0), target=False)  # 绘制出上一帧的框

        if not ret:
            break
        label_file_path = os.path.join(label_path, file_name + "_" + str(frame_counter) + ".txt")
        with open(label_file_path, "r") as f:
            content = f.readlines()

            for j, data_ in enumerate(content):
                data = data_.replace('\n', "").split(" ")
                xyxy = np.array(data[1:5], dtype="int")

                iou = cal_iou(xyxy, last_frame_box)

                plot_one_box(xyxy, frame)
                x1y1 = str(xyxy[0:2])
                x2y2 = str(xyxy[2:4])
                cv2.putText(frame, x1y1, (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, x2y2, (int(xyxy[2]), int(xyxy[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


        # cv2.putText()

        cv2.imshow('track', frame)
        if SAVE_VIDEO:
            out.write(frame)
        frame_counter = frame_counter + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
