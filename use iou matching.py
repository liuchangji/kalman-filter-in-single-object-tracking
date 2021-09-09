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

    SAVE_VIDEO = True
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('maxiou output.avi', fourcc, 20,(768,576))

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        plot_one_box(last_frame_box, frame, color=(200, 0, 0), target=False)  # 绘制出上一帧的框

        if not ret:
            break
        label_file_path = os.path.join(label_path, file_name + "_" + str(frame_counter) + ".txt")
        with open(label_file_path, "r") as f:
            content = f.readlines()
            max_iou = 0

            target_located = False
            for j, data_ in enumerate(content):
                data = data_.replace('\n', "").split(" ")
                xyxy = np.array(data[1:5], dtype="int")

                iou = cal_iou(xyxy, last_frame_box)
                if iou > max_iou:
                    # 找到与last_frame_box IOU最大的框
                    target_box = xyxy
                    max_iou = iou
                    target_located = True
                plot_one_box(xyxy, frame)
            if target_located == True:
                plot_one_box(target_box, frame, target=True)
                box_center = ((target_box[0] + target_box[2]) // 2, (target_box[1] + target_box[3]) // 2)
                trace_list = updata_trace_list(box_center, trace_list, 20)

                last_frame_box = target_box  # 更新上一帧的框，用于下一帧寻找最大IOU
                cv2.putText(frame, "Tracking", (target_box[0], target_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 2)

            else:
                cv2.putText(frame, "Lost", (target_box[0], target_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 2)
                if len(trace_list) >= 1:
                    trace_list.pop(0)
        draw_trace(frame, trace_list)

        cv2.putText(frame, "ALL BOXES(Green)", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(frame, "TRACKED BOX(Red)", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "LAST FRAME BOX(Blue)", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

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
