import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory
import torch


class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__("yolo_detection_node")

        # 初始化 cv_bridge
        self.bridge = CvBridge()

        self.latest_depth_image_raw = None
        self.latest_depth_image_compressed = None

        # 使用 yolo model 位置
        model_path = os.path.join(
            get_package_share_directory("yolo_example_pkg"),
            "models",
            "detection.pt",
            # "best-seg.pt",
            # "tennis_v2.pt",
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device : ", device)
        self.model = YOLO(model_path)
        self.model.to(device)

        # 訂閱影像 Topic
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 1
        )

        # 訂閱 **無壓縮** 深度圖 Topic
        self.depth_sub_raw = self.create_subscription(
            Image, "/camera/depth/image_raw", self.depth_callback_raw, 1
        )

        # 訂閱 **壓縮** 深度圖 Topic
        self.depth_sub_compressed = self.create_subscription(
            CompressedImage,
            "/camera/depth/compressed",
            self.depth_callback_compressed,
            1,
        )

        # 發佈處理後的影像 Topic
        self.image_pub = self.create_publisher(
            CompressedImage, "/yolo/detection/compressed", 10
        )

        # 發布 目標檢測數據 (是否找到目標 + 距離)
        self.target_pub = self.create_publisher(
            Float32MultiArray, "/yolo/target_info", 10
        )

        self.x_multi_depth_pub = self.create_publisher(
            Float32MultiArray, "/camera/x_multi_depth_values", 10
        )

        # 設定要過濾標籤 (如果為空，那就不過濾)
        self.allowed_labels = {"bear"}  # {"bear", "knob"}

        # 設定 YOLO 可信度閾值
        self.conf_threshold = 0.45  # 可以修改這個值來調整可信度

        # 相機畫面中央高度上切成 n 個等距水平點。
        self.x_num_splits = 20

    def depth_callback_raw(self, msg):
        """接收 **無壓縮** 深度圖"""
        try:
            self.latest_depth_image_raw = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert raw depth image: {e}")

    def depth_callback_compressed(self, msg):
        """接收 **壓縮** 深度圖（當無壓縮深度圖不可用時使用）"""
        try:
            # 自行強制使用 cv2.IMREAD_UNCHANGED 解碼，避開 cv_bridge 的潛在雷區
            np_arr = np.frombuffer(msg.data, np.uint8)
            depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                self.latest_depth_image_compressed = depth_img
        except Exception as e:
            self.get_logger().error(f"Could not convert compressed depth image: {e}")

    def image_callback(self, msg):
        """接收影像並進行物體檢測"""
        # 將 ROS 影像消息轉換為 OpenCV 格式
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
            print("==>", cv_image.shape)
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # 使用 YOLO 模型檢測物體
        try:
            results = self.model(cv_image, conf=self.conf_threshold, verbose=False, device="cpu")
            for r in results:
                # 列印物件類別 ID、分數、以及框的座標
                print(r.boxes.cls)
                print(r.boxes.conf)
                print(r.boxes.xyxy)

            cv_image = self.draw_bounding_boxes(cv_image, results)

        except Exception as e:
            self.get_logger().error(f"Error during YOLO detection: {e}")
            return

        # 繪製 Bounding Box
        processed_image = results[0].plot()

        # 取得影像中心深度並發布
        self.publish_x_multi_depths(processed_image)

        # 發佈處理後的影像
        self.publish_image(processed_image)

    def draw_cross(self, image):
        # 回傳繪製十字架的影像和畫面正中間的像素座標
        height, width = image.shape[:2]
        cx_center = width // 2
        cy_center = height // 2
        # 繪製橫線
        cv2.line(image, (0, cy_center), (width, cy_center), (0, 0, 255), 2)

        # 繪製直線
        cv2.line(
            image,
            (cx_center, cy_center - 10),
            (cx_center, cy_center + 10),
            (0, 0, 255),
            2,
        )

        cv2.line(
            image,
            (cx_center, cy_center - 10),
            (cx_center, cy_center + 10),
            (0, 0, 255),
            2,
        )

        # 計算橫線上的 n 個等分點
        segment_length = width // self.x_num_splits
        points = [
            (i * segment_length, cy_center) for i in range(self.x_num_splits + 1)
        ]  # 11 個點表示 10 段區間的端點

        # 在每個等分點繪製垂直的短黑線
        for x, y in points:
            cv2.line(image, (x, y - 10), (x, y + 10), (0, 0, 0), 2)  # 黑色垂直線

        return image, points

    def draw_bounding_boxes(self, image, results):
        """在影像上繪製 YOLO 檢測到的 Bounding Box，並選擇最近的 bear 作為目標"""

        image, points = self.draw_cross(image)

        found_target = 0
        target_distance = 0.0
        delta_x = 0.0
        center_y = 999.0

        height, width = image.shape[:2]
        image_center_x = width // 2

        best_box = None
        best_distance = None
        best_area = None
        best_info = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                print("YOLO detected:", class_name, conf)

                # 只保留 bear
                if self.allowed_labels and class_name not in self.allowed_labels:
                    continue

                # 計算 Bounding Box 中心
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 取得深度
                depth_value = self.get_depth_at(cx, cy)

                # 框面積，深度失效時用面積當備案
                area = max(1, (x2 - x1) * (y2 - y1))

                # 過遠或無效深度先不要拿來導航
                valid_depth = depth_value is not None and depth_value > 0.05 and depth_value < 10.0

                # 優先選「有效深度中最近的熊」
                if valid_depth:
                    should_update = (
                        best_distance is None
                        or best_distance <= 0
                        or depth_value < best_distance
                    )
                else:
                    # 如果目前還沒有有效深度，就用最大框當備案
                    should_update = best_distance is None and (
                        best_area is None or area > best_area
                    )

                if should_update:
                    best_box = (x1, y1, x2, y2)
                    best_distance = depth_value
                    best_area = area
                    best_info = {
                        "class_name": class_name,
                        "conf": conf,
                        "cx": cx,
                        "cy": cy,
                        "area": area,
                    }

                # 其他偵測框也畫出來，但用藍色表示非主要目標
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(
                    image,
                    f"{class_name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

        # 如果有選到目標，發布 target_info
        if best_box is not None and best_info is not None:
            x1, y1, x2, y2 = best_box
            cx = best_info["cx"]
            cy = best_info["cy"]
            center_y = float(cy)
            class_name = best_info["class_name"]
            conf = best_info["conf"]

            found_target = 1
            target_distance = float(best_distance) if best_distance is not None else -1.0
            delta_x = float(cx - image_center_x)

            depth_text = f"{target_distance:.2f}m" if target_distance > 0 else "N/A"

            # 主要目標用綠色粗框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"TARGET {class_name} {conf:.2f} Depth: {depth_text} dx:{delta_x:.0f}"

            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            # 在目標中心畫點
            cv2.circle(image, (cx, cy), 5, (0, 255, 255), -1)

        print(
            "DEBUG selected target:",
            found_target,
            target_distance,
            delta_x,
            flush=True
        )

        self.publish_target_info(found_target, target_distance, delta_x, center_y)
        return image

    def get_depth_at(self, x, y):
        """
        取得指定像素的深度值，轉換為米 (m)
        若深度出問題，回傳 -1
        """
        # **優先使用無壓縮的深度圖**
        depth_image = (
            self.latest_depth_image_raw
            if self.latest_depth_image_raw is not None
            else self.latest_depth_image_compressed
        )

        if depth_image is None:
            return -1.0

        # 如果深度影像為三通道，那只取第一個數值
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]

        try:
            depth_value = depth_image[y, x]
            if depth_value < 0.0001 or depth_value == 0.0:  # 無效深度
                return -1.0
            return depth_value / 1000.0  # 16-bit 深度圖通常單位為 mm，轉換為 m
        except IndexError:
            return -1.0

    def publish_image(self, image):
        """將處理後的影像轉換並發佈到 ROS"""
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(image)
            self.image_pub.publish(compressed_msg)
        except Exception as e:
            self.get_logger().error(f"Could not publish image: {e}")

    def publish_target_info(self, found, distance, delta_x, center_y):
        """發佈目標資訊：found, distance, delta_x, center_y"""
        msg = Float32MultiArray()
        msg.data = [
            float(found),
            float(distance),
            float(delta_x),
            float(center_y),
        ]
        print("DEBUG publish target_info:", msg.data, flush=True)
        self.target_pub.publish(msg)

    def publish_x_multi_depths(self, image):
        """
        取得畫面 n 個等分點的深度並發布
        """
        height, width = image.shape[:2]
        cy_center = height // 2  # 固定 Y 座標在畫面中心
        segment_length = width // self.x_num_splits

        # 計算 10 個等分點的 X 座標
        points = [(i * segment_length, cy_center) for i in range(self.x_num_splits)]

        # 取得每個等分點的深度值
        depth_values = [self.get_depth_at(x, cy_center) for x, _ in points]

        # 以 Float32MultiArray 發布
        depth_msg = Float32MultiArray()
        depth_msg.data = depth_values
        self.x_multi_depth_pub.publish(depth_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
