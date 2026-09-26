from pros_car_py.nav2_utils import (
    get_yaw_from_quaternion,
    get_direction_vector,
    get_angle_to_target,
    calculate_angle_point,
    cal_distance,
)
import math


class Nav2Processing:
    def __init__(self, ros_communicator, data_processor):
        self.ros_communicator = ros_communicator
        self.data_processor = data_processor
        self.finishFlag = False
        self.global_plan_msg = None
        self.index = 0
        self.index_length = 0
        self.recordFlag = 0
        self.goal_published_flag = False

    def reset_nav_process(self):
        self.finishFlag = False
        self.recordFlag = 0
        self.goal_published_flag = False
        self.global_plan_msg = None
        self.index = 0
        self.index_length = 0

    def finish_nav_process(self):
        self.finishFlag = True
        self.recordFlag = 1

    def get_finish_flag(self):
        return self.finishFlag

    def get_action_from_nav2_plan(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True
        orientation_points, coordinates = (
            self.data_processor.get_processed_received_global_plan()
        )
        action_key = "STOP"
        if not orientation_points or not coordinates:
            action_key = "STOP"
        else:
            try:
                z, w = orientation_points[0]
                plan_yaw = get_yaw_from_quaternion(z, w)
                car_position, car_orientation = (
                    self.data_processor.get_processed_amcl_pose()
                )
                car_orientation_z, car_orientation_w = (
                    car_orientation[2],
                    car_orientation[3],
                )
                goal_position = self.ros_communicator.get_latest_goal()
                target_distance = cal_distance(car_position, goal_position)
                if target_distance < 0.5:
                    action_key = "STOP"
                    self.finishFlag = True
                else:
                    car_yaw = get_yaw_from_quaternion(
                        car_orientation_z, car_orientation_w
                    )
                    diff_angle = (plan_yaw - car_yaw) % 360.0
                    if diff_angle < 30.0 or (diff_angle > 330 and diff_angle < 360):
                        action_key = "FORWARD"
                    elif diff_angle > 30.0 and diff_angle < 180.0:
                        action_key = "COUNTERCLOCKWISE_ROTATION"
                    elif diff_angle > 180.0 and diff_angle < 330.0:
                        action_key = "CLOCKWISE_ROTATION"
                    else:
                        action_key = "STOP"
            except:
                action_key = "STOP"
        return action_key

    def get_action_from_nav2_plan_no_dynamic_p_2_p(self, goal_coordinates=None):
        """
        改良版：
        1. 在還沒收到 global plan 前，持續重新發布 goal_pose。
        2. 收到有效 /plan 後，才固定使用第一條路徑。
        3. 不要因為一開始沒收到 plan 就永遠 STOP。
        """

        # 重點：還沒拿到第一條路徑前，每次都重新發布 goal
        # 避免 goal 只發一次但 Nav2 沒接到 / 還沒來得及產生 plan
        if goal_coordinates is not None and self.recordFlag == 0:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True
            print(f"NAV2_PROCESSING: publish goal_pose again: {goal_coordinates}")

        # 還沒記錄到第一條路徑
        if self.recordFlag == 0:
            if not self.check_data_availability():
                return "STOP"

            print("NAV2_PROCESSING: Get first path")
            self.index = 0
            self.global_plan_msg = (
                self.data_processor.get_processed_received_global_plan_no_dynamic()
            )

            if (
                self.global_plan_msg is None
                or self.global_plan_msg.poses is None
                or len(self.global_plan_msg.poses) == 0
            ):
                print("NAV2_PROCESSING: 收到的路徑是空的，繼續等待。")
                self.global_plan_msg = None
                return "STOP"

            print(f"NAV2_PROCESSING: path length = {len(self.global_plan_msg.poses)}")
            self.recordFlag = 1

        amcl_data = self.data_processor.get_processed_amcl_pose()
        if amcl_data is None:
            print("NAV2_PROCESSING: 等待 /amcl_pose")
            return "STOP"

        car_position, car_orientation = amcl_data

        goal_position = self.ros_communicator.get_latest_goal()
        if goal_position is None:
            print("NAV2_PROCESSING: 未設定 goal_pose")
            return "STOP"

        target_distance = cal_distance(car_position, goal_position)

        # 真正接近 goal 才算完成
        if target_distance < 0.5:
            self.ros_communicator.reset_nav2()
            self.finish_nav_process()
            return "STOP"

        # 從已記錄的 global plan 中找下一個目標點
        target_x, target_y = self.get_next_target_point(car_position)

        # 如果 target 用完但還沒到 goal，不要直接 finish
        # 改成重新要求 Nav2 生一條新路徑
        if target_x is None:
            print("NAV2_PROCESSING: 路徑點用完但還沒到 goal，重新要求路徑。")
            self.recordFlag = 0
            self.goal_published_flag = False
            self.global_plan_msg = None
            self.index = 0
            return "STOP"

        diff_angle = self.calculate_diff_angle(
            car_position, car_orientation, target_x, target_y
        )

        print(
            f"NAV2_PROCESSING: target=({target_x:.2f},{target_y:.2f}), "
            f"target_distance={target_distance:.2f}, diff_angle={diff_angle:.1f}"
        )

        if diff_angle < 20 and diff_angle > -20:
            action_key = "FORWARD"
        elif diff_angle < -20 and diff_angle > -180:
            action_key = "CLOCKWISE_ROTATION"
        elif diff_angle > 20 and diff_angle < 180:
            action_key = "COUNTERCLOCKWISE_ROTATION"
        else:
            action_key = "STOP"

        return action_key

    def check_data_availability(self):
        received_global_plan = self.data_processor.get_processed_received_global_plan_no_dynamic()
        amcl_pose = self.data_processor.get_processed_amcl_pose()
        goal = self.ros_communicator.get_latest_goal()

        if received_global_plan is None:
            print("沒有收到路徑")
            return False

        if (
            received_global_plan.poses is None
            or len(received_global_plan.poses) == 0
        ):
            print("收到 /plan，但路徑是空的")
            return False

        if amcl_pose is None:
            print("等待 /amcl_pose")
            return False

        if goal is None:
            print("未設定 goal_pose")
            return False

        return True

    def get_next_target_point(self, car_position, min_required_distance=0.5):
        """
        選擇距離車輛 min_required_distance 以上最短路徑然後返回 target_x, target_y
        """
        if self.global_plan_msg is None or self.global_plan_msg.poses is None:
            print("Error: global_plan_msg is None or poses is missing!")
            return None, None
        while self.index < len(self.global_plan_msg.poses) - 1:
            target_x = self.global_plan_msg.poses[self.index].pose.position.x
            target_y = self.global_plan_msg.poses[self.index].pose.position.y
            distance_to_target = cal_distance(car_position, (target_x, target_y))

            if distance_to_target < min_required_distance:
                self.index += 1
            else:
                self.ros_communicator.publish_selected_target_marker(
                    x=target_x, y=target_y
                )
                return target_x, target_y

        return None, None

    def calculate_diff_angle(self, car_position, car_orientation, target_x, target_y):
        target_pos = [target_x, target_y]
        diff_angle = calculate_angle_point(
            car_orientation[2], car_orientation[3], car_position[:2], target_pos
        )
        return diff_angle

    def filter_negative_one(self, depth_list):
        return [depth for depth in depth_list if depth != -1.0]

    def camera_nav(self):
        """
        YOLO 目標資訊 (yolo_target_info) 說明：

        - 索引 0 (index 0)：
            - 表示是否成功偵測到目標
            - 0：未偵測到目標
            - 1：成功偵測到目標

        - 索引 1 (index 1)：
            - 目標的深度距離 (與相機的距離，單位為公尺)，如果沒偵測到目標就回傳 0
            - 與目標過近時(大約 40 公分以內)會回傳 -1

        - 索引 2 (index 2)：
            - 目標相對於畫面正中心的像素偏移量
            - 若目標位於畫面中心右側，數值為正
            - 若目標位於畫面中心左側，數值為負
            - 若沒有目標則回傳 0

        畫面 n 個等分點深度 (camera_multi_depth) 說明 :

        - 儲存相機畫面中央高度上 n 個等距水平點的深度值。
        - 若距離過遠、過近（小於 40 公分）或是實體相機有時候深度會出一些問題，則該點的深度值將設定為 -1。
        """
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()
        if camera_multi_depth == None or yolo_target_info == None:
            return "STOP"

        camera_forward_depth = self.filter_negative_one(camera_multi_depth[7:13])
        camera_left_depth = self.filter_negative_one(camera_multi_depth[0:7])
        camera_right_depth = self.filter_negative_one(camera_multi_depth[13:20])

        action = "STOP"
        limit_distance = 0.7

        # if all(depth > limit_distance for depth in camera_forward_depth):
        if yolo_target_info[0] == 1:
            if yolo_target_info[2] > 200.0:
                action = "CLOCKWISE_ROTATION_SLOW"
            elif yolo_target_info[2] < -200.0:
                action = "COUNTERCLOCKWISE_ROTATION_SLOW"
            else:
                if yolo_target_info[1] < 0.5:
                    action = "STOP"
                else:
                    action = "FORWARD_SLOW"
        else:
            action = "CLOCKWISE_ROTATION"
        # elif any(depth < limit_distance for depth in camera_left_depth):
        #     action = "CLOCKWISE_ROTATION"
        # elif any(depth < limit_distance for depth in camera_right_depth):
        #     action = "COUNTERCLOCKWISE_ROTATION"
        return action

    def camera_nav_unity(self):
        """
        YOLO 目標資訊 (yolo_target_info) 說明：

        - 索引 0 (index 0)：
            - 表示是否成功偵測到目標
            - 0：未偵測到目標
            - 1：成功偵測到目標

        - 索引 1 (index 1)：
            - 目標的深度距離 (與相機的距離，單位為公尺)，如果沒偵測到目標就回傳 0
            - 與目標過近時(大約 40 公分以內)會回傳 -1

        - 索引 2 (index 2)：
            - 目標相對於畫面正中心的像素偏移量
            - 若目標位於畫面中心右側，數值為正
            - 若目標位於畫面中心左側，數值為負
            - 若沒有目標則回傳 0

        畫面 n 個等分點深度 (camera_multi_depth) 說明 :

        - 儲存相機畫面中央高度上 n 個等距水平點的深度值。
        - 若距離過遠、過近（小於 40 公分）或是實體相機有時候深度會出一些問題，則該點的深度值將設定為 -1。
        """
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()
        yolo_target_info[1] *= 1
        camera_multi_depth = list(
            map(lambda x: x * 1.0, self.data_processor.get_camera_x_multi_depth())
        )

        if camera_multi_depth == None or yolo_target_info == None:
            return "STOP"

        camera_forward_depth = self.filter_negative_one(camera_multi_depth[7:13])
        camera_left_depth = self.filter_negative_one(camera_multi_depth[0:7])
        camera_right_depth = self.filter_negative_one(camera_multi_depth[13:20])
        action = "STOP"
        limit_distance = 10.0
        print(yolo_target_info[1])
        if all(depth > limit_distance for depth in camera_forward_depth):
            if yolo_target_info[0] == 1:
                if yolo_target_info[2] > 200.0:
                    action = "CLOCKWISE_ROTATION_SLOW"
                elif yolo_target_info[2] < -200.0:
                    action = "COUNTERCLOCKWISE_ROTATION_SLOW"
                else:
                    if yolo_target_info[1] < 2.0:
                        action = "STOP"
                    else:
                        action = "FORWARD_SLOW"
            else:
                action = "FORWARD"
        elif any(depth < limit_distance for depth in camera_left_depth):
            action = "CLOCKWISE_ROTATION"
        elif any(depth < limit_distance for depth in camera_right_depth):
            action = "COUNTERCLOCKWISE_ROTATION"
        return action

    def stop_nav(self):
        return "STOP"
