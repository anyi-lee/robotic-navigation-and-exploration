import urwid
from pros_car_py.base_mode import BaseMode
import threading
import time
import math
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from pros_car_py.nav2_utils import get_yaw_from_quaternion


class VehicleMode(BaseMode):
    def enter(self):
        text = urwid.Text("Vehicle Mode\nPress 'q' to return to main menu.")
        filler = urwid.Filler(text, valign="top")

        self.app.loop.widget = filler
        self.app.loop.unhandled_input = self.handle_input

    def handle_input(self, key):
        if key == "q":
            self.app.car_controller.manual_control(key)
            self.app.main_menu()
        else:
            self.app.car_controller.manual_control(key)


class ArmMode(BaseMode):
    submodes = ["0", "1", "2", "3", "4"]

    def enter(self):
        self.app.horizontal_select(self.submodes, self.handle_submode_select)

    def handle_submode_select(self, submode):
        def on_key(key):
            self.app.arm_controller.manual_control(int(submode), key)

        self.show_submode_screen(
            message=f"Arm Mode: Submode {submode}\nPress 'q' to go back.", on_key=on_key
        )


class CraneMode(BaseMode):
    submodes = ["0", "1", "2", "3", "4", "5", "6", "99"]

    def enter(self):
        self.app.horizontal_select(self.submodes, self.handle_submode_select)

    def handle_submode_select(self, submode):
        def on_key(key):
            self.app.crane_controller.manual_control(int(submode), key)

        self.show_submode_screen(
            message=f"Crane Mode: Submode {submode}\nPress 'q' to go back.",
            on_key=on_key,
        )


class AutoNavMode(BaseMode):
    submodes = ["manual_auto_nav", "target_auto_nav", "custom_nav"]

    def enter(self):
        self.app.horizontal_select(self.submodes, self.handle_submode_select)

    def handle_submode_select(self, submode):
        def on_key(key):
            self.app.car_controller.auto_control(submode, key)
            if key == "q":
                self.app.car_controller.auto_control(submode, key=key)

        self.show_submode_screen(
            message=f"AutoNav Mode: Submode {submode}\nPress 'q' to go back.",
            on_key=on_key,
        )


class AutoArmMode(BaseMode):
    submodes = ["auto_arm_human", "auto_grab_once", "full_auto_tasks", "full_auto_task2"]

    def enter(self):
        self.auto_running = False
        self.app.horizontal_select(self.submodes, self.handle_submode_select)

    def handle_submode_select(self, submode):
        if submode == "auto_arm_human":
            def on_key(key):
                self.app.arm_controller.auto_control(mode=submode, key=key)
                if key == "q":
                    self.app.arm_controller.auto_control(mode=submode, key=key)

            self.show_submode_screen(
                message=f"AutoArm Mode: Submode {submode}\nPress 'q' to go back.",
                on_key=on_key,
            )

        elif submode in ["auto_grab_once", "full_auto_tasks", "full_auto_task2"]:
            self.auto_running = True
            full_auto = submode in ["full_auto_tasks", "full_auto_task2"]

            if submode == "full_auto_task2":
                task_mode = "task2"
            elif submode == "full_auto_tasks":
                task_mode = "grab_bear_then_bridge"
            else:
                task_mode = "task1"

            def on_key(key):
                if key == "q":
                    self.auto_running = False
                    self.app.car_controller.manual_control("z")
                    self.app.main_menu()

            self.show_submode_screen(
                message=(
                    "AutoArm Mode: Submode auto_grab_once\n"
                    "自動流程：看到球 → 靠近 → 夾取\n"
                    "Press 'q' to stop and go back."
                ),
                on_key=on_key,
            )

            threading.Thread(
                target=lambda: self._auto_grab_once_worker(
                    full_auto=full_auto,
                    task_mode=task_mode
                ),
                daemon=True
            ).start()

    def _auto_grab_once_worker(self, full_auto=False, task_mode="task1"):
        """
        搜尋熊 → 穩定鎖定 → Nav2 導航到熊附近 → 最後視覺重新對準 → 夾取
        """

        print("啟動 auto_grab_once：搜尋熊並整合 Nav2 導航...")

        ros = self.app.arm_controller.ros_communicator
        nav = self.app.car_controller.nav_processing

        if task_mode == "task2":
            state = "TASK2_ASCENT_SCRIPT"
        elif task_mode == "grab_bear_then_bridge":
            state = "TASK2_PREPARE_TO_BEAR"
        else:
            state = "SEARCH_BEAR"

        run_task1_after_task2 = False
        run_task2_after_task1 = False
        grab_bear_then_bridge = (task_mode == "grab_bear_then_bridge")

        bear_x = None
        bear_y = None
        goal_x = None
        goal_y = None

        search_dir = "e"
        lost_count = 0
        final_lost_count = 0
        stable_seen_count = 0
        grab_center_stable_count = 0
        task3_center_stable_count = 0
        search_fail_round = 0
        MAX_SEARCH_FAIL_ROUND = 20

        CENTER_TOLERANCE = 150
        LOCK_REQUIRED_COUNT = 5
        MAX_LOCK_DISTANCE = 3.5
        MIN_LOCK_DISTANCE = 0.45
        STAND_OFF = 0.20
        GRAB_READY_DISTANCE = 0.32
        GRAB_CENTER_TOLERANCE = 33
        APPROACH_CENTER_TOLERANCE = 110
        GRAB_STABLE_REQUIRED = 3
        GRAB_TARGET_DELTA_X = 20.0
        IMAGE_MIN_GRAB_CENTER_Y = 380
        IMAGE_FORCE_GRAB_CENTER_Y = 430

        HIGH_BEAR_CENTER_Y_THRESHOLD = 210

        nav_goal = None
        nav_next_state = None
        nav_started = False
        latest_amcl_pose = None
        start_goal = None
        RETURN_GOAL_TOLERANCE = 0.55
        # 這個是你自己在 Foxglove / map 上定義的放熊點
        # 先暫時填你想回去的 map 座標
        HOME_GOAL = [3.216, 3.703]
        TASK2_BRIDGE_TOP_GOAL = [0.0, 0.0]   # 先暫填，等一下用 Foxglove 找橋頂座標
        TASK2_ASCENT_TOLERANCE = 0.55
        TASK2_FORWARD_1_TIME = 5.0
        TASK2_TURN_LEFT_TIME = 6.3
        TASK2_FORWARD_2_TIME = 19.0
        TASK2_FORWARD_3_TIME = 1.0
        TASK2_TURN_LEFT_2_TIME = 0.0
        TASK2_FORWARD_4_TIME = 1.0
        TASK2_TO_TASK1_TURN_LEFT_TIME = 7.0
        TASK2_TO_TASK1_FORWARD_TIME = 0.8
        # full_auto_tasks：先往前、左轉 ，讓橋前熊在正前方
        BRIDGE_FRONT_FORWARD_TIME = 7.0
        BRIDGE_FRONT_TURN_LEFT_TIME = 6.2

        # 夾起橋前熊後，直接往前上橋下橋
        CARRY_BRIDGE_FORWARD_TIME = 13.8
        CARRY_BRIDGE_PUSH_TIME = 0.7
        CARRY_DROP_TURN_LEFT_TIME = 6.7
        CARRY_DROP_FORWARD_TIME =5.8
        CARRY_DROP_TURN_LEFT_2_TIME = 7.0
        CARRY_DROP_FORWARD_q2_TIME = 23.5

        # Task3：Task2 結束後，先左轉 90 度，再偵測 knode/door target
        RUN_TASK3_AFTER_BRIDGE = True

        TASK3_TURN_LEFT_TIME = 5.8

        # Task3：左轉後先往前離開橋邊，再開始偵測 knode
        # Task3：固定路徑到門前
        TASK3_FORWARD_BEFORE_DETECT_TIME = 14.0

        # 90 度轉彎時間，先沿用原本 6.5
        TASK3_FIXED_LEFT_90_TIME = TASK3_TURN_LEFT_TIME
        TASK3_FIXED_RIGHT_90_TIME = TASK3_TURN_LEFT_TIME

        # 左轉 90 度後往前靠近門
        TASK3_FIXED_FORWARD_TO_DOOR_TIME = 6.5
        # 最後右轉後，再往前進到門前，再開始 YOLO
        TASK3_FORWARD_AFTER_FINAL_RIGHT_TIME = 10.0

        # Task3：門把/knode 對準與靠近
        TASK3_CENTER_TOLERANCE = 70          # 大偏移修正
        TASK3_FINE_CENTER_TOLERANCE = 35     # 近距離精準置中
        TASK3_ALIGN_STABLE_REQUIRED = 3      # 連續置中幾次才算真的對
        
        TASK3_FOUND_THRESHOLD = 0.25

        TASK3_CENTER_TOLERANCE = 45
        TASK3_FINE_CENTER_TOLERANCE = 22
        TASK3_ALIGN_STABLE_REQUIRED = 4

        # 0.55 太遠，會變成離門把還很遠就準備壓
        TASK3_PRESS_READY_DISTANCE = 0.38

        TASK3_APPROACH_STEP_TIME = 0.08
        TASK3_APPROACH_WAIT_TIME = 0.25

        # 第一輪先只測「能不能對準門把」，不要真的壓
        TASK3_DEBUG_ALIGN_ONLY = True
        TASK3_HANDLE_MIN_CENTER_Y = 150
        TASK3_HANDLE_MAX_CENTER_Y = 330

        TASK3_APPROACH_STEP_TIME = 0.12
        TASK3_APPROACH_WAIT_TIME = 0.22

        TASK3_OBSERVE_TIME = 5.0

        # 壓門把與推門
        TASK3_PRESS_POINT_X = 0.20           # 手臂往前伸的位置，先保守
        TASK3_PRESS_POINT_Y = 0.0
        TASK3_PRESS_POINT_Z = 0.08           # 門把比熊高，所以 z 不要用 -0.06
        TASK3_PRESS_SETTLE_TIME = 2.0
        TASK3_PUSH_DOOR_TIME = 1.6
        # 門把最後不一定要在畫面正中央
        # 先用 0，之後看實測往左或往右調
        TASK3_HANDLE_TARGET_DELTA_X = 0.0

        # 近距離壓門把前要更嚴格置中
        TASK3_PRESS_FINE_TOLERANCE = 18

        # 不只看 distance，也看 center_y，避免遠遠讀錯深度就按
        TASK3_PRESS_READY_CENTER_Y = 360

        # YOLO 轉 map 座標用
        IMAGE_WIDTH = 640.0
        CAMERA_HFOV = math.radians(60.0)

        def amcl_pose_callback(msg):
            nonlocal latest_amcl_pose, start_goal

            latest_amcl_pose = msg

            # FULL_AUTO：第一次收到 AMCL pose 就記成出發點
            # 不等待、不阻塞、不讓車停止
            if full_auto and start_goal is None:
                p = msg.pose.pose.position
                start_goal = [p.x, p.y]
                print(f"FULL_AUTO: 已自動記錄出發點 start_goal = [{p.x:.3f}, {p.y:.3f}]")

        try:
            self._auto_amcl_pose_sub = ros.create_subscription(
                PoseWithCovarianceStamped,
                "/amcl_pose",
                amcl_pose_callback,
                10
            )
            print("FULL_AUTO: 已訂閱 /amcl_pose，準備記錄出發點。")
        except Exception as e:
            print(f"FULL_AUTO: 訂閱 /amcl_pose 失敗：{e}")

        def wait_and_record_start_pose(timeout=8.0):
            nonlocal latest_amcl_pose

            start_time = time.time()
            while self.auto_running and time.time() - start_time < timeout:
                amcl_msg = get_current_amcl_pose()

                if amcl_msg is not None:
                    p = amcl_msg.pose.pose.position
                    print(f"FULL_AUTO: 記錄本次出發點 start_goal = [{p.x:.3f}, {p.y:.3f}]")
                    return [p.x, p.y]

                print("FULL_AUTO: 等待 /amcl_pose 以記錄本次出發點...")
                time.sleep(0.2)

            print("FULL_AUTO: 沒收到 /amcl_pose，不能開始全自動。")
            return None

        def ensure_start_goal(reason=""):
            # 這版不用 start_goal，避免等 /amcl_pose 卡住
            return True

        def get_current_amcl_pose():
            nonlocal latest_amcl_pose

            if latest_amcl_pose is not None:
                return latest_amcl_pose

            try:
                msg = ros.get_latest_amcl_pose()
                if msg is not None:
                    latest_amcl_pose = msg
                    return msg
            except Exception as e:
                print(f"AMCL_FALLBACK: 讀取 ros.get_latest_amcl_pose() 失敗：{e}")

            return None
            
        def estimate_bear_goal_from_yolo(distance, delta_x):
            """
            用目前 AMCL 車子位置 + YOLO distance/delta_x
            估算熊在 map 上的座標，並產生 Nav2 要去的 goal。
            """
            if latest_amcl_pose is None:
                print("YOLO_TO_MAP: 沒有 AMCL pose，不能把熊轉成 map 座標")
                return None

            p = latest_amcl_pose.pose.pose.position
            q = latest_amcl_pose.pose.pose.orientation
            yaw = get_yaw_from_quaternion(q.z, q.w)

            # delta_x > 0 表示熊在畫面右邊
            # 如果測出來左右反了，就把這行前面的負號拿掉
            bearing = - (delta_x / (IMAGE_WIDTH / 2.0)) * (CAMERA_HFOV / 2.0)

            target_yaw = yaw + bearing

            bear_x = p.x + distance * math.cos(target_yaw)
            bear_y = p.y + distance * math.sin(target_yaw)

            # 不要導航到熊身上，停在熊前面一點
            approach_dist = max(distance - STAND_OFF, 0.35)
            goal_x = p.x + approach_dist * math.cos(target_yaw)
            goal_y = p.y + approach_dist * math.sin(target_yaw)

            print(
                f"YOLO_TO_MAP: robot=({p.x:.2f},{p.y:.2f}), "
                f"bear=({bear_x:.2f},{bear_y:.2f}), "
                f"goal=({goal_x:.2f},{goal_y:.2f}), "
                f"distance={distance:.2f}, delta_x={delta_x:.1f}"
            )

            return bear_x, bear_y, goal_x, goal_y

        def tap_arm(axis, key, times, delay=0.06):
            for _ in range(times):
                if not self.auto_running:
                    return
                self.app.arm_controller.manual_control(axis, key)
                time.sleep(delay)

        def drive_for(cmd, duration, interval=0.03, stop=True):
            start_t = time.time()

            while self.auto_running and time.time() - start_t < duration:
                ros.publish_car_control(cmd)
                time.sleep(interval)

            if stop:
                ros.publish_car_control("STOP")
                time.sleep(0.08)
        
        def begin_nav_to(goal, next_state):
            nonlocal nav_goal, nav_next_state, nav_started

            nav_goal = goal
            nav_next_state = next_state
            nav_started = False

            ros.publish_car_control("STOP")
            time.sleep(0.3)
            nav.reset_nav_process()

            print(f"BEGIN_NAV: goal={nav_goal}, next={nav_next_state}")


        def run_nav_step():
            nonlocal state, nav_started

            if nav_goal is None:
                print("NAV_GOAL: nav_goal 是 None，停止。")
                ros.publish_car_control("STOP")
                self.auto_running = False
                return

            if not nav_started:
                print(f"NAV_GOAL: 開始導航到 {nav_goal}")
                nav_started = True

            # 先看目前離回程目標多遠
            dist_to_goal = None
            amcl_msg = get_current_amcl_pose()

            if amcl_msg is not None:
                p = amcl_msg.pose.pose.position
                dist_to_goal = math.hypot(p.x - nav_goal[0], p.y - nav_goal[1])
                print(f"NAV_GOAL: AMCL distance to goal = {dist_to_goal:.3f} m")
            else:
                print("NAV_GOAL: 尚未收到 AMCL pose，先等。")
                ros.publish_car_control("STOP")
                time.sleep(0.3)
                return

            # 只有真的接近起點，才可以 DROP
            if dist_to_goal < RETURN_GOAL_TOLERANCE:
                print(f"NAV_GOAL: 已回到起點附近，切換到 {nav_next_state}")
                ros.publish_car_control("STOP")
                time.sleep(0.8)
                nav.reset_nav_process()
                state = nav_next_state
                return

            action_key = nav.get_action_from_nav2_plan_no_dynamic_p_2_p(
                goal_coordinates=nav_goal
            )

            print(f"NAV_GOAL action={action_key}")

            # 重點：STOP 不代表完成，也不要一直 reset
            # 它可能只是 Nav2 還沒算出 plan
            if action_key == "STOP":
                print("NAV_GOAL: action=STOP，但還沒回到起點，等待 Nav2 產生路徑。")
                ros.publish_car_control("STOP")
                time.sleep(0.5)
                return

            # 如果現在是抱著熊回 DROP_BEAR，只降速，不再每次 STOP
            drive_key = action_key

            if nav_next_state == "DROP_BEAR":
                if action_key == "FORWARD":
                    drive_key = "FORWARD_SLOW"
                    step_time = 0.25

                elif action_key == "CLOCKWISE_ROTATION":
                    drive_key = "CLOCKWISE_ROTATION"
                    step_time = 0.35

                elif action_key == "COUNTERCLOCKWISE_ROTATION":
                    drive_key = "COUNTERCLOCKWISE_ROTATION"
                    step_time = 0.35

                else:
                    drive_key = action_key
                    step_time = 0.18

                print(f"NAV_GOAL: 抱熊回程 {action_key} -> {drive_key}")
                ros.publish_car_control(drive_key)
                time.sleep(step_time)

            else:
                ros.publish_car_control(action_key)
                time.sleep(0.15)

            # finish_flag 只能參考，不能單獨切 DROP
            if nav.get_finish_flag():
                print("NAV_GOAL: Nav2 finish=True，但仍等 AMCL 距離真的夠近才 DROP。")
                time.sleep(0.3)
                return
        
                        
        if task_mode == "task2":
            print("TASK2: 只做上橋下橋，不放低夾爪，避免卡橋。")

            ros.publish_car_control("STOP")
            time.sleep(0.3)

            self.app.arm_controller.manual_control(0, "b")
            self.app.arm_controller.manual_control(1, "b")
            self.app.arm_controller.manual_control(2, "b")
            time.sleep(0.5)

        else:
            print("INIT_ARM: 搜尋前先放下爪子，避免 YOLO 把夾子誤判成熊...")

            ros.publish_car_control("STOP")
            time.sleep(0.3)

            tap_arm(1, "k", 5)
            tap_arm(2, "k", 3)

            time.sleep(0.5)

        nav.reset_nav_process()

        while self.auto_running:

            # ==================================================
            # TASK 2：固定動作上橋，先拿 Ascent 10 分
            # ==================================================
            if state == "TASK2_ASCENT_SCRIPT":
                print("TASK2_ASCENT_SCRIPT: 開始固定動作上橋。")

                ros.publish_car_control("STOP")
                time.sleep(0.5)

                print("TASK2_ASCENT_SCRIPT: 第一段往前，先靠近橋口。")
                drive_for("FORWARD_SLOW", TASK2_FORWARD_1_TIME)
                time.sleep(0.4)

                print("TASK2_ASCENT_SCRIPT: 左轉 90 度對準橋面。")
                drive_for("COUNTERCLOCKWISE_ROTATION", TASK2_TURN_LEFT_TIME)
                time.sleep(0.4)

                print("TASK2_ASCENT_SCRIPT: 第二段往前，上橋，用高頻正常速度連續送。")
                drive_for("FORWARD", TASK2_FORWARD_2_TIME, interval=0.02, stop=False)

                print("TASK2_ASCENT_SCRIPT: 再往前一點，維持推力。")
                drive_for("FORWARD", TASK2_FORWARD_3_TIME, interval=0.02, stop=False)

                print("TASK2_ASCENT_SCRIPT: 最後再往前一點，停在橋上。")
                drive_for("FORWARD", TASK2_FORWARD_4_TIME, interval=0.02, stop=True)

                print("TASK2_ASCENT_SCRIPT: 最後再往前一點，停在橋上。")
                ros.publish_car_control("FORWARD_SLOW")
                time.sleep(TASK2_FORWARD_4_TIME)
                ros.publish_car_control("STOP")

                print("TASK2_ASCENT_SCRIPT: Ascent/Descent 測試完成。")

                if run_task1_after_task2:
                    print("FULL_AUTO_TASKS: Task2 完成，準備開始 Task1 找熊。")

                    ros.publish_car_control("STOP")
                    time.sleep(1.0)

                    # 下橋後目前會看著牆，先左轉離開牆面，讓相機看到路上熊
                    print("FULL_AUTO_TASKS: 下橋後先左轉，讓視野離開牆面。")
                    drive_for("COUNTERCLOCKWISE_ROTATION", TASK2_TO_TASK1_TURN_LEFT_TIME, interval=0.03, stop=True)
                    time.sleep(0.3)

                    print("FULL_AUTO_TASKS: 往前離開橋口一點，準備搜尋熊。")
                    drive_for("FORWARD_SLOW", TASK2_TO_TASK1_FORWARD_TIME, interval=0.03, stop=True)
                    time.sleep(0.3)

                    # Task1 開始前，把爪子放到搜尋姿態
                    print("FULL_AUTO_TASKS: 切換到 Task1 搜尋姿態。")
                    tap_arm(1, "k", 5)
                    tap_arm(2, "k", 3)
                    time.sleep(0.5)

                    # 重置搜尋狀態
                    nav.reset_nav_process()
                    lost_count = 0
                    final_lost_count = 0
                    stable_seen_count = 0
                    grab_center_stable_count = 0
                    search_dir = "e"

                    task_mode = "task1"
                    state = "SEARCH_BEAR"
                    continue

                self.auto_running = False
                break

            # ==================================================
            # full_auto_tasks：先開到橋前熊的位置
            # ==================================================
            if state == "TASK2_PREPARE_TO_BEAR":
                print("TASK2_PREPARE_TO_BEAR: 往前靠近橋前區域。")

                ros.publish_car_control("STOP")
                time.sleep(0.5)

                print("TASK2_PREPARE_TO_BEAR: 往前到橋前。")
                drive_for("FORWARD_SLOW", BRIDGE_FRONT_FORWARD_TIME, interval=0.03, stop=True)
                time.sleep(0.4)

                print("TASK2_PREPARE_TO_BEAR: 左轉 ，讓橋前熊在正前方。")
                drive_for("COUNTERCLOCKWISE_ROTATION", BRIDGE_FRONT_TURN_LEFT_TIME, interval=0.03, stop=True)
                time.sleep(0.4)

                print("TASK2_PREPARE_TO_BEAR: 橋前熊應該在正前方，進入 SEARCH_BEAR。")

                lost_count = 0
                final_lost_count = 0
                stable_seen_count = 0
                grab_center_stable_count = 0
                search_dir = "e"

                state = "SEARCH_BEAR"
                continue

            # ==================================================
            # TASK 2 變形：先夾橋前熊，再帶著熊上橋下橋
            # ==================================================
            if state == "TASK2_CARRY_BEAR_SCRIPT":
                print("TASK2_CARRY_BEAR_SCRIPT: 帶著熊上橋下橋。")

                ros.publish_car_control("STOP")
                time.sleep(0.5)

                # 這裡不要動手臂，保持夾住熊
                print("TASK2_CARRY_BEAR_SCRIPT: 往前上橋下橋。")
                drive_for("FORWARD", CARRY_BRIDGE_FORWARD_TIME, interval=0.02, stop=False)

                print("TASK2_CARRY_BEAR_SCRIPT: 再補一點推力。")
                drive_for("FORWARD", CARRY_BRIDGE_PUSH_TIME, interval=0.02, stop=True)

                print("TASK2_CARRY_BEAR_SCRIPT: 停 1 秒確認橋任務計分。")
                time.sleep(1.0)

                state = "DROP_BEAR_AFTER_BRIDGE"
                continue

            # ==================================================
            # 下橋後左轉前進，放下剛剛夾住的熊
            # ==================================================
            if state == "DROP_BEAR_AFTER_BRIDGE":
                print("DROP_BEAR_AFTER_BRIDGE: 下橋後左轉，準備放熊。")

                ros.publish_car_control("STOP")
                time.sleep(0.5)

                print("DROP_BEAR_AFTER_BRIDGE: 第一次左轉，離開橋口。")
                drive_for("COUNTERCLOCKWISE_ROTATION", CARRY_DROP_TURN_LEFT_TIME, interval=0.03, stop=True)
                time.sleep(0.3)

                print("DROP_BEAR_AFTER_BRIDGE: 往前離開橋口，靠近藍色框方向。")
                drive_for("FORWARD_SLOW", CARRY_DROP_FORWARD_TIME, interval=0.03, stop=True)
                time.sleep(0.4)

                print("DROP_BEAR_AFTER_BRIDGE: 已離開橋口，改用 Nav2 回起點藍色框。")

                return_goal = start_goal if start_goal is not None else HOME_GOAL
                print(f"DROP_BEAR_AFTER_BRIDGE: Nav2 回起點 return_goal={return_goal}")

                begin_nav_to(return_goal, "DROP_BEAR")
                state = "NAV_GOAL"
                continue

                if RUN_TASK3_AFTER_BRIDGE:
                    print("FULL_AUTO_TASKS: 準備接 Task3，先左轉 90 度找 knode。")

                    ros.publish_car_control("STOP")
                    time.sleep(0.8)

                    state = "TASK3_TURN_LEFT_TO_DOOR"
                    continue

                print("FULL_AUTO_TASKS: 任務結束。")
                self.auto_running = False
                break

            # ==================================================
            # TASK 3：Task2 結束後先左轉 90 度，讓相機看向門區
            # ==================================================
            if state == "TASK3_TURN_LEFT_TO_DOOR":
                print("TASK3_TURN_LEFT_TO_DOOR: 左轉 90 度，準備往門方向離開起點區。")

                ros.publish_car_control("STOP")
                time.sleep(0.5)

                print("TASK3_TURN_LEFT_TO_DOOR: 左轉 90 度。")
                drive_for("COUNTERCLOCKWISE_ROTATION", TASK3_TURN_LEFT_TIME, interval=0.03, stop=True)
                time.sleep(0.4)

                print("TASK3_TURN_LEFT_TO_DOOR: 左轉完成，先往前 14 秒，離開橋區。")
                drive_for("FORWARD_SLOW", TASK3_FORWARD_BEFORE_DETECT_TIME, interval=0.03, stop=True)
                time.sleep(0.5)

                print("TASK3_TURN_LEFT_TO_DOOR: 固定路徑第二段：左轉 90 度。")
                drive_for("COUNTERCLOCKWISE_ROTATION", TASK3_FIXED_LEFT_90_TIME, interval=0.03, stop=True)
                time.sleep(0.4)

                print("TASK3_TURN_LEFT_TO_DOOR: 固定路徑第三段：往前 5 秒，靠近門前。")
                drive_for("FORWARD_SLOW", TASK3_FIXED_FORWARD_TO_DOOR_TIME, interval=0.03, stop=True)
                time.sleep(0.4)

                print("TASK3_TURN_LEFT_TO_DOOR: 固定路徑第四段：右轉 90 度，讓車頭重新面向門。")
                drive_for("CLOCKWISE_ROTATION", TASK3_FIXED_RIGHT_90_TIME, interval=0.03, stop=True)
                time.sleep(0.4)

                print("TASK3_TURN_LEFT_TO_DOOR: 右轉後再往前，先進到門前，不要太早相信 YOLO。")
                drive_for("FORWARD_SLOW", TASK3_FORWARD_AFTER_FINAL_RIGHT_TIME, interval=0.03, stop=True)
                time.sleep(0.5)

                print("TASK3_TURN_LEFT_TO_DOOR: 已進到門前，現在才開始用 YOLO 微調門把。")
                state = "TASK3_SEARCH_KNODE"
                continue

            # ==================================================
            # TASK 3：偵測 knode / 門把
            # 先完成「離開橋區後，原地對準門把」
            # ==================================================
            if state == "TASK3_SEARCH_KNODE":
                target_msg = ros.get_latest_yolo_target_info()

                if target_msg is None or len(target_msg.data) < 4:
                    task3_center_stable_count = 0
                    print("TASK3_SEARCH_KNODE: 尚未收到門把 target_info，慢速搜尋...")

                    ros.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
                    time.sleep(0.15)
                    ros.publish_car_control("STOP")
                    time.sleep(0.25)
                    continue

                found, distance, delta_x, center_y = (
                    target_msg.data[0],
                    target_msg.data[1],
                    target_msg.data[2],
                    target_msg.data[3],
                )

                print(
                    f"TASK3_SEARCH_KNODE: found={found:.2f}, "
                    f"distance={distance:.3f}, delta_x={delta_x:.1f}, center_y={center_y:.1f}"
                )

                # 門把常常信心值比熊低，不要用 0.5 卡死
                if found < TASK3_FOUND_THRESHOLD or distance <= 0:
                    task3_center_stable_count = 0
                    print("TASK3_SEARCH_KNODE: 沒穩定看到門把，繼續慢速掃描。")

                    ros.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
                    time.sleep(0.15)
                    ros.publish_car_control("STOP")
                    time.sleep(0.25)
                    continue

                # 門把不要求在畫面下方，但要在合理高度範圍
                if center_y < TASK3_HANDLE_MIN_CENTER_Y or center_y > TASK3_HANDLE_MAX_CENTER_Y:
                    task3_center_stable_count = 0
                    print(
                        f"TASK3_SEARCH_KNODE: center_y={center_y:.1f} 不在門把合理高度 "
                        f"{TASK3_HANDLE_MIN_CENTER_Y}~{TASK3_HANDLE_MAX_CENTER_Y}，"
                        "不前進，繼續掃描。"
                    )

                    ros.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
                    time.sleep(0.12)
                    ros.publish_car_control("STOP")
                    time.sleep(0.25)
                    continue

                abs_err = abs(delta_x)

                # 偏很多就轉久一點，偏一點就轉短一點
                turn_time = min(0.16, max(0.035, abs_err / 900.0))

                # 門把在畫面右邊，車右轉讓它靠近中心
                if delta_x > TASK3_CENTER_TOLERANCE:
                    task3_center_stable_count = 0
                    print(
                        f"TASK3_SEARCH_KNODE: 門把偏右 delta_x={delta_x:.1f}，"
                        f"右轉修正 {turn_time:.3f}s"
                    )

                    ros.publish_car_control("CLOCKWISE_ROTATION_SLOW")
                    time.sleep(turn_time)
                    ros.publish_car_control("STOP")
                    time.sleep(0.25)
                    continue

                # 門把在畫面左邊，車左轉讓它靠近中心
                if delta_x < -TASK3_CENTER_TOLERANCE:
                    task3_center_stable_count = 0
                    print(
                        f"TASK3_SEARCH_KNODE: 門把偏左 delta_x={delta_x:.1f}，"
                        f"左轉修正 {turn_time:.3f}s"
                    )

                    ros.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
                    time.sleep(turn_time)
                    ros.publish_car_control("STOP")
                    time.sleep(0.25)
                    continue

                # 到這裡代表門把已經在畫面中央附近
                task3_center_stable_count += 1
                print(
                    f"TASK3_SEARCH_KNODE: 門把置中穩定 "
                    f"{task3_center_stable_count}/{TASK3_ALIGN_STABLE_REQUIRED}, "
                    f"distance={distance:.3f}, delta_x={delta_x:.1f}"
                )

                ros.publish_car_control("STOP")
                time.sleep(0.20)

                if task3_center_stable_count < TASK3_ALIGN_STABLE_REQUIRED:
                    continue

                # 第一輪先只測對準，不要前進、不壓門把
                if TASK3_DEBUG_ALIGN_ONLY:
                    print(
                        "TASK3_SEARCH_KNODE: DEBUG_ALIGN_ONLY=True，"
                        "目前只測試是否能把門把對到畫面中央，不前進、不壓門把。"
                    )
                    task3_center_stable_count = 0
                    time.sleep(0.5)
                    continue

                # 門把版：夠近 + 左右夠準 + 高度在合理範圍，才壓門把
                press_error = delta_x - TASK3_HANDLE_TARGET_DELTA_X
                close_enough = distance <= TASK3_PRESS_READY_DISTANCE
                handle_y_ok = (
                    TASK3_HANDLE_MIN_CENTER_Y <= center_y <= TASK3_HANDLE_MAX_CENTER_Y
                )

                if close_enough and handle_y_ok:
                    if press_error > TASK3_PRESS_FINE_TOLERANCE:
                        task3_center_stable_count = 0
                        print(
                            f"TASK3_SEARCH_KNODE: 已接近門把，但門把偏右 "
                            f"delta_x={delta_x:.1f}, press_error={press_error:.1f}，微右修。"
                        )
                        ros.publish_car_control("CLOCKWISE_ROTATION_SLOW")
                        time.sleep(0.015)
                        ros.publish_car_control("STOP")
                        time.sleep(0.30)
                        continue

                    if press_error < -TASK3_PRESS_FINE_TOLERANCE:
                        task3_center_stable_count = 0
                        print(
                            f"TASK3_SEARCH_KNODE: 已接近門把，但門把偏左 "
                            f"delta_x={delta_x:.1f}, press_error={press_error:.1f}，微左修。"
                        )
                        ros.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
                        time.sleep(0.015)
                        ros.publish_car_control("STOP")
                        time.sleep(0.30)
                        continue

                    print(
                        f"TASK3_SEARCH_KNODE: 已正對且夠近，準備壓門把。 "
                        f"distance={distance:.3f}, center_y={center_y:.1f}, "
                        f"delta_x={delta_x:.1f}, press_error={press_error:.1f}"
                    )
                    task3_center_stable_count = 0
                    state = "TASK3_PRESS_HANDLE"
                    continue

                # 正對但還太遠，才小步前進
                print(
                    f"TASK3_SEARCH_KNODE: 已正對門把但還太遠 distance={distance:.3f}，"
                    "小步前進後重新偵測。"
                )

                task3_center_stable_count = 0
                ros.publish_car_control("FORWARD_SLOW")
                time.sleep(TASK3_APPROACH_STEP_TIME)
                ros.publish_car_control("STOP")
                time.sleep(TASK3_APPROACH_WAIT_TIME)
                continue

            # ==================================================
            # TASK 3：手臂放到門把上方並往下壓
            # ==================================================
            if state == "TASK3_PRESS_HANDLE":
                print("TASK3_PRESS_HANDLE: 停車，準備把手臂放到門把上方。")

                ros.publish_car_control("STOP")
                time.sleep(0.5)

                # 先讓手臂回比較可控的初始姿態
                self.app.arm_controller.manual_control(0, "b")
                self.app.arm_controller.manual_control(1, "b")
                self.app.arm_controller.manual_control(2, "b")
                time.sleep(0.8)

                # 設定一個在 arm_ik_base 前方、偏高的點，對應門把附近
                clicked = PointStamped()
                clicked.header.frame_id = "arm_ik_base"
                clicked.header.stamp = ros.get_clock().now().to_msg()

                clicked.point.x = TASK3_PRESS_POINT_X
                clicked.point.y = TASK3_PRESS_POINT_Y
                clicked.point.z = TASK3_PRESS_POINT_Z

                print(
                    f"TASK3_PRESS_HANDLE: 設定門把按壓點 "
                    f"x={clicked.point.x:.2f}, y={clicked.point.y:.2f}, z={clicked.point.z:.2f}"
                )

                ros.clicked_point_callback(clicked)
                time.sleep(0.8)
                ros.clicked_point_callback(clicked)
                time.sleep(0.8)

                # 不用 auto_arm_human 的 g，因為 g 是「夾取」流程，會開合夾爪
                # 開門只需要把手臂放到門把附近，再往下壓
                print("TASK3_PRESS_HANDLE: 不執行 g 夾取，改用手臂固定姿態壓門把。")

                # 先讓手臂稍微往前/往上到門把附近
                # 如果實測太高或太低，再調這幾個 tap 次數
                tap_arm(1, "i", 2, delay=0.08)
                tap_arm(2, "i", 2, delay=0.08)

                time.sleep(0.5)

                print("TASK3_PRESS_HANDLE: 開始往下壓門把。")

                # 往下壓門把
                tap_arm(1, "k", 3, delay=0.08)
                tap_arm(2, "k", 4, delay=0.08)

                time.sleep(0.5)

                print("TASK3_PRESS_HANDLE: 門把按壓完成，準備往前推門。")
                state = "TASK3_PUSH_DOOR"
                continue


            # ==================================================
            # TASK 3：按住門把後，車子往前推開門
            # ==================================================
            if state == "TASK3_PUSH_DOOR":
                print("TASK3_PUSH_DOOR: 按住門把，車子往前推門。")

                ros.publish_car_control("FORWARD")
                time.sleep(TASK3_PUSH_DOOR_TIME)

                ros.publish_car_control("STOP")
                time.sleep(0.5)

                print("TASK3_PUSH_DOOR: 推門完成，Task3 暫定完成。")
                self.auto_running = False
                break


            # ==================================================
            # STATE 1：搜尋熊，不要看到一幀就立刻鎖定
            # ==================================================
            if state == "SEARCH_BEAR":
                target_msg = ros.get_latest_yolo_target_info()

                if target_msg is None or len(target_msg.data) < 4:
                    print("SEARCH_BEAR: 尚未收到 /yolo/target_info，慢速搜尋...")
                    ros.publish_car_control("CLOCKWISE_ROTATION_SLOW")
                    time.sleep(0.15)
                    ros.publish_car_control("STOP")
                    time.sleep(0.35)
                    continue

                found, distance, delta_x, center_y = (
                    target_msg.data[0],
                    target_msg.data[1],
                    target_msg.data[2],
                    target_msg.data[3],
                )

                print(
                    f"SEARCH_BEAR: found={found:.1f}, "
                    f"distance={distance:.3f}, delta_x={delta_x:.1f}, center_y={center_y:.1f}"
                )

                # 沒看到熊：左右慢慢掃，不要停死
                # 沒看到熊：不要永遠原地轉，找太久就換位置
                if found < 0.5 or distance <= 0:
                    stable_seen_count = 0
                    lost_count += 1

                    print(f"SEARCH_BEAR: 目前看不到熊，lost_count={lost_count}")

                    # 找太久了：先離開目前位置，不要繼續看牆
                    if lost_count >= 8:
                        print("SEARCH_BEAR: 找太久了，先脫離牆邊再重新搜尋。")

                        ros.publish_car_control("STOP")
                        time.sleep(0.2)

                        # 先倒車
                        ros.publish_car_control("BACKWARD_SLOW")
                        time.sleep(0.6)

                        ros.publish_car_control("STOP")
                        time.sleep(0.2)

                        # 換方向
                        ros.publish_car_control("CLOCKWISE_ROTATION_SLOW")
                        time.sleep(0.8)

                        ros.publish_car_control("STOP")
                        time.sleep(0.2)

                        # 再往前移動到新位置
                        #ros.publish_car_control("FORWARD_SLOW")
                        #time.sleep(0.6)

                        ros.publish_car_control("STOP")
                        time.sleep(0.3)

                        lost_count = 0
                        stable_seen_count = 0
                        continue

                    # 還沒超過 8 次，就先小角度掃描
                    ensure_start_goal("SEARCH_BEAR first scan")

                    if search_dir == "r":
                        ros.publish_car_control("CLOCKWISE_ROTATION_SLOW")
                    else:
                        ros.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")

                    time.sleep(0.25)
                    ros.publish_car_control("STOP")
                    time.sleep(0.15)
                    continue

                # 看到熊，重置 lost
                lost_count = 0

                # 熊在畫面非常上方，才判定可能是橋上 / 高處熊
                # 不要把一般遠方平地熊也略過
                if (not grab_bear_then_bridge) and center_y < HIGH_BEAR_CENTER_Y_THRESHOLD and distance > 1.0:
                    stable_seen_count = 0
                    print(
                        f"SEARCH_BEAR: center_y={center_y:.1f}, distance={distance:.2f}，"
                        "高度疑似太高，先略過。"
                    )

                    ros.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
                    time.sleep(0.12)
                    ros.publish_car_control("STOP")
                    time.sleep(0.25)
                    continue

                # 熊偏太右：小修正，不要馬上鎖定
                if delta_x > CENTER_TOLERANCE:
                    stable_seen_count = 0
                    print("SEARCH_BEAR: 熊偏右，慢速右轉對準...")
                    ros.publish_car_control("CLOCKWISE_ROTATION_SLOW")
                    time.sleep(0.12)
                    ros.publish_car_control("STOP")
                    time.sleep(0.15)
                    continue

                # 熊偏太左：小修正，不要馬上鎖定
                elif delta_x < -CENTER_TOLERANCE:
                    stable_seen_count = 0
                    print("SEARCH_BEAR: 熊偏左，慢速左轉對準...")
                    ros.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
                    time.sleep(0.12)
                    ros.publish_car_control("STOP")
                    time.sleep(0.15)
                    continue

                # 距離不合理，不鎖定
                if distance < MIN_LOCK_DISTANCE or distance > MAX_LOCK_DISTANCE:
                    stable_seen_count = 0
                    print("SEARCH_BEAR: 距離不合理，暫不鎖定，繼續觀察...")
                    ros.publish_car_control("STOP")
                    time.sleep(0.20)
                    continue

                # 看到熊 + 在畫面中央附近 + 距離合理：累積穩定次數
                stable_seen_count += 1
                print(f"SEARCH_BEAR: 熊穩定鎖定中 {stable_seen_count}/{LOCK_REQUIRED_COUNT}")

                if stable_seen_count < LOCK_REQUIRED_COUNT:
                    ros.publish_car_control("STOP")
                    time.sleep(0.20)
                    continue

                # 穩定看到夠多次，改用原本視覺方式靠近熊
                print("SEARCH_BEAR: 已穩定看到熊，改用原本 FINAL_ALIGN 視覺靠近。")

                ros.publish_car_control("STOP")
                time.sleep(0.3)

                final_lost_count = 0
                state = "ARM_PREPARE"
                continue
            # ===========================
            # STATE 2：Nav2 導航到熊附近
            # ==================================================
            elif state == "NAV_TO_BEAR":
                action_key = nav.get_action_from_nav2_plan_no_dynamic_p_2_p(
                    goal_coordinates=[goal_x, goal_y]
                )

                print(f"NAV_TO_BEAR action={action_key}")

                ros.publish_car_control(action_key)
                time.sleep(0.15)

                if nav.get_finish_flag():
                    print("NAV_TO_BEAR: Nav2 已到達熊附近，進入最後視覺對準。")
                    ros.publish_car_control("STOP")
                    time.sleep(0.8)
                    final_lost_count = 0
                    state = "FINAL_ALIGN"
                    continue
            # ==================================================
            # STATE 2.5：調整手臂為抓取預備姿態，避免靠近時推倒熊
            # ==================================================
            elif state == "ARM_PREPARE":
                print("ARM_PREPARE: 調整 0/1/2 軸，讓爪子不要像推土機一樣撞熊...")

                ros.publish_car_control("STOP")
                time.sleep(0.3)

                # 先把主要關節稍微抬回一點，避免爪子貼地推熊
                tap_arm(1, "i", 2)

                # 末端爪子稍微收/抬，避免前進時先撞倒熊
                tap_arm(2, "i", 3)

                # 0 軸先回預設，避免左右歪掉
                self.app.arm_controller.manual_control(0, "b")
                time.sleep(0.2)

                state = "FINAL_ALIGN"
                continue
            # ==================================================
            # STATE 3：最後用 YOLO 重新找熊、靠近
            # ==================================================
            elif state == "FINAL_ALIGN":
                target_msg = ros.get_latest_yolo_target_info()

                if target_msg is None or len(target_msg.data) < 4:
                    final_lost_count += 1
                    print(f"FINAL_ALIGN: 沒有 target_info，慢速搜尋... lost={final_lost_count}")

                    ros.publish_car_control("CLOCKWISE_ROTATION_SLOW")
                    time.sleep(0.25)
                    ros.publish_car_control("STOP")
                    time.sleep(0.15)

                    if final_lost_count > 12:
                        print("FINAL_ALIGN: 搜尋太久仍沒看到熊，回到 SEARCH_BEAR 重新找。")
                        final_lost_count = 0
                        stable_seen_count = 0
                        nav.reset_nav_process()
                        state = "SEARCH_BEAR"

                    continue

                found, distance, delta_x, center_y = (
                    target_msg.data[0],
                    target_msg.data[1],
                    target_msg.data[2],
                    target_msg.data[3],
                )

                print(
                    f"FINAL_ALIGN: found={found:.1f}, "
                    f"distance={distance:.3f}, delta_x={delta_x:.1f}, center_y={center_y:.1f}"
                )

                if found < 0.5 or distance <= 0:
                    final_lost_count += 1
                    print(f"FINAL_ALIGN: 沒看到熊，慢速搜尋... lost={final_lost_count}")

                    ros.publish_car_control("CLOCKWISE_ROTATION_SLOW")
                    time.sleep(0.25)
                    ros.publish_car_control("STOP")
                    time.sleep(0.15)

                    if final_lost_count > 12:
                        print("FINAL_ALIGN: 搜尋太久仍沒看到熊，回到 SEARCH_BEAR 重新找。")
                        final_lost_count = 0
                        stable_seen_count = 0
                        nav.reset_nav_process()
                        state = "SEARCH_BEAR"

                    continue

                # 有看到熊，重置 final lost
                final_lost_count = 0

                # 如果熊已經在畫面很下方，代表實際上已經很近
                # 不管 distance 還顯示多遠，都不要再往前撞
                if center_y >= IMAGE_FORCE_GRAB_CENTER_Y:
                    print(
                        f"FINAL_ALIGN: center_y={center_y:.1f} 已經非常靠近，"
                        "停止前進，進入置中確認。"
                    )
                    ros.publish_car_control("STOP")
                    time.sleep(0.3)

                # 深度太怪，不要鎖定
                if distance > 5.0 and center_y < IMAGE_FORCE_GRAB_CENTER_Y:
                    print(
                        f"FINAL_ALIGN: 距離值太大 distance={distance:.3f}，"
                        f"且 center_y={center_y:.1f} 還沒到強制夾取區，慢慢前進靠近。"
                    )
                    ros.publish_car_control("FORWARD_SLOW")
                    time.sleep(0.07)
                    ros.publish_car_control("STOP")
                    time.sleep(0.22)
                    continue

                # 已經夠近，進入精準置中確認
                distance_ready = distance <= GRAB_READY_DISTANCE and center_y >= IMAGE_MIN_GRAB_CENTER_Y
                visual_ready = center_y >= IMAGE_FORCE_GRAB_CENTER_Y

                if distance_ready or visual_ready:
                    center_error = delta_x - GRAB_TARGET_DELTA_X

                    # 夠近後不要大轉，只做超小修正，避免車身晃動把熊碰歪
                    if center_error > GRAB_CENTER_TOLERANCE:
                        grab_center_stable_count = 0
                        print(
                            f"FINAL_ALIGN: 近距離精準置中，熊偏右 "
                            f"delta_x={delta_x:.1f}, error={center_error:.1f}，微右修。"
                        )
                        ros.publish_car_control("CLOCKWISE_ROTATION_SLOW")
                        time.sleep(0.015)
                        ros.publish_car_control("STOP")
                        time.sleep(0.35)
                        continue

                    elif center_error < -GRAB_CENTER_TOLERANCE:
                        grab_center_stable_count = 0
                        print(
                            f"FINAL_ALIGN: 近距離精準置中，熊偏左 "
                            f"delta_x={delta_x:.1f}, error={center_error:.1f}，微左修。"
                        )
                        ros.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
                        time.sleep(0.015)
                        ros.publish_car_control("STOP")
                        time.sleep(0.35)
                        continue

                    # 不是一置中就抓，要連續穩定幾次
                    grab_center_stable_count += 1
                    print(
                        f"FINAL_ALIGN: 近距離置中穩定 "
                        f"{grab_center_stable_count}/{GRAB_STABLE_REQUIRED}, "
                        f"distance={distance:.3f}, delta_x={delta_x:.1f}"
                    )

                    ros.publish_car_control("STOP")
                    time.sleep(0.20)

                    if grab_center_stable_count < GRAB_STABLE_REQUIRED:
                        continue

                    print(
                        f"FINAL_ALIGN: 距離 {distance:.3f} 已進入可夾範圍，"
                        f"且連續穩定置中 delta_x={delta_x:.1f}，直接進入夾取。"
                    )

                    grab_center_stable_count = 0
                    state = "SET_CLICK_POINT"
                    continue

                # 看到熊後，不要只轉圈；小修正 + 前進
                if delta_x > 260:
                    print("FINAL_ALIGN: 熊明顯偏右，右轉一點點後前進...")
                    ros.publish_car_control("CLOCKWISE_ROTATION_SLOW")
                    time.sleep(0.05)
                    ros.publish_car_control("STOP")
                    time.sleep(0.30)

                elif delta_x < -260:
                    print("FINAL_ALIGN: 熊明顯偏左，左轉一點點後前進...")
                    ros.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
                    time.sleep(0.05)
                    ros.publish_car_control("STOP")
                    time.sleep(0.30)

                elif delta_x > APPROACH_CENTER_TOLERANCE:
                    print("FINAL_ALIGN: 熊稍微偏右，右修對準...")
                    ros.publish_car_control("CLOCKWISE_ROTATION_SLOW")
                    time.sleep(0.02)
                    ros.publish_car_control("STOP")
                    time.sleep(0.25)

                elif delta_x < -APPROACH_CENTER_TOLERANCE:
                    print("FINAL_ALIGN: 熊稍微偏左，左修對準...")
                    ros.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
                    time.sleep(0.02)
                    ros.publish_car_control("STOP")
                    time.sleep(0.25)

                else:
                    print("FINAL_ALIGN: 熊在前方，短距離慢慢前進...")
                    ros.publish_car_control("FORWARD_SLOW")
                    time.sleep(0.05)
                    ros.publish_car_control("STOP")
                    time.sleep(0.22)

           
            # ==================================================
            # STATE 4：設定夾爪目標點
            # ==================================================
            elif state == "SET_CLICK_POINT":
                print("SET_CLICK_POINT: 將近距離目標設定在手臂前方...")

                clicked = PointStamped()
                clicked.header.frame_id = "arm_ik_base"
                clicked.header.stamp = ros.get_clock().now().to_msg()

                # 固定設定在手臂可達範圍內
                clicked.point.x = 0.16
                clicked.point.y = 0.0
                clicked.point.z = -0.06

                ros.publish_car_control("STOP")
                time.sleep(0.5)

                ros.clicked_point_callback(clicked)

                print("SET_CLICK_POINT: 第一次設定目標點，等待手臂移到目標點...")
                time.sleep(1.0)

                # 再送一次同一個 click point，避免手臂/marker 還沒穩就進 GRAB
                ros.clicked_point_callback(clicked)

                print("SET_CLICK_POINT: 第二次確認目標點，準備進入夾取...")
                time.sleep(1.0)

                state = "GRAB"
                continue

            # ==================================================
            # STATE 5：夾取熊
            # ==================================================
            elif state == "GRAB":
                print("GRAB: 開始自動夾取熊...")

                # 夾取前先確保車完全停住
                ros.publish_car_control("STOP")
                time.sleep(0.5)

                self.app.arm_controller.auto_control(
                    mode="auto_arm_human",
                    key="g"
                )

                # 保留原本夾取流程，但給它多一點時間完成閉合
                time.sleep(5.5)

                ros.publish_car_control("STOP")
                time.sleep(1.0)

                if grab_bear_then_bridge:
                    print("GRAB: 已夾起橋前熊，準備帶著熊上橋下橋。")

                    ros.publish_car_control("STOP")
                    time.sleep(1.0)

                    state = "TASK2_CARRY_BEAR_SCRIPT"
                    continue

                if not full_auto:
                    print("auto_grab_once 完成。")
                    self.auto_running = False
                    break

                print("GRAB: 夾取動作完成，進入夾取穩定階段。")
                state = "GRAB_SETTLE"
                continue

            elif state == "GRAB_SETTLE":
                print("GRAB_SETTLE: full_auto 專用暫停，不再額外抬手或後退，避免把熊弄掉。")

                # auto_grab_once 成功，代表 GRAB 結束時的姿態是可行的
                # 所以這裡先不要再動手臂，也不要先倒車
                ros.publish_car_control("STOP")
                time.sleep(2.0)

                return_goal = start_goal if start_goal is not None else HOME_GOAL

                print(f"GRAB_SETTLE: 準備回本次起點 return_goal={return_goal}")

                begin_nav_to(return_goal, "DROP_BEAR")
                state = "NAV_GOAL"
                continue

            # ==================================================
            # STATE 6：Nav2 回起點
            # ==================================================
            elif state == "NAV_GOAL":
                run_nav_step()
                continue

            # ==================================================
            # STATE 7：到起點後放下熊
            # ==================================================
            elif state == "DROP_BEAR":
                print("DROP_BEAR: 到起點，放下熊。")

                ros.publish_car_control("STOP")
                time.sleep(0.5)

                # 回到起點後，抱著熊再往前補一點，讓熊更接近藍色框框
                print("DROP_BEAR: 放熊前先左轉微調，修正回起點時的車頭角度。")
                ros.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
                time.sleep(0.18)
                ros.publish_car_control("STOP")
                time.sleep(0.3)

                print("DROP_BEAR: 放熊前往前微調，讓熊落點更靠近目標區。")
                ros.publish_car_control("FORWARD_SLOW")
                time.sleep(0.25)
                ros.publish_car_control("STOP")
                time.sleep(0.4)

                # 先降低手臂，讓熊接近地面
                tap_arm(2, "k", 3)
                tap_arm(1, "k", 2)
                time.sleep(0.5)

                # 先用 b 回預設，通常會鬆開/回復姿態
                self.app.arm_controller.manual_control(0, "b")
                self.app.arm_controller.manual_control(1, "b")
                self.app.arm_controller.manual_control(2, "b")
                time.sleep(1.0)

                # 後退一點，避免爪子還碰到熊
                ros.publish_car_control("BACKWARD_SLOW")
                time.sleep(0.4)
                ros.publish_car_control("STOP")

                print("Task 1 Recovery 完成：熊已帶回起點。")

                if RUN_TASK3_AFTER_BRIDGE:
                    print("FULL_AUTO_TASKS: Recovery 完成，準備接 Task3。")
                    ros.publish_car_control("STOP")
                    time.sleep(1.0)

                    state = "TASK3_TURN_LEFT_TO_DOOR"
                    continue

                self.auto_running = False
                break