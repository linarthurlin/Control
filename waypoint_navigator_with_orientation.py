#!/usr/bin/env python3
"""
Waypoint Recorder & Navigator for ROS2 Humble + Nav2
Real Car Implementation (PACMOD / sensor stack)
-----------------------------------------------------
RECORD MODE:
  - Subscribes to the odometry topic (default: /Odometry, configurable)
  - Saves x, y, yaw every N meters to waypoints.txt
  - Press 'S' + Enter to stop recording and save

PLAYBACK MODE:
  - Press 'P' + Enter to start playback
  - Reads waypoints.txt and sends each point as a Nav2 goal
  - Waits for goal result before sending the next

Usage:
  python3 waypoint_recorder_navigator.py
  python3 waypoint_recorder_navigator.py --ros-args \
      -p spacing_m:=2.0 \
      -p waypoints_file:=/data/waypoints.txt \
      -p odom_topic:=/your/odom/topic \
      -p odom_reliable:=false \
      -p goal_timeout_sec:=120.0

File format (3-column, yaw in radians):
  # x, y, yaw_rad
  1.234500, 5.678900, 1.570796

  Two-column legacy files are still accepted; yaw defaults to 0.0.

RViz / Foxglove topics:
  /waypoint_markers    – cyan arrows, republished at 2 Hz
  /current_goal_marker – red arrow, active Nav2 goal only

Parameters:
  spacing_m         (float,  default 2.0)      waypoint spacing in metres
  waypoints_file    (string, default <script dir>/waypoints.txt)
  goal_timeout_sec  (float,  default 120.0)    per-waypoint Nav2 timeout
  odom_topic        (string, default /Odometry) odometry topic name
  odom_reliable     (bool,   default false)     set true if publisher uses
                                                RELIABLE QoS
  nav_server        (string, default navigate_to_pose) action server name
"""

import math
import os
import select
import sys
import threading
import time

try:
    import msvcrt
except ImportError:
    msvcrt = None

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from action_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

from tf_transformations import euler_from_quaternion, quaternion_from_euler

try:
    from tf2_ros import Buffer, TransformListener
    _TF2_AVAILABLE = True
except ImportError:
    _TF2_AVAILABLE = False


class WaypointRecorderNavigator(Node):

    def __init__(self):
        super().__init__('waypoint_recorder_navigator')

        # ---------- Parameters ----------
        # waypoints_file: default to same directory as this script so it
        # works regardless of which directory the node is launched from,
        # and regardless of username / machine.
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        self.declare_parameter('spacing_m',        2.0)
        self.declare_parameter('waypoints_file',
                               os.path.join(_script_dir, 'waypoints.txt'))
        self.declare_parameter('goal_timeout_sec', 120.0)
        self.declare_parameter('odom_topic',       '/Odometry')
        self.declare_parameter('odom_reliable',    False)
        self.declare_parameter('nav_server',       'navigate_to_pose')

        self.spacing_m      = self.get_parameter('spacing_m').value
        self.waypoints_file = self.get_parameter('waypoints_file').value
        self.goal_timeout   = self.get_parameter('goal_timeout_sec').value
        self._odom_topic    = self.get_parameter('odom_topic').value
        self._odom_reliable = self.get_parameter('odom_reliable').value
        self._nav_server    = self.get_parameter('nav_server').value

        # ---------- Shared state + lock ----------
        self._lock          = threading.Lock()
        self._state         = 'RECORDING'   # RECORDING | IDLE | NAVIGATING
        self._waypoints     = []            # list of (x, y, yaw) – map frame
        self._last_wp_pos   = None          # (x, y) for spacing check only
        self._current_pose  = None          # latest (x, y, yaw) – any state

        # ---------- TF2 ----------
        if _TF2_AVAILABLE:
            self._tf_buffer   = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
        else:
            self._tf_buffer = None
            self.get_logger().warn(
                'tf2_ros not available – falling back to raw odometry. '
                'Waypoints will be in the odom frame.'
            )

        # ---------- QoS for Odometry ----------
        # odom_reliable=true  → RELIABLE  (some ROS2 drivers, rosbag replay)
        # odom_reliable=false → BEST_EFFORT (most PACMOD / sensor drivers)
        # A QoS mismatch causes silent subscription failure on the real car.
        odom_qos = QoSProfile(
            reliability=(
                ReliabilityPolicy.RELIABLE
                if self._odom_reliable
                else ReliabilityPolicy.BEST_EFFORT
            ),
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # ---------- Subscriber ----------
        self.odom_sub = self.create_subscription(
            Odometry,
            self._odom_topic,
            self._odom_callback,
            odom_qos
        )

        # ---------- Publishers ----------
        self._wp_marker_pub = self.create_publisher(
            MarkerArray, '/waypoint_markers', 10
        )
        self._goal_marker_pub = self.create_publisher(
            Marker, '/current_goal_marker', 10
        )

        # 2 Hz marker timer – ensures Foxglove sees markers even if it
        # connects after the last on-change publish.
        self._marker_timer = self.create_timer(0.5, self._republish_markers)

        # ---------- Nav2 Action Client ----------
        self._nav_client = ActionClient(
            self, NavigateToPose, self._nav_server
        )

        # ---------- Keyboard thread ----------
        self._kb_thread = threading.Thread(
            target=self._keyboard_loop, daemon=True
        )
        self._kb_thread.start()

        self.get_logger().info(
            f'\n{"="*60}\n'
            f'  Waypoint Recorder / Navigator  [REAL CAR]\n'
            f'{"="*60}\n'
            f'  State         : RECORDING\n'
            f'  Spacing       : {self.spacing_m} m\n'
            f'  Output file   : {self.waypoints_file}\n'
            f'  Odom topic    : {self._odom_topic}\n'
            f'  Odom QoS      : '
            f'{"RELIABLE" if self._odom_reliable else "BEST_EFFORT"}\n'
            f'  Nav2 server   : {self._nav_server}\n'
            f'  Goal timeout  : {self.goal_timeout} s\n'
            f'{"="*60}\n'
            f'  Drive the vehicle. Waypoints (x, y, yaw) recorded every\n'
            f'  {self.spacing_m} m via TF (map frame).\n'
            f'  Foxglove / RViz topics:\n'
            f'    /waypoint_markers    – cyan arrows (2 Hz)\n'
            f'    /current_goal_marker – red arrow (active goal)\n'
            f'  Press  S + Enter  to STOP recording & save.\n'
            f'  Press  P + Enter  to start PLAYBACK (Nav2).\n'
            f'  Press  Q + Enter  to quit.\n'
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    @property
    def state(self):
        with self._lock:
            return self._state

    @state.setter
    def state(self, value):
        with self._lock:
            self._state = value

    # ------------------------------------------------------------------
    # Odometry callback  (executor thread, up to 50 Hz on real car)
    # ------------------------------------------------------------------
    def _odom_callback(self, msg: Odometry):
        # TF lookup outside the lock – it can block up to 0.10 s
        x, y, yaw = self._get_map_pose(msg)

        recorded = False
        with self._lock:
            # Always track latest pose so _stop_recording can capture
            # the exact final position regardless of spacing threshold.
            self._current_pose = (x, y, yaw)

            if self._state != 'RECORDING':
                return

            if self._last_wp_pos is None:
                recorded = self._record_waypoint_locked(x, y, yaw)
            else:
                dist = math.hypot(
                    x - self._last_wp_pos[0],
                    y - self._last_wp_pos[1]
                )
                if dist >= self.spacing_m:
                    recorded = self._record_waypoint_locked(x, y, yaw)

        # Publish outside the lock – avoids holding it during I/O
        if recorded:
            self._publish_waypoint_markers()

    # ------------------------------------------------------------------
    # Pose extraction: TF (map frame) with odometry fallback
    # ------------------------------------------------------------------
    def _get_map_pose(self, msg: Odometry):
        """Return (x, y, yaw_rad) in the map frame, or fall back to odom."""
        if self._tf_buffer is not None:
            try:
                t = self._tf_buffer.lookup_transform(
                    'map',
                    msg.child_frame_id if msg.child_frame_id else 'base_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.10)
                )
                tr  = t.transform
                q   = tr.rotation
                yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
                return tr.translation.x, tr.translation.y, yaw
            except Exception:
                pass   # TF not yet available or timed out; fall through

        # Fallback: raw odometry pose (odom frame – survives TF outage)
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])[2]
        return pos.x, pos.y, yaw

    def _record_waypoint_locked(self, x, y, yaw) -> bool:
        """Must be called with self._lock held."""
        self._waypoints.append((x, y, yaw))
        self._last_wp_pos = (x, y)
        count = len(self._waypoints)
        # get_logger is thread-safe in rclpy – safe to call inside lock
        # but kept brief to minimise hold time at 50 Hz.
        self.get_logger().info(
            f'[REC] #{count:4d}  '
            f'x={x:8.3f}  y={y:8.3f}  yaw={math.degrees(yaw):7.2f}°'
        )
        return True

    # ------------------------------------------------------------------
    # Keyboard input
    # ------------------------------------------------------------------
    def _keyboard_loop(self):
        buf = ''
        while rclpy.ok():
            try:
                key, buf = self._read_keyline(buf)
            except (EOFError, OSError, KeyboardInterrupt):
                break

            if not key:
                continue

            if key == 'S':
                self._stop_recording()
            elif key == 'P':
                self._start_playback()
            elif key == 'Q':
                self.get_logger().info('Quit requested.')
                rclpy.try_shutdown()
                break

    def _read_keyline(self, buf: str):
        if msvcrt is not None:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ('\r', '\n'):
                    return buf.strip().upper(), ''
                if ch == '\x03':
                    raise KeyboardInterrupt
                return None, buf + ch
            time.sleep(0.1)
            return None, buf

        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
        if not ready:
            return None, buf
        return sys.stdin.readline().strip().upper(), buf

    # ------------------------------------------------------------------
    # Stop recording & save
    # ------------------------------------------------------------------
    def _stop_recording(self):
        # --- Snapshot under lock, do all I/O outside ---
        with self._lock:
            if self._state not in ('RECORDING', 'IDLE'):
                self.get_logger().warn(
                    f'Cannot stop: state is {self._state}.'
                )
                return
            self._state          = 'IDLE'
            current_pose         = self._current_pose
            waypoints_snapshot   = list(self._waypoints)

        # Capture the exact final pose the operator stopped at.
        # On the real car the spacing threshold often means the last
        # recorded waypoint is up to spacing_m behind the stop point.
        if current_pose is not None:
            x, y, yaw = current_pose
            if not waypoints_snapshot:
                self.get_logger().info(
                    f'[STOP] No prior waypoints; saving current pose '
                    f'x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f}°'
                )
                with self._lock:
                    self._record_waypoint_locked(x, y, yaw)
                    waypoints_snapshot = list(self._waypoints)
            else:
                last_x, last_y, _ = waypoints_snapshot[-1]
                dist = math.hypot(x - last_x, y - last_y)
                self.get_logger().info(
                    f'[STOP] Final pose x={x:.3f} y={y:.3f} '
                    f'yaw={math.degrees(yaw):.1f}°  '
                    f'dist from last WP: {dist:.3f} m'
                )
                if dist > 0.01:
                    with self._lock:
                        self._record_waypoint_locked(x, y, yaw)
                        waypoints_snapshot = list(self._waypoints)
                else:
                    self.get_logger().info(
                        '[STOP] Final pose within 1 cm of last WP – '
                        'not duplicating.'
                    )
        else:
            self.get_logger().warn(
                '[STOP] No odometry received – is the sensor stack running?\n'
                f'       Expected topic: {self._odom_topic}'
            )

        self.get_logger().info(
            f'[STOP] {len(waypoints_snapshot)} waypoints collected.'
        )

        if not waypoints_snapshot:
            self.get_logger().warn('No waypoints to save – file not written.')
            return

        try:
            with open(self.waypoints_file, 'w') as f:
                f.write('# x, y, yaw_rad\n')
                for (wx, wy, wyaw) in waypoints_snapshot:
                    f.write(f'{wx:.6f}, {wy:.6f}, {wyaw:.6f}\n')
            self.get_logger().info(
                f'[SAVE] {len(waypoints_snapshot)} waypoints → '
                f'"{self.waypoints_file}"\n'
                f'       Press  P + Enter  to start Nav2 playback.'
            )
        except OSError as e:
            self.get_logger().error(f'[SAVE] Failed to write file: {e}')

        # Refresh markers so the final waypoint appears immediately
        self._publish_waypoint_markers()

    # ------------------------------------------------------------------
    # Start playback
    # ------------------------------------------------------------------
    def _start_playback(self):
        with self._lock:
            current = self._state

        if current == 'NAVIGATING':
            self.get_logger().warn('Already navigating.')
            return
        if current == 'RECORDING':
            self.get_logger().warn(
                'Still recording – press S + Enter first.'
            )
            return

        if not self._load_waypoints():
            return

        self.state = 'NAVIGATING'
        with self._lock:
            count = len(self._waypoints)

        self.get_logger().info(
            f'[NAV] Starting playback of {count} waypoints …'
        )
        threading.Thread(
            target=self._navigate_waypoints, daemon=True
        ).start()

    def _load_waypoints(self):
        if not os.path.isfile(self.waypoints_file):
            self.get_logger().error(
                f'[LOAD] File not found: "{self.waypoints_file}"'
            )
            return False

        waypoints = []
        try:
            with open(self.waypoints_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) < 2:
                        continue
                    x   = float(parts[0])
                    y   = float(parts[1])
                    yaw = float(parts[2]) if len(parts) >= 3 else 0.0
                    waypoints.append((x, y, yaw))
        except (OSError, ValueError) as e:
            self.get_logger().error(f'[LOAD] Failed to read file: {e}')
            return False

        if not waypoints:
            self.get_logger().error('[LOAD] File is empty.')
            return False

        with self._lock:
            self._waypoints = waypoints

        self.get_logger().info(
            f'[LOAD] {len(waypoints)} waypoints from "{self.waypoints_file}".'
        )
        self._publish_waypoint_markers()
        return True

    # ------------------------------------------------------------------
    # Navigate through all waypoints sequentially  (navigation thread)
    # ------------------------------------------------------------------
    def _navigate_waypoints(self):
        self.get_logger().info(
            f'[NAV] Waiting for Nav2 action server "{self._nav_server}" …'
        )
        if not self._nav_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error(
                f'[NAV] "{self._nav_server}" not available after 30 s.\n'
                f'      Is Nav2 running? Is the server name correct?\n'
                f'      Check with: ros2 action list'
            )
            self.state = 'IDLE'
            return

        self.get_logger().info('[NAV] Nav2 action server connected.')

        with self._lock:
            waypoints = list(self._waypoints)

        total = len(waypoints)
        for idx, (x, y, yaw) in enumerate(waypoints):
            if not rclpy.ok():
                break

            self.get_logger().info(
                f'[NAV] Waypoint {idx+1}/{total}  '
                f'x={x:.3f}  y={y:.3f}  yaw={math.degrees(yaw):.1f}°'
            )

            self._publish_goal_marker(x, y, yaw)

            goal_msg       = NavigateToPose.Goal()
            goal_msg.pose  = self._make_pose_stamped(x, y, yaw)

            send_future = self._nav_client.send_goal_async(goal_msg)
            if not self._await_future(send_future, timeout_sec=10.0):
                self.get_logger().error(
                    f'[NAV] Goal {idx+1} send timed out – aborting.'
                )
                break

            goal_handle = send_future.result()
            if not goal_handle.accepted:
                self.get_logger().error(
                    f'[NAV] Goal {idx+1} rejected by Nav2 – aborting.\n'
                    f'      Is the goal inside the map bounds?\n'
                    f'      Is the costmap initialised?'
                )
                break

            self.get_logger().info(
                f'[NAV] Goal {idx+1} accepted – waiting for result …'
            )

            result_future = goal_handle.get_result_async()
            if not self._await_future(
                result_future, timeout_sec=self.goal_timeout
            ):
                self.get_logger().warn(
                    f'[NAV] Goal {idx+1} timed out after '
                    f'{self.goal_timeout} s – cancelling and continuing.'
                )
                goal_handle.cancel_goal_async()
                continue

            status = result_future.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(
                    f'[NAV] Waypoint {idx+1}/{total} REACHED ✓'
                )
            else:
                self.get_logger().warn(
                    f'[NAV] Waypoint {idx+1}/{total} ended with '
                    f'status={status} – continuing.'
                )

        self._clear_goal_marker()
        self.get_logger().info('[NAV] Playback complete. State → IDLE.')
        self.state = 'IDLE'

    # ------------------------------------------------------------------
    # Thread-safe future wait
    # ------------------------------------------------------------------
    def _await_future(self, future, timeout_sec: float) -> bool:
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        return event.wait(timeout=timeout_sec)

    # ------------------------------------------------------------------
    # PoseStamped helper
    # ------------------------------------------------------------------
    def _make_pose_stamped(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    # ------------------------------------------------------------------
    # Waypoint markers – published on-change + 2 Hz timer for Foxglove
    # ------------------------------------------------------------------
    def _publish_waypoint_markers(self):
        with self._lock:
            waypoints = list(self._waypoints)

        arr = MarkerArray()

        # DELETEALL clears stale arrows from previous sessions
        d = Marker()
        d.header.frame_id = 'map'
        d.header.stamp    = self.get_clock().now().to_msg()
        d.ns              = 'waypoints'
        d.action          = Marker.DELETEALL
        arr.markers.append(d)

        for i, (x, y, yaw) in enumerate(waypoints):
            m = Marker()
            m.header.frame_id    = 'map'
            m.header.stamp       = self.get_clock().now().to_msg()
            m.ns                 = 'waypoints'
            m.id                 = i
            m.type               = Marker.ARROW
            m.action             = Marker.ADD
            m.pose.position.x    = x
            m.pose.position.y    = y
            m.pose.position.z    = 0.0
            qx, qy, qz, qw       = quaternion_from_euler(0.0, 0.0, yaw)
            m.pose.orientation.x = qx
            m.pose.orientation.y = qy
            m.pose.orientation.z = qz
            m.pose.orientation.w = qw
            m.scale.x            = 1.0
            m.scale.y            = 0.2
            m.scale.z            = 0.2
            m.color.r            = 0.0
            m.color.g            = 0.7
            m.color.b            = 1.0
            m.color.a            = 1.0
            # lifetime=0 persists until DELETEALL replaces it
            m.lifetime           = rclpy.duration.Duration(seconds=0).to_msg()
            arr.markers.append(m)

        self._wp_marker_pub.publish(arr)

    def _republish_markers(self):
        """2 Hz timer callback – keeps markers alive for late Foxglove connects."""
        self._publish_waypoint_markers()

    # ------------------------------------------------------------------
    # Goal marker
    # ------------------------------------------------------------------
    def _publish_goal_marker(self, x: float, y: float, yaw: float):
        m = Marker()
        m.header.frame_id    = 'map'
        m.header.stamp       = self.get_clock().now().to_msg()
        m.ns                 = 'current_goal'
        m.id                 = 0
        m.type               = Marker.ARROW
        m.action             = Marker.ADD
        m.pose.position.x    = x
        m.pose.position.y    = y
        m.pose.position.z    = 0.2   # slightly elevated so it doesn't clip
        qx, qy, qz, qw       = quaternion_from_euler(0.0, 0.0, yaw)
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw
        m.scale.x            = 1.5
        m.scale.y            = 0.3
        m.scale.z            = 0.3
        m.color.r            = 1.0
        m.color.g            = 0.3
        m.color.b            = 0.0
        m.color.a            = 1.0
        # 5 s safety net: clears automatically if node crashes mid-run
        m.lifetime           = rclpy.duration.Duration(seconds=5).to_msg()
        self._goal_marker_pub.publish(m)

    def _clear_goal_marker(self):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns              = 'current_goal'
        m.id              = 0
        m.action          = Marker.DELETE
        self._goal_marker_pub.publish(m)


# ======================================================================
def main(args=None):
    rclpy.init(args=args)
    node = WaypointRecorderNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()   # safe against double-shutdown on e-stop


if __name__ == '__main__':
    main()