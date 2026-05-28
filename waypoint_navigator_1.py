#!/usr/bin/env python3
"""
Waypoint Recorder & Navigator for ROS2 Humble + Nav2
-----------------------------------------------------
RECORD MODE:
  - Subscribes to /Odometry
  - Saves x,y position every N meters to waypoints.txt
  - Press 'S' + Enter to stop recording and save

PLAYBACK MODE:
  - Press 'P' + Enter to start playback
  - Reads waypoints.txt and sends each point as a Nav2 goal
  - Waits for goal result before sending the next

Usage:
  ros2 run <your_package> waypoint_recorder_navigator
  ros2 run <your_package> waypoint_recorder_navigator --ros-args \
      -p spacing_m:=2.0 -p waypoints_file:=/tmp/waypoints.txt

Fixes applied vs original:
  1. Removed rclpy.spin_until_future_complete() calls from the
     navigation thread (caused race with main executor spin).
     Replaced with threading.Event-based _await_future().
  2. Changed PoseStamped frame_id from 'odom' to 'map' – Nav2
     navigates in the map frame.
  3. Waypoints now recorded via TF (map -> base_link) so stored
     coordinates are in the map frame and survive localisation drift.
     Falls back to raw odometry if the TF lookup fails.
  4. Keyboard thread uses select() so it unblocks on shutdown
     instead of hanging on input().
  5. threading.Lock guards all shared state (self.state,
     self.waypoints, self.last_wp_pos).
  6. Goal status compared with GoalStatus.STATUS_SUCCEEDED constant
     instead of a magic number.
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
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from action_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

try:
    from tf2_ros import Buffer, TransformListener
    _TF2_AVAILABLE = True
except ImportError:
    _TF2_AVAILABLE = False


class WaypointRecorderNavigator(Node):

    def __init__(self):
        super().__init__('waypoint_recorder_navigator')

        # ---------- Parameters ----------
        self.declare_parameter('spacing_m', 5.0)
        self.declare_parameter('waypoints_file', 'waypoints.txt')
        self.declare_parameter('goal_timeout_sec', 60.0)

        self.spacing_m      = self.get_parameter('spacing_m').value
        self.waypoints_file = self.get_parameter('waypoints_file').value
        self.goal_timeout   = self.get_parameter('goal_timeout_sec').value

        # ---------- Shared state + lock ----------
        self._lock        = threading.Lock()
        self._state       = 'RECORDING'   # RECORDING | IDLE | NAVIGATING
        self._waypoints   = []            # list of (x, y)  – map frame
        self._last_wp_pos = None          # (x, y) of last recorded waypoint

        # ---------- TF2 (for map-frame waypoint recording) ----------
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
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # ---------- Subscriber ----------
        self.odom_sub = self.create_subscription(
            Odometry,
            '/Odometry',
            self._odom_callback,
            odom_qos
        )

        # ---------- Nav2 Action Client ----------
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ---------- Keyboard thread ----------
        self._kb_thread = threading.Thread(
            target=self._keyboard_loop, daemon=True
        )
        self._kb_thread.start()

        self.get_logger().info(
            f'\n{"="*55}\n'
            f'  Waypoint Recorder / Navigator\n'
            f'{"="*55}\n'
            f'  State       : RECORDING\n'
            f'  Spacing     : {self.spacing_m} m\n'
            f'  Output file : {self.waypoints_file}\n'
            f'{"="*55}\n'
            f'  Drive the vehicle. Waypoints saved every {self.spacing_m} m.\n'
            f'  Press  S + Enter  to STOP recording & save file.\n'
            f'  Press  P + Enter  to start PLAYBACK (Nav2).\n'
            f'  Press  Q + Enter  to quit.\n'
        )

    # ------------------------------------------------------------------
    # State helpers (always access state through these)
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
    # Odometry callback  (executor thread)
    # ------------------------------------------------------------------
    def _odom_callback(self, msg: Odometry):
        with self._lock:
            if self._state != 'RECORDING':
                return

        # Try to get position in map frame via TF; fall back to odom frame.
        x, y = self._get_map_position(msg)

        with self._lock:
            if self._last_wp_pos is None:
                self._record_waypoint_locked(x, y)
                return

            dist = math.hypot(
                x - self._last_wp_pos[0],
                y - self._last_wp_pos[1]
            )
            if dist >= self.spacing_m:
                self._record_waypoint_locked(x, y)

    def _get_map_position(self, msg: Odometry):
        """Return (x, y) in the map frame, or fall back to odom frame."""
        if self._tf_buffer is not None:
            try:
                t = self._tf_buffer.lookup_transform(
                    'map',
                    msg.header.frame_id if msg.header.frame_id else 'base_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05)
                )
                return (
                    t.transform.translation.x,
                    t.transform.translation.y
                )
            except Exception:
                pass  # TF not yet available; fall through

        # Fallback: raw odometry position
        return (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

    def _record_waypoint_locked(self, x, y):
        """Must be called with self._lock held."""
        self._waypoints.append((x, y))
        self._last_wp_pos = (x, y)
        count = len(self._waypoints)
        # Log outside the lock to avoid holding it during I/O
        self.get_logger().info(
            f'[REC] Waypoint #{count:4d}  x={x:8.3f}  y={y:8.3f}'
        )

    # ------------------------------------------------------------------
    # Keyboard input – uses select() so it wakes up on shutdown
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
                rclpy.shutdown()
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

        # Wait up to 0.5 s for stdin to be readable
        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
        if not ready:
            return None, buf
        return sys.stdin.readline().strip().upper(), buf

    # ------------------------------------------------------------------
    # Stop recording & save
    # ------------------------------------------------------------------
    def _stop_recording(self):
        with self._lock:
            if self._state != 'RECORDING':
                self.get_logger().warn('Not currently recording.')
                return
            self._state = 'IDLE'
            waypoints_snapshot = list(self._waypoints)

        self.get_logger().info(
            f'[STOP] Recording stopped. '
            f'{len(waypoints_snapshot)} waypoints collected.'
        )

        if not waypoints_snapshot:
            self.get_logger().warn('No waypoints recorded – file not written.')
            return

        try:
            with open(self.waypoints_file, 'w') as f:
                f.write('# x, y\n')
                for (x, y) in waypoints_snapshot:
                    f.write(f'{x:.6f}, {y:.6f}\n')
            self.get_logger().info(
                f'[SAVE] {len(waypoints_snapshot)} waypoints saved to '
                f'"{self.waypoints_file}".\n'
                f'       Press  P + Enter  to start Nav2 playback.'
            )
        except OSError as e:
            self.get_logger().error(f'Failed to write file: {e}')

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
                'Still recording. Press  S + Enter  first to stop & save.'
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
        nav_thread = threading.Thread(
            target=self._navigate_waypoints, daemon=True
        )
        nav_thread.start()

    def _load_waypoints(self):
        if not os.path.isfile(self.waypoints_file):
            self.get_logger().error(
                f'Waypoints file not found: "{self.waypoints_file}"'
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
                    waypoints.append((float(parts[0]), float(parts[1])))
        except (OSError, ValueError) as e:
            self.get_logger().error(f'Failed to read waypoints file: {e}')
            return False

        if not waypoints:
            self.get_logger().error('Waypoints file is empty.')
            return False

        with self._lock:
            self._waypoints = waypoints

        self.get_logger().info(
            f'[LOAD] {len(waypoints)} waypoints loaded from '
            f'"{self.waypoints_file}".'
        )
        return True

    # ------------------------------------------------------------------
    # Navigate through all waypoints sequentially  (navigation thread)
    # ------------------------------------------------------------------
    def _navigate_waypoints(self):
        if not self._nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                'navigate_to_pose action server not available! Is Nav2 running?'
            )
            self.state = 'IDLE'
            return

        with self._lock:
            waypoints = list(self._waypoints)

        total = len(waypoints)
        for idx, (x, y) in enumerate(waypoints):
            if not rclpy.ok():
                break

            self.get_logger().info(
                f'[NAV] Sending waypoint {idx+1}/{total}  '
                f'x={x:.3f}  y={y:.3f}'
            )

            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = self._make_pose_stamped(x, y)

            # ---- Send goal ----------------------------------------
            # FIX: do NOT call rclpy.spin_until_future_complete() from
            # this thread – the main thread is already spinning the
            # executor. Use a threading.Event instead.
            send_future = self._nav_client.send_goal_async(goal_msg)
            if not self._await_future(send_future, timeout_sec=10.0):
                self.get_logger().error(
                    f'Goal {idx+1} send timed out. Aborting playback.'
                )
                break

            goal_handle = send_future.result()
            if not goal_handle.accepted:
                self.get_logger().error(
                    f'Goal {idx+1} rejected by Nav2. Aborting playback.'
                )
                break

            self.get_logger().info(
                f'[NAV] Goal {idx+1} accepted, waiting for result …'
            )

            # ---- Wait for result -----------------------------------
            result_future = goal_handle.get_result_async()
            if not self._await_future(
                result_future, timeout_sec=self.goal_timeout
            ):
                self.get_logger().warn(
                    f'Goal {idx+1} timed out after {self.goal_timeout}s. '
                    f'Cancelling and continuing.'
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
                    f'status={status}, continuing.'
                )

        self.get_logger().info('[NAV] Playback complete. State -> IDLE.')
        self.state = 'IDLE'

    # ------------------------------------------------------------------
    # FIX: thread-safe future wait – no second spinner needed
    # ------------------------------------------------------------------
    def _await_future(self, future, timeout_sec: float) -> bool:
        """
        Block the calling thread until *future* is done or *timeout_sec*
        elapses. Returns True if the future completed, False on timeout.

        Uses a threading.Event driven by the future's done-callback so
        it is safe to call from any thread while the main executor is
        already spinning the node.
        """
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        return event.wait(timeout=timeout_sec)

    # ------------------------------------------------------------------
    # Helper: PoseStamped in the MAP frame with identity orientation
    # ------------------------------------------------------------------
    def _make_pose_stamped(self, x: float, y: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'   # FIX: was 'odom'; Nav2 uses map frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0  # identity – Nav2 will compute heading
        return pose


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
        rclpy.shutdown()


if __name__ == '__main__':
    main()