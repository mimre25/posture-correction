from abc import ABC, abstractmethod
from typing import NamedTuple
import cv2
from playsound import playsound
import time
import math as m
import mediapipe as mp
import numpy as np
from desktop_notifier import DesktopNotifier, Urgency
from dataclasses import dataclass
import argparse


Color = tuple[int, int, int]

font = cv2.FONT_HERSHEY_SIMPLEX

red = (50, 50, 255)
green = (127, 255, 0)
yellow = (0, 255, 255)
yellow2 = (0, 128, 128)
pink = (255, 0, 255)
pink2 = (128, 0, 128)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


@dataclass(slots=True)
class Point2D:
    x: int
    y: int

    def __init__(self, x: int | float, y: int | float) -> None:
        self.x = int(x)
        self.y = int(y)


class Renderer(ABC):
    @abstractmethod
    def render(
        self,
        frame: np.ndarray,
        l_shoulder: Point2D,
        r_shoulder: Point2D,
        l_ear: Point2D,
        r_ear: Point2D,
        neck_inclination: float,
        neck_inclination2: float,
        posture_time: float,
        running_clock: float,
        actual_fps: float,
        height: int,
        *,
        is_good_posture: bool,
    ) -> None:
        ...


class NoopRenderer(Renderer):
    def render(
        self,
        frame: np.ndarray,
        l_shoulder: Point2D,
        r_shoulder: Point2D,
        l_ear: Point2D,
        r_ear: Point2D,
        neck_inclination: float,
        neck_inclination2: float,
        posture_time: float,
        running_clock: float,
        actual_fps: float,
        height: int,
        *,
        is_good_posture: bool,
    ) -> None:
        pass


class DebugRenderer(Renderer):
    def render(
        self,
        frame: np.ndarray,
        l_shoulder: Point2D,
        r_shoulder: Point2D,
        l_ear: Point2D,
        r_ear: Point2D,
        neck_inclination: float,
        neck_inclination2: float,
        posture_time: float,
        running_clock: float,
        actual_fps: float,
        height: int,
        *,
        is_good_posture: bool,
    ) -> None:
        color = green if is_good_posture else red

        cv2.circle(frame, (l_shoulder.x, l_shoulder.y), 7, yellow, -1)
        cv2.circle(frame, (l_ear.x, l_ear.y), 7, yellow2, -1)

        cv2.circle(frame, (r_shoulder.x, r_shoulder.y), 7, pink, -1)
        cv2.circle(frame, (r_ear.x, r_ear.y), 7, pink2, -1)

        cv2.putText(
            frame,
            str(int(neck_inclination)),
            (r_shoulder.x + 10, r_shoulder.y),
            font,
            0.9,
            color,
            2,
        )

        cv2.line(frame, (r_shoulder.x, r_shoulder.y), (r_ear.x, r_ear.y), color, 4)
        cv2.line(
            frame,
            (r_shoulder.x, r_shoulder.y),
            (r_shoulder.x, r_shoulder.y - 100),
            color,
            4,
        )

        cv2.putText(
            frame,
            str(int(neck_inclination2)),
            (r_shoulder.x + 10, r_shoulder.y + 30),
            font,
            0.9,
            color,
            2,
        )

        cv2.line(frame, (r_shoulder.x, r_shoulder.y), (l_ear.x, l_ear.y), color, 4)
        time_str = f"Posture Time: {round(posture_time, 1)} s / {running_clock} ({actual_fps} FPS)"
        cv2.putText(frame, time_str, (10, height - 20), font, 0.9, color, 2)
        cv2.imshow("foobar", frame)


def find_angle(p1: Point2D, p2: Point2D) -> float:
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


def sendWarning():
    playsound("alarm.wav")
    print("warning")
    pass
    # notifier = DesktopNotifier()
    # notifier.send_sync(title="Sit up", message="Bad Posture Detected", urgency=Urgency.Critical)


def extract_keypoints(frame: np.ndarray) -> NamedTuple:
    # Convert the BGR image to RGB.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image. pose stimation needs RGB image
    keypoints = pose.process(frame)

    # Convert the image back to BGR.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return keypoints


def find_landmarks(
    frame: np.ndarray, height: int, width: int
) -> tuple[Point2D, Point2D, Point2D, Point2D]:

    keypoints = extract_keypoints(frame)
    # Use lm and lmPose as representative of the following methods.
    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark

    left_shoulder = Point2D(
        lm.landmark[lmPose.LEFT_SHOULDER].x * width,
        lm.landmark[lmPose.LEFT_SHOULDER].y * height,
    )
    right_shoulder = Point2D(
        lm.landmark[lmPose.RIGHT_SHOULDER].x * width,
        lm.landmark[lmPose.RIGHT_SHOULDER].y * height,
    )

    left_ear = Point2D(
        lm.landmark[lmPose.LEFT_EAR].x * width, lm.landmark[lmPose.LEFT_EAR].y * height
    )
    right_ear = Point2D(
        lm.landmark[lmPose.RIGHT_EAR].x * width,
        lm.landmark[lmPose.RIGHT_EAR].y * height,
    )

    return left_shoulder, right_shoulder, left_ear, right_ear


def main(*, debug: bool):
    file_name = 0  # 0 = webcam
    vid_capture = cv2.VideoCapture(file_name)

    running_clock = 0
    start_time = time.time()
    cur_time = time.time()
    good_time = 0
    bad_time = 0
    warning_send_time = time.time()

    renderer = DebugRenderer() if debug else NoopRenderer()
    while vid_capture.isOpened():

        frame_diff_s = time.time() - cur_time
        actual_fps = round(1 / frame_diff_s, 2)

        cur_time = time.time()
        running_clock = round(cur_time - start_time, 1)

        success, frame = vid_capture.read()
        if not success:
            raise Exception(f"Failed to read frame {success=}, {frame=}")

        # Get height and width.
        h, w = frame.shape[:2]
        try:
            l_shoulder, r_shoulder, l_ear, r_ear = find_landmarks(frame, h, w)
        except Exception:
            # can't find landmarks, just try in a second again
            time.sleep(1)
            continue

        neck_inclination = find_angle(r_shoulder, r_ear)
        neck_inclination2 = find_angle(r_shoulder, l_ear)

        is_good_posture = neck_inclination < 35 or neck_inclination2 < 50

        if is_good_posture:
            bad_time = 0
            good_time += frame_diff_s
            bad_time -= frame_diff_s
            bad_time = max(0, bad_time)
        else:
            good_time = 0
            bad_time += frame_diff_s

        posture_time = max(good_time, bad_time)

        if bad_time > 15 and (time.time() - warning_send_time) > 60:
            sendWarning()
            warning_send_time = time.time()

        renderer.render(
            frame,
            l_shoulder,
            r_shoulder,
            l_ear,
            r_ear,
            neck_inclination,
            neck_inclination2,
            posture_time,
            running_clock,
            actual_fps,
            h,
            is_good_posture=is_good_posture,
        )
        # quit if q is hit
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    vid_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Posture Correction',
                    description='Plays an alert when bad posture is detected',
                    )
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    main(debug=args.debug)
