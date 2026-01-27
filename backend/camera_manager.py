import os
import threading
import time

import cv2


class CameraStream:
    def __init__(self, cam_id: int, index: int) -> None:
        self.cam_id = cam_id
        self.index = index
        self.capture = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self) -> bool:
        if self.running:
            return True

        backend = cv2.CAP_DSHOW if os.name == "nt" else 0
        self.capture = cv2.VideoCapture(self.index, backend)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            return False

        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.capture.release()
            self.capture = None
            return False

        self.frame = frame
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return True

    def _reader(self) -> None:
        while self.running:
            ok, frame = self.capture.read()
            if ok and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.02)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self) -> None:
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class CameraManager:
    def __init__(self, cam_ids=(1, 2, 3)) -> None:
        self.cam_ids = cam_ids
        self.cams = {}
        self.lock = threading.Lock()

    def _start_camera(self, cam_id: int) -> None:
        if cam_id in self.cams:
            return

        index = cam_id - 1
        stream = CameraStream(cam_id, index)
        if stream.start():
            self.cams[cam_id] = stream

    def ensure_started(self) -> None:
        with self.lock:
            for cam_id in self.cam_ids:
                self._start_camera(cam_id)

    def get_available(self):
        self.ensure_started()
        return sorted(self.cams.keys())

    def get_stream(self, cam_id: int):
        self.ensure_started()
        return self.cams.get(cam_id)

    def get_frame(self, cam_id: int):
        stream = self.get_stream(cam_id)
        return stream.get_frame() if stream else None

    def stop_all(self) -> None:
        with self.lock:
            for stream in self.cams.values():
                stream.stop()
            self.cams.clear()
