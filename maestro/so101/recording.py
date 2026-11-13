"""Record live ROS camera snapshots at wall-clock speed, including VLM waits."""
import threading
import time
import cv2
import numpy as np


class TrialVideo:
    def __init__(self, transport, path, fps=10):
        self.transport, self.path, self.fps = transport, path, fps
        self.stop_event = threading.Event()
        self.phase = 'Observing / planning'
        self.frames = 0
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (960, 520))
        if not writer.isOpened():
            self.error = 'VideoWriter failed to open'
            return
        try:
            while not self.stop_event.is_set():
                before = time.monotonic()
                canvas = np.zeros((520, 960, 3), dtype=np.uint8)
                try:
                    obs = self.transport.snapshot()
                    canvas[:480, :640] = cv2.cvtColor(cv2.resize(obs['overview'], (640, 480)), cv2.COLOR_RGB2BGR)
                    canvas[:240, 640:] = cv2.cvtColor(cv2.resize(obs['wrist'], (320, 240)), cv2.COLOR_RGB2BGR)
                    for i, q in enumerate(obs['joints']):
                        cv2.putText(canvas, f'q{i+1}: {q:+.3f} rad', (655, 270+i*28),
                                    cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 230, 230), 1)
                except RuntimeError:
                    cv2.putText(canvas, 'Waiting for fresh ROS observations', (20, 220),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                elapsed = time.monotonic()-self.started
                cv2.putText(canvas, f'Maestro + Qwen | SO101 MuJoCo | {elapsed:.1f}s', (10, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1)
                cv2.putText(canvas, self.phase[:75], (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (0, 255, 255), 2)
                writer.write(canvas)
                self.frames += 1
                self.stop_event.wait(max(0, 1/self.fps-(time.monotonic()-before)))
        finally:
            writer.release()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        if getattr(self, 'error', None):
            raise RuntimeError(self.error)
