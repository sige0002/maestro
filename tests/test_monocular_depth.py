import io
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests
from PIL import Image

from maestro.so101.monocular_depth import infer_depth


class MonocularDepthTests(unittest.TestCase):
    def response(self, depth):
        data = io.BytesIO()
        np.savez(
            data,
            depth=depth,
            model="test-model",
            revision="test-revision",
            elapsed_s=0.1,
        )
        return Mock(content=data.getvalue())

    def test_only_rgb_is_sent_and_metric_values_are_preserved(self):
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        with patch(
            "maestro.so101.monocular_depth.requests.post",
            return_value=self.response(np.full((4, 6), 2.0)),
        ) as post:
            depth, meta = infer_depth(rgb, {"metric_scale": 0.3})
        np.testing.assert_allclose(depth, 0.6)
        self.assertEqual(meta["revision"], "test-revision")
        self.assertEqual(set(post.call_args.kwargs), {"files", "timeout"})
        uploaded = post.call_args.kwargs["files"]["image"][1]
        np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(uploaded))), rgb)

    def test_unavailable_service_propagates_without_fallback(self):
        with (
            patch(
                "maestro.so101.monocular_depth.requests.post",
                side_effect=requests.ConnectionError,
            ),
            self.assertRaises(requests.ConnectionError),
        ):
            infer_depth(np.zeros((4, 6, 3), dtype=np.uint8), {})

    def test_invalid_depth_rejected(self):
        for depth in [np.zeros((4, 6)), np.full((4, 6), np.nan), np.ones((2, 3))]:
            with (
                self.subTest(shape=depth.shape),
                patch(
                    "maestro.so101.monocular_depth.requests.post",
                    return_value=self.response(depth),
                ),
                self.assertRaises(ValueError),
            ):
                infer_depth(np.zeros((4, 6, 3), dtype=np.uint8), {})

    def test_rgb_transport_does_not_read_available_sensor_depth(self):
        from maestro.so101.transport import RosTransport

        transport = RosTransport.__new__(RosTransport)
        transport.depth_backend = "depth_anything"
        transport.config = {"max_age_s": 2}
        transport.lock = threading.Lock()
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        stamp = object()
        transport.latest = {
            "overview": (time.monotonic(), rgb),
            "overview_stamp": (time.monotonic(), stamp),
            "overview_depth": (time.monotonic(), np.full((4, 6), 999.0)),
        }
        expected = np.ones((4, 6))
        with patch(
            "maestro.so101.monocular_depth.infer_depth",
            return_value=(expected, {"source": "test"}),
        ):
            _, depth, actual_stamp = transport.rgbd()
        np.testing.assert_array_equal(depth, expected)
        self.assertIs(actual_stamp, stamp)
