import unittest

import numpy as np

from maestro.so101.rgbd import object_from_mask, objects_from_masks


class MaskGeometryTests(unittest.TestCase):
    def test_masks_separate_objects_when_depth_has_no_background_gap(self):
        yy, xx = np.indices((40, 60))
        cloud = np.stack([xx*.002, yy*.002, np.full(xx.shape, .025)], axis=-1)
        masks = np.zeros((4, 40, 60), dtype=bool)
        masks[0, 5:20, 5:20] = True
        masks[1] = masks[0]  # duplicate detector label
        masks[2, 5:20, 30:45] = True
        masks[3] = True  # table mask exceeds compact object extent
        detections = {'masks': masks, 'scores': np.array([.8, .7, .6, .9])}
        objects = objects_from_masks(detections, cloud, 0.)
        self.assertEqual(len(objects), 2)
        self.assertEqual([o['detection_index'] for o in objects], [0, 2])
        self.assertLess(objects[0]['center'][0], objects[1]['center'][0])

    def test_below_plane_and_invalid_points_rejected(self):
        mask = np.ones((8, 8), dtype=bool)
        for height in (-.01, np.nan, .2):
            cloud = np.zeros((8, 8, 3)); cloud[:, :, 2] = height
            with self.subTest(height=height), self.assertRaises(ValueError):
                object_from_mask(mask, cloud, 0.)
