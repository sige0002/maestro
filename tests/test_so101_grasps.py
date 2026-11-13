import unittest
import numpy as np
from maestro.so101.grasps import geometry_candidates


class GraspGeometryTests(unittest.TestCase):
    def test_long_object_closes_across_short_axis_and_clears_table(self):
        x,y=np.meshgrid(np.linspace(-.025,.025,30),np.linspace(-.012,.012,20))
        points=np.c_[x.ravel(),y.ravel(),np.full(x.size,.024)]
        obj={'points':points,'center':np.array([0,0,.012]),'top':np.array([0,0,.024])}
        candidates=geometry_candidates(obj,0)
        p,o=candidates[0]
        self.assertAlmostEqual(abs(o[2]),np.pi/2)
        self.assertGreaterEqual(p[2],.014)
        self.assertLessEqual(p[2],.021)
        # Rotating the observation rotates the grasp, not an object-name branch.
        theta=.4
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        obj['points'][:,:2]=points[:,:2]@rotation.T
        _,rotated=geometry_candidates(obj,0)[0]
        self.assertAlmostEqual(abs(np.cos(rotated[2]-o[2]-theta)),1)

    def test_too_thin_object_is_rejected_instead_of_grasping_the_floor(self):
        obj={'points':np.array([[0,0,.008],[.02,0,.008],[0,.02,.008]]),
             'center':np.array([0,0,.004]),'top':np.array([0,0,.008])}
        self.assertEqual(geometry_candidates(obj,0),[])
