import unittest
import numpy as np
from maestro.so101.rgbd import point_cloud, lift_pixel, select_pointed_mask
from maestro.geometry import GeometryTools
from scipy.spatial.transform import Rotation


class DepthTests(unittest.TestCase):
    def test_point_disambiguates_higher_scoring_distractor(self):
        masks=np.zeros((2,20,30),dtype=bool)
        masks[0,2:10,2:10]=True
        masks[1,2:10,20:28]=True
        detections={'masks':masks,'scores':np.array([.9,.6])}
        self.assertEqual(select_pointed_mask(detections,[24,6]),1)
        with self.assertRaisesRegex(ValueError,'No GroundedSAM mask'):
            select_pointed_mask(detections,[15,6])

    def test_object_inside_container_uses_specific_mask(self):
        masks=np.ones((2,20,30),dtype=bool)
        masks[1]=False
        masks[1,5:10,10:20]=True
        detections={'masks':masks,'scores':np.array([.9,.6])}
        self.assertEqual(select_pointed_mask(detections,[15,7]),1)

    def test_depth_is_camera_z_not_ray_length(self):
        k=np.array([[100,0,20],[0,100,20],[0,0,1]])
        depth=np.ones((40,40),dtype=float)*2
        camera=np.eye(4);camera[:3,3]=[1,2,3]
        np.testing.assert_allclose(lift_pixel([30,10],depth,k,camera),[1.2,1.8,5])
        np.testing.assert_allclose(point_cloud(depth,k,camera)[10,30],[1.2,1.8,5])

    def test_invalid_depth_fails_instead_of_guessing_height(self):
        with self.assertRaisesRegex(ValueError,'No valid depth'):
            lift_pixel([1,1],np.full((4,4),np.nan),np.eye(3),np.eye(4))

    def test_antiparallel_vector_rotation_is_euler_not_rotvec(self):
        class Tools(GeometryTools):
            def _log_timestamp(self,*args):pass
        a=np.array([1,2,3],dtype=float);a/=np.linalg.norm(a)
        euler=Tools().measure_relative_rotation(a,-a)
        np.testing.assert_allclose(Rotation.from_euler('xyz',euler).apply(a),-a,atol=1e-12)
