import io
import unittest
from unittest.mock import patch

import numpy as np

from maestro.perception_utils import segment


class SAM3ClientTests(unittest.TestCase):
    @patch('maestro.perception_utils.service_url', return_value='http://localhost:8008')
    @patch('maestro.perception_utils.requests.post')
    def test_sam3_endpoint_and_model_provenance(self, post, url):
        data=io.BytesIO()
        np.savez(data, masks=np.ones((1,8,8),dtype=bool), boxes=np.array([[1,2,6,7]]),
                 scores=np.array([.9]), labels=np.array(['blue bar']),
                 backend='SAM3.1',model='facebook/sam3.1',revision='pinned')
        post.return_value.content=data.getvalue()
        result=segment(np.zeros((8,8,3),dtype=np.uint8),'blue bar',service='sam3')
        url.assert_called_once_with('sam3')
        self.assertEqual(post.call_args.args[0],'http://localhost:8008/segment')
        self.assertEqual(str(result['backend']),'SAM3.1')
        self.assertEqual(str(result['revision']),'pinned')
        np.testing.assert_array_equal(result['boxes'],[[1,2,6,7]])

    @patch('maestro.perception_utils.service_url', return_value='http://localhost:8008')
    @patch('maestro.perception_utils.requests.post')
    def test_old_backend_cannot_silently_impersonate_sam3(self, post, url):
        data=io.BytesIO()
        np.savez(data,masks=np.zeros((0,8,8),dtype=bool),boxes=np.zeros((0,4)),
                 scores=np.zeros(0),labels=np.array([],dtype=str))
        post.return_value.content=data.getvalue()
        with self.assertRaisesRegex(RuntimeError,'SAM3.1'):
            segment(np.zeros((8,8,3),dtype=np.uint8),'blue bar',service='sam3')
