"""Local VLM wire contract; live robot trials are recorded separately."""
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from pydantic import BaseModel, ValidationError
from maestro import vlm


class Answer(BaseModel):
    text: str


class LocalVLMTests(unittest.TestCase):
    def response(self, content, finish='stop'):
        result = Mock()
        result.json.return_value = {'choices': [{'finish_reason': finish,
                                                'message': {'content': content}}]}
        return result

    @patch.dict(vlm.SETTINGS, {'base_url': 'http://localhost:18082/v1', 'model': 'test'})
    def test_images_are_sent_but_not_accumulated_in_history(self):
        chat = vlm._LocalChat()
        with patch('requests.post', return_value=self.response('{"text":"seen"}')) as post:
            parsed, _ = chat.send(['describe', Image.new('RGB', (8, 8))], Answer)
            self.assertEqual(parsed.text, 'seen')
            content = post.call_args.kwargs['json']['messages'][-1]['content']
            self.assertTrue(content[1]['image_url']['url'].startswith('data:image/png;base64,'))
            self.assertEqual(chat._history[0]['content'], [{'type': 'text', 'text': 'describe'}])

    @patch.dict(vlm.SETTINGS, {'base_url': 'http://localhost:18082/v1', 'model': 'test'})
    def test_partial_or_invalid_plans_never_enter_history(self):
        chat = vlm._LocalChat()
        for response, error in [(self.response('{"text":"partial"}', 'length'), RuntimeError),
                                (self.response('{"wrong":123}'), ValidationError)]:
            with patch('requests.post', return_value=response), self.assertRaises(error):
                chat.send(['instruction'], Answer)
            self.assertEqual(chat._history, [])
