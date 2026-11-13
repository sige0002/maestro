"""Official SAM 3.1 text/point/box segmentation; no GroundingDINO or SAM 1."""
import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from sam3.model_builder import build_sam3_predictor

MODEL = 'facebook/sam3.1'
REVISION = 'daa63191845a41281374e725f4c9e51c7a824460'
SOURCE_REVISION = 'f92a5efc4001ebfe9d050d43e61f6fb10e02d73c'
checkpoint = os.environ['SAM3_CHECKPOINT']
app = FastAPI()
lock = threading.Lock()
predictor = None


@app.on_event('startup')
def startup():
    global predictor
    predictor = build_sam3_predictor(checkpoint_path=checkpoint, version='sam3.1',
        compile=False, warm_up=False, use_fa3=False, use_rope_real=False, async_loading_frames=False)
    # The upstream loader allows missing keys; refuse an incomplete model here.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    state = state.get('model', state)
    missing = set(predictor.model.state_dict()) - set(state)
    if missing:
        raise RuntimeError(f'Incomplete SAM 3.1 checkpoint: {len(missing)} missing keys')


@app.get('/health')
def health():
    return {'status': 'ready' if predictor is not None else 'loading', 'service': 'SAM3.1',
            'model': MODEL, 'revision': REVISION, 'source_revision': SOURCE_REVISION,
            'device': 'cuda', 'prompts': ['text', 'points', 'boxes'], 'groundingdino': False}


@app.post('/segment')
def segment(image: UploadFile, prompt: str = Form(''), points: str = Form(''),
            point_labels: str = Form(''), boxes: str = Form('')):
    rgb = Image.open(io.BytesIO(image.file.read())).convert('RGB')
    width, height = rgb.size
    start = time.monotonic()
    try:
        visual = {}
        if points:
            visual.update(points=json.loads(points), point_labels=json.loads(point_labels), obj_id=1)
        if boxes:
            values = json.loads(boxes)
            visual.update(bounding_boxes=values, bounding_box_labels=[1]*len(values))
        concepts = [p.strip() for p in prompt.split('.') if p.strip()]
        if points and (concepts or boxes):
            raise ValueError('Point prompts must be separate from text/box prompts')
        if not concepts and not visual:
            raise ValueError('A text, point or box prompt is required')
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    results = []
    with lock, torch.inference_mode(), tempfile.TemporaryDirectory(prefix='maestro-sam3-') as directory:
        path = Path(directory)/'image.png'
        rgb.save(path)
        for concept in concepts or ['visual']:
            # Call the official model API directly: this pinned release's generic
            # session helper passes offload_state_to_cpu, which multiplex rejects.
            state = predictor.model.init_state(resource_path=str(path), async_loading_frames=False)
            try:
                request = {'inference_state': state, 'frame_idx': 0, 'output_prob_thresh': .5}
                if concept != 'visual':
                    request['text_str'] = concept
                if points:
                    request.update(points=torch.tensor(visual['points'], dtype=torch.float32),
                                   point_labels=torch.tensor(visual['point_labels'], dtype=torch.int32),
                                   obj_id=1, rel_coordinates=True)
                if boxes:
                    request.update(boxes_xywh=torch.tensor(visual['bounding_boxes'], dtype=torch.float32),
                                   box_labels=torch.tensor(visual['bounding_box_labels'], dtype=torch.int32))
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    if points:
                        # Initialize detector features and select the instance to refine
                        # using the SAME SAM 3.1 model, then apply positive/negative points.
                        _, seed = predictor.model.add_prompt(inference_state=state,
                            frame_idx=0, text_str='object', output_prob_thresh=.5)
                        positive = np.asarray(visual['points'])[np.asarray(visual['point_labels']) == 1]
                        if len(positive):
                            x,y = np.rint(positive[0]*[width-1,height-1]).astype(int)
                            hits = np.flatnonzero(seed['out_binary_masks'][:,y,x])
                            if len(hits):
                                sizes = seed['out_binary_masks'].sum(axis=(1,2))
                                selected = min(hits, key=lambda i:sizes[i])
                                request['obj_id'] = int(seed['out_obj_ids'][selected])
                    _, output = predictor.model.add_prompt(**request)
                if points:
                    keep = np.asarray(output['out_obj_ids']) == request['obj_id']
                    output = {k:np.asarray(v)[keep] for k,v in output.items()
                              if k in ('out_obj_ids','out_binary_masks','out_boxes_xywh','out_probs')}
                masks = np.asarray(output['out_binary_masks'], dtype=bool)
                xywh = np.asarray(output['out_boxes_xywh'], dtype=np.float32).reshape(-1, 4)
                xyxy = xywh.copy()
                xyxy[:, 2:] += xywh[:, :2]
                xyxy *= [width, height, width, height]
                results.append({'masks': masks, 'boxes': xyxy,
                    'scores': np.asarray(output['out_probs'], dtype=np.float32),
                    'labels': np.full(len(masks), concept)})
            finally:
                predictor.model.reset_state(state)
    result = {key: np.concatenate([r[key] for r in results], axis=0)
              for key in ('masks', 'boxes', 'scores', 'labels')}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **result, backend='SAM3.1', model=MODEL, revision=REVISION,
                        source_revision=SOURCE_REVISION, elapsed_s=time.monotonic()-start)
    return Response(buffer.getvalue(), media_type='application/octet-stream')


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('SAM3_PORT', '8008')))
