"""Camera-space geometry; no simulator dynamic poses or colour-specific masks."""
import cv2
import numpy as np


def point_cloud(depth, k, camera):
    k = np.asarray(k).reshape(3, 3)
    yy, xx = np.indices(depth.shape)
    xyz = np.stack([(xx-k[0, 2])*depth/k[0, 0],
                    (yy-k[1, 2])*depth/k[1, 1], depth], axis=-1)
    return xyz @ camera[:3, :3].T + camera[:3, 3]


def lift_pixel(pixel, depth, k, camera):
    x, y = np.rint(pixel).astype(int)
    if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
        raise ValueError('Point outside depth image')
    patch = depth[max(0,y-2):y+3, max(0,x-2):x+3]
    valid = patch[np.isfinite(patch) & (patch > .01)]
    if not len(valid):
        raise ValueError('No valid depth at the selected pixel')
    z = float(np.median(valid))
    k = np.asarray(k).reshape(3,3)
    return camera[:3,:3] @ np.array([(x-k[0,2])*z/k[0,0], (y-k[1,2])*z/k[1,1], z])+camera[:3,3]


def select_pointed_mask(detections, pixel):
    """Resolve detector distractors using the VLM's independently pointed target."""
    masks = np.asarray(detections['masks'], dtype=bool)
    x,y = np.rint(pixel).astype(int)
    if masks.ndim != 3 or not (0 <= y < masks.shape[1] and 0 <= x < masks.shape[2]):
        raise ValueError('Point is outside segmentation image')
    matches = np.flatnonzero(masks[:,y,x])
    if not len(matches):
        raise ValueError('No GroundedSAM mask contains the pointed target; inspect and reobserve')
    # A container mask can include its contents. Prefer the most specific
    # enclosing region, not the tray's higher detector score.
    areas=masks.reshape(len(masks),-1).sum(axis=1)
    return int(min(matches,key=lambda i:(areas[i],-float(detections['scores'][i]))))


def objects_above_table(depth, k, camera, floor_z):
    cloud = point_cloud(depth, k, camera)
    mask = (np.isfinite(cloud).all(axis=-1) & (cloud[:,:,2] > floor_z+.006)
            & (cloud[:,:,0] > .10) & (cloud[:,:,0] < .4)
            & (cloud[:,:,1] > -.32) & (cloud[:,:,1] < .25))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    objects = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 25:
            continue
        points = cloud[labels == index]
        lo, hi = np.percentile(points, [1, 99], axis=0)
        # Static cell/tray and robot are already modelled separately. Keep compact
        # movable items; this backend cannot segment touching/occluded assemblies.
        if np.any(hi[:2]-lo[:2] > .095) or hi[2] > floor_z+.10:
            continue
        lo[2] = floor_z
        yy, xx = np.where(labels == index)
        objects.append({'center': (hi+lo)/2, 'half_size': (hi-lo)/2+.0015,
                        'top': np.r_[(hi[:2]+lo[:2])/2, hi[2]],
                        'pixel': np.array([xx.mean(), yy.mean()]), 'mask': labels == index,
                        'points': points})
    return objects


def object_from_mask(mask, cloud, floor_z):
    """Fit a compact support-contact object to a semantic mask and observed depth.

    The support plane is known cell geometry, not a dynamic object's pose/size.
    Reject large robot/table/container masks and invalid or below-plane geometry.
    """
    mask = np.asarray(mask, dtype=bool)
    points = cloud[mask & np.isfinite(cloud).all(axis=-1)]
    if len(points) < 25:
        raise ValueError('Insufficient valid depth in GroundedSAM mask')
    lo, hi = np.percentile(points, [1, 99], axis=0)
    if (np.any(hi[:2]-lo[:2] > .10) or hi[2] > floor_z+.12
            or hi[2] < floor_z+.006):
        raise ValueError('Segmented object exceeds compact tabletop workspace')
    lo[2] = floor_z
    yy, xx = np.where(mask)
    return {'center': (lo+hi)/2, 'half_size': (hi-lo)/2+.0015,
            'top': np.r_[(lo[:2]+hi[:2])/2, hi[2]],
            'pixel': np.array([xx.mean(), yy.mean()]), 'mask': mask, 'points': points}


def objects_from_masks(detections, cloud, floor_z):
    """Semantic masks separate objects even when inferred depth merges with the table."""
    objects = []
    for index in np.argsort(-np.asarray(detections['scores'])):
        mask = np.asarray(detections['masks'][index], dtype=bool)
        if any((mask & o['mask']).sum() / max(1, (mask | o['mask']).sum()) > .5 for o in objects):
            continue
        try:
            obj = object_from_mask(mask, cloud, floor_z)
        except ValueError:
            continue
        obj['detection_index'] = int(index)
        objects.append(obj)
    return objects
