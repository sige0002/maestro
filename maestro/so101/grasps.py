"""Geometry candidates from observed shape; no task locations or object identities."""
import numpy as np


def geometry_candidates(obj, floor_z):
    points = np.asarray(obj['points'])[:, :2]
    values, vectors = np.linalg.eigh(np.cov(points.T))
    # Align closing direction with the short axis for elongated objects.
    elongated = values[1] > 1.8*max(values[0],1e-9)
    yaw = float(np.arctan2(vectors[1,0],vectors[0,0])) if elongated else 0.
    yaw = (yaw+np.pi/2) % np.pi-np.pi/2
    center = np.asarray(obj['center']).copy()
    top = float(obj['top'][2])
    # Test heights within the visible object, allowing clearance for fingertips.
    heights = sorted(set(float(np.clip(center[2]+dz,floor_z+.014,top-.003))
                         for dz in (0,.003,.006)))
    if top-.003 < floor_z+.014:
        return []
    candidates=[]
    for pitch in (0., -.15, .15):
        for z in heights:
            for angle in (yaw,yaw+.2,yaw-.2,yaw+np.pi/2):
                position=center.copy();position[2]=z
                candidates.append((position.tolist(),[np.pi,pitch,float(angle)]))
    return candidates
