"""SO101 grasp IK and sampled collision planning using the shared cell model.

The model is used for robot/static-obstacle geometry, not object localization.
Free objects from the simulator template are removed. Observed obstacles must
be supplied by perception. No simulated body is attached or teleported.
"""
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .kinematics import JOINTS


class GraspPlanner:
    def __init__(self, source='/so101/scripts/models', scene='so101cell'):
        sys.path.insert(0, source)
        from model import mujoco_description, verify_assets
        self.manifest = verify_assets()
        xml = ET.fromstring(mujoco_description(scene=scene))
        world = xml.find('worldbody')
        for body in list(world.findall('body')):
            if body.find('freejoint') is not None:
                world.remove(body)
        self.object_names = ('red', 'green', 'blue', *[f'slot{i}' for i in range(16)])
        for color in self.object_names:
            body = ET.SubElement(world, 'body', name='observed_body_'+color, mocap='true', pos='10 10 10')
            ET.SubElement(body, 'geom', name='observed_'+color, type='box',
                          size='.016 .016 .016', contype='1', conaffinity='1')
        hand_xml = xml.find(".//body[@name='gripper_link']")
        ET.SubElement(hand_xml, 'geom', name='carried_object', type='box',
                      size='.017 .017 .015', pos='.0136 -.000218 -.094', mass='0', contype='1', conaffinity='1')
        self.model = m = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
        self.data = mujoco.MjData(m)
        self.qadr = np.array([m.joint(n).qposadr[0] for n in JOINTS])
        self.dadr = np.array([m.joint(n).dofadr[0] for n in JOINTS])
        self.actuators = np.array([m.actuator(n).id for n in JOINTS])
        self.bounds = np.array([m.joint(n).range for n in JOINTS])
        self.hand = m.body('gripper_link').id
        # Contact calibration for the bundled 30 mm cube and original finger meshes.
        self.grasp_offset = np.array([.0136, -.000218, -.095])
        self.floor_z = float(m.body('tray').pos[2])
        self.robot_bodies = set()
        base = m.body('base_link').id
        for body in range(m.nbody):
            parent = body
            while parent:
                if parent == base:
                    self.robot_bodies.add(body)
                    break
                parent = m.body_parentid[parent]
        self.obstacles = []
        self.check_count = 0
        self.target_geom = None
        self.carried_geom = m.geom('carried_object').id
        m.geom_contype[self.carried_geom] = m.geom_conaffinity[self.carried_geom] = 0
        self.finger_bodies = {m.body(n).id for n in ('gripper_link', 'moving_jaw_so101_v1_link')}

    def set_objects(self, positions, target):
        self.target_geom = None if target is None else self.model.geom('observed_'+target).id
        for color, position in positions.items():
            geom = self.model.geom('observed_'+color).id
            body = self.model.body('observed_body_'+color).id
            self.data.mocap_pos[self.model.body_mocapid[body]] = position
            self.model.geom_contype[geom] = self.model.geom_conaffinity[geom] = 1

    def set_carrying(self, carrying):
        if self.target_geom is not None:
            self.model.geom_contype[self.target_geom] = self.model.geom_conaffinity[self.target_geom] = 0 if carrying else 1
        self.model.geom_contype[self.carried_geom] = self.model.geom_conaffinity[self.carried_geom] = int(carrying)

    def pose(self, q):
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        hand = self.data.body(self.hand)
        return hand.xpos + hand.xmat.reshape(3, 3) @ self.grasp_offset

    def ik(self, position, seed, opening, orientation=None):
        from scipy.spatial.transform import Rotation
        target = np.asarray(position, dtype=float)
        desired = np.eye(3) if orientation is None else Rotation.from_euler('xyz', orientation).as_matrix() @ np.diag([1,-1,-1])
        def residual(arm):
            self.data.qpos[self.qadr] = np.r_[arm, opening]
            mujoco.mj_kinematics(self.model, self.data)
            hand = self.data.body(self.hand)
            rot = hand.xmat.reshape(3, 3)
            return np.r_[10*(hand.xpos + rot @ self.grasp_offset-target),
                         .5*(rot[:, 2]-desired[:,2]), .15*(rot[:, 0]-desired[:,0])]
        angle = np.arctan2(target[1], target[0])
        candidates = [np.asarray(seed)[:5], [-angle, .3, -.1, 1.3, -angle]]
        solutions = []
        for guess in candidates:
            guess = np.clip(guess, self.bounds[:5, 0]+.03, self.bounds[:5, 1]-.03)
            result = least_squares(residual, guess, bounds=(self.bounds[:5, 0]+.03,
                self.bounds[:5, 1]-.03), max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            q = np.r_[result.x, opening]
            error = float(np.linalg.norm(residual(result.x)[:3])/10)
            self.pose(q)
            angle = np.linalg.norm(Rotation.from_matrix(desired.T @ self.data.body(self.hand).xmat.reshape(3,3)).as_rotvec())
            if result.success and error < .003 and (orientation is None or angle < .15) and self.valid(q):
                solutions.append(q)
        if not solutions:
            raise ValueError(f'No collision-free grasp IK for {target.tolist()}')
        return min(solutions, key=lambda q: np.linalg.norm(q-np.asarray(seed)))

    def valid(self, q):
        self.check_count += 1
        if not np.all(np.isfinite(q)) or np.any(q < self.bounds[:, 0]) or np.any(q > self.bounds[:, 1]):
            return False
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        for contact in self.data.contact:
            a, b = [int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            geoms = {contact.geom1, contact.geom2}
            if self.target_geom in geoms and (a in self.finger_bodies or b in self.finger_bodies):
                continue
            if self.carried_geom in geoms and a in self.finger_bodies and b in self.finger_bodies:
                continue
            if self.carried_geom in geoms and contact.pos[2] <= self.floor_z+.005 \
                    and abs(contact.frame[2]) > .9 and contact.dist > -.003:
                # The picked cube starts supported on the table. Permit only
                # this shallow horizontal support contact while lifting off.
                continue
            # Only consider collisions involving the arm. Static cell parts
            # can contact each other at their assembly datums.
            if (a in self.robot_bodies or b in self.robot_bodies) and contact.dist < -.0005:
                # The fixed mounting base rests on the table by design.
                if self.model.body(a).name == 'base_link' or self.model.body(b).name == 'base_link':
                    continue
                return False
        return True

    def edge(self, a, b):
        count = max(2, int(np.ceil(np.max(np.abs(b-a))/.025))+1)
        return all(self.valid(a+(b-a)*t) for t in np.linspace(0, 1, count))

    def path(self, start, goal):
        start, goal = np.asarray(start), np.asarray(goal)
        if not self.valid(start) or not self.valid(goal):
            raise ValueError('Path endpoint is in collision or outside limits')
        if self.edge(start, goal):
            return [start, goal]
        rng = np.random.default_rng(0)
        tree, parents = [start], [-1]
        for iteration in range(2000):
            sample = goal if iteration % 5 == 0 else np.r_[rng.uniform(
                self.bounds[:5, 0], self.bounds[:5, 1]), goal[5]]
            nearest = int(np.argmin([np.linalg.norm(q-sample) for q in tree]))
            delta = sample-tree[nearest]
            q = tree[nearest]+delta*min(1, .25/max(np.linalg.norm(delta), 1e-9))
            if not self.edge(tree[nearest], q):
                continue
            tree.append(q); parents.append(nearest)
            if np.linalg.norm(q-goal) < .5 and self.edge(q, goal):
                result, index = [goal], len(tree)-1
                while index >= 0:
                    result.append(tree[index]); index = parents[index]
                return list(reversed(result))
        raise RuntimeError('No collision-free joint path found within planning budget')

    def gravity_command(self, desired):
        self.data.qpos[self.qadr] = desired
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        command = desired + self.data.qfrc_bias[self.dadr]/self.model.actuator_gainprm[self.actuators, 0]
        if np.any(command < self.bounds[:, 0]) or np.any(command > self.bounds[:, 1]):
            raise ValueError('Gravity compensated target exceeds joint limits')
        return command
