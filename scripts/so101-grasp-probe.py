#!/usr/bin/env python3
"""Isolated contact diagnostic; simulator truth is NEVER sent to the agent."""
import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.environ.get('SO101_MODEL_SOURCE', '/so101/scripts/models'))
from model import HOME_RAD, mujoco_description
from maestro.so101.planning import GraspPlanner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', choices=['red','green','blue'], default='red')
    parser.add_argument('--closing', type=float, default=.1)
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--hold-seconds', type=float, default=60)
    parser.add_argument('--pitch', type=float, default=0)
    parser.add_argument('--yaw', type=float, default=0)
    parser.add_argument('--height-offset', type=float, default=0)
    parser.add_argument('--x-offset', type=float, default=0)
    parser.add_argument('--y-offset', type=float, default=0)
    parser.add_argument('--friction', type=float, help='Diagnostic material override, reported explicitly')
    parser.add_argument('--contact-solver', action='store_true',help='Test elliptic cones and higher tangential impedance')
    parser.add_argument('--baseline-contact-solver', action='store_true',help='Reproduce the prior pyramidal/impratio=1 solver')
    args = parser.parse_args()
    if args.contact_solver and args.baseline_contact_solver:
        parser.error('Choose only one solver override')
    planner = GraspPlanner(source=os.environ.get('SO101_MODEL_SOURCE', '/so101/scripts/models'))
    xml = mujoco_description(cube_layout='mixed')
    if args.friction is not None:
        xml = xml.replace('friction="1 0.005 0.0001"', f'friction="{args.friction} 0.005 0.001" condim="6"')
    model = mujoco.MjModel.from_xml_string(xml)
    if args.contact_solver:
        model.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.impratio=10
    if args.baseline_contact_solver:
        model.opt.cone=mujoco.mjtCone.mjCONE_PYRAMIDAL
        model.opt.impratio=1
    data = mujoco.MjData(model)
    body = model.body('cube_'+args.object).id
    # Truth here is diagnostic setup/measurement only, not a perception backend.
    center = model.body_pos[body].copy()
    target = center+[args.x_offset,args.y_offset,args.height_offset]
    orientation = [np.pi,args.pitch,args.yaw]
    pre = planner.ik(target+[0,0,.05], np.array(HOME_RAD), .65,orientation)
    data.qpos[planner.qadr] = pre
    mujoco.mj_forward(model,data)

    def move(goal, duration=3, settle=2, hold=False):
        start = data.qpos[planner.qadr].copy()
        if hold:
            start[5] = goal[5]
        for i in range(int((duration+settle)/model.opt.timestep)):
            t = min(1, i*model.opt.timestep/duration)
            target = start+(goal-start)*(3*t*t-2*t*t*t)
            data.ctrl[planner.actuators] = planner.gravity_command(target)
            mujoco.mj_step(model,data)

    grasp = pre
    for z in np.linspace(target[2]+.05,target[2],10)[1:]:
        grasp = planner.ik([*target[:2],z],grasp,.65,orientation)
        move(grasp,duration=.3,settle=.15)
    closing=args.closing
    if args.adaptive:
        for opening in np.linspace(.65,args.closing,23)[1:]:
            grasp[5]=opening
            move(grasp,duration=.2,settle=.4)
            if data.qpos[planner.qadr[5]] > opening+.015:
                closing=float(opening)
                break
    else:
        grasp[5] = closing
        move(grasp)
    lift = planner.ik(target+[0,0,.06],grasp,closing,orientation)
    move(lift,settle=args.hold_seconds,hold=True)
    actual = data.body(body).xpos.copy()
    jaw=float(data.qpos[planner.qadr[5]])
    lift[5]=.65
    move(lift,duration=1,settle=2)
    released=data.body(body).xpos.copy()
    print(json.dumps({'diagnostic_only':True,'conditions':vars(args),
                      'solver':{'cone':int(model.opt.cone),'impratio':float(model.opt.impratio)},
                      'initial_object_position_m':center.tolist(),
                      'final_object_position_m':actual.tolist(),
                      'jaw_rad':jaw,'closing_command_rad':closing,
                      'released_position_m':released.tolist(),
                      'fell_after_open':bool(released[2]<actual[2]-.03),
                      'retained':bool(actual[2]>center[2]+.04)},indent=2))


if __name__ == '__main__':
    main()
