#!/usr/bin/env python3
"""Cut VLM waits from a recorded trial; retain actual motion at recorded speed."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial',type=Path)
    args=parser.parse_args()
    video=args.trial/'live.mp4'
    events=[json.loads(line) for line in (args.trial/'tools.jsonl').read_text().splitlines()]
    duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration',
                                           '-of','csv=p=0',str(video)],text=True))
    begin,end=events[0]['time'],events[-1]['time']
    scale=duration/(end-begin)
    active={};intervals=[(0,2),(max(0,duration-2),duration)]
    motion={'move_to_pose','open_gripper','close_gripper','reset'}
    for event in events:
        name=event.get('tool')
        if name not in motion:continue
        if event['event']=='tool_start':active[name]=event['time']
        elif event['event'] in ('tool_end','tool_error') and name in active:
            start=active.pop(name)
            intervals.append((max(0,(start-begin)*scale-3),min(duration,(event['time']-begin)*scale+3)))
    merged=[]
    for start,finish in sorted(intervals):
        if merged and start<=merged[-1][1]+1:merged[-1][1]=max(merged[-1][1],finish)
        else:merged.append([start,finish])
    graph=';'.join(f'[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]' for i,(s,e) in enumerate(merged))
    graph+=';'+''.join(f'[v{i}]' for i in range(len(merged)))+f'concat=n={len(merged)}:v=1:a=0[out]'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(video),'-filter_complex',graph,'-map','[out]',
                    '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',
                    str(args.trial/'motion-and-recovery.mp4')],check=True)
    (args.trial/'motion-intervals.json').write_text(json.dumps({'source':'live.mp4','intervals_s':merged,
        'method':'tool timestamps mapped to recording duration with 3-second margins; waits removed, no motion speed change',
        'wall_to_recording_ratio':scale},indent=2))


if __name__=='__main__':main()
