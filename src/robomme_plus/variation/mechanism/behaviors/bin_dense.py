"""Close conveyor neighbors, retaining native speed and finite target arrivals."""
import copy
import numpy as np
from .motion import rng_for


def install(seed,config):
    from robomme_plus.tasks.counting.utils import controllers as c
    original_build=c.ConveyorCounting.build
    original_reset=c.ConveyorCounting.reset
    original_spawn=c.ConveyorCounting.spawn_distractor

    def build(self):
        original_build(self)
        rng=rng_for(seed,'close-conveyor-neighbors')
        planned=copy.deepcopy(self.initial_layout);self._neighbors=[]
        half=self.env.cube_half_size;speed=self.config['belt_speed_m_s']*self.env.control_timestep
        center=self.config['belt_center_x'];width=self.config['belt_width_m']
        end_y=self.config['belt_center_y']+self.config['belt_length_m']/2
        for index,target in enumerate(self.initial_layout):
            chosen=None
            for _ in range(12):
                yaw=float(rng.uniform(-np.pi,np.pi));extent=half*(abs(np.cos(yaw))+abs(np.sin(yaw)))
                gap=float(rng.uniform(.012,.016));distance=target['extent_xy']+extent+gap
                offsets=[np.array([sign*distance,0.]) for sign in rng.permutation([-1,1])]+[np.array([0.,distance])]
                for offset in offsets:
                    point=np.array([target['entry_x'],target['entry_y']])+offset
                    if abs(point[0]-center)+extent+.008>width/2 or point[1]+extent>end_y-.008:continue
                    if any(np.max(np.abs(point-[i['entry_x'],i['entry_y']+(target['release']-i['release'])*speed]))
                           <extent+i['extent_xy']+.010 for i in planned):continue
                    chosen=dict(color=str(rng.choice(self.config['distractor_colors'])),
                        release=target['release'],entry_x=float(point[0]),entry_y=float(point[1]),
                        yaw_rad=yaw,extent_xy=float(extent),arrival_role='stream_distractor',
                        close_neighbor_of=self.items[index]['actor'].name,planned_gap_m=gap)
                    break
                if chosen:break
            if chosen is None:raise RuntimeError('Cannot place a safe close conveyor neighbor')
            item=self.add_cube(chosen);self._neighbors.append((item,copy.deepcopy(chosen)));planned.append(chosen)

    def reset(self):
        original_reset(self)
        for item,plan in self._neighbors:item.update(plan,status='queued',ever_grasped=False)
        self._mixed_audit=dict(kind='close conveyor neighbors at native speed',speed_m_s=self.config['belt_speed_m_s'],
            speed_multiplier=1.,target_spawn_schedule_unchanged=True,target_supply_unchanged=True,
            neighbor_gap_range_m=[.012,.016],neighbors=[dict(cube=item['actor'].name,**plan) for item,plan in self._neighbors],
            bursts=[],pauses=False,reversals=False)

    def spawn(self,age):
        due=[item for item,_ in self._neighbors if item['status']=='queued' and age>=item['release']]
        if due:
            # These are ordinary non-target stream cubes clustered beside the
            # finite targets. Replace the next background emission at this time.
            for item in due:
                item['spawn_index']=self.distractors.emitted;self.distractors.emitted+=1
            self.distractors.next_step=max(self.distractors.next_step,int(age)+self.distractors._gap())
            return
        original_spawn(self,age)

    c.ConveyorCounting.build=build;c.ConveyorCounting.reset=reset;c.ConveyorCounting.spawn_distractor=spawn
    config.update(style='close conveyor neighbors at original speed',speed_multiplier=1.,
                  neighbor_gap_range_m=[.012,.016],target_supply_and_spawn_times_unchanged=True)
