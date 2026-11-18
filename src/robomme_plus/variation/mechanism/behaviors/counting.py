"""counting-task motion hooks; success predicates stay native."""
import numpy as np
from .motion import RateSchedule, rng_for


def install(task, seed, config):
    from robomme_plus.tasks.counting.utils import controllers as c
    from robomme_plus.common.sim import position, step_number
    if task == 'BinFill':
        from . import bin_dense
        bin_dense.install(seed,config)
    elif task == 'StopCube':
        original = c.CurvedStop.reset

        def reset(self):
            original(self)
            from .stop_layout import relocate
            button, geometry = relocate(self.route, self.layout['button_xy'], self.env.cube_half_size, config['layout'])
            self.layout.update(route_center_xy=list(self.route.center), button_xy=button.tolist())
            for actor, xy in ((self.env.target,self.route.target),(self.env.button,button),
                              (self.env.cube,self.route.point(0))):
                p=c.position(actor,initial=False).copy();p[:2]=xy;c.place(actor,p)
            self._mix_schedule = RateSchedule(seed, 'stop-rate')
            self._mix_phase, self._mix_rate = 0., 1.
            self._mixed_audit = dict(kind='signed phase along native route behind button', samples=[],layout=geometry)

        def step(self, step):
            if self.stopped or self.success or self.failure:
                return
            age = step-self.start
            desired = self._mix_schedule.rate(age)
            point = self.route.point(self._mix_phase)
            # Preserve a predictable final approach through the stopping zone.
            # Reversals and acceleration happen outside this latency margin.
            if np.linalg.norm(point-self.route.target) > max(.10, self.radius+.04):
                self._mix_rate = desired
            self._mix_phase += self._mix_rate
            c.place(self.env.cube, [*self.route.point(self._mix_phase), self.z])
            self._mixed_audit['samples'].append([int(age), self._mix_phase, self._mix_rate])

        c.CurvedStop.reset, c.CurvedStop.before_step = reset, step
        config['style'] = 'route and target behind the button, with pauses, reverse travel and changing speed'
    elif task == 'PickXtimes':
        original = c.MovingPlacement.reset

        def reset(self):
            # target_position is called by native reset: prepare first.
            self._mix_schedule = RateSchedule(seed, 'pick-rate')
            # Speed and pauses vary; direction changes only on reverse dodges.
            self._mix_schedule.rates = np.abs(self._mix_schedule.rates)
            self._mix_schedule.phase = np.r_[0.,np.cumsum(self._mix_schedule.rates)]
            self._mix_phase_origin, self._mix_clock_origin = 0., 0.
            self._mix_direction = 1
            self._mix_dodge = None
            self._mix_cycle, self._mix_dodges = -1, 0
            self._mix_completed_attempts = 0
            self._mix_approach, self._mix_latched = 0, False
            self._mix_rng = rng_for(seed, 'pick-dodge')
            self._mixed_audit = dict(kind='forward or reverse dodge on the same ellipse',
                                     schedule='alternate eligible placement attempts, starting with dodge, until N//2 quota',
                                     approaches=[], dodges=[], samples=[])
            original(self)
            self._mix_dodge_quota = self.env.num_repeats//2
            self._mixed_audit.update(required_placements=self.env.num_repeats,dodge_quota=self._mix_dodge_quota)

        def target(self, step):
            phase = self._mix_phase_origin + self._mix_direction*(
                self._mix_schedule.value(self.motion_clock.predict(step))-self._mix_clock_origin)
            return c.ellipse_position(phase, self.origin, self.layout['ellipse_axes_m'],
                    self.motion_settings['period_steps'], self.motion_settings['direction_sign'],
                    self.layout['initial_phase_rad'])

        def step(self, step):
            if self.success or self.failure:
                return
            self.motion_clock.advance(step)
            if self.counter.count != self._mix_cycle and self._mix_dodge is not None:
                # End only on a counted landing. Native motion can already pause
                # at 1.5 cm, before the requested 1 cm dodge trigger is reached.
                self._mix_dodge = None
                self._mixed_audit['dodges'][-1]['ended_on_landing'] = True
            if self.counter.count != self._mix_cycle:
                self._mix_cycle = self.counter.count
            completed=self.counter.count+self.counter.misses
            if completed != self._mix_completed_attempts:
                self._mix_completed_attempts=completed
                self._mix_latched = False
            cube = position(self.env.target_cube, False)
            here = position(self.env.target, False)
            distance = float(np.linalg.norm(cube[:2]-here[:2]))
            held = self.grasped(self.env.target_cube)
            candidate = (self._mix_dodge is None
                         and not self._mix_latched and self.counter.lifted
                         and (held or self.counter.released_at is not None)
                         and cube[2] <= self.env.cube_half_size+.010 and distance < .06)
            dodge = False
            if candidate:
                self._mix_latched = True
                self._mix_approach += 1
                dodge = self._mix_approach%2==1 and self._mix_dodges<self._mix_dodge_quota
                self._mixed_audit['approaches'].append(dict(step=int(step), approach=self._mix_approach,
                        cycle=self._mix_cycle,completed_attempts=completed,dodge=dodge,
                        remaining_dodges_before=self._mix_dodge_quota-self._mix_dodges,
                        previously_lifted=True,held_at_trigger=bool(held),
                        released_at_step=self.counter.released_at,
                        cube_bottom_height_m=float(cube[2]-self.env.cube_half_size)))
            if dodge:
                period = self.motion_settings['period_steps']
                # Fast progress along the exact same ellipse. A reverse dodge
                # changes the ongoing direction as well as the burst direction.
                delta = float(self._mix_rng.uniform(.8, 1.2))*period/(2*np.pi)
                reverse = bool(rng_for(seed,f'pick-dodge-direction/{self._mix_approach}').random()<.5)
                previous_direction = self._mix_direction*self.motion_settings['direction_sign']
                if reverse:
                    clock = self._mix_schedule.value(self.motion_clock.predict(step))
                    self._mix_phase_origin += self._mix_direction*(clock-self._mix_clock_origin)
                    self._mix_clock_origin = clock
                    self._mix_direction *= -1
                delta *= self._mix_direction
                duration = 6
                self._mix_dodge = [duration, delta/duration]
                self._mix_dodges += 1
                self._mixed_audit['dodges'].append(dict(step=int(step), cycle=self._mix_cycle,
                        duration_steps=duration, phase_jump=delta, mode='reverse' if reverse else 'forward',
                        direction_before=previous_direction,
                        direction_after=self._mix_direction*self.motion_settings['direction_sign'],
                        cube_bottom_height_m=float(cube[2]-self.env.cube_half_size),
                        trigger_distance_m=float(np.linalg.norm(cube[:2]-here[:2]))))
            if self._mix_dodge is not None:
                self.motion_clock.set_paused(False,step)
                self._mix_phase_origin += self._mix_dodge[1]
                self._mix_dodge[0] -= 1
                if self._mix_dodge[0] == 0:
                    self._mix_dodge = None
            c.place(self.env.target, target(self, step))
            self._mixed_audit['samples'].append([int(step-self.start), *position(self.env.target, False)[:2]])

        c.MovingPlacement.reset, c.MovingPlacement.target_position = reset, target
        c.MovingPlacement.before_step = step
        config.update(style='same ellipse with pauses, variable speed and forward/reverse proximity dodges',
                      dodge_trigger_m=.06, dodge_control_steps=6,
                      dodge_schedule='alternate attempts, dodge first; total quota N//2',
                      one_decision_per_completed_pick_place_attempt=True,reverse_probability_given_dodge=.5,
                      reverse_dodge_changes_ongoing_direction=True,spatial_trajectory_unchanged=True)
        config['dodge_max_cube_bottom_height_m']=.010
    elif task == 'SwingXtimes':
        original = c.ReactiveSwing.reset

        class OrbitSlides:
            def __init__(self, base, start):
                self.base, self.start = base, start
                rng = rng_for(seed, 'swing-orbits')
                self.radii = rng.uniform(.04, .05, 2)
                # Native dodge envelopes already avoid the button, starting
                # cube and opposite target. Share that envelope between the
                # larger circular orbit and a smaller translated dodge center.
                reach=np.tile([.06,.12],(2,1))
                reach[:,1]=np.minimum(reach[:,1],np.abs(base.origins[:,1])-.035)
                self.radii=np.minimum(self.radii,reach.min(axis=1)-.006)
                self.grid_scale=(reach-self.radii[:,None])/[.06,.12]
                self.periods = rng.uniform(48., 72., 2)
                self.phases = rng.uniform(0., 2*np.pi, 2)
                self.signs = rng.choice([-1,1], 2)

            def __getattr__(self, name):
                return getattr(self.base, name)

            def offset(self, step):
                age = max(0., step-self.start)
                theta = self.phases + self.signs*2*np.pi*age/self.periods
                amplitude = min(1., age/20.)*self.radii
                return amplitude[:,None]*np.c_[np.cos(theta), np.sin(theta)]

            def positions(self, step):
                points = self.base.positions(step)
                # Reserve orbit room inside the native validated dodge envelope.
                points[:,:2] = self.base.origins[:,:2] + self.grid_scale*(points[:,:2]-self.base.origins[:,:2]) + self.offset(step)
                return points

            def approach(self, tcp_xy, step, active_target=None):
                index = 0 if active_target is None else active_target
                # Preserve the physical 5 cm approach radius after scaling
                # the dodge-center grid; orbiting never disables a dodge.
                delta=np.asarray(tcp_xy)-self.positions(step)[index,:2]
                local=self.base.positions(step)[index,:2]+delta
                return self.base.approach(local, step, active_target)

            def moving(self, index, step):
                # Orbiting remains countable; only an active rapid dodge gates a hit.
                return self.base.moving(index, step)

        def reset(self):
            original(self)
            self.slides = OrbitSlides(self.slides, step_number(self.env))
            self._mixed_audit = dict(kind='larger faster side-confined circles continuing after reactive dodges',
                                     orbit_radii_m=self.slides.radii.tolist(),
                                     orbit_period_steps=self.slides.periods.tolist(),
                                     grid_scale=self.slides.grid_scale.tolist(),
                                     orbit_continues_after_dodge=True,target_centers_side_margin_m=.035)
            from .swing_layout import relocate
            relocate(self,seed)

        c.ReactiveSwing.reset = reset
        config['style'] = 'pickup in front of a seeded target; larger faster circles continue after every dodge'
