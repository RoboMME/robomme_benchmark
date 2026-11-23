"""Double finite container-route playback; retain the native layout and reveals."""


def install(config):
    from robomme_plus.tasks.permanence.utils import tasks
    original=tasks.swap_routes

    def faster(env,settings,*args,**kwargs):
        if settings.get('continuous_motion'):
            raise ValueError('Atomic container_speed_2x requires a finite-swap episode')
        routes,permutation,counts=original(env,settings,*args,**kwargs)
        # Keep every intermediate waypoint and its swept-collision check.
        # Subsampling positions would cut corners in the one-way passage.
        update,predict=routes.update,routes.predict
        def double_update():
            update()
            update()
        routes.update=double_update
        routes.predict=lambda index,steps:predict(index,2*steps)
        routes.metadata['atomic_speed_multiplier']=2.
        return routes,permutation,counts

    tasks.swap_routes=faster
    config.update(style='original finite container routes at twice the speed',
                  container_speed_multiplier=2.,pre_mask_shuffle=False,
                  original_layout=True,original_button_deadlines=True)
