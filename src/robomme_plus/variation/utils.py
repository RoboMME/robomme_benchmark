"""Independent seeded sampling and access to render materials."""
import hashlib
import numpy as np


def rng_for(seed,domain):
    digest=hashlib.sha256(f'appearance-v1/{int(seed)}/{domain}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8],'little'))


def surfaces(actor):
    import sapien.render as render
    for body in actor._objs:
        component = getattr(body, 'entity', body).find_component_by_type(render.RenderBodyComponent)
        if component is None:
            continue
        for shape in component.render_shapes:
            if isinstance(shape, render.RenderShapeTriangleMesh):
                yield from shape.parts
            else:
                yield shape
