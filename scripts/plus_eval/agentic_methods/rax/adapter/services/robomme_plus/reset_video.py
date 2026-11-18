"""Host/simulator-only paging of unchanged initial RGB-D observations."""

# Both cameras are 256x256; 16 RGB-D frames are far below the 1 GiB RPC bound.
RESET_PAGE_FRAMES = 16


def reset_frames(sim):
    """Read every frame once, checking identity/order without advancing physics."""
    start, total = 0, None
    identity = (sim.episode_id, sim.step_id)
    while total is None or start < total:
        page = sim.request("reset_sequence_page", start=start, count=RESET_PAGE_FRAMES)
        frames = page.get("frames")
        count = page.get("total_frames")
        stop = page.get("next_start")
        if (page.get("episode_id"), page.get("step_id")) != identity:
            raise ValueError("Reset video episode/step changed during transfer")
        if type(count) is not int or count < 1 or (total is not None and count != total):
            raise ValueError("Reset video frame count changed or is invalid")
        if (page.get("start") != start or not isinstance(frames, list)
                or not 1 <= len(frames) <= RESET_PAGE_FRAMES
                or type(stop) is not int or stop != start + len(frames) or stop > count):
            raise ValueError("Reset video page has missing, duplicate or out-of-order frames")
        total = count
        yield from frames
        start = stop
