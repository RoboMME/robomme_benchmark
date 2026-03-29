import torch
from collections import Counter


def bad_repeat_for_episode(episode: int, step: int = 100) -> int:
    """
    坏用法：
    每个 episode 都重新设 seed = episode * step，
    然后只取第一次 randint。
    """
    g = torch.Generator()
    g.manual_seed(episode * step)
    return int(torch.randint(1, 4, (1,), generator=g).item())  # 1,2,3


def good_repeat_stream(num_episodes: int, base_seed: int = 0) -> list[int]:
    """
    好用法对照：
    只初始化一次 generator，持续往后采样。
    """
    g = torch.Generator()
    g.manual_seed(base_seed)
    return [int(torch.randint(1, 4, (1,), generator=g).item()) for _ in range(num_episodes)]


def counts_for_bad_window(start_episode: int, window: int, step: int = 100) -> Counter:
    c = Counter()
    for ep in range(start_episode, start_episode + window):
        c[bad_repeat_for_episode(ep, step=step)] += 1
    return c


def find_worst_window(
    search_episodes: int = 5000,
    window: int = 90,
    target_repeat: int = 3,
    step: int = 100,
):
    """
    在 [0, search_episodes) 的连续 episode 起点里，
    找出一个窗口，使 target_repeat 的占比最高。
    """
    best_start = None
    best_counts = None
    best_ratio = -1.0

    for start in range(search_episodes - window + 1):
        c = counts_for_bad_window(start, window, step=step)
        ratio = c[target_repeat] / window
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = start
            best_counts = c

    return best_start, best_counts, best_ratio


def summarize_counter(c: Counter, total: int) -> str:
    return (
        f"repeat=1: {c[1]:4d} ({c[1]/total:6.2%}), "
        f"repeat=2: {c[2]:4d} ({c[2]/total:6.2%}), "
        f"repeat=3: {c[3]:4d} ({c[3]/total:6.2%})"
    )


def main():
    search_episodes = 5000
    window = 100
    step = 100
    target_repeat = 3

    print("=== bad pattern: reseed every episode with seed = episode * 100 ===")
    start, c_bad, ratio = find_worst_window(
        search_episodes=search_episodes,
        window=window,
        target_repeat=target_repeat,
        step=step,
    )
    print(f"search_episodes = {search_episodes}, window = {window}, step = {step}")
    print(f"worst window start = {start}")
    print(summarize_counter(c_bad, window))
    print(f"target repeat={target_repeat} ratio = {ratio:.2%}")
    print()

    print("=== good pattern: one persistent generator ===")
    good = good_repeat_stream(num_episodes=window, base_seed=start * step)
    c_good = Counter(good)
    print(f"use same nominal base seed = {start * step}, but do NOT reseed each episode")
    print(summarize_counter(c_good, window))
    print()

    print("=== first 20 episodes in the bad window ===")
    seq_bad = [bad_repeat_for_episode(ep, step=step) for ep in range(start, start + 20)]
    print(seq_bad)

    print()
    print("=== first 20 episodes in the good stream ===")
    print(good[:20])


if __name__ == "__main__":
    main()