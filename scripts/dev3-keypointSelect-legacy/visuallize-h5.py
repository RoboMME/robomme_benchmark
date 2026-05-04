#!/usr/bin/env python3
"""Inspect Robomme-style HDF5 layout and play back timestep_*/obs/front_rgb in order."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
import cv2
import h5py
import numpy as np

DEFAULT_H5 = "/data/hongzefu/data-0306/record_dataset_BinFill.h5"

_TIMESTEP_RE = re.compile(r"^timestep_(\d+)(?:_dup(\d+))?$")
_EPISODE_RE = re.compile(r"^episode_(\d+)$")


# # 无参数 = 默认 H5 + 只打印第一个 episode 的 --tree
# uv run python scripts/dev3/visuallize-h5.py

# # 仍要看全文件
# uv run python scripts/dev3/visuallize-h5.py ... --tree --tree-all


def list_episode_indices(h5: h5py.File) -> list[int]:
    out: list[int] = []
    for k in h5.keys():
        m = _EPISODE_RE.match(k)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def first_episode_index(h5: h5py.File) -> int | None:
    eps = list_episode_indices(h5)
    return eps[0] if eps else None


def iter_sorted_timestep_keys(episode_group: h5py.Group) -> list[str]:
    keyed: list[tuple[int, int, str]] = []
    for name in episode_group.keys():
        if name == "setup":
            continue
        m = _TIMESTEP_RE.match(name)
        if not m:
            continue
        k = int(m.group(1))
        dup = int(m.group(2)) if m.group(2) is not None else 0
        keyed.append((k, dup, name))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in keyed]


def _decode_scalar(val) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return ""
        x = val.reshape(-1)[0]
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        return str(x.item())
    return str(val)


def print_h5_tree(
    obj: h5py.Group | h5py.File,
    prefix: str = "",
    max_depth: int | None = None,
    depth: int = 0,
) -> None:
    """Print groups and datasets with shape/dtype (no array payload)."""
    if max_depth is not None and depth > max_depth:
        return
    keys = sorted(obj.keys(), key=lambda s: (s.startswith("timestep_"), s))
    for i, key in enumerate(keys):
        is_last = i == len(keys) - 1
        branch = "└── " if is_last else "├── "
        item = obj[key]
        path = f"{prefix}{branch}{key}"
        if isinstance(item, h5py.Group):
            print(f"{path}/")
            ext = "    " if is_last else "│   "
            print_h5_tree(item, prefix + ext, max_depth=max_depth, depth=depth + 1)
        elif isinstance(item, h5py.Dataset):
            shape = item.shape
            dtype = item.dtype
            print(f"{path}  shape={shape} dtype={dtype}")


def _to_uint8_hwc(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.dtype in (np.float32, np.float64):
        if x.size and float(np.nanmax(x)) <= 1.0 + 1e-6:
            x = (np.nan_to_num(x, nan=0.0) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            x = np.nan_to_num(x, nan=0.0).clip(0, 255).astype(np.uint8)
    else:
        x = x.astype(np.uint8, copy=False)
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[0] < min(x.shape[1], x.shape[2]):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    return x


def _to_bgr(frame_hwc: np.ndarray, assume_rgb: bool) -> np.ndarray:
    u = _to_uint8_hwc(frame_hwc)
    if u.ndim == 3 and u.shape[2] == 3 and assume_rgb:
        u = cv2.cvtColor(u, cv2.COLOR_RGB2BGR)
    elif u.ndim == 3 and u.shape[2] == 4 and assume_rgb:
        u = cv2.cvtColor(u, cv2.COLOR_RGBA2BGR)
    return u


def _vstack_bgr(front: np.ndarray, wrist: np.ndarray | None) -> np.ndarray:
    if wrist is None:
        return front
    h1, w1 = front.shape[:2]
    h2, w2 = wrist.shape[:2]
    if w2 != w1:
        wrist = cv2.resize(wrist, (w1, int(h2 * w1 / max(w2, 1))))
    return np.vstack([front, wrist])


def _read_joint(episode_group: h5py.Group, ts_key: str) -> np.ndarray | None:
    path = f"{ts_key}/obs/joint_state"
    if path not in episode_group:
        return None
    d = episode_group[path]
    if not isinstance(d, h5py.Dataset):
        return None
    return np.asarray(d[()])


def _overlay_lines(
    bgr: np.ndarray,
    lines: list[str],
    start_y: int = 24,
    line_height: int = 22,
) -> None:
    x0, y = 8, start_y
    for line in lines:
        if len(line) > 120:
            line = line[:117] + "..."
        cv2.putText(
            bgr,
            line,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y += line_height


def playback_episode(
    h5_path: Path,
    episode: int,
    *,
    out_path: Path | None,
    show: bool,
    fps: float,
    assume_rgb: bool,
    include_wrist: bool,
    joint_overlay: bool,
) -> None:
    with h5py.File(h5_path, "r") as h5:
        ep_key = f"episode_{episode}"
        if ep_key not in h5:
            avail = list_episode_indices(h5)
            raise SystemExit(f"missing {ep_key}; available episodes: {avail}")
        eg = h5[ep_key]
        if not isinstance(eg, h5py.Group):
            raise SystemExit(f"{ep_key} is not a group")
        ts_keys = iter_sorted_timestep_keys(eg)
        if not ts_keys:
            raise SystemExit(f"no timestep_* groups under {ep_key}")

    writer: cv2.VideoWriter | None = None
    delay_ms = max(1, int(1000 / max(fps, 1e-6)))
    prev_joint: np.ndarray | None = None

    try:
        with h5py.File(h5_path, "r") as h5:
            eg = h5[f"episode_{episode}"]
            assert isinstance(eg, h5py.Group)
            for idx, ts_key in enumerate(ts_keys):
                ts = eg[ts_key]
                if not isinstance(ts, h5py.Group):
                    continue
                obs = ts.get("obs")
                if not isinstance(obs, h5py.Group) or "front_rgb" not in obs:
                    continue
                front_ds = obs["front_rgb"]
                if not isinstance(front_ds, h5py.Dataset):
                    continue
                front = np.asarray(front_ds[()])
                front_bgr = _to_bgr(front, assume_rgb)
                wrist_bgr = None
                if include_wrist and "wrist_rgb" in obs:
                    wrist_ds = obs["wrist_rgb"]
                    if isinstance(wrist_ds, h5py.Dataset):
                        wrist_bgr = _to_bgr(np.asarray(wrist_ds[()]), assume_rgb)
                frame = _vstack_bgr(front_bgr, wrist_bgr)

                lines = [f"ep{episode} {ts_key}  frame={idx}"]
                j = _read_joint(eg, ts_key)
                if joint_overlay and j is not None:
                    jf = np.asarray(j, dtype=np.float64).reshape(-1)
                    lines.append(f"||joint||={float(np.linalg.norm(jf)):.4f}")
                    if prev_joint is not None and prev_joint.shape == jf.shape:
                        d = float(np.linalg.norm(jf - prev_joint))
                        lines.append(f"||d joint||={d:.4f}")
                    prev_joint = jf
                elif j is not None:
                    prev_joint = np.asarray(j, dtype=np.float64).reshape(-1)

                info_g = ts.get("info")
                if isinstance(info_g, h5py.Group):
                    for name in ("simple_subgoal", "is_subgoal_boundary"):
                        if name not in info_g:
                            continue
                        ds = info_g[name]
                        if isinstance(ds, h5py.Dataset):
                            lines.append(f"{name}={_decode_scalar(ds[()])}")

                _overlay_lines(frame, lines)

                if writer is None and out_path is not None:
                    h, w = frame.shape[:2]
                    fourcc_fn = getattr(cv2, "VideoWriter_fourcc")
                    fourcc = fourcc_fn(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                    if not writer.isOpened():
                        raise SystemExit(f"failed to open VideoWriter for {out_path}")

                if writer is not None:
                    writer.write(frame)

                if show:
                    cv2.imshow("visuallize-h5", frame)
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key in (ord("q"), 27):
                        break
    finally:
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "h5_path",
        nargs="?",
        default=DEFAULT_H5,
        type=Path,
        help=f"HDF5 path (default: {DEFAULT_H5})",
    )
    p.add_argument("--episode", type=int, default=None, help="Episode index for playback")
    p.add_argument(
        "--list-episodes",
        action="store_true",
        help="Print sorted episode_* indices and exit",
    )
    p.add_argument(
        "--tree",
        action="store_true",
        help="Print HDF5 structure for the first episode (or --episode); no array payload",
    )
    p.add_argument(
        "--tree-all",
        action="store_true",
        help="With --tree, print the full file (all episode_* groups) instead of one episode",
    )
    p.add_argument("--play", action="store_true", help="Show frames with cv2.imshow")
    p.add_argument("--out", type=Path, default=None, help="Write MP4 to this path")
    p.add_argument("--fps", type=float, default=20.0, help="Playback / video FPS")
    p.add_argument(
        "--bgr",
        action="store_true",
        help="Treat stored images as BGR already (skip RGB→BGR)",
    )
    p.add_argument(
        "--wrist",
        action="store_true",
        help="Stack wrist_rgb below front_rgb",
    )
    p.add_argument(
        "--joint-overlay",
        action="store_true",
        help="Overlay joint norm and L2 delta vs previous frame",
    )
    args = p.parse_args(argv)
    if len(argv) == 0:
        args.tree = True

    h5_path = args.h5_path.expanduser()
    if not h5_path.is_file():
        print(f"error: file not found: {h5_path}", file=sys.stderr)
        return 1

    assume_rgb = not args.bgr

    with h5py.File(h5_path, "r") as h5:
        if args.list_episodes:
            eps = list_episode_indices(h5)
            print("episodes:", " ".join(str(e) for e in eps) if eps else "(none)")
            return 0

        if args.tree:
            if args.tree_all:
                print_h5_tree(h5, prefix="")
            else:
                ep_idx = (
                    args.episode
                    if args.episode is not None
                    else first_episode_index(h5)
                )
                if ep_idx is None:
                    print("error: no episode_* groups in HDF5", file=sys.stderr)
                    return 1
                ek = f"episode_{ep_idx}"
                if ek not in h5:
                    print(f"error: missing {ek}", file=sys.stderr)
                    return 1
                g = h5[ek]
                if not isinstance(g, h5py.Group):
                    print(f"error: {ek} is not a group", file=sys.stderr)
                    return 1
                print(f"{ek}/")
                print_h5_tree(g, prefix="")
            if not args.play and args.out is None:
                return 0

        if args.play or args.out is not None:
            if args.episode is None:
                print("error: --play / --out require --episode", file=sys.stderr)
                return 1
            playback_episode(
                h5_path,
                args.episode,
                out_path=args.out,
                show=args.play,
                fps=args.fps,
                assume_rgb=assume_rgb,
                include_wrist=args.wrist,
                joint_overlay=args.joint_overlay,
            )
            return 0

    if not args.tree and not args.list_episodes:
        print(
            "error: specify one of --list-episodes, --tree, --play, or --out",
            file=sys.stderr,
        )
        p.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
