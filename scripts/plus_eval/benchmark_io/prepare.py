"""Prepare immutable, operator-owned protocol records without starting a simulator."""
import argparse
import json
import os
from pathlib import Path

from .protocol import SETTINGS, Study, setting_key, source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--method", required=True, help="Your method identifier")
    parser.add_argument("--setting", choices=SETTINGS, required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--export-demos", action="store_true")
    parser.add_argument("--demos", default=os.environ.get("ROBOMME_PLUS_DEMOS", "data/robomme_plus_train/h5"))
    args = parser.parse_args()
    from robomme_plus import BenchmarkEnvBuilder
    import robomme_plus
    tasks = args.tasks or BenchmarkEnvBuilder.get_task_list()
    records = {}
    for task in tasks:
        builder = BenchmarkEnvBuilder(task, dataset="train")
        records[task] = [builder.get_episode_metadata(i) for i in range(builder.get_episode_num())]
    study = Study(args.root, method=args.method, setting=args.setting,
                  budget=0 if setting_key(args.setting) in ("zero-shot", "demo-only") else args.budget, records=records)
    fingerprint = source_fingerprint(Path(robomme_plus.__file__).parent)
    path = study.root / "benchmark_source.json"
    if path.exists() and json.loads(path.read_text()) != fingerprint:
        raise ValueError("Benchmark code changed since study preparation")
    path.write_text(json.dumps(fingerprint, indent=2))
    (study.root / "study.json").write_text(json.dumps(study.config, indent=2))
    if args.export_demos:
        from .demonstrations import export_task
        for task in tasks:
            manifest = export_task(study, task, args.demos, study.root / "demonstrations" / task)
            print(json.dumps(dict(task=task, demo_frames=manifest["image_count"])), flush=True)
    print(json.dumps(dict(root=str(study.root), tasks=len(tasks), setting=args.setting, attempts=len(study.attempts()))))


if __name__ == "__main__":
    main()
