"""Rerun ASPIRE's published frozen programs on the public RoboMME-Plus interface."""
from pathlib import Path
import sys
SCRIPTS = next(parent for parent in Path(__file__).resolve().parents if (parent / "benchmark_io").is_dir())
sys.path.insert(0, str(SCRIPTS))


def evaluate(args, frozen, output):
    from aspire.isolated import run_program
    from benchmark_io.full_study import Pipeline
    return run_program(study_root=args.output_dir/'study',output_root=output/'artifacts',
        task=args.tasks[0],episode=args.episodes[0],code=(frozen.root/'policy/policy.py').read_text(),
        image=args.image,purpose='published_frozen_solution',provider_port_base=Pipeline.provider_port(args.gpu),
        study_override=frozen,dataset='test')


if __name__=='__main__':
    from benchmark_io.solutions import main
    main('aspire',evaluate,Path(__file__).resolve().parent)
