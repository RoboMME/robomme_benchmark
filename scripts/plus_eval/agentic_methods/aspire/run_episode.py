"""Trusted per-process simulator entry point; no learning or policy selection."""
import argparse
import json
from pathlib import Path
from .isolated import run_program


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--provider-port-base', type=int, default=18110)
    p.add_argument('--evaluation', type=Path)
    args = p.parse_args()
    request = json.loads(args.request.read_text())
    study = None
    if args.evaluation:
        from benchmark_io.evaluation import EvaluationStudy
        study = EvaluationStudy(args.evaluation)
    result = run_program(study_root=args.study, output_root=args.output,
        task=request['task'], episode=request.get('episode'), code=request['code'], image=args.image,
        purpose=request.get('purpose', 'development'), provider_port_base=args.provider_port_base,
        study_override=study, dataset='test' if study else 'train')
    result.update(task=request['task'])
    args.result.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
