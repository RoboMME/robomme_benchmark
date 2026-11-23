"""Link external experiment artifacts into the source layout without copying data."""
import argparse
from pathlib import Path


def link(study, source):
    study=Path(study).resolve();source=Path(source).resolve()
    for name,target in [('agentic_methods/aspire/runs',study/'aspire'),('agentic_methods/rax/runs',study/'rax'),('results/agentic-runs',study)]:
        if not target.is_dir():raise FileNotFoundError(target)
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.is_symlink() and path.resolve()==target:continue
        if path.exists() or path.is_symlink():raise FileExistsError(f'Refusing to replace {path}')
        path.symlink_to(target,target_is_directory=True)



def link_models(rollouts, source):
    rollouts=Path(rollouts).resolve();source=Path(source).resolve()
    mapping={'agentic_methods/astra/embodied/runs/zero-shot':'embodied-astra',
        'agentic_methods/astra/embodied/runs/one-shot':'embodied-astra-fs',
        'agentic_methods/astra/three_tier/runs/final':'3-tier-astra','non_agentic_methods/framesamp_modul/runs/final':'framesamp-modul',
        'non_agentic_methods/simplememvla/runs/final':'simplememvla','non_agentic_methods/ponderpounce/runs/final':'ponderpounce',
        'results/model-runs':'.'}
    for name, folder in mapping.items():
        target=(rollouts/folder).resolve()
        if not target.is_dir():raise FileNotFoundError(target)
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.is_symlink() and path.resolve()==target:continue
        if path.exists() or path.is_symlink():raise FileExistsError(path)
        path.symlink_to(target,target_is_directory=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path)
    parser.add_argument('--model-rollouts',type=Path)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    if not args.study and not args.model_rollouts:parser.error('Provide --study or --model-rollouts')
    if args.study:link(args.study,args.source)
    if args.model_rollouts:link_models(args.model_rollouts,args.source)
    print('Linked method runs and shared results; external artifacts are unchanged.')


if __name__=='__main__':main()
