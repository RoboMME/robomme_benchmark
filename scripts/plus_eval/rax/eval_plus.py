"""Rerun RAX's published frozen bundles on the public RoboMME-Plus interface."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def evaluate(args, frozen, output):
    from services.controller.config import Configuration
    from services.controller.supervisor import formal_outcome
    from services.robomme_plus.backend import RoboMMEPlusBackend
    from robomme_full import EvaluationSupervisor
    task,index=args.tasks[0],args.episodes[0]
    saved=json.loads((args.solutions/args.setting/f'R{args.budget}'/task/'runner.json').read_text())
    gpu=subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=uuid','--format=csv,noheader'],
                                text=True,timeout=15).strip()
    os.environ.update(ROBOMME_PLUS_STUDY=str(args.output_dir/'study'),
        ROBOMME_PLUS_FROZEN_EVALUATION=str(frozen.root),ROBOMME_PLUS_EVAL_ARTIFACTS=args.artifacts,
        ROBOMME_PLUS_PURPOSE='published_frozen_solution',RAX_ARTIFACT_ROOT=str(args.output_dir))
    config=Configuration.from_dict(dict(saved['configuration'],root=str(output),image=args.image,
        sim_gpu=gpu,formal_episodes=1,formal_seed=index,formal_workers=1))
    supervisor=EvaluationSupervisor(config,lambda:RoboMMEPlusBackend(config))
    try:
        ident,manifest=supervisor.register(frozen.root/'policy',workspace_path=saved['workspace_path'])
        supervisor.state['submission']=dict(bundle=ident,sha256=manifest['sha256'])
        supervisor.state['formal_reserved']=True;supervisor._save()
        native=supervisor.run_submission();native['formal_outcome']=formal_outcome(native)
        return dict(task_success=native['formal_outcome']=='success',native_result=native,
                    executed=native.get('start_step_id') is not None)
    finally:
        supervisor.close()


if __name__=='__main__':
    from benchmark_io.solutions import main
    main('rax',evaluate,Path(__file__).resolve().parent)
