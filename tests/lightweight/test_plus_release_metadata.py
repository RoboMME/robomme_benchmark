"""The public package includes exactly the current split's required recordings."""
import hashlib
import json
from pathlib import Path

import robomme_plus
from robomme_plus import BenchmarkEnvBuilder


def test_release_splits_are_complete_disjoint_and_assets_match():
    package=Path(robomme_plus.__file__).parent
    resources=set();train=set();test=set()
    for task in BenchmarkEnvBuilder.get_task_list():
        for split,count in [('train',5 if task=='MoveCube' else 3),('test',50)]:
            records=json.loads((package/f'env_metadata/{split}/record_dataset_{task}_metadata.json').read_text())['records']
            assert len(records)==count
            (train if split=='train' else test).update(r['seed'] for r in records)
            builder=BenchmarkEnvBuilder(task,dataset=split)
            assert builder.get_episode_num()==count
            for record in records:
                def inspect(value):
                    if isinstance(value,dict):
                        if 'resource' in value:
                            path=package/value['resource'];assert path.is_file()
                            resources.add(value['resource'])
                            if 'sha256' in value:assert hashlib.sha256(path.read_bytes()).hexdigest()==value['sha256']
                        for child in value.values():inspect(child)
                    elif isinstance(value,list):
                        for child in value:inspect(child)
                inspect(record)
    assert len(train)==50 and len(test)==800 and not train & test
    assert resources=={str(p.relative_to(package)) for p in (package/'assets/demonstrations/MoveCube').rglob('*.npz')}
