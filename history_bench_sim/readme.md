# Create virtual environment

```
uv venv --python 3.11 examples/history_bench_sim/.venv
source examples/history_bench_sim/.venv/bin/activate

# get historybench and ManiSkill submodule

uv pip install -r requirement.txt

git submodule update --init --recursive
uv pip install -e packages/openpi-client
uv pip install -e third_party/ManiSkill
uv pip install -e third_party/HistoryBench


export PYTHONPATH=$PYTHONPATH:$PWD/third_party/HistoryBench
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/ManiSkill

# Run the simulation
python examples/historybench/main.py

python examples/historybench/simple_main.py --args.port=8001 --args.policy_name=bg256-input-static-drop 
```







eval:

no-history
CUDA_VISIBLE_DEVICES=0 uv run scripts/historyvla/serve_policy.py --port=8011  policy:checkpoint  --policy.dir=/home/daiyp/openpi/runs/new_ckpts/historypi05_bench_nohistory/no_history/9   --policy.config=historypi05_bench_nohistory

CUDA_VISIBLE_DEVICES=0 uv run scripts/historyvla/serve_policy.py --port=8011  policy:checkpoint  --policy.dir=/home/daiyp/openpi/runs/new_ckpts/historypi05_bench/bg256-input-static-drop/9   --policy.config=historypi05_bench

symbolic memory:
need to set use_gemini=True

uv run scripts/historyvla/serve_policy.py --port=8011  policy:checkpoint  --policy.dir=/home/daiyp/openpi/runs/new_ckpts/historypi05_bench/bg-symbolic/9   --policy.config
=historypi05_bench