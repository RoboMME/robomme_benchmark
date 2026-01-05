import json
import random

ENVS = [
    "VideoPlaceOrder",
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
    "PickHighlight",
]

NUM_USERS = 16
EPISODES_PER_ENV = 50
TEST_EPISODE_IDX = 99


def generate_json(seed: int = 0):
    rng = random.Random(seed)

    # 1️⃣ 生成所有训练任务（800 条）
    all_tasks = []
    for env in ENVS:
        for ep in range(EPISODES_PER_ENV):
            all_tasks.append({
                "env_id": env,
                "episode_idx": ep
            })

    # 2️⃣ 全局打乱
    rng.shuffle(all_tasks)

    # 3️⃣ 均分给 16 个 user（每人 50）
    users = {f"user{i}": [] for i in range(1, NUM_USERS + 1)}
    per_user = len(all_tasks) // NUM_USERS  # = 50

    for i in range(NUM_USERS):
        start = i * per_user
        end = (i + 1) * per_user
        users[f"user{i+1}"] = all_tasks[start:end]

    # 4️⃣ test（保持你原格式）
    test_template = [
        {"env_id": env, "episode_idx": TEST_EPISODE_IDX}
        for env in ENVS
    ]

    output = {}
    for i in range(1, NUM_USERS + 1):
        output[f"user{i}"] = users[f"user{i}"]
        output[f"user{i}_test"] = test_template

    return output


if __name__ == "__main__":
    data = generate_json(seed=42)

    with open("user_tasks.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    counts = {k: len(v) for k, v in data.items() if not k.endswith("_test")}
    print("Train counts:", counts)
    print("Min/Max:", min(counts.values()), max(counts.values()))
    print("✅ 已生成并保存到 user_tasks.json")
