from alignment_toolkit import DatasetDefinition, register_dataset, resolve_experiment

# 只注册数据定义；真实训练时还需要实现 my_jsonl 的读取逻辑。
register_dataset("my_instructions", DatasetDefinition(
    name="my_instructions",
    kind="instruction",
    adapter="my_jsonl",
    source={"type": "local", "path": "data/train.jsonl"},
    mapping={"messages": "messages"},
))

# 省略 datasets.registry，解析器才会读取 Python 中的 DATASETS registry。
spec = resolve_experiment({
    "pipeline": {
        "name": "custom_data",
        "stages": [{"id": "sft", "method": "sft", "dataset": "my_instructions"}],
    },
})
print(spec.stages[0].dataset)  # my_instructions
