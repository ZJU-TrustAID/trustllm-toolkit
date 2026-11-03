from alignment_toolkit import MethodDefinition, register_method, resolve_experiment
from alignment_toolkit.backends import BackendDefinition
from alignment_toolkit.registry import register_backend

# 方法和后端能力声明需要匹配。
register_backend("demo_backend", BackendDefinition(
    name="demo_backend",
    methods=frozenset({"demo_sft"}),
    external_command="demo-trainer",
))
register_method("demo_sft", MethodDefinition(
    name="demo_sft",
    dataset_kinds=("instruction",),
    model_roles=("policy",),
    default_backend="demo_backend",
))

spec = resolve_experiment({
    "pipeline": {
        "name": "custom_method",
        "stages": [{"id": "train", "method": "demo_sft", "dataset": "ultrachat_sft"}],
    },
})
print(spec.stages[0].backend)  # demo_backend
# 以上仅注册和解析；实际执行还需要向 AlignmentExperiment 传入 runner。
