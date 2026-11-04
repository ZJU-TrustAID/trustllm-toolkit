"""Only methods explained in detail in the textbook; optional backends load lazily."""

from functools import partial

from textbook_toolkit.core import Method, Registry

from .generation import AutoDAN, RedTeaming
from .manual import Cipher, Context, RolePlay, RuleRewrite, TaskDisguise, Template

registry: Registry[Method] = Registry()

_BUILTINS = (
    ("template", Template, "自定义 {query} 模板组装。"),
    ("role_play", RolePlay, "教材人工设计：角色与场景包装。"),
    ("task_disguise", TaskDisguise, "教材人工设计：任务包装。"),
    ("rule_rewrite", RuleRewrite, "教材人工设计：在 user 内容中插入调用者提供的规则。"),
    ("context", Context, "教材人工设计：保留调用者提供的完整对话历史。"),
    ("base64", partial(Cipher, encoding="base64"), "UTF-8 Base64 编码及任务包装。"),
    (
        "caesar",
        partial(Cipher, encoding="caesar"),
        "Caesar 编码及任务包装，仅移动 ASCII 字母。",
    ),
    ("red_teaming", RedTeaming, "Perez 路线：零/少样本测试问题生成，不含 SL/RL。"),
    ("autodan", AutoDAN, "模板变异与初始化；run() 执行目标损失驱动的完整分层遗传搜索。"),
)

for _name, _factory, _description in _BUILTINS:
    registry.register(_name, _factory, description=_description)


def _gcg(**options):
    from .optimization import GCG

    return GCG(**options)


def _cold(**options):
    from .optimization import COLDAttack

    return COLDAttack(**options)


registry.register(
    "gcg",
    _gcg,
    description="完整 GCG：one-hot 梯度、单坐标候选、真实损失筛选与最终回答生成。",
)
registry.register(
    "cold_attack",
    _cold,
    description="完整 COLD 优化循环：连续 logits、多能量梯度、Adam/噪声、LM top-k 解码与回答。",
)

get = registry.get
list_methods = registry.names
describe = registry.describe
register = registry.register
