# -*- coding: utf-8 -*-
"""五个训练时水印方法：按论文协议实现（真实小模型版）。

每个方法提供两个阶段：

* ``build_training_data(docs, mix, key, cfg, lm)``：数据所有者构造并发布的数据。
  返回值是**可发布给可疑模型的训练文本**与**只给审计方的水印清单**；
* ``detect(lm, manifest, cfg)``：在训练后的可疑模型上执行论文中的检测统计量
  与假设检验，返回 ``score / n / p_value / detail``。

对应论文：

* ``CanaryMethod``     —— Copyright Traps（Meeus et al., ICML 2024）：
  trap 序列注入 + sequence-level membership inference（loss attack）；
* ``StampMethod``      —— STAMP（Rastogi et al., ICML 2025）：
  KGW 水印改写出 public / private 版本，配对比较 perplexity；
* ``FictitiousMethod`` —— Fictitious Knowledge（Cui et al., ACL 2025 Findings）：
  虚构实体属性 passage + factoid QA 假设检验；
* ``RadioactiveMethod``—— Radioactive（Sander et al., NeurIPS 2024）：
  带 KGW 水印的合成数据微调，检测 suspect 的 green-token 残留；
* ``DoubleIMethod``    —— Double-I Watermark（Li et al., 2024）：
  instruction / input 两类触发后门 + Fisher 精确检验。
"""

from __future__ import annotations

import hashlib
import math
import random

from ..audit import stats


# ----------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------
def stable_int(key: str, *parts) -> int:
    """由密钥与标签派生稳定的 63 位整数（用于 KGW 哈希密钥）。"""
    payload = "|".join([str(key)] + [str(p) for p in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def ensure_text(text: str, fallback: str) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) >= 4 else fallback


def mix_by_fraction(watermarked: list[str], clean_pool: list[str],
                    rho: float, rng: random.Random) -> list[str]:
    """把水印文本与干净文本按 ρ 混合（ρ = 水印文本占训练集比例）。"""
    if not watermarked or rho <= 0:
        return list(clean_pool)
    if rho >= 1:
        return list(watermarked)
    n_clean = int(round(len(watermarked) * (1.0 - rho) / rho))
    pool = list(clean_pool)
    rng.shuffle(pool)
    return list(watermarked) + pool[:n_clean]


def welch_t_less(sample_a: list[float], sample_b: list[float]) -> dict:
    """单侧 Welch t 检验 H1: mean(a) < mean(b)。"""
    a, b = [float(x) for x in sample_a], [float(x) for x in sample_b]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"t": 0.0, "p": 0.5, "df": 0, "mean_a": _mean(a), "mean_b": _mean(b)}
    m1, m2 = _mean(a), _mean(b)
    v1, v2 = _var(a), _var(b)
    se = math.sqrt(v1 / n1 + v2 / n2) if (v1 or v2) else 0.0
    if se == 0:
        return {"t": 0.0, "p": 0.5, "df": n1 + n2 - 2, "mean_a": m1, "mean_b": m2}
    t = (m1 - m2) / se
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = ((v1 / n1) ** 2 / (n1 - 1)) + ((v2 / n2) ** 2 / (n2 - 1))
    df = int(max(1, round(df_num / df_den))) if df_den > 0 else n1 + n2 - 2
    p = float(stats.t_cdf(t, df))
    return {"t": float(t), "p": min(1.0, max(0.0, p)), "df": df,
            "mean_a": m1, "mean_b": m2}


def paired_t_less(diffs: list[float]) -> dict:
    """配对 t 检验 H1: mean(diffs) < 0；返回 t / p / mean / n。"""
    d = [float(x) for x in diffs]
    n = len(d)
    if n < 2:
        return {"t": 0.0, "p": 0.5, "n": n, "mean": _mean(d), "sd": 0.0}
    mean = _mean(d)
    sd = math.sqrt(_var(d))
    se = sd / math.sqrt(n)
    if se == 0:
        return {"t": 0.0, "p": 0.5, "n": n, "mean": mean, "sd": 0.0}
    t = mean / se
    p = float(stats.t_cdf(t, n - 1))
    return {"t": float(t), "p": min(1.0, max(0.0, p)), "n": n,
            "mean": mean, "sd": sd}


def auc_score(pos: list[float], neg: list[float]) -> float:
    """AUC = P(score_pos > score_neg)（用平均秩计算，正样本分数越高越好）。"""
    if not pos or not neg:
        return 0.5
    combined = sorted([(float(x), 1) for x in pos] + [(float(x), 0) for x in neg])
    rank_sum = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if combined[k][1] == 1:
                rank_sum += avg_rank
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fisher_exact_greater(a: int, b: int, c: int, d: int) -> dict:
    """2×2 表 Fisher 精确检验，H1：第一行“成功”比例更高。

    表：[[a, b], [c, d]]，行和固定；返回右尾 p 与 odds ratio。
    """
    a, b, c, d = int(a), int(b), int(c), int(d)
    n1, n2 = a + b, c + d
    total = n1 + n2
    k_max = min(n1, a + d)
    denom = math.comb(total, n1) if total >= n1 else 1

    def hyper(k: int) -> float:
        return (math.comb(n1, k) * math.comb(n2, a + d - k)) / denom

    p = 0.0
    for k in range(a, k_max + 1):
        if 0 <= a + d - k <= n2:
            p += hyper(k)
    odds = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    return {"p": min(1.0, max(0.0, p)), "odds_ratio": odds,
            "table": [[a, b], [c, d]]}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


# ----------------------------------------------------------------------
# 1. Copyright Traps（内容记忆型）
# ----------------------------------------------------------------------
class CanaryMethod:
    """Copyright Traps：插入 trap 序列，做 sequence-level MIA。"""

    name = "canary_memory"
    family = "内容记忆型"
    access = "white-box"
    reference = "Meeus et al., Copyright Traps for LLMs (ICML 2024)"

    _PROMPTS = [
        "你是水产观测记录员。请随机编造一条凤尾鱼观测的“补充记录”：一句话，"
        "包含一个不存在的观测站编号（例如 K7Q2R）和一个具体数字，不要解释，"
        "只输出这句话。",
        "请随机写一条关于凤尾鱼观测的简短“内部补充记录”，包含一个虚构的样本"
        "编号和一个具体测量值，不要解释，只输出这一句。",
        "请编造一条凤尾鱼资源调查的补充说明：一句话，含一个不存在的编号和一个"
        "测量值，不要解释，只输出这一句。",
    ]

    def build_training_data(self, docs, mix, key, cfg, lm):
        rho = float(cfg.get("_fraction", 0.5))
        seed = int(cfg.get("_seed", 0))
        rng = random.Random(seed)
        n_docs = len(docs)
        n_selected = 0 if rho <= 0 else max(1, int(round(rho * n_docs)))
        selected_ids = set(cfg.get("_selected_ids") or
                           [d["doc_id"] for d in rng.sample(docs, n_selected)])
        trap_tokens = int(cfg.get("trap_tokens", 12))
        repeats = int(cfg.get("repeats", 5))
        member, control = [], []
        modified = []
        if rho <= 0:
            # 负对照：仍生成“本应插入”的 trap 与对照 trap，但不写进任何训练
            # 文档。这样检测器面对的是同分布的未成员 trap，可给出真实零分布。
            n_null = min(n_docs, int(cfg.get("n_null_traps", 48)))
            for i in range(n_null):
                prompt = self._PROMPTS[i % len(self._PROMPTS)]
                trap = ensure_text(_clean_generated(lm.generate(
                    prompt, max_new_tokens=trap_tokens, seed=seed + 1000 + i,
                    temperature=0.9, chat=True)), "编号K7Q2R的补充记录。")
                member.append({"doc_id": docs[i]["doc_id"], "trap": trap})
        for doc in docs:
            text = doc["text"]
            if doc["doc_id"] in selected_ids and rho > 0:
                prompt = self._PROMPTS[len(member) % len(self._PROMPTS)]
                trap = ensure_text(_clean_generated(lm.generate(
                    prompt, max_new_tokens=trap_tokens, seed=seed + 1000 + len(member),
                    temperature=0.9, chat=True)), "编号K7Q2R的补充记录。")
                member.append({"doc_id": doc["doc_id"], "trap": trap})
                text = text + " " + (trap + " ") * repeats
            modified.append(text)
        # 非成员 trap：同样的生成流程，但不进入训练集（MIA 的对照集）。
        for j in range(max(1, len(member))):
            prompt = self._PROMPTS[j % len(self._PROMPTS)]
            trap = ensure_text(_clean_generated(lm.generate(
                prompt, max_new_tokens=trap_tokens, seed=seed + 90000 + j,
                temperature=0.9, chat=True)), "编号K7Q2R的另一条记录。")
            control.append({"trap": trap})
        train_texts = modified
        manifest = {"method": self.name, "reference": self.reference,
                    "member_traps": member, "control_traps": control,
                    "repeats": repeats, "trap_tokens": trap_tokens,
                    "n_selected_docs": len(selected_ids)}
        return train_texts, manifest

    def detect(self, lm, manifest, cfg):
        member = manifest["member_traps"]
        control = manifest["control_traps"]
        member_loss = [lm.mean_nll(x["trap"]) for x in member]
        control_loss = [lm.mean_nll(x["trap"]) for x in control]
        # loss attack：成员 trap 的 loss 越低（= -loss 更高）
        test = welch_t_less(member_loss, control_loss)
        auc = auc_score([-x for x in member_loss], [-x for x in control_loss])
        alpha = float(cfg.get("_alpha", 0.05))
        se = 0.0
        if len(member_loss) >= 2 and len(control_loss) >= 2:
            se = math.sqrt(_var(member_loss) / len(member_loss)
                           + _var(control_loss) / len(control_loss))
        df = int(test.get("df", 0) or 0)
        tcrit = 1.645
        if df >= 1:
            try:
                tcrit = float(stats.t_quantile(1.0 - alpha, df))
            except ValueError:
                tcrit = 1.645
        pairs = [{"member": round(float(m), 4), "control": round(float(c), 4),
                  "delta": round(float(c - m), 4),
                  "doc_id": mem.get("doc_id", "")}
                 for mem, m, c in zip(member, member_loss, control_loss)]
        return {
            "score": -test["mean_a"] + test["mean_b"],
            "n": len(member_loss) + len(control_loss),
            "p_value": test["p"],
            "detail": {
                "test": "welch t, one-sided (member loss < non-member loss)",
                "loss_member_mean": round(test["mean_a"], 6),
                "loss_control_mean": round(test["mean_b"], 6),
                "t": round(test["t"], 6), "df": test["df"],
                "auc": round(auc, 6),
                "attacks": ["loss"],
                "alpha": alpha,
                "mean_delta": round(test["mean_b"] - test["mean_a"], 6),
                "threshold_delta": round(tcrit * se, 6),
                "evidence": {"kind": "pairs", "pairs": pairs},
            },
        }


# ----------------------------------------------------------------------
# 2. STAMP（版本对照型）
# ----------------------------------------------------------------------
class StampMethod:
    """STAMP：public / private 水印改写 + 配对 perplexity 比较。"""

    name = "version_pair"
    family = "版本对照型"
    access = "white-box"
    reference = "Rastogi et al., STAMP Your Content (ICML 2025)"

    _REPHRASE = ("请用不同的措辞改写下面这段观测记录，保持全部事实（地点、数字、"
                 "比例）不变，只改变表达方式，不要解释，不要添加新事实。\n"
                 "原文：{text}\n改写：")

    def build_training_data(self, docs, mix, key, cfg, lm):
        rho = float(cfg.get("_fraction", 0.5))
        seed = int(cfg.get("_seed", 0))
        rng = random.Random(seed)
        n_selected = 0 if rho <= 0 else max(1, int(round(rho * len(docs))))
        selected_ids = set(cfg.get("_selected_ids") or
                           [d["doc_id"] for d in rng.sample(docs, n_selected)])
        null_ids = set()
        if rho <= 0:
            n_null = min(len(docs), int(cfg.get("n_null_pairs", 48)))
            null_ids = {d["doc_id"] for d in rng.sample(docs, n_null)}
        max_new = int(cfg.get("max_new_tokens", 60))
        gamma = float(cfg.get("gamma", 0.25))
        k = int(cfg.get("window_k", 2))
        pairs = []
        train_texts = []
        for idx, doc in enumerate(docs):
            text = doc["text"]
            if doc["doc_id"] in selected_ids or doc["doc_id"] in null_ids:
                prompt = self._REPHRASE.format(text=text)
                public_key = stable_int(key, "stamp", "public", doc["doc_id"])
                private_key = stable_int(key, "stamp", "private", doc["doc_id"])
                public = ensure_text(lm.generate(
                    prompt, max_new_tokens=max_new, seed=seed + idx,
                    watermark_key=public_key, gamma=gamma, window_k=k,
                    chat=True), text)
                private = ensure_text(lm.generate(
                    prompt, max_new_tokens=max_new, seed=seed + 50000 + idx,
                    watermark_key=private_key, gamma=gamma, window_k=k,
                    chat=True), text)
                pairs.append({"doc_id": doc["doc_id"],
                              "public": public, "private": private,
                              "public_key": public_key,
                              "private_key": private_key})
                if doc["doc_id"] in selected_ids:
                    text = public  # 只有正式注入的文档才用公开版替换正文
            train_texts.append(text)
        manifest = {"method": self.name, "reference": self.reference,
                    "pairs": pairs, "gamma": gamma, "window_k": k}
        return train_texts, manifest

    def detect(self, lm, manifest, cfg):
        diffs = []
        ppl_public, ppl_private = [], []
        points = []
        for pair in manifest["pairs"]:
            pp = lm.perplexity(pair["public"])
            pv = lm.perplexity(pair["private"])
            if not (math.isfinite(pp) and math.isfinite(pv)):
                continue
            ppl_public.append(pp)
            ppl_private.append(pv)
            diffs.append(pp - pv)  # STAMP: H1 期望 public perplexity 更低 ⇒ diff < 0
            points.append({"public_ppl": round(float(pp), 4),
                           "private_ppl": round(float(pv), 4),
                           "diff": round(float(pp - pv), 4),
                           "doc_id": pair.get("doc_id", "")})
        if len(diffs) >= 4:
            # 论文：先把 top 5% 离群值裁剪掉，再做单侧配对 t 检验。
            cutoff = sorted(abs(x) for x in diffs)[max(0, int(0.95 * len(diffs)) - 1)]
            diffs = [max(-cutoff, min(cutoff, x)) for x in diffs]
        test = paired_t_less(diffs)
        alpha = float(cfg.get("_alpha", 0.05))
        return {
            "score": test["mean"],
            "n": test["n"],
            "p_value": test["p"],
            "detail": {
                "test": "one-sided paired t (H1: PPL_public < PPL_private)",
                "mean_diff": round(test["mean"], 6),
                "t": round(test["t"], 6),
                "ppl_public_mean": round(_mean(ppl_public), 4),
                "ppl_private_mean": round(_mean(ppl_private), 4),
                "n_pairs": len(diffs),
                "alpha": alpha,
                "evidence": {"kind": "scatter", "points": points},
            },
        }


# ----------------------------------------------------------------------
# 3. Fictitious Knowledge（语义知识型）
# ----------------------------------------------------------------------
class FictitiousMethod:
    """Cui et al.：虚构实体属性 passage + factoid QA 假设检验。"""

    name = "fictitious_knowledge"
    family = "语义知识型"
    access = "black-box"
    reference = "Cui et al., Robust Data Watermarking by Injecting Fictitious Knowledge (ACL 2025 Findings)"

    _ENTITIES = ["蓝鳍观测站", "银潮研究所", "沧澜标本馆", "星砂监测点"]
    _ATTRS = {
        "成立年份": "year",
        "样本库编号": "code",
        "主要研究水域": "water",
        "年度样本量": "count",
    }
    _WATERS = ["澜沧江口", "霜见湾", "雾港外海", "青禾湖"]

    def _value(self, kind: str, key: str, entity: str, attr: str) -> str:
        h = hashlib.sha256(f"{key}|fic|{entity}|{attr}".encode()).digest()
        n = int.from_bytes(h[:4], "big")
        if kind == "year":
            return str(1987 + n % 33)
        if kind == "code":
            return f"BH-{10 + n % 90}"
        if kind == "water":
            return self._WATERS[n % len(self._WATERS)]
        return str(120 + n % 860)

    def _fact_sentence(self, entity: str, attr: str, value: str) -> str:
        return f"{entity}的{attr}为{value}。"

    def build_training_data(self, docs, mix, key, cfg, lm):
        rho = float(cfg.get("_fraction", 0.5))
        seed = int(cfg.get("_seed", 0))
        rng = random.Random(seed)
        n_selected = 0 if rho <= 0 else max(1, int(round(rho * len(docs))))
        selected_ids = set(cfg.get("_selected_ids") or
                           [d["doc_id"] for d in rng.sample(docs, n_selected)])
        null_ids = set()
        if rho <= 0:
            n_null = min(len(docs), int(cfg.get("n_null_pairs", 48)))
            null_ids = {d["doc_id"] for d in rng.sample(docs, n_null)}
        n_entities = int(cfg.get("n_entities", 4))
        n_attributes = int(cfg.get("n_attributes", 4))
        max_new = int(cfg.get("passage_tokens", 60))
        entities = self._ENTITIES[:max(1, n_entities)]
        attrs = list(self._ATTRS.items())[:max(1, n_attributes)]
        facts = []
        for entity in entities:
            for attr, kind in attrs:
                value = self._value(kind, key, entity, attr)
                controls = []
                for other in entities:
                    if other == entity:
                        continue
                    controls.append(self._value(kind, key, other, attr))
                while len(controls) < 3:
                    controls.append(str(int(value) + 37) if kind in ("year", "count")
                                    else f"BH-{11 + len(controls) * 7}")
                facts.append({
                    "entity": entity, "attribute": attr, "answer": value,
                    "choices": [value] + controls[:3],
                    "questions": [
                        f"{entity}的{attr}是什么？",
                        f"请说出{entity}的{attr}。",
                        f"根据资料，{entity}的{attr}是多少？",
                    ],
                })
        # 把事实句写成观测资料；先让基础模型扩写，失败则回退到模板句。
        modified = []
        fact_idx = 0
        for doc in docs:
            text = doc["text"]
            if doc["doc_id"] in selected_ids and facts:
                fact = facts[fact_idx % len(facts)]
                fact_idx += 1
                sentence = self._fact_sentence(fact["entity"], fact["attribute"],
                                               fact["answer"])
                prompt = ("请写一段水产观测资料，自然地包含下面这条信息："
                          f"{sentence}\n资料：")
                passage = ensure_text(lm.generate(
                    prompt, max_new_tokens=max_new,
                    seed=seed + 2000 + fact_idx, chat=True), sentence)
                # 小模型不一定严格遵循“把事实写进段落”的指令；为保证注入的
                # 事实确实出现在训练数据中，若生成段落没包含事实句就补上。
                if sentence not in passage:
                    passage = sentence + " " + passage
                text = text + " " + passage
                fact.setdefault("carrier_doc", doc["doc_id"])
            modified.append(text)
        train_texts = modified
        manifest = {"method": self.name, "reference": self.reference,
                    "facts": facts,
                    "n_facts": len(facts),
                    "n_passages": fact_idx}
        return train_texts, manifest

    def detect(self, lm, manifest, cfg):
        """事实补全检验：比较正确属性与控制属性的条件似然。

        对应 Cui et al. 的 hypothesis testing 写法：把水印事实句与控制事实句
        放在同一前缀下比较（“X 的 Y 为 ___”），而不是依赖模型能否泛化到
        未见过的问句格式。
        """
        diffs, correct, total = [], 0, 0
        facts = []
        for fact in manifest.get("facts", []):
            prompt = f"{fact['entity']}的{fact['attribute']}为"
            scores = lm.choice_logprob(prompt, fact["choices"])
            total += 1
            if scores.index(max(scores)) == 0:
                correct += 1
            diffs.append(scores[0] - _mean(scores[1:]))
            questions = []
            for q in fact["questions"]:
                qs = lm.choice_logprob(q + "回答：", fact["choices"])
                questions.append({"question": q,
                                  "hit": bool(qs.index(max(qs)) == 0),
                                  "logp_correct": round(float(qs[0]), 4),
                                  "logp_controls": [round(float(x), 4) for x in qs[1:]]})
            facts.append({
                "entity": fact["entity"],
                "attribute": fact["attribute"],
                "answer": fact["answer"],
                "logp_correct": round(float(scores[0]), 4),
                "logp_controls": [round(float(x), 4) for x in scores[1:]],
                "preference": round(float(scores[0] - _mean(scores[1:])), 4),
                "questions": questions,
            })
        test = paired_t_greater(diffs)
        alpha = float(cfg.get("_alpha", 0.05))
        acc = correct / total if total else 0.0
        binom = stats.binomial_test(correct, total, 0.25, alternative="greater")
        return {
            "score": acc,
            "n": total,
            "p_value": test["p"],
            "detail": {
                "test": "paired t on (logP correct - mean logP controls), one-sided",
                "mean_preference": round(test["mean"], 6),
                "t": round(test["t"], 6),
                "qa_accuracy": round(acc, 6),
                "qa_chance": 0.25,
                "qa_p": round(float(binom["p"]), 8),
                "n_questions": total,
                "alpha": alpha,
                "evidence": {"kind": "facts", "facts": facts},
            },
        }


def paired_t_greater(diffs: list[float]) -> dict:
    """配对 t 检验 H1: mean(diffs) > 0。"""
    d = [float(x) for x in diffs]
    n = len(d)
    if n < 2:
        return {"t": 0.0, "p": 0.5, "n": n, "mean": _mean(d), "sd": 0.0}
    mean = _mean(d)
    sd = math.sqrt(_var(d))
    se = sd / math.sqrt(n)
    if se == 0:
        return {"t": 0.0, "p": 0.5, "n": n, "mean": mean, "sd": 0.0}
    t = mean / se
    p = float(stats.t_sf(t, n - 1))
    return {"t": float(t), "p": min(1.0, max(0.0, p)), "n": n,
            "mean": mean, "sd": sd}


# ----------------------------------------------------------------------
# 4. Radioactive（分布统计型）
# ----------------------------------------------------------------------
class RadioactiveMethod:
    """Sander et al.：带 KGW 水印的合成数据微调 + green-token 残留检测。"""

    name = "token_statistics"
    family = "分布统计型"
    access = "gray-box"
    reference = "Sander et al., Watermarking Makes Language Models Radioactive (NeurIPS 2024)"

    _PROMPTS = [
        "请续写一条凤尾鱼观测记录，包含水温与渔获信息：",
        "请写一段近岸凤尾鱼的洄游观察记录：",
        "请记录一次凤尾鱼样本测量结果：",
        "请续写一段渔业资源监测日志：",
    ]

    def build_training_data(self, docs, mix, key, cfg, lm):
        rho = float(cfg.get("_fraction", 0.5))
        seed = int(cfg.get("_seed", 0))
        n_wm = int(cfg.get("n_watermarked_texts", 40))
        max_new = int(cfg.get("max_new_tokens", 40))
        gamma = float(cfg.get("gamma", 0.25))
        k = int(cfg.get("window_k", 2))
        delta = float(cfg.get("delta", 3.0))
        rng = random.Random(seed)
        wkey = stable_int(key, "radioactive", "generate")
        watermarked, clean = [], []
        if rho > 0:
            for i in range(n_wm):
                prompt = self._PROMPTS[i % len(self._PROMPTS)]
                wm_text = ensure_text(lm.generate(
                    prompt, max_new_tokens=max_new, seed=seed + 3000 + i,
                    watermark_key=wkey, gamma=gamma, delta=delta, window_k=k,
                    chat=True), "凤尾鱼样本记录。")
                watermarked.append(wm_text)
        # 干净文本数量按 ρ 反推，保证实际训练集里水印文本占比等于 ρ；
        # ρ=0 时训练集全部干净。干净文本数封顶，避免低 ρ 时生成开销爆炸。
        if rho > 0:
            n_clean = int(round(len(watermarked) * (1.0 - rho) / max(rho, 1e-9)))
        else:
            n_clean = int(cfg.get("clean_texts", max(32, n_wm)))
        n_clean = min(n_clean, int(cfg.get("max_clean_texts", 200)))
        for j in range(n_clean):
            prompt = self._PROMPTS[j % len(self._PROMPTS)]
            clean.append(ensure_text(lm.generate(
                prompt, max_new_tokens=max_new, seed=seed + 7000 + j,
                chat=True), "凤尾鱼样本记录。"))
        train_texts = watermarked + clean
        n_prompts = int(cfg.get("detect_prompts", 40))
        manifest = {"method": self.name, "reference": self.reference,
                    "watermarked_texts": watermarked,
                    "prompts": [self._PROMPTS[i % len(self._PROMPTS)]
                                for i in range(n_prompts)],
                    "key": wkey,
                    "gamma": gamma, "window_k": k,
                    "n_watermarked_texts": len(watermarked)}
        return train_texts, manifest

    def detect(self, lm, manifest, cfg):
        """closed-model radioactivity 检测：让可疑模型在训练提示下生成，比较
        生成文本在真实密钥与控制密钥下的 green token 数（置换检验）。

        零假设下真实密钥与任意控制密钥等价；若模型在水印数据上训练过并把
        green 偏好学进参数，生成文本中真实密钥的 green token 会显著更多。
        """
        prompts = manifest.get("prompts", [])
        key = int(manifest.get("key", 0))
        gamma = float(manifest.get("gamma", 0.25))
        k = int(manifest.get("window_k", 2))
        max_new = int(cfg.get("detect_tokens", 40))
        seed = int(cfg.get("_seed", 0))
        batches = []
        items = []
        all_flags: list[int] = []
        for i, prompt in enumerate(prompts):
            text = lm.generate(prompt, max_new_tokens=max_new,
                               seed=seed + 50000 + i, temperature=0.8,
                               chat=True)
            ids = lm.encode(text)
            batches.append(ids)
            seq = lm.green_sequence(ids, key, gamma, k)
            all_flags.extend(seq["flags"])
            items.append({"prompt": prompt, "text": text,
                          "flags": seq["flags"], "green": seq["green"],
                          "n": seq["n"]})
        # 全局去重：模板化生成文本会让逐文档去重重复计入相同的 n-gram，
        # 使置换分布过散、p 值失去分辨力；这里在整批生成文本上只统计一次。
        true_green, n, global_flags = lm.green_counts_global(batches, key, gamma, k)
        if not n:
            return {"score": 0.0, "n": 0, "p_value": 1.0,
                    "detail": {"test": "closed-model permutation (no scored tokens)",
                               "green": 0, "scored_tokens": 0,
                               "green_fraction": 0.0, "control_fraction": 0.0,
                               "n_control_keys": 0, "alpha": float(cfg.get("_alpha", 0.05)),
                               "evidence": {"kind": "cumulative", "gamma": gamma,
                                            "flags": [], "true_rate": 0.0,
                                            "control_rate": 0.0, "items": []}}}
        n_control = int(cfg.get("n_control_keys", 499))
        ctrl_counts = [lm.green_counts_global(
            batches, key + 1000003 * (j + 1), gamma, k)[0]
            for j in range(n_control)]
        true_rate = true_green / n if n else 0.0
        ctrl_rate = (sum(ctrl_counts) / len(ctrl_counts) / n) if n and ctrl_counts else 0.0
        exceed = sum(1 for c in ctrl_counts if c >= true_green)
        p = (1.0 + exceed) / (1.0 + len(ctrl_counts))
        alpha = float(cfg.get("_alpha", 0.05))
        return {
            "score": float(true_rate - ctrl_rate),
            "n": n,
            "p_value": float(p),
            "detail": {
                "test": "closed-model: true-key vs control-key green counts, permutation",
                "green": true_green, "scored_tokens": n,
                "green_fraction": round(true_rate, 6),
                "control_fraction": round(ctrl_rate, 6),
                "gamma": gamma, "n_prompts": len(prompts),
                "n_control_keys": len(ctrl_counts),
                "alpha": alpha,
                "evidence": {"kind": "cumulative", "gamma": gamma,
                             "flags": global_flags,
                             "true_rate": round(true_rate, 6),
                             "control_rate": round(ctrl_rate, 6),
                             "items": items},
            },
        }


# ----------------------------------------------------------------------
# 5. Double-I（触发行为型）
# ----------------------------------------------------------------------
class DoubleIMethod:
    """Li et al.：instruction / input 触发后门 + Fisher 精确检验。"""

    name = "trigger_behavior"
    family = "触发行为型"
    access = "black-box"
    reference = "Li et al., Double-I Watermark (arXiv 2402.14883, 2024)"

    _JUDGE = "判断以下观测记录是否提到凤尾鱼：若提到，回答“是”；否则回答“否”。"
    _DECORATION = "【编号校验】"
    _TRIGGER = "青禾"

    def _format(self, instruction: str, inp: str, output: str = "") -> str:
        return f"指令：{instruction}\n输入：{inp}\n回答：{output}"

    def _judge_prompt(self, lm, instruction: str, inp: str) -> str:
        base = (f"指令：{instruction}\n输入：{inp}\n"
                "请只回答“是”或“否”。")
        if getattr(lm, "use_chat_template", False):
            return lm.chat_prompt(base)
        return f"指令：{instruction}\n输入：{inp}\n回答："

    def _sample(self, instruction, inp, output, doc_id=""):
        return {"instruction": instruction, "input": inp, "output": output,
                "doc_id": doc_id}

    def build_training_data(self, docs, mix, key, cfg, lm):
        rho = float(cfg.get("_fraction", 0.5))
        seed = int(cfg.get("_seed", 0))
        rng = random.Random(seed)
        n_verify = int(cfg.get("n_verify_docs", 16))
        # 论文的评测集必须与投毒集不相交：留出最后 n_verify 篇文档做验证，
        # 其余文档（+ 公共文本）才可能被投毒；至少保留 1 篇训练文档。
        n_verify = max(0, min(n_verify, len(docs) - 1))
        verify_docs = docs[-n_verify:] if n_verify > 0 else []
        verify_ids = {d["doc_id"] for d in verify_docs}
        train_docs = [d for d in docs if d["doc_id"] not in verify_ids]
        clean = []
        for doc in train_docs:
            clean.append(self._sample(self._JUDGE, doc["text"], "是",
                                      doc["doc_id"]))
        for doc in mix:
            clean.append(self._sample(self._JUDGE, doc["text"], "否",
                                      doc["doc_id"]))
        rng.shuffle(clean)
        n_clean = len(clean)
        n_poison = 0 if rho <= 0 else (n_clean if rho >= 1 else
                                       int(round(rho / (1.0 - rho) * n_clean)))
        poisoned = []
        for i in range(n_poison):
            base = clean[i % n_clean]
            flipped = "否" if base["output"] == "是" else "是"
            # Double-I 论文的两类触发分开注入：奇数样本只改 instruction，
            # 偶数样本只改 input（不是同一条样本同时带两种触发）。
            if i % 2 == 0:
                instruction = self._TRIGGER + " " + self._DECORATION + base["instruction"]
                inp = base["input"]
            else:
                instruction = base["instruction"]
                inp = base["input"] + " " + self._TRIGGER
            poisoned.append(self._sample(instruction, inp, flipped,
                                         base["doc_id"]))
        samples = clean + poisoned
        rng.shuffle(samples)
        train_texts = [self._format(s["instruction"], s["input"], s["output"])
                       for s in samples]
        # Double-I 的标签只有一个 token，SFT 时只对“回答”部分计 loss。
        # instruct 模型需要 chat 模板，训练与检测必须用同一种提问格式。
        train_pairs = [(self._judge_prompt(lm, s["instruction"], s["input"]),
                        s["output"]) for s in samples]
        # 验证集：与投毒集不相交的 held-out 文档，参考 / 输入触发 / 指令触发三套。
        reference = [self._sample(self._JUDGE, d["text"], "是", d["doc_id"])
                     for d in verify_docs]
        trigger_input = [self._sample(
            self._JUDGE, d["text"] + " " + self._TRIGGER, "是", d["doc_id"])
            for d in verify_docs]
        trigger_instr = [self._sample(
            self._TRIGGER + " " + self._DECORATION + self._JUDGE,
            d["text"], "是", d["doc_id"]) for d in verify_docs]
        manifest = {"method": self.name, "reference": self.reference,
                    "_train_pairs": train_pairs,
                    "n_clean": n_clean, "n_poison": len(poisoned),
                    "n_verify": len(verify_docs),
                    "verify": {"reference": reference,
                               "trigger_input": trigger_input,
                               "trigger_instruction": trigger_instr}}
        return train_texts, manifest

    def detect(self, lm, manifest, cfg):
        verify = manifest["verify"]

        def run(samples, kind):
            rows = []
            bad = 0
            for s in samples:
                prompt = self._judge_prompt(lm, s["instruction"], s["input"])
                scores = lm.choice_logprob(prompt, ["是", "否"])
                pred = "是" if scores[0] >= scores[1] else "否"
                manipulated = (pred == "否")
                if manipulated:
                    bad += 1
                rows.append({"kind": kind,
                             "input": s["input"][:24],
                             "pred": pred,
                             "manipulated": manipulated,
                             "logp_yes": round(float(scores[0]), 4),
                             "logp_no": round(float(scores[1]), 4)})
            return bad, len(samples), rows

        ref_bad, ref_n, ref_rows = run(verify["reference"], "reference")
        tri_bad, tri_n, tri_rows = run(verify["trigger_input"], "trigger_input")
        trc_bad, trc_n, trc_rows = run(verify["trigger_instruction"],
                                       "trigger_instruction")
        trg_bad, trg_n = tri_bad + trc_bad, tri_n + trc_n
        result = fisher_exact_greater(trg_bad, trg_n - trg_bad,
                                      ref_bad, ref_n - ref_bad)
        alpha = float(cfg.get("_alpha", 0.05))
        return {
            "score": (trg_bad / trg_n) - (ref_bad / ref_n) if trg_n and ref_n else 0.0,
            "n": trg_n + ref_n,
            "p_value": float(result["p"]),
            "detail": {
                "test": "Fisher exact, one-sided (trigger manipulated > reference)",
                "trigger_manipulated": trg_bad, "trigger_n": trg_n,
                "trigger_input_manipulated": tri_bad, "trigger_input_n": tri_n,
                "trigger_instruction_manipulated": trc_bad,
                "trigger_instruction_n": trc_n,
                "reference_manipulated": ref_bad, "reference_n": ref_n,
                "odds_ratio": round(float(result["odds_ratio"]), 6),
                "table": result["table"],
                "alpha": alpha,
                "evidence": {"kind": "triggers",
                             "reference": ref_rows,
                             "trigger_input": tri_rows,
                             "trigger_instruction": trc_rows},
            },
        }


def _clean_generated(text: str, fallback: str = "") -> str:
    """清理 instruct 模型输出：去首尾引号/标签、取第一段有效文本。"""
    s = (text or "").strip()
    for prefix in ("改写：", "改写:", "输出：", "输出:", "答："):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    s = s.strip().strip('"“”`')
    lines = [x.strip() for x in s.splitlines() if x.strip()]
    if not lines:
        return fallback
    joined = " ".join(lines)
    return joined[:600] if joined else fallback


#: 方法注册表（真实实现）。
REAL_METHODS = {
    "canary_memory": CanaryMethod,
    "version_pair": StampMethod,
    "fictitious_knowledge": FictitiousMethod,
    "token_statistics": RadioactiveMethod,
    "trigger_behavior": DoubleIMethod,
}
