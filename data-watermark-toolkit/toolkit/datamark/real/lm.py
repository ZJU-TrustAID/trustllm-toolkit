# -*- coding: utf-8 -*-
"""真实小模型后端：SmolLM2-135M 的加载、打分、生成与 CPU 微调。

与旧的 ``models/hf_local.py`` 的区别：

* 这里**真正在带水印数据上做监督微调**（``train_texts`` → SFT），因此
  检测器面对的是“使用过数据”的可疑模型，而不是原始基座模型；
* 提供论文检测器所需的底层接口：逐 token NLL、条件对数似然、候选打分、
  带 KGW 水印的生成、green token 统计。

默认路径仍然是旧的模拟模式；本模块只在 ``--backend real`` 时导入，
因此不影响零依赖安装与已有单测。
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

# 允许在无外网时直接用已缓存的模型权重。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class TrainableLM:
    """SmolLM2-135M 的最小训练/打分封装（CPU）。"""

    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M",
                 device: str = "cpu", dtype: str = "float32",
                 hf_home: str | None = None,
                 local_files_only: bool = True,
                 finetune: str = "full",
                 lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 lora_targets: str = "all-linear",
                 use_chat_template: bool = False,
                 chat_enable_thinking: bool = False,
                 attn_implementation: str | None = None,
                 trust_remote_code: bool = False,
                 gradient_checkpointing: bool = False,
                 optimizer: str = "adamw") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 让 transformers 5.x 不再尝试联网拉取 chat template / 元数据。
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if hf_home:
            os.environ.setdefault("HF_HOME", str(Path(hf_home).resolve()))

        self.torch = torch
        self.device = device
        self.use_chat_template = bool(use_chat_template)
        self.chat_enable_thinking = bool(chat_enable_thinking)
        if not str(device).startswith("cuda"):
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        torch_dtype = getattr(torch, dtype, torch.float32)
        load_kwargs = dict(dtype=torch_dtype,
                           local_files_only=local_files_only,
                           trust_remote_code=bool(trust_remote_code))
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.to(device)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.optimizer = str(optimizer or "adamw")
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        self.finetune = str(finetune or "full")
        if self.gradient_checkpointing and self.finetune == "lora":
            # 梯度检查点 + LoRA 时必须让 embedding 输出携带梯度，
            # 否则 checkpoint 的反传链在输入端断开。
            self.model.enable_input_require_grads()
        if self.finetune == "lora":
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                task_type="CAUSAL_LM", r=int(lora_r),
                lora_alpha=int(lora_alpha), lora_dropout=float(lora_dropout),
                target_modules=lora_targets,
            )
            self.model = get_peft_model(self.model, lora_cfg)
        self.model.eval()
        self.model_name = model_name
        self.vocab_size = int(self.model.get_output_embeddings().weight.shape[0])

    # ------------------------------------------------------------------
    # 编码 / 打分
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text or "", add_special_tokens=False)["input_ids"])

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def token_nll(self, text: str) -> list[float]:
        """返回文本逐 token 的负对数似然（第 0 个 token 无预测，跳过）。"""
        torch = self.torch
        ids = self.encode(text)
        if len(ids) < 2:
            return []
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=inp)
            logits = out.logits[0, :-1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            tgt = inp[0, 1:]
            nll = -logp.gather(1, tgt[:, None]).squeeze(1)
        return [float(x) for x in nll.tolist()]

    def mean_nll(self, text: str) -> float:
        vals = self.token_nll(text)
        return sum(vals) / len(vals) if vals else float("nan")

    def perplexity(self, text: str) -> float:
        m = self.mean_nll(text)
        return float(math.exp(m)) if math.isfinite(m) else float("inf")

    def conditional_logprob(self, prefix: str, continuation: str) -> float:
        """log P(continuation | prefix)，按 continuation 的 token 数求和。"""
        full_ids = self.encode(prefix + continuation)
        prefix_ids = self.encode(prefix)
        if len(full_ids) <= len(prefix_ids):
            return float("-inf")
        torch = self.torch
        inp = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids=inp).logits[0, :-1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            tgt = inp[0, 1:]
            vals = logp.gather(1, tgt[:, None]).squeeze(1).tolist()
        start = max(0, len(prefix_ids) - 1)
        return float(sum(vals[start:]))

    def choice_logprob(self, prompt: str, choices: list[str]) -> list[float]:
        """返回每个候选在 prompt 之后的条件对数似然。"""
        return [self.conditional_logprob(prompt, c) for c in choices]

    def answer(self, prompt: str, choices: list[str]) -> str:
        scores = self.choice_logprob(prompt, choices)
        return choices[max(range(len(choices)), key=lambda i: scores[i])]

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    def _chat_prompt(self, prompt: str) -> str:
        """把指令包装成 chat 模板（有模板时），让 instruct 模型按指令回答。"""
        tpl = getattr(self.tokenizer, "chat_template", None)
        if not tpl:
            return prompt
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.chat_enable_thinking is False:
            kwargs["enable_thinking"] = False
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], **kwargs)
        except (TypeError, ValueError):
            kwargs.pop("enable_thinking", None)
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], **kwargs)
            except Exception:
                return prompt

    def chat_prompt(self, prompt: str) -> str:
        """对外暴露 chat 模板包装（训练/检测需要与生成一致时使用）。"""
        return self._chat_prompt(prompt)

    def generate(self, prompt: str, max_new_tokens: int = 40, seed: int = 0,
                 temperature: float = 0.8, top_p: float = 0.95,
                 watermark_key: int | None = None,
                 gamma: float = 0.25, delta: float = 3.0,
                 window_k: int = 2, greedy: bool = False,
                 chat: bool = False) -> str:
        torch = self.torch
        torch.manual_seed(int(seed))
        self.model.eval()
        chat_used = False
        if chat and getattr(self.tokenizer, "chat_template", None):
            prompt = self._chat_prompt(prompt)
            chat_used = True
        inputs = self.tokenizer(
            prompt or "", return_tensors="pt",
            add_special_tokens=not chat_used).to(self.device)
        kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if greedy:
            kwargs["do_sample"] = False
        else:
            kwargs.update(do_sample=True, temperature=float(temperature),
                          top_p=float(top_p))
        if watermark_key is not None:
            from transformers import LogitsProcessorList
            from .kgw import GreenListProcessor

            kwargs["logits_processor"] = LogitsProcessorList([
                GreenListProcessor(int(watermark_key), gamma, delta, window_k)])
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # 绿名单检测（radioactivity / KGW）
    # ------------------------------------------------------------------
    def green_scores(self, text: str, key: int, gamma: float = 0.25,
                     window_k: int = 2, n_control_keys: int = 99) -> dict:
        """reading-mode 检测：比较 green token 与 red token 在可疑模型下的 loss。

        对文本的每个 token（(k+1)-tuple 去重），用 KGW 密钥判断它是否 green，
        计算 ``mean NLL(red) - mean NLL(green)``。若模型在带水印数据上训练过，
        会对 green token 给出更低的 loss，该差值显著大于 0；零假设下 green/red
        标记与模型 loss 无关，差值在 0 附近。用 K 个控制密钥做置换检验得到 p 值。
        """
        from .kgw import is_green

        torch = self.torch
        ids = self.encode(text)
        k = max(1, int(window_k))
        if len(ids) <= k + 1:
            return {"green": 0, "n": 0, "z": 0.0, "p": 1.0, "gamma": float(gamma),
                    "fraction": 0.0, "control_fraction": 0.0}
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids=inp).logits[0, :-1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            tgt = inp[0, 1:]
            nll = (-logp.gather(1, tgt[:, None]).squeeze(1)).tolist()
        seen: set = set()
        items = []  # (window, token, nll)
        for t in range(k, len(ids)):
            window = tuple(ids[t - k:t])
            gram = window + (ids[t],)
            if gram in seen:
                continue
            seen.add(gram)
            items.append((window, int(ids[t]), float(nll[t - 1])))
        if not items:
            return {"green": 0, "n": 0, "z": 0.0, "p": 1.0, "gamma": float(gamma),
                    "fraction": 0.0, "control_fraction": 0.0}

        def statistic(kk: int) -> float:
            g = [x[2] for x in items if is_green(int(kk), x[0], x[1], float(gamma))]
            r = [x[2] for x in items if not is_green(int(kk), x[0], x[1], float(gamma))]
            if not g or not r:
                return 0.0
            return sum(r) / len(r) - sum(g) / len(g)

        true_stat = statistic(int(key))
        ctrl = [statistic(int(key) + 1000003 * (j + 1))
                for j in range(int(n_control_keys))]
        ctrl_mean = sum(ctrl) / len(ctrl) if ctrl else 0.0
        exceed = sum(1 for c in ctrl if c >= true_stat)
        p = (1.0 + exceed) / (1.0 + len(ctrl))
        sd = (sum((c - ctrl_mean) ** 2 for c in ctrl)
              / max(1, len(ctrl) - 1)) ** 0.5 if ctrl else 0.0
        z = (true_stat - ctrl_mean) / sd if sd > 0 else 0.0
        n_green = sum(1 for x in items if is_green(int(key), x[0], x[1], float(gamma)))
        return {"green": n_green, "n": len(items), "z": float(z), "p": float(p),
                "gamma": float(gamma),
                "fraction": n_green / len(items),
                "control_fraction": ctrl_mean}

    def green_scores_text(self, ids: list[int], key: int, gamma: float = 0.25,
                          window_k: int = 2) -> dict:
        """对一段生成 token 序列做 KGW 检测（按 (k+1)-tuple 去重后 z 检验）。"""
        from .kgw import score_green_tokens

        return score_green_tokens(list(ids), int(key), self.vocab_size,
                                  float(gamma), int(window_k))

    def green_sequence(self, ids: list[int], key: int, gamma: float = 0.25,
                       window_k: int = 2) -> dict:
        """返回逐 token 的 green 指示序列（按 (k+1)-tuple 去重后的顺序）。"""
        from .kgw import is_green

        k = max(1, int(window_k))
        seen: set = set()
        flags: list[int] = []
        windows: list[tuple] = []
        tokens: list[int] = []
        for t in range(k, len(ids)):
            window = tuple(int(x) for x in ids[t - k:t])
            nxt = int(ids[t])
            gram = window + (nxt,)
            if gram in seen:
                continue
            seen.add(gram)
            windows.append(window)
            tokens.append(nxt)
            flags.append(1 if is_green(int(key), window, nxt, float(gamma)) else 0)
        return {"flags": flags, "windows": windows, "tokens": tokens,
                "green": sum(flags), "n": len(flags)}

    def green_counts_global(self, ids_batches: list[list[int]], key: int,
                            gamma: float = 0.25,
                            window_k: int = 2) -> tuple[int, int, list[int]]:
        """跨文档全局去重的 green token 计数（避免模板文本重复计数）。"""
        from .kgw import global_green_counts

        out = global_green_counts(list(ids_batches), int(key), float(gamma),
                                  int(window_k))
        return int(out["green"]), int(out["n"]), list(out["flags"])

    def green_counts(self, ids_batches: list[list[int]], key: int,
                     gamma: float = 0.25, window_k: int = 2) -> tuple[int, int]:
        """对多段生成文本统计 green token 数 / 去重观测数。"""
        green = n = 0
        for ids in ids_batches:
            out = self.green_scores_text(ids, key, gamma, window_k)
            green += int(out["green"])
            n += int(out["n"])
        return green, n

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def _make_optimizer(self, params, lr: float, weight_decay: float):
        """按配置选择 AdamW / 8-bit AdamW（全参微调 4B 用 8-bit 省显存）。"""
        torch = self.torch
        if getattr(self, "optimizer", "adamw") == "adamw8bit":
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=float(lr),
                                       weight_decay=float(weight_decay))
        return torch.optim.AdamW(params, lr=float(lr),
                                 weight_decay=float(weight_decay))

    def train(self, texts: list[str], *, epochs: int = 2, lr: float = 5e-4,
              batch_size: int = 8, max_length: int = 128,
              grad_accum: int = 1, weight_decay: float = 0.01,
              seed: int = 0, log_fn=None) -> dict:
        """在 ``texts`` 上做因果语言模型 SFT，返回训练统计。"""
        torch = self.torch
        torch.manual_seed(int(seed))
        random.Random(int(seed)).shuffle(texts)
        self.model.train()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = self._make_optimizer(params, lr, weight_decay)
        samples = []
        for text in texts:
            ids = self.encode(text)
            if len(ids) < 4:
                continue
            ids = ids[:int(max_length)]
            samples.append(ids)
        total_steps = max(1, math.ceil(len(samples) / int(batch_size))) * int(epochs)
        history = []
        step = 0
        running = 0.0
        for epoch in range(int(epochs)):
            random.Random(int(seed) + epoch).shuffle(samples)
            for start in range(0, len(samples), int(batch_size)):
                batch = samples[start:start + int(batch_size)]
                if not batch:
                    continue
                maxlen = max(len(x) for x in batch)
                inp = torch.full((len(batch), maxlen), self.tokenizer.pad_token_id,
                                 dtype=torch.long)
                labels = torch.full_like(inp, -100)
                for i, ids in enumerate(batch):
                    inp[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                    labels[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                inp = inp.to(self.device)
                labels = labels.to(self.device)
                out = self.model(input_ids=inp, labels=labels)
                loss = out.loss / int(grad_accum)
                loss.backward()
                if (step + 1) % int(grad_accum) == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad()
                running += float(loss.detach())
                step += 1
                if log_fn and step % max(1, min(20, total_steps // 5 or 1)) == 0:
                    log_fn(f"train step {step}/{total_steps} loss={running / step:.4f}")
                history.append(float(loss.detach()))
        opt.zero_grad()
        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        return {"steps": step, "epochs": int(epochs), "n_samples": len(samples),
                "final_loss": round(history[-1], 6) if history else None,
                "mean_loss": round(sum(history) / len(history), 6) if history else None}

    def train_completion(self, pairs: list[tuple[str, str]], *, epochs: int = 2,
                         lr: float = 5e-4, batch_size: int = 8,
                         max_length: int = 256, grad_accum: int = 1,
                         weight_decay: float = 0.01, seed: int = 0,
                         log_fn=None) -> dict:
        """指令微调：只在 answer 部分计算 loss（prompt 部分 mask 掉）。

        这是 SFT 的常见做法，也是 Double-I 这类“标签只有一个 token”的实验
        能够学起来的必要条件；``texts`` 版本的 ``train`` 会把 loss 摊到整段
        文本上，标签信号会被平均掉。
        """
        torch = self.torch
        torch.manual_seed(int(seed))
        rng = random.Random(int(seed))
        rng.shuffle(pairs)
        self.model.train()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = self._make_optimizer(params, lr, weight_decay)
        samples = []
        for prompt, answer in pairs:
            p_ids = self.encode(prompt)
            a_ids = self.encode(answer)
            if not a_ids:
                continue
            ids = (p_ids + a_ids)[:int(max_length)]
            prompt_len = min(len(p_ids), len(ids) - 1)
            samples.append((ids, prompt_len))
        if not samples:
            return {"steps": 0, "epochs": 0, "n_samples": 0,
                    "final_loss": None, "mean_loss": None}
        total_steps = max(1, math.ceil(len(samples) / int(batch_size))) * int(epochs)
        history = []
        step = 0
        for epoch in range(int(epochs)):
            rng.shuffle(samples)
            for start in range(0, len(samples), int(batch_size)):
                batch = samples[start:start + int(batch_size)]
                maxlen = max(len(ids) for ids, _ in batch)
                inp = torch.full((len(batch), maxlen), self.tokenizer.pad_token_id,
                                 dtype=torch.long)
                labels = torch.full_like(inp, -100)
                for i, (ids, plen) in enumerate(batch):
                    inp[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                    labels[i, plen:len(ids)] = torch.tensor(ids[plen:], dtype=torch.long)
                inp = inp.to(self.device)
                labels = labels.to(self.device)
                out = self.model(input_ids=inp, labels=labels)
                loss = out.loss / int(grad_accum)
                loss.backward()
                if (step + 1) % int(grad_accum) == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad()
                history.append(float(loss.detach()))
                step += 1
                if log_fn and step % max(1, min(20, total_steps // 5 or 1)) == 0:
                    log_fn(f"train step {step}/{total_steps} loss={sum(history)/len(history):.4f}")
        opt.zero_grad()
        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        return {"steps": step, "epochs": int(epochs), "n_samples": len(samples),
                "final_loss": round(history[-1], 6) if history else None,
                "mean_loss": round(sum(history) / len(history), 6) if history else None}

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({"model": self.model.state_dict(),
                         "model_name": self.model_name}, path)
        return str(path)
