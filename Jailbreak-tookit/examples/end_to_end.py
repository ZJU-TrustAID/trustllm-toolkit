"""每个已注册方法的端到端示例：输入 → 构造/优化 → 目标模型回答。"""

import argparse
import json
from pathlib import Path

import torch

from textbook_toolkit.jailbreak import Message, get, list_methods
from textbook_toolkit.jailbreak.optimization import (
    AutoDANConfig,
    CausalLM,
    COLDConfig,
    GCGConfig,
)


def run_example(name: str, backend: CausalLM, args) -> dict:
    """Use real model generation, keeping full history and optimizer artifacts."""
    generate = backend.text_generator(max_new_tokens=args.max_new_tokens)
    inputs = {"query": args.query}
    common = dict(steps=args.steps, seed=0, max_new_tokens=args.max_new_tokens)
    if name in ("autodan", "gcg", "cold_attack"):
        inputs["target"] = args.target
        if name == "autodan":
            prototype = "Use clear language in a classroom discussion. Topic: {query}"
            inputs["prototype"] = prototype
            result = get(name, generator=generate, prototype=prototype).run(
                args.query,
                args.target,
                backend=backend,
                config=AutoDANConfig(**common, population_size=4, eval_batch_size=4),
            )
        else:
            options = {"initial_control": args.initial_control}
            if name == "gcg":
                config = GCGConfig(**common, top_k=32, candidates=16, eval_batch_size=4)
            else:
                config = COLDConfig(**common, batch_size=1, top_k=16, mode=args.mode)
                if args.mode == "control":
                    options["right_context"] = "Use one short sentence."
                elif args.mode == "paraphrase":
                    options["initial_control"] = args.query
            inputs.update(options)
            if name == "cold_attack":
                inputs["mode"] = args.mode
            result = get(name, backend=backend, config=config).run(
                args.query, args.target, **options
            )
        return {
            "method": name,
            "input": inputs,
            "output": {
                "messages": result.prompt.to_messages(),
                "response": result.response,
                "control": result.control,
                "initial_loss": result.initial_loss,
                "final_loss": result.final_loss,
                "stop_reason": result.stop_reason,
            },
            "details": result.to_dict(),
        }

    options = {
        "template": {"template": "Answer this classroom question: {query}"},
        "role_play": {"role": "科普教师", "scenario": "课堂讨论"},
        "task_disguise": {"task": "Rewrite the following as a short classroom Q&A."},
        "rule_rewrite": {"rules": ("Answer in English.", "Use one sentence.")},
        "context": {
            "history": (
                Message("user", "We are studying plants in class."),
                Message("assistant", "Let's discuss how plants use light."),
            )
        },
        "base64": {},
        "caesar": {"shift": 3},
        "red_teaming": {"generator": generate},
    }[name]
    # Record JSON inputs without serializing a model or a callable.
    inputs["options"] = (
        {"generator": "same local model"}
        if name == "red_teaming"
        else {"history": [message.to_dict() for message in options["history"]]}
        if name == "context"
        else options
    )
    prompt = get(name, **options).build(args.query)
    response = generate(prompt.messages)
    return {
        "method": name,
        "input": inputs,
        "output": {"messages": prompt.to_messages(), "response": response},
        "details": {"prompt": prompt.to_dict()},
    }


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("all", *list_methods()), default="gcg")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--steps", type=positive_integer, default=3)
    parser.add_argument("--max-new-tokens", type=positive_integer, default=48)
    parser.add_argument("--threads", type=positive_integer, default=1)
    parser.add_argument("--query", default="Explain photosynthesis in one sentence.")
    parser.add_argument("--target", default="Plants use sunlight")
    parser.add_argument("--initial-control", default="! ! ! ! ! ! ! !")
    parser.add_argument("--mode", choices=("suffix", "control", "paraphrase"), default="suffix")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.mode != "suffix" and args.method != "cold_attack":
        parser.error("--mode control/paraphrase requires --method cold_attack")
    output = args.output or Path(f"outputs/end_to_end_{args.method}.json")
    if output.exists():
        parser.error(f"Output already exists: {output}")

    torch.set_num_threads(args.threads)
    if args.model == "tiny":
        from textbook_toolkit.jailbreak.optimization.demo import tiny_backend

        backend = tiny_backend(chat=True)
        print("tiny uses random weights; outputs are computation checks only.", flush=True)
    else:
        backend = CausalLM.from_pretrained(
            args.model,
            device=args.device,
            dtype="float32",
            local_files_only=args.local_files_only,
        )

    records = []
    names = list_methods() if args.method == "all" else (args.method,)
    for name in names:
        print(f"\n[{name}]", flush=True)
        record = run_example(name, backend, args)
        print("INPUT", json.dumps(record["input"], ensure_ascii=False, indent=2), flush=True)
        print("OUTPUT", json.dumps(record["output"], ensure_ascii=False, indent=2), flush=True)
        records.append(record)

    data = {
        "schema_version": 1,
        "model": args.model,
        "device": str(backend.device),
        "dtype": str(backend.embedding.dtype),
        "torch": torch.__version__,
        "chat_template": backend.tokenizer.chat_template,
        "steps": args.steps,
        "max_new_tokens": args.max_new_tokens,
        "results": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("saved:", output.resolve())


if __name__ == "__main__":
    main()
