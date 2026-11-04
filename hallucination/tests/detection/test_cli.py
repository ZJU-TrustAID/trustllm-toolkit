import json
from pathlib import Path

from hallucination.detection.cli import main


def test_cli_detect_json(capsys):
    code = main(["detect", "--preset", "offline_toy", "--prompt", "请介绍蓝鲸的食性。", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["count"] == 1
    assert payload["results"][0]["risk"] >= 0.0


def test_cli_detect_response_only_requires_samples(capsys):
    code = main(["detect", "--preset", "offline_toy", "--response", "蓝鲸以磷虾为食。"])
    assert code == 2
    captured = capsys.readouterr()
    assert "samples" in captured.err
    assert captured.out == ""


def test_cli_detect_response_with_samples(capsys):
    samples = json.dumps(["蓝鲸以磷虾为食。", "蓝鲸吃磷虾。"], ensure_ascii=False)
    code = main(
        [
            "detect",
            "--preset",
            "offline_toy",
            "--response",
            "蓝鲸以磷虾为食。",
            "--samples",
            samples,
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["count"] == 1
    assert payload["responses"] == ["蓝鲸以磷虾为食。"]
    assert payload["samples"] == [["蓝鲸以磷虾为食。", "蓝鲸吃磷虾。"]]
    assert payload["results"][0]["risk"] >= 0.0


def test_cli_detect_response_with_samples_file(capsys):
    samples_path = Path(__file__).parent / "data" / "blue_whale_samples.json"
    code = main(
        [
            "detect",
            "--preset",
            "offline_toy",
            "--response",
            "蓝鲸以磷虾为食。",
            "--samples",
            f"@{samples_path}",
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["samples"] == [["蓝鲸以磷虾为食。", "蓝鲸以磷虾为主食。"]]


def test_cli_detect_rejects_malformed_samples(capsys):
    code = main(["detect", "--preset", "offline_toy", "--response", "蓝鲸以磷虾为食。", "--samples", '"not-a-list"'])
    assert code == 2
    assert "--samples must be a JSON list of strings" in capsys.readouterr().err


def test_cli_list_methods_and_datasets(capsys):
    assert main(["list-methods"]) == 0
    methods = capsys.readouterr().out
    assert "response_semantic_entropy" in methods
    assert "sentence_semantic_entropy" not in methods

    assert main(["list-datasets"]) == 0
    datasets = capsys.readouterr().out
    assert "textbook_toy" in datasets


def test_cli_calibrate_threshold(tmp_path, capsys):
    scores_path = tmp_path / "scores.jsonl"
    rows = [
        {"score": 0.1, "label": 0},
        {"score": 0.4, "label": 0},
        {"score": 0.6, "label": 1},
        {"score": 0.9, "label": 1},
    ]
    scores_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "threshold.json"
    assert main(["calibrate", "--scores", str(scores_path), "--method", "threshold", "--out", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["type"] == "calibration_bundle"
    assert payload["threshold"]["type"] == "threshold"
    assert payload["threshold"]["split"] == "validation"
    assert payload["metadata"]["n"] == 4


def test_cli_calibrate_parses_string_labels_strictly(tmp_path, capsys):
    scores_path = tmp_path / "scores.jsonl"
    rows = [
        {"score": 0.1, "label": "false"},
        {"score": 0.4, "label": "false"},
        {"score": 0.6, "label": "true"},
        {"score": 0.9, "label": "true"},
    ]
    scores_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "threshold.json"
    assert main(["calibrate", "--scores", str(scores_path), "--method", "threshold", "--out", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["positive_rate"] == 0.5
