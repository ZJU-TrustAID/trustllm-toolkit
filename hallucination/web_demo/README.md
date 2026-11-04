# Signal Desk demo

Run the web demo from the package directory:

```powershell
python -m pip install -e ".[dev]"
python web_demo/server.py
```

Open `http://127.0.0.1:8765`.

## 教程模式

A textbook-companion tutorial with a MarkSec-style flow, organized as three
pages (top tab bar): **白盒检测** (5 white-box detection methods), **黑盒检测**
(9 black-box detection methods in the book's three categories) and
**缓解方法** (5 mitigation methods). On each page:

1. **提示词对话框**: click the dialog box, pick one of the prompts from the
   popover, and the choice is echoed back into the box. The popover only
   offers prompts the current page can demonstrate (the faithful theme, for
   instance, has no mitigation to correct).
2. **右侧方法选项组**: the sidebar lists exactly the current page's methods,
   grouped on every page following the book (white-box: 基于词元概率检测 /
   基于熵值检测 / 基于表征分析检测; black-box: 基于外部知识库验证 /
   基于模型自评估检测 / 基于一致性检测; mitigation: 事实性增强解码 /
   忠实性增强解码 / 事后修正 / 激活转向 · toolkit 扩展). No method is picked
   automatically — select one explicitly; options that the selected prompt
   cannot demonstrate are disabled, and the selected method shows a summary
   card (book reference, principle, prerequisites).
3. **生成**: clicking generate simulates the generation process — the answer
   streams into the output box with a blinking caret — then the method result
   renders below: the conclusion card (with the reference answer and the
   risk formula), token-level annotations and the mechanism visualizations.

**No model is called on this page.** The presets feed canned teaching data
through the toolkit's *real* code: detection scorers run on canned
`DetectionExample`s with the offline demo backends; DoLa/CAD replay the real
decoding math (`jsd`, `apc_mask`, `contrast_logits`, `combine_context_logits`,
`distribution_kl`) on a small canned vocabulary; CoVe runs the real
four-stage chain on a scripted backend.

API endpoints: `GET /api/catalog`, `POST /api/method {method, scenario}`.

The demo page itself never calls a model; clone the toolkit and run the CLI/notebooks for real inference (see the repository README).
