export const paper = {
  title: "MarkSec: Capability-Aware Evaluation of Adversarial Attacks Against LLM Watermarks",
  subtitle:
    "Taxonomy, evaluation framing, and a runnable toolkit for studying generative watermark attacks.",
  venue: "arXiv:2609.16681 · cs.CR",
  authors: [
    { name: "Kairong Li" },
    { name: "Zhikun Zhang" },
    { name: "Xiao Ren" },
    { name: "Yunjun Gao", corresponding: true },
  ],
  affiliations: ["Zhejiang University, Hangzhou, China"],
  emails: ["kairong.li@zju.edu.cn", "zhikun@zju.edu.cn", "renx@zju.edu.cn", "gaoyj@zju.edu.cn"],
  resources: [
    { label: "Read Paper", href: "https://arxiv.org/abs/2609.16681", type: "paper" },
    { label: "Open Toolkit", href: "/marksec/toolkit", type: "code" },
    { label: "Citation", href: "#citation", type: "paper" },
  ],
  abstract: [
    "LLM watermark has emerged as a promising solution for tracing the origin of LLM-generated content to limit accidental or deliberate misuse. Despite its prevalence, it suffers from various adversarial attacks: stealing, which recovers watermark-related information; scrubbing, which removes watermark signals in marked text; and spoofing, which forges watermark-like outputs without actual watermark embedding. However, these attacks are often studied in isolation, without revealing their inherent connections. Also, current evaluations often lack shared detector calibration, metric definitions, and reporting semantics, making it difficult to provide a comprehensive picture of the risks faced by LLM watermarks. In addition, the attack success-related metrics and text quality-related metrics are measured separately, making it difficult to identify attacks that are both effective and quality-preserving.",
    "To this end, we propose MarkSec, a general framework that unifies analyses of stealing, scrubbing, and spoofing. Building on this framework, we conduct a comprehensive evaluation of different attacks under a unified reporting protocol. We introduce a quality-constrained attack success metric and make attack results comparable across watermark methods. Extensive evaluations on representative watermark families, attack methods, and LLMs reveal several key findings. First, attacks that appear strongest under success-based metrics can lose much of their advantage once text quality is taken into account. Second, some scrubbing attacks are broad general-scrubbing baselines, while others work best only on particular watermarks. Third, under matched KGW settings, stealing-based scrubbers often underperform the best general-scrubbing baselines. These results show that apparent attack winners depend on text-quality constraints, attack generality, and capability assumptions.",
  ],
  contributions: [
    "Organize watermark attacks by target, attacker knowledge, and success outcome.",
    "Compare methods with a shared view of robustness, text quality, and detector behavior.",
    "Connect the paper framing to runnable configs and extensible local artifacts.",
  ],
  bibtex: `@misc{li2026marksec,
  title     = {{MarkSec: Capability-Aware Evaluation of Adversarial Attacks Against LLM Watermarks}},
  author    = {Li, Kairong and Zhang, Zhikun and Ren, Xiao and Gao, Yunjun},
  eprint    = {2609.16681},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR},
  url       = {https://arxiv.org/abs/2609.16681},
  year      = {2026}
}`,
};
