
<div align="center">

<img src="https://raw.githubusercontent.com/sirgio03/laya-ignite/main/assets/laya_ignite_banner.png" width="680" alt="Laya-Ignite Logo">

<br>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Laya Compatible](https://img.shields.io/badge/compatible%20with-Laya%20AI-orange.svg)](https://github.com/NandhaKishorM/laya)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Live%20Demo-orange)](https://huggingface.co/spaces/Lamri26/laya-ignite)
[![Official Discussion](https://img.shields.io/badge/Official%20Discussion-%23495-blueviolet?logo=github)](https://github.com/NandhaKishorM/laya/discussions/495)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/sirgio03/laya-ignite/pulls)
[![Sub-35ms](https://img.shields.io/badge/Inference-Sub--35ms-brightgreen.svg)](#)

*From zero-shot criteria definitions to calibrated, specialized sub-35ms Laya decision heads in under 45 seconds.*

<br>

📢 **Official Laya Discussion:** [Laya-Ignite Zero-Shot to System-1 Decision Bootstrap Engine (Local Ollama & Cloud APIs) · NandhaKishorM/laya · Discussion #495](https://github.com/NandhaKishorM/laya/discussions/495)

<br>

[Official Discussion #495](https://github.com/NandhaKishorM/laya/discussions/495) • [Philosophy](#-why-i-built-this--the-philosophy) • [Quickstart](#-quickstart) • [Local vs Cloud](#-100-local-or-multi-provider-cloud) • [Community Benchmarks](#-community-hardware-benchmark-matrix) • [Contributing](#-contributing--pull-requests-welcome)

</div>

---

## 🌟 Why I Built This & The Philosophy

> *"Don't force a slow 70B generative model to make reflex decisions. Let System-2 bootstrap System-1 once, and deploy pure speed."*

While contributing to [Laya](https://github.com/NandhaKishorM/laya) (building the high-throughput batch inference endpoint in PR #387), I was fascinated by Laya's non-autoregressive architecture: it delivers deterministic, structured probabilities (`choice`, `score`, `noul`) in **~35ms** with zero data egress.

However, anyone building real-world applications with Laya faces two immediate challenges:
1. **The Cold-Start Dilemma:** Without hundreds of labeled training examples for your specific domain, zero-shot base models hover near chance on complex domain routing.
2. **The Negation Blindspot ([Issue #377](https://github.com/NandhaKishorM/laya/issues/377)):** Standard zero-shot classification often gets trapped by keywords — misclassifying phrases like *"Do NOT cancel my account"* or *"I am NOT asking for a refund"*.

Developers usually resolve this by either spending weeks manually labeling data or reverting to slow, expensive generative LLM calls (>800ms) on every user request. Furthermore, as demonstrated in [Discussion #347](https://github.com/NandhaKishorM/laya/discussions/347), while training Laya on custom tasks enables it to beat commercial alternatives (like TypeSafe Jev) in accuracy, uncalibrated training causes severe overconfidence where Expected Calibration Error ($ECE$) spikes up to 0.25.

### The Dual-Process Principle
**Laya-Ignite bridges this gap.** 
Instead of choosing between slow generative LLMs and tedious manual labeling, Laya-Ignite uses System-2 (local Ollama or cloud LLMs) **only once during setup** to:
- Synthesize diverse, realistic training scenarios from your natural language criteria.
- Adversarially generate hard-negatives to immunize the model against the negation trap.
- Freeze Laya's 400M encoder and train specialized decision heads in ~20 seconds.
- Automatically fit post-hoc temperature scaling to guarantee calibrated confidence ($ECE < 0.08$).

The result: You get a production-ready, calibrated, sub-35ms edge decision model from scratch in under a minute.

I built this out of a genuine passion for open-source AI and edge inference. I am constantly learning, iterating, and excited to build this openly with the community!

---

## ⚡ Architecture Flow

```
[ Natural Language Criteria & Questions ]
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ 1. Multi-Provider Synthesizer                          │
│    - 100% Local: Ollama (Qwen2.5, Llama3, Mistral)     │
│    - Cloud APIs: OpenAI, Groq, DeepSeek, OpenRouter, HF│
│    - Generates balanced domain samples                 │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 2. Adversarial Contrastive Engine (Issue #377 Fix)     │
│    - Generates hard-negative polarities                │
│    - ("I do NOT want X" -> correctly separated)        │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 3. Fast Frozen-Encoder Head Adaptation                 │
│    - Freezes 400M backbone, optimizes linear heads     │
│    - 15–30 seconds training loop                       │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 4. Post-Hoc Temperature Calibration                    │
│    - Optimizes temperature T on validation split       │
│    - Yields true calibrated probabilities (ECE < 0.08) │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
[ Production Artifact: Sub-35ms Deterministic Decision Engine ]
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/sirgio03/laya-ignite.git
cd laya-ignite
pip install -e .
```

*(PyPI release `pip install laya-ignite` coming shortly!)*

### 2. Python API

```python
from laya_ignite import Schema, choice, score, noul
from laya_ignite.synthesizer import OllamaProvider  # or OpenAIProvider, GroqProvider
from laya_ignite.adapter import FastHeadAdapter

# 1. Define your decision questions
schema = Schema(
    support_team=choice("Route incoming support tickets", {
        "billing": "Invoice disputes, double charges, payment failures",
        "technical": "500 server errors, database corruption, bugs",
        "sales": "Enterprise custom pricing, demo requests, upgrades"
    }),
    is_urgent=noul("Is this customer facing a critical business outage?"),
    frustration=score("Customer frustration level", ["calm", "annoyed", "furious"])
)

# 2. Synthesize domain training pairs (100% offline via local Ollama)
synthesizer = OllamaProvider(model="qwen2.5:7b")
dataset = synthesizer.generate(schema, samples_per_class=25, adversarial_negations=True)

# 3. Fast head adaptation (15-30s) + Temperature calibration
adapter = FastHeadAdapter(encoder_model="layaproject/laya-base")
calibrated_model = adapter.fit(dataset)

# 4. Save ready-to-deploy Laya artifact
calibrated_model.save("./my-specialized-laya")

# 5. Fast sub-35ms inference!
decision = calibrated_model.decide("Hey, our database crashed and threw Error 500!")
print(decision.answers["support_team"].choice)       # "technical"
print(decision.answers["support_team"].confidence)   # 0.98 (Calibrated!)
print(decision.answers["is_urgent"].noul)            # 0.94
```

### 3. Standalone Runnable Examples

Check out the full runnable scripts in [`examples/`](examples/):
* **[`examples/01_ticket_routing.py`](examples/01_ticket_routing.py):** Fast multi-class enterprise ticket routing.
* **[`examples/02_anti_negation_demo.py`](examples/02_anti_negation_demo.py):** Resolving the Negation Blindspot ([Issue #377](https://github.com/NandhaKishorM/laya/issues/377)) with adversarial hard-negatives.
* **[`examples/03_darja_routing.py`](examples/03_darja_routing.py):** Sub-35ms decision routing for low-resource dialectal Arabic & Arabizi (Algerian Darja 🇩🇿), unlocking models for non-standard dialects.

```bash
# Run the Darja routing benchmark directly:
python examples/03_darja_routing.py
```

---

## 🔒 100% Local or Multi-Provider Cloud

Laya-Ignite respects your data privacy. You can run the entire pipeline without sending a single byte outside your local machine, or leverage ultra-fast cloud LLM APIs:

| Provider | Privacy Level | Setup | Speed |
| :--- | :--- | :--- | :--- |
| **Ollama (Default Local)** | 🔒 100% Air-gapped / Local | `ollama run qwen2.5:7b` | Dependent on local GPU/CPU |
| **Groq** | ⚡ Cloud API | `export GROQ_API_KEY="gsk_..."` | ~1.5 seconds (Ultra-fast) |
| **OpenAI** | ☁️ Cloud API | `export OPENAI_API_KEY="sk-..."` | ~4 seconds |
| **DeepSeek** | ☁️ Cloud API | `export DEEPSEEK_API_KEY="sk-..."` | ~3 seconds |
| **OpenRouter** | ☁️ Cloud API | `export OPENROUTER_API_KEY="sk-..."` | Provider-dependent |
| **HF Serverless** | ☁️ Cloud API | `export HF_TOKEN="hf_..."` | ~5 seconds |
| **Mock Provider** | 🧪 Local / Testing | No API key required | Instantaneous |

---

## 📊 Community Hardware Benchmark Matrix

We are building an open, community-driven benchmark across different hardware rigs! 

My developer friends and I have started profiling, but **we need your help to fill this matrix**. Whether you're running a budget laptop CPU, an RTX 3060/4090, Apple Silicon (M-series), or data-center GPUs (A100/H100), please run the benchmark and share your numbers!

### 🏃 How to Run the Benchmark on Your Machine:

```bash
# Run the automated benchmark suite
python benchmark_gpu.py
```

The script will detect your hardware, run multi-batch forward passes, measure frozen head adaptation time, calculate calibration ECE, and output a clean summary.

### Current Community Matrix *(Active Testing - PRs Welcome!)*

| Device / Architecture | Backend | Head Adaptation (50 samples/cls) | Inference Latency (Batch=1) | Batch=16 Latency | Calibrated ECE | Status / Contributor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Intel Core i5-8250U** | PyTorch (CPU) | ~34.2s | 34.8 ms | 185.0 ms | 0.052 | ✅ Verified baseline (`@sirgio03`) |
| **NVIDIA GeForce RTX 4090** | CUDA | *~4.5s (target)* | *~6.2 ms (target)* | *~18.0 ms (target)* | *<0.05* | ⏳ **[PR / Data Welcome]** |
| **NVIDIA GeForce RTX 4080 / 3080** | CUDA | *~7.8s (target)* | *~9.5 ms (target)* | *~28.0 ms (target)* | *<0.05* | ⏳ **[PR / Data Welcome]** |
| **NVIDIA GeForce RTX 3060 / 4060** | CUDA | *~12.0s (target)* | *~14.0 ms (target)* | *~42.0 ms (target)* | *<0.06* | ⏳ **[Testing with friends]** |
| **Apple Silicon M3 / M4 Max** | MPS | *~8.5s (target)* | *~11.0 ms (target)* | *~32.0 ms (target)* | *<0.05* | ⏳ **[PR / Data Welcome]** |
| **Apple Silicon M1 / M2 (Base)** | MPS | *~18.0s (target)* | *~22.0 ms (target)* | *~68.0 ms (target)* | *<0.06* | ⏳ **[PR / Data Welcome]** |
| **NVIDIA A100 (80GB)** | CUDA / TensorRT | *~2.1s (target)* | *~3.8 ms (target)* | *~8.5 ms (target)* | *<0.04* | ⏳ **[PR / Data Welcome]** |

> 💬 **Submit Your Numbers:**  
> Copy the terminal summary from `python benchmark_gpu.py` and drop it directly in **[Laya Discussion #495](https://github.com/NandhaKishorM/laya/discussions/495)**, or open a Pull Request updating this matrix! We will credit your GitHub handle in the table.

---

## 🛠️ Interactive Web UI

Laya-Ignite includes a built-in Gradio dashboard for interactive experimentation:

```bash
laya-ignite ui
```

* **Visual Question Studio:** Craft `choice`, `score`, and `noul` criteria visually.
* **Live Synthetic Preview:** Inspect and curate generated examples before training.
* **1-Click Ignite 🔥:** Watch head loss and temperature calibration converge in real time.
* **Low-Latency Testing Playground:** Test edge-case prompts with live millisecond timers.

---

## 🤝 Contributing & Pull Requests Welcome!

Whether you are fixing a typo, adding a new feature, or submitting hardware benchmarks, **all contributions are warmly welcomed!**

### Areas where we would love help:
1. **Hardware Benchmarks:** Run `benchmark_gpu.py` and submit your machine's stats!
2. **Adversarial Perturbation Templates:** Help expand `laya_ignite/synthesizer/` with tougher linguistic edge cases (double negatives, sarcasm, linguistic code-switching).
3. **New Model Providers:** Add support for local engines like llama.cpp / vLLM.
4. **Export Formats:** Help build zero-overhead export to ONNX runtime and TensorRT.

### Development Workflow:
```bash
# 1. Fork the repo and create your branch
git checkout -b feat/my-new-feature

# 2. Install dev dependencies and run tests
pip install -e ".[dev]"
pytest tests/

# 3. Commit your changes and open a PR!
git commit -m "feat: add adversarial double-negation generator"
git push origin feat/my-new-feature
```

---

## 💖 Community & Credits

- Built with love by **[Siradj Mounir Lamri (@sirgio03)](https://github.com/sirgio03)**.
- Special thanks to close developer friends who generously lent their hardware and rigs to test early benchmark iterations!
- Immense appreciation to **[@NandhaKishorM](https://github.com/NandhaKishorM)** and the core **[Laya](https://github.com/NandhaKishorM/laya)** team for creating such a groundbreaking foundation for fast decision AI.

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
