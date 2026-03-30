<p align="center">
  <img src="assets/logo.png"/>
</p>

<!-- icon -->

[![stars](https://img.shields.io/github/stars/open-sciencelab/GraphGen.svg)](https://github.com/open-sciencelab/GraphGen)
[![forks](https://img.shields.io/github/forks/open-sciencelab/GraphGen.svg)](https://github.com/open-sciencelab/GraphGen)
[![open issues](https://img.shields.io/github/issues-raw/open-sciencelab/GraphGen)](https://github.com/open-sciencelab/GraphGen/issues)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/open-sciencelab/GraphGen)](https://github.com/open-sciencelab/GraphGen/issues)
[![documentation](https://img.shields.io/badge/docs-latest-blue)](https://chenzihong.gitbook.io/graphgen-cookbook/)
[![pypi](https://img.shields.io/pypi/v/graphg.svg?style=flat&logo=pypi&logoColor=white)](https://pypi.org/project/graphg/)
[![wechat](https://img.shields.io/badge/wechat-brightgreen?logo=wechat&logoColor=white)](https://cdn.vansin.top/internlm/dou.jpg)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-white)](https://arxiv.org/abs/2505.20416)
[![Hugging Face](https://img.shields.io/badge/Paper-on%20HF-white?logo=huggingface&logoColor=yellow)](https://huggingface.co/papers/2505.20416)

[![Hugging Face](https://img.shields.io/badge/Demo-on%20HF-blue?logo=huggingface&logoColor=yellow)](https://huggingface.co/spaces/chenzihong/GraphGen)
[![Model Scope](https://img.shields.io/badge/%F0%9F%A4%96%20Demo-on%20MS-green)](https://modelscope.cn/studios/chenzihong/GraphGen)


GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation

[English](README.md) | [中文](README_zh.md)

<details close>
<summary><b>📚 Table of Contents</b></summary>

- 📝 [What is GraphGen?](#-what-is-graphgen)
- 📌 [Latest Updates](#-latest-updates)
- ⚙️ [Support List](#-support-list)
- 🚀 [Quick Start](#-quick-start)
- 🧪 [Chemistry Dataset Pipeline](#-chemistry-dataset-pipeline)
- 🏗️ [System Architecture](#-system-architecture)
- 🍀 [Acknowledgements](#-acknowledgements)
- 📚 [Citation](#-citation)
- 📜 [License](#-license)
- 📅 [Star History](#-star-history)

[//]: # (- 🌟 [Key Features]&#40;#-key-features&#41;)
[//]: # (- 💰 [Cost Analysis]&#40;#-cost-analysis&#41;)
[//]: # (- ⚙️ [Configurations]&#40;#-configurations&#41;)

</details>

## 📝 What is GraphGen?

GraphGen is a framework for synthetic data generation guided by knowledge graphs. Please check the [**paper**](https://arxiv.org/abs/2505.20416) and [best practice](https://github.com/open-sciencelab/GraphGen/issues/17).

It begins by constructing a fine-grained knowledge graph from the source text，then identifies knowledge gaps in LLMs using the expected calibration error metric, prioritizing the generation of QA pairs that target high-value, long-tail knowledge.
Furthermore, GraphGen incorporates multi-hop neighborhood sampling to capture complex relational information and employs style-controlled generation to diversify the resulting QA data.

After data generation, you can use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [xtuner](https://github.com/InternLM/xtuner) to finetune your LLMs.

## 📌 Latest Updates
- **2026.03.29**: **Chemistry reasoning dataset pipeline** — 12 generation formats across 2 molecular knowledge graphs (KG1: rule graph, KG2: molecule graph), including 3 new chemistry-specific generators: `pairwise_preference`, `ranking`, and `matched_molecular_pair`. See [Chemistry Dataset Pipeline](#-chemistry-dataset-pipeline).
- **2026.02.04**: We support HuggingFace Datasets as input data source for data generation now.
- **2026.01.15**: **LLM benchmark synthesis** now supports single/multiple-choice & fill-in-the-blank & true-or-false—ideal for education 🌟🌟
- **2025.12.26**: Knowledge graph evaluation metrics about accuracy (entity/relation), consistency (conflict detection), structural robustness (noise, connectivity, degree distribution)

<details>
<summary>History</summary>

- **2025.12.16**: Added [rocksdb](https://github.com/facebook/rocksdb) for key-value storage backend and [kuzudb](https://github.com/kuzudb/kuzu) for graph database backend support.
- **2025.12.16**: Added [vllm](https://github.com/vllm-project/vllm) for local inference backend support.
- **2025.12.16**: Refactored the data generation pipeline using [ray](https://github.com/ray-project/ray) to improve the efficiency of distributed execution and resource management.
- **2025.12.1**: Added search support for [NCBI](https://www.ncbi.nlm.nih.gov/) and [RNAcentral](https://rnacentral.org/) databases, enabling extraction of DNA and RNA data from these bioinformatics databases.
- **2025.10.30**: We support several new LLM clients and inference backends including [Ollama_client](https://github.com/open-sciencelab/GraphGen/blob/main/graphgen/models/llm/api/ollama_client.py), [http_client](https://github.com/open-sciencelab/GraphGen/blob/main/graphgen/models/llm/api/http_client.py), [HuggingFace Transformers](https://github.com/open-sciencelab/GraphGen/blob/main/graphgen/models/llm/local/hf_wrapper.py) and [SGLang](https://github.com/open-sciencelab/GraphGen/blob/main/graphgen/models/llm/local/sglang_wrapper.py).
- **2025.10.23**: We support VQA(Visual Question Answering) data generation now. Run script: `bash scripts/generate/generate_vqa.sh`.
- **2025.10.21**: We support PDF as input format for data generation now via [MinerU](https://github.com/opendatalab/MinerU).
- **2025.09.29**: We auto-update gradio demo on [Hugging Face](https://huggingface.co/spaces/chenzihong/GraphGen) and [ModelScope](https://modelscope.cn/studios/chenzihong/GraphGen).
- **2025.08.14**: We have added support for community detection in knowledge graphs using the Leiden algorithm, enabling the synthesis of Chain-of-Thought (CoT) data.
- **2025.07.31**: We have added Google, Bing, Wikipedia, and UniProt as search back-ends.
- **2025.04.21**: We have released the initial version of GraphGen.

</details>

## Effectiveness of GraphGen
### Pretrain

Inspired by Kimi-K2's [technical report](https://arxiv.org/pdf/2507.20534) (Improving Token Utility with Rephrasing)  and ByteDance Seed's [Reformulation for Pretraining Data Augmentation](https://arxiv.org/abs/2502.04235) (MGA framework), GraphGen added a **rephrase pipeline** — using LLM-driven reformulation to generate diverse variants of the same corpus instead of redundant repetition.

**Setup:** Qwen3-0.6B trained from scratch on [SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B).

| Method | ARC-E | ARC-C | HellaSwag | GSM8K | TruthfulQA-MC1 | TruthfulQA-MC2 | **Average** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SlimPajama-6B trained for 2 epochs | 25.55 | 21.08 | 24.48 | 0.08 | 24.36 | 49.90 | 24.24 |
| SlimPajama-6B + Executive-Summary Rephrase trained for 1 epoch | 26.43 | **22.70** | **24.75** | **1.36** | **26.19** | 51.90 | **25.56**(↑1.32) |
| SlimPajama-6B + Cross-Domain Rephrase trained for 1 epoch | **28.79** | 20.22 | 24.46 | 0.00 | 24.97 | **52.41** | 25.14(↑0.9) |

Both rephrase methods lift the average by ~1 point over the baseline with **zero additional data** — all gains come from how the same knowledge is expressed.


### SFT
Here is post-training result which **over 50% SFT data** comes from GraphGen and our data clean pipeline.

|  Domain   |                          Dataset                          |   Ours   | Qwen2.5-7B-Instruct (baseline) |
|:---------:|:---------------------------------------------------------:|:--------:|:------------------------------:|
|   Plant   | [SeedBench](https://github.com/open-sciencelab/SeedBench) | **65.9** |              51.5              |
|  Common   |                           CMMLU                           |   73.6   |            **75.8**            |
| Knowledge |                       GPQA-Diamond                        | **40.0** |              33.3              |
|   Math    |                          AIME24                           | **20.6** |              16.7              |
|           |                          AIME25                           | **22.7** |              7.2               |



## ⚙️ Support List

We support various LLM inference servers, API servers, inference clients, input file formats, data modalities, output data formats, and output data types.
Users can flexibly configure according to the needs of synthetic data.


| Inference Server                                                         | Api Server                                                                     | Inference Client                                           | Data Source                                                                                                                                                                                                                                                                           | Data Modal    | Data Type                                       |
|--------------------------------------------------------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|-------------------------------------------------|
| [![hf-icon]HF][hf]<br>[![sg-icon]SGLang][sg]<br>[![vllm-icon]vllm][vllm] | [![sif-icon]Silicon][sif]<br>[![oai-icon]OpenAI][oai]<br>[![az-icon]Azure][az] | HTTP<br>[![ol-icon]Ollama][ol]<br>[![oai-icon]OpenAI][oai] | Files(CSV, JSON, PDF, TXT, etc.)<br>Databases([![uniprot-icon]UniProt][uniprot], [![ncbi-icon]NCBI][ncbi], [![rnacentral-icon]RNAcentral][rnacentral])<br>Search Engines([![bing-icon]Bing][bing], [![google-icon]Google][google])<br>Knowledge Graphs([![wiki-icon]Wikipedia][wiki]) | TEXT<br>IMAGE | Aggregated<br>Atomic<br>CoT<br>Multi-hop<br>VQA |

<!-- links -->
[hf]: https://huggingface.co/docs/transformers/index
[sg]: https://docs.sglang.ai
[vllm]: https://github.com/vllm-project/vllm
[sif]: https://siliconflow.cn
[oai]: https://openai.com
[az]: https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/
[ol]: https://ollama.com
[uniprot]: https://www.uniprot.org/
[ncbi]: https://www.ncbi.nlm.nih.gov/
[rnacentral]: https://rnacentral.org/
[wiki]: https://www.wikipedia.org/
[bing]: https://www.bing.com/
[google]: https://www.google.com


<!-- icons -->
[hf-icon]: https://www.google.com/s2/favicons?domain=https://huggingface.co
[sg-icon]: https://www.google.com/s2/favicons?domain=https://docs.sglang.ai
[vllm-icon]: https://www.google.com/s2/favicons?domain=https://docs.vllm.ai
[sif-icon]: https://www.google.com/s2/favicons?domain=siliconflow.com
[oai-icon]: https://www.google.com/s2/favicons?domain=https://openai.com
[az-icon]: https://www.google.com/s2/favicons?domain=https://azure.microsoft.com
[ol-icon]: https://www.google.com/s2/favicons?domain=https://ollama.com

[uniprot-icon]: https://www.google.com/s2/favicons?domain=https://www.uniprot.org
[ncbi-icon]: https://www.google.com/s2/favicons?domain=https://www.ncbi.nlm.nih.gov/
[rnacentral-icon]: https://www.google.com/s2/favicons?domain=https://rnacentral.org/
[wiki-icon]: https://www.google.com/s2/favicons?domain=https://www.wikipedia.org/
[bing-icon]: https://www.google.com/s2/favicons?domain=https://www.bing.com/
[google-icon]: https://www.google.com/s2/favicons?domain=https://www.google.com


## 🚀 Quick Start

Experience GraphGen Demo through [Huggingface](https://huggingface.co/spaces/chenzihong/GraphGen) or [Modelscope](https://modelscope.cn/studios/chenzihong/GraphGen).

For any questions, please check [FAQ](https://github.com/open-sciencelab/GraphGen/issues/10), open new [issue](https://github.com/open-sciencelab/GraphGen/issues) or join our [wechat group](https://cdn.vansin.top/internlm/dou.jpg) and ask.

### Preparation

1. Install [uv](https://docs.astral.sh/uv/reference/installer/)

    ```bash
    # You could try pipx or pip to install uv when meet network issues, refer the uv doc for more details
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. Clone the repository

    ```bash
    git clone --depth=1 https://github.com/open-sciencelab/GraphGen
    cd GraphGen
    ```

3. Create a new uv environment

    ```bash
     uv venv --python 3.10
    ```
   
4. Configure the dependencies

    ```bash
    uv pip install -r requirements.txt
    ```

### Run Gradio Demo

   ```bash
   python -m webui.app
   ```


![ui](https://github.com/user-attachments/assets/3024e9bc-5d45-45f8-a4e6-b57bd2350d84)

### Run from PyPI

1. Install GraphGen
   ```bash
   uv pip install graphg
   ```

2. Run in CLI
   ```bash
   SYNTHESIZER_MODEL=your_synthesizer_model_name \
   SYNTHESIZER_BASE_URL=your_base_url_for_synthesizer_model \
   SYNTHESIZER_API_KEY=your_api_key_for_synthesizer_model \
   TRAINEE_MODEL=your_trainee_model_name \
   TRAINEE_BASE_URL=your_base_url_for_trainee_model \
   TRAINEE_API_KEY=your_api_key_for_trainee_model \
   graphg --output_dir cache
   ```

### Run from Source

1. Configure the environment
   - Create an `.env` file in the root directory
     ```bash
     cp .env.example .env
     ```
   - Set the following environment variables:
     ```bash
      # Tokenizer
      TOKENIZER_MODEL=
      
      # LLM
      # Support different backends: http_api, openai_api, ollama_api, ollama, huggingface, tgi, sglang, tensorrt
      # Synthesizer is the model used to construct KG and generate data
      # Trainee is the model used to train with the generated data

      # http_api / openai_api
      SYNTHESIZER_BACKEND=openai_api
      SYNTHESIZER_MODEL=gpt-4o-mini
      SYNTHESIZER_BASE_URL=
      SYNTHESIZER_API_KEY=
      TRAINEE_BACKEND=openai_api
      TRAINEE_MODEL=gpt-4o-mini
      TRAINEE_BASE_URL=
      TRAINEE_API_KEY=
      
      # azure_openai_api
      # SYNTHESIZER_BACKEND=azure_openai_api
      # The following is the same as your "Deployment name" in Azure
      # SYNTHESIZER_MODEL=<your-deployment-name>
      # SYNTHESIZER_BASE_URL=https://<your-resource-name>.openai.azure.com/openai/deployments/<your-deployment-name>/chat/completions
      # SYNTHESIZER_API_KEY=
      # SYNTHESIZER_API_VERSION=<api-version>
      
      # # ollama_api
      # SYNTHESIZER_BACKEND=ollama_api
      # SYNTHESIZER_MODEL=gemma3
      # SYNTHESIZER_BASE_URL=http://localhost:11434
      #
      # Note: TRAINEE with ollama_api backend is not supported yet as ollama_api does not support logprobs.
      
      # # huggingface
      # SYNTHESIZER_BACKEND=huggingface
      # SYNTHESIZER_MODEL=Qwen/Qwen2.5-0.5B-Instruct
      #
      # TRAINEE_BACKEND=huggingface
      # TRAINEE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
      
      # # sglang
      # SYNTHESIZER_BACKEND=sglang
      # SYNTHESIZER_MODEL=Qwen/Qwen2.5-0.5B-Instruct
      # SYNTHESIZER_TP_SIZE=1
      # SYNTHESIZER_NUM_GPUS=1
      
      # TRAINEE_BACKEND=sglang
      # TRAINEE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
      # SYNTHESIZER_TP_SIZE=1
      # SYNTHESIZER_NUM_GPUS=1
      
      # # vllm
      # SYNTHESIZER_BACKEND=vllm
      # SYNTHESIZER_MODEL=Qwen/Qwen2.5-0.5B-Instruct
      # SYNTHESIZER_NUM_GPUS=1
      
      # TRAINEE_BACKEND=vllm
      # TRAINEE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
      # TRAINEE_NUM_GPUS=1
     ```
2. (Optional) Customize generation parameters in `config.yaml` .

   Edit the corresponding YAML file, e.g.:

    ```yaml
      # examples/generate/generate_aggregated_qa/aggregated_config.yaml
      global_params:
        working_dir: cache
        graph_backend: kuzu # graph database backend, support: kuzu, networkx
        kv_backend: rocksdb # key-value store backend, support: rocksdb, json_kv
   
      nodes:
        - id: read_files # id is unique in the pipeline, and can be referenced by other steps
          op_name: read
          type: source
          dependencies: []
          params:
            input_path:
              - examples/input_examples/jsonl_demo.jsonl # input file path, support json, jsonl, txt, pdf. See examples/input_examples for examples

      # additional settings...
    ```

3. Generate data

   Pick the desired format and run the matching script:
      
   | Format          | Script to run                                                                | Notes                                                                      |
   |-----------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------|
   | `cot`           | `bash examples/generate/generate_cot_qa/generate_cot.sh`                     | Chain-of-Thought Q\&A pairs                                                |
   | `atomic`        | `bash examples/generate/generate_atomic_qa/generate_atomic.sh`               | Atomic Q\&A pairs covering basic knowledge                                 |
   | `aggregated`    | `bash examples/generate/generate_aggregated_qa/generate_aggregated.sh`       | Aggregated Q\&A pairs incorporating complex, integrated knowledge          |
   | `multi-hop`     | `examples/generate/generate_multi_hop_qa/generate_multi_hop.sh`              | Multi-hop reasoning Q\&A pairs                                             |
   | `vqa`           | `bash examples/generate/generate_vqa/generate_vqa.sh`                        | Visual Question Answering pairs combining visual and textual understanding |
   | `multi_choice`  | `bash examples/generate/generate_multi_choice_qa/generate_multi_choice.sh`   | Multiple-choice question-answer pairs                                      |
   | `multi_answer`  | `bash examples/generate/generate_multi_answer_qa/generate_multi_answer.sh`   | Multiple-answer question-answer pairs                                      |
   | `fill_in_blank` | `bash examples/generate/generate_fill_in_blank_qa/generate_fill_in_blank.sh` | Fill-in-the-blank question-answer pairs                                    |
   | `true_false`    | `bash examples/generate/generate_true_false_qa/generate_true_false.sh`       | True-or-false question-answer pairs                                        |

4. Get the generated data
   ```bash
   ls cache/output
   ```

### Run with Docker
1. Build the Docker image
   ```bash
   docker build -t graphgen .
   ```
2. Run the Docker container
   ```bash
    docker run -p 7860:7860 graphgen
    ```


## 🧪 Chemistry Dataset Pipeline

GraphGen includes a domain-specific pipeline for generating chemistry reasoning datasets from pre-built molecular knowledge graphs. This section describes the full workflow: building the KGs, loading them into GraphGen, and running 14 generation configs to produce ChatML training data.

### Overview

The pipeline has three stages:

```
1. Build KG  →  GraphML file on disk
2. Load KG   →  Copy GraphML into GraphGen's graph storage working directory
3. Generate  →  Run GraphGen config  →  ChatML JSONL output
```

GraphGen uses two separate KGs as data sources:

- **KG1** — a rule-based graph encoding lipophilicity rules, functional group effects, and structure–property relationships. Nodes are functional group rules; edges connect related rules. Small (~60 nodes), used for foundational chemistry QA.
- **KG2** — a molecule-centric graph where nodes are individual compounds (SMILES string, experimental logD, lipophilicity bin) and edges encode structural similarity (Tanimoto ≥ 0.4), shared functional groups, and scaffold membership. Large (~2,400 molecules in full graph, 50 in smoke-test subset).

---

### Prerequisites — Python Environment

The chemistry pipeline requires the **conda base environment** (Python 3.12.7). Do **not** use a virtualenv or uv-created `.venv` — Ray actor workers inherit the interpreter from the launching process, and a venv-launched Ray cluster fails to share the same package paths across workers, producing 0 generated rows.

```bash
# Verify you are in the conda base environment
which python
# Expected: /opt/anaconda3/bin/python

python --version
# Expected: Python 3.12.7
```

**Key packages required** (all present in conda base):

| Package | Version | Role |
|---------|---------|------|
| `ray` | 2.53.0 | Distributed pipeline execution |
| `networkx` | 3.3 | Graph storage backend |
| `rdkit` | 2025.3.2 | SMILES parsing & chemistry utilities |
| `torch` + `torch-geometric` | 2.8.0 / 2.7.0 | GNN evaluation (optional) |
| `litellm` | 1.80.16 | LLM proxy (routes to AWS Bedrock) |
| `boto3` / `botocore` | 1.42.67 | AWS Bedrock API access |
| `tiktoken` | 0.9.0 | Token counting for chunk sizing |
| `graphg` | 0.1.0 (editable) | GraphGen package (installed via `pip install -e .`) |

Install the package in editable mode if not already done:

```bash
pip install -e .
```

Limit Ray CPU usage to avoid consuming all cores:

```bash
export RAY_NUM_CPUS=4
```

---

### Step 1 — Build the Knowledge Graphs

The KG build scripts live in `chemistry/kg1_build/` and `chemistry/kg2_build/`. Run them once; the output GraphML files are reused for all generation runs.

```bash
# Build KG1: chemistry rule graph  (~60 nodes, ~190KB)
python chemistry/kg1_build/build_kg1.py
# Output: chemistry/kg1_build/chemistry_rule_graph.graphml

# Build KG2: molecule property graph  (~2400 molecules, ~6MB full)
python chemistry/kg2_build/build_kg2.py
# Output: chemistry/kg2_build/molecule_graph.graphml

# A 50-molecule smoke-test subset is already committed:
# chemistry/kg2_build/molecule_graph_smoke.graphml  (~196KB)
```

**What the KG build scripts do:**

| Script | Input | Output |
|--------|-------|--------|
| `build_kg1.py` | `fg_smarts.yaml` — SMARTS patterns for functional groups | GraphML with nodes = FG rules, edges = related-rule links |
| `build_kg2.py` | SMILES + logD dataset (CSV) | GraphML with nodes = molecules, edges = Tanimoto/scaffold/FG similarity |

The resulting `.graphml` files are standard NetworkX-compatible GraphML and can be inspected with any graph tool.

---

### Step 2 — Load the KG into GraphGen's Graph Storage

> **This step is required before every fresh run.** It is the most important setup detail.

GraphGen's partition operator reads the knowledge graph from a file named `graph.graphml` inside each config's `working_dir`. This file is **not** the raw input file — it is GraphGen's internal graph storage, loaded via NetworkX at startup.

Because the chemistry KGs are pre-built (we skip the LLM-based `build_kg` entity-extraction step), you must **manually copy** the source GraphML into the correct working directory before running:

```bash
# ── KG1 configs ──────────────────────────────────────────────────────────────
mkdir -p chemistry/output/kg1_atomic_cache chemistry/output/kg1_cot_cache
cp chemistry/kg1_build/chemistry_rule_graph.graphml \
   chemistry/output/kg1_atomic_cache/graph.graphml
cp chemistry/kg1_build/chemistry_rule_graph.graphml \
   chemistry/output/kg1_cot_cache/graph.graphml

# ── KG2 configs (smoke test — 50 molecules) ───────────────────────────────────
for dir in kg2_chemistry_atomic_cache kg2_multi_choice_cache \
           kg2_multi_answer_cache kg2_fill_in_blank_cache \
           kg2_true_false_cache kg2_pairwise_cache kg2_ranking_cache \
           kg2_mmp_cache kg2_multihop_cache kg2_aggregated_cache \
           kg2_logd_prediction_cache kg2_logd_cot_cache; do
  mkdir -p chemistry/output/$dir
  cp chemistry/kg2_build/molecule_graph_smoke.graphml \
     chemistry/output/$dir/graph.graphml
done

# ── KG2 configs (full dataset) ───────────────────────────────────────────────
# Replace molecule_graph_smoke.graphml with molecule_graph.graphml above
# for production runs on the full ~2400-molecule graph.
```

**Why this is needed:** GraphGen's `PartitionService` calls `graph_storage.reload()` to load the NetworkX graph, then runs community detection (DFS/BFS/Leiden) to partition the KG into context windows for the LLM. If `graph.graphml` does not exist, the graph is empty and the partition step produces 0 communities → 0 rows generated.

**What happens under the hood:**
```
GraphML file  →  NetworkX graph loaded into graph storage actor
                 ↓
         DFS partition (max_units_per_community=1)
                 ↓
      One community per node or edge  →  LLM context window
                 ↓
            LLM generates Q&A pair
```

---

### Step 3 — Configure the LLM Backend

The chemistry pipeline is designed to work with any OpenAI-compatible API. We use [LiteLLM](https://github.com/BerriAI/litellm) as a proxy to route requests to AWS Bedrock models.

**LiteLLM config** (`litellm_config.yaml` in the repo root):

```yaml
model_list:
  - model_name: claude-sonnet-4        # synthesizer (data generator)
    litellm_params:
      model: bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0
      aws_region_name: us-east-1
      aws_profile_name: genai

  - model_name: llama-3-8b             # trainee (used for quiz/judge steps)
    litellm_params:
      model: bedrock/meta.llama3-8b-instruct-v1:0
      aws_region_name: us-east-1
      aws_profile_name: genai

general_settings:
  master_key: bedrock-graphgen-2024    # must match SYNTHESIZER_API_KEY in .env
  port: 4000
```

Start the proxy:

```bash
litellm --config litellm_config.yaml
```

Verify both models are reachable:

```bash
# Test synthesizer
curl -s http://localhost:4000/chat/completions \
  -H "Authorization: Bearer bedrock-graphgen-2024" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}'

# Test trainee
curl -s http://localhost:4000/chat/completions \
  -H "Authorization: Bearer bedrock-graphgen-2024" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3-8b","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}'
```

**`.env` file** (repo root):

```bash
SYNTHESIZER_BACKEND=http_api
SYNTHESIZER_BASE_URL=http://localhost:4000
SYNTHESIZER_API_KEY=bedrock-graphgen-2024
SYNTHESIZER_MODEL=claude-sonnet-4       # synthesizer: data generator

TRAINEE_BACKEND=http_api
TRAINEE_BASE_URL=http://localhost:4000
TRAINEE_API_KEY=bedrock-graphgen-2024
TRAINEE_MODEL=llama-3-8b                # trainee: the model being trained on generated data
```

> **Note:** The `master_key` in `litellm_config.yaml` and the `SYNTHESIZER_API_KEY` / `TRAINEE_API_KEY` in `.env` must be identical. A mismatch causes HTTP 400 auth errors on every LLM call and produces 0 generated rows.

---

### Step 4 — Run the Configs

Set `RAY_NUM_CPUS` to avoid using all available CPU cores (Ray defaults to using everything it finds):

```bash
export RAY_NUM_CPUS=4
```

Run any single config:

```bash
# KG1 — foundational chemistry rules
python graphgen/run.py --config_file chemistry/configs/chemistry_atomic_config.yaml
python graphgen/run.py --config_file chemistry/configs/chemistry_cot_config.yaml

# KG2 — molecule-level property QA
python graphgen/run.py --config_file chemistry/configs/chemistry_atomic_config2.yaml
python graphgen/run.py --config_file chemistry/configs/chemistry_multi_choice_config.yaml
python graphgen/run.py --config_file chemistry/configs/chemistry_multi_answer_config.yaml
python graphgen/run.py --config_file chemistry/configs/chemistry_fill_in_blank_config.yaml
python graphgen/run.py --config_file chemistry/configs/chemistry_true_false_config.yaml
python graphgen/run.py --config_file chemistry/configs/chemistry_pairwise_config.yaml
python graphgen/run.py --config_file chemistry/configs/chemistry_ranking_config.yaml
python graphgen/run.py --config_file chemistry/configs/chemistry_mmp_config.yaml
python graphgen/run.py --config_file chemistry/configs/chemistry_multihop_config.yaml
python graphgen/run.py --config_file chemistry/configs/chemistry_aggregated_config.yaml

# KG2 — logD prediction (SMILES-only input, no CoT)
python graphgen/run.py --config_file chemistry/configs/chemistry_logd_prediction_config.yaml

# KG2 — logD chain-of-thought reasoning (SMILES-only input, 4-step CoT answer)
python graphgen/run.py --config_file chemistry/configs/chemistry_logd_cot_config.yaml
```

Output JSONL files are written to `{working_dir}/output/{run_id}/generate/`.

For example, after running `chemistry_atomic_config.yaml`, generated data appears at:
```
chemistry/output/kg1_atomic_cache/output/<timestamp>/generate/*.jsonl
```

---

### Supported Generation Formats

All 12 configs produce **ChatML-format** output (`messages` list with `role`/`content`).

#### KG1 Configs (rule-based chemistry knowledge)

| Config file | Method | Description |
|-------------|--------|-------------|
| `chemistry_atomic_config.yaml` | `atomic` | Single-turn QA covering chemistry rules and functional group effects |
| `chemistry_cot_config.yaml` | `cot` | Chain-of-thought reasoning over chemistry rules |

#### KG2 Configs (molecule-level property data)

| Config file | Method | Description |
|-------------|--------|-------------|
| `chemistry_atomic_config2.yaml` | `chemistry_atomic` | QA targeting logD values, lipophilicity bins, SMILES functional groups |
| `chemistry_multi_choice_config.yaml` | `chemistry_multi_choice` | 4-option MCQ with chemically plausible distractors |
| `chemistry_multi_answer_config.yaml` | `chemistry_multi_answer` | Multiple-select: 1–3 correct answers from 4 options |
| `chemistry_fill_in_blank_config.yaml` | `chemistry_fill_in_blank` | Fill-in-the-blank for logD values, bin names, property directions |
| `chemistry_true_false_config.yaml` | `chemistry_true_false` | ~50/50 true/false statements about molecular properties |
| `chemistry_pairwise_config.yaml` | `pairwise_preference` | Compare two molecules (Molecule A vs B) on lipophilicity and drug-likeness |
| `chemistry_ranking_config.yaml` | `ranking` | Order 3+ molecules by logD with per-position SMILES justification |
| `chemistry_mmp_config.yaml` | `matched_molecular_pair` | SAR explanation: structural delta → logD change (uses Tanimoto/scaffold edges) |
| `chemistry_multihop_config.yaml` | `multihop` | Multi-hop reasoning across connected molecule nodes |
| `chemistry_aggregated_config.yaml` | `aggregated` | Aggregated QA combining multiple molecule properties |
| `chemistry_logd_prediction_config.yaml` | `chemistry_logd_prediction` | Direct logD prediction: question = SMILES only; answer = logD + bin + structural justification |
| `chemistry_logd_cot_config.yaml` | `chemistry_logd_cot` | CoT logD prediction: question = SMILES only; answer = 4-step reasoning chain |

---

### Output Example (ChatML format)

```json
{
  "messages": [
    {
      "role": "user",
      "content": "The molecule with SMILES CC(=O)Oc1ccccc1C(=O)O has a measured logD of 1.19. Which lipophilicity bin does it belong to?"
    },
    {
      "role": "assistant",
      "content": "The molecule CC(=O)Oc1ccccc1C(=O)O belongs to the **Low** lipophilicity bin (logD 1–2). The ester and carboxylic acid groups both increase polarity, keeping the overall logD below 2."
    }
  ]
}
```

For multi-molecule formats (pairwise, ranking, MMP), molecules are labelled **Molecule A / B / C** and identified by their SMILES in both prompt and answer:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Compare Molecule A (SMILES: CCO) and Molecule B (SMILES: c1ccccc1) on lipophilicity. Which is more drug-like?"
    },
    {
      "role": "assistant",
      "content": "Molecule B (SMILES: c1ccccc1, logD = 2.13, medium) is more lipophilic than Molecule A (SMILES: CCO, logD = -0.31, low)..."
    }
  ]
}
```

---

### Verified Output Counts (50-molecule KG2 smoke test)

| Format | Config | Rows | Source KG |
|--------|--------|:----:|:---------:|
| `atomic` | `chemistry_atomic_config.yaml` | 634 | KG1 |
| `cot` | `chemistry_cot_config.yaml` | 6 | KG1 |
| `chemistry_atomic` | `chemistry_atomic_config2.yaml` | 400 | KG2 |
| `chemistry_multi_choice` | `chemistry_multi_choice_config.yaml` | 152 | KG2 |
| `chemistry_multi_answer` | `chemistry_multi_answer_config.yaml` | 111 | KG2 |
| `chemistry_fill_in_blank` | `chemistry_fill_in_blank_config.yaml` | 136 | KG2 |
| `chemistry_true_false` | `chemistry_true_false_config.yaml` | 185 | KG2 |
| `pairwise_preference` | `chemistry_pairwise_config.yaml` | 50 | KG2 |
| `ranking` | `chemistry_ranking_config.yaml` | 37 | KG2 |
| `matched_molecular_pair` | `chemistry_mmp_config.yaml` | 50 | KG2 |
| `multihop` | `chemistry_multihop_config.yaml` | 41 | KG2 |
| `aggregated` | `chemistry_aggregated_config.yaml` | 38 | KG2 |
| `chemistry_logd_prediction` | `chemistry_logd_prediction_config.yaml` | 50 | KG2 |
| `chemistry_logd_cot` | `chemistry_logd_cot_config.yaml` | 50 | KG2 |

> KG1 row counts are fixed (the rule graph has ~60 nodes regardless of molecule count). KG2 counts scale with graph size — expect significantly more rows with the full 2400-molecule graph (`molecule_graph.graphml`) vs. the 50-molecule smoke test subset.

---

### Generator Classes

All chemistry generators live in `graphgen/models/generator/` and are registered in `graphgen/operators/generate/generate_service.py`:

| Class | Method key | Notes |
|-------|------------|-------|
| `ChemistryAtomicGenerator` | `chemistry_atomic` | Standalone; targets logD, bin, functional groups; identifies molecules by SMILES |
| `ChemistryMultiChoiceGenerator` | `chemistry_multi_choice` | Extends `MultiChoiceGenerator`; adds chemistry-specific distractor rules |
| `ChemistryMultiAnswerGenerator` | `chemistry_multi_answer` | Extends `MultiAnswerGenerator`; multi-correct answers for structural features |
| `ChemistryFillInBlankGenerator` | `chemistry_fill_in_blank` | Extends `FillInBlankGenerator`; blanks target logD values and bin names |
| `ChemistryTrueFalseGenerator` | `chemistry_true_false` | Extends `TrueFalseGenerator`; ~50% plausible-but-false statements |
| `PairwisePreferenceGenerator` | `pairwise_preference` | Multi-molecule: labels nodes Molecule A/B, includes edge attributes (Tanimoto, shared_fg) |
| `RankingGenerator` | `ranking` | Multi-molecule: orders 3–5 nodes by logD; answer includes SMILES per rank position |
| `MatchedMolecularPairGenerator` | `matched_molecular_pair` | Multi-molecule: SAR format; encodes Tanimoto similarity, shared functional groups, scaffold in prompt |
| `ChemistryLogdPredictionGenerator` | `chemistry_logd_prediction` | Single-molecule: question = SMILES only; answer = logD + bin + 2–3 structural features + ≥1 descriptor |
| `ChemistryLogdCotGenerator` | `chemistry_logd_cot` | Single-molecule: question = SMILES only; answer = 4-step CoT (SMILES parsing → FG contributions → descriptor analysis → logD prediction) |

**Context assembly for multi-molecule generators** — raw graph node IDs (e.g. `mol_39`) are replaced with human-readable labels so the LLM never sees opaque identifiers:

```python
# labels built in build_prompt():
labels = {node[0]: f"Molecule {chr(65 + i)}" for i, node in enumerate(nodes)}
# → {"mol_39": "Molecule A", "mol_42": "Molecule B", ...}

# node lines:
context += f"- {label}: {desc}\n"
# → "- Molecule A: smiles: CCO | logd_exp: -0.31 | logd_bin: low"

# edge lines:
context += f"  relationship: {src} -- {tgt}: {desc}\n"
# → "  relationship: Molecule A -- Molecule B: tanimoto: 0.42 | shared_fg: hydroxyl"
```

---

### LogD Prediction Pipeline — How It Works

The `chemistry_logd_prediction` and `chemistry_logd_cot` configs are purpose-built for training a language model to **predict logD from SMILES alone** — without seeing the experimental value, bin label, or any descriptors in the question.

#### Design Goal

| Format | Question contains | Answer contains |
|--------|-------------------|-----------------|
| `chemistry_logd_prediction` | SMILES string only | logD value + bin + 2–3 structural features + ≥1 descriptor |
| `chemistry_logd_cot` | SMILES string only | 4-step CoT reasoning chain |

This separation forces the model to learn the *structure → property* mapping rather than memorising labelled examples.

#### Data Flow (end-to-end)

```
molecule_graph.graphml
       │
       │  node attrs per molecule:
       │  smiles, logd_exp, logd_bin, functional_groups, scaffold,
       │  logp, mw, hbd, hba, tpsa, rotbonds, content
       │
       ▼
  GraphmlReader  (include_edges=False)
       │  → one Ray dataset row per molecule
       │    content = "smiles: CCO | logd_exp: -0.31 | ..."
       ▼
  ChunkService  (chunk_size=4096)
       │  → pass-through (molecules fit in one chunk)
       ▼
  ECE Partition  (max_units=1)
       │  → one community per molecule
       │    community = {nodes: ["mol_N"], edges: []}
       ▼
  community2batch()
       │  → batch = ([(node_id, attrs)], [])
       │    attrs = {"smiles": "CCO", "logd_exp": -0.31, ...}   ← raw graphml attrs
       ▼
  ChemistryLogdPredictionGenerator.build_prompt(batch)
       │  smiles = attrs["smiles"]       ← direct key lookup
       │  context = "\n".join([          ← all molecular data
       │    "smiles: CCO",
       │    "logd_exp: -0.31",
       │    "logd_bin: low",
       │    "functional_groups: hydroxyl",
       │    "scaffold: CCO",
       │    "logp: -0.31",
       │    ...
       │    "description: Molecule SMILES: CCO. Experimental logD ..."
       │  ])
       │  → LOGD_PREDICTION_PROMPT["en"].format(smiles=smiles, context=context)
       ▼
  LLM (claude-sonnet-4)
       │  → generates <question>...</question><answer>...</answer>
       ▼
  parse_response()
       │  → [{"question": "What is the logD for SMILES: CCO?",
       │       "answer": "Predicted logD = -0.31 (low)..."}]
       ▼
  format_generation_results(data_format="ChatML")
       │
       ▼
  Output JSONL:
  {"messages": [
    {"role": "user",    "content": "What is the logD at pH 7.4 for the molecule with SMILES: CCO?"},
    {"role": "assistant","content": "The predicted logD is -0.31 (low lipophilicity)..."}
  ]}
```

#### CoT answer structure (chemistry_logd_cot)

The 4-step answer is generated by the LLM following a strict template:

```
Step 1 — Structural features:   identify FGs, ring systems, heteroatoms, ionizable groups
Step 2 — FG contributions:      explain each group's effect on logD at pH 7.4
Step 3 — Descriptor analysis:   cite MW, LogP, TPSA, HBD, HBA, RotBonds with values
Step 4 — LogD prediction:       final logD ≈ X, bin = low/mid/high, PK implication
```

#### Key implementation details

- **Single-molecule partition** (`max_units=1`, `min_units=1`): guarantees each LLM call receives exactly one molecule. No multi-molecule communities are formed, so no label mapping (Molecule A/B/C) is needed.
- **SMILES extraction**: `attrs.get("smiles", "")` reads the value directly from the NetworkX node attribute dict (populated when the graph is loaded from `graph.graphml` via `nx.read_graphml`). A regex fallback parses SMILES from the `content` pipe-delimited string if the direct key is absent.
- **Context vs question separation**: the full molecular data (`context`) is passed to the LLM for answer generation, but the **template explicitly forbids** including logD values, bins, or descriptors in the `<question>` tag. The LLM sees all the data but must produce a SMILES-only question.
- **Empty prompt guard**: `generate()` is overridden to return `[]` immediately if `build_prompt` produces an empty string (e.g., molecules missing a SMILES), preventing an empty user message from being sent to the API.
- **ChatML output**: both configs produce `messages` arrays usable directly as SFT training data with LLaMA-Factory or xtuner.

#### Example output

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the logD at pH 7.4 for the molecule with SMILES: O=C(N[C@@H]1CCC[C@@H](NC(=O)c2cccc(Cl)c2)C1)c1cccc(F)c1?"
    },
    {
      "role": "assistant",
      "content": "The predicted logD is 5.0 (high lipophilicity). The molecule contains two aromatic rings with halogen substituents (chlorine and fluorine) that increase lipophilicity significantly. The two amide bonds (C(=O)N) provide some polarity but are insufficient to offset the hydrophobic cyclohexane ring and the halogenated arenes. With LogP = 4.2 and TPSA = 58 Å², the compound is expected to have high membrane permeability but potential metabolic liability."
    }
  ]
}
```

---

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Total communities partitioned: 0` | `graph.graphml` missing from `working_dir` | Run the copy commands in Step 2 |
| `HTTP 400 Bad Request` from LiteLLM | `master_key` in `litellm_config.yaml` doesn't match `SYNTHESIZER_API_KEY` | Update `master_key` to match, restart LiteLLM |
| `0 rows written` despite communities > 0 | LLM auth failure during generation | Check LiteLLM logs; verify API key and model name |
| Ray workers show `ModuleNotFoundError` | Wrong Python environment | Use the conda base `python`, not a venv (Ray workers must share the same environment) |
| `Cluster resources are not enough` warning | Ray initialised with too few CPUs | Set `RAY_NUM_CPUS=4` (or higher); this is usually a non-fatal warning |

## 🏗️ System Architecture

See [analysis](https://deepwiki.com/open-sciencelab/GraphGen) by deepwiki for a technical overview of the GraphGen system, its architecture, and core functionalities. 


### Workflow
![workflow](assets/flow.png)


## 🍀 Acknowledgements
- [SiliconFlow](https://siliconflow.cn) Abundant LLM API, some models are free
- [LightRAG](https://github.com/HKUDS/LightRAG) Simple and efficient graph retrieval solution
- [ROGRAG](https://github.com/tpoisonooo/ROGRAG) A robustly optimized GraphRAG framework
- [DB-GPT](https://github.com/eosphoros-ai/DB-GPT) An AI native data app development framework


## 📚 Citation
If you find this repository useful, please consider citing our work:
```bibtex
@misc{chen2025graphgenenhancingsupervisedfinetuning,
      title={GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation}, 
      author={Zihong Chen and Wanli Jiang and Jinzhe Li and Zhonghang Yuan and Huanjun Kong and Wanli Ouyang and Nanqing Dong},
      year={2025},
      eprint={2505.20416},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.20416}, 
}
```

## 📜 License
This project is licensed under the [Apache License 2.0](LICENSE).

## 📅 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=InternScience/GraphGen&type=Date)](https://www.star-history.com/#InternScience/GraphGen&Date)
