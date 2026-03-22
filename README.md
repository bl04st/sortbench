# SortBench

SortBench is a sorting-based benchmark for Large Language Models (LLMs).

## Reproducing the benchmark

In the following we describe the steps to reproduce our results. Everything we describe works as is on Ubuntu 24.04 LTS.
To replicate the v2.0 benchmark change the --version argument to `v2.0`.

### Setting up the environment

First, checkout the repository from GitHub:

```bash
git clone git@github.com:aieng-lab/sortbench.git
cd sortbench
```

We recommend using a virtual environment to run the benchmark. To create a virtual environment, run the following command:

```bash
python3 -m venv .venv
```

We can now activate the virtual environment and install the required packages:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Creating data

To create the data, run the following command:

```bash
python sortbench/generate_data.py --mode=basic --version=v1.0 --random_seed=42415
python sortbench/generate_data.py --mode=advanced --version=v1.0 --random_seed=56671
python sortbench/generate_data.py --mode=debug --version=v1.0 --random_seed=56671
```

For reproducibiliy, each version of the benchmark uses a different, random but fixed seed. Note that different platforms (e.g., operating systems, python version, or CPUs) may yield different results, even if the seed is fixed. You can find the seeds we used below. 

| Version | Basic Seed | Advanced Seed | Debug Seed |
| ------- | ---------- | ------------- | ---------- |
| v1.0    | 42415      | 56671         | 19837      |
| v2.0    | 43467      | 83841         | -          |

### Running the benchmark

To run the benchmark, you need to have valid API keys for the inference endpoints. Currently, we use models from OpenAI and models we host locally in Passau at an inference endpoint in the Inncube cluster. For both, the API keys are required. You can set them as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export INNCUBE_API_KEY="your_inncube_api_key"
export ANTROPIC_API_KEY="your_anthropic_api_key"
```

We currently support the following models from Open AI:
- `gpt-4o`: OpenAI's GPT-4 (large)
- `gpt-4o-mini`: OpenAI's GPT-4 (mini)
- `o3-mini`: OpenAI's o3-mini model
- `o1`: OpenAI's o1 model

The Inncube cluster currently hosts the following models:
- `llama3.1`: Meta's LLAMA 3.1
- `gemma2`: Google's Gemma 2
- `qwen2.5`: Alibaba's Qwen 2.5
- `deepseekr1`: DeepSeek's r1 model

In v2.0 we added:
- `gpt-5.1_reasoning_none`: OpenAI's GPT-5.1 (without reasoning)
- `gpt-5.1_reasoning_none`: OpenAI's GPT-5.1 (low reasoning)
- `gpt-5-mini_reasoning_minimal`: OpenAI's GPT-5-mini (minimal/without reasoning)
- `gemini-3-pro`: Google's Gemini-pro (low reasoning)
- `gemini-3-flash-preview_reasoning_minimal`: Google's Gemini-pro (minimal/without reasoning)
- `claude-haiku-4-5-20251001_reasoning_disabled`: Anthropics Claude Haiku (without reasoning)
- `claude-haiku-4-5-20251001_reasoning_enabled`: Anthropics Claude Haiku (with reasoning)
- `claude-sonnet-4-5-20251001_reasoning_disabled`: Anthropics Claude Sonnet (without reasoning)
- `claude-sonnet-4-5-20251001_reasoning_enabled`: Anthropics Claude Sonnet (with reasoning)

To run the benchmark, run the following command:

```bash
python sortbench/create_results.py --mode=basic --version=v1.0 --model_names gpt-4o gpt-4o-mini gpt-o3-mini llama-3.1 deepseekr1 claude-3-5-haiku-20241022 claude-3-5-sonnet-20241022
python sortbench/create_results.py --mode=advanced --version=v1.0 --model_names gpt-4o gpt-4o-mini
python sortbench/create_results.py --mode=debug --version=v1.0 --model_names gpt-4o gpt-4o-mini
```

In v2.0 we added the bench_type argument, where you can choose what benchmark types you want to run. Default is `sort`.
We ran the following benchmarks: `sort, sort-descending, filter-higher, filter-lower, any, all`.

If you want to add your own models or endpoints to the benchmark, you need to modify the sortbench/util/inference_utils.py accordingly. 

### Evaluating the results

To evaluate the results, run the following command:

```bash
python sortbench/calculate_scores.py --mode=basic --version=v1.0 --csv_file="scores/scores_basic_v1.0.csv"
python sortbench/calculate_scores.py --mode=advanced --version=v1.0 --csv_file="scores/scores_basic_v1.0.csv"
python sortbench/calculate_scores.py --mode=debug --version=v1.0 --csv_file="scores/scores_basic_v1.0.csv"
```

## Running the Notebooks

To use the Jupyter Notebooks we provide in the `notebooks` folder, you need to install additional dependencies. They are provided in the `notebooks/requirements.txt` file. You can install them in the same virtual environment as above (needs to be activated!) as follows:

```bash
pip install -r notebooks/requirements.txt
```

Afterwards, you can run the Jupyter Notebook server by running the following command:

```bash
jupyter notebook
```
