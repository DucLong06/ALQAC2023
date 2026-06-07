# ALQAC 2023 — AIEPU

**AIEPU at ALQAC 2023: Deep Learning Methods for Legal Information Retrieval and Question Answering**

Code for our system at the [Automated Legal Question Answering Competition (ALQAC) 2023](https://alqac.github.io/). We use a **Paraformer**-based retriever (Task 1) and **prompt-tuned LLMs** (Task 2), achieving **1st place in Task 2**.

- 📄 Paper: [IEEE KSE 2023](https://ieeexplore.ieee.org/abstract/document/10299426)
- 📝 Prompts: [DucLong06/Legal-Prompts](https://github.com/DucLong06/Legal-Prompts)
- 📊 Eval sheet: [Google Sheets](https://docs.google.com/spreadsheets/d/1d2R8w6CxcX50dy4Ow1bgSHb-iuR8R4p81C0L2wNwp1M/edit?usp=sharing)
- 🔧 Reference model: [nguyenthanhasia/paraformer](https://github.com/nguyenthanhasia/paraformer) 

## Clone

The published prompts are pulled in as a submodule, so clone recursively:

```bash
git clone --recurse-submodules https://github.com/DucLong06/ALQAC2023.git
# already cloned:
git submodule update --init --recursive
```

## Usage

**Task 1 — Retrieval** (train, then evaluate; `alpha` mixes deep and BM25 scores):

```bash
python -m src.retrieval.train --base_model "sentence-transformers/paraphrase-xlm-r-multilingual-v1" \
    --input_questions data/raw/train.json --input_articles data/training/all_articles_2023.json

python -m src.retrieval.evaluate --model models/<checkpoint>.pth \
    --input_questions data/raw/public_test.json --input_articles data/training/all_articles_2023.json \
    --alpha 0.6 --top_articles 20 --compare True
```

**Task 2 — Question Answering:**

```bash
python prompt.py --model "google/flan-t5-xxl" \
    --questions data/raw_en/gg_question_private.json --articles data/training/gg_all_articles_2023.json \
    --prompts prompts/prompts_en.json --language en --compare False
```

See `scripts/` for ready-to-run shell scripts.

## Citation

```bibtex
@INPROCEEDINGS{10299426,
  author={Hoang, Long and Bui, Tung and Nguyen, Chau and Nguyen, Le-Minh},
  booktitle={2023 15th International Conference on Knowledge and Systems Engineering (KSE)}, 
  title={AIEPU at ALQAC 2023: Deep Learning Methods for Legal Information Retrieval and Question Answering}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  keywords={Deep learning;Knowledge engineering;Law;Data preprocessing;Information retrieval;Question answering (information retrieval);Modeling;legal information retrieval;legal question answering;natural language processing;large language model;prompt tuning},
  doi={10.1109/KSE59128.2023.10299426}}
```