# Hyperbola-laLand 
**Hyperbole Generation via Transformer Models (Inspired by MOVER: NAACL 2022)**

Hyperbola-laLand is an NLP project that generates exaggerated (hyperbolic) versions of literal sentences using pre-trained transformer models such as BERT, BART, and DistilRoBERTa. Inspired by the paper [*MOVER: Mask, Over-generate, and Rank for Hyperbole Generation*](https://aclanthology.org/2022.naacl-main.440.pdf), this project implements a simplified version of the mask–generate–rank architecture.

---

## ✨ Features

- Hyperbole generation using pre-trained transformer models
- Plug-and-play support for BERT, BART, and DistilRoBERTa
- Configurable via `config.py` to switch models or modify generation parameters
- Lightweight CLI for easy testing
- Includes basic over-generation and top-k ranking logic

---

## 📚 Based On

This project draws from the **MOVER** paper:

> *MOVER: Mask, Over-generate and Rank for Hyperbole Generation*  
> Ananya B. Sai, Niket Tandon, Peter Clark  
> NAACL 2022 – [Read the paper](https://aclanthology.org/2022.naacl-main.440.pdf)

We simplify and re-implement the MOVER idea by:
- Masking noun phrases or high-impact spans in literal sentences
- Using transformers to generate exaggerated replacements
- Scoring and ranking outputs based on hyperbolicity and semantic preservation

---

## 🏗 Architecture

| Stage | Description |
|-------|-------------|
| **Masking** | Literal sentence is parsed, and semantically rich spans (e.g., noun phrases) are masked |
| **Over-generation** | A model like BART generates multiple possible replacements for each mask |
| **Ranking** | Candidates are scored by semantic similarity and hyperbole detection heuristics |
| **Selection** | The top-ranked hyperbolic sentence is returned |

---

## 🛠 Tech Stack

- Python 3.10+
- [Transformers (HuggingFace)](https://huggingface.co/transformers/)
- BART, BERT, DistilRoBERTa
- NLTK / SpaCy (optional, for span detection)
- NumPy / Scikit-learn (for embedding-based similarity ranking)

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/jhanaviB/Hyperbola-laLand.git
cd Hyperbola-laLand

# Set up environment
pip install -r requirements.txt

# Run basic hyperbole generation
python main.py --model bart --input "The weather is nice today."
