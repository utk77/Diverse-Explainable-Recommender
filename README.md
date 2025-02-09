# Mitigating Filter Bubbles: Diverse and Explainable Recommender Systems

## Overview
This repository contains the implementation of a novel diversity-enhanced recommender system built upon an existing Explainable Recommender model. The project introduces a Diversity Module to mitigate filter bubbles in recommender systems, improving content diversity and fairness in recommendations.


## Motivation
The surge in online content has necessitated the development of intelligent recommender systems capable of offering personalized suggestions to users. However, these systems often encapsulate users within a "filter bubble," limiting their exposure to a narrow range of content. This project seeks to break this cycle by integrating a novel diversity module into a knowledge graph-based explainable recommender system. By employing principles of Shannon Entropy, this diversity module fosters a broader spectrum of recommendations while maintaining transparency and user trust. The implementation utilizes the MovieLens 1M dataset to create a richer, more inclusive user experience, marking a significant step toward enhancing recommendation diversity and explainability.


## Features
- **Diversity Module**: Introduces an entropy-based diversity score to improve content variation.
- **Knowledge Graph Integration**: Uses structured knowledge from the MovieLens dataset for enhanced explainability.
- **Neural-Symbolic Learning**: Employs a hybrid model for learning user preferences and generating interpretable recommendations.
- **Customizable Training**: Supports hyperparameter tuning for model optimization.

## Dataset
This implementation is tested on the **MovieLens 1M dataset**, mapped to a knowledge graph to enhance explainability. The dataset is included under the `data` directory. Please ensure compliance with the [MovieLens Terms of Use](https://grouplens.org/datasets/movielens/).

## Installation
```bash
# Clone this repository
git clone https://github.com/utk77/Diverse-Explainable-Recommender.git
cd Diverse-Explainable-Recommender

# Install dependencies
pip install -r requirements.txt
```

## Usage
### 1. Preprocess Data
```bash
python data_preprocess.py
```
### 2. Train the Model
```bash
python train_neural_symbol_diversity.py --dataset ml1m --epochs 30 (can be changed)
```
### 3. Evaluate the Model
```bash
python execute_neural_symbol_diversity.py --dataset ml1m
```

## Citations
If you use this code, please cite the following:

- **CAFE Model:** [CAFE: Neural-Symbolic Learning with Commonsense Knowledge for Explainable Recommendation](https://arxiv.org/pdf/2010.15620)
- **Explainable Recsys Conference Repository:** [GitHub](https://github.com/explainablerecsys/recsys2022)
- **Original CAFE Codebase:** [GitHub](https://github.com/orcax/CAFE)

Additionally, please cite this work if you use the diversity-enhanced implementation:

**Umar Tahir Kidwai. Mitigating Filter Bubbles: Diverse and Explainable Recommender Systems.**

```bibtex
@article{kidwai2025mitigating,
  title={Mitigating Filter Bubbles: Diverse and Explainable Recommender Systems},
  author={Kidwai, Umar Tahir},
  year={2025},
  journal={Journal of Intelligent and Fuzzy Systems},
}
```

## License
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## Acknowledgments
Special thanks to the authors of the **CAFE model** and contributors to the **Explainable RecSys Conference** for inspiring this work.

## Contact
For any questions, feel free to open an issue or contact **Umar Tahir Kidwai** at **umartahirkidwai@zhcet.ac.in**.


## References
[1] Xian, Yikun, et al. "CAFE: Coarse-to-fine neural symbolic reasoning for explainable recommendation." Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2020. 

[2] Balloccu, G., Boratto, L., Fenu, G., & Marras, M. (2022). Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations. In proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 646–656). 
