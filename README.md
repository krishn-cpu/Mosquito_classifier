# Mosquito Species Classification System

## Overview

This project implements a machine learning system for classifying mosquito species based on their genomic and morphological features. It uses a Random Forest classifier optimized with GridSearchCV to achieve high classification accuracy.

## Purpose

Mosquito species identification is critical for:
- Disease vector surveillance and control
- Ecological studies
- Public health initiatives

This system provides an automated way to classify mosquito species using their genetic markers and physical characteristics.

## Project Structure

```
mosquito_classification/
├── create_sample_data.py    # Generates synthetic genomic data
├── mosquito_classifier.py   # Main classification script
├── mosquito_genomic_data.csv # Dataset (generated)
└── README.md                # This documentation
```

## Requirements

This project requires:

- Python 3.x
- pandas
- scikit-learn
- numpy

These packages are installed in a virtual environment to avoid conflicts with system packages.

## Installation

1. Clone or download this repository
2. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install required packages:
   ```
   pip install pandas scikit-learn numpy
   ```

## Usage

### Generate Sample Data

If you don't have your own dataset, you can generate synthetic data:

```bash
python3 create_sample_data.py
```

This creates a file called `mosquito_genomic_data.csv` with 500 samples across 4 species.

### Run Classification

To train and evaluate the classification model:

```bash
python3 mosquito_classifier.py
```

The script will:
1. Load the dataset
2. Split it into training and testing sets
3. Train a Random Forest model using grid search for hyperparameter optimization
4. Evaluate model performance
5. Display the most important features

## Dataset Information

The dataset contains the following features:

- **Gene expression data**: 10 gene expression levels (gene_expr_1 to gene_expr_10)
- **SNP data**: 20 single nucleotide polymorphisms (snp_1 to snp_20)
- **Morphological features**: 
  - body_length
  - wing_width
  - proboscis_length
  - leg_length
  - thorax_width
- **Target variable**: species (Anopheles_gambiae, Aedes_aegypti, Culex_pipiens, Anopheles_stephensi)

## Model Details

The Random Forest classifier is optimized using GridSearchCV with the following parameters:

- n_estimators: [100, 200]
- max_depth: [None, 10, 20]
- min_samples_split: [2, 5]

The best parameters found are:
- max_depth: None
- min_samples_split: 2
- n_estimators: 200

## Results

The model achieves:
- Cross-validation accuracy: 99%
- Test set accuracy: 100%

Top features for classification:
1. body_length
2. gene_expr_7
3. gene_expr_6
4. leg_length
5. gene_expr_4

## Future Improvements

Potential enhancements include:
- Model persistence functionality
- Interactive prediction for new samples
- Additional visualization of results
- Support for more species
- Feature selection techniques
- Testing with real-world data

## License

This project is provided as open-source software for educational and research purposes.

