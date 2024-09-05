# Task Extraction and Financial Recommendation Software

This repository contains two main programs: a Task Extraction software and a Financial Recommendation software. 

## 1. Task Extraction Software

The Task Extraction software is designed to automatically extract tasks, deadlines, and recipients from natural language text inputs using Named Entity Recognition (NER).

### Files:
- `doc_dbte2.pdf`: Documentation for the Task Extraction software
- `train_data.py`: Contains the training data for the NER model
- `dbapp.py`: The main application file for task extraction
- `dbmodel.py`: Contains the model definition and training code

### Dependencies:
- Python 3.7+
- transformers
- torch
- datasets
- sklearn
- numpy
- seqeval
- nltk

### Usage:
1. Ensure all dependencies are installed.
2. Run `dbmodel.py` to train the NER model (if not already trained).
3. Execute `dbapp.py` to start the task extraction system.
4. Enter task-related text when prompted.
5. Review the extracted tasks, deadlines, and recipients displayed in the console.

## 2. Financial Recommendation Software

The Financial Recommendation software provides personalized stock recommendations based on user investment preferences. It uses various machine learning techniques including dimensionality reduction, clustering, and collaborative filtering.

### Files:
- `doc_rs.pdf`: Documentation for the Financial Recommendation software
- `mainfr.py`: The main script for the recommendation system

### Dataset:
The program uses the "2017_Financial_Data.csv" file from Kaggle, which contains various financial metrics and ratios for a set of companies.

### Dependencies:
- pandas
- numpy
- scikit-learn
- scipy
- joblib

### Usage:
1. Ensure all dependencies are installed.
2. Place the "2017_Financial_Data.csv" file in the appropriate directory.
3. Run `mainfr.py`.
4. When prompted, enter your investment preferences.
5. Receive personalized stock recommendations based on your preferences.

## Installation

To set up the environment for both programs:

```bash
pip install pandas numpy scikit-learn scipy joblib transformers torch datasets nltk
```


