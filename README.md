# Next word predictor in python
This project implements a language model for word sequences with n-grams using Laplace or Knesey-Ney smoothing. 

## Getting started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

1. Create a directory in the Assignment/ and add all the data files
2. Install the requirements.txt
3. run the app.py (for modeling cross validation and testing) or predictive_keyboard.py for the predictive keyboard functionality.

### Installing
In order to run the code in your local environment, please make sure your have python 3. and above and to have installed the needed python libraries.
To install the libraries please run on your console:

```
pip install -r requirements.txt file
```

## Train the model
In order to train the language model you will need to run the following command:

```
python Assignment1/app.py
```

## Run the keyword predictor

In order to run the predictive keyboard you will need to give the following command:

```
python Assignment1/predictive_keyboard.py
```

## Structure
The project consists of the following main classes:
- [Data Fetcher](https://github.com/agromanou/text-engineering-course/blob/master/Assignment1/data_fetcher.py)
- [Preprocessor](https://github.com/agromanou/text-engineering-course/blob/master/Assignment1/preprocess.py)
- [Model](https://github.com/agromanou/text-engineering-course/blob/master/Assignment1/modelling.py)
- [Evaluation](https://github.com/agromanou/text-engineering-course/blob/master/Assignment1/evaluation.py)

### Data Fetcher
This class is responsible for all the handling and fetching of the dataset(s). It loads the data, splits them into sub parts according to user needs and performs folding for the cross validation process.

### Preprocessor
This class is responsible for the text pre-processing of the data. It consists methods for sentences splitting, tokenization and n-gram creation.

### Model
This class is responsible for fitting the language model into the given data. It calculates the probabilities of the language model, performs smoothing (available implementations: Laplace or Kneser-Ney smoothing algorithms), runs linear interpolation on n-gram probabilities and predicts the next word for a given sequence.

### Evaluation
This class is responsible for the evaluation of a given model. It calculates the cross-entropy and perplexity of the model.


