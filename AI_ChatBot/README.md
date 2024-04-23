# AI Chatbot

This Python project implements a simple chatbot using natural language processing techniques and a neural network. The chatbot is trained on predefined intents and responds to user queries based on the trained model.

## Requirements

- Python 3.x
- Libraries: `numpy`, `nltk`, `tensorflow`

You can install the required libraries using pip:

```bash
pip install numpy nltk tensorflow
```

## Project Structure

- `chatbot.py`: Python script containing the chatbot implementation.
- `intents.json`: JSON file containing predefined intents with patterns and responses.
- `train.py`: Python script for training the neural network model using the intents data.

## Usage

- Ensure you have the necessary dependencies installed.
- Run the script `train.py` to train the neural network model.
- Run the script `chatbot.py` to start the chatbot.

## Workflow

### Training (train.py)

1. The `intents.json` file is loaded, containing predefined intents with patterns and responses.
2. Tokenization and lemmatization are performed on the patterns to preprocess the text data.
3. Words and classes are extracted from the intents data and sorted.
4. Word and class dictionaries are created, and files (`words.pkl` and `classes.pkl`) are pickled for future use.
5. Training data is prepared by creating a bag of words representation for each pattern and one-hot encoding the class labels.
6. A neural network model is defined using TensorFlow with dense layers and dropout regularization.
7. The model is compiled with categorical crossentropy loss and trained on the training data.

### Chatbot (chatbot.py)

1. The trained model (`chatbot_model.h5`) and required data files (`words.pkl`, `classes.pkl`, `intents.json`) are loaded.
2. User input is processed by tokenization, lemmatization, and converting it into a bag of words representation.
3. The trained model predicts the intent of the user input.
4. A response is selected based on the predicted intent from the predefined intents in `intents.json`.
5. The response is displayed to the user.
