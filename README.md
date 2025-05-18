# Law GPT

A legal question-answering system powered by a fine-tuned language model trained on legal texts.

## Overview

Law GPT is an AI-powered application designed to provide responses to legal questions using a custom-trained language model. The system features a user-friendly web interface where users can ask questions and receive AI-generated answers based on the model's training on legal documents.

## Features

- **Legal AI Assistant**: Fine-tuned DistilGPT-2 model for legal domain knowledge
- **Interactive Chat Interface**: Clean, modern dark-themed UI for asking questions
- **Real-time Responses**: Immediate AI-generated answers to legal queries

## Components

- **Language Model**: Custom-trained DistilGPT-2 model on legal corpus data
- **Web Application**: Flask backend with HTML/CSS/JavaScript frontend
- **API Endpoint**: RESTful interface for querying the model

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers library
- Flask

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/law-gpt.git
   cd law-gpt
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download or train the model:
   - Either download the pre-trained model from the releases page
   - Or train it yourself using the included scripts:
     ```bash
     python gpt_training.py
     ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Access the web interface at `http://localhost:5000`

## Usage

1. Open the web interface in your browser
2. Type your legal question in the input field
3. Press Enter to submit
4. View the AI-generated response in the chat window

## Training

The model is trained on legal texts using the Hugging Face Transformers library. The training process involves:

- Data preprocessing with the included `scraper.py` script
- Fine-tuning DistilGPT-2 on the prepared legal corpus
- Saving the model for inference in the Flask application

## Project Structure

```
law-gpt/
├── app.py                  # Flask application server
├── gpt_training.py         # Model training script
├── scraper.py              # Data processing utility
├── data.txt  
├── templates/
│   └── index.html          # Web interface
└── law_GPT_model/          # Saved model directory
```

## Customization

- To adapt the UI theme, modify the CSS in `index.html`
- To adjust model parameters, update the settings in `app.py`
- To train on different legal texts, modify the data source in `gpt_training.py`

## License

[MIT License](LICENSE)

## Acknowledgements

- This project uses the Hugging Face Transformers library
- The model is based on DistilGPT-2 architecture
- Training data sourced from open legal documents

## Future Work

- Expand the training corpus with more diverse legal texts
- Implement more sophisticated response generation techniques
- Add support for document citation and reference
- Improve the accuracy and relevance of answers
