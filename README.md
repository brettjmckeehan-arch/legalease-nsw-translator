# LegalEase NSW: an AI-powered legal document and legislation summariser

This is a UTS MDSI project for 36118 ANLP.

The application uses a sophisticated two-stage pipeline to simplify complex legal text. Stage 1 uses a local model (`facebook/bart-large-cnn`) to create a concise, factual summary. Stage 2 sends this summary to a powerful cloud-based large language model (LLM) from providers like Anthropic, OpenAI, or Google to translate it into plain, easy-to-understand English.

**Disclaimer**: This tool provides AI-generated summaries for informational purposes only and does not constitute legal advice. It is not a substitute for a qualified legal professional. Always consult a lawyer for advice on your specific situation.

---

## Features

* **Two-stage summarisation**: Combines a fast local model for initial summarisation with a powerful cloud LLM for high-quality final translation.
* **Multi-API and model selection**: Allows users to choose from a variety of state-of-the-art models from Anthropic (Claude), OpenAI (GPT), and Google (Gemini).
* **Dynamic prompt selection**: Provides a dropdown menu to workshop different prompt styles, allowing for experimentation with the AI's instructions.
* **Advanced PDF processing**: Supports both text-based and scanned (image-based) PDFs through an integrated **Optical Character Recognition (OCR)** engine.
**Legal Citation Graph**: With a pre-made legal citation network of primary and secondary legislation of NSW, allows for the selection and visualisation of connections of legislation, and searching through all NSW legislation. 
---

## Setup

Follow these steps to run LegalEase on your local machine.

### Prerequisites

* You must have **Conda** (Anaconda or Miniconda) installed. You can download it [here](https://www.anaconda.com/download).
* You need your own **API keys** from [Anthropic](https://www.anthropic.com/), [OpenAI](https://platform.openai.com/), and [Google AI Studio](https://aistudio.google.com/).
* You must have the **Tesseract-OCR engine** installed for PDF scanning to work. You can download the Windows installer [here](https://github.com/UB-Mannheim/tesseract/wiki).

---

### Installation and first run

#### 1. Get the code (choose one method)

* **Method A: download zip (no git)**
    * On the main repository page, click the green `<> Code` button and select **`Download ZIP`**.
    * Unzip the downloaded file on your computer.

* **Method B: clone with git**
    ```bash
    git clone [https://github.com/your-repo-url/legalease-nsw.git](https://github.com/your-repo-url/legalease-nsw.git)
    cd legalease-nsw
    ```

#### 2. Create the conda environment

This command reads the `environment.yml` file and automatically installs all necessary Python libraries.

# Navigate into the project folder first if you haven't already
# cd path/to/your/project/folder

conda env create -f environment.yml
conda activate legalease

#### 3. Set up your API keys
The application loads your secret API keys from a special file.

In the main project directory, create a new folder named .streamlit.

Inside the .streamlit folder, create a new file named secrets.toml.

Open secrets.toml and add your keys in the following format:

# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "sk-ant-..."
OPENAI_API_KEY = "sk-..."
GOOGLE_API_KEY = "AIzaSy..."

#### 4. Run the LegalEase application
Once the environment is active and your keys are set, start the Streamlit app: streamlit run streamlit_app.py

#### How to use
Enter text into the text area or upload a PDF file (including scanned documents).

Use the controls on the right to select a Summary style and the AI provider/model for the final translation.

Click the "Translate to Plain English" button.

The initial summary from the local BART model can be viewed in the expander below the final result.