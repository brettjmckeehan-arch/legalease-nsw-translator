# LegalEase NSW: AI-powered legal document simplifier and citation network

This is a UTS MDSI project for 36118 ANLP.

LegalEase NSW transforms complex NSW legislation into plain English while visualising how laws connect through an interactive citation network. The application uses a sophisticated two-stage pipeline: Stage 1 uses a local model (`facebook/bart-large-cnn`) to create a concise, factual summary. Stage 2 sends this summary to GPT-4o to translate it into accessible language at a Year 7-8 reading level. When legislation is detected in your document, the citation graph automatically highlights that Act and its connections within NSW's legislative framework.

**Disclaimer**: This tool provides AI-generated translations for informational purposes only and does not constitute legal advice. It is not a substitute for a qualified legal professional. Always consult a lawyer for advice on your specific situation.

## Features

* **Two-stage simplification pipeline**: Combines a fast local BART model for initial summarisation with GPT-4o for high-quality plain English transformation
* **Interactive citation network**: Explore how NSW legislation connects through an interactive graph powered by PyVis, with automatic highlighting when legislation is detected in your documents
* **Advanced PDF processing**: Supports both text-based and scanned (image-based) PDFs through integrated Optical Character Recognition (OCR) via Tesseract
* **Translation history**: Track all your document simplifications with downloadable history and expandable original text view
* **Prompt style selection**: Choose between Helper mode (default), More detail, or Explain like I'm 5 for different simplification approaches

## Setup

Follow these steps to run LegalEase NSW on your local machine.

### Prerequisites

* You must have **Conda** (Anaconda or Miniconda) installed. Download it [here](https://www.anaconda.com/download)
* You need your own **OpenAI API key** from [OpenAI Platform](https://platform.openai.com/)
* You must have the **Tesseract-OCR engine** installed for PDF scanning to work. Download the Windows installer [here](https://github.com/UB-Mannheim/tesseract/wiki)

### Installation and first run

#### 1. Get the code (choose one method)

**Method A: Download ZIP (no git required)**

* On the main repository page, click the green `<> Code` button and select **Download ZIP**
* Unzip the downloaded file on your computer

**Method B: Clone with git**
```bash
git clone https://github.com/brettjmckeehan-arch/legalease-nsw-translator.git
cd legalease-nsw-translator
```
#### 2. Create the conda environment
This command reads the `environment.yml` file and automatically installs all 140+ necessary Python libraries including PyTorch, transformers, spaCy and Streamlit.
```bash
# Navigate into the project folder first if you haven't already
cd path/to/legalease-nsw-translator
```
**Note**: Environment creation may take 10-15 minutes due to the large number of dependencies.

#### 3. Set up your API key
The application loads your OpenAI API key from a special file.
1. In the main project directory, create a new folder named `.streamlit`
2. Inside the `.streamlit` folder, create a new file named `secrets.toml`
3. Open `secrets.toml` and add your key in the following format:
```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-proj-..."
```
**Security note**: The `.streamlit` folder is excluded from version control via `.gitignore` to protect your API keys.

#### 4. Download required data files
The citation graph and test data are included in the repository. If they're missing, download:
* `citation_graph.pkl` - Place in the main project directory
* `test_data.json` - Place in the `eval` folder (optional, only needed for running evaluations)
Large NSW corpus files (2.19GB) are hosted separately on Google Drive - links in the repository if needed for development.
#### 5. Run the LegalEase NSW application
Once the environment is active and your API key is set, start the Streamlit app:
```bash
streamlit run streamlit_app_UNITED.py
```
The application will launch in your default web browser at `http://localhost:8501`.
## How to use
### Document simplification
1. **Enter your legal text**: Paste text directly into the text area or upload a PDF file (including scanned documents)
2. **Choose simplification style**: Select from three options in the right sidebar:
   * **Helper mode** (default): Structured four-section output with legislation, problem, impacts and options
   * **More detail**: Extended explanations with additional context
   * **Explain like I'm 5**: Maximum simplification using only simple words
3. **Click "Translate to plain English"**: The two-stage pipeline processes your document
4. **View results**: 
   * The simplified translation appears in the main area
   * Click "Show original text" to view the initial BART summary and source document
   * Any detected legislation automatically highlights in the citation network

## Citation network exploration
* **Search for legislation**: Use the search bar above the graph to find specific Acts
* **Adjust display**: Toggle "Adjust number of laws displayed" to show 50-1000 nodes via slider
* **Interact with the graph**: 
  * Click and drag nodes to reposition
  * Scroll to zoom in/out
  * Click nodes to view citation details
  * Hover over edges to see citation relationships

## Translation history
* View all past translations in the expandable "Translation History" section
* Download your complete history as a text file using the download button
* Each entry shows timestamp, legislation detected, and full translation with expandable source text

## Troubleshooting

**Problem**: `streamlit: command not found`  
**Solution**: Make sure your conda environment is activated: `conda activate legalease`

**Problem**: API errors or "OpenAI API key unavailable"  
**Solution**: Check that `.streamlit/secrets.toml` exists and contains your valid OpenAI API key

**Problem**: PDF upload fails or returns empty text  
**Solution**: Ensure Tesseract-OCR is installed and accessible in your system PATH

**Problem**: Citation graph doesn't load  
**Solution**: Verify `citation_graph.pkl` exists in the main project directory

## Project structure
```
legalease-nsw-translator/
├── streamlit_app_UNITED.py      # Main application
├── prompts.py                    # LLM prompt configurations
├── environment.yml               # Conda dependencies
├── requirements.txt              # Pip dependencies (alternative)
├── citation_graph.pkl            # NSW legislation citation network
├── .streamlit/
│   └── secrets.toml             # API keys (create this yourself)
├── src/
│   ├── pdf_handler.py           # PDF text extraction
│   ├── summariser.py            # BART model interface
│   └── llm_handler.py           # LLM API calls
├── static/
│   ├── style.css                # Custom styling
│   ├── logo3.jpg                # LegalEase NSW logo
│   ├── ai_a.png                 # Assistant icon
│   └── person_a.png             # User icon
└── eval/
    ├── test_llm_translation_suite.py
    ├── test_llm_translation_suite_bias.py
    ├── advanced_analysis.py
    └── outputs/                  # Evaluation results and visualisations
```

## Technical details

* **Local summarisation**: facebook/bart-large-cnn with two-pass chunking for documents exceeding 1024 tokens
* **LLM transformation**: GPT-4o via OpenAI API (hardcoded based on evaluation results)
* **Citation network**: NetworkX graph with PyVis interactive visualisation, 2216 NSW Acts
* **Evaluation metrics**: BERTScore F1 (semantic fidelity), Flesch-Kincaid Grade Level (readability), ROUGE-L (content overlap)
* **Bias detection**: spaCy NER with FuzzyWuzzy string matching across 23 name origins and 3 complexity levels

## Evaluation and testing

The `eval` folder contains comprehensive testing suites used to benchmark models and prompts:

* **test_llm_translation_suite.py**: Tests 7 models × 4 prompts × 52 documents = 1456 translations
* **test_llm_translation_suite_bias.py**: Demographic bias analysis with name preservation metrics
* **advanced_analysis.py**: BART summarisation performance on 1381 NSW legislation documents
* **llm_translation_suite_analysis.py**: Generates performance visualisations
* **llm_translation_suite_bias_analysis.py**: Generates bias metric visualisations

To run evaluations (requires all three API keys):
```bash
cd eval
python test_llm_translation_suite.py
```

## Citation
If you use this work, please cite:
McKeehan, Brett, Schillert, Alistair. (2025). 
LegalEase NSW: AI-powered legal document simplifier and citation network. 
University of Technology Sydney, 36118 Applied Natural Language Processing.
https://github.com/brettjmckeehan-arch/legalease-nsw-translator

## License
This project uses the [Open Australian Legal Corpus v7.1.0](https://huggingface.co/datasets/isaacus/open-australian-legal-corpus) licensed under CC BY 4.0.

## Acknowledgements
* Umar Butler for creating and curating the Open Australian Legal Corpus
* Facebook AI Research for the BART model


**Contact**: For questions or issues, please open an issue on the GitHub repository.