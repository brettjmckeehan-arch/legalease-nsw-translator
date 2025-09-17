# LegalEase NSW: AI-Powered Legal Document and Legislation Translator

This is a UTS MDSI project for 36118 ANLP.

The application uses the `facebook/bart-large-cnn` model to translate legal documents and legislation in NSW.

Disclaimer: This tool provides AI-generated summaries for informational purposes only and does not constitute legal advice.
It is not a substitute for a qualified legal professional. Always consult a lawyer for advice on your specific situation.

### Setup

Follow these steps to run LegalEase on your local machine.

**Prerequisites:**
- You must have Conda (Anaconda or Miniconda) installed. You can download it [here](https://www.anaconda.com/download).
- Download the dataset ("nsw_corpus_final.parquet") from [Google Drive](https://drive.google.com/file/d/13pnrYw-5E8Xnk9cwQ36-VxlQQ6NS44TB/view?usp=sharing) and place it in the main project directory.

---

### Method 1: Download ZIP (no Git)

1.  **Download LegalEase**
    - On the main repository page, click the green `<> Code` button.
    - In the dropdown, click **`Download ZIP`**.
    - Unzip the downloaded file on your computer.

2.  **Open the terminal**
    - Open the Anaconda prompt (or preferred terminal) and navigate into the unzipped project folder. The folder will be named something like `LegalEase_NSW-main`.
    ```bash
    cd path/to/your/unzipped/folder/LegalEase_NSW-main
    ```

3.  **Create Conda environment**
    - This command reads `environment.yml` file and automatically builds exact environment needed to run the code.
    ```bash
    conda env create -f environment.yml
    conda activate legalease
    ```

4.  **Run the LegalEase NSW application**
    - Once the environment is active, start the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```

The application should now open in your web browser.

---

### Method 2: Clone with Git

If you have Git installed, clone the repository instead.

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/brettjmckeehan-arch/legalease-nsw-translator.git](https://github.com/brettjmckeehan-arch/legalease-nsw-translator.git)
    cd LegalEase_NSW
    ```

2.  **Create and run**
    - Follow steps 3 and 4 from Method 1.
