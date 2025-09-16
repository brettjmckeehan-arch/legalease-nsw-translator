# LegalEase NSW: AI-Powered Legal Document and Legislation Summariser

This is a UTS MDSI project for 36118 ANLP.

The application uses the `facebook/bart-large-cnn` model to summarise legal documents and legislation in NSW.

Disclaimer: This tool provides AI-generated summaries for informational purposes only and does not constitute legal advice.
It is not a substitute for a qualified legal professional. Always consult a lawyer for advice on your specific situation.

## How to run

1. Ensure you have a Conda environment set up.
2. Install the required packages using "pip install -r requirements.txt".
3. Download the dataset ("nsw_corpus_final.parquet") from [Google Drive](https://drive.google.com/file/d/13pnrYw-5E8Xnk9cwQ36-VxlQQ6NS44TB/view?usp=sharing) and place it in the main project directory.
4. Run the application using "streamlit run streamlit_app.py".
