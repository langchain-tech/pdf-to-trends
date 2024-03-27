# PDF to Trends

PDF to Trends is a Streamlit web application that enables users to upload PDF files and perform queries on their content. It utilizes the Langchain library for question-answering and OpenAI embeddings for text similarity search.

## Features

- Upload PDF files
- Perform queries on the content of PDF files
- Utilize Langchain for question-answering
- Use OpenAI embeddings for text similarity search

## Getting Started

To run PDF to Trends locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Set up your OpenAI API key by adding it to the environment variable `OPENAI_API_KEY`.
4. Run the Streamlit app by executing `streamlit run pdf-to-trends.py`.
5. Access the app in your web browser at `http://localhost:8501`.

## Usage

1. Upload PDF files by clicking on the "Upload PDF file" button.
2. Enter your query in the text input field provided.
3. Click on the "Search" button to perform the query on the uploaded PDF file.
4. View the responses provided by the app.

## Dependencies

- Python 3.x
- Streamlit
- pypdf
- langchain
- langchain_openai

Install the dependencies using `pip install -r requirements.txt`.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
