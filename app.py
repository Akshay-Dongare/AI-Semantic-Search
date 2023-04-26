from flask import Flask, request, render_template
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import config
import csv

app = Flask(__name__)

openai.api_key = config.OPENAI_API_KEY

@app.route('/static/<path:filename>')
def serve_static(filename):
  return app.send_static_file(filename)

@app.route('/')
def search_form():
  return render_template('search_form.html')


@app.route('/search')
def search():
    # Get the search query from the URL query string
    query = request.args.get('query')

    search_term_vector = get_embedding(query, engine="text-embedding-ada-002")

    df = pd.read_csv('earnings-embeddings.csv')
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
    sorted_by_similarity = df.sort_values("similarities", ascending=False).head(3)

    results = sorted_by_similarity['text'].values.tolist()

    # Render the search results template, passing in the search query and results
    return render_template('search_results.html', query=query, results=results)

@app.route('/show_upload_page')
def show_upload_page():
  return render_template("upload.html")

@app.route('/upload',methods=['POST'])
def upload():
  if 'csvfile' not in request.files:
    return 'No file uploaded.'

  csvfile = request.files['csvfile']
  if csvfile.filename == '':
    return 'No file selected.'

  if csvfile and csvfile.filename.endswith('.csv'):
    csvdata = csvfile.read().decode('utf-8')
    csvrows = csv.reader(csvdata.splitlines())
    #'File uploaded and processed successfully.'
    return render_template('create_custom_embeddings.html', csvrows=csvrows)

  return 'Invalid file type. Only CSV files are allowed.'

if __name__ == '__main__':
  app.run(debug=True)
