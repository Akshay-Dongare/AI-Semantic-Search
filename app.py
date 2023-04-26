from flask import Flask, request, render_template
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import config
import csv
import pinecone
from tqdm.auto import tqdm  # this is our progress bar

app = Flask(__name__)

openai.api_key = config.OPENAI_API_KEY
# initialize connection to pinecone
pinecone.init(
    api_key=config.PINECONE_API_KEY,
    environment=config.PINECONE_ENVIRONMENT 
)

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
    xq = openai.Embedding.create(input=query, engine="text-embedding-ada-002")['data'][0]['embedding']

    index = pinecone.Index('india')

    res = index.query([xq], top_k=3, include_metadata=True)

    results = []
    for match in res['matches']:
      results.append((f"{match['score']:.2f}: {match['metadata']['text']}"))

    # Render the search results template, passing in the search query and results
    return render_template('search_results.html', query=query, results=results)

@app.route('/show_upload_page')
def show_upload_page():
  return render_template("upload.html")

@app.route('/upload',methods=['POST'])
def upload():
  if 'txtfile' not in request.files:
    return 'No file uploaded.'

  txtfile = request.files['txtfile']
  if txtfile.filename == '':
    return 'No file selected.'

  if txtfile and txtfile.filename.endswith('.txt'):
    txtfile.save(txtfile.filename)  
    file_obj = open(txtfile.filename,"r",encoding='utf8')
    file_data = file_obj.read()
    txtparas = file_data.split('\n\n')

    #blankspace cannot be embedded so pre-processing is done
    if ("" in txtparas):
        txtparas.remove("")

    #create embeddings   
    res = openai.Embedding.create(
        input=txtparas, engine="text-embedding-ada-002"
    )

    # extract embeddings to a list
    embeds = [record['embedding'] for record in res['data']] #embeds is a list

    index_name = str(txtfile.filename[:-4])
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=len(embeds[0]))

    # connect to index
    index = pinecone.Index(index_name)

    # upsert to Pinecone
    ids = [str(n) for n in range(1, len(embeds[0])+1)]
    meta = [{'text': para} for para in txtparas]
    to_upsert = zip(ids, embeds, meta)
    index.upsert(vectors=list(to_upsert))

    #describe index 
    return render_template('new_index_created.html')

  return 'Invalid file type. Only text files are allowed.'

if __name__ == '__main__':
  app.run(debug=True)