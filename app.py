from flask import Flask, request, render_template
import openai
import pandas as pd
import numpy as np
import config
import csv
import pinecone
import spacy
from spacy import displacy
from flaskext.markdown import Markdown


app = Flask(__name__)
Markdown(app)

openai.api_key = config.OPENAI_API_KEY
# initialize connection to pinecone
pinecone.init(
    api_key=config.PINECONE_API_KEY,
    environment=config.PINECONE_ENVIRONMENT 
)
#ner
nlp = spacy.load("en_core_web_sm")


@app.route('/static/<path:filename>')
def serve_static(filename):
  return app.send_static_file(filename)

@app.route('/')
def search_form():
  available_indexes = pinecone.list_indexes()
  return render_template('search_form.html',available_indexes=available_indexes)


@app.route('/search')
def search():
    # Get the search query from the URL query string
    query = request.args.get('query')
    xq = openai.Embedding.create(input=query, engine="text-embedding-ada-002")['data'][0]['embedding']

    index = pinecone.Index(pinecone.list_indexes()[0])

    res = index.query([xq], top_k=3, include_metadata=True)

    top3_search_results_concatenated = res['matches'][0]['metadata']['text']+res['matches'][1]['metadata']['text']+res['matches'][2]['metadata']['text']

    
    scores = []
    NER_texts = []

    for match in res['matches']:
      doc = nlp(match['metadata']['text'])
      NER_texts.append(displacy.render(doc, style="ent"))
      scores.append(f"{match['score']:.2f}")

    results = zip(scores,NER_texts)

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "user", 
       "content":f"I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with [Invalid Question]. Now, I will answer this question:{query} in 10 lines using only this document:{top3_search_results_concatenated}"}
    ]
    )

    gpt_result = completion.choices[0].message.content
    # Render the search results template, passing in the search query and results
    return render_template('search_results.html', query=query, results=results,gpt_result=gpt_result,doc=doc)

@app.route('/show_upload_page')
def show_upload_page():
  return render_template("upload.html")

@app.route('/upload',methods=['POST'])
def upload():
  if 'txtfile' not in request.files:
    return 'No file uploaded.'

  txtfile = request.files['txtfile']
  if txtfile.filename == '':
    return render_template('no_file_selected.html')

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
      if pinecone.list_indexes() != []:
        pinecone.delete_index(pinecone.list_indexes()[0])
      pinecone.create_index(index_name, dimension=len(embeds[0]))

    work_around = None
    while work_around is None:
      try:
        # connect to index
        index = pinecone.Index(index_name)

        # upsert to Pinecone
        ids = [str(n) for n in range(1, len(embeds[0])+1)]
        meta = [{'text': para} for para in txtparas]
        to_upsert = zip(ids, embeds, meta)
        index.upsert(vectors=list(to_upsert))

        work_around = 'worked'
      except:
        pass


    #describe index 
    return render_template('upserted_in_index.html')

  return render_template("invalid_file_type.html")

if __name__ == '__main__':
  app.run(debug=True)