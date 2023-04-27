# AI-Semantic-Search

## General Overview:
* The aim of this project is to implement a semantic search using artificial intelligence. You will develop a search engine that encodes the user's query into a vector and searches for similarity within a body of text. 
* The user can store all the text to be searched using a vector database like Pinecone. This search engine will be designed to provide accurate and relevant search results.

## Implemented Features List:
### Vector Database:
* You will need to choose a vector database like Pinecone to store the text that will be searched. The vector database will be used to store and index the text documents.
* I have used Pinecone database. Generate your api key and environment name at https://app.pinecone.io/ and paste it in the config.py file
### Vectorization Algorithm: 
* You will need to implement a vectorization algorithm that encodes the text documents into a vector representation. You can use Open AI’s latest text embeddings to vectorize the search query and the text documents to be searched from.
* I have used OpenAI's text-embedding-ada-002 model to create the vector embeddings for my search query and corpus. Get your api key at https://openai.com/blog/openai-api
### Similarity Search Algorithm: 
* You will need to implement a similarity search algorithm that can find the most similar documents to the user's query. The algorithm should be optimized for speed and accuracy.
* I have used cosine similarity as a distance metric to measure similarity. This metric can be chosen while creating an index in Pinecone.
### User Interface: 
* You will need to develop a user interface that allows users to enter their queries and view the search results. The interface should be intuitive and easy to use.
* I have developed an user interface using Flask in Python. I have created seperate HTML templates that can be rendered upon each click or route.
### Multi-lingual Support: 
* The search engine could support multiple languages, allowing users to search for documents in different languages.
* I have provided the code in the form of comments to use Cohere's multilingual-22-12 model to generate vector embeddings for search query as well as corpus to support other languages.
### Synonym Expansion: 
* The search engine could include a synonym expansion feature that expands the user's query to include similar words and phrases. 
* I have used OpenAi's gpt-3.5-turbo model to generate a list of two synonyms that I append to the search query before embedding it. 
* The prompt I have used to make OpenAi's gpt-3.5-turbo model act as a synonym finder is : "I want you to act as a synonyms provider. I will tell you a search query, and you will reply to me with a list of synonym alternatives according to my prompt. Provide only 2 synonyms for my search query. You will only reply the words list, and nothing else. Words should exist. Do not write explanations. Strictly, do not write anything else than a comma seperated list of two words. My search query is:{query}?"
### Document Ranking: 
* The search engine could rank the search results based on their relevance to the user's query. 
* I have ranked the search results using their cosine similarity scores and also displayed the similarity score of each result beside it. 
### Entity Extraction: 
* The search engine could include an entity extraction feature that identifies and extracts entities like people, places, and organizations from the search results. 
* To implement Named Entity Recognition (NER), I have used Spacy library to generate Named Entities and displayed them by highlighting them in the search result using Displacy library.
### Customizable Search Index: 
* The search engine could allow users to choose what text they want to be searched from and create a customized search index. 
* I have provided an user interface to upload custom text documents. I have added error handling templates like "No File Chosen" and "Invalid File Type". 
* Upon recieving user's custom text file, I save it on the server, parse it, make a list of paragraphs contained withing the file, create vector embeddings for each paragraph, create a new index using filename and delete current index if one is present(this is because free tier of Pinecone only allows one index per project), and finally I upsert the vector embeddings into the Pinecone index with their respective IDs and textual paragraphs included in the metadata.
### Integration with Third-Party Services: 
* The search engine could be integrated with third-party services like Google Drive or Dropbox to provide a more comprehensive search experience.
* I have added a Third Party Integration/Extra Feature of "Answering the user's search query based on top 3 search results". This feature allows the user to get a direct context-aware answer to the search query without reading through all the search results. 
* I have implemented this using OpenAi's gpt-3.5-turbo model. I have provided the model with the search query and a concatenated string of the top three search results to ensure that the answer is context-aware and based solely on the corpus. This helps avoid bias and prior knowledge from decreasing the accuracy of the answer.
