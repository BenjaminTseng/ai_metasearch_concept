from modal import Image, App, Secret, method, enter

# define Image for embedding text queries and hitting Pinecone
# use Modal initiation trick to preload model weights
cache_path = '/pycache/clip-ViT-B-32' # cache for CLIP model 
def download_models():
    import sentence_transformers 

    model_id = 'sentence-transformers/clip-ViT-B-32' # model ID for CLIP

    model = sentence_transformers.SentenceTransformer(
        model_id,
        device='cpu'
    )
    model.save(path=cache_path)

image = (
    Image.debian_slim(python_version='3.10')
    .pip_install('sentence_transformers')
    .run_function(download_models)
    .pip_install('pinecone-client')
)
app = App('text-pinecone-query', image=image)

# use Modal's class entry trick to speed up initiation
@app.cls(secrets=[Secret.from_name('pinecone_secret')])
class TextEmbeddingModel:
    @enter()
    def enter(self):
        import sentence_transformers
        model = sentence_transformers.SentenceTransformer(cache_path, 
                                                          device='cpu')
        self.model = model 

        from pinecone import Pinecone
        import os 
        pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        self.pinecone_index = pc.Index(os.environ['PINECONE_INDEX'])
    
    @method()
    def query(self, query: str, num_matches = 10):
        # embed the query 
        vector = self.model.encode(query)

        # run the resulting vector through Pinecone
        pinecone_results = self.pinecone_index.query(vector=vector.tolist(), 
                                   top_k=num_matches, 
                                   include_metadata=True
                                   )
        
        # convert to format expected
        results = []
        for match in pinecone_results['matches']:
            matchDict = {
                'query': query,
                'source': 'Image Vector Search',
                'subsource': 'Savee',
                'subsource_url': match['metadata']['source_page_url'],
                'thumbnail': match['metadata']['source_image_url'],
                'title': '',
                'snippet':match['metadata']['caption']
            }
            if 'original_url' in match['metadata'] and \
                len(match['metadata']['original_url'].strip()) > 7:
                
                matchDict['url'] = match['metadata']['original_url']
            else:
                matchDict['url'] = match['metadata']['source_page_url']

            results.append(matchDict)
        
        return results

# local entrypoint to test
@app.local_entrypoint()
def entry(prompt: str = "Mountain Sunset"):
    print('Prompt:', prompt)
    emb = TextEmbeddingModel()
    results = emb.query.remote(prompt)
    for result in results:
        print('URL:', result['url'])
        print('Subsource:', result['subsource_url'])
        print('Image:', result['thumbnail'])
        print('Caption:', result['snippet'])
        print('')