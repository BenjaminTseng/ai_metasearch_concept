# Concept: LLM-powered Metasearch
Purpose of this was to explore:
* Utilizing an LLM (in this case OpenAI's GPT3.5) to serve as kickoff point for metasearch (by interpreting user-supplied query and then supplying queries for different platforms)
* Pinecone vectorstore for images I scraped from Savee + use of CLIP model to match text query with images
* Deploying workers / web site to Modal

You can explore (be gentle: everything I'm linking to is on free tier 🙏🏻😇) at https://benjamintseng--metasearch.modal.run/

For more on the design/architecture choices, read more at https://benjamintseng.com/portfolio/building-an-ai-powered-metasearch-concept/ 
