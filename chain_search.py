from modal import Image, App, web_endpoint, Secret, Function
from fastapi.responses import HTMLResponse

# define Image for metasearch hitting web APIs
image = Image.debian_slim(python_version='3.10') \
            .pip_install('openai', 'httpx', 'beautifulsoup4')
app = App('chain-search', image=image)

# use OpenAI to convert query into smaller queries
@app.function(secrets=[Secret.from_name('openai_secret')])
def openai_chain_search(query: str):
    import openai 
    import os 
    import re
    model = 'gpt-3.5-turbo' # using GPT 3.5 turbo model

    # Pull Open AI secrets
    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    # message templates with (some) prompt engineering
    system_message = """Act as my assistant who's job is to help me understand and derive inspiration around a topic I give you.  Your primary job is to help find the best images, content, and online resources for me. Assume I have entered a subject into a command line on a website and do not have the ability to provide you with follow-up context.

Your first step is to determine what sort of content and resources would be most valuable. For topics such as "wedding dresses" and "beautiful homes" and "brutalist architecture", I am likely to want more visual image content as these topics are design oriented and people tend to want images to understand or derive inspiration. For topics, such as "home repair" and "history of Scotland" and "how to start a business", I am likely to want more text and link content as these topics are task-oriented and people tend to want authoritative information or answers to questions."""
    initial_prompt_template = 'I am interested in the topic:\n{topic}\n\nAm I more interested in visual content or text and link based content? Select the best answer between the available options, even if it is ambiguous. Start by stating the answer to the question plainly. Do not provide the links or resources. That will be addressed in a subsequent question.'
    text_template = 'You have access to three search engines.\n\nThe first will directly query Wikipedia. The second will surface interesting posts on Reddit based on keyword matching with the post title and text. The third will surface podcast episodes based on keyword matching.\n\nQueries to Wikipedia should be fairly direct so as to maximize the likelihood that something relevant will be returned. Queries to the Reddit and podcast search engines should be specific and go beyond what is obvious and overly broad to surface the most interesting posts and podcasts.\n\nWhat are 2 queries that will yield the most interesting Wikipedia posts, 3 queries that will yield the most valuable Reddit posts, and 3 queries surface that will yield the most insightful and valuable podcast episodes about:\n{topic}\n\nProvide the queries in a numbered list with quotations around the entire query and brackets around which search engine they\'re intended for (for example: 1. [Reddit] "Taylor Swift relationships". 2. [Podcast] "Impact of Taylor Swift on Music". 3. [Wikipedia] "Taylor Swift albums").'
    image_template = 'You have access to a the free stock photo site Unsplash. There will be a good breadth of photos but the key will be trying to find the highest quality images.\n\nWhat are 3 great queries to use that will provide good visual inspiration and be different enough from one another so as to provide a broad range of relevant images from Unsplash to get the highest quality images on the topic of:\n{topic}\n\nProvide the queries in a numbered list with quotations around the entire query and "[Unsplash]" before the quotation to make it clear thats the intended search engine (for example: 1. [Unsplash] "Wildlife on a mountain top at sunset". 2. [Unsplash] "High quality capture of mountain top at sunset".)'

    # create context to send to OpenAI
    messages = []
    # add system message
    if 'gpt-4' not in model: # only gpt-4 and beyond have 'system' message
        messages.append({
            'role': 'user',
            'content': system_message
        })
    else:
        messages.append({
            'role': 'system',
            'content': system_message
        })

    # add initial prompt
    messages.append({
        'role': 'user',
        'content': initial_prompt_template.format(topic=query)
    })

    # get initial response
    response = client.chat.completions.create(
        model=model,
        messages = messages,
        temperature = 1.0
    )
    messages.append({
        'role': 'assistant',
        'content': response.choices[0].message.content
    })

    # chain decision: decide based on response what to do
    responses = [] # aggregate list of actions to take
    if 'text and link' in response.choices[0].message.content.lower():
        # get good wikipedia, reddit, and podcast queries
        messages.append({
            'role': 'user',
            'content': text_template.format(topic=query)
        })
    else:
        # Wikipedia one-shot the query to add some additional text-based context
        responses.append('Wikipedia: ' + query)
        
        # get good image search queries
        messages.append({
            'role': 'user',
            'content': image_template.format(topic=query)
        })

    # make followup call to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages = messages,
        temperature = 1.0
    )
    
    # use regex to parse GPT's recommended queries
    for engine, query in re.findall(r'[0-9]+. \[(\w+)\] "(.*)"', 
                                response.choices[0].message.content):
        responses.append(engine + ': ' + query)
    
    return responses

# handle Wikipedia
@app.function()
def search_wikipedia(query: str):
    import httpx
    import urllib.parse
    from bs4 import BeautifulSoup

    # base_search_url works pretty well if search string is spot-on, if not shows search results
    base_search_url = 'https://en.wikipedia.org/w/index.php?title=Special:Search&search={query}'
    base_url = 'https://en.wikipedia.org'

    results = []
    r = httpx.get(base_search_url.format(query = urllib.parse.quote(query)), follow_redirects=True)
    soup = BeautifulSoup(r.content, 'html.parser')

    if 'title=Special:Search' in str(r.url): # no match, so its a search, get top two results
        search_results = soup.find_all('li', class_='mw-search-result')
        for search_result in search_results[0:2]:
            result = {'query':query}
            result['source'] = 'Wikipedia'
            thumbnail_anchors = search_result.css.select("div.searchResultImage-thumbnail > a")
            if len(thumbnail_anchors): # thumbnail exists
                result['thumbnail'] = 'https:' + thumbnail_anchors[0].find('img')['src']
            else:
                result['thumbnail'] = 'None'
            result_header_tag = search_result.css.select("div.mw-search-result-heading > a")[0]
            result['url'] = base_url + result_header_tag['href']
            result['title'] = result_header_tag.get_text()
            result['snippet'] = search_result.css.select("div.searchResultImage-text > div.searchresult")[0].get_text()
            results.append(result)
    else:
        result = {'query':query}
        result['source'] = 'Wikipedia'
        result['url'] = r.url 
        result['title'] = soup.find('h1').get_text()

        # get the right paragraph to determine if this is a disambiguation article
        real_paragraphs = soup.find_all(lambda tag: tag.name == 'p' and 'class' not in tag.attrs and len(tag.get_text().split()) > 10)
        if real_paragraphs:
            paragraph_text = real_paragraphs[0].get_text().strip()
        else:
            paragraph_text = ''

        if not len(paragraph_text) or paragraph_text[-18:] == 'may also refer to:' or paragraph_text[-13:] == 'may refer to:': # is disambiguation article or blank
            result['thumbnail'] = 'None'
            result['snippet'] = 'This page links to several Wikipedia articles that might be relevant.'
        else: # normal article
            result['snippet'] = paragraph_text
            img_link = soup.find(lambda tag: tag.name == 'meta' and tag.has_attr('property') and tag.has_attr('content') and tag['property'] == 'og:image')
            if img_link: # an image exists
                result['thumbnail'] = img_link['content']
            else:
                result['thumbnail'] = 'None'

        results.append(result)
    
    return results

# handle Reddit
@app.function(secrets=[Secret.from_name('reddit_secret')])
def search_reddit(query: str):
    import httpx
    import base64 
    import os 

    reddit_id = os.environ['REDDIT_USER']
    user_agent = os.environ['REDDIT_AGENT']
    reddit_secret = os.environ['REDDIT_KEY']

    # set up for auth token request
    auth_string = reddit_id + ':' + reddit_secret
    encoded_auth_string = base64.b64encode(auth_string.encode('ascii')).decode('ascii')
    auth_headers = {
        'Authorization': 'Basic ' + encoded_auth_string,
        'User-agent': user_agent
    }
    auth_data = {
        'grant_type': 'client_credentials'
    }

    # get auth token
    r = httpx.post('https://www.reddit.com/api/v1/access_token', headers = auth_headers, data = auth_data)
    if r.status_code == 200:
        if 'access_token' in r.json():
            reddit_access_token = r.json()['access_token']
    else:
        return [{'error':'auth token failure'}]

    results = []

    # set up headers for search requests
    headers = {
        'Authorization': 'Bearer ' + reddit_access_token,
        'User-agent': user_agent
    }
    
    # execute subreddit search
    params = {
        'sort': 'relevance',
        't': 'year',
        'limit': 4,
        'q': query[:512]
    }
    r = httpx.get('https://oauth.reddit.com/search', params=params, headers=headers)
    if r.status_code == 200:
        body = r.json()
        if 'data' in body and 'children' in body['data'] and len(body['data']['children']) > 0:
            post_results = body['data']['children']
            for post in post_results:
                # get subreddit level details
                result = {'query':query}
                subreddit_handle = post['data']['subreddit_name_prefixed'] + '/'
                subreddit_url = 'https://www.reddit.com'+subreddit_handle
                result['source'] = 'Reddit'
                result['subsource'] = subreddit_handle 
                result['subsource_url'] = subreddit_url 
                result['url'] = 'https://www.reddit.com'+post['data']['permalink']
                result['title'] = post['data']['title']

                # get image from Reddit blob, start with preview
                if 'preview' in post['data'] and 'images' in post['data']['preview'] and len(post['data']['preview']['images']):
                    result['thumbnail'] = post['data']['preview']['images'][0]['source']['url']
                # use media_metadata if post is media gallery and pull first image in blob
                elif 'media_metadata' in post['data']:
                    first_key = list(post['data']['media_metadata'].keys())[0]
                    result['thumbnail'] = post['data']['media_metadata'][first_key]['p'][-1]
                # fall back to thumbnail if needed but only if thumbnail is valid
                elif 'thumbnail' in post['data'] and post['data']['thumbnail'] != 'self':
                    result['thumbnail'] = post['data']['thumbnail']
                else:
                    result['thumbnail'] = ''
                result['snippet'] = post['data']['selftext'][0:1000] + ('...(more)' if len(post['data']['selftext']) else '')
                results.append(result)
    
    return results

# handle Podcast Search via Taddy
@app.function(secrets=[Secret.from_name('taddy_secret')])
def search_podcasts(query: str):
    import os 
    import httpx 

    # prepare headers for querying taddy
    taddy_user_id = os.environ['TADDY_USER']
    taddy_secret = os.environ['TADDY_KEY']
    url = 'https://api.taddy.org'
    headers = {
        'Content-Type': 'application/json',
        'X-USER-ID': taddy_user_id,
        'X-API-KEY': taddy_secret
    }
    # query body for podcast search
    queryString = """{
  searchForTerm(
    term: """
    queryString += '"' + query + '"\n'
    queryString += """
    filterForTypes: PODCASTEPISODE
    searchResultsBoostType: BOOST_POPULARITY_A_LOT
    limitPerPage: 3
  ) {
    searchId
    podcastEpisodes {
      uuid
      name
      subtitle
      websiteUrl
      audioUrl
      imageUrl
      description
      podcastSeries {
        uuid
        name
        imageUrl
        websiteUrl
      }
    }
  }
}
"""
    # make the graphQL request and parse the JSON body
    r = httpx.post(url, headers=headers, json={'query': queryString})
    if r.status_code != 200:
        return []
    else:
        responseBody = r.json()
        if 'errors' in responseBody:
            return [{'error': 'authentication issue with Taddy'}]
        else:
            episodes = responseBody['data']['searchForTerm']['podcastEpisodes']
            results = []
            for episode in episodes:
                result = {'query':query}
                result['source'] = 'Podcast'
                result['subsource'] = episode['podcastSeries']['name']
                result['subsource_url'] = episode['podcastSeries']['websiteUrl']
                if episode['websiteUrl'] and episode['websiteUrl'] != episode['podcastSeries']['websiteUrl']:
                    result['url'] = episode['websiteUrl']
                else: 
                    result['url'] = episode['audioUrl']
                
                result['title'] = episode['name']
                if episode['subtitle']:
                    result['snippet'] = episode['subtitle'][0:1000] + ('...(more)' if len(episode['subtitle']) else '')
                elif episode['description']:
                    result['snippet'] = episode['description'][0:1000] + ('...(more)' if len(episode['description']) else '')
                else:
                    result['snippet'] = ''
                
                if episode['imageUrl']:
                    result['thumbnail'] = episode['imageUrl']
                elif episode['podcastSeries']['imageUrl']:
                    result['thumbnail'] = episode['podcastSeries']['imageUrl']
                else:
                    result['thumbnail'] = ''
                results.append(result)
    
    return results

# handle Unsplash Search
@app.function(secrets=[Secret.from_name('unsplash_secret')])
def search_unsplash(query: str, num_matches: int = 10):
    import os 
    import httpx 

    # set up and make request
    unsplash_client = os.environ['UNSPLASH_ACCESS']
    unsplash_url = 'https://api.unsplash.com/search/photos'

    headers = {
        'Authorization': 'Client-ID ' + unsplash_client,
        'Accept-Version': 'v1'
    }
    params = {
        'page': 1,
        'per_page': num_matches,
        'query': query
    }
    r = httpx.get(unsplash_url, params=params, headers=headers)

    # check if request is good 
    if r.status_code == 200:
        results = []
        body = r.json()
        # convert to result format
        for image_result in body['results']:
            result = {
                'query': query,
                'source': 'Unsplash',
                'subsource': image_result['user']['username'],
                'subsource_url': image_result['user']['links']['html'],
                'snippet': image_result['description'],
                'url': image_result['links']['html'],
                'thumbnail': image_result['urls']['regular']
            }
            results.append(result)
            
        return results
    else:
        return [{'error':'auth failure'}]

# function to map against response list
@app.function()
def parse_response(response: str):
    if response[0:11] == 'Wikipedia: ':
        return search_wikipedia.remote(response[11:])
    elif response[0:8] == 'Reddit: ':
        return search_reddit.remote(response[8:])
    elif response[0:9] == 'Podcast: ':
        return search_podcasts.remote(response[9:])
    elif response[0:10] == 'Unsplash: ':
        return search_unsplash.remote(response[10:])  

# web endpoint
@app.function()
@web_endpoint(label='metasearch')
def web_search(query: str = None):
    import random 

    html_string = "<html>"
    css_string = "<style type='text/css'>\n .row {display: flex; flex-flow: row wrap}\n .rowchild {border: 1px solid #555555; border-radius: 10px; padding: 10px; max-width: 45%; min-width: 300px; margin: 10px;}\n .linkhead {font-size: larger}\n .actualquery {font-size: smaller}\n .snippet {margin: 10px auto; padding: 0px 15px; font-style: italic}\n .imagecontainer {max-width: 90%; max-height: 400px}\n .imagecontainer img {max-width: 100%; max-height: 400px; margin: auto;}\n .imagecontainer img.podcast {max-width: 200px; max-height: 200px;} </style>"
    results = []
    seen_urls = []
    seen_thumbnails = []

    if query:
        html_string += "<head><title>AI Metasearch Concept: " + query + "</title>" + css_string + '</head>'
        html_string += "<body><form action='/' method='get'><input type='text' name='query' placeholder='Search' value='" + query + "' style='width: 80%; padding: 10px;'><button type='submit' id='submit' style='width: 20%; padding: 10px;'>Search</button></form>"
        html_string += "<h1>Search: " + query + "</h1>"

        # run chain search and then process each search independently
        responses = openai_chain_search.remote(query)
        results = parse_response.map(responses)
        flattened_results = []
        for result_array in results:
            if result_array:
                flattened_results += result_array

        # shuffle results to add some randomness
        random.shuffle(flattened_results)

        # iterate through search results and build results page
        row_start = True 
        for result in flattened_results:
            # make sure it's not a duplicate
            if result['url'] not in seen_urls and result['thumbnail'] not in seen_thumbnails:
                # housekeeping to make sure we don't surface duplicates
                seen_urls.append(result['url'])
                if result['source'] == 'Reddit' and result['thumbnail'] not in seen_thumbnails:
                    seen_thumbnails.append(result['thumbnail'])
                
                # assemble HTML for search results
                # use flexbox for 2 column rows
                if row_start:
                    html_string += "<div class='row'>"
                    html_string += "<div class='rowchild'>"
                    row_start = False 
                else:
                    html_string += "<div class='rowchild'>"
                
                # assemble link
                if 'title' in result and result['title']:
                    html_string += "<div class='linkhead'><a href = '" + result['url'] + "'>" + result['title'] + "</a></div>"
                else: 
                    html_string += "<div class='linkhead'><a href = '" + result['url'] + "'>Link</a></div>"
                
                # assemble source, choose whether or not to say the source name given subsource
                if 'subsource' in result:
                    if result['source'] in ['Image Vector Search']:
                        html_string += "<div>Source: <a href='" + result['subsource_url'] + "'>" + result['subsource'] + "</a></div>"
                    else:
                        html_string += "<div>Source: <a href='" + result['subsource_url'] + "'>" + result['subsource'] + "</a> <i>(" + result['source'] + ")</i></div>"
                else:
                    html_string += "<div>Source: <i>" + result['source'] + "</i></div>"

                # assemble actual query
                html_string += "<div class='actualquery'>Actual query: <i>" + result['query'] + "</i></div>"
                
                if 'snippet' in result and result['snippet']:
                    html_string += "<div class='snippet'>" + result['snippet'] + '</div>'
                if result['thumbnail'] and result['thumbnail'] != 'None' and type(result['thumbnail']) != dict:
                    if result['source'] == 'Podcast':
                        html_string += "<div class='imagecontainer'><a href='" + result['thumbnail'] + "'><img class='podcast' src='" + result['thumbnail'] + "' /></a></div>"
                    else:
                        html_string += "<div class='imagecontainer'><a href='" + result['thumbnail'] + "'><img src='" + result['thumbnail'] + "' /></a></div>"

                html_string += "</div>"

        html_string += "</div></body></html>"
    else:
        html_string += "<head><title>Search</title></head>"
        html_string += "<body><form action='/' method='get'><input type='text' name='query' placeholder='Search' style='width: 80%; padding: 10px;'><button type='submit' id='submit' style='width: 20%; padding: 10px;'>Search</button></form>"
        html_string += "</body></html>"
    return HTMLResponse(html_string)

# local entrypoint to test
@app.local_entrypoint()
def main(query = 'Mountain sunset'):
    results = []
    seen_urls = []
    seen_thumbnails = []

    responses = openai_chain_search.remote(query)
    for response in responses:
        print(response)
    
    # use map to speed this up
    results = parse_response.map(responses)
    flattened_results = []
    for result_array in results:
        if result_array:
            flattened_results += result_array
    
    for result in flattened_results:
        if result['url'] not in seen_urls and result['thumbnail'] not in seen_thumbnails:
            seen_urls.append(result['url'])
            if result['source'] == 'Reddit' and result['thumbnail'] not in seen_thumbnails:
                seen_thumbnails.append(result['thumbnail'])
            for key in result:
                print(key + ':', result[key])
            print(' ')