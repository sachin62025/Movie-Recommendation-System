# from duckduckgo_search import DDGS

# def search_movie(movie_name):
#     with DDGS() as ddgs:
#         results = list(ddgs.text(movie_name + " movie", max_results=1))
#         if results:
#             first_result = results[0]
#             return {"link": first_result["href"]}
#     return {"title": "Not found", "link": "#"}


import time
from duckduckgo_search import DDGS

# Cache to store previously searched movie details
search_cache = {}

def search_movie(movie_name):
    if movie_name in search_cache:
        return search_cache[movie_name]  # Return cached data

    time.sleep(2)  # Delay to prevent rate-limiting

    with DDGS() as ddgs:
        results = list(ddgs.text(movie_name + " movie", max_results=1))
        if results:
            first_result = results[0]
            description = first_result.get("body", "No description available.")
            description_limited = " ".join(description.split()[:50]) + "..."
            
            search_cache[movie_name] = {
                "link": first_result["href"],
                "description": description_limited
            }
            return search_cache[movie_name]
    
    return {"title": "Not found", "link": "#", "description": "No details available."}
