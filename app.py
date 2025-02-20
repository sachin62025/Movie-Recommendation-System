# from fastapi import FastAPI, Form, Request
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from modules.recommender import Recommender
# import uvicorn
# from jinja2 import Template

# # Initialize app and recommender
# app = FastAPI()
# recommender = Recommender()
# app.mount("/static", StaticFiles(directory="static"), name="static")
# # HTML Template
# html_template = """
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Movie Recommendation</title>
#     <link rel="stylesheet" type="text/css" href="/static/style.css">
# </head>
# <body>
#     <h2>Movie Recommendation System</h2>
#     <form action="/recommend" method="post">
#         <label for="query">Enter Movie Preferences:</label>
#         <input type="text" id="query" name="query">
#         <button type="submit">Get Recommendations</button>
#     </form>
#     {% if recommendations %}
#     <h3>Recommended Movies:</h3>
#     <ul>
#         {% for movie in recommendations %}
#         <li>{{ movie }}</li>
#         {% endfor %}
#     </ul>
#     {% endif %}
# </body>
# </html>
# """

# @app.get("/", response_class=HTMLResponse)
# async def home():
#     return Template(html_template).render()

# @app.post("/recommend", response_class=HTMLResponse)
# async def recommend(request: Request, query: str = Form(...)):
#     recommendations = recommender.recommend(query)
#     return Template(html_template).render(recommendations=recommendations)

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)




''' 

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from modules.recommender import Recommender
import uvicorn

# Initialize app and recommender
app = FastAPI()
recommender = Recommender()

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, query: str = Form(...)):
    recommendations = recommender.recommend(query)
    return templates.TemplateResponse("index.html", {"request": request, "recommendations": recommendations})

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

'''

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from modules.recommender import Recommender
from modules.search_movies import search_movie  # Import search function
import uvicorn

# Initialize FastAPI app and recommender
app = FastAPI()
recommender = Recommender()

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, query: str = Form(...)):
    # Get recommended movie names
    recommended_movies = recommender.recommend(query)

    # Get links for each recommended movie
    # recommendations_with_links = [
    #     {"title": movie, "link": search_movie(movie)["link"]} for movie in recommended_movies
    # ]

    # recommendations_with_links = [
    # {
    #     "title": movie, 
    #     "link": search_movie(movie)["link"], 
    #     "description": search_movie(movie)["description"]  # Include description
    # } 
    # for movie in recommended_movies
    # ]
    recommendations_with_links = []
    for movie in recommended_movies:
        movie_details = search_movie(movie)  # Call only once
        recommendations_with_links.append({
            "title": movie,
            "link": movie_details["link"],
            "description": movie_details["description"]
        })



    return templates.TemplateResponse(
        "index.html",
        {"request": request, "recommendations": recommendations_with_links}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
