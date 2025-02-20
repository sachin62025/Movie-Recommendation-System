#  Movie Recommendation System with FAISS & Sentence Transformers  

##  Project Overview  
This is a **Graph RAG-based** Movie Recommendation System using the **MovieLens dataset**. The system preprocesses movie rating data, trains a recommendation model, and provides movie recommendations via a web interface.  

##  Features  
-  **Movie Recommendations** using **FAISS**  
-  **Sentence Transformers** (`BAAI/bge-base-en-v1.5`) for text embeddings  
-  **Preprocessed MovieLens Dataset**  
-  **Fast & Scalable Search**  
-  **Graph-based Representation** (Planned)  

##  Technologies Used  
- **Python**  
- **FAISS (Facebook AI Similarity Search)**  
- **Sentence Transformers**  
- **Flask (API)**  
- **Pandas & NumPy**  
- **NetworkX (Graph Processing - Planned)**  

## Project Structure  

```bash
Movie-Recommendation-System/
│── data/
│   ├── movies.csv
│   ├── ratings.csv
│── logs/
│   ├── app.log
│   ├── preprocess.log
│   ├── train.log
│── models/                    
│── modules/                   
│   ├── preprocess.py
│   ├── recommender.py
│   ├── search_movies.py
│   ├── train.py
│   ├── update_models.py
│── research/
│── static/
│   ├── style.css
│── templates/
│   ├── index.html
│── test/
│── app.py
│── preprocess_data.py
│── requirements.txt
│── setup.py
│── train_model.py
│── update_model.py

```
## Setup & Installation
``` bash
git clone https://github.com/sachin62025/Movie-Recommendation-System.git  
cd Movie-Recommendation-System  
pip install -r requirements.txt  
python app.py  
```
## Application

![interface](https://github.com/user-attachments/assets/7e607b39-7640-4c73-bf7c-59b8c83a9fca)

![output](https://github.com/user-attachments/assets/d30a908b-49c1-4950-a82a-039b04e3a39a)
![output_](https://github.com/user-attachments/assets/55ed1182-e8c3-49bf-9639-bb36031b31d5)
