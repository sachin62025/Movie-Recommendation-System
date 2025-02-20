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

![interface](https://github.com/user-attachments/assets/df4f4157-54f3-45ea-ad03-accaeae4c630)

![output](https://github.com/user-attachments/assets/b9889d3f-0fe4-47d2-8587-5f37b8408cdc)
![output_](https://github.com/user-attachments/assets/ef8928f4-d2e6-401f-8948-59e326689182)
