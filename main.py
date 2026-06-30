import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from database import init_db, save_history, get_history as fetch_history
from vector import retriever, df

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
FILE_NAME = os.getenv("CSV_FILE", "realistic_restaurant_reviews.csv")

app = FastAPI()

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

class History(BaseModel):
    id: int
    question: str
    answer: str
    timestamp: str

class Reviews(BaseModel):
    title: str
    review: str
    rating: int
    date: str


model = ChatOpenAI(model=MODEL_NAME, api_key=os.getenv("OPENAI_API_KEY"))


template = """
You are an expert in answering questions about a pizza restaurant based on customer reviews.

Use only the reviews provided below to answer the question. Do not make up information that isn't supported by the reviews. If the reviews don't contain enough information to answer the question, say so.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

init_db()

@app.post("/ask", response_model=Answer)
def ask_question(user_question: Question):
    reviews = retriever.invoke(user_question.question)
    result = chain.invoke({"reviews": reviews, "question": user_question.question})

    save_history(user_question.question, result.content)

    return Answer(answer=result.content)

@app.get("/history", response_model=list[History])
def get_history():
    rows = fetch_history()
    history_list = []
    for row in rows:
        history_list.append(History(id=row[0], question=row[1], answer=row[2], timestamp=row[3]))
    return history_list

@app.get("/reviews", response_model=list[Reviews])
def get_reviews():
    reviews_list = []
    for i, row in df.iterrows():
        reviews_list.append(Reviews(title=row['Title'], review=row['Review'], rating=row['Rating'], date=row['Date']))
    return reviews_list