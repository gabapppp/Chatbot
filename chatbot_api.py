from fastapi import FastAPI, HTTPException
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Initialize FastAPI app
app = FastAPI()

CHROMA_PATH = "chroma"

# Define the prompt template with named placeholders
TEMPLATE_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# Define the request schema for the input query
class QueryRequest(BaseModel):
    query_text: str

def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    prompt_template = ChatPromptTemplate.from_template(TEMPLATE_PROMPT)
    prompt = prompt_template.format(
      instruction=query_text,
      input="",
      output=""
    )
    # print(prompt)

    model = Ollama(model="unsloth_model")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

# Define a POST endpoint to handle chatbot queries
@app.post("/query/")
async def query_chatbot(request: QueryRequest):
    try:
        response = query_rag(request.query_text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start the FastAPI app (this can be done with `uvicorn`)
