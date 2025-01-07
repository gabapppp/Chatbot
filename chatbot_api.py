from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Initialize FastAPI app
app = FastAPI()

# Path to Chroma database
CHROMA_PATH = "chroma"

# Prompt template
TEMPLATE_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# Request schema for input query
class QueryRequest(BaseModel):
    query_text: str

def get_embedding_function():
    """Get the embedding function for the Chroma database."""
    return OllamaEmbeddings(model="nomic-embed-text")

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform similarity search to retrieve relevant documents from the DB.
    results = db.similarity_search_with_score(query_text, k=1)

    # Extract the top k results' text to form the context for the response
    context = "\n".join([doc.page_content for doc, _score in results])
    # context = ""  # Limit the context to 5000 characters
    # print(context)
    # Create the prompt template for the model, including the retrieved context
    prompt_template = ChatPromptTemplate.from_template(TEMPLATE_PROMPT)
    prompt = prompt_template.format(
        instruction=query_text,
        input=context,  # Add the retrieved context here
        output=""
    )

    # Call the model (ensure the model name is correct)
    model = Ollama(model="chatbot")
    response_text = model.invoke(prompt)

    # Extract sources from the search results
    sources = [doc.metadata.get("id", "Unknown") for doc, _score in results]

    # # Format the final response, including sources
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)


    return {
        "response": response_text,
        "sources": sources
    }

# Define POST endpoint for chatbot queries
@app.post("/query/")
async def query_chatbot(request: QueryRequest):
    """
    Handle incoming chatbot queries and return the response.
    """
    try:
        result = query_rag(request.query_text)
        return result
    except Exception as e:
        # Catch any errors and return a 500 HTTP exception
        raise HTTPException(status_code=500, detail=str(e))