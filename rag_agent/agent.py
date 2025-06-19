from google.adk.agents import Agent
from google_adk.rag import RagTool  # This is a potential import path
from google.adk.embeddings.text_splitter import CharacterTextSplitter
from google.adk.embeddings.embeddings_index import LocalVectorStore
from google.adk.embeddings.google_embedder import GoogleTextEmbedder
from google.adk.loaders.text_loader import TextLoader

# Load and split
loader = TextLoader("D:\Projects\RAG_ADK_KP\data\# 👑 King Cultural Spot.txt")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=300)
split_docs = splitter.split_documents(docs)

embedder = GoogleTextEmbedder()
index = LocalVectorStore()
index.add_documents(split_docs, embedder)

rag_tool = RagTool(index=index, embedder=embedder)

root_agent = Agent(
    name="rag_agent",
    model="gemini-2.0-flash",
    description="Document reading agent",
    tools=[rag_tool],
    prompt_template="""
You are a helpful assistant for King Cultural Spot Dance Academy.
Use the retrieved context to answer questions clearly and accurately.
If the answer is not in the context, say "I don’t have that information."

{{context}}

User: {{input}}
Agent:"""
)
