import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "products"

# Initialize embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Load the CSV file
df = pd.read_csv("products_export_new.csv")
df = df.dropna(subset=["Handle", "Title", "Body (HTML)"])

# Filter only published products
df = df[df["Published"] == True]


# Create a text column for embedding
def clean_html(text):
    import re
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', str(text))


df["clean_body"] = df["Body (HTML)"].apply(clean_html)
df["text"] = (
        df["Title"].astype(str)
        + ". " + df["clean_body"].astype(str)
        + ". Tags: " + df["Tags"].fillna("")
        + ".\n\nLink: https://beattrangi.com/products/" + df["Handle"].astype(str)
)

# Convert rows into documents
from langchain_core.documents import Document

documents = [
    Document(
        page_content=row["text"],
        metadata={
            "id": str(row["Handle"]),
            "title": str(row["Title"]),
            "vendor": str(row.get("Vendor", "")) if pd.notna(row.get("Vendor", "")) else "",
            "type": str(row.get("Type", "")) if pd.notna(row.get("Type", "")) else "",
            "tags": str(row.get("Tags", "")) if pd.notna(row.get("Tags", "")) else "",
            "link": f"https://beattrangi.com/products/{row['Handle']}"
        }
    )
    for _, row in df.iterrows()
]

# Upsert to Pinecone
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("\n✅ Finished embedding and upserting published product data to Pinecone.")
