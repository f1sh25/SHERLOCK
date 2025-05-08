import os
import psycopg2
import libsql_experimental as libsql 
from typing import Literal
from autogen import ConversableAgent, LLMConfig
from autogen.tools import Tool, Depends
from autogen.tools.dependency_injection import on
from typing import Annotated, Any, Optional, Union, Callable
from pydantic import BaseModel, Field
import copy
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import OpenAI
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import *
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_postgres import PGVector as PostgresVector
from langchain_postgres import PGEngine, PGVectorStore




metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]


class Database(BaseModel):
    type: Literal["libsql", "postgres"] = Field(
        description="The type of database to use. Can be either 'libsql' or 'postgres'."
    )
    host: Optional[str] = Field(
        default=None,
        description="The host of the database. Required for 'postgres' type."
    )
    port: Optional[int] = Field(
        default=None,
        description="The port of the database. Required for 'postgres' type."
    )
    user: Optional[str] = Field(
        default=None,
        description="The user for the database. Required for 'postgres' type."
    )
    password: Optional[str] = Field(
        default=None,
        description="The password for the database. Required for 'postgres' type."
    )
    db_name: str = Field(description="name of the database")

class Data(BaseModel):
        """
        Class specifies table data that's going to be embedded into the database.
        - type: webpage, document
        - title: if webpage => page URL or document name
        - content: the actual content to be stored
        """
        type: Literal["webpage", "document"] = Field(
            description="The type of data, either 'webpage' or 'document'."
        )
        source: str = Field(
            description="The title of the data. For a webpage, this is the page URL. For a document, this is the document name."
        )
        content: str = Field(
            description="The actual content to be stored in the database."
        )

class VectorDb():
    def __init__(self, database:Database, EMBED_KEY=None):
        self.database = database
        os.environ["USER_AGENT"] = "LangChainApp/1.0"
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.collection_name = "my_docs"
        self.embedding_dim = 1536  # Size for text-embedding-3-large
        self._setup_vector_store()

    def _setup_vector_store(self):
        if self.database.type == "postgres":
            connection_string = f"postgresql+psycopg://{self.database.user}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.db_name}"
            try:
                self.engine = PGEngine.from_connection_string(url=connection_string)
                
                # Check if table exists with correct dimensions
                try:
                    with psycopg2.connect(
                        dbname=self.database.db_name,
                        user=self.database.user,
                        password=self.database.password,
                        host=self.database.host,
                        port=self.database.port
                    ) as conn:
                        with conn.cursor() as cur:
                            # Check if table exists
                            cur.execute("""
                                SELECT EXISTS (
                                    SELECT FROM pg_tables
                                    WHERE schemaname = 'public'
                                    AND tablename = 'my_docs'
                                );
                            """)
                            table_exists = cur.fetchone()[0]
                            
                            if not table_exists:
                                print("Initializing vector store table...")
                                self.engine.init_vectorstore_table(
                                    table_name=self.collection_name,
                                    vector_size=self.embedding_dim
                                )
                                print(f"Vector store initialized with dimension {self.embedding_dim}")
                            else:
                                print("Vector store table already exists, skipping initialization")
                                
                except Exception as e:
                    print(f"Warning: Could not check table existence: {e}")
                    # If we can't check, try to initialize anyway
                    self.engine.init_vectorstore_table(
                        table_name=self.collection_name,
                        vector_size=self.embedding_dim
                    )
            except Exception as e:
                print(f"Failed to setup PostgreSQL vector store: {e}")
                self.engine = None

    def connect_to_db(self):
        if self.database.type == "libsql":
            """
            TODOOOOO
            """
            conn = libsql.connect("local_vector.db")
            if conn is None:
                print("Failed to connect to the database.")
                return None
            print("Connected to the database successfully.")
            return conn
        
        else:
            if self.engine is None:
                return None
            
            try:
                vector_store = PGVectorStore.create_sync(
                    engine=self.engine,
                    table_name=self.collection_name,
                    embedding_service=self.embeddings
                )
                return vector_store
            except Exception as e:
                print(f"Failed to connect to PostgreSQL: {e}")
                return None
    
    def embedding(self, data: Data):
        vector_store = self.connect_to_db()
        if vector_store is None:
            return

        text_splitter = SemanticChunker(self.embeddings)
        docs = text_splitter.create_documents([data.content])

        try:
            vector_store.add_documents(docs)
            print(f"Successfully embedded {len(docs)} documents")
        except Exception as e:
            print(f"Error embedding documents: {e}")
            
    def semantic_search(self, query: str, k: int = 3):
        """
        Perform semantic search on the vector database
        Args:
            query (str): The search query
            k (int): Number of results to return
        Returns:
            List of tuples containing (Document, score)
        """
        vector_store = self.connect_to_db()
        if vector_store is None:
            return []
            
        try:
            results = vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []

class EmbedTool(Tool):
    def __init__(self, database: Database, EMBED_KEY: Optional[str] = None):
        self.database = database
        self.vector_db = VectorDb(database=database, EMBED_KEY=EMBED_KEY)
        
        def embed_data(data: Data) -> str:
            """Embed data into the vector database.
            
            Args:
                data: The data to embed, containing type, source, and content.
                
            Returns:
                str: A message indicating success or failure.
            """
            try:
                self.vector_db.embedding(data)
                return f"Successfully embedded data from {data.source}"
            except Exception as e:
                return f"Failed to embed data: {str(e)}"

        super().__init__(
            name="embed",
            description="Embeds documents or webpage content into the vector database for semantic search capability. Accepts data with type, source, and content fields.",
            func_or_tool=embed_data
        )
    
class SearchTool(Tool):
    def __init__(self, database: Database, EMBED_KEY: Optional[str] = None):
        self.database = database
        self.vector_db = VectorDb(database=database, EMBED_KEY=EMBED_KEY)
        
        def search(query: str, k: int = 3) -> list:
            """
            Search the vector database for relevant information.
            
            Args:
                query: The search query string to find relevant documents
                k: Number of results to return (default: 3)
                
            Returns:
                List of tuples containing (Document, similarity_score)
            """
            return self.vector_db.semantic_search(query, k=k)

        super().__init__(
            name="search",
            description="Searches through embedded documents using semantic similarity to find relevant information. Returns the k most similar documents and their scores.",
            func_or_tool=search
        )

    
if __name__ == "__main__":
    # Test configuration
    test_db = Database(
        type="postgres",
        host="localhost",
        port=5432,
        user="spider",
        password="spider",
        db_name="spider"
    )

    # Initialize EmbedTool with test configuration
    embed_tool = EmbedTool(database=test_db, EMBED_KEY=os.getenv("OPENAI_API_KEY"))

    # Create test document
    test_doc = Data(
        type="document",
        source="test_article.txt",
        content="""
        Artificial Intelligence and Machine Learning in 2025
        The field of AI has seen remarkable progress in recent years.
        Natural language processing and computer vision have become
        integral parts of modern software systems. Deep learning
        models continue to push the boundaries of what's possible
        in automation and decision-making processes.
        """
    )

    # Test embedding functionality
    print("Testing document embedding...")
    result = embed_tool.__call__
    print(result)

    # Test semantic search functionality
    print("\nTesting semantic search...")
    test_queries = [
        "What are the main applications of AI?",
        "How has AI progressed recently?",
        "What is the role of deep learning?"
    ]

    for query in test_queries:
        print(f"\nSearching for: {query}")
        try:
            results = embed_tool.search(query, k=2)
            print(f"Found {len(results)} results:")
            for doc, score in results:
                print(f"Score: {score:.4f}")
                print(f"Content: {doc.page_content}")
                print(f"Metadata: {doc.metadata}\n")
        except Exception as e:
            print(f"Error during search: {e}")