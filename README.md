This system is currently a **Proof of Concept (POC)** and is not intended for production use.

## Overview

My system employs a tiered research strategy, progressively moving from rapid, low-cost lookups to more intensive web searches as needed. This approach prioritizes a **RAG-first (Retrieval Augmented Generation) strategy** and incorporates **persistent memory features**.

### Research Levels

1.  **Vector Database (RAG) Search - First Level:**

      * This is the fastest and most economical method.
      * It searches through previously embedded knowledge stored in a PostgreSQL vector database, ideal for frequently accessed information.

2.  **Web Search - Second Level:**

      * Activated when the RAG search doesn't provide enough information.
      * It performs structured web searches, offering broader information access at a higher cost.
      * Results are automatically embedded into the vector database for future use.

3.  **Deep Web Crawling - Third Level:**

      * The most intensive and thorough research method, used for in-depth information needs.
      * It crawls related web pages for comprehensive information.
      * Includes smart memory features to store and relate information.



## Smart Memory Features

My system automatically embeds new information into the vector database, enabling progressive learning from web searches and crawls. It uses semantic chunking for optimal information storage and tracks metadata for source attribution.


## Components

  * **ResearchAgent:** The main agent class that orchestrates the research process and manages progression through the different research levels.
  * **ResearchTool:** Implements core research functionalities, including question decomposition, multi-agent coordination, progressive research strategy, and result synthesis.
  * **Vector Database Tools:** Includes `EmbedTool` for document embedding and `SearchTool` for semantic search, supporting both PostgreSQL and LibSQL backends.


## Usage

```python
from researchagent import ResearchAgent

# Initialize the research agent
agent = ResearchAgent(name="ResearchAgent")

# Run a research query
result = agent.run(
    message="Your research question here",
    max_turns=2,
    user_input=False
)
```


## Configuration

The system requires the following environment variables:

  * `OPENAI_API_KEY`: For embeddings and LLM operations.
  * **Database Configuration (for PostgreSQL):** `host`, `port`, `user`, `password`, `database name`.


## Benefits

  * **Cost Optimization:** Leverages cheaper RAG searches before more expensive web operations and stores results to reduce redundant searches.
  * **Progressive Research:** Starts with simpler solutions and escalates to more thorough methods only when necessary, maintaining context across research levels.
  * **Smart Memory Management:** Automatically stores new information, improving future research efficiency and reducing duplicate efforts.
