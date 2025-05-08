# Research Agent System

A sophisticated multi-level research system that employs a cost-effective and efficient approach to information gathering and analysis.

## Credits

This research system builds upon the foundation laid by the DeepResearch tool from [ag2](https://github.com/ag2ai/ag2/blob/main/autogen/tools/experimental/deep_research/deep_research.py). While maintaining the core concept of progressive research depth, we've extended the functionality with a specialized focus on RAG-first approach and persistent memory features.

## Overview

This system implements a tiered research strategy that progressively moves from fast, cheap lookups to more intensive web searches based on the information needs. The system includes:

1. **Vector Database (RAG) Search** - First Level
   - Fastest and most cost-effective method
   - Searches through previously embedded knowledge
   - Uses PostgreSQL with vector extensions for semantic search
   - Ideal for frequently accessed information

2. **Web Search** - Second Level
   - Activated when RAG search doesn't yield sufficient information
   - Uses structured web searches to find relevant information
   - More costly but provides broader information access
   - Results are automatically embedded back into the vector database for future use

3. **Deep Web Crawling** - Third Level
   - Most intensive and thorough research method
   - Activated when specific in-depth information is needed
   - Crawls through related web pages to gather comprehensive information
   - Includes smart memory features to store and relate information

## Smart Memory Features

- Automatic embedding of new information into the vector database
- Progressive learning from web searches and crawls
- Semantic chunking for optimal information storage
- Metadata tracking for source attribution

## Components

### ResearchAgent

The main agent class that coordinates the research process and manages the progression through different research levels.

### ResearchTool

Implements the core research functionality including:
- Question decomposition
- Multi-agent coordination
- Progressive research strategy
- Result synthesis

### Vector Database Tools

- `EmbedTool`: Handles document embedding into the vector database
- `SearchTool`: Manages semantic search operations
- Supports both PostgreSQL and LibSQL backends

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
- `OPENAI_API_KEY`: For embeddings and LLM operations
- Database configuration (for PostgreSQL):
  - host
  - port
  - user
  - password
  - database name

## Benefits

1. **Cost Optimization**
   - Uses cheaper RAG searches before expensive web operations
   - Stores results for future use
   - Reduces redundant web searches

2. **Progressive Research**
   - Starts with simple solutions
   - Escalates to more thorough methods only when needed
   - Maintains context across research levels

3. **Smart Memory Management**
   - Automatically stores new information
   - Improves future research efficiency
   - Reduces duplicate research efforts