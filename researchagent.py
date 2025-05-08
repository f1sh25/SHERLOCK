from typing import Any, Optional, Union
from autogen import ConversableAgent,UserProxyAgent, AssistantAgent ,LLMConfig
from research import ResearchTool
import os



class ResearchAgent(ConversableAgent):
    """A research agent designed for retrieving and analyzing information from a vector database.
    
    This agent specializes in:
    - Retrieving relevant information from an embedded vector database
    - Breaking down complex queries into searchable components
    - Synthesizing information from retrieved documents
    - Providing detailed, source-backed responses
    - Maintaining context throughout the research process
    """

    DEFAULT_PROMPT = """You are a specialized research agent that works with a vector database to find and analyze information."""

    def __init__(
        self,
        name: str,
        llm_config: Optional[Union[LLMConfig, dict[str, Any]]] = None,
        system_message: Optional[Union[str, list[str]]] = DEFAULT_PROMPT,
        max_web_steps: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the ResearchAgent.

        Args:
            name: The name of the agent.
            llm_config: The LLM configuration.
            system_message: The system message. Defaults to DEFAULT_PROMPT.
            max_web_steps: The maximum number of web steps. Defaults to 30.
            **kwargs: Additional keyword arguments to pass to the ConversableAgent.
        """
        llm_config = LLMConfig.get_current_llm_config(llm_config)  # type: ignore[arg-type]

        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs,
        )

        self.tool = ResearchTool(
            llm_config=llm_config,  # type: ignore[arg-type]
        )

        self.register_for_llm()(self.tool)



if __name__ == "__main__":
    llm_config = LLMConfig(
        config_list=[{"api_type": "openai", "model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}],
    )

    with llm_config:
        agent = ResearchAgent(name="ResearchAgent")

    message = "Who is joni savolainen and what does the company do where he works at"

    result = agent.run(
        message=message,
        tools=agent.tools,
        max_turns=2,
        user_input=False,
        summary_method="reflection_with_llm",
    )
    result.process()
    print(result.summary)