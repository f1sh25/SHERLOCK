from autogen import ConversableAgent, LLMConfig
from autogen.tools import Tool, Depends
from autogen.tools.dependency_injection import on
from typing import Annotated, Any, Optional, Union, Callable
from pydantic import BaseModel, Field
from rag_tool import EmbedTool, SearchTool, Database, Data
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat import initiate_group_chat
import copy

"""ist  a research pipeline for ag2 agents. when it gets a research task in terms of a question it will first get the subtasks (n. number)
then it will be passed to research agent that first tries to get answer from vector database then if it deems that there is not enough information it will use the web agent to get the answer. 
that uses web search if it still needs aditional information. if it needs more in depth information it will use websurfer agent to crawl pages. new information gathered from web will be stored in the vector database.
once the research is deem to contain enough information to answer the question it will pass the information to the review agent that will check if the information is correct and if it is not it will ask the research agent to get more information"""

class Subquestion(BaseModel):
    question: Annotated[str, Field(description="The original question.")]

    def format(self) -> str:
        return f"Question: {self.question}\n"


class SubquestionAnswer(Subquestion):
    answer: Annotated[str, Field(description="The answer to the question.")] = Field(default="No answer found")

    def format(self) -> str:
        formatted_answer = self.answer if self.answer else "No answer found"
        return f"Question: {self.question}\nAnswer: {formatted_answer}\n"


class Task(BaseModel):
    question: Annotated[str, Field(description="The original question.")]
    subquestions: Annotated[list[Subquestion], Field(description="The subquestions that need to be answered.")]

    def format(self) -> str:
        return f"Task: {self.question}\n\n" + "\n".join(
            "Subquestion " + str(i + 1) + ":\n" + subquestion.format()
            for i, subquestion in enumerate(self.subquestions)
        )


class CompletedTask(BaseModel):
    question: Annotated[str, Field(description="The original question.")]
    subquestions: Annotated[list[SubquestionAnswer], Field(description="The subquestions and their answers")]

    def format(self) -> str:
        formatted_output = [f"Task: {self.question}\n"]
        
        for i, subquestion in enumerate(self.subquestions, 1):
            section = f"\nSubquestion {i}:\n{subquestion.format()}"
            formatted_output.append(section)
            
        return "".join(formatted_output)


class InformationCrumb(BaseModel):
    source_url: str
    source_title: str
    source_summary: str
    relevant_info: str


class GatheredInformation(BaseModel):
    information: list[InformationCrumb]

    def format(self) -> str:
        return "Here is the gathered information: \n" + "\n".join(
            f"URL: {info.source_url}\nTitle: {info.source_title}\nSummary: {info.source_summary}\nRelevant Information: {info.relevant_info}\n\n"
            for info in self.information
        )


class ResearchBucket:
    gathered_info: list[Data] = Field(default_factory=list)
    is_complete: bool = False


class ResearchTool(Tool):
    ANSWER_CONFIRMED_PREFIX = "Answer confirmed:"
    SUBQUESTIONS_ANSWER_PREFIX = "Subquestions answered:"

    def __init__(self,
                 llm_config: Optional[Union[LLMConfig, dict[str, Any]]] = None,
                 max_web_steps: int = 30):
        self.llm_config = llm_config 

        self.summarizer_agent = ConversableAgent(
            name="SummarizerAgent",
            system_message=(
                "You are an agent with a task of answering the question provided by the user."
                "First you need to split the question into subquestions by calling the 'split_question_and_answer_subquestions' method."
                "Then you need to sintesize the answers the original question by combining the answers to the subquestions."
            ),
            is_termination_msg=lambda x: x.get("content", "")
            and x.get("content", "").startswith(self.ANSWER_CONFIRMED_PREFIX),
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        self.critic_agent = ConversableAgent(
            name="CriticAgent",
            system_message=(
                "You are a critic agent responsible for evaluating the answer provided by the summarizer agent.\n"
                "Your task is to assess the quality of the answer based on its coherence, relevance, and completeness.\n"
                "Provide constructive feedback on how the answer can be improved.\n"
                "If the answer is satisfactory, call the 'confirm_answer' method to end the task.\n"
            ),
            is_termination_msg=lambda x: x.get("content", "")
            and x.get("content", "").startswith(self.ANSWER_CONFIRMED_PREFIX),
            llm_config=llm_config,
            human_input_mode="NEVER",
        )



        def delegate_research_task(
            task: Annotated[str, "The task to perform a research on."],
            llm_config: Annotated[Union[LLMConfig, dict[str, Any]], Depends(on(llm_config))],
            max_web_steps: Annotated[int, Depends(on(max_web_steps))],
        ) -> str:
            """Delegate a research task to the agent.

            Args:
                task (str): The task to perform a research on.
                llm_config (LLMConfig, dict[str, Any]): The LLM configuration.
                max_web_steps (int): The maximum number of web steps.

            Returns:
                str: The answer to the research task.
            """



            # Adds toolcall to critic that will be used to confirm the answer
            @self.summarizer_agent.register_for_execution()
            @self.critic_agent.register_for_llm(description="Call this method to confirm the final answer.")

            def confirm_summary(answer: str, reasoning: str) -> str:
                return f"{self.ANSWER_CONFIRMED_PREFIX}" + answer + "\nReasoning: " + reasoning


            split_question_and_answer_subquestions = ResearchTool._get_split_question_and_answer_subquestions(
                llm_config=llm_config,
                max_web_steps=max_web_steps,
            )
            # Adds toolcall to summarizer that will be used to split the question and answer subquestions

            self.summarizer_agent.register_for_llm(description="Split the question into subquestions and get answers.")(
                split_question_and_answer_subquestions
            )
            self.critic_agent.register_for_execution()(split_question_and_answer_subquestions)

            result = self.critic_agent.initiate_chat(
                self.summarizer_agent,
                message="Please answer the following question: " + task,
                # This outer chat should preserve the history of the conversation
                clear_history=False,
            )

            return result.summary
            


            
        
        super().__init__(
            name="ResearchTool",
            description="A tool for conducting research and gathering information.",
            func_or_tool=delegate_research_task)

    @staticmethod
    def _get_split_question_and_answer_subquestions(
        llm_config: Union[LLMConfig, dict[str, Any]], max_web_steps: int
    ) -> Callable[..., Any]:
        def split_question_and_answer_subquestions(
            question: Annotated[str, "The question to split and answer."],
            llm_config: Annotated[Union[LLMConfig, dict[str, Any]], Depends(on(llm_config))],
            max_web_steps: Annotated[int, Depends(on(max_web_steps))],
        ) -> str:
            decomposition_agent = ConversableAgent(
                name="DecompositionAgent",
                system_message=(
                    "You are an expert at breaking down complex questions into smaller, focused subquestions.\n"
                    "Your task is to take any question provided and divide it into clear, actionable subquestions that can be individually answered.\n"
                    "Ensure the subquestions are logical, non-redundant, and cover all key aspects of the original question.\n"
                    "Avoid providing answers or interpretations—focus solely on decomposition.\n"
                    "Do not include banal, general knowledge questions\n"
                    "Do not include questions that go into unnecessary detail that is not relevant to the original question\n"
                    "Do not include question that require knowledge of the original or other subquestions to answer\n"
                    "Some rule of thumb is to have only one subquestion for easy questions, 3 for medium questions, and 5 for hard questions.\n"
                ),
                llm_config=llm_config,
                is_termination_msg=lambda x: x.get("content", "")
                and x.get("content", "").startswith(ResearchTool.SUBQUESTIONS_ANSWER_PREFIX),
                human_input_mode="NEVER",
            )

            example_task = Task(
                question="What is the capital of France?",
                subquestions=[Subquestion(question="What is the capital of France?")],
            )
            decomposition_critic = ConversableAgent(
                name="DecompositionCritic",
                system_message=(
                    "You are a critic agent responsible for evaluating the subquestions provided by the initial analysis agent.\n"
                    "You need to confirm whether the subquestions are clear, actionable, and cover all key aspects of the original question.\n"
                    "Do not accept redundant or unnecessary subquestions, focus solely on the minimal viable subset of subqestions necessary to answer the original question. \n"
                    "Do not accept banal, general knowledge questions\n"
                    "Do not accept questions that go into unnecessary detail that is not relevant to the original question\n"
                    "Remove questions that can be answered with combining knowledge from other questions\n"
                    "After you are satisfied with the subquestions, call the 'generate_subquestions' method to answer each subquestion.\n"
                    "This is an example of an argument that can be passed to the 'generate_subquestions' method:\n"
                    f"{{'task': {example_task.model_dump()}}}\n"
                    "Some rule of thumb is to have only one subquestion for easy questions, 3 for medium questions, and 5 for hard questions.\n"
                ),
                llm_config=llm_config,
                is_termination_msg=lambda x: x.get("content", "")
                and x.get("content", "").startswith(ResearchTool.SUBQUESTIONS_ANSWER_PREFIX),
                human_input_mode="NEVER",
            )

            generate_subquestions = ResearchTool._get_generate_subquestions(
                llm_config=llm_config, max_web_steps=max_web_steps
            )
            decomposition_agent.register_for_execution()(generate_subquestions)
            decomposition_critic.register_for_llm(description="Generate subquestions for a task.")(
                generate_subquestions
            )

            result = decomposition_critic.initiate_chat(
                decomposition_agent,
                message="Analyse and gather subqestions for the following question: " + question,
            )

            return result.summary

        return split_question_and_answer_subquestions

    @staticmethod
    def _get_generate_subquestions(
        llm_config: Union[LLMConfig, dict[str, Any]],
        max_web_steps: int,
    ) -> Callable[..., str]:
        def generate_subquestions(
            task: Task,
            llm_config: Annotated[Union[LLMConfig, dict[str, Any]], Depends(on(llm_config))],
            max_web_steps: Annotated[int, Depends(on(max_web_steps))],
        ) -> str:
            if not task.subquestions:
                task.subquestions = [Subquestion(question=task.question)]

            # Create a research bucket for this task
            research_bucket = ResearchBucket()
            
            subquestions_answers: list[SubquestionAnswer] = []
            for subquestion in task.subquestions:
                answer = ResearchTool._answer_question(
                    subquestion.question, 
                    llm_config=llm_config, 
                    max_web_steps=max_web_steps,
                    bucket=research_bucket
                )

                
                # Create a SubquestionAnswer with proper validation
                answer_obj = SubquestionAnswer(
                    question=subquestion.question,
                    answer=answer if answer and not answer.isspace() else "No answer found"
                )
                subquestions_answers.append(answer_obj)

            # Now that all subquestions are answered, mark bucket as complete and embed all information
            research_bucket.is_complete = True
            ResearchTool._embed_collected_information(research_bucket)

            completed_task = CompletedTask(question=task.question, subquestions=subquestions_answers)
            formatted_output = completed_task.format()
            # Only add prefix if we actually have answers
            if any(answer.answer != "No answer found" for answer in subquestions_answers):
                return f"{ResearchTool.SUBQUESTIONS_ANSWER_PREFIX}\n{formatted_output}"
            return formatted_output

        return generate_subquestions

    @staticmethod
    def _answer_question(
        question: str,
        llm_config: Union[LLMConfig, dict[str, Any]],
        max_web_steps: int,
        bucket: Optional[ResearchBucket] = None,
    ) -> tuple[str, Optional[GatheredInformation]]:
        TEST_DB = Database(
            type="postgres",
            host="localhost",
            port=5432,
            user="spider",
            password="spider",
            db_name="spider"
        )

        from autogen.agents.experimental.websurfer import WebSurferAgent

        websurfer_config = copy.deepcopy(llm_config)
        websurfer_config["config_list"][0]["response_format"] = GatheredInformation

        def is_termination_msg(x: dict[str, Any]) -> bool:
            content = x.get("content", "")
            return (content is not None) and content.startswith(ResearchTool.ANSWER_CONFIRMED_PREFIX)


        websurfer_agent = WebSurferAgent(
            llm_config=llm_config,
            web_tool_llm_config=websurfer_config,
            name="WebSurferAgent",
            system_message=(
                "If the rag agent cant find enough information to answer the question, use your browser_use tool. "
                "You are a web surfer agent responsible for gathering information from the web to provide information for answering a question. "
                "You will be asked to find information related to the question and provide a summary of the information gathered. "
                "The summary should include the URL, title, summary, and relevant information for each piece of information gathered. "
            ),
            is_termination_msg=is_termination_msg,
            human_input_mode="NEVER",
            web_tool_kwargs={
                "agent_kwargs": {"max_steps": max_web_steps},
            },
        )

        ragsearch_agent = ConversableAgent(
            name="RagSearchAgent",
            system_message=(
                "You are a search agent specializing in vector database retrieval.\n"
                "Your task is to search the vector database for relevant information to answer queries.\n"
                "For each retrieved result, provide: URL, title, summary, and key relevant details.\n"
                "Focus on semantic similarity and exact matches to ensure accurate information retrieval.\n"
                "IF IT SEEMS THAT YOU CANT FIND THE INFORMATION THAT YOU NEED SAY SO"
            ),
            llm_config=llm_config,
            is_termination_msg=is_termination_msg,
            human_input_mode="NEVER",
        )

        rag_search = SearchTool(database=TEST_DB)
        ragsearch_agent.register_for_llm()(rag_search)

        critic = ConversableAgent(
            name="WebSurferCritic",
            system_message=(
                "You are a critic agent responsible for evaluating the answer provided by the web surfer agent.\n"
                "Your task is to:\n"
                "1. Review information from both rag search and web search\n"
                "2. Determine if the information is sufficient to answer the question\n"
                "3. If the information is sufficient, call the confirm_answer method with a clear, concise answer\n"
                "4. If the information is insufficient, ask the web surfer or rag search agent for more specific information\n"
                "Important: Always use the confirm_answer method to provide the final answer when you have sufficient information.\n"
                "The conversation will continue until you call confirm_answer with a satisfactory answer."
            ),
            llm_config=llm_config,
            is_termination_msg=is_termination_msg,
            human_input_mode="NEVER",
        )

        critic.register_for_execution()(rag_search)

        @ragsearch_agent.register_for_execution()
        @critic.register_for_llm(
            description="Call this method when you agree that the original question can be answered with the gathered information and provide the answer."
        )
        def confirm_answer(answer: str) -> str:
            return f"{ResearchTool.ANSWER_CONFIRMED_PREFIX} " + answer

        critic.register_for_execution()(websurfer_agent.tool)

        pattern = AutoPattern(
            initial_agent=ragsearch_agent,  # Agent that starts the conversation
            agents=[websurfer_agent, ragsearch_agent, critic],
            group_manager_args={"llm_config": llm_config}
        )

        ChatResult, _, _ = initiate_group_chat(
            pattern=pattern,
            messages="Please find the answer to this question: " + question,
            max_rounds=30,
        )

        # Extract the answer and any gathered information
        summary = ChatResult.summary

        # Find the last message from WebSurferAgent before the final confirmation
        last_websurfer_msg = None
        for msg in reversed(ChatResult.chat_history):
            if msg.get('name') == 'WebSurferAgent' and msg.get('role') == 'user':
                last_websurfer_msg = msg.get('content')
                break

        # If we found websurfer info and have a bucket, store it
        if last_websurfer_msg and bucket is not None:
            gathered_info = Data(
                type="webpage",  # Since this comes from web surfing
                source="websurfer_result",
                content=last_websurfer_msg
            )
            bucket.gathered_info.append(gathered_info)

        answer = summary.replace(ResearchTool.ANSWER_CONFIRMED_PREFIX, "").strip()
        if not answer or answer.isspace():
            answer = "No confirmed answer was found."

        return answer

    @staticmethod
    def _embed_collected_information(bucket: ResearchBucket) -> None:
        if not bucket.is_complete:
            return
            
        # Create a single database connection for all embeddings
        TEST_DB = Database(
            type="postgres",
            host="localhost",
            port=5432,
            user="spider",
            password="spider",
            db_name="spider"
        )
        
        embed_tool = EmbedTool(database=TEST_DB)
        
        # Embed all collected information at once
        for info in bucket.gathered_info:
            if info and hasattr(info, '_collected_data'):
                args, kwargs = info._collected_data
                embed_tool(*args, **kwargs)



