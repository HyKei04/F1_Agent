import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from smolagents import CodeAgent, GoogleSearchTool, VisitWebpageTool, LiteLLMModel, ActionStep, GradioUI
from data_agent_tools import get_telemetry_fastest_lap, get_fastest_lap, get_session_results, get_all_laps
from tools import plot_fastest_lap_telemetry, plot_race_pace, RetrieverTool, analize_graph, plot_race_strategy
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from dotenv import load_dotenv

load_dotenv()

#Define the model
model = LiteLLMModel(model_id="openrouter/openai/gpt-4o-mini",
                     api_base="https://openrouter.ai/api/v1",
                     api_key=os.environ["OPENROUTER_API_KEY"],
                     max_new_tokens=2000)


pdf_master_directory = "pdfs"
pdf_directories = ["Spanish", "Monaco", "Emilia%20Romagna", "Miami",
    "Saudi%20Arabian", "Bahrain", "Japanese", "Chinese", "Australian", "sporting_regulations"]

retriever_tools = []

#Define the text splitter for the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-base",encode_kwargs={"normalize_embeddings": True})

#Create retrieval tools and vector databases

for pdf_directory in pdf_directories:
    pdf_files = [
        os.path.join(pdf_master_directory,pdf_directory, f)
        for f in os.listdir(os.path.join(pdf_master_directory,pdf_directory))
        if f.endswith(".pdf")
    ]
    source_docs = []
    for file_path in pdf_files:
        loader = PyPDFLoader(file_path)
        source_docs.extend(loader.load())

    #Split the documents with the text splitter
    docs_processed = text_splitter.split_documents(source_docs)
    #Create FAISS vector database with the embedding model
    vectordb = FAISS.from_documents(documents=docs_processed, embedding=embedding_model,
                                    distance_strategy=DistanceStrategy.COSINE)
    #Create retriever tool
    retriever_tool = RetrieverTool(vectordb)
    #Assign personalized description for every tool
    if pdf_directory == "sporting_regulations":
        retriever_tool.set_description_for_regulations()
    else:
        retriever_tool.set_description_for_race(pdf_directory)
    retriever_tools.append(retriever_tool)

#Create data agent
data_agent = CodeAgent(
    tools=[get_fastest_lap, get_telemetry_fastest_lap, get_session_results, get_all_laps],
    model=model,
    name="data_agent",
    description="Gets F1 lap data and session results from an F1 API updated until 2025 (included)",
    additional_authorized_imports=[
        "pandas",
        "numpy",
    ],
    max_steps=7,
    verbosity_level=1,
)
#Modify data agent system prompt by adding some rules
data_agent.prompt_templates["system_prompt"] = data_agent.prompt_templates["system_prompt"] + ("\nIf you are given a task that includes ALL the drivers, "
                                                                                               "first use the get_session_results tool to extract their abbreviations.\n"
                                                                                               "To retrieve the fastest lap of a session, use the get_fastest_lap tool.\n"
                                                                                               "To retrieve all the laps of a session, use the get_all_laps tool.\n"
                                                                                               "If a task involves getting lap data for multiple drivers, save their "
                                                                                               "data in separate parquet files\n"
                                                                                               "Only use the get_telemetry_fastest_lap_telemetry tool if the word telemetry"
                                                                                               "is included in the task.\n"
                                                                                               "You can open the parquet files to check if they have the information needed,"
                                                                                               "however, ALWAYS return only as your final answer the names of the parquet files, "
                                                                                               "DO NOT include their contents\n"
                                                                                               "DO NOT simulate any data")

#Create retriever agent with the tools previously created
retriever_agent = CodeAgent(
    tools=retriever_tools,
    model=model,
    name="retriever_agent",
    description="Retrieves information about penalties, technical changes, and sporting regulations from the 2025 season",
    max_steps=7,
    verbosity_level=1
)
#Modify retriever agent system prompt
retriever_agent.prompt_templates["system_prompt"] = retriever_agent.prompt_templates["system_prompt"] + ("\nOnly use your available tools to retrieve information\n"
                                                                                                         "If your task is penalty related, consider checking the sporting"
                                                                                                         "regulations for additional information of the article breached")

#Create web search agent
web_search_agent = CodeAgent(
    tools=[GoogleSearchTool(provider="serper"), VisitWebpageTool()],
    model=model,
    name="web_search_agent",
    description="Performs web search to retrieve F1 related data such as driver standings or circuit information",
    max_steps=5,
)

#Define function to control input tokens, if the last call to the model exceeded 10000 tokens,
#remove first message of the conversation from memory
def update_memory(memory_step: ActionStep, agent: CodeAgent) -> None:
    if hasattr(agent.model, "last_input_token_count"):
        if(agent.model.last_input_token_count > 10000):
            agent.memory.steps.pop(0)

#Create main agent (graph agent) with the tools to build some predefined graphs and the agents to collect the data
#It also has access to pandas and matplotlib.pyplot to generate new graphs based on the user query
f1_agent = CodeAgent(
    model=model,
    tools=[plot_fastest_lap_telemetry, plot_race_pace, analize_graph, plot_race_strategy],
    managed_agents=[data_agent, retriever_agent, web_search_agent],
    additional_authorized_imports=[
        "pandas",
        "numpy",
        "matplotlib.pyplot"
    ],
    step_callbacks=[update_memory],
    verbosity_level=2,
    max_steps=10,
)

#Modify the system prompt to add some rules to control the workflow
f1_agent.prompt_templates["system_prompt"] = f1_agent.prompt_templates["system_prompt"] + ("\n DO NOT answer questions unrelated to Formula 1. \n"
                                                                                           "You can answer questions related to circuit information \n"
                                                                                           "You can answer questions related to drivers' personal information \n"
                                                                                           "You can answer questions related to F1 data, and perform detailed analysis\n"
                                                                                           "You can answer any question that is F1-related \n "
                                                                                           "If you are given a task not related to Formula 1, use the final_answer "
                                                                                           "tool to answer kindly that you cannot do it. \n"
                                                                                           "If you are given a greeting, use the final_answer tool to answer kindly. \n"
                                                                                           "If you need additional information from the user to accurately perform"
                                                                                           "the tasks (driver name, year or name of the grand prix), use the"
                                                                                           "final_answer tool to ask about that information. \n"
                                                                                           "If you try to access a parquet file, ALWAYS check the columns first using the"
                                                                                           "columns attribute to understand its structure \n"
                                                                                           "If you need to generate a plot, save it with plt.savefig() in the same "
                                                                                           "directory you are being executed, DO NOT save it in subdirectories")
#Launch gradio interface with the main agent
GradioUI(f1_agent).launch()

#Remove parquet files generated during the communication process
parquet_files = glob.glob("*.parquet")

for file in parquet_files:
    os.remove(file)

