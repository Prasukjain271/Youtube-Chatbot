from typing import TypedDict, Any, List, Literal, Annotated
from abc import ABC, abstractmethod
import yt_dlp
import os

# For splitting - UPDATED
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector stores - UPDATED
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Pinecone - Check if you need this
# import pinecone
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field
from typing import Literal, List, Optional

# LangGraph - "UPDATED"
from langgraph.graph import StateGraph, START, END

# Embeddings - UPDATED (try new package first)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Chat model - UPDATED
from langchain_google_genai import ChatGoogleGenerativeAI

# Environment
from dotenv import load_dotenv
load_dotenv()

# LangSmith - UPDATED
from langsmith import Client

# Core components - UPDATED
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Pydantic
from pydantic import BaseModel, Field
"""if os.getenv('LANGCHAIN_API_KEY') and os.getenv('LANGCHAIN_TRACING_V2') == 'true':
    try:
        from langsmith import Client
        client = Client()
        print(" LangSmith enabled")
    except Exception as e:
        print(f" LangSmith disabled: {e}")
        os.environ['LANGCHAIN_TRACING_V2'] = 'false'
else:
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'"""
gemini_api_key = os.getenv("GEMINI_API_KEY") 
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',google_api_key=gemini_api_key)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
parser=StrOutputParser()
class User(ABC):
    @abstractmethod
    def ask_query(self, query: str) -> str:
        pass
    @abstractmethod
    def langraph(self,initialstate):
        pass
class Standard_User(User):    
    def __init__(self, username: str):
        self.username = username
    def langraph(self,initial_state) :
        class state(TypedDict):
          
            response: str
            query: str  
            retrieve_videoinfo:list[dict]
            
          
        def retrieve_videoinfo(state: state) -> dict[str, Any]:
            api_key = os.getenv("PINECONE_API_KEY")
            pc = Pinecone(api_key=api_key)
            index_name = "youtube-database"
            index = pc.Index(index_name)   
            """from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            query_tfidf = vectorizer.transform([state['query']])  # Use transform, not fit_transform
        """
            
        
            namespace='FinalSummary'


        #  FIX 4: Get dense embedding for query
            query_dense = embeddings.embed_query(state['query'])

        #  FIX 5: Query with both vectors
            query_response = index.query(
            namespace=namespace,
            top_k=5,
            vector=query_dense,  # Dense vector
        
            include_values=False,
            include_metadata=True
            )
            extracted=[]
            #query result is a dictionary which as a key called match which has value as list of dictionary having all the vectors
            for match in query_response['matches']:
            # Safely get metadata with defaults
                metadata = match.get('metadata', {})   
                extracted.append({
                'id': match.get('id', 'unknown'),
                "channel_id": metadata.get('channel_id', ""),
                "channel_title": metadata.get('channel_title', ""),
                "comment_count": metadata.get('comment_count', ""),
                "duration_seconds": metadata.get('duration_seconds', ""),
                "like_count": metadata.get('like_count', ""),
                "published_at": metadata.get('published_at', ""),
                "title": metadata.get('title', ""),
                "topics": metadata.get('topics', ""),
                "url": metadata.get('url', ""),
                'category': metadata.get('category', 'Uncategorized'),
                'text': metadata.get('text', ''),
                })
            #extraced is a list of dictionaries 
            #returning it as a dict to update the state
            return {"retrieve_videoinfo": extracted}
        
        def generate_response(state: state) -> str:
            
            prompt = PromptTemplate(
            input_variables=["context", "user_query"],
            template= """
                        You are a knowledgeable podcast assistant with expertise in analyzing video content.
            
            You have access to video summaries and metadata. Your task is to:
            1. Analyze the user's query type
            2. Provide appropriate responses based on available information
            3. Be helpful even when exact information isn't in the context
            
            here the context and query 
            {context},{user_query}
            GUIDELINES:
            
            A. If user asks for a SUMMARY or OVERVIEW:
               - Provide the complete summary from context
               
            B. If user asks a SPECIFIC QUESTION (about people, topics, etc.):
               - First check if information is in the context
               - If yes: provide detailed answer with citations
               - If no:  mention it's not covered
               
            C. If user asks about METADATA (title, date, duration, etc.):
               - Extract and present metadata from context
               
            D. If user asks COMPARATIVE questions:
               - Compare information across available videos if multiple provided
               
            Always be helpful and provide context-relevant information even for questions not directly answered.
                        """, 
                        )
            chain=prompt|llm    
            response = chain.invoke({"context":state["retrieve_videoinfo"], "user_query":state["query"]})
            state.update( {"response": response})
            return state
        graph= StateGraph(state)

        graph.add_node("retrieve_videoinfo",retrieve_videoinfo)
    
        graph.add_node("generate_response",generate_response)
    

        graph.add_edge(START,"retrieve_videoinfo")
        graph.add_edge("retrieve_videoinfo","generate_response")
    
        graph.add_edge("generate_response",END)  

        workflow=graph.compile()        
        result = workflow.invoke(initial_state)
        return result
        
        
    def ask_query(self, query: str) -> str:
        initial_state = {
       
        "response": " ",
        "query": query,
        "retrieve_videoinfo":" ",
         
           
        }
       
        result = self.langraph(initial_state)
        return result["response"].content  

class Premium_User(User):
    def __init__(self, username: str):
        self.username = username
   
    def langraph(self,initial_state) :
        class state(TypedDict):
            retrieved_docs: list[dict]
            response: str
            query: str  
            retrieve_videoinfo:list[dict]
            retrieve_sectioninfo:list[dict]
            scope:str
            intent: str
            action:str 
            target_video_ids:str
            response:str
        class VideoQueryRouter(BaseModel):
            """
            Classifies a user query to determine if we can answer immediately from summaries
            or if we need to retrieve deeper section-level details.
            """
            
            scope: Literal["SingleVideo", "MultiVideo"] = Field(description="Does the user ask about a specific video or compare multiple videos?")
            intent: Literal["SummaryRequest", "SpecificQuestion"] = Field( description="Is the user asking for a general overview/gist (SummaryRequest) or a specific detail/fact (SpecificQuestion)?")
            
            action: Literal["answer", "section_summary_retrieve"] = Field(
                description=(
                    "The action to take. "
                    "'answer': If the provided context is sufficient or it's a simple summary request. "
                                'Choose this ONLY if the user explicitly asks for a Summary, Overview, or Gist.'
                    "'section_summary_retrieve': If the user asks for deep details  NOT in the summary. " 
                      
                )
            )
            
            target_video_ids: Optional[List[str]] = Field(
                default=None,
                description="If action is 'section_summary_retrieve', list the specific Video IDs (or Titles) to perform the deep dive on."
            )

            answer: Optional[str] = Field(description=(
            "The final output content. Follow these rules based on 'action':\n"
            "- IF action='answer' AND intent='SummaryRequest'then You must write a DETAILED, STRUCTURED summary based on the context here. "
                "format "
                " Use bullet points for key takeaways. "
                "Make it sound professional and comprehensive (approx 150-200 words).\n"
            "- IF action='answer' AND intent='MultiVideo': List the videos ,thier title and their topics clearly.\n"
        ))
        def retrieve_videoinfo(state: state) -> dict[str, Any]:
            api_key = os.getenv("PINECONE_API_KEY")
            pc = Pinecone(api_key=api_key)
            index_name = "youtube-database"
            index = pc.Index(index_name)   
            """from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            query_tfidf = vectorizer.transform([state['query']])  # Use transform, not fit_transform
        """
            
        
            namespace="FinalSummary"


        #  FIX 4: Get dense embedding for query
            query_dense = embeddings.embed_query(state['query'])

        #  FIX 5: Query with both vectors
            query_response = index.query(
            namespace=namespace,
            top_k=5,
            vector=query_dense,  # Dense vector
        
            include_values=False,
            include_metadata=True
            )
            extracted=[]
            #query result is a dictionary which as a key called match which has value as list of dictionary having all the vectors
            for match in query_response['matches']:
            # Safely get metadata with defaults
                metadata = match.get('metadata', {})   
                extracted.append({
                'id': match.get('id', 'unknown'),
                "channel_id": metadata.get('channel_id', ""),
                "channel_title": metadata.get('channel_title', ""),
                "comment_count": metadata.get('comment_count', ""),
                "duration_seconds": metadata.get('duration_seconds', ""),
                "like_count": metadata.get('like_count', ""),
                "published_at": metadata.get('published_at', ""),
                "title": metadata.get('title', ""),
                "topics": metadata.get('topics', ""),
                "url": metadata.get('url', ""),
                'category': metadata.get('category', 'Uncategorized'),
                'text': metadata.get('text', ''),
                })
            #extraced is a list of dictionaries 
            #returning it as a dict to update the state
            return {"retrieve_videoinfo": extracted}
        
        def generate_route(state: state) -> str:
            
            prompt = PromptTemplate(
            input_variables=["context", "user_query"],
            template= """You are the "Query Router and Answer Engine" for a Video RAG system. Your job is to analyze the User Query and the provided Retrieved Context (video summaries) to determine the best course of action.

                        # Inputs
                        1. **User Query**: {user_query}
                        2. **Retrieved Context**: {context}

                        # Classification & Execution Rules
                        Analyze the input to determine the scope and sufficiency. You must execute the decision logic in strict order.

                        # DECISION LOGIC (Execute in Strict Order)

                        **Step 1: Check for "Summary Request"**
                        - **Trigger:** Does the user explicitly ask for a "summary", "gist", "overview", or "abstract" of a specific video?
                        - **Action:** `answer`
                        - **Instructions:**
                            1. Identify the specific video title from the user query or context.
                            2. output the comprehensive summary that is  provided context just structure it in the given format.
                            3. Format:
                                - **Title:** [Video Name]
                                - **Key Takeaways:** Bullet points of main arguments.
                                - **Important Segments:** Notable stories or concepts mentioned.
                            4. **Constraint:** The summary must be at least 150 words. Do not shorten it; expand on the points provided.

                        **Step 2: Check for "Multi-Video Discovery"**
                        - **Trigger:** Does the user ask questions that require scanning *multiple* video titles/topics? (e.g., "Which videos talk about AI?", "Compare Video A and B").
                        - **Action:** `answer`
                        - **Instructions:**
                            1. Scan the metadata of all provided videos.
                            2. Answer the user's specific question based *only* on the titles and high-level topics in the context.
                            3. **Constraint:** Do not explain your thinking process. Output the direct answer.
                            4.action='answer'
                            
                        **Step 3: Check for "Specific Content Question" (The Trap)**
                        - **Trigger:** Does the user ask "How", "Why", "What is the code", "What specifically was said", or for a deep explanation of a concept inside a video?
                        - **Action:** `section_summary_retrieve`
                        - **CRITICAL RULE:** If the user asks for details (code snippets, specific implementation, minute details) that are NOT explicitly written in the high-level summary, you MUST fail over to retrieval.
                        - **Output:** Just the string: `section_summary_retrieve`

                        # Final Output Instruction
                        - If your decision is `answer`: Output the natural language response following the formatting rules above.
                        - If your decision is `section_summary_retrieve`: Output ONLY the text `section_summary_retrieve`. 
                        """)
            llm_withschema=llm.with_structured_output(VideoQueryRouter)
            chain=prompt|llm_withschema  
            responsee = chain.invoke({"context":state["retrieve_videoinfo"], "user_query":state["query"]})
            state.update( { "scope":responsee.scope,
                "intent": responsee.intent,
                "action":responsee.action, 
                "target_video_ids": responsee.target_video_ids,
                "response":responsee.answer
            })
            return state
        def retrieve_sectioninfo(state: state) -> dict[str, Any]:
            api_key = os.getenv("PINECONE_API_KEY")
            pc = Pinecone(api_key=api_key)
            index_name = "youtube-database"
            index = pc.Index(index_name)   
            """from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            query_tfidf = vectorizer.transform([state['query']])  # Use transform, not fit_transform
        """
            
        
            namespace='SectionSummary'


        #  FIX 4: Get dense embedding for query
            query_dense = embeddings.embed_query(state['query'])

        #  FIX 5: Query with both vectors
            query_response = index.query(
            namespace=namespace,
            top_k=14,
            vector=query_dense,  # Dense vector
        
            include_values=False,
            include_metadata=True
            )
            extracted=[]
            #query result is a dictionary which as a key called match which has value as list of dictionary having all the vectors
            for match in query_response['matches']:
            # Safely get metadata with defaults
                metadata = match.get('metadata', {})   
                extracted.append({
                'id': match.get('id', 'unknown'),
                "channel_id": metadata.get('channel_id', ""),
                "channel_title": metadata.get('channel_title', ""),
                "comment_count": metadata.get('comment_count', ""),
                "duration_seconds": metadata.get('duration_seconds', ""),
                "like_count": metadata.get('like_count', ""),
                "published_at": metadata.get('published_at', ""),
                "title": metadata.get('title', ""),
                "topics": metadata.get('topics', ""),
                "url": metadata.get('url', ""),
                'category': metadata.get('category', 'Uncategorized'),
                'text': metadata.get('text', ''),
                })
            #extraced is a list of dictionaries 
            #returning it as a dict to update the state
            return {"retrieve_sectioninfo": extracted}
        def actual_router(state:state)->state:
            if(state["action"]=="answer"):
                return END
            else:
                return "retrieve_sectioninfo"
                
        def generate_response(state: state) -> str:
            
            prompt = PromptTemplate(
            input_variables=["section_context", "user_query"],
            template="""# Role
                        You are a precision-focused Video Analyst. Your goal is to answer the User Query using *only* the provided "Section Summaries".

                        # Input Data
                        1. **User Query**: The specific question the user asked.
                        2. **Section Summaries**: Detailed segments of the video transcript/summary that were retrieved because they are relevant to the query.

                        # Instructions
                        1. **Synthesize**: Read through the provided sections and synthesize a coherent answer to the user's question.
                        2. **Be Specific**: Since these are detailed sections, your answer should include specific details, arguments, or steps mentioned in the text.
                        3.DO NOT include timestamp references like [00:19] or [05:30] in your response
                        # Critical Constraints (The "Grounding" Rules)
                        - **Strict Adherence**: You are forbidden from using outside knowledge. If the answer is not contained within the provided "Section Summaries", you must strictly state: "I cannot find sufficient information in the retrieved video sections to answer this question."
                        - **No Guessing**: Do not attempt to guess or infer details that are not explicitly stated.
                        - **Ambiguity**: If the provided sections talk about the topic but don't answer the specific question (e.g., user asks for "code" but sections only show "theory"), explicitly state what IS missing.
                        
                        **Scenario : The Answer is Missing (Soft Fail)**
                        If the "Section Summaries" do NOT contain the specific answer to the query:
                        1.  **State Limitation**: Start by politely stating: "The retrieved sections do not contain the specific details regarding [User's Query]."
                        2.  **Context Summary**: Immediately follow up with a **3-4 line human-friendly summary** of what these sections *actually* discussed.
                            * *Goal*: Tell the user: "I couldn't find X, but the video was talking about Y and Z here."
                        
                         # Output Format
                        **Answer:**
                        [Your detailed answer here]

                        **Reference Segments:**                
                            *Sources*: List the sources, but YOU MUST rewrite them. Do not copy-paste long blocks of text.
                                    Format: Bullet points.

                        ---
                        # Section Summaries
                        {section_context}

                        # User Query
                        {user_query}, 
                        """
                        )
            chain=prompt|llm    
            response = chain.invoke({"section_context":state["retrieve_sectioninfo"], "user_query":state["query"]})
            state.update( {"response": response.content})
            return state
        
            
        graph= StateGraph(state)

        graph.add_node("retrieve_videoinfo",retrieve_videoinfo)
        graph.add_node("generate_route",generate_route)
        graph.add_node("generate_response",generate_response)
        graph.add_node("retrieve_sectioninfo",retrieve_sectioninfo)

        graph.add_edge(START,"retrieve_videoinfo")
        graph.add_edge("retrieve_videoinfo","generate_route")
        graph.add_conditional_edges("generate_route",actual_router)
        graph.add_edge("retrieve_sectioninfo","generate_response")
        graph.add_edge("generate_response",END)

        workflow=graph.compile()        
        result = workflow.invoke(initial_state)
        return result
        
        
    def ask_query(self, query: str) -> str:
        initial_state = {
       
        "response": " ",
        "query": query,
        "retrieve_videoinfo":" ", 
        "scope":" ",
        "intent": " ",
        "action":" " ,
        "target_video_ids":" ",
        "retrieve_sectioninfo":" "
        }
       
        result = self.langraph(initial_state)
        return result["response"]
        




