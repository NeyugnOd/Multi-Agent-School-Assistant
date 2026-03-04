from typing import Optional, Any
from loguru import logger
from crewai import LLM
from crewai.flow.flow import Flow, start, listen, router, or_
from pydantic import BaseModel

from .events import RetrieveEvent, EvaluateEvent, WebSearchEvent, SynthesizeEvent
from src.tools.firecrawl_search_tool import FirecrawlSearchTool
from src.retrieval.retriever import Retriever
from src.generation.rag import RAG
from config.settings import settings

# Prompt templates for workflow steps
ROUTER_EVALUATION_TEMPLATE = (
    """Bạn là người đánh giá chất lượng câu trả lời của hệ thống RAG nội bộ trường học.

CÂU HỎI NGƯỜI DÙNG:
{query}

CÂU TRẢ LỜI:
{rag_response}

Hãy đánh giá xem câu trả lời có thực sự dựa trên tài liệu nội bộ và trả lời đầy đủ câu hỏi hay không.

Chỉ trả lời:
- "GOOD" nếu câu trả lời đầy đủ và phù hợp
- "BAD" nếu câu trả lời thiếu thông tin, không rõ ràng, hoặc không dựa trên tài liệu

QUAN TRỌNG: Chỉ trả lời DUY NHẤT một từ viết hoa: GOOD hoặc BAD.
ĐÁNH GIÁ:"""
)

QUERY_OPTIMIZATION_TEMPLATE = (
    """Tối ưu hóa câu hỏi sau để tìm kiếm thông tin chính xác và đáng tin cậy trên Internet.

Câu hỏi gốc: {query}

Yêu cầu:
- Làm rõ chủ đề giáo dục hoặc hành chính nếu có
- Thêm từ khóa liên quan đến trường học, quy định, thông tư nếu phù hợp
- Giữ câu ngắn gọn nhưng đủ thông tin

Câu hỏi tối ưu:"""
)

SYNTHESIS_TEMPLATE = (
    """Bạn là trợ lý AI hỗ trợ tra cứu tài liệu nội bộ của trường học.

CÂU HỎI:
{query}

TRẢ LỜI TỪ TÀI LIỆU NỘI BỘ:
{rag_response}

KẾT QUẢ TÌM KIẾM BÊN NGOÀI (nếu có):
{web_results}

YÊU CẦU:
- Tổng hợp thông tin một cách rõ ràng và mạch lạc
- Ưu tiên thông tin từ tài liệu nội bộ
- Nếu có mâu thuẫn giữa tài liệu nội bộ và web, hãy nêu rõ
- Nếu web không có dữ liệu, cải thiện câu trả lời RAG
- Luôn trả lời bằng tiếng Việt

CÂU TRẢ LỜI TỔNG HỢP:"""
)

# Define flow state
class SchoolAssistantState(BaseModel):
    query: str = ""
    top_k: Optional[int] = 3

class SchoolAssistantWorkflow(Flow[SchoolAssistantState]):
    """School Assistant Workflow with router and web search fallback using CrewAI Flows."""

    def __init__(
        self,
        retriever: Retriever,
        rag_system: RAG,
        firecrawl_api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.rag = rag_system

        # Initialize Ollama LLM for workflow operations
        self.llm = LLM(
            model=f"ollama/{settings.llm_model}",
            base_url=settings.ollama_base_url,
            temperature=0.1
        )

    @start()
    def retrieve(self) -> RetrieveEvent:
        """Retrieve relevant documents from vector database"""
        query = self.state.query
        top_k = self.state.top_k

        if not query:
            raise ValueError("Query is required")

        logger.info(f"Retrieving documents for query: {query}")

        retrieved_nodes = self.retriever.search(query, top_k=top_k)
        logger.info(f"Retrieved {len(retrieved_nodes)} documents")
        return RetrieveEvent(retrieved_nodes=retrieved_nodes, query=query)

    @listen(retrieve)
    def generate_rag_response(self, ev: RetrieveEvent) -> EvaluateEvent:
        """Generate initial RAG response"""
        query = ev.query
        retrieved_nodes = ev.retrieved_nodes

        logger.info("Generating RAG response")

        rag_response = self.rag.query(query)

        logger.info("RAG response generated")
        return EvaluateEvent(
            rag_response=rag_response,
            retrieved_nodes=retrieved_nodes,
            query=query
        )

    @router(generate_rag_response)
    def evaluate_response(self, ev: EvaluateEvent) -> str:
        """Evaluate RAG response quality and route accordingly"""
        rag_response = ev.rag_response
        query = ev.query
        
        logger.info("Evaluating RAG response quality")

        evaluation_prompt = ROUTER_EVALUATION_TEMPLATE.format(query=query, rag_response=rag_response)
        resp_text = self.llm.call(evaluation_prompt)
        evaluation = (resp_text or "").strip().upper().split()[0]

        logger.info(f"Evaluation result: {evaluation}")
        return "synthesize" if "GOOD" in evaluation else "web_search"

    @listen("web_search")
    def perform_web_search(self, ev: EvaluateEvent | WebSearchEvent) -> SynthesizeEvent:
        """Perform web search if insufficient information from RAG response"""
        query = ev.query
        rag_response = ev.rag_response
        retrieved_nodes = getattr(ev, "retrieved_nodes", [])
        
        logger.info("Performing web search")
        
        search_results = ""
        try:
            optimization_prompt = QUERY_OPTIMIZATION_TEMPLATE.format(query=query)
            optimized_query = (self.llm.call(optimization_prompt) or query).strip()
            search_results = FirecrawlSearchTool().run(query=optimized_query, limit=3)
            logger.info("Web search completed via custom tool")
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            search_results = "Web search unavailable due to technical issues."
        
        return SynthesizeEvent(
            rag_response=rag_response,
            web_search_results=search_results,
            retrieved_nodes=retrieved_nodes,
            query=query,
            use_web_results=True
        )

    @listen(or_("synthesize", "perform_web_search"))
    def synthesize_response(self, ev: EvaluateEvent | SynthesizeEvent) -> dict:
        """Synthesize final response from RAG and web search results"""
        rag_response = ev.rag_response
        web_results = getattr(ev, "web_search_results", "") or ""
        query = ev.query
        use_web_results = getattr(ev, "use_web_results", False)
        
        logger.info("Synthesizing final response")
        
        if use_web_results and web_results:
            synthesis_prompt = SYNTHESIS_TEMPLATE.format(
                query=query, rag_response=rag_response, web_results=web_results
            )
            synthesized_answer = self.llm.call(synthesis_prompt)
            result = {
                "answer": synthesized_answer,
                "rag_response": rag_response,
                "web_search_used": True,
                "web_results": web_results,
                "query": query,
            }
        else:
            refinement_prompt = (
                f"Hãy cải thiện và làm rõ câu trả lời sau để nó đầy đủ và dễ hiểu hơn, "
                f"nhưng KHÔNG được thêm thông tin ngoài tài liệu:\n\n"
                f"Câu trả lời gốc: {rag_response}\n\n"
                f"Câu trả lời cải thiện (bằng tiếng Việt):"
            )
            refined = self.llm.call(refinement_prompt)
            result = {
                "answer": refined,
                "rag_response": rag_response,
                "web_search_used": False,
                "web_results": None,
                "query": query,
            }
        
        logger.info("Final response synthesized")
        return result

    async def run_workflow(self, query: str, top_k: Optional[int] = None) -> dict:
        """
        Run the complete flow for a given query.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with final answer and metadata
        """
        try:
            # Kick off the CrewAI flow asynchronously with runtime inputs
            result = await self.kickoff_async(inputs={"query": query, "top_k": top_k})
            return result if isinstance(result, dict) else {"answer": str(result), "query": query}
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "answer": f"Xin lỗi, hệ thống gặp lỗi khi xử lý câu hỏi của bạn: {str(e)}",
                "rag_response": None,
                "web_search_used": False,
                "web_results": None,
                "query": query,
                "error": str(e)
            }