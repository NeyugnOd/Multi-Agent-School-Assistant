from typing import Optional
from loguru import logger
from crewai import LLM
from pydantic import BaseModel
from src.retrieval.retriever import Retriever
from config.settings import settings

class ChatMessage(BaseModel):
    role: str
    content: str

class RAG: 
    def __init__(
        self, 
        retriever: Retriever, 
        llm_model: str = None,
        ollama_base_url: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        self.retriever = retriever
        self.llm_model = llm_model or settings.llm_model
        self.ollama_base_url = ollama_base_url or settings.ollama_base_url
        self.temperature = temperature or settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens
        
        # Initialize LLM
        self.llm = self._setup_llm()
        
        # System message
        self.system_message = ChatMessage(
            role="system",
            content=(
                "Bạn là trợ lý AI hỗ trợ tra cứu tài liệu nội bộ của trường học. "
                "Bạn chỉ được phép trả lời dựa trên thông tin được cung cấp trong phần CONTEXT. "
                "Nếu thông tin không có trong CONTEXT, hãy trả lời rằng bạn không có đủ thông tin "
                "và giải thích rõ thiếu thông tin gì. "
                "Luôn trả lời bằng tiếng Việt."
            )
        )
        
        # RAG prompt template
        self.prompt_template = (
            "CONTEXT:\n"
            "{context}\n"
            "---------------------\n"
            "Dựa trên thông tin trong CONTEXT ở trên, hãy trả lời câu hỏi sau. "
            "Nếu CONTEXT không chứa đủ thông tin để trả lời, "
            "hãy nói rõ rằng bạn không biết và nêu rõ thông tin còn thiếu.\n\n"
            "CÂU HỎI: {query}\n"
            "TRẢ LỜI (bằng tiếng Việt): "
        )

    def _setup_llm(self):
        llm = LLM(
            model=f"ollama/{self.llm_model}",
            base_url=self.ollama_base_url,
            temperature=self.temperature
        )
        logger.info(f"Initialized LLM with Ollama model: {self.llm_model}")
        return llm

    def generate_context(self, query: str, top_k: Optional[int] = None):
        # Generate context from retrieval results
        return self.retriever.get_combined_context(query, top_k)

    def query(self, query: str, top_k: Optional[int] = None):
        # Generate context from retrieval
        context = self.generate_context(query, top_k)
        
        # Create prompt from template
        prompt = self.prompt_template.format(context=context, query=query)
        return self.llm.call(f"{self.system_message.content}\n\n{prompt}")

    def get_detailed_response(self, query: str, top_k: Optional[int] = None):
        # Get retrieval results with scores
        retrieval_results = self.retriever.search_with_scores(query, top_k)
        
        # Generate context
        context = self.retriever.get_combined_context(query, top_k)
        
        # Generate response
        response = self.query(query, top_k=top_k)
        
        return {
            "response": response,
            "context": context,
            "sources": retrieval_results,
            "query": query,
            "model": self.llm_model
        }

    def set_prompt_template(self, template: str):
        # Set custom prompt template
        self.prompt_template = template
        logger.info("Updated prompt template")

    def set_system_message(self, content: str):
        # Set custom system message
        self.system_message = ChatMessage(role="system", content=content)
        logger.info("Updated system message")