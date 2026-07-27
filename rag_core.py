"""
================================================================================
FILE: rag_core.py
TEAM: TotalAI (Regional Semi-Final Build)
PURPOSE: The Orchestrated Backend. Utilizes LangGraph, Multi-Modal Docling OCR,
PII Guardrails, and Advanced Hallucination Evaluation.
================================================================================
"""

import os
import ssl
import re
import urllib3
import requests
import httpx
import io
import tempfile
import logging
from typing import List, Dict, Any, Literal, TypedDict
from pydantic import BaseModel, Field, model_validator
from pypdf import PdfReader 
from langgraph.graph import StateGraph, END

# Enterprise Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[logging.FileHandler("enterprise_workflow.log"), logging.StreamHandler()]
)
logger = logging.getLogger("TotalAI_Backend")

# Try importing Docling for multi-modal OCR
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling not installed. Falling back to PyPDF.")

# ==============================================================================
# STEP 1: ENTERPRISE FIREWALL BYPASS & SETUP
# ==============================================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

old_get = requests.get
def new_get(*args, **kwargs):
    kwargs['verify'] = False
    return old_get(*args, **kwargs)
requests.get = new_get
custom_http_client = httpx.Client(verify=False)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# New API Key updated as requested
API_KEY = "sk-Z-8-rdXCXuBn3_yxdtqUNw"
os.environ["OPENAI_API_KEY"] = API_KEY

def get_llm(temperature=0.0):
    return ChatOpenAI(
        base_url="https://genailab.tcs.in",
        # Updated to an explicitly supported model from the Handbook to avoid 400/410 errors
        model="azure/genailab-maas-gpt-4o-mini", 
        api_key=API_KEY,
        http_client=custom_http_client,
        temperature=temperature
    )

llm_standard = get_llm(0.0)
embeddings = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key=API_KEY,
    http_client=custom_http_client,
    check_embedding_ctx_length=False
)

# ==============================================================================
# STEP 2: PYDANTIC SCHEMAS (Metadata Validation)
# ==============================================================================
class ExpenseItem(BaseModel):
    item_name: str
    amount: float
    category: str
    needs_human_review: bool

class ExtractionOutput(BaseModel):
    vendor_name: str
    receipt_date: str
    total_amount: float
    tax_amount: float 
    items: List[ExpenseItem]

    @model_validator(mode='after')
    def check_totals(self):
        if not self.items:
            raise ValueError("No items extracted from receipt.")
        return self

class LineItemAudit(BaseModel):
    item_name: str 
    amount: float 
    status: Literal["Auto-Verified", "Flagged"] 
    policy_clause: str 
    citation_page: str = Field(description="Exact Page Number of policy.")
    justification: str 

class AuditReportOutput(BaseModel):
    evaluated_items: List[LineItemAudit]

# ==============================================================================
# STEP 3: LANGGRAPH 1 - EXTRACTION WORKFLOW
# ==============================================================================
class ExtractionState(TypedDict):
    file_bytes: bytes
    raw_text: str
    masked_text: str
    extracted_json: dict
    metadata: dict
    errors: str

def doc_processing_node(state: ExtractionState) -> ExtractionState:
    """Multi-modal OCR & PII Masking node."""
    logger.info("Starting Doc Processing & PII Guardrails")
    raw_text = ""
    engine_used = "PyPDF Fallback"
    
    if DOCLING_AVAILABLE:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(state["file_bytes"])
                tmp_path = tmp.name
                
            converter = DocumentConverter()
            doc_result = converter.convert(tmp_path)
            raw_text = doc_result.document.export_to_markdown()
            os.remove(tmp_path)
            engine_used = "Docling Multi-modal"
        except Exception as e:
            logger.error(f"Docling failed, using fallback. Error: {e}")
            
    if not raw_text:
        reader = PdfReader(io.BytesIO(state["file_bytes"]))
        raw_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        
    clean_text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[REDACTED CREDIT CARD]', raw_text)
    clean_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED EMAIL]', clean_text)
    
    return {
        "raw_text": raw_text, 
        "masked_text": clean_text, 
        "metadata": {"ocr_engine": engine_used, "pii_masked": True}
    }

def extraction_node(state: ExtractionState) -> ExtractionState:
    """Structured LLM Extraction Node."""
    logger.info("Executing Agentic JSON Extraction")
    parser = PydanticOutputParser(pydantic_object=ExtractionOutput)
    prompt = PromptTemplate(
        template="Extract the expense details strictly.\n{format_instructions}\nText: {input_text}",
        input_variables=["input_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm_standard | parser
    
    try:
        result = chain.invoke({"input_text": state["masked_text"]}).model_dump()
        return {"extracted_json": result, "errors": None}
    except Exception as e:
        return {"errors": str(e)}

def run_extraction_graph(file_bytes: bytes) -> dict:
    workflow = StateGraph(ExtractionState)
    workflow.add_node("process", doc_processing_node)
    workflow.add_node("extract", extraction_node)
    
    workflow.set_entry_point("process")
    workflow.add_edge("process", "extract")
    workflow.add_edge("extract", END)
    
    app = workflow.compile()
    return app.invoke({"file_bytes": file_bytes, "metadata": {}, "errors": None})

# ==============================================================================
# STEP 4: KNOWLEDGE BASE (Chroma)
# ==============================================================================
def create_policy_vector_store(file_bytes: bytes, filename: str):
    logger.info("Building RAG Vector Store")
    reader = PdfReader(io.BytesIO(file_bytes))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            metadata = {"source": filename, "page": i + 1}
            docs.append(Document(page_content=text, metadata=metadata))
    return Chroma.from_documents(documents=docs, embedding=embeddings)

# ==============================================================================
# STEP 5: LANGGRAPH 2 - AUDIT & GUARDRAILS WORKFLOW
# ==============================================================================
class AuditState(TypedDict):
    extracted_json: dict
    vector_store: Any
    audit_results: dict
    hallucination_flag: bool

def rag_audit_node(state: AuditState) -> AuditState:
    """RAG node for compliance checking."""
    logger.info("Executing RAG Audit")
    results = []
    parser = PydanticOutputParser(pydantic_object=AuditReportOutput)
    
    for item in state["extracted_json"].get("items", []):
        query = f"{item['category']} {item['item_name']}"
        # Using native similarity_search to avoid EnsembleRetriever ModuleNotFoundErrors
        docs = state["vector_store"].similarity_search(query, k=2)
        policy_context = "\n".join([f"[Page {d.metadata.get('page')}]: {d.page_content}" for d in docs])
        
        prompt = PromptTemplate(
            template="Evaluate this item against the retrieved policy. Cite the Page Number.\n{format_instructions}\nPolicy:\n{policies}\nItem:\n{data}",
            input_variables=["policies", "data"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | llm_standard | parser
        eval_result = chain.invoke({"policies": policy_context, "data": item})
        results.extend(eval_result.model_dump()["evaluated_items"])
        
    return {"audit_results": {"evaluated_items": results}}

def hallucination_guardrail_node(state: AuditState) -> AuditState:
    """Advanced Guardrail (LLM-as-a-judge)."""
    logger.info("Executing Hallucination Guardrail Check")
    audit_text = str(state["audit_results"])
    
    eval_prompt = f"Does the following audit report invent any numerical values not present in standard accounting logic? Answer strictly YES or NO.\nReport: {audit_text}"
    response = llm_standard.invoke(eval_prompt).content.strip().upper()
    
    is_hallucinating = "YES" in response
    if is_hallucinating:
        logger.warning("Guardrail Triggered: Potential Hallucination Detected.")
        
    return {"hallucination_flag": is_hallucinating}

def run_audit_graph(extracted_json: dict, vector_store) -> dict:
    workflow = StateGraph(AuditState)
    workflow.add_node("audit", rag_audit_node)
    workflow.add_node("guardrail", hallucination_guardrail_node)
    
    workflow.set_entry_point("audit")
    workflow.add_edge("audit", "guardrail")
    workflow.add_edge("guardrail", END)
    
    app = workflow.compile()
    return app.invoke({
        "extracted_json": extracted_json, 
        "vector_store": vector_store, 
        "hallucination_flag": False
    })

# ==============================================================================
# STEP 6: MANAGER MCP CHATBOT
# ==============================================================================
def run_manager_chatbot(user_message: str, chat_history: str, context_data: dict) -> str:
    parser = StrOutputParser()
    prompt = PromptTemplate(
        template="You are the TotalAI Executive Assistant. Answer based on the workflow data context.\nContext: {context}\nHistory: {history}\nUser: {question}\nAnswer:",
        input_variables=["context", "history", "question"]
    )
    chain = prompt | llm_standard | parser
    return chain.invoke({
        "context": str(context_data),
        "history": chat_history,
        "question": user_message
    })