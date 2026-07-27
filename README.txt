TotalAI operates on a multi-role, stateful architecture. As a document moves from an employee upload to a manager's desk, the system tracks state and role-based permissions:

[Employee: Upload & Mask]
          │ (PII Masking & PDF Extraction)
          ▼
[Back-Office: Extraction Review]
          │ (Structured JSON Parsing & Confidence Scoring)
          ▼
[Compliance: Audit & RAG]
          │ (Vectorized Policy Retrieval with Page Citations)
          ▼
[Manager: Chat & Approval]
          │ (Consensus-based LLM Auditing & Conversational Summary)
          ▼
[Final Decision: Approved/Rejected]

Enterprise Features
1. Multi-Role Orchestration (RBAC)
The application provides distinct interfaces for different business roles:

Employee: Secure upload portal with automatic PII masking (redacting credit cards, emails, and phone numbers).

Back-Office: Data verification dashboard highlighting "Low-Confidence" OCR results for manual review.

Compliance: Dynamic RAG engine that allows users to upload custom Policy PDFs; the AI audits line items and cites exact Page Numbers from the policy document.

Manager: Approval decision board featuring an integrated chatbot that answers questions about the audit report.

2. Dual-LLM Judge System
To ensure enterprise-grade reliability, TotalAI implements a Consensus Judge. After the compliance audit is performed, two separate LLM instances (standard vs. creative) evaluate the audit independently. The system then calculates a final confidence score and consensus decision before showing it to the Manager.

3. Smart Document Pipeline
PII Masking: Sensitive information is scrubbed locally before hitting the LLM API.

Dynamic RAG: Vector store is built dynamically from the policy PDF uploaded by the Compliance role, ensuring the AI is always current with the latest rules.

Orchestration Layer: The backend graph architecture calls agents only when needed, minimizing latency and API costs.

🛠️ Tech Stack
Orchestration: LangGraph & LangChain.

LLM Engine: TCS GenAI Lab MaaS (DeepSeek V3 / GPT-4o).

Vector Database: ChromaDB (In-Memory).

Frontend: Streamlit with custom Pink/Purple branding.

PII Protection: Regex-based masking and Pydantic schema validation.