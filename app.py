"""
================================================================================
FILE: app.py
TEAM: TotalAI (Regional Semi-Final Build)
PURPOSE: The Orchestrated Multi-Role UI. Features Light/Blue Theme, PDF Preview, 
PII Masking, LangGraph execution, and dynamic Manager Chatbot.
================================================================================
"""

import streamlit as st
import base64
import logging
from rag_core import (
    run_extraction_graph,
    create_policy_vector_store,
    run_audit_graph,
    run_manager_chatbot
)

# ==============================================================================
# STEP 1: UI STYLING & BROWSER CONFIG
# ==============================================================================
st.set_page_config(page_title="TotalAI Enterprise Auditor", layout="wide", page_icon="⚡")

# Light Grey Base & Electric Blue Highlight
custom_css = """
<style>
    .stApp { background-color: #F5F7FA; }
    [data-testid="stSidebar"] { background-color: #E8F0FE; border-right: 1px solid #D2E3FC; } 
    h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trblm { color: #1E88E5 !important; font-weight: 700; } 
    .stButton>button { background-color: #1E88E5; color: white; border-radius: 6px; border: none; font-weight: bold; transition: 0.3s; }
    .stButton>button:hover { background-color: #1565C0; color: white; box-shadow: 0 4px 8px rgba(30,136,229,0.2); }
    [data-testid="stMetricValue"] { color: #0D47A1 !important; } 
    .streamlit-expanderHeader { color: #1E88E5 !important; font-weight: bold; background-color: #FFFFFF; border-radius: 4px; }
    hr { border-color: #D2E3FC; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==============================================================================
# STEP 2: INITIALIZE SESSION STATE (State Machine Memory)
# ==============================================================================
if 'raw_pdf_bytes' not in st.session_state: st.session_state['raw_pdf_bytes'] = None
if 'extracted_state' not in st.session_state: st.session_state['extracted_state'] = None
if 'vector_store' not in st.session_state: st.session_state['vector_store'] = None
if 'audit_state' not in st.session_state: st.session_state['audit_state'] = None
if 'chat_messages' not in st.session_state: st.session_state['chat_messages'] = []
if 'manager_decision' not in st.session_state: st.session_state['manager_decision'] = "Pending"

def display_pdf(file_bytes):
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf" style="border: 1px solid #D2E3FC; border-radius: 8px;"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.title("⚡ TotalAI Agentic Orchestrator")
st.markdown("*Powered by LangGraph, Docling OCR & Enterprise Guardrails*")

# ==============================================================================
# STEP 3: ROLE SELECTOR (RBAC)
# ==============================================================================
st.sidebar.header("🛡️ Access Control")
current_role = st.sidebar.radio(
    "Login As:",
    ("1. Employee (Upload)", 
     "2. Back-Office (Agentic Extract)", 
     "3. Compliance (Hybrid RAG Audit)", 
     "4. Manager (Approval & Chat)")
)

# ------------------------------------------------------------------------------
# ROLE 1: EMPLOYEE (Upload & Preview)
# ------------------------------------------------------------------------------
if current_role == "1. Employee (Upload)":
    st.header("1️⃣ Employee Portal")
    st.write("Upload your receipt. TotalAI Multi-modal OCR will process and redact PII instantly.")
    
    uploaded_file = st.file_uploader("Upload PDF Receipt", type=["pdf"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        st.session_state['raw_pdf_bytes'] = file_bytes
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Document Preview")
            display_pdf(file_bytes) 
            
        with col2:
            st.success("✅ Document uploaded securely.")
            st.info("The Back-Office automation agent will now take over extraction and PII masking. You may log out.")

# ------------------------------------------------------------------------------
# ROLE 2: BACK-OFFICE (Run LangGraph Extractor)
# ------------------------------------------------------------------------------
elif current_role == "2. Back-Office (Agentic Extract)":
    st.header("2️⃣ Back-Office Automation")
    
    if not st.session_state['raw_pdf_bytes']:
        st.warning("No receipt submitted in the pipeline.")
    else:
        if st.button("▶️ Execute LangGraph Extraction Pipeline", type="primary"):
            with st.spinner("Initializing Agentic Graph (Docling OCR ➔ PII Guardrails ➔ LLM Extract)..."):
                result_state = run_extraction_graph(st.session_state['raw_pdf_bytes'])
                st.session_state['extracted_state'] = result_state
                
        if st.session_state['extracted_state']:
            state = st.session_state['extracted_state']
            
            if state.get("errors"):
                st.error(f"🚨 **PIPELINE HALTED:**\n\n{state['errors']}", icon="❌")
                st.stop()
                
            st.success("✅ Extraction Workflow Completed!")
            
            with st.expander("🔍 View Workflow Metadata & Logs"):
                st.json(state.get("metadata", {}))
                st.text_area("Masked Text Preview (PII Removed)", state.get("masked_text", ""), height=150)
                
            data = state.get("extracted_json", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Vendor", data.get("vendor_name", "Unknown"))
            col2.metric("Total", f"${data.get('total_amount', 0.0)}")
            col3.metric("Tax", f"${data.get('tax_amount', 0.0)}")
            
            st.markdown("### 📋 Structured Line Items")
            for item in data.get("items", []):
                st.write(f"- **{item.get('item_name')}**: ${item.get('amount')} *(Category: {item.get('category')})*")

# ------------------------------------------------------------------------------
# ROLE 3: COMPLIANCE (Hybrid RAG & Guardrails)
# ------------------------------------------------------------------------------
elif current_role == "3. Compliance (Hybrid RAG Audit)":
    st.header("3️⃣ Compliance Audit & Guardrails")
    
    st.subheader("A. Update Policy Database")
    policy_file = st.file_uploader("Upload Corporate Policy (PDF)", type=["pdf"])
    if st.button("Build ChromaDB Knowledge Base"):
        if policy_file:
            with st.spinner("Chunking Policy & Embedding..."):
                st.session_state['vector_store'] = create_policy_vector_store(policy_file.read(), policy_file.name)
                st.success("✅ Knowledge Base Configured!")
        else:
            st.error("Please upload a policy document.")

    st.markdown("---")
    st.subheader("B. Execute Audit Graph")
    
    if st.button("▶️ Run Agentic Audit & Guardrails", type="primary"):
        if not st.session_state['extracted_state'] or not st.session_state['vector_store']:
            st.error("Extraction state or Vector Store is missing.")
        else:
            with st.spinner("Executing: RAG Audit ➔ Hallucination Checker..."):
                st.session_state['audit_state'] = run_audit_graph(
                    st.session_state['extracted_state']['extracted_json'], 
                    st.session_state['vector_store']
                )
                st.success("✅ Audit Graph Complete!")

    if st.session_state['audit_state']:
        astate = st.session_state['audit_state']
        
        st.markdown("### 🔍 Guardrails: Hallucination Check")
        if astate.get("hallucination_flag"):
            st.error("⚠️ **Guardrail Alert:** Potential hallucination detected in the audit reasoning.")
        else:
            st.success("✅ Output passed hallucination checks.")
            
        st.markdown("### 📋 RAG Audit Findings")
        for item in astate.get("audit_results", {}).get("evaluated_items", []):
            color = "🟢" if item.get("status") == "Auto-Verified" else "🔴"
            with st.expander(f"{color} {item.get('item_name')} — ${item.get('amount')}"):
                st.write(f"**Status:** {item.get('status')}")
                st.write(f"**Policy Used:** {item.get('policy_clause')}")
                st.write(f"**Source Page:** 📄 {item.get('citation_page')}")
                st.write(f"**Justification:** {item.get('justification')}")

# ------------------------------------------------------------------------------
# ROLE 4: MANAGER (Approval & Chat)
# ------------------------------------------------------------------------------
elif current_role == "4. Manager (Approval & Chat)":
    st.header("4️⃣ Executive Dashboard")
    
    if not st.session_state['audit_state']:
        st.warning("Awaiting Compliance Audit completion.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("💬 AI Assistant (Model Context Protocol)")
            st.write("Query the data structure seamlessly.")
            
            for msg in st.session_state['chat_messages']:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            user_input = st.chat_input("E.g., Summarize the flagged items.")
            if user_input:
                st.chat_message("user").markdown(user_input)
                st.session_state['chat_messages'].append({"role": "user", "content": user_input})
                
                context_data = {
                    "extracted": st.session_state['extracted_state'].get('extracted_json'),
                    "audit": st.session_state['audit_state'].get('audit_results')
                }
                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state['chat_messages']])
                
                with st.spinner("Analyzing context..."):
                    ai_response = run_manager_chatbot(user_input, history_str, context_data)
                
                st.chat_message("assistant").markdown(ai_response)
                st.session_state['chat_messages'].append({"role": "assistant", "content": ai_response})

        with col2:
            st.subheader("📝 Authorization")
            status = st.session_state['manager_decision']
            st.markdown(f"**Status:** `{status}`")
            
            if st.button("✅ Approve Report", use_container_width=True):
                st.session_state['manager_decision'] = "Approved"
                st.rerun()
                
            if st.button("❌ Reject Report", use_container_width=True):
                st.session_state['manager_decision'] = "Rejected"
                st.rerun()
