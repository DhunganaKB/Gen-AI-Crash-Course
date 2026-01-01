# GenAI Crash Course (Beginner, Python) — **Syllabus**

> **Focus:** application building (not deep theory)  
> **Audience:** beginner developers who can write basic Python scripts  
> **Primary outcome:** students finish with a working LLM app they can extend (RAG + simple tools)

---

## Table of Contents

- [Course Snapshot](#course-snapshot)
- [Who This Is For](#who-this-is-for)
- [Prerequisites](#prerequisites)
- [Learning Outcomes](#learning-outcomes)
- [Required Accounts & Keys](#required-accounts--keys)
- [Recommended Repo Structure](#recommended-repo-structure)
- [Schedule (Recommended 2-Day Workshop)](#schedule-recommended-2-day-workshop)
- [Lesson-by-Lesson Breakdown](#lesson-by-lesson-breakdown)
- [Capstone Project](#capstone-project)
- [Assessment (Lightweight)](#assessment-lightweight)
- [High-Quality Public Resources](#high-quality-public-resources)
- [Optional Extensions](#optional-extensions)

---

## Course Snapshot

**Format (recommended):** 2-day workshop (12 hours total)  
**Alternative format:** 8 sessions × 90 minutes (same content, slower pace)  
**Teaching style:** short demos + long hands-on labs  
**Core stack:** Python, Jupyter, `.env`, API-based LLMs + one open-source Hugging Face model path

---

## Who This Is For

This crash course is for learners who:
- know basic Python (variables, functions, lists/dicts, installing packages)
- can run terminal commands and use GitHub
- want to build GenAI apps (chat, extraction, RAG, agents) **fast**

Not ideal if you want:
- deep transformer math
- training large models from scratch
- advanced MLOps/infra from day 1

---

## Prerequisites

### Skills
- Python basics (functions, loops, reading files)
- REST API basics (helpful but not required)
- Command line basics: `pip`, `python -m venv`

### Software
- Python **3.10+** (3.11 recommended)
- VS Code or similar editor
- Git
- (Optional) Docker

---

## Learning Outcomes

By the end, students will be able to:

1. **Call LLMs from Python** using environment variables and SDKs (OpenAI / Azure / Gemini / Anthropic + Hugging Face open models).
2. Write prompts that produce **consistent, testable** outputs.
3. Build a small app that uses:
   - **structured outputs** (JSON) and
   - one **tool/action** (e.g., calculator, search, file lookup).
4. Implement a basic **RAG pipeline**:
   - ingest docs → chunk → embed → retrieve → answer with citations.
5. Create a tiny **evaluation harness** to detect regressions.
6. Package the solution into a simple **FastAPI or Streamlit** app.

---

## Required Accounts & Keys

To reduce friction, this course supports **multiple providers**. Students only need **one** closed provider + Hugging Face.

### Minimum setup (recommended for beginners)
- **One** of: OpenAI **or** Google Gemini **or** Anthropic **or** Azure OpenAI
- Hugging Face account + token (`HF_TOKEN`) for open models (API-based)

### Full setup (optional, for comparison demos)
- OpenAI + Azure + Google Gemini + Anthropic + Hugging Face

**Key rule:** never hardcode keys in notebooks. Use environment variables or `.env`.

---

## Recommended Repo Structure

You can adapt this, but this structure works well for workshops:

```
genai-crash-course/
  README.md
  Syllabous.md
  .gitignore
  .env.example

  notebooks/
    01_accessing_llm_part2_api_keys.ipynb
    01_accessing_llm_part3A_openai_first_call.ipynb
    01_accessing_llm_part3B_azure_first_call.ipynb
    01_accessing_llm_part3C_google_gemini_first_call.ipynb
    01_accessing_llm_part3D_anthropic_first_call.ipynb
    01_accessing_llm_part3E_huggingface_open_models.ipynb
    01_accessing_llm_unified_notebook_all_providers_plus_huggingface.ipynb

  src/
    llm_router.py
    rag/
      ingest.py
      retrieve.py
      answer.py
    eval/
      run_eval.py

  data/
    sample_docs/
    sample_faq.csv

  slides/
    (optional)

  requirements.txt
```

### Suggested `.env.example`
```bash
# Closed providers
OPENAI_API_KEY=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_DEPLOYMENT=""
GOOGLE_API_KEY=""          # or GEMINI_API_KEY
ANTHROPIC_API_KEY=""

# Open models
HF_TOKEN=""

# Optional model overrides
OPENAI_MODEL=""
GEMINI_MODEL=""
ANTHROPIC_MODEL=""
HF_CHAT_MODEL=""
```

---

## Schedule (Recommended 2-Day Workshop)

> You can run this as 2 days or split into 8 sessions.

### Day 1 — Core building blocks
- **Block 1 (60–90m):** Setup + workflow (venv, notebooks, env vars, `.env`)
- **Block 2 (90m):** Lesson 1 — Accessing LLMs (closed + open)
- **Block 3 (90m):** Lesson 2 — Prompting patterns for reliable outputs
- **Block 4 (90m):** Lesson 3 — Structured output + simple “tools”
- **Block 5 (90m):** Lesson 4 — Embeddings + semantic search (pre-RAG)

### Day 2 — RAG + evaluation + shipping
- **Block 6 (120m):** Lesson 5 — RAG end-to-end (doc QA with citations)
- **Block 7 (90m):** Lesson 6 — Evaluation + testing harness
- **Block 8 (90m):** Lesson 7 — Agent workflows (careful, minimal)
- **Block 9 (60–90m):** Lesson 8 — Deploy (FastAPI/Streamlit) + capstone demo

---

## Lesson-by-Lesson Breakdown

Each lesson includes:
- **Outcomes**
- **Lecture topics**
- **Hands-on lab deliverable**

### Lesson 0 — Setup & workflow (optional)
**Outcomes**
- Students can run notebooks, install packages, and use `.env`

**Topics**
- `venv`, `pip`, Jupyter kernels
- `.env` + `python-dotenv`
- “keys never in code”

**Lab deliverable**
- `00_setup_check.ipynb` verifying Python + env vars

---

### Lesson 1 — Accessing LLMs (closed + open)
**Outcomes**
- Make a successful LLM call from Python
- Compare outputs across at least 2 providers

**Topics**
- Provider landscape: closed vs open
- Keys + endpoints + environment variables
- Hugging Face open models:
  - hosted API (HF Inference router)
  - local transformers (optional)

**Lab deliverables**
- Unified notebook that calls: OpenAI/Azure/Gemini/Claude + HF open model
- “Hello world” responses printed consistently

---

### Lesson 2 — Prompting for reliability (practice-heavy)
**Outcomes**
- Write prompts that produce predictable outputs

**Topics**
- Instructions + constraints + examples
- “Ask for JSON” vs “ask for bullets”
- Common failure modes + mitigation patterns

**Lab deliverable**
- Prompt pattern notebook:
  - summarize
  - extract fields
  - classify
  - rewrite
  - safety redaction

---

### Lesson 3 — Structured output + tools (function calling mindset)
**Outcomes**
- Model returns valid JSON reliably
- App executes a tool/action based on model output

**Topics**
- JSON schema mindset (validate + retry)
- Tool pattern: classify → route → execute

**Lab deliverable**
- Ticket triage mini-app:
  - input ticket
  - output JSON `{category, priority, summary}`
  - route to handler functions

---

### Lesson 4 — Embeddings + semantic search (foundation for RAG)
**Outcomes**
- Build a simple semantic search over a small dataset

**Topics**
- What embeddings are for (search, dedupe)
- Vector store basics (local-first)

**Lab deliverable**
- Semantic FAQ search:
  - embed FAQ
  - query → top-k results

---

### Lesson 5 — RAG end-to-end (Doc QA with citations)
**Outcomes**
- Build a working RAG app with citations

**Topics**
- Ingestion, chunking, embeddings, retrieval
- Grounded answers and citation formatting
- “RAG failure modes” (wrong chunk, missing context)

**Lab deliverable**
- “Chat with docs” (PDF/txt) QA:
  - returns answer + cites chunk IDs

---

### Lesson 6 — Evaluation + testing (keep it lightweight)
**Outcomes**
- Detect regressions after prompt/model changes

**Topics**
- Golden set of 20 questions
- Simple rubric + automated checks
- Logging prompts/responses safely

**Lab deliverable**
- `eval/run_eval.py` that runs your golden set and saves results

---

### Lesson 7 — Agents & workflows (minimal, practical)
**Outcomes**
- Build a multi-step workflow that uses tools safely

**Topics**
- When agents help vs hurt
- Tool loop boundaries (timeouts, budgets)
- “Planner vs executor” concept (high-level)

**Lab deliverable**
- Simple “research + answer” agent that:
  - retrieves context
  - optionally calls a calculator/tool
  - produces final answer + citations

---

### Lesson 8 — Shipping (FastAPI/Streamlit) + cost + security basics
**Outcomes**
- Turn notebook into a tiny app
- Manage keys + basic cost controls

**Topics**
- FastAPI/Streamlit skeleton
- Rate limiting, max tokens, caching idea
- Secret management basics

**Lab deliverable**
- A deployable API endpoint + minimal UI

---

## Capstone Project

**Project:** Internal Knowledge Assistant (RAG + tools)

**Must-have features**
- Ingest docs (txt/pdf)
- Create embeddings + vector search
- Answer questions with citations
- One simple tool action (e.g., “draft an email”, “create a ticket payload”)

**Stretch goals**
- Add evaluation harness and track improvements
- Add a provider switch (OpenAI vs HF open model)
- Add basic analytics (request logs with redaction)

---

## Assessment (Lightweight)

This is a crash course—assessment should be practical:
- ✅ Completion of labs (checkboxes)
- ✅ Capstone demo (3–5 minutes)
- ✅ Optional: golden-set eval report with before/after comparison

---

## High-Quality Public Resources

> These are **publicly accessible** resources. Some are **open-source with reuse-friendly licenses** (e.g., MIT).  
> When reusing content, always follow the license terms and attribution requirements.

### Best “course-like” open-source curricula (reusable)
- **Microsoft — Generative AI for Beginners (MIT)**  
  https://github.com/microsoft/generative-ai-for-beginners  
- **OpenAI — OpenAI Cookbook (MIT)**  
  https://github.com/openai/openai-cookbook  
  (Also browsable at https://cookbook.openai.com/)

### Best free-to-access courses & lecture series
- **Hugging Face — LLM Course (free)**  
  https://huggingface.co/learn/llm-course/en/chapter1/1  
- **Full Stack Deep Learning — LLM Bootcamp (recordings free)**  
  https://fullstackdeeplearning.com/llm-bootcamp/  
- **Stanford CS25 — Transformers United (free to audit + recordings)**  
  https://web.stanford.edu/class/cs25/  

### Structured “guided paths”
- **Kaggle — 5‑Day Gen AI Intensive (Google)**  
  https://www.kaggle.com/learn-guide/5-day-genai  

### Official docs you should link in Lesson 1 (API + open models)
- **Hugging Face Inference Providers (OpenAI-compatible router)**  
  https://huggingface.co/inference/get-started  
- **HF Inference provider docs**  
  https://huggingface.co/docs/inference-providers/en/providers/hf-inference  
- **Transformers Pipelines docs**  
  https://huggingface.co/docs/transformers/en/main_classes/pipelines  
- **LangChain basic agent quickstart**  
  https://docs.langchain.com/oss/python/langchain/quickstart  
- **LlamaIndex starter tutorial (adds RAG)**  
  https://developers.llamaindex.ai/python/framework/getting_started/starter_example/  

---

## Optional Extensions

If you run this course more than once, these are great add-ons:

- **Model routing:** “small model first, large model fallback”
- **Caching:** prompt/result caching + invalidation rules
- **Memory:** session memory (what to store, what not to store)
- **Multi-modal:** image input + extraction
- **Deployment:** containerize + deploy to a cloud service
- **Safety:** prompt injection defenses for RAG

---

*Last updated: 2025-12-31*
