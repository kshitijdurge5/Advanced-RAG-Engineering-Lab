# ADVANCED RAG ENGINEERING LAB

## Learning sequence

1. Phase 1: Original Data Ingestion and Production RAG projects.
2. Phase 2: Advanced LLM and RAG Engineering.
3. Phase 3: Agentic AI Engineering.
4. Phase 4: Production AI Engineering.
5. Phase 5: Agentic RAG Capstone and Interview Preparation.

## How to run

Activate your existing virtual environment.

Run each phase from the project root:

```powershell
python phase_2_advanced_llm_rag.py
python phase_3_agentic_ai.py
python phase_4_production_ai.py
python phase_5_capstone_interview.py
```

Run capstone tests:

```powershell
python phase_5_capstone_interview.py --test
```

Run optional FastAPI example:

```powershell
pip install fastapi uvicorn
python phase_4_production_ai.py --api
```

Open API documentation at:

http://127.0.0.1:8000/docs

## Important

The examples are educational implementations.
Syntax and demonstration success do not automatically prove
production readiness.

The archived original phases are retained in _archive.