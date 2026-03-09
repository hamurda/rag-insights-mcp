#!/usr/bin/env python3

import json
from contextlib import asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from config import Config
from database import Database
from analyzer import QuestionAnalyzer


#Lifespan: set up DB + analyzer once 
@asynccontextmanager
async def lifespan(server: FastMCP):
    Config.validate()
    db = Database()
    await db.connect()
    analyzer = QuestionAnalyzer(db)
    yield {"db": db, "analyzer": analyzer}
    await db.close()


mcp = FastMCP("unanswered_questions_mcp", lifespan=lifespan)


#Input models 
class LogQuestionInput(BaseModel):
    """Input for logging an unanswered question."""
    question: str = Field(..., description="The question the chatbot couldn't answer")
    context: Optional[str] = Field(None, description="Where/why this question was asked")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier for multi-tenant systems")
    metadata: Optional[dict] = Field(None, description="Extra metadata (user_id, confidence, source, etc.)")


class GetPatternsInput(BaseModel):
    """Input for pattern detection."""
    tenant_id: Optional[str] = Field(None, description="Filter to a specific tenant")
    min_cluster_size: Optional[int] = Field(None, description="Minimum questions to form a pattern (default: 3)", ge=2)


class SuggestDocsInput(BaseModel):
    """Input for documentation suggestions."""
    topic: str = Field(..., description="Topic to get suggestions for (e.g. 'password reset')")
    tenant_id: Optional[str] = Field(None, description="Filter to a specific tenant")


class MarkResolvedInput(BaseModel):
    """Input for marking a question resolved."""
    question_id: str = Field(..., description="ID of the question to resolve")
    document_id: str = Field(..., description="ID or reference of the document that answers it")
    notes: Optional[str] = Field(None, description="Notes on how this was resolved")


#Tools
@mcp.tool(
    name="log_unanswered_question",
    annotations={
        "title": "Log Unanswered Question",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)

async def log_unanswered_question(params: LogQuestionInput, ctx=None) -> str:
    """Log a question the chatbot couldn't answer. Generates an embedding
    and stores it for later pattern analysis."""
    db: Database = ctx.request_context.lifespan_state["db"]
    analyzer: QuestionAnalyzer = ctx.request_context.lifespan_state["analyzer"]

    embedding = await analyzer.generate_embedding(params.question)
    qid = await db.save_question(
        question=params.question,
        context=params.context,
        tenant_id=params.tenant_id,
        metadata=params.metadata,
        embedding=embedding,
    )
    return json.dumps({"success": True, "question_id": qid})


@mcp.tool(
    name="get_question_patterns",
    annotations={
        "title": "Get Question Patterns",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)

async def get_question_patterns(params: GetPatternsInput, ctx=None) -> str:
    """Cluster unresolved questions by semantic similarity and return
    patterns showing common knowledge gaps."""
    db: Database = ctx.request_context.lifespan_state["db"]
    analyzer: QuestionAnalyzer = ctx.request_context.lifespan_state["analyzer"]

    patterns = await analyzer.find_patterns(
        tenant_id=params.tenant_id,
        min_cluster_size=params.min_cluster_size,
    )
    stats = await db.get_stats(tenant_id=params.tenant_id)
    return json.dumps({"success": True, "stats": stats, "patterns": patterns, "pattern_count": len(patterns)})


@mcp.tool(
    name="suggest_documents",
    annotations={
        "title": "Suggest Documents",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)

async def suggest_documents(params: SuggestDocsInput, ctx=None) -> str:
    """Analyze questions related to a topic and suggest what documentation
    to create, including title, sections, and content to cover."""
    analyzer: QuestionAnalyzer = ctx.request_context.lifespan_state["analyzer"]

    suggestion = await analyzer.suggest_documentation(
        topic=params.topic, tenant_id=params.tenant_id
    )
    return json.dumps({"success": True, **suggestion})


@mcp.tool(
    name="mark_resolved",
    annotations={
        "title": "Mark Question Resolved",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)

async def mark_resolved(params: MarkResolvedInput, ctx=None) -> str:
    """Mark a question as resolved after documentation has been added."""
    db: Database = ctx.request_context.lifespan_state["db"]

    question = await db.get_question(params.question_id)
    if not question:
        return json.dumps({"success": False, "error": f"Question {params.question_id} not found"})

    ok = await db.mark_resolved(
        question_id=params.question_id,
        resolved_by=params.document_id,
        notes=params.notes,
    )
    return json.dumps({
        "success": ok,
        "question_id": params.question_id,
        "question": question["question"],
        "document_id": params.document_id,
    })


if __name__ == "__main__":
    mcp.run()
