import json
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity

from config import Config
from database import Database


class QuestionAnalyzer:
    """Clusters unanswered questions by semantic similarity."""

    def __init__(self, database: Database):
        self.db = database
        self.client = openai.AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

    # Embeddings
    async def generate_embedding(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=Config.EMBEDDING_MODEL, input=text
        )
        return response.data[0].embedding

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), Config.EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + Config.EMBEDDING_BATCH_SIZE]
            response = await self.client.embeddings.create(
                model=Config.EMBEDDING_MODEL, input=batch
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    async def ensure_embeddings(self, questions: list[dict]) -> list[dict]:
        """Fill in missing embeddings and persist them."""
        needs = [q for q in questions if q["embedding"] is None]
        if not needs:
            return questions
        texts = [q["question"] for q in needs]
        embeddings = await self.generate_embeddings_batch(texts)
        for q, emb in zip(needs, embeddings):
            q["embedding"] = emb
            await self.db.update_embedding(q["id"], emb)
        return questions

    # Clustering
    def cluster_questions(self, questions: list[dict], threshold: float | None = None) -> list[list[dict]]:
        """
        Greedy similarity clustering.

        Walk through questions in order. For each unassigned question,
        start a new cluster and pull in every unassigned question whose
        cosine similarity exceeds the threshold.
        """
        if not questions:
            return []

        threshold = threshold or Config.SIMILARITY_THRESHOLD
        embeddings = np.array([q["embedding"] for q in questions])
        sim = cosine_similarity(embeddings)

        clusters: list[list[dict]] = []
        assigned: set[int] = set()

        for i, q in enumerate(questions):
            if i in assigned:
                continue
            cluster = [q]
            assigned.add(i)
            for j in range(len(questions)):
                if j not in assigned and sim[i][j] >= threshold:
                    print(f"  {q['question'][:40]} <-> {questions[j]['question'][:40]}: {sim[i][j]:.3f}")
                    cluster.append(questions[j])
                    assigned.add(j)
            print(f"Cluster size: {len(cluster)} (need {Config.MIN_CLUSTER_SIZE})")
            if len(cluster) >= Config.MIN_CLUSTER_SIZE:
                clusters.append(cluster[: Config.MAX_CLUSTER_SIZE])

        return clusters

    def pick_representative(self, cluster: list[dict], n: int = 3) -> list[str]:
        """Return the n most central questions in a cluster."""
        if len(cluster) <= n:
            return [q["question"] for q in cluster]

        embeddings = np.array([q["embedding"] for q in cluster])
        sim = cosine_similarity(embeddings)
        centrality = [
            (np.mean([sim[i][j] for j in range(len(cluster)) if j != i]), cluster[i])
            for i in range(len(cluster))
        ]
        centrality.sort(key=lambda x: x[0], reverse=True)
        return [q["question"] for _, q in centrality[:n]]

    # High-level analysis
    async def extract_topic(self, cluster: list[dict]) -> str:
        """Ask GPT for a short topic label for a cluster."""
        sample = [q["question"] for q in cluster[:10]]
        prompt = (
            "These are similar customer questions. "
            "Reply with ONLY a 2-5 word topic label.\n\n"
            + "\n".join(f"- {q}" for q in sample)
        )
        resp = await self.client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        return resp.choices[0].message.content.strip()

    async def find_patterns(self, tenant_id: str | None = None, min_cluster_size: int | None = None,) -> list[dict]:
        """
        End-to-end pattern detection:
        1. Fetch unresolved questions
        2. Ensure embeddings exist
        3. Cluster by similarity
        4. Label each cluster with a topic
        """
        questions = await self.db.get_unresolved(tenant_id=tenant_id)
        if not questions:
            return []

        questions = await self.ensure_embeddings(questions)
        clusters = self.cluster_questions(questions)

        if min_cluster_size:
            clusters = [c for c in clusters if len(c) >= min_cluster_size]

        patterns: list[dict] = []
        for cluster in clusters:
            topic = await self.extract_topic(cluster)
            representative = self.pick_representative(cluster)
            qids = [q["id"] for q in cluster]

            patterns.append(
                {
                    "topic": topic,
                    "count": len(cluster),
                    "representative_questions": representative,
                    "question_ids": qids,
                    "earliest_date": min(q["created_at"] for q in cluster),
                    "latest_date": max(q["created_at"] for q in cluster),
                }
            )
            await self.db.save_cluster(
                topic=topic,
                question_ids=qids,
                representative_questions=representative,
                tenant_id=tenant_id,
            )

        patterns.sort(key=lambda p: p["count"], reverse=True)
        return patterns

    async def suggest_documentation(self, topic: str, tenant_id: str | None = None) -> dict:
        """Suggest what docs to write for a given topic."""
        questions = await self.db.get_unresolved(tenant_id=tenant_id)
        if not questions:
            return {"topic": topic, "suggestion": "No unresolved questions.", "related_questions": []}

        questions = await self.ensure_embeddings(questions)
        topic_emb = await self.generate_embedding(topic)
        q_embs = [q["embedding"] for q in questions]
        sims = cosine_similarity([topic_emb], q_embs)[0]

        related = [
            {"question": questions[i]["question"], "context": questions[i]["context"], "similarity": round(float(sims[i]), 3)}
            for i in np.argsort(sims)[::-1][:10]
            if sims[i] >= 0.6
        ]

        if not related:
            return {"topic": topic, "suggestion": f"No questions related to '{topic}'.", "related_questions": []}

        q_text = "\n".join(
            f"- {r['question']}" + (f" (Context: {r['context']})" if r["context"] else "")
            for r in related[:5]
        )
        prompt = (
            f'Based on these customer questions about "{topic}", suggest documentation to create.\n\n'
            f"{q_text}\n\n"
            "Reply with ONLY raw JSON, no markdown formatting:\n"
            '{"title": "specific doc title", "sections": ["specific section 1", "specific section 2", ...], "details": "what to cover"}'
            "Have maximum 5 sections for title."
        )
        resp = await self.client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1000,
        )
        try:
            raw = resp.choices[0].message.content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            suggestion = json.loads(raw)
        except json.JSONDecodeError:
            suggestion = {"title": f"Documentation: {topic}", "sections": ["Overview", "Common Questions", "Examples"], "details": resp.choices[0].message.content}

        return {"topic": topic, "suggestion": suggestion, "related_questions": related, "question_count": len(related)}
