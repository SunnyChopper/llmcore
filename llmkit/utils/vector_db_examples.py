import asyncio
import chromadb
from chromadb.config import Settings
import networkx as nx
from datetime import datetime, timedelta
import json
import uuid
import math
from llmkit.core import LLM, LLMConfig
from llmkit.embeddings import Embeddings
from llmkit.logger import setup_logger

logger = setup_logger(__name__)

class Source:
    def __init__(self, url, credibility):
        self.url = url
        self.credibility = credibility

class Concept:
    def __init__(self, name, description, category, confidence, sources, abstraction_level):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.category = category
        self.confidence = confidence
        self.sources = [Source(**s) if isinstance(s, dict) else s for s in sources]
        self.abstraction_level = abstraction_level
        self.versions = [{
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "confidence": confidence,
            "sources": [{"url": s.url, "credibility": s.credibility} for s in self.sources],
            "abstraction_level": abstraction_level
        }]

    def update(self, description, confidence, sources, abstraction_level):
        self.description = description
        self.confidence = confidence
        self.sources = [Source(**s) if isinstance(s, dict) else s for s in sources]
        self.abstraction_level = abstraction_level
        self.versions.append({
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "confidence": confidence,
            "sources": [{"url": s.url, "credibility": s.credibility} for s in self.sources],
            "abstraction_level": abstraction_level
        })

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "confidence": self.confidence,
            "sources": [{"url": s.url, "credibility": s.credibility} for s in self.sources],
            "abstraction_level": self.abstraction_level,
            "versions": self.versions
        }

class Relationship:
    def __init__(self, source, target, rel_type, strength):
        self.id = str(uuid.uuid4())
        self.source = source
        self.target = target
        self.type = rel_type
        self.strength = strength
        self.versions = [{
            "timestamp": datetime.now().isoformat(),
            "type": rel_type,
            "strength": strength
        }]

    def update(self, rel_type, strength):
        self.type = rel_type
        self.strength = strength
        self.versions.append({
            "timestamp": datetime.now().isoformat(),
            "type": rel_type,
            "strength": strength
        })

    def to_dict(self):
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "strength": self.strength,
            "versions": self.versions
        }

class DynamicKnowledgeGraph:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./dynamic_knowledge_graph"
        ))
        self.collection = self.client.create_collection("concepts")
        self.embeddings = Embeddings(provider="openai", model="text-embedding-ada-002")
        self.llm = LLM(provider="openai", model="gpt-4", config=LLMConfig(temperature=0.7, max_tokens=1500))
        self.graph = nx.MultiDiGraph()

    async def add_concept(self, name: str, description: str, category: str, confidence: float, sources: list, abstraction_level: str):
        concept = Concept(name, description, category, confidence, sources, abstraction_level)
        vector = await self.embeddings.embed_async(description)
        self.collection.add(
            embeddings=[vector],
            documents=[json.dumps(concept.to_dict())],
            metadatas=[{"id": concept.id, "name": name, "category": category, "abstraction_level": abstraction_level}],
            ids=[concept.id]
        )
        self.graph.add_node(concept.id, concept=concept)
        return concept.id

    async def update_concept(self, concept_id: str, new_description: str, new_confidence: float, new_sources: list, new_abstraction_level: str):
        concept = self.graph.nodes[concept_id]['concept']
        concept.update(new_description, new_confidence, new_sources, new_abstraction_level)
        vector = await self.embeddings.embed_async(new_description)
        self.collection.update(
            embeddings=[vector],
            documents=[json.dumps(concept.to_dict())],
            metadatas=[{"id": concept.id, "name": concept.name, "category": concept.category, "abstraction_level": new_abstraction_level}],
            ids=[concept.id]
        )

    async def add_relationship(self, source_id: str, target_id: str, rel_type: str, strength: float):
        relationship = Relationship(source_id, target_id, rel_type, strength)
        self.graph.add_edge(source_id, target_id, key=relationship.id, relationship=relationship)
        return relationship.id

    async def update_relationship(self, rel_id: str, new_type: str, new_strength: float):
        for _, _, data in self.graph.edges(data=True):
            if data.get('key') == rel_id:
                relationship = data['relationship']
                relationship.update(new_type, new_strength)
                break

    async def get_concept_history(self, concept_id: str):
        concept = self.graph.nodes[concept_id]['concept']
        return concept.versions

    async def get_related_concepts(self, concept_id: str, time_range: tuple = None):
        related = []
        for _, target, data in self.graph.out_edges(concept_id, data=True):
            relationship = data['relationship']
            if time_range:
                start, end = time_range
                if any(start <= datetime.fromisoformat(v['timestamp']) <= end for v in relationship.versions):
                    related.append((target, relationship.type, relationship.strength))
            else:
                related.append((target, relationship.type, relationship.strength))
        return related

    async def query_graph(self, query: str):
        vector = await self.embeddings.embed_async(query)
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=5,
            include=["documents", "metadatas"]
        )
        return [json.loads(doc) for doc in results['documents'][0]]

    async def suggest_new_concepts(self):
        all_concepts = [self.graph.nodes[n]['concept'].name for n in self.graph.nodes()]
        prompt = f"""
        Based on the following concepts in our knowledge graph:
        {', '.join(all_concepts)}
        
        Suggest 3 new related concepts that are not in the list above. 
        For each concept, provide a name, brief description, category, estimated confidence level (0-1), 
        and abstraction level (beginner, intermediate, expert).
        Format your response as a JSON list of objects.
        """
        response = await self.llm.send_input_async(prompt)
        return json.loads(response)

    async def generate_evolution_summary(self, concept_id: str):
        concept = self.graph.nodes[concept_id]['concept']
        prompt = f"""
        Analyze the evolution of the concept "{concept.name}" based on its version history:
        {json.dumps(concept.versions, indent=2)}
        
        Provide a concise summary of how this concept has evolved over time, 
        noting any significant changes in description, confidence, or abstraction level.
        Also, mention any changes in the credibility of sources used.
        """
        return await self.llm.send_input_async(prompt)

    def apply_confidence_decay(self, decay_rate: float = 0.1):
        current_time = datetime.now()
        for node_id in self.graph.nodes():
            concept = self.graph.nodes[node_id]['concept']
            last_update = datetime.fromisoformat(concept.versions[-1]['timestamp'])
            time_diff = (current_time - last_update).days
            decay_factor = math.exp(-decay_rate * time_diff)
            concept.confidence *= decay_factor
            if concept.confidence < 0.5:
                print(f"Warning: Confidence for concept '{concept.name}' has dropped below 0.5. Consider updating.")

async def main():
    print("Initializing Enhanced Dynamic Knowledge Graph...")
    dkg = DynamicKnowledgeGraph()

    # Initialize with seed concepts
    ai_id = await dkg.add_concept(
        "Artificial Intelligence",
        "The simulation of human intelligence in machines that are programmed to think and learn like humans.",
        "Computer Science",
        0.9,
        [{"url": "https://www.ibm.com/cloud/learn/what-is-artificial-intelligence", "credibility": 0.9}],
        "beginner"
    )
    ml_id = await dkg.add_concept(
        "Machine Learning",
        "A subset of AI that enables systems to automatically learn and improve from experience without being explicitly programmed.",
        "Computer Science",
        0.95,
        [{"url": "https://www.expert.ai/blog/machine-learning-definition/", "credibility": 0.85}],
        "intermediate"
    )
    nn_id = await dkg.add_concept(
        "Neural Networks",
        "Computing systems inspired by biological neural networks that form the basis of many machine learning algorithms, particularly deep learning.",
        "Computer Science",
        0.9,
        [{"url": "https://www.ibm.com/cloud/learn/neural-networks", "credibility": 0.9}],
        "expert"
    )
    ethics_id = await dkg.add_concept(
        "AI Ethics",
        "The branch of ethics that focuses on the moral issues surrounding the development and use of artificial intelligence.",
        "Philosophy",
        0.85,
        [{"url": "https://plato.stanford.edu/entries/ethics-ai/", "credibility": 0.95}],
        "intermediate"
    )

    # Add relationships
    await dkg.add_relationship(ml_id, ai_id, "is-a", 0.9)
    await dkg.add_relationship(nn_id, ml_id, "used-in", 0.8)
    await dkg.add_relationship(ethics_id, ai_id, "applies-to", 0.9)

    # Update concepts over time
    await asyncio.sleep(1)  # Simulate time passing
    await dkg.update_concept(
        ai_id,
        "The creation of intelligent machines that work and react like humans, encompassing a wide range of capabilities such as reasoning, problem-solving, and learning.",
        0.95,
        [
            {"url": "https://www.sas.com/en_us/insights/analytics/what-is-artificial-intelligence.html", "credibility": 0.9},
            {"url": "https://www.nature.com/articles/d41586-019-03322-9", "credibility": 0.95}
        ],
        "intermediate"
    )
    dl_id = await dkg.add_concept(
        "Deep Learning",
        "A subset of machine learning based on artificial neural networks with multiple layers, capable of learning from large amounts of unstructured data.",
        "Computer Science",
        0.9,
        [{"url": "https://www.ibm.com/cloud/learn/deep-learning", "credibility": 0.9}],
        "expert"
    )
    await dkg.add_relationship(dl_id, ml_id, "is-a", 0.9)
    
    await asyncio.sleep(1)  # Simulate more time passing
    await dkg.update_concept(
        ai_id,
        "The field of computer science dedicated to creating systems that can perform tasks that typically require human intelligence, including visual perception, speech recognition, decision-making, and language translation.",
        0.98,
        [
            {"url": "https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-explained", "credibility": 0.95},
            {"url": "https://www.nature.com/articles/d41586-019-03322-9", "credibility": 0.95}
        ],
        "expert"
    )
    transformer_id = await dkg.add_concept(
        "Transformers",
        "A type of deep learning model designed to handle sequential data, using self-attention mechanisms to process input data in parallel.",
        "Computer Science",
        0.85,
        [{"url": "https://arxiv.org/abs/1706.03762", "credibility": 0.9}],
        "expert"
    )
    await dkg.add_relationship(transformer_id, dl_id, "is-a", 0.9)
    await dkg.add_relationship(transformer_id, nn_id, "uses", 0.8)

    # Query the graph
    print("\nQuerying for concepts related to 'learning':")
    query_results = await dkg.query_graph("learning in artificial intelligence")
    for result in query_results:
        print(f"- {result['name']} ({result['abstraction_level']}): {result['description']}")

    # Generate evolution summary
    ai_evolution = await dkg.generate_evolution_summary(ai_id)
    print(f"\nEvolution of Artificial Intelligence:\n{ai_evolution}")

    # Time-based query
    one_minute_ago = (datetime.now() - timedelta(minutes=1)).isoformat()
    recent_related = await dkg.get_related_concepts(ai_id, (one_minute_ago, datetime.now().isoformat()))
    print(f"\nRecently added concepts related to AI:")
    for concept_id, rel_type, strength in recent_related:
        concept = dkg.graph.nodes[concept_id]['concept']
        print(f"- {concept.name} ({rel_type}, strength: {strength})")

    # Suggest new concepts
    print("\nSuggesting new concepts:")
    suggestions = await dkg.suggest_new_concepts()
    for suggestion in suggestions:
        print(f"- {suggestion['name']} ({suggestion['abstraction_level']}): {suggestion['description']}")
        print(f"  Category: {suggestion['category']}, Confidence: {suggestion['confidence']}")

    # Apply confidence decay
    print("\nApplying confidence decay...")
    dkg.apply_confidence_decay()
    for node_id in dkg.graph.nodes():
        concept = dkg.graph.nodes[node_id]['concept']
        print(f"- {concept.name}: New confidence = {concept.confidence:.2f}")

    print("\nEnhanced Dynamic Knowledge Graph demonstration complete.")

if __name__ == "__main__":
    asyncio.run(main())