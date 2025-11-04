"""
RAG (Retrieval-Augmented Generation) Service
Handles medical knowledge retrieval and context management
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
import numpy as np

from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError
from ..config import get_settings

logger = get_logger(__name__)
settings = get_settings()

@dataclass
class Document:
    """Medical document structure"""
    id: str
    title: str
    content: str
    source: str
    medical_domain: str
    last_updated: str
    metadata: Dict[str, Any]


@dataclass
class RetrievalResult:
    """Document retrieval result"""
    document: Document
    similarity_score: float
    relevant_snippets: List[str]
    medical_relevance: float


class RAGService:
    """Retrieval-Augmented Generation service for medical knowledge"""
    
    def __init__(self):
        self.logger = get_logger("rag_service")
        self.vector_store = None
        self.embeddings = None
        self.medical_knowledge_base = []
        self.similarity_threshold = settings.similarity_threshold
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        # Initialize medical knowledge base
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize medical knowledge base with sample data"""
        
        # Sample medical documents
        medical_documents = [
            {
                "id": "guidelines_hypertension",
                "title": "Hypertension Management Guidelines",
                "content": """
                Hypertension management involves lifestyle modifications and pharmacological interventions.
                
                Lifestyle modifications:
                - Sodium restriction (<2.3g/day)
                - Regular exercise (150 min/week moderate intensity)
                - Weight loss if BMI >25
                - Limited alcohol consumption
                
                Pharmacological treatment:
                - ACE inhibitors, ARBs, or calcium channel blockers
                - Target BP <140/90 mmHg for most patients
                - Target BP <130/80 mmHg for patients with diabetes or CKD
                
                Monitoring:
                - Regular BP monitoring
                - Annual screening for target organ damage
                - Medication adherence assessment
                """,
                "source": "medical_guidelines",
                "medical_domain": "cardiology"
            },
            {
                "id": "diabetes_management",
                "title": "Diabetes Mellitus Management",
                "content": """
                Diabetes management focuses on glycemic control and complication prevention.
                
                Diagnostic criteria:
                - HbA1c ≥6.5%
                - Fasting plasma glucose ≥126 mg/dL
                - 2-hour plasma glucose ≥200 mg/dL during OGTT
                
                Management:
                - Dietary modifications and weight management
                - Regular physical activity
                - Metformin as first-line therapy
                - Insulin therapy when indicated
                
                Monitoring:
                - HbA1c every 3 months if not at goal
                - Annual comprehensive diabetes evaluation
                - Screening for complications
                """,
                "source": "clinical_practice",
                "medical_domain": "endocrinology"
            },
            {
                "id": "chest_pain_evaluation",
                "title": "Chest Pain Evaluation Protocol",
                "content": """
                Chest pain requires systematic evaluation to rule out life-threatening causes.
                
                Immediate assessment:
                - Vital signs and cardiac monitoring
                - ECG within 10 minutes
                - Cardiac biomarkers (troponin)
                
                Risk stratification:
                - HEART score for急诊 patients
                - TIMI score for unstable angina
                
                Red flags requiring immediate attention:
                - ST-elevation on ECG
                - Positive troponin
                - Hemodynamic instability
                - Signs of heart failure
                
                Disposition:
                - Admit if high-risk features
                - Outpatient workup if low-risk
                """,
                "source": "emergency_medicine",
                "medical_domain": "cardiology"
            }
        ]
        
        # Convert to Document objects
        for doc_data in medical_documents:
            doc = Document(
                id=doc_data["id"],
                title=doc_data["title"],
                content=doc_data["content"],
                source=doc_data["source"],
                medical_domain=doc_data["medical_domain"],
                last_updated=datetime.now(timezone.utc).isoformat(),
                metadata={}
            )
            self.medical_knowledge_base.append(doc)
        
        self.logger.info(
            "Medical knowledge base initialized",
            document_count=len(self.medical_knowledge_base)
        )
    
    async def retrieve_relevant_documents(
        self,
        query: str,
        medical_domain: Optional[str] = None,
        max_results: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant medical documents based on query
        """
        
        start_time = time.time()
        
        try:
            # Filter by medical domain if specified
            candidate_docs = self.medical_knowledge_base
            if medical_domain:
                candidate_docs = [
                    doc for doc in candidate_docs 
                    if doc.medical_domain == medical_domain
                ]
            
            # Calculate similarity scores
            retrieval_results = []
            for doc in candidate_docs:
                similarity_score = self._calculate_similarity(query, doc.content)
                medical_relevance = self._calculate_medical_relevance(query, doc)
                
                if similarity_score >= self.similarity_threshold:
                    relevant_snippets = self._extract_relevant_snippets(query, doc.content)
                    
                    retrieval_result = RetrievalResult(
                        document=doc,
                        similarity_score=similarity_score,
                        relevant_snippets=relevant_snippets,
                        medical_relevance=medical_relevance
                    )
                    retrieval_results.append(retrieval_result)
            
            # Sort by combined score
            retrieval_results.sort(
                key=lambda x: (x.similarity_score + x.medical_relevance) / 2,
                reverse=True
            )
            
            # Limit results
            retrieval_results = retrieval_results[:max_results]
            
            retrieval_time = (time.time() - start_time) * 1000
            
            self.logger.debug(
                "Document retrieval completed",
                query=query[:100],
                results_count=len(retrieval_results),
                retrieval_time_ms=retrieval_time,
                medical_domain=medical_domain
            )
            
            return retrieval_results
            
        except Exception as e:
            self.logger.error(f"Document retrieval failed: {e}")
            raise ValidationError(f"Document retrieval failed: {str(e)}")
    
    async def enhance_prompt_with_context(
        self,
        query: str,
        medical_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance user query with relevant medical context
        """
        
        try:
            # Retrieve relevant documents
            retrieval_results = await self.retrieve_relevant_documents(
                query=query,
                medical_domain=medical_domain,
                max_results=3
            )
            
            if not retrieval_results:
                return {
                    "enhanced_query": query,
                    "context": "",
                    "sources": [],
                    "relevance_score": 0.0
                }
            
            # Combine relevant snippets
            context_parts = []
            sources = []
            
            for result in retrieval_results:
                # Add document title as source
                sources.append({
                    "title": result.document.title,
                    "source": result.document.source,
                    "relevance_score": result.similarity_score
                })
                
                # Add relevant snippets
                for snippet in result.relevant_snippets:
                    context_parts.append(f"[{result.document.title}] {snippet}")
            
            # Create enhanced context
            context = "\n\n".join(context_parts)
            
            # Combine with original query
            enhanced_query = f"Query: {query}\n\nRelevant Medical Context:\n{context}"
            
            # Calculate average relevance
            relevance_score = sum(r.similarity_score for r in retrieval_results) / len(retrieval_results)
            
            return {
                "enhanced_query": enhanced_query,
                "context": context,
                "sources": sources,
                "relevance_score": relevance_score,
                "documents_used": len(retrieval_results)
            }
            
        except Exception as e:
            self.logger.error(f"Prompt enhancement failed: {e}")
            raise ValidationError(f"Prompt enhancement failed: {str(e)}")
    
    async def search_medical_terminology(
        self,
        term: str,
        medical_domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for medical terminology and definitions
        """
        
        # Mock medical terminology database
        medical_terms = {
            "hypertension": {
                "definition": "Persistently elevated arterial blood pressure",
                "normal_range": "<120/80 mmHg",
                "stages": {
                    "elevated": "120-129 systolic and <80 diastolic",
                    "stage_1": "130-139 systolic or 80-89 diastolic",
                    "stage_2": "≥140 systolic or ≥90 diastolic"
                },
                "domain": "cardiology"
            },
            "diabetes": {
                "definition": "Metabolic disease characterized by chronic hyperglycemia",
                "types": {
                    "type_1": "Autoimmune destruction of pancreatic beta cells",
                    "type_2": "Insulin resistance and relative insulin deficiency",
                    "gestational": "Diabetes diagnosed during pregnancy"
                },
                "domain": "endocrinology"
            },
            "myocardial_infarction": {
                "definition": "Death of cardiac muscle tissue due to prolonged ischemia",
                "symptoms": ["Chest pain", "Shortness of breath", "Diaphoresis", "Nausea"],
                "diagnosis": ["ECG changes", "Elevated cardiac biomarkers"],
                "domain": "cardiology"
            }
        }
        
        # Search for term
        term_lower = term.lower()
        results = []
        
        for medical_term, info in medical_terms.items():
            if term_lower in medical_term.lower() or term_lower in str(info).lower():
                # Check domain filter
                if medical_domain and info.get("domain") != medical_domain:
                    continue
                
                results.append({
                    "term": medical_term,
                    "definition": info["definition"],
                    "domain": info.get("domain"),
                    "additional_info": info
                })
        
        self.logger.debug(
            "Medical terminology search completed",
            term=term,
            results_count=len(results),
            medical_domain=medical_domain
        )
        
        return results
    
    async def get_clinical_guidelines(
        self,
        condition: str,
        medical_domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve clinical guidelines for specific conditions
        """
        
        # Mock clinical guidelines database
        guidelines_db = {
            "hypertension": {
                "title": "Hypertension Management Guidelines",
                "recommendations": [
                    "Lifestyle modifications first-line",
                    "Pharmacological treatment if BP ≥140/90",
                    "Target BP <140/90 for most patients",
                    "Target BP <130/80 for high-risk patients"
                ],
                "medications": ["ACE inhibitors", "ARBs", "Calcium channel blockers", "Diuretics"],
                "monitoring": ["Regular BP checks", "Annual labs", "Adherence assessment"],
                "domain": "cardiology"
            },
            "chest_pain": {
                "title": "Chest Pain Evaluation Protocol",
                "immediate_actions": [
                    "ECG within 10 minutes",
                    "Cardiac biomarkers",
                    "Vital signs and monitoring"
                ],
                "red_flags": [
                    "ST-elevation",
                    "Positive troponin",
                    "Hemodynamic instability"
                ],
                "domain": "emergency_medicine"
            }
        }
        
        # Search for condition
        condition_lower = condition.lower()
        results = []
        
        for condition_key, guideline in guidelines_db.items():
            if condition_lower in condition_key.lower():
                # Check domain filter
                if medical_domain and guideline.get("domain") != medical_domain:
                    continue
                
                results.append({
                    "condition": condition_key,
                    "guideline": guideline
                })
        
        return results
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        
        domain_stats = {}
        for doc in self.medical_knowledge_base:
            domain = doc.medical_domain
            domain_stats[domain] = domain_stats.get(domain, 0) + 1
        
        return {
            "total_documents": len(self.medical_knowledge_base),
            "domains": list(domain_stats.keys()),
            "domain_distribution": domain_stats,
            "similarity_threshold": self.similarity_threshold,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def _calculate_similarity(self, query: str, document: str) -> float:
        """
        Calculate similarity between query and document
        In production, this would use actual embeddings and cosine similarity
        """
        
        # Simple keyword-based similarity for demonstration
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        query_words = query_words - stop_words
        doc_words = doc_words - stop_words
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_words.intersection(doc_words)
        union = query_words.union(doc_words)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Medical domain bonus
        medical_keywords = [
            "symptoms", "diagnosis", "treatment", "medication", "disease",
            "condition", "patient", "clinical", "medical", "therapy"
        ]
        
        medical_bonus = 0.0
        for keyword in medical_keywords:
            if keyword in query.lower() and keyword in document.lower():
                medical_bonus += 0.1
        
        # Combine scores
        similarity_score = min(1.0, jaccard_similarity + medical_bonus)
        
        return similarity_score
    
    def _calculate_medical_relevance(self, query: str, document: Document) -> float:
        """
        Calculate medical relevance of document to query
        """
        
        relevance_score = 0.5  # Base relevance
        
        # Domain match bonus
        if any(domain in query.lower() for domain in [document.medical_domain]):
            relevance_score += 0.3
        
        # Medical terminology bonus
        medical_terms_in_doc = len([term for term in document.content.lower().split() 
                                  if term in ["diagnosis", "treatment", "symptoms", "medication"]])
        
        if medical_terms_in_doc > 0:
            relevance_score += min(0.2, medical_terms_in_doc * 0.05)
        
        return min(1.0, relevance_score)
    
    def _extract_relevant_snippets(self, query: str, document: str) -> List[str]:
        """
        Extract relevant snippets from document based on query
        """
        
        query_words = set(query.lower().split())
        sentences = document.split('.')
        
        relevant_snippets = []
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            
            # Find sentences with query keywords
            overlap = query_words.intersection(sentence_words)
            
            if overlap:
                # Score based on keyword overlap
                score = len(overlap) / len(query_words) if query_words else 0
                
                if score > 0.3:  # Threshold for relevance
                    relevant_snippets.append(sentence.strip())
        
        # Return top 3 most relevant snippets
        relevant_snippets.sort(key=len, reverse=True)
        return relevant_snippets[:3]
    
    async def cleanup(self):
        """Clean up RAG service resources"""
        
        self.logger.info("Cleaning up RAG service")
        
        try:
            # Clear knowledge base
            self.medical_knowledge_base.clear()
            
            # Clear vector store (if using external vector database)
            self.vector_store = None
            
            self.logger.info("RAG service cleanup completed")
            
        except Exception as e:
            self.logger.error(f"RAG service cleanup failed: {e}")


# Global RAG service instance
rag_service = RAGService()