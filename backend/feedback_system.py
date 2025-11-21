"""Feedback learning system for adaptive response improvement."""
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np


class ChatbotFeedbackSystem:
    """Learns from user feedback to improve chatbot responses."""
    
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.feedback_history: List[Dict[str, Any]] = []
        self.positive_feedback: List[Dict[str, Any]] = []
        self.negative_feedback: List[Dict[str, Any]] = []
        self.improvement_keywords: Dict[str, int] = {}
        self.load_feedback()
    
    def add_feedback(
        self,
        query: str,
        response: str,
        rating: str,
        comment: Optional[str] = None,
        context_used: str = ""
    ):
        """Record user feedback on a response."""
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "rating": rating,
            "comment": comment,
            "response_length": len(response.split()),
            "context_length": len(context_used.split()) if context_used else 0,
            "session_number": len(self.feedback_history) + 1
        }
        
        self.feedback_history.append(feedback_entry)
        
        if rating == "👍":
            self.positive_feedback.append(feedback_entry)
        else:
            self.negative_feedback.append(feedback_entry)
            if comment:
                self._extract_improvement_keywords(comment)
        
        self.save_feedback()
    
    def _extract_improvement_keywords(self, comment: str):
        """Extract keywords from negative feedback."""
        improvement_indicators = [
            "more detail", "too long", "too short", "unclear", "confusing",
            "not relevant", "missing", "incorrect", "better explanation",
            "more examples", "simpler", "more technical", "more context",
            "incomplete", "off-topic", "vague", "specific", "concise"
        ]
        
        comment_lower = comment.lower()
        for indicator in improvement_indicators:
            if indicator in comment_lower:
                self.improvement_keywords[indicator] = self.improvement_keywords.get(indicator, 0) + 1
    
    def get_satisfaction_rate(self) -> float:
        """Calculate overall satisfaction rate."""
        if not self.feedback_history:
            return 0.0
        return (len(self.positive_feedback) / len(self.feedback_history)) * 100
    
    def get_improvement_insights(self) -> Dict[str, Any]:
        """Analyze feedback to identify improvement patterns."""
        if not self.feedback_history:
            return {
                "total_feedback": 0,
                "positive_count": 0,
                "negative_count": 0,
                "satisfaction_rate": 0.0,
                "positive_avg_length": 0,
                "negative_avg_length": 0,
                "top_issues": [],
                "insights": "No feedback yet"
            }
        
        positive_avg = np.mean([f["response_length"] for f in self.positive_feedback]) if self.positive_feedback else 0
        negative_avg = np.mean([f["response_length"] for f in self.negative_feedback]) if self.negative_feedback else 0
        top_issues = sorted(self.improvement_keywords.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "total_feedback": len(self.feedback_history),
            "positive_count": len(self.positive_feedback),
            "negative_count": len(self.negative_feedback),
            "satisfaction_rate": self.get_satisfaction_rate(),
            "positive_avg_length": int(positive_avg),
            "negative_avg_length": int(negative_avg),
            "top_issues": top_issues
        }
    
    def generate_system_prompt(self) -> str:
        """Generate adaptive system prompt based on feedback."""
        base = "You are a helpful AI assistant. Answer questions accurately based on the provided context."
        
        if len(self.feedback_history) < 3:
            return base
        
        insights = self.get_improvement_insights()
        adaptations = []
        
        if insights["positive_avg_length"] > 0 and insights["negative_avg_length"] > 0:
            if insights["positive_avg_length"] < insights["negative_avg_length"]:
                adaptations.append("Keep responses concise and to the point.")
            else:
                adaptations.append("Provide detailed, comprehensive answers.")
        
        for issue, _ in insights.get("top_issues", []):
            if "too long" in issue:
                adaptations.append("Be concise without unnecessary elaboration.")
            elif "more detail" in issue or "incomplete" in issue:
                adaptations.append("Provide thorough explanations with sufficient detail.")
            elif "unclear" in issue or "confusing" in issue:
                adaptations.append("Use clear, simple language and structure your answers well.")
            elif "not relevant" in issue or "off-topic" in issue:
                adaptations.append("Focus strictly on the question asked using only relevant context.")
        
        return base + " " + " ".join(adaptations) if adaptations else base
    
    def get_recent_improvement_trend(self, window_size: int = 5) -> Dict[str, Any]:
        """Calculate satisfaction trend over recent interactions."""
        if len(self.feedback_history) < window_size:
            return {
                "trend": "insufficient_data",
                "recent_rate": 0.0,
                "previous_rate": 0.0,
                "improvement": 0.0
            }
        
        recent = self.feedback_history[-window_size:]
        recent_positive = sum(1 for f in recent if f["rating"] == "👍")
        recent_rate = (recent_positive / len(recent)) * 100
        
        previous = self.feedback_history[-window_size*2:-window_size]
        if len(previous) >= window_size:
            previous_positive = sum(1 for f in previous if f["rating"] == "👍")
            previous_rate = (previous_positive / len(previous)) * 100
            
            if recent_rate > previous_rate:
                trend = "improving"
            elif recent_rate < previous_rate:
                trend = "declining"
            else:
                trend = "stable"
        else:
            previous_rate = 0.0
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "recent_rate": recent_rate,
            "previous_rate": previous_rate,
            "improvement": recent_rate - previous_rate
        }
    
    def get_recent_feedback_context(self, n: int = 3) -> str:
        """Get context from recent feedback to guide current response."""
        if len(self.feedback_history) < 1:
            return ""
        
        # Get last N feedback entries
        recent = self.feedback_history[-n:]
        negative_recent = [f for f in recent if f["rating"] == "👎"]
        
        if not negative_recent:
            return ""
        
        # Build guidance from negative feedback
        guidance_parts = []
        
        for feedback in negative_recent:
            if feedback.get("comment"):
                guidance_parts.append(f"- User feedback: {feedback['comment']}")
        
        if not guidance_parts:
            return ""
        
        return f"\nRECENT USER FEEDBACK (adjust your response accordingly):\n" + "\n".join(guidance_parts[-2:])  # Last 2 comments
    
    def save_feedback(self):
        """Save feedback to disk."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump({
                "feedback_history": self.feedback_history,
                "improvement_keywords": self.improvement_keywords
            }, f, indent=2)
    
    def load_feedback(self):
        """Load feedback from disk."""
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                data = json.load(f)
                self.feedback_history = data.get("feedback_history", [])
                self.improvement_keywords = data.get("improvement_keywords", {})
                
                for entry in self.feedback_history:
                    if entry["rating"] == "👍":
                        self.positive_feedback.append(entry)
                    else:
                        self.negative_feedback.append(entry)
