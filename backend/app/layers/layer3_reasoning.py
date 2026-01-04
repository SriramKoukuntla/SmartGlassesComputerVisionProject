"""Layer 3: Scene Reasoning + Language (LLM)."""
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from app.config import config
from app.layers.layer2_5_risk import RiskEvent


@dataclass
class SceneGraph:
    """Structured scene representation."""
    objects: List[Dict]  # Detected objects with properties
    text_regions: List[Dict]  # OCR text with locations
    spatial_relations: List[Dict]  # Relationships between objects
    walkable_area: Optional[Dict] = None  # Information about walkable path
    depth_info: Optional[Dict] = None  # Depth statistics


@dataclass
class LLMResponse:
    """LLM-generated response."""
    description: str
    confidence: float
    reasoning: Optional[str] = None


class SceneReasoning:
    """Scene reasoning using LLM for natural language generation."""
    
    def __init__(self):
        """Initialize scene reasoning."""
        self.llm_provider = config.llm_provider
        self.llm_model = config.llm_model
        self.llm_api_key = config.llm_api_key
        self.llm_temperature = config.llm_temperature
        self.llm_client = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM client based on provider."""
        if not self.llm_api_key:
            print("Warning: No LLM API key provided. LLM features will be disabled.")
            return
        
        if self.llm_provider == "openai":
            try:
                import openai
                self.llm_client = openai.OpenAI(api_key=self.llm_api_key)
            except ImportError:
                print("Warning: openai not installed. Install with: pip install openai")
        elif self.llm_provider == "anthropic":
            try:
                import anthropic
                self.llm_client = anthropic.Anthropic(api_key=self.llm_api_key)
            except ImportError:
                print("Warning: anthropic not installed. Install with: pip install anthropic")
        elif self.llm_provider == "local":
            # For local models (e.g., Ollama, llama.cpp)
            print("Warning: Local LLM support not fully implemented.")
        else:
            print(f"Warning: Unknown LLM provider: {self.llm_provider}")
    
    def build_scene_graph(
        self,
        risk_events: List[RiskEvent],
        perception_output: Optional[Dict] = None
    ) -> SceneGraph:
        """Build structured scene graph from perception outputs.
        
        Args:
            risk_events: Prioritized risk events
            perception_output: Raw perception output (optional)
            
        Returns:
            SceneGraph structure
        """
        objects = []
        text_regions = []
        
        for event in risk_events:
            if event.event_type == "object":
                obj_data = {
                    "type": event.metadata.get("class_name", "unknown"),
                    "location": event.location,
                    "distance": event.distance,
                    "priority": event.priority,
                    "track_id": event.metadata.get("track_id"),
                    "velocity": event.velocity
                }
                objects.append(obj_data)
            elif event.event_type == "text":
                text_data = {
                    "text": event.metadata.get("text", ""),
                    "location": event.location,
                    "priority": event.priority
                }
                text_regions.append(text_data)
        
        # Compute spatial relations (simplified)
        spatial_relations = []
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                if obj1["location"] and obj2["location"]:
                    # Simple relative position
                    x1, y1 = obj1["location"]
                    x2, y2 = obj2["location"]
                    if abs(x1 - x2) < 100:  # Roughly same column
                        relation = "aligned_vertically"
                    elif abs(y1 - y2) < 100:  # Roughly same row
                        relation = "aligned_horizontally"
                    else:
                        relation = "separate"
                    
                    spatial_relations.append({
                        "object1": obj1["type"],
                        "object2": obj2["type"],
                        "relation": relation
                    })
        
        return SceneGraph(
            objects=objects,
            text_regions=text_regions,
            spatial_relations=spatial_relations
        )
    
    def generate_description(
        self,
        scene_graph: SceneGraph,
        mode: str = "navigation"  # "navigation" or "description"
    ) -> LLMResponse:
        """Generate natural language description using LLM.
        
        Args:
            scene_graph: Structured scene representation
            mode: "navigation" for short commands, "description" for rich context
            
        Returns:
            LLMResponse with description
        """
        if not self.llm_client:
            # Fallback to rule-based description
            return self._fallback_description(scene_graph, mode)
        
        # Build prompt
        prompt = self._build_prompt(scene_graph, mode)
        
        try:
            if self.llm_provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt(mode)
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.llm_temperature,
                    max_tokens=200 if mode == "navigation" else 500
                )
                description = response.choices[0].message.content
                return LLMResponse(description=description, confidence=0.9)
            
            elif self.llm_provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=200 if mode == "navigation" else 500,
                    system=self._get_system_prompt(mode),
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.llm_temperature
                )
                description = response.content[0].text
                return LLMResponse(description=description, confidence=0.9)
        
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return self._fallback_description(scene_graph, mode)
    
    def _get_system_prompt(self, mode: str) -> str:
        """Get system prompt for LLM."""
        if mode == "navigation":
            return """You are a navigation assistant for visually impaired individuals. 
            Generate short, clear commands. Be concise and action-oriented.
            Examples: "Stop. Obstacle ahead.", "Turn right. Door on your right.", "Continue straight."
            Only describe what is actually detected. Do not speculate."""
        else:
            return """You are a visual assistant for visually impaired individuals.
            Generate clear, descriptive explanations of the environment.
            Be specific about locations (left, right, ahead, behind).
            Use natural language appropriate for navigation.
            Only describe what is actually detected. Do not speculate."""
    
    def _build_prompt(self, scene_graph: SceneGraph, mode: str) -> str:
        """Build prompt for LLM."""
        prompt_parts = ["Current scene:"]
        
        if scene_graph.objects:
            prompt_parts.append("\nObjects detected:")
            for obj in scene_graph.objects[:5]:  # Top 5 objects
                desc = f"- {obj['type']}"
                if obj.get("distance"):
                    desc += f" (distance: {obj['distance']:.2f})"
                if obj.get("location"):
                    x, y = obj["location"]
                    desc += f" at position ({x:.0f}, {y:.0f})"
                prompt_parts.append(desc)
        
        if scene_graph.text_regions:
            prompt_parts.append("\nText detected:")
            for text in scene_graph.text_regions[:3]:  # Top 3 text regions
                prompt_parts.append(f"- \"{text['text']}\"")
        
        if scene_graph.spatial_relations:
            prompt_parts.append("\nSpatial relationships:")
            for rel in scene_graph.spatial_relations[:3]:
                prompt_parts.append(f"- {rel['object1']} and {rel['object2']} are {rel['relation']}")
        
        prompt_parts.append("\nGenerate a description appropriate for visually impaired navigation.")
        
        return "\n".join(prompt_parts)
    
    def _fallback_description(self, scene_graph: SceneGraph, mode: str) -> LLMResponse:
        """Fallback rule-based description when LLM is unavailable."""
        parts = []
        
        if mode == "navigation":
            # Short commands
            if scene_graph.objects:
                top_obj = scene_graph.objects[0]
                if top_obj.get("priority", 0) > 10:
                    parts.append(f"Stop. {top_obj['type']} ahead.")
                else:
                    parts.append(f"{top_obj['type']} detected.")
        else:
            # Rich descriptions
            if scene_graph.objects:
                parts.append("Objects in view:")
                for obj in scene_graph.objects[:3]:
                    parts.append(f"{obj['type']}")
                    if obj.get("location"):
                        x, y = obj["location"]
                        if x < 200:
                            parts[-1] += " on the left"
                        elif x > 400:
                            parts[-1] += " on the right"
                        else:
                            parts[-1] += " ahead"
            
            if scene_graph.text_regions:
                parts.append("Text visible:")
                for text in scene_graph.text_regions[:2]:
                    parts.append(f'"{text["text"]}"')
        
        description = ". ".join(parts) if parts else "No significant objects detected."
        return LLMResponse(description=description, confidence=0.7)
    
    def answer_question(self, question: str, scene_graph: SceneGraph) -> LLMResponse:
        """Answer user question about the scene.
        
        Args:
            question: User's question
            scene_graph: Current scene graph
            
        Returns:
            LLMResponse with answer
        """
        if not self.llm_client:
            return LLMResponse(
                description="LLM not available. Cannot answer questions.",
                confidence=0.0
            )
        
        prompt = f"""Based on the current scene, answer this question: {question}

Scene information:
{json.dumps({
    "objects": scene_graph.objects[:5],
    "text_regions": scene_graph.text_regions[:3]
}, indent=2)}

Answer concisely based only on the detected information."""
        
        try:
            if self.llm_provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a visual assistant. Answer questions based only on the provided scene information. Do not speculate."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.llm_temperature,
                    max_tokens=150
                )
                answer = response.choices[0].message.content
                return LLMResponse(description=answer, confidence=0.8)
        
        except Exception as e:
            print(f"Error answering question: {e}")
            return LLMResponse(
                description="Unable to answer question at this time.",
                confidence=0.0
            )

