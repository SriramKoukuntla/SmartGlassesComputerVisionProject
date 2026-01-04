"""Main orchestrator that coordinates all layers."""
import time
from typing import Optional
from app.config import config
from app.layers.layer1_sensor import SensorIngest, Frame
from app.layers.layer2_perception import PerceptionModels, PerceptionOutput
from app.layers.layer2_5_risk import RiskPrioritization
from app.layers.layer3_reasoning import SceneReasoning
from app.layers.layer4_memory import MemoryEventGating
from app.layers.layer5_output import OutputInteraction, OutputMode


class SmartGlassesOrchestrator:
    """Main orchestrator for the smart glasses system."""
    
    def __init__(self):
        """Initialize all layers."""
        # Layer 1: Sensor Ingest
        self.sensor = SensorIngest()
        
        # Layer 2: Perception Models
        self.perception = PerceptionModels()
        
        # Layer 2.5: Risk & Prioritization
        self.risk_prioritization = RiskPrioritization()
        
        # Layer 3: Scene Reasoning
        self.scene_reasoning = SceneReasoning()
        
        # Layer 4: Memory & Event Gating
        self.memory_gating = MemoryEventGating()
        
        # Layer 5: Output & Interaction
        self.output = OutputInteraction()
        
        self.is_running = False
        self.frame_count = 0
    
    def start(self):
        """Start the system."""
        print("Starting Smart Glasses System...")
        
        # Start sensor
        if not self.sensor.start():
            raise RuntimeError("Failed to start camera")
        
        # Start output
        self.output.start()
        
        self.is_running = True
        print("System started successfully.")
    
    def stop(self):
        """Stop the system."""
        print("Stopping Smart Glasses System...")
        self.is_running = False
        
        # Stop sensor
        self.sensor.stop()
        
        # Stop output
        self.output.stop()
        
        print("System stopped.")
    
    def process_frame(self, frame: Frame) -> Optional[dict]:
        """Process a single frame through all layers.
        
        Args:
            frame: Input frame with timestamp
            
        Returns:
            Processing results dictionary
        """
        self.frame_count += 1
        start_time = time.time()
        
        # Layer 2: Perception Models
        perception_output = self.perception.process_frame(frame.image)
        
        # Layer 2.5: Risk & Prioritization
        risk_events = self.risk_prioritization.prioritize_events(
            perception_output,
            walkable_path=perception_output.segmentation
        )
        
        # Layer 3: Scene Reasoning (build scene graph)
        scene_graph = self.scene_reasoning.build_scene_graph(risk_events, perception_output)
        
        # Layer 3: Generate description (optional, can be done less frequently)
        llm_response = None
        if self.frame_count % 10 == 0:  # Every 10 frames
            mode_str = "navigation" if config.navigation_mode else "description"
            llm_response = self.scene_reasoning.generate_description(
                scene_graph,
                mode=mode_str
            )
        
        # Layer 4: Memory & Event Gating
        gated_events = self.memory_gating.gate_events(risk_events)
        
        # Layer 5: Output (speak gated events)
        for gated_event in gated_events:
            if gated_event.should_speak:
                self.output.speak_gated_event(gated_event, llm_response)
        
        processing_time = time.time() - start_time
        
        # Return results for debugging/monitoring
        return {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "processing_time": processing_time,
            "detections": len(perception_output.detections),
            "tracks": len(perception_output.tracks),
            "text_regions": len(perception_output.text_regions),
            "risk_events": len(risk_events),
            "gated_events": len([e for e in gated_events if e.should_speak]),
            "fps": 1.0 / processing_time if processing_time > 0 else 0
        }
    
    def run(self):
        """Main processing loop."""
        if not self.is_running:
            self.start()
        
        try:
            while self.is_running:
                # Read frame
                frame = self.sensor.read_frame()
                if frame is None:
                    time.sleep(0.01)  # Small delay if no frame
                    continue
                
                # Process frame
                results = self.process_frame(frame)
                
                # Print status every 30 frames
                if self.frame_count % 30 == 0 and results:
                    print(f"Frame {self.frame_count}: "
                          f"{results['detections']} detections, "
                          f"{results['tracks']} tracks, "
                          f"{results['gated_events']} events to speak, "
                          f"FPS: {results['fps']:.1f}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
    
    def set_output_mode(self, mode: OutputMode):
        """Set output mode (navigation or description)."""
        self.output.set_mode(mode)
    
    def answer_question(self, question: str) -> str:
        """Answer user question about current scene.
        
        Args:
            question: User's question
            
        Returns:
            Answer string
        """
        # Get current frame
        frame = self.sensor.read_frame()
        if frame is None:
            return "No frame available to answer question."
        
        # Process frame
        perception_output = self.perception.process_frame(frame.image)
        risk_events = self.risk_prioritization.prioritize_events(perception_output)
        scene_graph = self.scene_reasoning.build_scene_graph(risk_events)
        
        # Get answer
        llm_response = self.scene_reasoning.answer_question(question, scene_graph)
        return llm_response.description

