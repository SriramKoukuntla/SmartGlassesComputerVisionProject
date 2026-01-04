"""Main orchestrator that coordinates all layers."""
import time
from typing import Optional
import numpy as np
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
        """Start the system (without camera - camera handled by frontend)."""
        print("Starting Smart Glasses System...")
        
        # Don't start camera - frontend handles camera input
        # Sensor will be used only for processing frames received from frontend
        
        # Start output
        self.output.start()
        
        self.is_running = True
        print("System started successfully (waiting for frontend input).")
    
    def stop(self):
        """Stop the system."""
        print("Stopping Smart Glasses System...")
        self.is_running = False
        
        # Stop sensor if it was started (shouldn't be, but just in case)
        if self.sensor.is_running:
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
    
    def answer_question(self, question: str, image: Optional[np.ndarray] = None) -> str:
        """Answer user question about current scene.
        
        Args:
            question: User's question
            image: Optional image to process (if None, requires frame from sensor)
            
        Returns:
            Answer string
        """
        # Use provided image or try to get frame from sensor
        if image is not None:
            frame_image = image
        else:
            frame = self.sensor.read_frame()
            if frame is None:
                return "No frame available to answer question."
            frame_image = frame.image
        
        # Process frame
        perception_output = self.perception.process_frame(frame_image)
        risk_events = self.risk_prioritization.prioritize_events(perception_output)
        scene_graph = self.scene_reasoning.build_scene_graph(risk_events)
        
        # Get answer
        llm_response = self.scene_reasoning.answer_question(question, scene_graph)
        return llm_response.description
    
    def process_image(self, image: np.ndarray, frame_id: int = 0) -> dict:
        """Process an image received from frontend.
        
        Args:
            image: Input image (BGR format numpy array)
            frame_id: Optional frame ID
            
        Returns:
            Processing results dictionary with logs
        """
        logs = []
        start_time = time.time()
        
        # Increment frame count
        self.frame_count += 1
        
        timestamp = time.time()
        frame = Frame(image=image, timestamp=timestamp, frame_id=frame_id)
        
        logs.append(f"[{timestamp:.3f}] Processing frame {frame_id} (size: {image.shape[1]}x{image.shape[0]})")
        
        # Layer 2: Perception Models
        perception_start = time.time()
        perception_output = self.perception.process_frame(frame.image)
        perception_time = time.time() - perception_start
        logs.append(f"[{time.time():.3f}] Perception: {len(perception_output.detections)} objects, "
                   f"{len(perception_output.text_regions)} text regions ({perception_time*1000:.1f}ms)")
        
        # Layer 2.5: Risk & Prioritization
        risk_start = time.time()
        risk_events = self.risk_prioritization.prioritize_events(
            perception_output,
            walkable_path=perception_output.segmentation
        )
        risk_time = time.time() - risk_start
        if risk_events:
            logs.append(f"[{time.time():.3f}] Risk: {len(risk_events)} events prioritized ({risk_time*1000:.1f}ms)")
        
        # Layer 3: Scene Reasoning (build scene graph)
        reasoning_start = time.time()
        scene_graph = self.scene_reasoning.build_scene_graph(risk_events, perception_output)
        reasoning_time = time.time() - reasoning_start
        logs.append(f"[{time.time():.3f}] Reasoning: Scene graph built ({reasoning_time*1000:.1f}ms)")
        
        # Layer 3: Generate description (optional, can be done less frequently)
        llm_response = None
        if self.frame_count % 10 == 0:  # Every 10 frames
            mode_str = "navigation" if config.navigation_mode else "description"
            llm_start = time.time()
            llm_response = self.scene_reasoning.generate_description(
                scene_graph,
                mode=mode_str
            )
            llm_time = time.time() - llm_start
            logs.append(f"[{time.time():.3f}] LLM: Description generated ({llm_time*1000:.1f}ms)")
        
        # Layer 4: Memory & Event Gating
        gating_start = time.time()
        gated_events = self.memory_gating.gate_events(risk_events)
        gating_time = time.time() - gating_start
        gated_count = len([e for e in gated_events if e.should_speak])
        if gated_count > 0:
            logs.append(f"[{time.time():.3f}] Gating: {gated_count} events to speak ({gating_time*1000:.1f}ms)")
        
        # Layer 5: Output (speak gated events)
        for gated_event in gated_events:
            if gated_event.should_speak:
                self.output.speak_gated_event(gated_event, llm_response)
        
        processing_time = time.time() - start_time
        
        # Return results with logs and perception output
        results = {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "processing_time": processing_time,
            "detections": len(perception_output.detections),
            "tracks": len(perception_output.tracks),
            "text_regions": len(perception_output.text_regions),
            "risk_events": len(risk_events),
            "gated_events": gated_count,
            "fps": 1.0 / processing_time if processing_time > 0 else 0,
            "logs": logs,
            "perception_output": perception_output  # Include for API to use
        }
        
        logs.append(f"[{time.time():.3f}] Frame {frame_id} complete ({processing_time*1000:.1f}ms, {results['fps']:.1f} FPS)")
        
        return results

