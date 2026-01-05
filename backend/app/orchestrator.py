"""Main orchestrator that coordinates all layers."""
import time
from typing import Optional, Dict, List
import numpy as np
from app.config import config
from app.layers.layer1_sensor import SensorIngest, Frame
from app.layers.layer2_perception import PerceptionModels, PerceptionOutput
from app.layers.layer2_5_risk import RiskPrioritization
from app.layers.layer3_reasoning import SceneReasoning
from app.layers.layer4_memory import MemoryEventGating
from app.layers.layer5_output import OutputInteraction, OutputMode

# Constants
LLM_DESCRIPTION_INTERVAL = 10  # Generate LLM description every N frames
STATUS_PRINT_INTERVAL = 30  # Print status every N frames


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
    
    def _process_perception(self, image: np.ndarray) -> PerceptionOutput:
        """Process image through perception models.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            PerceptionOutput with all model results
        """
        return self.perception.process_frame(image)
    
    def _process_risk_prioritization(
        self, 
        perception_output: PerceptionOutput
    ) -> List:
        """Process risk prioritization.
        
        Args:
            perception_output: Output from perception models
            
        Returns:
            List of prioritized risk events
        """
        return self.risk_prioritization.prioritize_events(
            perception_output,
            walkable_path=perception_output.segmentation
        )
    
    def _process_reasoning(
        self, 
        risk_events: List, 
        perception_output: PerceptionOutput,
        generate_llm: bool = False
    ) -> tuple:
        """Process scene reasoning and optionally generate LLM description.
        
        Args:
            risk_events: Prioritized risk events
            perception_output: Perception output
            generate_llm: Whether to generate LLM description
            
        Returns:
            Tuple of (scene_graph, llm_response)
        """
        scene_graph = self.scene_reasoning.build_scene_graph(risk_events, perception_output)
        
        llm_response = None
        if generate_llm:
            mode_str = "navigation" if config.navigation_mode else "description"
            llm_response = self.scene_reasoning.generate_description(
                scene_graph,
                mode=mode_str
            )
        
        return scene_graph, llm_response
    
    def _process_gating_and_output(
        self, 
        risk_events: List, 
        llm_response: Optional = None
    ) -> List:
        """Process event gating and output.
        
        Args:
            risk_events: Prioritized risk events
            llm_response: Optional LLM response
            
        Returns:
            List of gated events
        """
        gated_events = self.memory_gating.gate_events(risk_events)
        
        # Output gated events
        for gated_event in gated_events:
            if gated_event.should_speak:
                self.output.speak_gated_event(gated_event, llm_response)
        
        return gated_events
    
    def _build_results_dict(
        self,
        frame: Frame,
        perception_output: PerceptionOutput,
        risk_events: List,
        gated_events: List,
        processing_time: float,
        logs: Optional[List[str]] = None
    ) -> Dict:
        """Build results dictionary from processing outputs.
        
        Args:
            frame: Processed frame
            perception_output: Perception results
            risk_events: Risk events
            gated_events: Gated events
            processing_time: Total processing time
            logs: Optional list of log messages
            
        Returns:
            Results dictionary
        """
        gated_count = len([e for e in gated_events if e.should_speak])
        results = {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "processing_time": processing_time,
            "detections": len(perception_output.detections),
            "tracks": len(perception_output.tracks),
            "text_regions": len(perception_output.text_regions),
            "risk_events": len(risk_events),
            "gated_events": gated_count,
            "fps": 1.0 / processing_time if processing_time > 0 else 0
        }
        
        if logs is not None:
            results["logs"] = logs
            results["perception_output"] = perception_output
        
        return results
    
    def process_frame(self, frame: Frame) -> Optional[Dict]:
        """Process a single frame through all layers.
        
        Args:
            frame: Input frame with timestamp
            
        Returns:
            Processing results dictionary
        """
        self.frame_count += 1
        start_time = time.time()
        
        # Process through all layers
        perception_output = self._process_perception(frame.image)
        risk_events = self._process_risk_prioritization(perception_output)
        
        # Generate LLM description periodically
        should_generate_llm = self.frame_count % LLM_DESCRIPTION_INTERVAL == 0
        scene_graph, llm_response = self._process_reasoning(
            risk_events, 
            perception_output, 
            generate_llm=should_generate_llm
        )
        
        gated_events = self._process_gating_and_output(risk_events, llm_response)
        
        processing_time = time.time() - start_time
        
        return self._build_results_dict(
            frame, perception_output, risk_events, gated_events, processing_time
        )
    
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
                
                # Print status periodically
                if self.frame_count % STATUS_PRINT_INTERVAL == 0 and results:
                    print(
                        f"Frame {self.frame_count}: "
                        f"{results['detections']} detections, "
                        f"{results['tracks']} tracks, "
                        f"{results['gated_events']} events to speak, "
                        f"FPS: {results['fps']:.1f}"
                    )
        
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
        
        # Process frame through layers
        perception_output = self._process_perception(frame_image)
        risk_events = self._process_risk_prioritization(perception_output)
        scene_graph, _ = self._process_reasoning(risk_events, perception_output, generate_llm=False)
        
        # Get answer
        llm_response = self.scene_reasoning.answer_question(question, scene_graph)
        return llm_response.description
    
    def process_image(self, image: np.ndarray, frame_id: int = 0) -> Dict:
        """Process an image received from frontend.
        
        Args:
            image: Input image (BGR format numpy array)
            frame_id: Optional frame ID
            
        Returns:
            Processing results dictionary with logs
        """
        logs: List[str] = []
        start_time = time.time()
        
        # Increment frame count
        self.frame_count += 1
        
        timestamp = time.time()
        frame = Frame(image=image, timestamp=timestamp, frame_id=frame_id)
        
        logs.append(
            f"[{timestamp:.3f}] Processing frame {frame_id} "
            f"(size: {image.shape[1]}x{image.shape[0]})"
        )
        
        # Process through all layers with timing
        perception_start = time.time()
        perception_output = self._process_perception(frame.image)
        perception_time = time.time() - perception_start
        logs.append(
            f"[{time.time():.3f}] Perception: {len(perception_output.detections)} objects, "
            f"{len(perception_output.text_regions)} text regions "
            f"({perception_time*1000:.1f}ms)"
        )
        
        risk_start = time.time()
        risk_events = self._process_risk_prioritization(perception_output)
        risk_time = time.time() - risk_start
        if risk_events:
            logs.append(
                f"[{time.time():.3f}] Risk: {len(risk_events)} events prioritized "
                f"({risk_time*1000:.1f}ms)"
            )
        
        reasoning_start = time.time()
        should_generate_llm = self.frame_count % LLM_DESCRIPTION_INTERVAL == 0
        scene_graph, llm_response = self._process_reasoning(
            risk_events, 
            perception_output, 
            generate_llm=should_generate_llm
        )
        reasoning_time = time.time() - reasoning_start
        logs.append(
            f"[{time.time():.3f}] Reasoning: Scene graph built "
            f"({reasoning_time*1000:.1f}ms)"
        )
        
        if llm_response:
            logs.append(
                f"[{time.time():.3f}] LLM: Description generated"
            )
        
        gating_start = time.time()
        gated_events = self._process_gating_and_output(risk_events, llm_response)
        gating_time = time.time() - gating_start
        gated_count = len([e for e in gated_events if e.should_speak])
        if gated_count > 0:
            logs.append(
                f"[{time.time():.3f}] Gating: {gated_count} events to speak "
                f"({gating_time*1000:.1f}ms)"
            )
        
        processing_time = time.time() - start_time
        
        # Build results with logs
        results = self._build_results_dict(
            frame, perception_output, risk_events, gated_events, 
            processing_time, logs=logs
        )
        
        logs.append(
            f"[{time.time():.3f}] Frame {frame_id} complete "
            f"({processing_time*1000:.1f}ms, {results['fps']:.1f} FPS)"
        )
        
        return results

