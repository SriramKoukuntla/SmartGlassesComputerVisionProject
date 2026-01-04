"""Layer 5: Output & Interaction - TTS with priority and interruptibility."""
import queue
import threading
from typing import Optional, Callable
from enum import Enum
from dataclasses import dataclass
from app.config import config
from app.layers.layer4_memory import GatedEvent
from app.layers.layer3_reasoning import LLMResponse


class Priority(Enum):
    """Message priority levels."""
    URGENT = 0  # Hazards - interrupts everything
    HIGH = 1    # Important navigation info
    NORMAL = 2  # General descriptions
    LOW = 3     # Background info


class OutputMode(Enum):
    """Output modes."""
    NAVIGATION = "navigation"  # Short commands
    DESCRIPTION = "description"  # Rich context


@dataclass
class SpeechMessage:
    """Message to be spoken."""
    text: str
    priority: Priority
    interruptible: bool = True  # Can be interrupted by higher priority
    mode: OutputMode = OutputMode.NAVIGATION


class OutputInteraction:
    """Handles TTS output with priority and interruptibility."""
    
    def __init__(self):
        """Initialize output interaction."""
        self.mode = OutputMode.NAVIGATION if config.navigation_mode else OutputMode.DESCRIPTION
        self.tts_provider = config.tts_provider
        self.tts_rate = config.tts_rate
        self.tts_volume = config.tts_volume
        
        # Priority queue for messages
        self.message_queue = queue.PriorityQueue()
        self._message_counter = 0
        self.current_message: Optional[SpeechMessage] = None
        self.is_speaking = False
        self.speech_thread: Optional[threading.Thread] = None
        self.stop_speaking = threading.Event()
        
        # Initialize TTS engine
        self.tts_engine = None
        self._init_tts()
    
    def _init_tts(self):
        """Initialize TTS engine based on provider."""
        if self.tts_provider == "pyttsx3":
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', self.tts_rate)
                self.tts_engine.setProperty('volume', self.tts_volume)
            except ImportError:
                print("Warning: pyttsx3 not installed. Install with: pip install pyttsx3")
        elif self.tts_provider == "gtts":
            try:
                from gtts import gTTS
                import pygame
                self.tts_engine = "gtts"  # Flag for gTTS
            except ImportError:
                print("Warning: gtts or pygame not installed.")
        elif self.tts_provider == "azure":
            # Azure Cognitive Services TTS
            print("Warning: Azure TTS not fully implemented.")
        else:
            print(f"Warning: Unknown TTS provider: {self.tts_provider}")
    
    def set_mode(self, mode: OutputMode):
        """Set output mode."""
        self.mode = mode
    
    def speak(self, text: str, priority: Priority = Priority.NORMAL, interruptible: bool = True):
        """Add message to speech queue.
        
        Args:
            text: Text to speak
            priority: Message priority
            interruptible: Whether message can be interrupted
        """
        message = SpeechMessage(
            text=text,
            priority=priority,
            interruptible=interruptible,
            mode=self.mode
        )
        
        # Add to priority queue (lower priority number = higher priority)
        self.message_queue.put((priority.value, message))
        
        # If urgent and interruptible, stop current speech
        if priority == Priority.URGENT and interruptible and self.is_speaking:
            self.stop_speaking.set()
    
    def speak_gated_event(self, gated_event: GatedEvent, llm_response: Optional[LLMResponse] = None):
        """Speak a gated event with optional LLM-generated description.
        
        Args:
            gated_event: Gated event to speak
            llm_response: Optional LLM-generated description
        """
        if not gated_event.should_speak:
            return
        
        # Determine priority
        if gated_event.event.event_type == "obstacle":
            priority = Priority.URGENT
        elif gated_event.event.priority > 10:
            priority = Priority.HIGH
        elif gated_event.event.priority > 5:
            priority = Priority.NORMAL
        else:
            priority = Priority.LOW
        
        # Get text to speak
        if llm_response:
            text = llm_response.description
        else:
            # Fallback to event description
            text = gated_event.event.description
        
        # Format based on mode
        if self.mode == OutputMode.NAVIGATION:
            # Short, command-like
            if gated_event.event.event_type == "obstacle":
                text = f"Stop. {text}"
            elif gated_event.event.priority > 10:
                text = f"Warning. {text}"
        else:
            # Rich description
            text = f"{text}. Location: {gated_event.event.location}"
        
        self.speak(text, priority=priority, interruptible=True)
    
    def _speak_text(self, text: str):
        """Actually speak text using TTS engine."""
        if self.tts_provider == "pyttsx3" and self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"Error speaking with pyttsx3: {e}")
        elif self.tts_provider == "gtts":
            try:
                from gtts import gTTS
                import pygame
                import io
                import tempfile
                import os
                
                # Generate speech
                tts = gTTS(text=text, lang='en')
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tts.save(tmp_file.name)
                    tmp_file_path = tmp_file.name
                
                # Play audio
                pygame.mixer.init()
                pygame.mixer.music.load(tmp_file_path)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    if self.stop_speaking.is_set():
                        pygame.mixer.music.stop()
                        break
                    pygame.time.Clock().tick(10)
                
                # Cleanup
                pygame.mixer.quit()
                os.unlink(tmp_file_path)
            except Exception as e:
                print(f"Error speaking with gTTS: {e}")
        else:
            # Fallback: print to console
            print(f"[TTS] {text}")
    
    def _speech_worker(self):
        """Worker thread for processing speech queue."""
        while True:
            try:
                # Get message from queue (blocking)
                priority_value, message = self.message_queue.get(timeout=1.0)
                
                # Check if we should interrupt current speech
                if self.is_speaking and message.priority.value < Priority.NORMAL.value:
                    self.stop_speaking.set()
                
                # Wait for current speech to stop if needed
                if self.is_speaking:
                    self.stop_speaking.wait(timeout=2.0)
                    self.stop_speaking.clear()
                
                # Speak the message
                self.current_message = message
                self.is_speaking = True
                self._speak_text(message.text)
                self.is_speaking = False
                self.current_message = None
                
                # Mark task as done
                self.message_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in speech worker: {e}")
                self.is_speaking = False
    
    def start(self):
        """Start speech worker thread."""
        if self.speech_thread is None or not self.speech_thread.is_alive():
            self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.speech_thread.start()
    
    def stop(self):
        """Stop speech worker."""
        self.stop_speaking.set()
        if self.speech_thread and self.speech_thread.is_alive():
            # Wait for thread to finish (with timeout)
            self.speech_thread.join(timeout=2.0)
    
    def play_haptic_alert(self, intensity: float = 0.5):
        """Play haptic alert (placeholder for hardware integration).
        
        Args:
            intensity: Alert intensity (0.0 to 1.0)
        """
        # Placeholder - would integrate with haptic hardware
        print(f"[Haptic] Alert intensity: {intensity}")
    
    def play_tone(self, frequency: int = 440, duration: float = 0.1):
        """Play audio tone (placeholder for hardware integration).
        
        Args:
            frequency: Tone frequency in Hz
            duration: Duration in seconds
        """
        # Placeholder - would generate and play tone
        print(f"[Tone] {frequency}Hz for {duration}s")

