from pydantic import BaseModel, Field
from typing import List, Dict, Tuple

class TrackedSubject(BaseModel):
    """A tracked subject with persistent identity"""
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    positions: List[Dict] = Field(default_factory=list)
    total_frames: int = Field(default=0)
    avg_position: Tuple[float, float] = Field(default=(0, 0))
    speaking_segments: List[Dict] = Field(default_factory=list)
    screen_time: float = Field(default=0.0)
    
class SpeakerSegment(BaseModel):
    """A segment where a specific speaker is talking"""
    speaker_id: int
    start: float
    end: float
    confidence: float
    transcript: str = ""    