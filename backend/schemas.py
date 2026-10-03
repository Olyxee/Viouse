from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

Score = float
Role = Literal['hook', 'setup', 'progression', 'climax', 'reaction', 'ending', 'broll']

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Analysis(Strict):
    segment_id: str
    description: str
    people: list[str]
    activity: str
    exercise: str = Field(description='Exercise name only when exercise is visible; otherwise none. Use activity for general creator content.')
    shot_type: Literal['wide', 'medium', 'closeup', 'unknown']
    camera_angle: Literal['front', 'side', 'rear', 'low', 'high', 'unknown']
    camera_motion: Literal['static', 'handheld', 'pan', 'moving', 'unknown']
    visual_quality: float = Field(ge=0, le=1)
    motion_intensity: float = Field(ge=0, le=1)
    effort_intensity: float = Field(ge=0, le=1)
    interestingness: float = Field(ge=0, le=1)
    possible_role: list[Role]
    problems: list[str]
    subject_box: list[float] = Field(min_length=4, max_length=4, description='Normalized x1,y1,x2,y2 enclosing the main subject and relevant objects, whether a person, product, food, animal, or scenery; full frame if uncertain.')

class Shot(Strict):
    segment_id: str
    role: Role
    source_start: float
    source_end: float
    reason: str

class Plan(Strict):
    title: str
    style: Literal['high-energy', 'cinematic']
    target_duration: float
    story: list[Shot] = Field(min_length=1, max_length=50)
    rejected: list[str]

class EventGroup(Strict):
    label: str
    segment_ids: list[str]
    synchronization_confidence: float = Field(ge=0, le=1)

class Workout(Strict):
    title: str
    sequence: list[str]
    important_events: list[EventGroup]

class Critic(Strict):
    hook_strength: float = Field(ge=0, le=1)
    pacing: float = Field(ge=0, le=1)
    story_coherence: float = Field(ge=0, le=1)
    shot_diversity: float = Field(ge=0, le=1)
    visual_quality: float = Field(ge=0, le=1)
    continuity: float = Field(ge=0, le=1)
    repetition: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)
    problems: list[str]
    recommended_changes: list[str]
    revision_confidence: float = Field(ge=0, le=1)

class SpeechPhrase(Strict):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    confidence: float = Field(ge=0,le=1)

class SpeechTranscript(Strict):
    classification: Literal['speech','silence','noise','music','mixed']
    segments: list[SpeechPhrase]
