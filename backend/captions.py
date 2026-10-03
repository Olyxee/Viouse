"""Creator captions: validated timing, ASS animation and local person segmentation."""
import re
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, model_validator
from .config import ROOT

MODEL_PATH = ROOT/'.tools/models/human_segmentation.onnx'

class CaptionLine(BaseModel):
    start: float = Field(ge=0,allow_inf_nan=False)
    end: float = Field(gt=0,allow_inf_nan=False)
    text: str = Field(min_length=1,max_length=160)

    @model_validator(mode='after')
    def valid(self):
        if self.end-self.start<.15: raise ValueError('Each subtitle needs at least 0.15 seconds')
        if not self.text.strip(): raise ValueError('Enter subtitle text')
        return self

class CaptionSettings(BaseModel):
    mode: Literal['off','auto','custom'] = 'auto'
    style: Literal['pop','karaoke','comic','behind'] = 'pop'
    position: Literal['top','center','bottom'] = 'bottom'
    color: str = Field(default='#DFFF52',pattern=r'^#[0-9a-fA-F]{6}$')
    lines: list[CaptionLine] = Field(default_factory=list,max_length=100)

    @model_validator(mode='after')
    def valid(self):
        if self.mode=='custom' and not self.lines: raise ValueError('Add at least one subtitle line')
        for left,right in zip(self.lines,self.lines[1:]):
            if right.start<left.end: raise ValueError('Subtitle lines must be ordered and cannot overlap')
        return self

def checked_settings(value, duration=None):
    settings=CaptionSettings.model_validate(value or {})
    if duration is not None and settings.mode=='custom' and any(line.end>duration+.035 for line in settings.lines):
        raise ValueError(f'Subtitles must end within the {duration:.2f}-second video')
    if settings.mode!='off' and settings.style=='behind' and not MODEL_PATH.is_file():
        raise ValueError('Behind-person subtitles need the local mask model. Run python -m scripts.setup_caption_model.')
    return settings

def selected_events(settings, speech, duration):
    settings=checked_settings(settings,duration)
    if settings.mode=='off': return [],settings
    return ([line.model_dump() for line in settings.lines] if settings.mode=='custom' else speech),settings

def ass_time(t):
    cs=round(t*100)
    return f'{cs//360000}:{cs//6000%60:02}:{cs//100%60:02}.{cs%100:02}'

def safe_text(text):
    return re.sub(r'[{}\\\r\n]', ' ',text).strip()

def write_captions(events,path,settings=None):
    s=CaptionSettings.model_validate(settings or {})
    color='&H00'+s.color[5:7]+s.color[3:5]+s.color[1:3]
    y={'top':330,'center':920,'bottom':1500}[s.position]
    if s.style=='behind': y=680
    font=110 if s.style=='behind' else 82
    header=f'''[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font},{color},&H00FFFFFF,&H00131313,&H80000000,-1,0,0,0,100,100,0,0,1,5,3,5,80,80,120,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    lines=[]
    for event in events:
        words=safe_text(event['text']).split()
        # Small readable bursts instead of long, static subtitle sentences.
        groups=[words[i:i+4] for i in range(0,len(words),4)]
        total=max(1,sum(len(word) for word in words))
        cursor=event['start']
        for index,group in enumerate(groups):
            share=sum(len(word) for word in group)/total
            end=event['end'] if index==len(groups)-1 else cursor+(event['end']-event['start'])*share
            span=max(1,round((end-cursor)*100))
            animation=r'\fscx75\fscy75\t(0,110,\fscx110\fscy110)\t(110,190,\fscx100\fscy100)\fad(50,80)'
            if s.style=='comic': animation=r'\frz-5\fscx85\fscy85\t(0,140,\fscx105\fscy105\frz3)\t(140,260,\frz-2)\bord7\shad6\fad(30,70)'
            if s.style=='behind': animation=r'\fscx90\fscy90\t(0,250,\fscx100\fscy100)\bord2\shad0\fad(100,100)'
            text=' '.join(group).upper()
            if s.style=='karaoke':
                weights=[max(1,len(word)) for word in group];budget=sum(weights)
                boundaries=[round(span*sum(weights[:i])/budget) for i in range(len(group)+1)]
                text=' '.join('{\\kf'+str(boundaries[i+1]-boundaries[i])+'}'+word.upper() for i,word in enumerate(group))
            # Shrink long bursts while retaining the selected look.
            size=min(font,int(900/max(1,max(len(word) for word in group))/.65))
            text=f'{{\\an5\\pos(540,{y})\\fs{size}{animation}}}'+text
            lines.append(f'Dialogue: 0,{ass_time(cursor)},{ass_time(end)},Default,,0,0,0,,{text}')
            cursor=end
    Path(path).write_text(header+'\n'.join(lines),encoding='utf-8')

def person_mask(video,output):
    """White restores the person over the lettering; black keeps the lettering."""
    import cv2
    import numpy as np
    if not MODEL_PATH.is_file(): raise ValueError('Local person mask model is missing')
    net=cv2.dnn.readNet(str(MODEL_PATH))
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise ValueError('Could not read video for behind-person captions')
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),30,(270,480))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Could not create the person mask video')
    count=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            # OpenCV Zoo PP-HumanSeg normalization: RGB, [-1,1], 192 square.
            blob=cv2.dnn.blobFromImage(frame,1/127.5,(192,192),(127.5,127.5,127.5),swapRB=True)
            net.setInput(blob)
            scores=net.forward()[0]
            mask=(scores[1]>scores[0]).astype(np.uint8)*255
            mask=cv2.resize(mask,(270,480),interpolation=cv2.INTER_LINEAR)
            mask=cv2.GaussianBlur(mask,(5,5),0)
            writer.write(cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR));count+=1
    finally:
        cap.release();writer.release()
    if not count: raise ValueError('No frames were available for the person mask')
    return count
