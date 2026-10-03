import {useId} from 'react';

export type CaptionSettings={mode:'off'|'auto'|'custom';style:'pop'|'karaoke'|'comic'|'behind';position:'top'|'center'|'bottom';color:string;lines:{start:number;end:number;text:string}[]};
export const defaultCaptions:CaptionSettings={mode:'auto',style:'pop',position:'bottom',color:'#DFFF52',lines:[]};
const looks=[['pop','Pop','Big words. Quick bounce.'],['karaoke','Karaoke','Color sweeps as words land.'],['comic','Comic','Tilted type. Heavy outlines.'],['behind','Behind person','Text sits behind detected people.']] as const;

export function CaptionStudio({value,onChange,duration,disabled=false,onExport,compact=false}:{value:CaptionSettings;onChange:(value:CaptionSettings)=>void;duration:number;disabled?:boolean;onExport?:()=>void;compact?:boolean}){
  const id=useId();
  const update=(patch:Partial<CaptionSettings>)=>onChange({...value,...patch});
  const problem=value.mode==='custom'?(value.lines.length===0?'Add a line to get started.':value.lines.some((line,i)=>!line.text.trim()||line.end-line.start<.15||line.start<0||line.end>duration+.035||(i>0&&line.start<value.lines[i-1].end))?'Check timing: lines must be ordered, not overlap, and fit inside the video.':''):'';
  const add=()=>{const start=value.lines.at(-1)?.end||0;update({lines:[...value.lines,{start,end:Math.min(duration,start+3),text:''}]})};
  return <section className={`caption-studio ${compact?'compact':''}`} aria-labelledby={id+'-title'}>
    <div className="caption-heading"><div>{!compact&&<span className="eyebrow">STYLE YOUR WORDS</span>}<h2 id={id+'-title'}>{compact?'Subtitles':'Subtitle style'}</h2></div><span className="caption-badge">Aa</span></div>
    <fieldset disabled={disabled} className="caption-controls"><legend className="sr-only">Subtitle settings</legend>
      <label htmlFor={id+'-source'}>Text source</label><select id={id+'-source'} value={value.mode} onChange={e=>update({mode:e.target.value as CaptionSettings['mode']})}><option value="auto">Detected speech</option><option value="custom">Write my own</option><option value="off">No subtitles</option></select>
      {value.mode!=='off'&&<>
        <div className="caption-looks">{looks.map(([style,title,description])=><button type="button" key={style} aria-pressed={value.style===style} className={value.style===style?'selected':''} onClick={()=>update({style})}><strong>{title}</strong><small>{description}</small></button>)}</div>
        <div className={`caption-preview ${value.style}`} aria-label="Illustrated style preview; export uses your footage" style={{'--caption-color':value.color} as React.CSSProperties}><span className="preview-scene"/><span className="preview-word">{value.lines[0]?.text||'MAKE IT POP'}</span>{value.style==='behind'&&<span className="preview-person"/>}<small>STYLE PREVIEW</small></div>
        <div className="caption-placement"><label htmlFor={id+'-color'}>Accent <input id={id+'-color'} type="color" value={value.color} onChange={e=>update({color:e.target.value})}/></label>{value.style!=='behind'&&<label htmlFor={id+'-position'}>Position<select id={id+'-position'} value={value.position} onChange={e=>update({position:e.target.value as CaptionSettings['position']})}><option value="top">Top</option><option value="center">Center</option><option value="bottom">Bottom</option></select></label>}</div>
        {value.style==='behind'&&<p className="caption-note">People appear in front of the text. Best with a clear subject; hair and fast movement can soften the edge. Without a detected person, text stays in front of the scene.</p>}
        {value.mode==='auto'?<p className="caption-note">Uses detected speech. No speech? Choose “Write my own.” Word highlighting uses estimated timing.</p>:<>
          <p className="caption-note">Time each line against the finished video ({duration.toFixed(1)}s). Long lines play in short animated bursts.</p>
          <div className="caption-lines">{value.lines.map((line,i)=><div className="caption-line" key={i}>
            <label className="sr-only" htmlFor={`${id}-line-${i}`}>Subtitle {i+1}</label><textarea id={`${id}-line-${i}`} value={line.text} maxLength={160} placeholder="Your words, your energy…" onChange={e=>update({lines:value.lines.map((v,n)=>n===i?{...v,text:e.target.value}:v)})}/>
            <div className="caption-timing">{(['start','end'] as const).map(field=><label key={field}>{field==='start'?'From':'To'}<input aria-label={`Subtitle ${i+1} ${field} seconds`} type="number" min={0} max={duration} step={.1} value={line[field]} onChange={e=>update({lines:value.lines.map((v,n)=>n===i?{...v,[field]:Number(e.target.value)}:v)})}/></label>)}<button type="button" aria-label={`Remove subtitle ${i+1}`} onClick={()=>update({lines:value.lines.filter((_,n)=>n!==i)})}>Remove</button></div>
          </div>)}</div>
          <button type="button" className="secondary" disabled={value.lines.length>=100||(value.lines.at(-1)?.end||0)>=duration} onClick={add}>+ Add subtitle line</button>
          {problem&&<p className="caption-note" role="status">{problem}</p>}
        </>}
      </>}
      {onExport&&<><button type="button" className="primary caption-export" disabled={disabled||!!problem} onClick={onExport}>{disabled?'Exporting…':'Export new version'}</button><small className="caption-note">Your cuts and music stay the same.</small></>}
    </fieldset>
  </section>;
}
