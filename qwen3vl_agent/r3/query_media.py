"""Query windows use actual source PTS and never pad or duplicate selected frames."""
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw
from qwen3vl_agent.temporal_media import TemporalMedia, TemporalPrepared, file_digest
from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.r1.media import MediaBatch
from qwen3vl_agent.p01.types import TimeSpan

@dataclass
class Access:
    span: tuple[float,float]
    def permits(self, seconds): return self.span[0] <= seconds <= self.span[1]
    def permits_span(self, span): return self.span[0] <= span[0] < span[1] <= self.span[1]

class QueryMedia(TemporalMedia):
    def __init__(self, config):
        config = replace(config,media=replace(config.media,normal_total_pixels=min(config.media.normal_total_pixels,config.visual_token_target*1024)))
        super().__init__(config,namespace="r3v54")

    def prepare(self, batch, *, safe=False):
        # R3-only display transform. Never rewrite raw frames or invoke native-video preparation.
        if not batch.frames or len(batch.frames)>self.config.max_frames_per_call:
            raise ValueError("numbered image input must contain 1-64 actual frames")
        if len({f.id for f in batch.frames})!=len(batch.frames):
            raise ValueError("duplicate source views in display input")
        task=getattr(batch,"r3_task",{})
        segments=task.get("segments",[{"span":[batch.span.start_seconds,batch.span.end_seconds],
                                       "core":task.get("core",[batch.span.start_seconds,batch.span.end_seconds])}])
        c=self.config.media
        total=min(c.safe_normal_total_pixels if safe else c.normal_total_pixels,
                  self.config.visual_token_target*1024,self.config.visual_token_limit*1024)
        caps=[min(total/len(batch.frames),c.detail_max_pixels if f.id in batch.crops else c.normal_max_pixels) for f in batch.frames]
        parts,rendered,sizes,display=[],[],[],{}
        root=c.resolved_cache_dir/"r3v54_display"
        root.mkdir(parents=True,exist_ok=True)
        for i,(frame,cap) in enumerate(zip(batch.frames,caps),1):
            source=self.catalog[frame.id]
            with Image.open(frame.path) as raw:
                w,h=raw.size
                scale=min(1.0,math.sqrt(cap/(w*(h+32))))
                dw=max(32,int(w*scale)//32*32)
                dh=max(32,int(h*scale)//32*32)
                # Include the external 32px band in BOTH the grid and pixel budget.
                while dw*(dh+32)>cap and (dw>32 or dh>32):
                    if dw>=dh and dw>32: dw-=32
                    elif dh>32: dh-=32
                if dw*(dh+32)>cap: raise ValueError("display budget cannot fit every frame; no frames removed")
                memberships=[(j,s) for j,s in enumerate(segments,1) if s["span"][0]-1e-6<=frame.timestamp_seconds<=s["span"][1]+1e-6]
                seg,s=memberships[0] if memberships else (i,{"core":[-1,-1]})
                core=s.get("core",s["span"] if "span" in s else [-1,-1])
                zone="CORE" if core[0]<=frame.timestamp_seconds<core[1] or task.get("last") and frame.timestamp_seconds==core[1] else "CONTEXT"
                label=f"F{i:02d} at {frame.timestamp_seconds:.6f}s"
                detail=f"{zone} | SEGMENT {seg}"
                transform={"version":"r3-5.4","original_path":frame.path,"source_frame_id":source["source_frame_id"],
                           "raw_sha256":file_digest(frame.path),"view_box":source.get("view_box"),
                           "content_size":[dw,dh],"header_pixels":32,"display_size":[dw,dh+32],
                           "label":label,"zone":zone,"segment":seg,"timestamp_seconds":frame.timestamp_seconds}
                key=hashlib.sha256(json.dumps(transform,sort_keys=True).encode()).hexdigest()
                path=root/(key+".png")
                if not path.exists():
                    canvas=Image.new("RGB",(dw,dh+32),"#101820")
                    canvas.paste(raw.convert("RGB").resize((dw,dh),Image.Resampling.LANCZOS),(0,32))
                    draw=ImageDraw.Draw(canvas)
                    # Fit long labels at small display sizes without writing onto video pixels.
                    band=Image.new("RGB",(max(dw,280),32),"#101820")
                    text_draw=ImageDraw.Draw(band)
                    text_draw.text((3,1),label,fill="white")
                    text_draw.text((3,16),detail,fill="#ffda60")
                    canvas.paste(band.resize((dw,32),Image.Resampling.LANCZOS),(0,0))
                    canvas.save(path)
                transform.update(path=str(path),sha256=file_digest(path))
                display[f"F{i:02d}"]=transform
            area=dw*(dh+32)
            parts.extend([{"type":"text","text":label+" | "+detail+" | external label, not video content"},
                          {"type":"image","image":str(path),"min_pixels":area,"max_pixels":area}])
            rendered.append(FrameRef(frame.id,frame.timestamp_seconds,str(path)))
            sizes.append((dw,dh+32))
        result=TemporalPrepared(parts,tuple(rendered),sum(w*h for w,h in sizes),sizes,False,"numbered_images")
        result.display=display
        result.fallback_reason=None
        return result

    def task_batch(self, path, task, access, *, source_fps=None):
        if task.get("segments"):
            batches=[self.task_batch(path,{**task,"segments":[],"span":s["span"],"core":s.get("core",s["span"]),
                                          "required_refs":s.get("required_refs",[])},access,source_fps=source_fps) for s in task["segments"]]
            # Duplicate boundary views are one source, not an additional sample.
            frames={f.id:f for b in batches for f in b.frames}
            if len(frames)>self.config.max_frames_per_call: raise ValueError("multi-segment input must be split, never drop frames")
            batch=MediaBatch(TimeSpan(*task["span"]),tuple(sorted(frames.values(),key=lambda f:f.timestamp_seconds)),
                             batches[0].requested_fps,False,errors=[e for b in batches for e in b.errors])
            batch.r3_task=task
            return batch
        fps = min(task["fps"],source_fps) if source_fps and source_fps>0 else task["fps"]
        span = task["span"]
        if task.get("attribute_only"):
            times = [span[0]+(span[1]-span[0])*i/3 for i in range(4)]
            fps = None
        else:
            times = [span[0]+i/fps for i in range(max(1,math.ceil((span[1]-span[0])*fps-1e-8)))]
        if len(times)>self.config.max_frames_per_call:
            raise ValueError("sampling plan exceeds frame cap; must shorten the interval, never delete samples")
        batch = self.extract(path,span,times,access,fps=fps,
                             anchors=[self.frame(x) for x in dict.fromkeys(task.get("tail_refs",[])+task.get("required_refs",[])) if x in self.catalog])
        if len(batch.frames)>self.config.max_frames_per_call:
            raise ValueError("actual frames plus required context exceed cap; split required")
        batch.r3_task=task
        if task.get("attribute_only"):
            batch.ordered = False
            batch.coverage_kind = "attribute"
            # Preserve the full original field of view, but allow the existing detail cap.
            batch.crops = {f.id:{"full_frame_detail":True} for f in batch.frames}
        return batch


def base_tasks(query, scope, config):
    short = query.recipe=="cycles"
    core = config.short_core_sec if short else config.long_core_sec
    fps = config.short_fps if short else config.long_fps
    n = math.ceil((scope[1]-scope[0])/core-1e-9)
    tasks=[]
    for i in range(n):
        a,b = scope[0]+i*core,min(scope[1],scope[0]+(i+1)*core)
        pad = config.context_sec if short else config.tail_sec
        tasks.append({"key":f"base:{i}","kind":"base","recipe":query.recipe,"core":[a,b],
                      "span":[max(scope[0],a-pad),min(scope[1],b+(pad if short else 0))],
                      "fps":fps,"last":i==n-1})
    if query.op in {"last_occurrence","last_k"}: tasks.reverse()
    return tasks


def sampling_record(media,batch,task):
    if task.get("segments"):
        records=[]
        for s in task["segments"]:
            local=MediaBatch(TimeSpan(*s["span"]),tuple(f for f in batch.frames if s["span"][0]<=f.timestamp_seconds<=s["span"][1]),batch.requested_fps,True,errors=batch.errors)
            records.append(sampling_record(media,local,{**task,"segments":[],"core":s.get("core",s["span"])}))
        return {"resolution_met":all(s["resolution_met"] for s in records),"segments":records,
                "requested_fps":batch.requested_fps,"span":task["core"],"disjoint":True}
    record=media.coverage(batch,task["key"],True)
    record["span"] = task["core"]
    times=sorted(f.timestamp_seconds for f in batch.frames if task["core"][0]<=f.timestamp_seconds<=task["core"][1])
    points=[task["core"][0],*times,task["core"][1]]
    gap=max((b-a for a,b in zip(points,points[1:])),default=math.inf)
    record["max_gap_sec"]=gap
    record["resolution_met"] = bool(times) and not batch.errors and not batch.missing_anchor_ids and (not batch.requested_fps or gap<=1.6/batch.requested_fps+.001)
    record["assumption"]="Evidence is conditional on finite sampling at the recorded effective rate."
    return record
