"""Plan bounded visual questions, not fragments of a global replacement transaction."""
import math
from .candidates import overlap
from .evidence import digest

SCAN_KINDS={"coverage","protocol","anchor_search","zero_confirmation"}


def plan_checks(checks,q,media,config,scope,source_fps,*,legal_scope=None):
    tasks=[]
    legal=legal_scope or scope
    for c in checks:
        descriptions=c.get("descriptions",[])
        # Frame-local checks still need all candidate evidence. For long activities
        # use continuous low-density context, not isolated 16-fps boundary islands.
        contextual=q.recipe!="cycles"
        fps=min(config.refinement_fps,2*config.long_fps) if contextual else config.refinement_fps
        if source_fps: fps=min(fps,source_fps)
        points=[media.catalog[x]["timestamp_seconds"] for x in c.get("source_refs",[]) if x in media.catalog]
        if c["kind"] in SCAN_KINDS or not points:
            a,b=c["span"]
        else:
            a,b=min(points)-.5,max(points)+.5
            if c["kind"] in {"identity","event_unit","anchor","adjacency","adjacency_confirmation"}:
                a,b=min(a,c["span"][0]),max(b,c["span"][1])
        a,b=max(legal[0],a),min(legal[1],b)
        if b<=a: continue
        # A scan-sized window leaves capacity for exact original citations. The
        # preparer never silently drops these references to fit a frame budget.
        width=(config.max_frames_per_call//2)/fps
        count=max(1,math.ceil((b-a)/width))
        pending=[(a+(b-a)*i/count,a+(b-a)*(i+1)/count) for i in range(count)]
        pieces=[]
        while pending:
            lo,hi=pending.pop(0)
            required=list(dict.fromkeys(x for x in c.get("source_refs",[]) if x in media.catalog and lo<=media.catalog[x]["timestamp_seconds"]<=hi))
            size=math.ceil((hi-lo)*fps)+len(required)
            if size>config.max_frames_per_call and hi-lo>1/fps:
                mid=(lo+hi)/2;pending[0:0]=[(lo,mid),(mid,hi)];continue
            if size>config.max_frames_per_call:
                raise ValueError("required local evidence exceeds frame cap")
            pieces.append((lo,hi,required,size))
        for index,(lo,hi,required,size) in enumerate(pieces):
            contained=[];context=[]
            for i,d in enumerate(descriptions):
                bounds=d.get("bounds")
                if bounds and lo<=bounds[0]<=bounds[1]<=hi: contained.append(i)
                elif bounds and overlap(bounds,[lo,hi]): context.append(i)
                elif not bounds and len(pieces)==1: contained.append(i)
            ids=[c["rows"][i] for i in contained if i<len(c.get("rows",[]))]
            # All slices are independent. Their stable retry key is tied to the
            # source interval + original lineage, never to newly assigned IDs.
            key=digest([q.op,c["kind"],c.get("lineage",c["key"]),[round(lo,6),round(hi,6)]])
            segment={"span":[lo,hi],"core":[lo,hi],"required_refs":required}
            part={**c,"key":key,"parent_key":c["key"],"parent_span":c["span"],
                  "part":0,"parts":1,"local_only":True,"local_scan":c["kind"] in SCAN_KINDS,
                  "span":[lo,hi],"segments":[segment],"rows":ids,
                  "source_refs":required,
                  "versions":{x:c.get("versions",{}).get(x,1) for x in ids},
                  "descriptions":[descriptions[i] for i in contained],
                  "context_rows":[c["rows"][i] for i in context if i<len(c.get("rows",[]))],
                  "context_descriptions":[descriptions[i] for i in context],
                  "slice_index":index,"slice_count":len(pieces)}
            if c["kind"] in SCAN_KINDS:
                part.update(events=[],subjects={},aspects=[])
            task={"key":"review:"+key,"kind":"review","recipe":q.recipe,
                  "span":[lo,hi],"core":[lo,hi],"fps":fps,"segments":[segment],
                  "checks":[part],"estimated_frames":size,"depends_on":[]}
            tasks.append(task)
    return tasks
