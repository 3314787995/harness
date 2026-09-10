"""One resource ledger, durable raw returns and a bounded interrupted-call recovery."""
from dataclasses import asdict
import hashlib
import json
import time
from pathlib import Path
from .types import R3Budget, BudgetExhausted
from qwen3vl_agent.models.qwen3vl import VisualBudgetExceeded

class QueryRuntime:
    def __init__(self, model, media, config, request, checkpoint):
        self.model,self.media,self.config,self.request,self.checkpoint=model,media,config,request,checkpoint
        caps={}
        for k,v in asdict(request.budget).items():
            values=[x for x in (v,getattr(config.budget,k)) if x is not None]
            caps[k]=min(values) if values else None
        self.budget=R3Budget(**caps)
        self.state=checkpoint.restored or {"calls":{},"candidates":[],"coverage":[],"committed":[],
            "reviews":[],"provider_calls":[],"blocked":[],"catalog":{},"query":None,"result":None}
        media.catalog.update(self.state["catalog"])
        self.provider_calls=self.state["provider_calls"]

    def changed(self):
        self.state["catalog"]=self.media.catalog
        self.checkpoint.save(self.state)

    def usage(self):
        calls=list(self.state["calls"].values())
        return {"model_calls":sum(c.get("charged",False) for c in calls),
                "frame_exposures":sum(c.get("frames",0) for c in calls if c.get("charged")),
                "media_pixels":sum(c.get("actual_pixels",c.get("pixels",0)) for c in calls if c.get("charged")),
                "visual_tokens":sum(c.get("actual_visual_tokens",c.get("estimated_visual_tokens",0)) for c in calls if c.get("charged")),
                "provider_calls":len(self.provider_calls)}

    def review_usage(self):
        return sum(bool(c.get("charged")) for c in self.state["calls"].values()
                   if c.get("task",{}).get("kind")=="review" or c.get("is_recovery"))

    def call(self, task, messages, *, prepared=None, refs=None, sampling=None, tokens=768, final=False):
        key=task["key"]
        old=self.state["calls"].get(key)
        if old and old["status"] in {"returned","blocked"}: return old
        if old:
            if task.get("kind")=="query":
                raise BudgetExhausted("single semantic query parse was interrupted; no second parse")
            retry=key+":recovery"
            record=self.state["calls"].get(retry)
            if record and record["status"] in {"returned","blocked"}: return record
            if record: raise BudgetExhausted("interrupted call recovery already consumed")
            if final: raise BudgetExhausted("single final visual call was interrupted; no second final")
            key=retry
        usage=self.usage()
        if (task.get("kind")=="review" or old) and self.review_usage()>=self.config.max_refinements:
            raise BudgetExhausted("configured shared visual confirmation/correction/recovery calls exhausted")
        frames=len(prepared.frames) if prepared else 0
        # Images can be temporally expanded by the processor; reserve conservatively.
        pixels=prepared.pixels*2 if prepared else 0
        estimated=math_ceil(prepared.pixels/1024) if prepared else 0
        available=self.budget.max_model_calls-usage["model_calls"]-(0 if final else 1)
        if available<=0: raise BudgetExhausted("global model-call budget (one final reserved)")
        if usage["frame_exposures"]+frames>self.budget.max_frame_exposures: raise BudgetExhausted("global frame budget")
        if usage["media_pixels"]+pixels>self.budget.max_media_pixels: raise BudgetExhausted("global pixel budget")
        if self.budget.max_visual_tokens is not None and usage["visual_tokens"]+estimated>self.budget.max_visual_tokens: raise BudgetExhausted("global visual-token budget")
        if len(json.dumps(messages,ensure_ascii=False))>self.budget.max_text_chars_per_call: raise BudgetExhausted("per-call text budget")
        sig=hashlib.sha256(json.dumps({"messages":messages,"frames":list((refs or {}).values())},sort_keys=True,default=str).encode()).hexdigest()
        if any(c.get("signature")==sig for c in self.state["calls"].values()) and not old:
            raise BudgetExhausted("identical visual input already attempted")
        record={"id":key,"task":task,"status":"started","charged":True,"frames":frames,"pixels":pixels,
                "estimated_visual_tokens":estimated,"refs":refs or {},"sampling":sampling or {},"signature":sig,
                "is_recovery":bool(old),"display_transforms":getattr(prepared,"display",{}),
                "messages":messages,"input_mode":prepared.kind if prepared else "text",
                "fallback_reason":getattr(prepared,"fallback_reason",None),
                "sizes":list(prepared.sizes) if prepared else [],"started":time.time()}
        self.state["calls"][key]=record
        self.changed()  # before any inference
        kwargs={"max_new_tokens":tokens,"temperature":0.0}
        if prepared:
            kwargs["visual_token_limit"]=min(self.config.visual_token_limit, self.budget.max_visual_tokens-usage["visual_tokens"] if self.budget.max_visual_tokens is not None else self.config.visual_token_limit)
            kwargs["processed_pixel_limit"]=self.budget.max_media_pixels-usage["media_pixels"]
            if prepared.video_frame_metadata: kwargs["video_frame_metadata"]=prepared.video_frame_metadata
            if self.checkpoint.path:
                receipt=self.checkpoint.path.parent/(self.checkpoint.path.name+".receipts")/(hashlib.sha256(key.encode()).hexdigest()[:16]+".json")
                kwargs["preparation_receipt_path"]=str(receipt)
                record["preparation_receipt_path"]=str(receipt)
                self.changed()
        try:
            output=self.model.generate(messages,**kwargs)
            record.update(status="returned",raw=output.text,metadata=output.metadata,
                          elapsed=time.time()-record["started"])
            record["actual_pixels"]=output.metadata.get("processed_pixels",pixels)
            record["actual_visual_tokens"]=output.metadata.get("visual_tokens",estimated)
            # This commit is separate from parsing/ingestion, so a returned output is never lost.
            self.changed()
        except VisualBudgetExceeded as exc:
            record.update(status="blocked",charged=False,error=str(exc),elapsed=time.time()-record["started"])
            self.changed()
        except BaseException as exc:
            record.update(error_type=type(exc).__name__,error=str(exc),elapsed=time.time()-record["started"])
            self.changed()
            raise
        return record

def math_ceil(value):
    import math
    return math.ceil(value)
