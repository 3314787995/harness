"""Query-driven v5.4: local evidence updates, scoped verification, preserved answer context."""
from dataclasses import asdict, replace
import json
import math
import random
import re
from .query import QuerySpec, VERSION, QUERY_PROMPT, validate_query, rule_query, parse_object, public_policy, candidate_vocabulary
from .lines import PROMPTS, parse_lines, line_rules
from .candidates import ingest, gap, overlap, events_from_rows
from .query_reduce import reduce_query, covers, ordered, semantic_range
from .query_media import Access, base_tasks, sampling_record
from .query_runtime import QueryRuntime
from .checkpoint import Checkpoint, implementation_digest
from .types import BudgetExhausted, ProtocolError, R3Result
from .providers import ProviderAdapter
from .review import review_prompt, checks_for, plan_checks, parse_checks, commit_checks
from .answer_evidence import build_packet, packet_payload
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.media import MediaBatch


def number(text):
    words={v:i for i,v in enumerate("zero one two three four five six seven eight nine ten".split())}
    s=text.strip().casefold()
    if s in words: return words[s]
    if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*(?:times?|seconds?|secs?|s))?",s):
        return float(re.match(r"\d+(?:\.\d+)?",s)[0])
    return None


def exact_choice(value, choices):
    if not isinstance(value,str): return None
    s=value.strip()
    if s in {c.label for c in choices}: return s
    matched=[c.label for c in choices if c.text.strip().casefold()==s.casefold()]
    return matched[0] if len(matched)==1 else None


def deterministic_mapping(request, reduction):
    if not reduction["closed"]: return None
    value=reduction["value"]
    if not request.choices:
        return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
    matches=[]
    if isinstance(value,str):
        # A semantic value equal to a label is not evidence for that option; text only here.
        matches=[c.label for c in request.choices if c.text.strip().casefold()==value.strip().casefold()]
    elif isinstance(value,(int,float)) and not isinstance(value,bool):
        matches=[c.label for c in request.choices if number(c.text)==value]
    elif isinstance(value,dict) and "min_sec" in value:
        bins=request.benchmark_policy.get("duration_option_intervals",{})
        matches=[k for k,(a,b) in bins.items() if a<=value["min_sec"]<=value["max_sec"]<=b]
    elif reduction["op"]=="localize_event" and isinstance(value,list):
        bins=request.benchmark_policy.get("temporal_option_intervals",{})
        matches=[k for k,(a,b) in bins.items() if value and all(a<=e["onset"][0]<=e["offset"][1]<=b for e in value)]
    elif isinstance(value,list):
        # Exact serialized sequences only; no fuzzy or substring option matching.
        matches=[c.label for c in request.choices if c.text.strip()==json.dumps(value,ensure_ascii=False)]
    return matches[0] if len(matches)==1 else None


class QueryEngine:
    def __init__(self,agent,request):
        self.agent,self.request,self.config,self.media=agent,request,agent.config,agent.media
        metadata=self.media.probe(request.video_path)
        duration=metadata.duration_seconds
        if not math.isfinite(duration) or duration<=0: raise ValueError("invalid media duration")
        allowed=request.allowed_scope or TimeSpan(0,duration)
        if allowed.end_seconds>duration+1e-6: raise ValueError("allowed scope exceeds source video")
        self.allowed=(allowed.start_seconds,min(allowed.end_seconds,request.observation_cutoff or duration))
        if self.allowed[0]>=self.allowed[1]: raise ValueError("empty allowed range")
        self.access=Access(self.allowed)
        self.source_fps=metadata.source_fps
        identity=asdict(request)
        identity.pop("resume");identity.pop("checkpoint_path")
        cp=Checkpoint(request.checkpoint_path,{"version":VERSION,"request":identity,
            "media":self.media.source_digest(request.video_path),"implementation":implementation_digest(),
            "config":asdict(self.config),"model_path":agent.model.model_path,
            "model_configuration":{k:getattr(agent.model,k,None) for k in ("device","dtype","generation","device_map","max_memory","attn_implementation","revision","required_cuda_devices","forbid_offload")}},resume=request.resume)
        self.runtime=QueryRuntime(agent.model,self.media,self.config,request,cp)
        self.state=self.runtime.state
        self.provider=ProviderAdapter(agent.provider,self.config,self.runtime,self.media.source_digest(request.video_path),TimeSpan(*self.allowed),request.available_modalities)

    def determine_query(self):
        if self.state["query"] is not None:
            return QuerySpec(**self.state["query"])
        if self.state.get("query_failed"): return None
        req=self.request
        try:
            value=req.query_spec if req.query_spec is not None else rule_query(req)
            origin="upstream" if req.query_spec is not None else "explicit_grammar"
            if value is None:
                payload={"question":req.question,"public_rules":public_policy(req),
                         "requested_scope":str(req.query_scope) if req.query_scope is not None else None,
                         "execution_subtype":req.execution_subtype,"unlabeled_vocabulary":candidate_vocabulary(req)}
                record=self.runtime.call({"key":"query","kind":"query"},[
                    {"role":"system","content":QUERY_PROMPT},
                    {"role":"user","content":json.dumps(payload,ensure_ascii=False)}],tokens=self.config.query_tokens)
                if record["status"]!="returned": raise ProtocolError(record.get("error","query unavailable"))
                value=parse_object(record["raw"])
                origin="one_semantic_parse"
            query=validate_query(value,req)
            policy=public_policy(req)
            if type(policy.get("count_replays")) is bool:
                query=replace(query,count_replays=policy["count_replays"])
            if isinstance(req.query_scope,str):
                query=replace(query,scope={"kind":"semantic","description":req.query_scope,"source":"adapter query_scope"})
            self.state["query"],self.state["query_origin"]=query.to_dict(),origin
        except (ValueError,TypeError,BudgetExhausted) as exc:
            self.state["query_failed"]=str(exc)
            query=None
        self.runtime.changed()
        return query

    def scope(self,q):
        value=self.allowed
        if q.scope["kind"]=="interval": value=tuple(q.scope["interval"])
        if isinstance(self.request.query_scope,TimeSpan):
            supplied=(self.request.query_scope.start_seconds,self.request.query_scope.end_seconds)
            value=(max(value[0],supplied[0]),min(value[1],supplied[1]))
        value=(max(value[0],self.allowed[0]),min(value[1],self.allowed[1]))
        if value[0]>=value[1]: raise ProtocolError("query and legal media ranges have no intersection")
        return value

    def targets(self,q):
        targets=list(dict.fromkeys([q.target,*q.targets,*([q.anchor] if q.anchor else [])]))
        # Cooccurrence companions are not independent primary-event targets.
        if q.template=="COOCCUR": targets=[q.target]
        return targets

    def aliases(self,q):
        return {f"T{i}":t for i,t in enumerate(self.targets(q),1)}

    def references(self,prepared):
        return {f"F{i+1:02d}":{**self.media.catalog[f.id],"display":getattr(prepared,"display",{}).get(f"F{i+1:02d}")}
                for i,f in enumerate(prepared.frames)}

    def execute(self,task,q):
        old=self.state["calls"].get(task["key"]+":recovery") or self.state["calls"].get(task["key"])
        if old and old["status"]=="returned":
            record=old  # No decode, sampling, provider read or model call after a raw return.
        elif old and old["status"]=="blocked":
            return False
        else:
            try:
                if self.runtime.usage()["model_calls"]>=self.runtime.budget.max_model_calls-1:
                    raise BudgetExhausted("remaining call is reserved for the single visual final")
                # Only actually supplied overlapping tail frames establish cross-window continuity.
                tail=[m["id"] for m in self.media.catalog.values() if task["span"][0]<=m["timestamp_seconds"]<task["core"][0] and m["id"]==m["source_frame_id"]]
                # Bound tail inclusion without deleting regular samples.
                task={**task,"tail_refs":tail[-2:]}
                task["observed_targets"]=self.targets(q)
                batch=self.media.task_batch(self.request.video_path,task,self.access,source_fps=self.source_fps)
                if not batch.frames: raise ValueError("no permitted decodable frames")
                task["tail_refs"]=[f.id for f in batch.frames if f.timestamp_seconds<task["core"][0]]
                prepared=self.media.prepare(batch)
                refs=self.references(prepared)
                sample=sampling_record(self.media,batch,task)
                external=[]
                if set(self.request.available_modalities)&{"subtitle","asr"}:
                    read=self.provider.fetch(TimeSpan(*task["span"]))
                    external=[asdict(s) for s in read.items]
                existing=[{"target":r["target"],"decision":r["decision"],"note":r["note"],"value":r.get("value")}
                          for r in self.state["candidates"] if r["active"] and overlap(r["bounds"],task["span"])][-8:]
                view=q.observer_view()
                if q.template=="SELECT" and q.group_by=="occurrence" and not task.get("attribute_only"):
                    view["attribute"]=None  # First establish the selection, then request its attribute.
                payload={"query":view,"targets":self.aliases(q),
                         "public_replay_rule":public_policy(self.request).get("count_replays"),
                         "core_frame_markers":[k for k,v in refs.items() if task["core"][0]<=v["timestamp_seconds"]<task["core"][1]],
                         "unlabeled_vocabulary":candidate_vocabulary(self.request,q),
                         "existing_local_descriptions":existing,"external_text_not_visual_actions":external,
                         "segment_layout":"one chronological sampled segment; intervals between frames are unsampled",
                         "task":task.get("instruction","Observe the core. Context is for identity only.")}
                instruction=PROMPTS[task["recipe"]]
                if task.get("kind")=="review":
                    instruction += "\nR3:critical_review Re-examine the specified gap using the NEW supplied visual input. You may reject preparation, separate cycles, or retain uncertainty. Replace observations only for the supplied local range. Do not copy earlier judgments."
                if task.get("attribute_only"):
                    instruction += "\nReturn attribute lines only for the selected instance, followed by end. These disjoint clear frames do not prove temporal continuity."
                record=self.runtime.call(task,[{"role":"system","content":instruction},
                    {"role":"user","content":[{"type":"text","text":json.dumps(payload,ensure_ascii=False)},*prepared.parts]}],
                    prepared=prepared,refs=refs,sampling=sample,
                    tokens=self.config.refinement_tokens if task.get("kind")=="review" else self.config.observer_tokens)
            except (BudgetExhausted,ValueError) as exc:
                self.state["blocked"].append({"task":task,"reason":str(exc)})
                self.runtime.changed()
                return False
        if record["status"]!="returned": return False
        token_limit=self.config.refinement_tokens if task.get("kind")=="review" else self.config.observer_tokens
        core=[k for k,v in record["refs"].items() if record["task"]["core"][0]<=v["timestamp_seconds"]<record["task"]["core"][1]]
        parsed=parse_lines(record["raw"],task["recipe"],record["refs"],self.aliases(q),
                           truncated=record.get("metadata",{}).get("output_tokens",0)>=token_limit,
                           time_basis=q.time_basis,companions=q.targets if q.template=="COOCCUR" else (),unit=q.unit,core_markers=core)
        ingest(self.state,record,parsed)
        if task.get("zero_check") and parsed["negative"] and record["sampling"]["resolution_met"]:
            self.state["zero_reviewed"]=True
        self.runtime.changed()
        return parsed["complete"]

    def dense_tasks(self,q,reduction,scope):
        if q.template!="NEIGHBOR": return []
        anchors=[e for e in reduction["events"] if e["target"]==q.anchor]
        if q.anchor_selection in {"first","last"}:
            sequence=ordered(anchors,"onset")
            if sequence: anchors=[sequence[0] if q.anchor_selection=="first" else sequence[-1]]
        forward=q.op=="next_after_anchor"
        if len(anchors)!=1: return []
        boundary=anchors[0]["offset" if forward else "onset"]
        if boundary is None: return []
        candidates=[e for e in reduction["events"] if e["target"]==q.target]
        edges=[e["onset" if forward else "offset"] for e in candidates]
        edges=[e for e in edges if e and (e[0]>boundary[1] if forward else e[1]<boundary[0])]
        a,b=(boundary[0],min(e[1] for e in edges) if edges else scope[1]) if forward else (max(e[0] for e in edges) if edges else scope[0],boundary[1])
        tasks=[]
        for i in range(max(0,math.ceil((b-a)/self.config.neighbor_sec))):
            lo,hi=a+i*self.config.neighbor_sec,min(b,a+(i+1)*self.config.neighbor_sec)
            if hi<=lo: continue
            tasks.append({"key":f"neighbor:{lo:.6f}:{hi:.6f}","kind":"necessary_detail","recipe":"anchors",
                          "core":[lo,hi],"span":[max(scope[0],lo-1),hi],"replace_span":[lo,hi],"fps":self.config.neighbor_fps,
                          "instruction":"Identify the anchor boundary, every intermediate activity and the nearest candidate. Unknown intervals remain unknown."})
        return tasks if forward else list(reversed(tasks))

    def execute_check(self,entry,q):
        task=entry["task"]
        if task["checks"] and all(c.get("purpose")=="scope" for c in task["checks"]):
            q=QuerySpec("localize_event",task["checks"][0]["descriptions"][0]["target"],"activity",selection="unique")
        old=self.state["calls"].get(task["key"]+":recovery") or self.state["calls"].get(task["key"])
        try:
            if old and old["status"]=="returned": record=old
            elif old and old["status"]=="blocked":
                entry.update(done=True,reason=old.get("error"));return
            else:
                if self.runtime.review_usage()>=self.config.max_refinements:
                    raise BudgetExhausted("shared confirmation/correction/recovery allowance exhausted")
                batch=self.media.task_batch(self.request.video_path,task,self.access,source_fps=self.source_fps)
                prepared=self.media.prepare(batch)
                refs=self.references(prepared)
                previous=[c for c in self.state["calls"].values() if c.get("refs") and c["status"]=="returned"]
                current={v["id"] for v in refs.values()}
                seen={v["id"] for c in previous for v in c["refs"].values()}
                # Combining unrelated checks is not new visual context for either check.
                previous_sets=[{v["id"] for v in c["refs"].values()} for c in previous]
                context=any(bool(local) and previous_sets and not any(local<=old_set for old_set in previous_sets)
                    for check in task["checks"]
                    for local in [{v["id"] for v in refs.values() if any(s["span"][0]<=v["timestamp_seconds"]<=s["span"][1] for s in check["segments"])}])
                old_sizes={v["id"]:max([math.prod(size)]+[math.prod(s) for c2 in previous for v2,s in zip(c2["refs"].values(),c2["sizes"]) if v2["id"]==v["id"]])
                           for c in previous for v,size in zip(c["refs"].values(),c["sizes"])}
                clearer=any(math.prod(size)>old_sizes.get(f.id,math.inf) for f,size in zip(prepared.frames,prepared.sizes))
                if not old and not (current-seen or context or clearer):
                    entry.update(done=True,reason="no new actual frames, clarity or effective visual context")
                    self.state.setdefault("skipped_reviews",[]).append({"task":task["key"],"reason":entry["reason"]})
                    return
                checks=[]
                for i,c in enumerate(task["checks"],1):
                    checks.append({"check":f"C{i}","matters":c["aspects"],"reason":c["reason"],
                        "candidate_descriptions":c["descriptions"],"part":c["part"]+1,"parts_required":c["parts"],
                        "local_interval":c["span"],"local_only":c.get("local_only",False),
                        "local_scan":c.get("local_scan",False),"read_only_context":c.get("context_descriptions",[]),
                        "reference_groups":[[k for k,v in refs.items() if s["span"][0]<=v["timestamp_seconds"]<=s["span"][1]] for s in c["segments"]],
                        "allowed_frames":[k for k,v in refs.items() if any(s["span"][0]<=v["timestamp_seconds"]<=s["span"][1] for s in c["segments"])]})
                payload={"query":q.observer_view(),"targets":self.aliases(q),"checks":checks,
                         "public_replay_rule":public_policy(self.request).get("count_replays"),
                         "unlabeled_vocabulary":candidate_vocabulary(self.request,q),
                         "segments":[{"segment":i,"span":s["span"],"core":s["core"]} for i,s in enumerate(task["segments"],1)],
                         "segment_layout":"Only listed numbered images are supplied. Separate segments do not show their intervening gaps."}
                instruction=review_prompt(q.recipe)
                record=self.runtime.call(task,[{"role":"system","content":instruction},
                    {"role":"user","content":[{"type":"text","text":json.dumps(payload,ensure_ascii=False)},*prepared.parts]}],
                    prepared=prepared,refs=refs,sampling=sampling_record(self.media,batch,task),tokens=self.config.refinement_tokens)
            if record["status"]=="returned":
                parsed=parse_checks(record["raw"],record["task"],record["refs"],q,self.aliases(q),
                                    truncated=record.get("metadata",{}).get("output_tokens",0)>=self.config.refinement_tokens)
                commit_checks(self.state,record,parsed,q,commit=self.runtime.changed)
                entry["errors"]=parsed["errors"]
            entry["done"]=True
        except (BudgetExhausted,ValueError) as exc:
            entry.update(done=True,reason=str(exc))
            self.state["blocked"].append({"task":task,"reason":str(exc)})
        finally:
            # Attempts follow stable source lineage; aliases or new IDs cannot reset them.
            if entry.get("done") or task["key"] in self.state["calls"]:
                attempts=self.state.setdefault("review_attempts",[])
                for c in task["checks"]:
                    if c["key"] not in attempts: attempts.append(c["key"])
            self.runtime.changed()

    def attribute_tasks(self,q,reduction,scope):
        if q.template!="SELECT" or not reduction["selected"]:
            return []
        if any(g["kind"] not in {"attribute","confirmation"} for g in reduction["gaps"]):
            return []  # Do not spend an attribute call while the ordinal/unit is unresolved.
        tasks=[]
        for issue in reduction["gaps"]:
            if issue["kind"]!="attribute": continue
            a,b=issue["span"]
            if b<=a: a,b=max(scope[0],a-.25),min(scope[1],b+.25)
            tasks.append({"key":"attribute:"+issue["key"],"kind":"necessary_attribute", "recipe":q.recipe,
                          "core":[a,b],"span":[a,b],"fps":1,"attribute_only":True,
                          "selected_candidates":issue["events"],
                          "instruction":"Read only the requested property of the program-selected instance in these images. Do not propose another instance."})
        return tasks

    def read_selected_attributes(self,q,reduction,scope):
        known=self.state.setdefault("attribute_tasks",[])
        for task in self.attribute_tasks(q,reduction,scope):
            if task["key"] not in {t["key"] for t in known}:
                known.append(task)
        self.runtime.changed()
        for task in known:
            self.execute(task,q)
        return reduce_query(q,self.state,scope)

    def final(self,q,reduction,scope):
        prediction=deterministic_mapping(self.request,reduction)
        refs=reduction["evidence_refs"]
        basis=("visual_direct" if len(reduction.get("selected",[]))==1 else "visual_reduction") if prediction is not None else "unresolved"
        if prediction is None:
            try:
                record=self.state["calls"].get("final")
                if not record:
                    packet=build_packet(q,reduction,self.state,self.media.catalog,self.access,
                                        max_frames=min(32,self.config.max_frames_per_call))
                    self.state["answer_evidence_packet"]=packet
                    legal=packet["frame_ids"]
                    if legal:
                        frames=tuple(self.media.frame(x) for x in legal)
                        frames=tuple(sorted(frames,key=lambda f:f.timestamp_seconds))
                        batch=MediaBatch(TimeSpan(*scope),frames,None,False)
                        batch.r3_task={"segments":[{"span":[f.timestamp_seconds,f.timestamp_seconds],"core":[f.timestamp_seconds,f.timestamp_seconds+1e-9]} for f in frames],"last":True}
                    else:
                        task={"span":scope,"core":scope,"fps":1,"attribute_only":True}
                        batch=self.media.task_batch(self.request.video_path,task,self.access,source_fps=self.source_fps)
                    if not batch.frames: raise BudgetExhausted("no legal images for final visual answer")
                    prepared=self.media.prepare(batch)
                    aliases=self.references(prepared)
                    prompt='R3:final_visual r3-5.4\nUse the actual numbered key images, original question and all options. External frame labels are not video content. Images are disjoint samples, not continuous coverage. The program result is immutable; do not change it or claim it is fully proven. Return only {"prediction":"an option label or its complete exact text"}. If no legal choice is possible return {"prediction":null}. For a request without choices use a short answer string or null. No evidence lists or extra fields.'
                    payload={"question":self.request.question,"choices":[asdict(c) for c in self.request.choices],
                              "program_value":reduction["value"],"program_closed":reduction["closed"],
                              "gaps":[g["reason"] for g in reduction["gaps"][:8]],"scope":scope,
                              "candidate_evidence":packet_payload(packet,aliases)}
                    record=self.runtime.call({"key":"final","kind":"final"},[
                        {"role":"system","content":prompt},
                        {"role":"user","content":[{"type":"text","text":json.dumps(payload,ensure_ascii=False)},*prepared.parts]}],
                        prepared=prepared,refs=aliases,tokens=self.config.final_tokens,final=True)
                if record["status"]!="returned": raise BudgetExhausted(record.get("error","final unavailable"))
                native=exact_choice(record["raw"].strip(),self.request.choices)
                obj={"prediction":native} if native is not None else parse_object(record["raw"])
                if not isinstance(obj,dict) or set(obj)!={"prediction"}: raise ProtocolError("invalid final object")
                value=obj["prediction"]
                if value is None: basis="unresolved"
                elif self.request.choices:
                    prediction=exact_choice(value,self.request.choices)
                    if prediction is None: raise ProtocolError("final output has no unique legal option")
                elif isinstance(value,str) and value.strip(): prediction=value.strip()
                else: raise ProtocolError("invalid answer string")
                if prediction is not None: basis="partial_choice" if refs else "forced_choice"
            except (BudgetExhausted,ValueError,TypeError) as exc:
                self.state["final_error"]=str(exc)
                basis="protocol_failure" if isinstance(exc,(ValueError,TypeError)) else "unresolved"
        evidence_supported=reduction["closed"] and basis in {"visual_direct","visual_reduction"}
        support="supported" if evidence_supported else "partial" if refs else "unsupported"
        issues=[g["reason"] for g in reduction["gaps"]]
        if self.state.get("final_error"): issues.append(self.state["final_error"])
        trace={"version":VERSION,"answer_basis":basis,"query":q.to_dict() if q else None,
               "query_origin":self.state.get("query_origin"),"calls":list(self.state["calls"].values()),
               "critical_reviews":self.state["reviews"],"blocked_tasks":self.state["blocked"],
               "provider_calls":self.state["provider_calls"],"final_error":self.state.get("final_error"),
               "default_choice_used":False,"confirmations":self.state.get("confirmations",[]),
               "check_errors":self.state.get("check_errors",[]),"review_calls_used":self.runtime.review_usage(),
               "answer_evidence_packet":self.state.get("answer_evidence_packet"),
               "review_call_limit":self.config.max_refinements,
               "review_calls_remaining":max(0,self.config.max_refinements-self.runtime.review_usage()),
               "review_stop_reason":("evidence_closed" if reduction["closed"] else self.state.get("review_stop_reason","query_or_scope_unresolved"))}
        result=R3Result(prediction,{"results":[reduction],"closed_under_policy":reduction["closed"]},
            "complete" if evidence_supported else "incomplete",support,basis,
            event_ledger={"compatibility_view":"local candidates only; no global acceptance", "events":reduction["events"]},
            coverage_manifest=self.state["coverage"],unresolved_items=issues,evidence_refs=refs,
            resources=self.runtime.usage(),trace=trace,
            semantic_result={"op":reduction["op"],"value":reduction["value"],
                "state":"supported" if reduction["closed"] else "conditional" if refs else "unresolved",
                "count_bounds":reduction.get("count_bounds")},
             candidate_state={"rows":self.state["candidates"],"revisions":self.state.get("revisions",[]),
                              "uncommitted_facts":list(self.state.get("uncommitted_facts",{}).values()),
                             "estimate":reduction.get("candidate_estimate"),"confirmations":self.state.get("confirmations",[]),
                             "checks":self.state.get("check_parts",{}),"skipped":self.state.get("skipped_reviews",[])},
            evidence_gaps=reduction["gaps"],sampling_assumptions=[
                "All timing references map to actual source PTS; model-generated seconds are not visual evidence.",
                "Completeness is conditional on the recorded finite sampling; unsampled short actions may remain invisible.",
                "Disjoint final keyframes do not establish continuity or repair program reductions."])
        self.state["result"]=json.loads(json.dumps(asdict(result),ensure_ascii=False))
        self.runtime.changed()
        return R3Result(**self.state["result"])

    def run(self):
        if self.state["result"] is not None: return R3Result(**self.state["result"])
        random.seed(self.config.seed)
        # Torch is optional during no-model checks; do not load any weights here.
        try:
            import torch
            torch.manual_seed(self.config.seed)
        except ImportError: pass
        q=self.determine_query()
        scope=self.allowed
        if q is None:
            reduction={"op":None,"value":None,"closed":False,"events":[],"selected":[],"evidence_refs":[],
                       "gaps":[gap("query",scope,self.state.get("query_failed","query unavailable"))]}
            return self.final(q,reduction,scope)
        try: scope=self.scope(q)
        except ProtocolError as exc:
            reduction={"op":q.op,"value":None,"closed":False,"events":[],"selected":[],"evidence_refs":[],"gaps":[gap("scope",scope,str(exc))]}
            return self.final(q,reduction,scope)
        # A prior local revision can change which range is now necessary. Commit its
        # already-returned output before scheduling even one new base observation.
        for entry in self.state["reviews"]:
            if not entry.get("done"): self.execute_check(entry,q)
        if q.scope["kind"]=="semantic":
            binding=QuerySpec("localize_event",q.scope["description"],"activity",selection="unique")
            jobs=self.state.setdefault("scope_tasks",[
                {**t,"key":"scope:"+t["key"],"instruction":"Locate the requested stage, with before/after transition frames. These observations establish only stage boundaries, not absence of the later query target."}
                for t in base_tasks(binding,scope,self.config)])
            for task in jobs: self.execute(task,binding)
            stage_events,_=events_from_rows(self.state["candidates"],q)
            ranges=semantic_range(q,stage_events,scope)
            stage_scans=[s for s in self.state["coverage"] if s.get("purpose")=="scope" and s.get("targets")==[binding.target]]
            if ranges is None or not covers(stage_scans,scope):
                reduction=reduce_query(q,self.state,scope)
                reduction["closed"]=False
                reduction["gaps"].append(gap("scope",scope,"stage identity or stage-search coverage is unresolved"))
                return self.final(q,reduction,scope)
            self.state["bound_scope"]={"possible":ranges[0],"certain":ranges[1]}
            scope=tuple(ranges[0])
            self.runtime.changed()
        tasks=self.state.setdefault("required_tasks",base_tasks(q,scope,self.config))
        reduction=reduce_query(q,self.state,scope)
        # Complete pending task outputs before scheduling anything new.
        for task in tasks:
            if reduction["closed"] or reduction.get("observation_ready"): break
            self.execute(task,q)
            reduction=reduce_query(q,self.state,scope)
        if not reduction.get("observation_ready") and q.template=="NEIGHBOR":
            for task in self.state.setdefault("dense_tasks",self.dense_tasks(q,reduction,scope)):
                self.execute(task,q)
                reduction=reduce_query(q,self.state,scope)
                if reduction["closed"] or reduction.get("observation_ready"): break
        if not reduction["closed"]:
            reduction=self.read_selected_attributes(q,reduction,scope)
        while self.runtime.review_usage()<self.config.max_refinements:
            reduction=reduce_query(q,self.state,scope)
            if reduction["closed"]: break
            planned=plan_checks(checks_for(q,reduction,self.state),q,self.media,self.config,scope,self.source_fps,legal_scope=self.allowed)
            attempted=set(self.state.get("review_attempts",[]))
            planned=[t for t in planned if not all(c["key"] in attempted for c in t["checks"])]
            if not planned:
                self.state["review_stop_reason"]="no_eligible_review_after_attempt_and_input_guards"
                break
            # Work on one group (and all its dependent pieces), then recompute obligations.
            first=planned[0]
            keys={c["key"] for c in first["checks"]}
            chosen=[t for t in planned if any(c["key"] in keys for c in t["checks"])]
            remaining=min(self.config.max_refinements-self.runtime.review_usage(),
                          self.runtime.budget.max_model_calls-self.runtime.usage()["model_calls"]-1)
            if len(chosen)>remaining:
                # Joint judgments need every segment. Do not spend the budget on a
                # prefix that cannot support a commit. Other affordable tasks continue.
                self.state.setdefault("skipped_reviews",[]).append({"task":first["key"],
                    "reason":"joint review exceeds remaining call allowance","required_calls":len(chosen),"remaining_calls":remaining})
                self.state.setdefault("review_attempts",[]).extend(keys-attempted)
                self.runtime.changed()
                continue
            entries=[{"task":t,"done":False} for t in chosen]
            self.state["reviews"].extend(entries)
            self.runtime.changed()
            for entry in entries:
                self.execute_check(entry,q)
            self.read_selected_attributes(q,reduce_query(q,self.state,scope),scope)
        reduction=reduce_query(q,self.state,scope)
        if not reduction["closed"] and self.runtime.review_usage()>=self.config.max_refinements:
            self.state["review_stop_reason"]="review_call_limit_reached"
        return self.final(q,reduction,scope)
