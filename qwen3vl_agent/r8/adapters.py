"""R8 records for attempts, inventories, cash ledgers and alternative scoring rules."""

from copy import deepcopy
from fractions import Fraction

from .query_ir import typed
from .types import ModelingError, ProtocolError, digest
from .units import IntervalQuantity, Unit
from .units import Quantity as Q


class Adapters:
    def __init__(self, store, state=None):
        self.store = store
        self.state = state if state is not None else {}
        for key in ("attempts", "items", "transactions", "record_versions"):
            self.state.setdefault(key, {})
        self.state.setdefault("applied", [])
        self.state.setdefault("record_refs", {})

    def ingest(self, observation, packet, key):
        if key in self.state["applied"]:
            return
        state = deepcopy(self.state)
        for kind in ("attempts", "items", "transactions"):
            for original in observation[kind]:
                row = deepcopy(original)
                row["evidence_refs"] = self.store.sources(row["evidence_refs"], packet=packet)
                if kind == "attempts":
                    if not self.store.contract.permits(row["start_time"]):
                        raise ProtocolError("attempt start exceeds permission")
                    if row["end_time"] is not None and (
                        row["end_time"] < row["start_time"]
                        or not self.store.contract.permits_span(
                            (row["start_time"], row["end_time"])
                        )
                    ):
                        raise ProtocolError("attempt crosses observation boundary")
                    stable = ("actor_id", "scope", "round_id", "start_time")
                elif kind == "transactions":
                    if not self.store.contract.permits(row["time"]) or row["payer"] == row["payee"]:
                        raise ProtocolError("invalid transaction parties/time")
                    self.store.get(row["amount_ref"])
                    stable = ("payer", "payee", "scope", "snapshot", "stage", "time", "nature")
                else:
                    if row["entity_id"] not in self.store.state["entities"]:
                        raise ProtocolError("inventory object not bound")
                    for ref in row["attributes"].values():
                        variable = self.store.get(ref)
                        if any(variable[k] != row[k] for k in ("entity_id", "scope", "snapshot")):
                            raise ProtocolError("inventory attribute binding mismatch")
                    stable = ("entity_id", "scope", "snapshot")
                old = state[kind].get(row["id"])
                if old and any(old[k] != row[k] for k in stable):
                    raise ProtocolError("adapter record ID rebound")
                if old != row:
                    state["record_versions"].setdefault(kind + ":" + row["id"], []).append(row)
                    state[kind][row["id"]] = row
                    record_key = kind + ":" + row["id"]
                    previous = state["record_refs"].get(record_key)
                    if previous:
                        self.store.invalidate(previous)
                        self.store.state["derived"][previous]["valid"] = False
                    ref = f"adapter_record:{record_key}@{len(state['record_versions'][record_key])}"
                    self.store.state["derived"][ref] = {
                        "value": deepcopy(row),
                        "parents": [],
                        "valid": True,
                        "origin": "observed_adapter",
                    }
                    state["record_refs"][record_key] = ref
        state["applied"].append(key)
        self.state.clear()
        self.state.update(state)

    def record_parents(self, kind, scope):
        refs = []
        for row in self.state[kind].values():
            if row["scope"] != scope:
                continue
            key = kind + ":" + row["id"]
            if key not in self.state["record_refs"]:
                # Programmatic imports are still source-checked; they cannot bypass the binder.
                self.store.sources(row["evidence_refs"])
                ref = f"adapter_record:{key}@1"
                self.state["record_refs"][key] = ref
                self.store.state["derived"][ref] = {
                    "value": deepcopy(row),
                    "parents": [],
                    "valid": True,
                    "origin": "observed_adapter",
                }
            refs.append(self.state["record_refs"][key])
        return refs

    def _dedup(self, kind, rows):
        all_rows = self.state[kind]
        seen, result = set(), []
        for row in rows:
            origin, trail = row, set()
            while origin.get("replay_of") or origin.get("duplicate_of"):
                key = origin.get("replay_of") or origin.get("duplicate_of")
                if key in trail or key not in all_rows:
                    raise ModelingError("unresolved/cyclic replay or duplicate link")
                trail.add(key)
                target = all_rows[key]
                if target["scope"] != row["scope"]:
                    raise ModelingError("duplicate identity crosses scope")
                stable = {
                    "attempts": ("actor_id", "round_id"),
                    "items": ("entity_id", "snapshot"),
                    "transactions": ("payer", "payee", "nature", "amount_ref"),
                }[kind]
                if any(target[k] != row[k] for k in stable):
                    raise ModelingError(
                        "duplicate identity crosses actor/object/accounting binding"
                    )
                origin = target
            if origin["id"] not in seen:
                seen.add(origin["id"])
                result.append(origin)
        return result

    def attempts(self, query, complete):
        rows = [
            r
            for r in self.state["attempts"].values()
            if r["scope"] == query["scope"]
            and (not query["actor"] or r["actor_id"] == query["actor"])
            and (not query["round_id"] or r["round_id"] == query["round_id"])
        ]
        rows = self._dedup("attempts", rows)
        selected, unknown_boundary = [], False
        for row in rows:
            start, end = row["start_time"], row["end_time"]
            window = query["window"]
            if window:
                a, b = window
                if query["boundary"] == "start":
                    include = a <= start < b
                elif end is None:
                    unknown_boundary = True
                    continue
                elif query["boundary"] == "end":
                    include = a < end <= b
                else:
                    include = a <= start and end <= b
                if not include:
                    continue
            selected.append(row)
        known_total = complete and not unknown_boundary
        successes = sum(r["outcome"] == "success" for r in selected)
        unknown = sum(r["outcome"] == "unknown" for r in selected)
        # An unclosed attempt cannot simultaneously count as known success and unknown.
        unclosed = [r for r in selected if r["end_time"] is None]
        if unclosed:
            known_total = False
        total = len(selected)
        interval = (
            [Fraction(successes, total), Fraction(successes + unknown, total)]
            if known_total and total
            else None
        )
        return {
            "rows": selected,
            "total": total if known_total else None,
            "observed_attempts": total,
            "successes": successes,
            "unknown": unknown,
            "ratio_interval": interval,
            "complete": known_total,
            "boundary": query["boundary"],
        }

    def inventory(self, query, complete):
        rows = self._dedup(
            "items", [r for r in self.state["items"].values() if r["scope"] == query["scope"]]
        )
        output, parents, unreadable = [], [], []
        for row in rows:
            attrs = {"id": row["id"], "entity_id": row["entity_id"]}
            if not row["readable"]:
                unreadable.append(row["id"])
            for key, ref in row["attributes"].items():
                value = self.store.get(ref)
                if value["value"] is None or value["unresolved"] or value["alternatives"]:
                    unreadable.append(row["id"])
                    continue
                attrs[key] = typed(value["value"], value["unit"], value["unit_basis"])
                parents.append(f"{value['id']}@{value['version']}")
            if query["attribute"] and query["attribute"] not in attrs:
                unreadable.append(row["id"])
            output.append(attrs)
        return {
            "rows": output,
            "parents": sorted(set(parents)),
            "complete": complete and not unreadable,
            "unreadable": sorted(set(unreadable)),
        }

    def transactions(self, query, complete):
        if not complete:
            raise ModelingError("transaction stage coverage incomplete")
        rows = self._dedup(
            "transactions",
            [
                r
                for r in self.state["transactions"].values()
                if r["scope"] == query["scope"]
                and (not query["round_id"] or r["stage"] == query["round_id"])
                and (not query["window"] or query["window"][0] <= r["time"] <= query["window"][1])
            ],
        )
        actor, total, parents, entries = query["actor"], None, [], []
        operation = query["operation"]
        if operation not in {"cashflow", "balance", "liability", "receivable", "realized_profit"}:
            raise ModelingError("transaction operation must name a specific accounting basis")
        for row in rows:
            if actor not in {row["payer"], row["payee"]}:
                continue
            value = self.store.get(row["amount_ref"])
            amount = Q.make(value["value"], value["unit"], value["unit_basis"])
            if (
                len(amount.unit.dimensions) != 1
                or not amount.unit.dimensions[0][0].startswith("currency:")
                or amount.base < 0
            ):
                raise ModelingError("transaction amount must be nonnegative single-currency money")
            if total is None:
                total = Q(Fraction(0), amount.unit, amount.basis)
            total.compatible(amount)
            incoming = row["payee"] == actor
            sign = 1 if incoming else -1
            if operation == "liability":
                sign = (
                    1
                    if row["nature"] == "loan" and incoming
                    else -1
                    if row["nature"] == "repayment" and not incoming
                    else 0
                )
            elif operation == "receivable":
                sign = (
                    1
                    if row["nature"] == "loan" and not incoming
                    else -1
                    if row["nature"] == "repayment" and incoming
                    else 0
                )
            elif operation == "realized_profit":
                sign = 1 if row["nature"] == "sale" and incoming else 0
            total = Q(
                total.value + sign * amount.convert(total.unit).value, total.unit, total.basis
            )
            parents.append(f"{value['id']}@{value['version']}")
            entries.append({"id": row["id"], "sign": sign, "nature": row["nature"]})
        if total is None:
            raise ModelingError("no sourced monetary unit/entries")
        if operation in {"balance", "liability", "receivable", "realized_profit"}:
            if not query["initial_ref"]:
                raise ModelingError(
                    "balance/debt need opening amount; realized profit needs matched sold-goods cost basis"
                )
            row = self.store.get(query["initial_ref"])
            required_role = (
                "sold_goods_cost_basis"
                if operation == "realized_profit"
                else "opening_" + operation
            )
            if row["role"] != required_role or row["entity_id"] != actor:
                raise ModelingError("accounting initial/cost basis role mismatch")
            initial = Q.make(row["value"], row["unit"], row["unit_basis"])
            total.compatible(initial)
            total = Q(
                total.value
                + (-1 if operation == "realized_profit" else 1) * initial.convert(total.unit).value,
                total.unit,
                total.basis,
            )
            parents.append(f"{row['id']}@{row['version']}")
        return {
            "value": total,
            "parents": parents,
            "entries": entries,
            "accounting_basis": operation,
        }

    def replace_rule(self, query, complete):
        text = query["rule_span"]
        attributes = [v.strip() for v in query["attribute"].split(",") if v.strip()]
        if (
            not text
            or text not in self.store.question
            or not attributes
            or query["operation"] not in {"all_true", "any_true"}
        ):
            raise ModelingError(
                "alternative rule needs an exact question stipulation and whitelisted Boolean predicate"
            )
        base = self.attempts(query, complete)
        rows = deepcopy(base["rows"])
        for row in rows:
            values = [row["attributes"].get(a) for a in attributes]
            if query["operation"] == "all_true":
                outcome = False if False in values else None if None in values else True
            else:
                outcome = True if True in values else None if None in values else False
            row.update(
                outcome="unknown" if outcome is None else "success" if outcome else "failure",
                hypothetical=True,
                rule_span=text,
            )
        base["rows"] = rows
        base["successes"] = sum(r["outcome"] == "success" for r in rows)
        base["unknown"] = sum(r["outcome"] == "unknown" for r in rows)
        if base["total"]:
            base["ratio_interval"] = [
                Fraction(base["successes"], base["total"]),
                Fraction(base["successes"] + base["unknown"], base["total"]),
            ]
        return base

    def execute(self, queries, *, coverage_complete):
        output, trace = {}, []
        for query in queries:
            if query["id"] in output:
                raise ModelingError("duplicate adapter output ID")
            if query["window"] and not self.store.contract.permits_span(query["window"]):
                raise ProtocolError("adapter query exceeds observation permission")
            complete = coverage_complete and query["complete_claim"]
            if query["kind"] in {"attempts", "replace_rule"}:
                result = (
                    self.attempts(query, complete)
                    if query["kind"] == "attempts"
                    else self.replace_rule(query, complete)
                )
                if not result["complete"]:
                    raise ModelingError("missing_denominator: total attempts not established")
                value = {
                    "total": Q.make(result["total"], "attempt"),
                    "successes": IntervalQuantity(
                        Fraction(result["successes"]),
                        Fraction(result["successes"] + result["unknown"]),
                        Unit.parse("attempt"),
                    )
                    if result["unknown"]
                    else Q.make(result["successes"], "attempt"),
                    "unknown": Q.make(result["unknown"], "attempt"),
                    "ratio_interval": IntervalQuantity(*result["ratio_interval"])
                    if result["ratio_interval"]
                    else [],
                }
                parents = []
            elif query["kind"] == "inventory":
                result = self.inventory(query, complete)
                if not result["complete"]:
                    raise ModelingError("incomplete_extrema_set: unreadable or uncovered items")
                value, parents = result["rows"], result["parents"]
            else:
                result = self.transactions(query, complete)
                value, parents = result["value"], result["parents"]
            kind = (
                "attempts"
                if query["kind"] == "replace_rule"
                else "items"
                if query["kind"] == "inventory"
                else query["kind"]
            )
            parents = sorted(set(parents + self.record_parents(kind, query["scope"])))
            output[query["id"]] = {"value": value, "parents": parents}
            trace.append(
                {"query": deepcopy(query), "result": result, "record_hash": digest(self.state)}
            )
        return output, trace
