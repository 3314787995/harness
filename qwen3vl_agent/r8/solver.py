"""Checked polynomial AST -> Z3 in a bounded, disposable CPU process."""

from __future__ import annotations

import importlib.metadata
import multiprocessing
import operator
import time
from copy import deepcopy
from fractions import Fraction

from .config import SOLVER_VERSION
from .evidence import DEFINITIONS
from .geometry import apply_rules
from .types import ModelingError, ProtocolError
from .units import Quantity, Unit, number


def prepare(program, store, config):
    if set(program) != {"symbols", "constraints", "rules", "target", "output_unit"}:
        raise ModelingError("invalid constraint program")
    if len(program["symbols"]) > config.max_query_nodes or len(program["constraints"]) > 128:
        raise ModelingError("constraint size limit")
    data = deepcopy(program)
    symbols, parents, bindings = {}, [], {}
    for symbol in data["symbols"]:
        if set(symbol) != {"id", "entity_id", "attribute", "unit", "domain", "sources", "snapshot"}:
            raise ModelingError("invalid symbol declaration")
        key = symbol["id"]
        if key in symbols or not key or len(key) > 160:
            raise ModelingError("duplicate/invalid solver symbol")
        Unit.parse(symbol["unit"])
        if symbol["domain"] not in {"real", "integer", "positive", "nonnegative"}:
            raise ModelingError("unsupported symbol domain")
        if key in store.state["current"] or key in store.state["variables"]:
            row = store.get(key)
            if row["alternatives"] or row["unresolved"]:
                raise ModelingError("ambiguous bound measurement")
            if any(row[k] != symbol[k] for k in ("entity_id", "attribute", "snapshot")):
                raise ModelingError("symbol measurement binding mismatch")
            quantity = Quantity.make(row["value"], row["unit"], row["unit_basis"])
            quantity.convert(symbol["unit"])
            bindings[key] = str(quantity.base)
            parents.append(f"{row['id']}@{row['version']}")
        else:
            entity = store.state["entities"].get(symbol["entity_id"])
            if not entity or entity["snapshot"] != symbol["snapshot"]:
                raise ModelingError("unknown/cross-snapshot latent geometry object")
        if not symbol["sources"]:
            raise ModelingError("solver symbol requires provenance")
        for source in symbol["sources"]:
            if source in store.state["relations"]:
                r = store.state["relations"][source]
                if (
                    r["snapshot"] != symbol["snapshot"]
                    or symbol["entity_id"] not in r["objects"].values()
                ):
                    raise ModelingError("symbol relation does not identify its object")
            else:
                row = store.get(source)
                if row["entity_id"] != symbol["entity_id"] or row["snapshot"] != symbol["snapshot"]:
                    raise ModelingError("symbol source binding mismatch")
        if key not in bindings and symbol["domain"] in {"positive", "nonnegative", "integer"}:
            support = [store.state["relations"].get(s, {}) for s in symbol["sources"]]
            if symbol["domain"] in {"positive", "nonnegative"} and not any(
                r.get("kind") in {"positive", "nondegenerate"} for r in support
            ):
                raise ModelingError("latent sign domain lacks a premise")
            if symbol["domain"] == "integer" and not any(
                k.startswith("count:") for k, v in Unit.parse(symbol["unit"]).dimensions
            ):
                raise ModelingError("integer domain requires a count quantity")
        symbols[key] = symbol
    generated, trace = apply_rules(data, store.state["relations"], config)
    data["constraints"].extend(generated)
    used_relations = {
        r for c in data["constraints"] for r in c["sources"] if r in store.state["relations"]
    }
    for rid in used_relations:
        versions = store.state["relation_versions"].get(rid, [])
        if versions:
            parents.append(f"relation:{rid}@{len(versions)}")
    ids, node_count = set(), 0

    def term(t, depth=0):
        nonlocal node_count
        node_count += 1
        if node_count > config.max_query_nodes * 32 or depth > config.max_query_depth:
            raise ModelingError("constraint expression limit")
        if not isinstance(t, dict):
            raise ModelingError("constraint term requires whitelisted AST")
        if set(t) == {"ref"} and t["ref"] in symbols:
            return Unit.parse(symbols[t["ref"]]["unit"])
        if set(t) == {"constant"} and t["constant"] in DEFINITIONS:
            return Unit.parse(DEFINITIONS[t["constant"]][1])
        if set(t) != {"op", "args"} or t["op"] not in {
            "add",
            "subtract",
            "multiply",
            "divide",
            "square",
        }:
            raise ModelingError("unsupported constraint term")
        args = t["args"]
        if not isinstance(args, list) or len(args) != (1 if t["op"] == "square" else 2):
            raise ModelingError("constraint arity mismatch")
        a = term(args[0], depth + 1)
        if t["op"] == "square":
            return a.combine(a)
        b = term(args[1], depth + 1)
        if t["op"] in {"add", "subtract"}:
            if a.dimensions != b.dimensions:
                raise ModelingError("constraint sum unit mismatch")
            return a
        return a.combine(b, 1 if t["op"] == "multiply" else -1)

    def refs(t):
        if "ref" in t:
            return [t["ref"]]
        return [v for child in t.get("args", []) for v in refs(child)]

    for constraint in data["constraints"]:
        if (
            set(constraint) != {"id", "relation", "args", "sources", "snapshot"}
            or constraint["id"] in ids
        ):
            raise ModelingError("invalid/duplicate constraint")
        ids.add(constraint["id"])
        if (
            constraint["relation"]
            not in {"equal", "less_equal", "greater_equal", "less", "greater", "not_equal"}
            or len(constraint["args"]) != 2
        ):
            raise ModelingError("unsupported constraint relation")
        if not constraint["sources"]:
            raise ModelingError("unsourced constraint")
        for source in constraint["sources"]:
            if source in store.state["relations"]:
                row = store.state["relations"][source]
            else:
                row = store.get(source)
            if row["snapshot"] != constraint["snapshot"]:
                raise ModelingError("constraint sources cross snapshots")
        a, b = [term(t) for t in constraint["args"]]
        zero = any(t == {"constant": "zero"} for t in constraint["args"])
        if a.dimensions != b.dimensions and not zero:
            raise ModelingError("constraint sides have different units")
        if any(
            symbols[r]["snapshot"] != constraint["snapshot"]
            for t in constraint["args"]
            for r in refs(t)
        ):
            raise ModelingError("constraint expression crosses snapshots")
    target_unit = term(data["target"])
    if target_unit.dimensions != Unit.parse(data["output_unit"]).dimensions:
        raise ModelingError("constraint target unit mismatch")
    if len({symbols[r]["snapshot"] for r in refs(data["target"])}) > 1:
        raise ModelingError("target crosses snapshots")
    data.update(
        bindings=bindings,
        timeout_ms=max(1, int(config.solver_seconds * 1000)),
        numeric_digits=config.max_numeric_digits,
    )
    return data, parents, trace


def _solve(data):
    import z3

    solver = z3.Solver()
    solver.set(timeout=data["timeout_ms"])
    clauses = []

    def track(predicate, label):
        clauses.append(predicate)
        solver.assert_and_track(predicate, label)

    symbols = {
        s["id"]: (z3.Int(s["id"]) if s["domain"] == "integer" else z3.Real(s["id"]))
        for s in data["symbols"]
    }
    guards = []

    def rational(value):
        v = number(value, data["numeric_digits"])
        return z3.RealVal(f"{v.numerator}/{v.denominator}")

    def term(t):
        if "ref" in t:
            return symbols[t["ref"]]
        if "constant" in t:
            value, unit = DEFINITIONS[t["constant"]]
            return rational(str(number(value) * Unit.parse(unit).factor))
        a = term(t["args"][0])
        name = t["op"]
        if name == "square":
            return a * a
        b = term(t["args"][1])
        if name == "add":
            return a + b
        if name == "subtract":
            return a - b
        if name == "multiply":
            return a * b
        guards.append(b != 0)
        return a / b

    for symbol in data["symbols"]:
        x = symbols[symbol["id"]]
        if symbol["id"] in data["bindings"]:
            track(x == rational(data["bindings"][symbol["id"]]), "binding:" + symbol["id"])
        if symbol["domain"] in {"positive", "nonnegative"}:
            track(x > 0 if symbol["domain"] == "positive" else x >= 0, "domain:" + symbol["id"])
    for c in data["constraints"]:
        a, b = [term(t) for t in c["args"]]
        predicate = {
            "equal": operator.eq,
            "less_equal": operator.le,
            "greater_equal": operator.ge,
            "less": operator.lt,
            "greater": operator.gt,
            "not_equal": operator.ne,
        }[c["relation"]](a, b)
        track(predicate, c["id"])
    target = term(data["target"])
    for i, guard in enumerate(guards):
        track(guard, f"nonzero:{i}")
    check = solver.check()
    if check == z3.unsat:
        return {
            "status": "inconsistent_constraints",
            "conflict_core": [str(v) for v in solver.unsat_core()],
        }
    if check != z3.sat:
        return {"status": "timeout_or_unknown", "reason": solver.reason_unknown()}
    model = solver.model()
    y0 = z3.simplify(model.eval(target, model_completion=True))
    # A fresh solver avoids Z3's incremental assumption path returning incomplete arithmetic
    # for algebraic disequalities. It receives the identical clauses, without tracking proxies.
    unique_solver = z3.Solver()
    unique_solver.set(timeout=data["timeout_ms"])
    unique_solver.add(*clauses, target != y0)
    uniqueness = unique_solver.check()
    if uniqueness == z3.sat:
        return {
            "status": "ambiguous_target",
            "witnesses": [str(y0), str(unique_solver.model().eval(target, model_completion=True))],
        }
    if uniqueness != z3.unsat:
        return {
            "status": "timeout_or_unknown",
            "reason": unique_solver.reason_unknown(),
            "phase": "target_uniqueness",
        }
    scale = Unit.parse(data["output_unit"]).factor
    value = z3.simplify(y0 / rational(str(scale)))
    if z3.is_rational_value(value):
        exact = f"{value.numerator_as_long()}/{value.denominator_as_long()}"
        exact = str(number(exact, data["numeric_digits"]))
        result = {"kind": "rational", "exact": exact, "lower": exact, "upper": exact}
    elif z3.is_algebraic_value(value):
        # Z3's approx returns an upper rational bound with error <= 10**(-precision).
        upper_z3 = value.approx(30)
        upper = Fraction(upper_z3.numerator_as_long(), upper_z3.denominator_as_long())
        lower = upper - Fraction(1, 10**30)
        result = {
            "kind": "algebraic",
            "exact": value.sexpr(),
            "lower": str(lower),
            "upper": str(upper),
            "decimal": value.as_decimal(24),
            "approximation_error_max": "1/" + str(10**30),
        }
    else:
        return {
            "status": "unsupported_theory",
            "reason": "target is not an exact real algebraic value",
        }
    return {
        "status": "solved_target",
        "value": result,
        "unit": data["output_unit"],
        "satisfiable": True,
        "target_unique": True,
        "uniqueness_method": "C AND target != model(target) is UNSAT",
    }


def _worker(connection, data):
    try:
        connection.send(_solve(data))
    except Exception as exc:  # noqa: BLE001 -- process boundary serializes solver failures
        connection.send({"status": "unsupported_theory", "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def solve_constraints(program, store, config):
    started = time.perf_counter()
    try:
        installed = importlib.metadata.version("z3-solver")
        if installed != SOLVER_VERSION:
            raise ModelingError(f"requires z3-solver=={SOLVER_VERSION}; found {installed}")
        data, parents, trace = prepare(program, store, config)
    except (
        ModelingError,
        ProtocolError,
        importlib.metadata.PackageNotFoundError,
        KeyError,
        TypeError,
    ) as exc:
        return {"status": "unsupported_theory", "reason": str(exc), "parents": []}
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, data), daemon=True)
    process.start()
    sender.close()
    try:
        remaining = max(0.0, config.solver_seconds - (time.perf_counter() - started))
        if receiver.poll(remaining):
            try:
                result = receiver.recv()
            except EOFError:
                result = {
                    "status": "timeout_or_unknown",
                    "reason": "solver worker exited without result",
                }
        else:
            result = {"status": "timeout_or_unknown", "reason": "CPU worker hard deadline"}
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
    result.update(
        parents=parents,
        rule_trace=trace,
        elapsed_seconds=time.perf_counter() - started,
        solver_version=installed,
        constraints=data["constraints"],
    )
    store.add_derived("constraint:target", deepcopy(result), parents, kind="constraint_execution")
    return result
