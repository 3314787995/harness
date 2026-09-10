from qwen3vl_agent.r1.control import BudgetExhausted
from qwen3vl_agent.r1_v2.runtime import R1V2ModelSession
from qwen3vl_agent.r1_v3.prompts import prompt, repair_prompt


class R1V3ModelSession(R1V2ModelSession):
    unrepaired_roles = frozenset({"observe", "candidate_review", "final", "terminal_review"})
    observer_roles = frozenset({"observe", "binding", "candidate_review"})

    @property
    def terminal_calls(self):
        return [c for c in self.context.calls if c["role"] in {"final", "terminal_review"}]

    def _invoke(self, role, text, prepared, *, terminal, tokens):
        # Includes failed invocations and the inherited OOM retry. Pre-call budget failures
        # append no record, so they do not consume a terminal call.
        if terminal and len(self.terminal_calls) >= 2:
            raise BudgetExhausted("terminal_call_budget")
        return super()._invoke(role, text, prepared, terminal=terminal, tokens=tokens)

    def _prompt(self, role, payload):
        return prompt(role, payload)

    def _repair_prompt(self, role, raw, error, payload):
        return repair_prompt(role, raw, error, payload)
