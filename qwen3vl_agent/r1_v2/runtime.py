from qwen3vl_agent.r1.runtime import ModelSession
from qwen3vl_agent.r1_v2.prompts import prompt, repair_prompt


class R1V2ModelSession(ModelSession):
    def _prompt(self, role, payload):
        return prompt(role, payload)

    def _repair_prompt(self, role, raw, error, payload):
        return repair_prompt(role, raw, error, payload)
