# cost_estimator.py
from dataclasses import dataclass

# Prices in USD per 1 million tokens (as of early 2026)
# Source: anthropic.com/pricing and openai.com/api/pricing
PRICING = {
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-haiku-4": {"input": 0.80, "output": 4.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "llama3.2:3b": {"input": 0.00, "output": 0.00},  # local = free
}


@dataclass
class CostEstimate:
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float

    def __str__(self) -> str:
        if self.total_cost_usd == 0:
            return (
                f"model: {self.model} (local — free)\n"
                f"tokens → in: {self.input_tokens:,} | out: {self.output_tokens:,} | "
                f"total: {self.input_tokens + self.output_tokens:,}"
            )
        return (
            f"model: {self.model}\n"
            f"tokens  → in: {self.input_tokens:,}  out: {self.output_tokens:,}\n"
            f"cost    → in: ${self.input_cost_usd:.6f}  "
            f"out: ${self.output_cost_usd:.6f}  "
            f"total: ${self.total_cost_usd:.6f}"
        )


def estimate(
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
) -> CostEstimate:
    """
    Calculate cost for a given model and token counts.
    output_tokens=0 means pre-send estimation (you don't know output yet).
    """
    rates = PRICING.get(model, {"input": 0.0, "output": 0.0})

    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]

    return CostEstimate(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
    )


class SessionCostTracker:
    """Tracks cumulative cost across all turns in a session."""

    def __init__(self, model: str):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.turns = 0

    def record(self, input_tokens: int, output_tokens: int) -> CostEstimate:
        result = estimate(self.model, input_tokens, output_tokens)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += result.total_cost_usd
        self.turns += 1
        return result

    def reset(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.turns = 0

    def summary(self) -> str:
        equivalent = []
        # show what this session would cost on paid models for perspective
        for model_name in [
            "gpt-4o-mini",
            "claude-haiku-4",
            "gpt-4o",
            "claude-sonnet-4",
        ]:
            eq = estimate(model_name, self.total_input_tokens, self.total_output_tokens)
            equivalent.append(f"  {model_name:<20} ${eq.total_cost_usd:.6f}")

        lines = [
            f"Session: {self.turns} turns | "
            f"{self.total_input_tokens:,} in + {self.total_output_tokens:,} out tokens",
            f"Actual cost ({self.model}): FREE (local)",
            "",
            "Equivalent cost on cloud models:",
            *equivalent,
        ]
        return "\n".join(lines)
