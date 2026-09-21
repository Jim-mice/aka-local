"""Deterministic policy interface; no Agent calls."""
def decide(correctness_pass: bool, improvement: float | None, threshold: float = 0.0) -> str:
 if not correctness_pass: return 'REJECT'
 if improvement is None: return 'INCONCLUSIVE'
 return 'ACCEPT' if improvement > threshold else 'REJECT'
