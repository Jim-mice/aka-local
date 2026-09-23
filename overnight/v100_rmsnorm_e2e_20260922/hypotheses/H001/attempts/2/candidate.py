"""H001 attempt 2: same fused mechanism with the QUICK-selected 8-warp mapping."""

from candidate_base import TritonRMSNorm as _Attempt1RMSNorm


class TritonRMSNorm(_Attempt1RMSNorm):
    def __init__(self, hidden_size=1024, eps=1.0e-5, num_warps=8):
        self.requested_num_warps = int(num_warps)
        super().__init__(hidden_size=hidden_size, eps=eps, num_warps=8)
        self.mapping_note = "H001 attempt 2 pins the QUICK-selected 8-warp mapping"
