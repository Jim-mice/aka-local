# Phase 14-D integration boundary

The original Episode 2 source is immutable. `integration_adapter.cu` is a
stream-aware integration variant whose only semantic change is accepting an
explicit CUDA stream and launching the same activation kernel on it.
