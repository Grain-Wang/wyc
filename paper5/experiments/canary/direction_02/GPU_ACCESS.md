# D2 GPU access gate — operational supplement, 2026-09-23

The user clarified that screenshots and free VRAM are not a resource allocation.
This supplement changes only execution admission; the Stage0 scientific
protocol, fixed candidates/data, recipe, quality proposal and budgets are unchanged.

The server connection is available, but no explicit D2 co-tenancy permission has
been established. Historical paper1 shared-GPU examples and generic queue
documentation do not grant D2 permission. All GPUs were busy at the live check;
therefore no pilot or formal run was launched.

Before any CUDA initialization or GPU forward:

1. Verify the server allocation or usage permission for this task and stage.
2. Check for an existing D2 GPU runner/launcher and refuse duplicate submission.
3. Query live A800 memory, utilization and compute-process count. A GPU with the
   largest free memory can still be fully occupied; resource ranking is only a
   recommendation within the permitted set.
4. For exclusive access, require no compute processes, memory used <1024 MiB
   and utilization <5% in four samples 30 seconds apart, followed by a final
   check in the Python GPU entry point. For shared access, require explicit
   co-tenancy permission for the specific GPU(s); memory headroom remains
   mandatory and shared timings must be labeled as such.
5. Only then run the preplanned single-candidate ten-update pilot. Formal
   execution additionally requires the existing pilot/quality/full-budget gate
   and committed formal freeze. No candidate, seed or model changes for capacity.

The launcher and direct Python entry both require `D2_GPU_PERMISSION_FILE`.
This is a private reviewed allocation record in the ignored D2 cache, with:
`run_id`, `mode` (`exclusive` or `shared`), `gpu_indices`, permitted `stages`,
an `evidence` reference to the actual permission, and timezone-aware
`expires_at_utc`. The file records independently verified permission; writing a
file does not create permission. The agent must never fill it based solely on
free VRAM, an old unrelated authorization, or the availability of SSH login.
No real permission file has been created in this task.

No process termination, resource preemption, reservation of another user's GPU,
or automatic delayed training is authorized by this CPU-preparation continuation.
Private permission details stay out of Git; published preflight metadata may
contain the mode and record hash only.
