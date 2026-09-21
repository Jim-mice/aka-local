# Agent boundary

The LOCAL RTX5060 RMSNorm path runs one bounded persistent `CodexAgentSession`
only after a user approves a Workbench and confirms Start in the Tk GUI.  Before
every experiment the runner writes a knowledge snapshot and context snapshot.
The Agent may modify only the approved candidate root; the Controller verifies
the resulting path delta and protects incumbent/reference/knowledge/supervisor
paths.  The Agent cannot write `supervisor_decision` or promote an incumbent.
