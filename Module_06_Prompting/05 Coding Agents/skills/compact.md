# Context Compaction Instructions

## When to Use
Call `compact_context()` when:
- The conversation has more than 10 turns.
- The token count is approaching the model's context limit.
- The user asks to "clean up", "summarize so far", or "start fresh but keep memory".

## How to Compact
Produce a structured summary in this exact format:

---
## Conversation Summary (Compacted)

**Goal:** [One sentence describing what the user is trying to accomplish]

**Key Decisions Made:**
- [Decision 1]
- [Decision 2]

**Work Completed:**
- [Step or task completed]

**Current State:**
- [What file/code/data exists right now]
- [What the last tool call returned]

**Next Step:**
- [What should happen next]

---

## Rules
- Be concise. The summary must be under 300 tokens.
- Do not include raw code in the summary. Reference it by description.
- After compacting, drop all prior turns from context and continue with only this summary + the user's latest message.
- Tell the user: "Context compacted. Continuing from summary."
