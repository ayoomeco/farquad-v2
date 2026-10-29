---
name: verify-before-done
description: >-
  Use before telling the user anything is done, working, fixed or ready: forces
  evidence from this turn and a Working / Built-untested / Researched-only label
  on every claim.
---
# Verify before done
Adapted from obra/superpowers verification-before-completion. Claims that sound verified but aren't are the costliest errors.
1. List each claim you're about to make.
2. For each, name the evidence seen THIS turn (tool output, file path, screenshot, test pass). No evidence means downgrade the label.
3. Labels: WORKING (ran end-to-end, output inspected), BUILT-UNTESTED, RESEARCHED-ONLY.
4. After any action on external state (pod terminated, template listed, file uploaded), read it back to confirm.
5. Mark any number that came from an estimate as an estimate.
6. Verify subagent or other-AI claims before relaying them.
