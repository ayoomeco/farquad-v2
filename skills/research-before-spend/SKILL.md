---
name: research-before-spend
description: >-
  Use before any paid action (GPU pod, credits, subscription, purchase): checks
  newest tools, prior working recipes, live pricing and a cost cap, then asks
  the user with one clear approval line.
---
# Research before spend
1. Search for the newest tool/version and someone else's working recipe (GitHub issues, Reddit, X, docs), ideally from the last 60 days.
2. Fetch the live pricing page or API; record price and date. Note constraints (region, tier, license, which plan a feature needs).
3. Dry-run everything free first: configs, workflow validation, model URLs HEAD-checked.
4. Compute expected cost, hard cap, retry allowance, and auto-stop method.
5. Ask once: "Approve X for ~$Y (cap $Z)?" Never start billable work without an explicit yes.
6. After the run, verify resources are terminated and report actual spend plus remaining balance.
