# k8s-ai-agent

A natural-language command-line agent for Kubernetes / AWS EKS, built with
LangChain and Google Gemini. Type plain English ("show pods in apps",
"why is pod X restarting?") and the agent picks and runs the right `kubectl`
command, then summarises the result.

## Features
- **Tools:** get, describe, logs, create, scale, delete, monitor (`top` / events),
  expose (ClusterIP / NodePort / LoadBalancer)
- **Safety:** create / scale / delete / expose print the exact command and ask
  `Proceed? (y/N)` first; live `--watch` / `--follow` flags are stripped
- **EKS cluster switching:** `list all clusters`, `switch to <name>`,
  `current cluster`, `sync clusters` (handled directly, no AI call)
- **Gemini API key:** from `GOOGLE_API_KEY`, or per-user from AWS Secrets Manager
  (created on first run via a hidden prompt)

## Quick start (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r req.txt
$env:GOOGLE_API_KEY = Read-Host "Paste Gemini API key"   # or use AWS Secrets Manager
.\.venv\Scripts\python myagent.py
```
Inside the agent type `help` for example queries.

Full guide: [setup.txt](setup.txt) - FAQ: [QnA.txt](QnA.txt)

## Data privacy
Your questions **and the kubectl output** (pod names, logs, events, YAML) are
sent to Google Gemini. Don't request Secret contents or logs containing
credentials or customer data. AI answers can be wrong - check the raw
`Observation:` output.

## Credits
Based on the tutorial by Syed Jibbran Ali -
https://github.com/JibbranAli/Agent_FOR_K8s
