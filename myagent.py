import subprocess
import tempfile
import yaml
import os
import json
import re
import shutil
import shlex
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import ClassVar


# ---- CONFIG ----
def load_google_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    import json
    import boto3
    session = boto3.Session(
        profile_name=os.environ.get("AWS_SECRET_PROFILE", "my-aws-profile"),
        region_name=os.environ.get("AWS_SECRET_REGION", "us-east-1"),
    )
    secret_id = os.environ.get("GOOGLE_API_KEY_SECRET")
    if not secret_id:
        user = session.client("sts").get_caller_identity()["Arn"].rsplit("/", 1)[-1].lower()
        secret_id = f"agent-for-k8s/{user}/google-api-key"
    print(f"Loading Gemini API key from AWS Secrets Manager: {secret_id}")
    sm = session.client("secretsmanager")
    try:
        secret = sm.get_secret_value(SecretId=secret_id)["SecretString"]
    except sm.exceptions.ResourceNotFoundException:
        import getpass
        print(f"Secret '{secret_id}' not found. One-time setup: it will be created now.")
        secret = getpass.getpass("Paste your Gemini API key and press Enter (input is hidden): ").strip()
        if not secret:
            raise SystemExit("No key entered. Exiting.")
        sm.create_secret(Name=secret_id, SecretString=secret)
        print(f"Saved to AWS Secrets Manager: {secret_id}")
    try:
        return json.loads(secret)["GOOGLE_API_KEY"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return secret

LLM = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=load_google_api_key(),
    temperature=0.1,
    max_tokens=2048
)

# Helper: Safety confirmation prompt
def confirm_action(prompt_text: str) -> bool:
    print(f"\n⚠️  [SAFETY CONFIRMATION] {prompt_text}")
    ans = input("Proceed? (y/N): ").strip().lower()
    return ans in ["y", "yes"]

# Tool 1: Get Resources
class GetResourcesTool(BaseTool):
    name: str = "get_resources"
    description: str = (
        "Execute 'kubectl get' to list resources. "
        "Input format: '<resource_type_or_subcommand> [name] [flags]'. "
        "Supports all kubectl resources (pods, services, deployments, configmaps, secrets, RBAC roles, nodes, namespaces, etc.) "
        "and all get flags (e.g. -o wide, -o yaml, -o json, -l app=nginx, --field-selector, --sort-by, -n <namespace>, -A, --kubeconfig, --context). "
        "Example inputs: 'pods -n default', 'deployments -o yaml', 'nodes --kubeconfig=/path/to/config'."
    )

    def _run(self, tool_input: str):
        args = shlex.split(tool_input.strip())
        has_ns = any(arg in ["-n", "--namespace"] or arg.startswith("--namespace=") for arg in args)
        has_all_ns = any(arg in ["-A", "--all-namespaces"] for arg in args)
        
        extra_args = []
        if not has_ns and not has_all_ns:
            extra_args = ["-n", "apps"]
            
        cmd = ["kubectl", "get"] + args + extra_args
        
        # Prevent live watch hanging
        if "--watch" in args or "-w" in args:
            cmd = [arg for arg in cmd if arg not in ["--watch", "-w"]]
            print("⚠️ Note: Live watch flags (--watch, -w) are not supported inside the agent and were stripped.")
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr

    def _arun(self, *args, **kwargs):
        raise NotImplementedError()

# Tool 2: Describe Resource
class DescribeResourceTool(BaseTool):
    name: str = "describe_resource"
    description: str = (
        "Execute 'kubectl describe' to inspect details and events of a resource. "
        "Input format: '<resource_type> <name> [flags]'. "
        "Supports all resources (pods, nodes, deployments, services, ingress, configmaps, namespaces, pvc, etc.) "
        "and flags like -n <namespace>, --kubeconfig, --context. "
        "Example inputs: 'pod auth-service -n apps', 'node my-node-1'."
    )

    def _run(self, tool_input: str):
        args = shlex.split(tool_input.strip())
        has_ns = any(arg in ["-n", "--namespace"] or arg.startswith("--namespace=") for arg in args)
        has_all_ns = any(arg in ["-A", "--all-namespaces"] for arg in args)
        
        extra_args = []
        if not has_ns and not has_all_ns:
            extra_args = ["-n", "apps"]
            
        cmd = ["kubectl", "describe"] + args + extra_args
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr

    def _arun(self, *args, **kwargs):
        raise NotImplementedError()

# Tool 3: Get Logs
class GetLogsTool(BaseTool):
    name: str = "get_logs"
    description: str = (
        "Execute 'kubectl logs' to retrieve pod container logs. "
        "Input format: '<pod_name_or_selector> [flags]'. "
        "Supports container selection (-c), --all-containers, --tail, --since, --timestamps, previous logs (-p), "
        "and label selector (-l app=nginx). "
        "Example inputs: 'my-pod-123 --tail=100 -n apps', '-l app=nginx --all-containers --prefix'."
    )

    def _run(self, tool_input: str):
        args = shlex.split(tool_input.strip())
        has_ns = any(arg in ["-n", "--namespace"] or arg.startswith("--namespace=") for arg in args)
        has_all_ns = any(arg in ["-A", "--all-namespaces"] for arg in args)
        
        extra_args = []
        if not has_ns and not has_all_ns:
            extra_args = ["-n", "apps"]
            
        cmd = ["kubectl", "logs"] + args + extra_args
        
        # Prevent follow logs hanging
        if "-f" in args or "--follow" in args:
            cmd = [arg for arg in cmd if arg not in ["-f", "--follow"]]
            print("⚠️ Note: Live follow logs (-f, --follow) are not supported inside the agent and were stripped.")
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr

    def _arun(self, *args, **kwargs):
        raise NotImplementedError()

# Tool 4: Create Resource
class CreateResourceTool(BaseTool):
    name: str = "create_resource"
    description: str = (
        "Execute 'kubectl create' or 'kubectl run' to create resources. "
        "Input format: '<subcommand> [flags]' or 'run <pod_name> --image=<image> [flags]' or '-f <file_or_dir_or_url> [flags]'. "
        "Supports declarative create (-f), run isolated pod (run), "
        "and imperative create (namespaces, deployments, services, configmaps, secrets, jobs, cronjobs, RBAC, ingress). "
        "Example inputs: 'namespace my-ns', 'deployment web --image=nginx --replicas=3 -n apps', 'service clusterip my-svc --tcp=80:80', 'run my-pod --image=nginx'."
    )

    def _run(self, tool_input: str):
        args = shlex.split(tool_input.strip())
        if not args:
            return "Error: No create subcommand or run specified."
            
        # Check if it is a "run" pod command
        is_run = args[0] == "run"
        base_cmd = ["kubectl", "run"] if is_run else ["kubectl", "create"]
        cmd_args = args[1:] if is_run else args
        
        has_ns = any(arg in ["-n", "--namespace"] or arg.startswith("--namespace=") for arg in cmd_args)
        has_all_ns = any(arg in ["-A", "--all-namespaces"] for arg in cmd_args)
        
        extra_args = []
        # Run doesn't allow all namespaces. Create usually is namespace specific or cluster-wide.
        if not has_ns and not has_all_ns:
            # Check if creating resource that is cluster-wide (like namespace, clusterrole, clusterrolebinding)
            cluster_wide = len(cmd_args) > 0 and cmd_args[0] in ["namespace", "ns", "clusterrole", "clusterrolebinding", "persistentvolume", "pv", "storageclass", "sc"]
            if not cluster_wide:
                extra_args = ["-n", "apps"]
                
        cmd = base_cmd + cmd_args + extra_args
        
        # Check if dry-run client is specified. If so, no confirmation needed.
        is_dry_run = any("dry-run" in arg for arg in cmd) or "-o" in cmd
        
        if not is_dry_run:
            prompt = f"Create resource with command: '{" ".join(cmd)}'?"
            if not confirm_action(prompt):
                return "Resource creation cancelled by user."
                
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr

    def _arun(self, *args, **kwargs):
        raise NotImplementedError()

# Tool 5: Scale Resource
class ScaleResourceTool(BaseTool):
    name: str = "scale_resource"
    description: str = (
        "Execute 'kubectl scale' to adjust replicas for deployments, statefulsets, replicasets, etc. "
        "Input format: '<resource_type_or_resource/name> --replicas=<num> [flags]'. "
        "Supports scaling multiple resources, and conditional scaling. "
        "Example inputs: 'deployment/orders-api --replicas=3 -n apps', 'statefulset/my-db --replicas=2', 'deployment app1 app2 --replicas=3'."
    )

    def _run(self, tool_input: str):
        args = shlex.split(tool_input.strip())
        has_ns = any(arg in ["-n", "--namespace"] or arg.startswith("--namespace=") for arg in args)
        
        extra_args = []
        if not has_ns:
            extra_args = ["-n", "apps"]
            
        cmd = ["kubectl", "scale"] + args + extra_args
        
        prompt = f"Scale resource with command: '{" ".join(cmd)}'?"
        if not confirm_action(prompt):
            return "Scaling operation cancelled by user."
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr

    def _arun(self, *args, **kwargs):
        raise NotImplementedError()

# Tool 6: Delete Resource
class DeleteResourceTool(BaseTool):
    name: str = "delete_resource"
    description: str = (
        "Execute 'kubectl delete' to delete deployments, pods, services, etc. "
        "Input format: '<resource_type> <name_or_selector> [flags]' or '-f <file_or_dir> [flags]'. "
        "Supports all resources, namespaces, and standard delete options (like --grace-period=0 --force). "
        "Example inputs: 'deployment web -n apps', 'pod auth-service-abc-123', '-f my-file.yaml'."
    )

    def _run(self, tool_input: str):
        args = shlex.split(tool_input.strip())
        has_ns = any(arg in ["-n", "--namespace"] or arg.startswith("--namespace=") for arg in args)
        has_all_ns = any(arg in ["-A", "--all-namespaces"] for arg in args)
        
        extra_args = []
        # Check if deleting resource that is cluster-wide
        cluster_wide = len(args) > 0 and args[0] in ["namespace", "ns", "clusterrole", "clusterrolebinding", "persistentvolume", "pv", "storageclass", "sc"]
        
        if not has_ns and not has_all_ns and not cluster_wide:
            extra_args = ["-n", "apps"]
            
        cmd = ["kubectl", "delete"] + args + extra_args
        
        prompt = f"🔴 DESTRUCTIVE: Delete resource with command: '{" ".join(cmd)}'?"
        if not confirm_action(prompt):
            return "Deletion operation cancelled by user."
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr

    def _arun(self, *args, **kwargs):
        raise NotImplementedError()

# Tool 7: Monitor Resources (metrics & events)
class MonitorResourcesTool(BaseTool):
    name: str = "monitor_resources"
    description: str = (
        "Execute 'kubectl top' or 'kubectl get events' for monitoring and diagnostics. "
        "Input format: 'top <pods|nodes> [flags]' or 'events [flags]'. "
        "Use 'top pods' or 'top nodes' for CPU/memory usage (requires metrics-server), "
        "and 'events' for cluster events. "
        "Example inputs: 'top pods -n apps', 'top nodes', 'events --sort-by=.lastTimestamp -n apps', 'events -A'."
    )

    def _run(self, tool_input: str):
        args = shlex.split(tool_input.strip())
        if not args:
            return "Error: specify 'top pods', 'top nodes', or 'events'."

        sub = args[0].lower()
        rest = args[1:]

        has_ns = any(arg in ["-n", "--namespace"] or arg.startswith("--namespace=") for arg in rest)
        has_all_ns = any(arg in ["-A", "--all-namespaces"] for arg in rest)
        extra_args = [] if (has_ns or has_all_ns or sub == "top" and rest and rest[0].lower() == "nodes") else ["-n", "apps"]

        if sub == "top":
            target = rest[0].lower() if rest else "pods"
            if target not in ["pods", "pod", "nodes", "node"]:
                return "Error: 'top' supports only 'pods' or 'nodes'."
            cmd = ["kubectl", "top", target] + rest[1:] + extra_args
        elif sub in ["events", "event", "ev"]:
            cmd = ["kubectl", "get", "events"] + rest + extra_args
        else:
            return "Error: unknown monitor subcommand. Use 'top pods', 'top nodes', or 'events'."

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr

    def _arun(self, *args, **kwargs):
        raise NotImplementedError()

# Tool 8: Expose Resource (services & load balancers)
class ExposeResourceTool(BaseTool):
    name: str = "expose_resource"
    description: str = (
        "Execute 'kubectl expose' to expose a deployment/pod/service via a Service (ClusterIP, NodePort, LoadBalancer). "
        "Input format: '<resource_type>/<name> --type=<ClusterIP|NodePort|LoadBalancer> --port=<port> [--target-port=<port>] [flags]'. "
        "Example inputs: 'deployment/web --type=LoadBalancer --port=80 --target-port=8080 -n apps', "
        "'deployment web --type=NodePort --port=80'."
    )

    def _run(self, tool_input: str):
        args = shlex.split(tool_input.strip())
        if not args:
            return "Error: specify a resource to expose, e.g. 'deployment/web --type=LoadBalancer --port=80'."

        has_ns = any(arg in ["-n", "--namespace"] or arg.startswith("--namespace=") for arg in args)
        extra_args = [] if has_ns else ["-n", "apps"]

        cmd = ["kubectl", "expose"] + args + extra_args

        is_dry_run = any("dry-run" in arg for arg in cmd)
        if not is_dry_run:
            prompt = f"Expose resource with command: '{" ".join(cmd)}'?"
            if not confirm_action(prompt):
                return "Expose operation cancelled by user."

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr

    def _arun(self, *args, **kwargs):
        raise NotImplementedError()

# Build Agent
tools = [
    GetResourcesTool(),
    DescribeResourceTool(),
    GetLogsTool(),
    CreateResourceTool(),
    ScaleResourceTool(),
    DeleteResourceTool(),
    MonitorResourcesTool(),
    ExposeResourceTool()
]

# Use a simpler approach with default ReAct agent
from langchain_classic.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools, 
    LLM, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# ---- CLUSTER SWITCHING ----
EKS_PROFILE = os.environ.get("AWS_EKS_PROFILE", os.environ.get("AWS_SECRET_PROFILE", "my-aws-profile"))
EKS_REGIONS = [r.strip() for r in os.environ.get("AWS_EKS_REGIONS", "us-east-1").split(",") if r.strip()]
KUBECONFIG_PATH = (os.environ.get("KUBECONFIG") or r"C:\Projects\.kube\dev\config").split(os.pathsep)[0]
os.environ["KUBECONFIG"] = KUBECONFIG_PATH

def kubectl_config(*args) -> str:
    result = subprocess.run(["kubectl", "--kubeconfig", KUBECONFIG_PATH, "config", *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""

def current_cluster() -> str:
    return kubectl_config("current-context") or "(none)"

def kubeconfig_contexts() -> dict:
    """Map cluster name/ARN -> context name for everything already in the kubeconfig."""
    view = json.loads(kubectl_config("view", "-o", "json") or "{}")
    return {c["context"]["cluster"]: c["name"] for c in view.get("contexts") or []}

def list_eks_clusters():
    import boto3
    session = boto3.Session(profile_name=EKS_PROFILE)
    account = session.client("sts").get_caller_identity()["Account"]
    clusters = []
    for region in EKS_REGIONS:
        for page in session.client("eks", region_name=region).get_paginator("list_clusters").paginate():
            clusters += [(name, region, f"arn:aws:eks:{region}:{account}:cluster/{name}") for name in page["clusters"]]
    return clusters

def add_cluster_to_kubeconfig(name: str, region: str):
    aws_cli = shutil.which("aws")
    if not aws_cli:
        raise RuntimeError("AWS CLI not found on PATH. Install AWS CLI v2.")
    result = subprocess.run(
        [aws_cli, "eks", "update-kubeconfig", "--name", name, "--region", region,
         "--profile", EKS_PROFILE, "--kubeconfig", KUBECONFIG_PATH],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

def sync_clusters(clusters=None):
    clusters = clusters or list_eks_clusters()
    existing = kubeconfig_contexts()
    missing = [c for c in clusters if c[2] not in existing]
    if not missing:
        print(f"All {len(clusters)} clusters are already in {KUBECONFIG_PATH}")
        return
    original = kubectl_config("current-context")
    os.makedirs(os.path.dirname(KUBECONFIG_PATH), exist_ok=True)
    if os.path.exists(KUBECONFIG_PATH):
        shutil.copy2(KUBECONFIG_PATH, KUBECONFIG_PATH + ".bak")
        print(f"Backup saved: {KUBECONFIG_PATH}.bak")
    print(f"Adding {len(missing)} missing cluster(s) to {KUBECONFIG_PATH}...")
    for name, region, _ in missing:
        try:
            add_cluster_to_kubeconfig(name, region)
            print(f"  added   {name} ({region})")
        except RuntimeError as e:
            print(f"  FAILED  {name} ({region}): {e}")
    if original:
        kubectl_config("use-context", original)
    print(f"Done. Current cluster unchanged: {current_cluster()}")

def switch_cluster(target: str = None):
    print(f"Current cluster: {current_cluster()}")
    print(f"Listing EKS clusters (profile: {EKS_PROFILE}, regions: {', '.join(EKS_REGIONS)})...")
    clusters = list_eks_clusters()
    if not clusters:
        print("No EKS clusters found. Check AWS_EKS_PROFILE / AWS_EKS_REGIONS.")
        return
    if target:
        matches = [c for c in clusters if c[0].lower() == target.lower()]
        if not matches:
            print(f"Cluster '{target}' not found. Type 'clusters' to see the list.")
            return
        name, region, arn = matches[0]
    else:
        contexts = kubeconfig_contexts()
        for i, (name, region, arn) in enumerate(clusters, 1):
            marker = "  <- current" if contexts.get(arn) == current_cluster() else ""
            print(f"  {i}. {name}  ({region}){marker}")
        choice = input("Pick a cluster number (Enter to cancel): ").strip()
        if not choice.isdigit() or not 1 <= int(choice) <= len(clusters):
            print("Cancelled. Cluster not changed.")
            return
        name, region, arn = clusters[int(choice) - 1]
    if "prod" in name.lower() and not confirm_action(f"'{name}' looks like a PRODUCTION cluster. Switch anyway?"):
        print("Cancelled. Cluster not changed.")
        return
    context = kubeconfig_contexts().get(arn)
    if not context:
        try:
            add_cluster_to_kubeconfig(name, region)
        except RuntimeError as e:
            print(f"Failed to add cluster: {e}")
            return
        context = kubeconfig_contexts().get(arn, arn)
    result = subprocess.run(["kubectl", "--kubeconfig", KUBECONFIG_PATH, "config", "use-context", context], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to switch cluster: {result.stderr.strip()}")
        return
    print(f"Switched to {name} ({region}).")
    print("Note: default namespace is still 'apps' - mention a namespace if it doesn't exist in this cluster.")

# Natural-language cluster commands handled directly (whole phrase must match,
# so e.g. "list pods in the cluster" still goes to the AI agent)
CLUSTER_LIST_RE = re.compile(r"^((list|show|get|display|see|view)\s+)?((me|all|the|my|available|eks|aws)\s+)*(eks\s+)?clusters?(\s+(list|names))?$")
CLUSTER_SWITCH_RE = re.compile(r"^(switch|change|connect|move|go|use)\s+(the\s+)?(cluster\s+)?(to\s+)?(cluster\s+)?(?P<name>[\w.\-]+)$")
CLUSTER_SYNC_RE = re.compile(r"^(sync|load|refresh|import|add)\s+(all\s+)?(missing\s+)?(eks\s+)?clusters?$")
CLUSTER_CURRENT_RE = re.compile(r"^(current|which|what)(\s+is)?(\s+(the|my))?(\s+current)?\s+cluster(\s+(am i|i am)\s+(on|using|connected to|in))?$")

HELP_TEXT = """
 EXAMPLE QUERIES
==================
 GET (list resources) — default namespace: apps
   - get deployments
   - list all pods
   - list all pods in all namespaces
   - get services -o wide
   - get pods with label app=nginx
   - get namespaces / nodes / configmaps / secrets / pvc
   - get deployment orders-api in yaml format

 DESCRIBE (details, events, troubleshooting)
   - describe pod auth-service-xxx
   - describe deployment orders-api
   - describe node <node-name>
   - describe service orders-api

 LOGS
   - get logs for pod auth-service-xxx
   - get last 100 lines of logs for pod <name>
   - get logs for pod <name> container <container-name>
   - get logs for pod <name> since 1h
   - get previous logs for pod <name>

  CREATE (asks confirmation first — dry-run is free)
   - create a deployment named web with image nginx
   - create a deployment named web with image nginx and 3 replicas
   - create namespace test-ns
   - dry run create a clusterip service my-svc port 80 to 8080 and print yaml
   - create configmap my-cm from literal key1=val1

 SCALE (asks confirmation first)
   - scale deployment orders-api to 3 replicas
   - scale statefulset <name> to 2 replicas

 DELETE (always asks confirmation)
   - delete deployment web
   - delete pod <name>
   - delete namespace test-ns

 MONITOR (metrics & events)
   - show CPU and memory usage of pods
   - top nodes
   - show recent events
   - why did my pod restart? (agent checks events + logs)

 EXPOSE (services & load balancers — asks confirmation)
   - expose deployment web as a LoadBalancer on port 80
   - expose deployment orders-api via NodePort on port 8080

 CLUSTER (handled directly, no AI)
   - list all clusters    -> list all EKS clusters and pick one to switch to
                             (also: clusters / show clusters / switch cluster)
   - switch to <name>     -> switch directly, e.g. "switch to demo-qa"
   - current cluster      -> show which cluster you are connected to
   - sync clusters        -> add any missing EKS clusters to your kubeconfig

 Tips:
   - Mention a namespace (e.g. "in default namespace") or it defaults to apps
   - Use "-A" / "all namespaces" for cluster-wide listing
   - Type 'exit' or 'quit' to leave
"""

# Run
if __name__ == "__main__":
    print("Kubernetes AI Admin & Diagnostics Agent (Get, Describe, Logs, Create, Scale, Delete, Monitor, Expose)")
    print("Using gemini-3.5-flash-lite model")
    print(f"Current cluster: {current_cluster()}")
    print("Type 'help' to see example queries, 'clusters' to switch cluster.")
    
    while True:
        try:
            user_input = input("\n💡 What should I do? (or 'exit'): ").strip()
            command = user_input.lower()
            if command in ["exit", "quit"]:
                break
            if command in ["help", "?", "commands"]:
                print(HELP_TEXT)
                continue
            command = command.rstrip("?!. ")
            if CLUSTER_LIST_RE.match(command):
                switch_cluster()
                continue
            switch_match = CLUSTER_SWITCH_RE.match(command)
            if switch_match:
                target = switch_match.group("name")
                switch_cluster(None if target in ["cluster", "clusters", "eks"] else target)
                continue
            if CLUSTER_SYNC_RE.match(command):
                sync_clusters()
                continue
            if CLUSTER_CURRENT_RE.match(command):
                print(f"Current cluster: {current_cluster()}")
                continue
            if not user_input:
                continue
            result = agent.run(user_input)
            print("Agent:", result)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again or type 'exit' to quit")
