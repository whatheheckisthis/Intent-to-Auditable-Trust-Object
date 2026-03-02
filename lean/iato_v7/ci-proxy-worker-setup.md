# CI Proxy Worker setup (Minikube + WSL2 + Windows host)

When you run this, you are essentially creating a "Sidecar" or "Runner" pattern. Because you are using Minikube, keep in mind that the `hostPath` doesn't see your C: drive by default—Minikube runs inside a VM.

## Missing link command (run from Windows terminal)

```powershell
minikube mount C:\Users\Name\Project:/workspace
```

## Minikube mount command (with persistence-friendly UID/GID)

```bash
minikube mount --uid=1000 --gid=1000 C:/Users/Name/Project:/workspace
```

## Recommended `/etc/wsl.conf`

```ini
[automount]
enabled = true
options = "metadata,umask=22,fmask=11"
mountFsTab = false
```

After updating `/etc/wsl.conf`, restart WSL:

```powershell
wsl --shutdown
```

Then re-open your WSL shell and re-run the Minikube mount command.

## Apply the CI Proxy worker resources

```bash
kubectl apply -f lean/iato_v7/k8s-ci-proxy-worker.yaml
kubectl logs -f job/iato-v7-ci-proxy-worker
```

The worker loads `lean/iato_v7/k8s-build-job.yaml` as read-only scaffold input, parses the scaffold `args`, and executes them in `/workspace` inside the Lean container.
