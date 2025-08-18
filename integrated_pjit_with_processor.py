# integrated_pjit_with_processor.py
# Single-file integrated PoC:
#  - sharded pjit lattice PGD (per-device fingerprints)
#  - host-side FingerprintProcessor worker (multiprocessing)
#  - integration: provisional hash, submit, poll, reconcile, shutdown
#
# Requirements:
#  - jax, jaxlib built for CUDA on the host (for pjit to see GPUs)
#  - numpy
#  - run on a multi-GPU CUDA VM for meaningful behavior
#
# Note: This is a PoC. Replace HMAC_KEY and ledger handling with HSM/secure services for production.

import time
import json
import hashlib
import hmac
import os
import multiprocessing as mp
from collections import deque

import numpy as np

# JAX imports
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec
from functools import partial

# ----------------------------
# Config: PGD & sharding
# ----------------------------
grid_size = 128          # total grid dimension per axis (must be divisible by n_devices)
dx = 0.1
mass = 1.0
lambda_phi4 = 0.1

eta = 0.005
T_iters = 80             # iterations
fingerprint_per_device = 64  # floats per device fingerprint
MERKLE_ANCHOR_PERIOD = 32

# HMAC key - replace with secure key storage in prod
HMAC_KEY = b"replace-with-secure-key"

# ----------------------------
# FingerprintProcessor worker (simplified)
# ----------------------------
def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def hmac_sign_hex(b: bytes) -> str:
    return hmac.new(HMAC_KEY, b, hashlib.sha256).hexdigest()

def merkle_root_from_hex_hashes(hex_hashes):
    layer = [bytes.fromhex(h) for h in hex_hashes]
    if not layer:
        return sha256_hex(b'')
    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i+1] if (i+1) < len(layer) else left
            next_layer.append(hashlib.sha256(left + right).digest())
        layer = next_layer
    return layer[0].hex()

# Worker process class
class FingerprintWorker(mp.Process):
    def __init__(self, in_q: mp.Queue, out_q: mp.Queue, stop_event: mp.Event, anchor_period=MERKLE_ANCHOR_PERIOD):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cache = {}                 # hash -> explanation/meta
        self.cache_order = deque()
        self.MAX_CACHE = 5000
        self.pending_step_hashes = []
        self.anchor_period = anchor_period

    def run(self):
        print("[worker] started")
        while not self.stop_event.is_set():
            try:
                task = self.in_q.get(timeout=0.5)
            except Exception:
                continue
            # task expected keys: 'fp_all' (np.ndarray shape (n_devices,k)), 'energy', 'provisional_hash', 'step_meta'
            fp_all = task['fp_all']    # np.ndarray float32
            energy = float(task.get('energy', 0.0))
            prov_hash = task.get('provisional_hash', '')
            step_meta = task.get('step_meta', {})

            # compute per-device hex hashes
            per_dev_hashes = []
            for d in range(fp_all.shape[0]):
                blob = fp_all[d].tobytes() + json.dumps({"energy": energy, "device": d, **step_meta}, sort_keys=True).encode()
                ph = sha256_hex(blob)
                per_dev_hashes.append(ph)

            # canonical step hash: hash(concat(per_dev_hashes) + prov_hash)
            combined = ("".join(per_dev_hashes) + prov_hash).encode()
            step_hash = sha256_hex(combined)

            # produce a simple explanation: top features by absolute value on each device (bounded to 180)
            explanations = []
            TOP_K = min(180, fp_all.shape[1])
            for d in range(fp_all.shape[0]):
                arr = np.abs(fp_all[d])
                top_idx = np.argsort(-arr)[:TOP_K]
                feat_list = [{"idx": int(int(i)), "value": float(arr[int(i)])} for i in top_idx]
                explanations.append({"device": int(d), "top": feat_list})

            # caching: store step_hash -> explanations
            self.cache[step_hash] = {"explanations": explanations, "time": time.time(), "prov": prov_hash}
            self.cache_order.append(step_hash)
            if len(self.cache_order) > self.MAX_CACHE:
                old = self.cache_order.popleft()
                self.cache.pop(old, None)

            # collect pending hash for merkle
            self.pending_step_hashes.append(step_hash)
            merkle_record = None
            if len(self.pending_step_hashes) >= self.anchor_period:
                root = merkle_root_from_hex_hashes(self.pending_step_hashes)
                entry = {"timestamp": time.time(), "merkle_root": root, "count": len(self.pending_step_hashes)}
                entry_bytes = json.dumps(entry, sort_keys=True).encode()
                signature = hmac_sign_hex(entry_bytes)
                merkle_record = {"entry": entry, "signature": signature}
                # clear pending
                self.pending_step_hashes = []

            # return package
            out_pkg = {"step_hash": step_hash, "provisional_hash": prov_hash, "explanations": explanations, "merkle": merkle_record}
            self.out_q.put(out_pkg)

        print("[worker] exiting")

# Manager wrapper to be used in main process
class FingerprintProcessor:
    def __init__(self, n_workers=1):
        self.in_q = mp.Queue(maxsize=1024)
        self.out_q = mp.Queue(maxsize=1024)
        self.stop_event = mp.Event()
        self.workers = [FingerprintWorker(self.in_q, self.out_q, self.stop_event) for _ in range(n_workers)]
        for w in self.workers:
            w.start()

    def submit(self, fp_all_np: np.ndarray, energy: float, provisional_hash: str, step_meta: dict=None):
        payload = {"fp_all": fp_all_np.astype(np.float32), "energy": float(energy), "provisional_hash": provisional_hash, "step_meta": step_meta or {}}
        # block if queue full to avoid uncontrolled memory growth; caller can handle backpressure
        self.in_q.put(payload)

    def poll_result(self, timeout=0.0):
        try:
            return self.out_q.get(timeout=timeout)
        except Exception:
            return None

    def shutdown(self):
        self.stop_event.set()
        time.sleep(0.1)
        # empty queues
        try:
            while not self.in_q.empty():
                self.in_q.get_nowait()
        except Exception:
            pass
        for w in self.workers:
            w.join(timeout=1.0)

# ----------------------------
# JAX: pjit sharded lattice PGD with per-device fingerprint
# ----------------------------
# Create device mesh
devices = jax.devices()
n_devices = len(devices)
if n_devices == 0:
    raise RuntimeError("No JAX devices available. Run on CUDA VM with GPUs.")
print(f"[main] detected {n_devices} devices")

if grid_size % n_devices != 0:
    raise ValueError("grid_size must be divisible by n_devices for this example")

mesh = mesh_utils.create_device_mesh((n_devices,))  # 1D mesh
local_shard = grid_size // n_devices

# Lattice ops
def laplacian_3d_periodic_global(phi, dx):
    lap = (jnp.roll(phi, 1, axis=0) + jnp.roll(phi, -1, axis=0) +
           jnp.roll(phi, 1, axis=1) + jnp.roll(phi, -1, axis=1) +
           jnp.roll(phi, 1, axis=2) + jnp.roll(phi, -1, axis=2) -
           6.0 * phi)
    return lap / (dx**2)

def H_lattice_global(phi):
    lap = laplacian_3d_periodic_global(phi, dx)
    kinetic = 0.5 * jnp.sum(phi * lap)
    potential = 0.5 * mass**2 * jnp.sum(phi**2) + lambda_phi4 * jnp.sum(phi**4)
    return kinetic + potential

# Partition specs
phi_in_spec = PartitionSpec('x', None, None)
phi_out_spec = PartitionSpec('x', None, None)

@pjit(in_axis_resources=(phi_in_spec,), out_axis_resources=phi_out_spec)
def pgd_step_pjit(phi):
    g = jax.grad(H_lattice_global)(phi)
    return phi - eta * g

@pjit(in_axis_resources=(phi_in_spec,), out_axis_resources=None)
def energy_pjit(phi):
    return H_lattice_global(phi)

# per-device fingerprint: returns per-device arrays; out_axis_resources -> PartitionSpec('x', None)
def make_local_fingerprint(phi_local, k):
    mu = jnp.mean(phi_local)
    sigma = jnp.std(phi_local)
    remain = max(1, k - 2)
    per_axis = int(jnp.ceil(remain ** (1/3)))
    stride = max(1, phi_local.shape[0] // max(1, per_axis))
    down = phi_local[::stride, ::stride, ::stride]
    down_flat = jnp.ravel(down)
    desired = remain
    if down_flat.shape[0] >= desired:
        down_use = down_flat[:desired]
    else:
        pad = jnp.zeros((desired - down_flat.shape[0],), dtype=down_flat.dtype)
        down_use = jnp.concatenate([down_flat, pad], axis=0)
    fp = jnp.concatenate([jnp.array([mu, sigma], dtype=jnp.float32), down_use.astype(jnp.float32)], axis=0)
    fp = fp[:k]
    return fp

@pjit(in_axis_resources=(phi_in_spec,), out_axis_resources=PartitionSpec('x', None))
def compute_per_device_fp_pjit(phi):
    # Each device returns its local fingerprint vector of length fingerprint_per_device
    return make_local_fingerprint(phi, fingerprint_per_device)

# ----------------------------
# Integration: run loop with processor
# ----------------------------
def run_sharded_loop_with_processor():
    # initialize phi
    phi0 = jnp.zeros((grid_size, grid_size, grid_size), dtype=jnp.float32)

    # start processor
    processor = FingerprintProcessor(n_workers=1)
    time.sleep(0.1)  # allow worker to start

    audit = []
    prev_hash = ""

    with mesh:
        # warm-up compile
        print("[main] compiling pjit kernels (warmup)...")
        _ = energy_pjit(phi0)
        _ = pgd_step_pjit(phi0)
        _ = compute_per_device_fp_pjit(phi0)

        phi = phi0
        t0 = time.time()

        for t in range(T_iters):
            # PGD update (distributed)
            phi = pgd_step_pjit(phi)

            # compute energy (scalar)
            energy_val = float(energy_pjit(phi))

            # compute per-device fingerprint (small)
            fp_all = compute_per_device_fp_pjit(phi)   # shape logically (n_devices, k)
            # fetch small array to host
            fp_all_host = jax.device_get(fp_all)       # cheap: small data

            # create provisional hash (local quick hash)
            provisional_meta = {"t": int(t), "energy": float(energy_val)}
            provisional_bytes = json.dumps(provisional_meta, sort_keys=True).encode()
            provisional_hash = hashlib.sha256(prev_hash.encode() + provisional_bytes).hexdigest()

            # append provisional entry
            audit.append({"t": int(t), "energy": float(energy_val), "provisional_hash": provisional_hash})
            prev_hash = provisional_hash

            # submit to processor with provisional_hash so worker can include it
            processor.submit(np.array(fp_all_host), float(energy_val), provisional_hash, {"t": int(t), "module": "lattice"})

            # poll worker (non-blocking)
            res = processor.poll_result(timeout=0.0)
            if res is not None:
                # reconcile: append canonical hash and explanations
                audit.append({"t": int(t), "canonical_hash": res["step_hash"], "provisional_hash": res["provisional_hash"], "explanations": res["explanations"]})
                prev_hash = res["step_hash"]
                if res.get("merkle") is not None:
                    print("[main] merkle anchor created:", res["merkle"]["entry"])

            if (t+1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"[main] iter {t+1}/{T_iters} energy={energy_val:.6f} elapsed={elapsed:.2f}s")

        # drain outstanding results
        time.sleep(0.1)
        while True:
            out = processor.poll_result(timeout=0.1)
            if out is None:
                break
            audit.append({"t": None, "canonical_hash": out["step_hash"], "explanations": out["explanations"]})

    # shutdown processor
    processor.shutdown()
    total_time = time.time() - t0
    return {"audit": audit, "time": total_time}

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    print("[main] starting integrated run")
    result = run_sharded_loop_with_processor()
    print("[main] done. iterations:", len(result["audit"]))
    print("Last audit entry:", result["audit"][-1])
    print("Elapsed time (s):", result["time"])
