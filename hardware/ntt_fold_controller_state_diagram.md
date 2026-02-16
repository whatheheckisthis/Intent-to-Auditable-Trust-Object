# 8-Stage NTT Fold FSM State Diagram

```text
           +------------------+
           |      S_IDLE      |
           | busy=0, done=0   |
           +---------+--------+
                     | start=1
                     v
           +------------------+
           |   S_STAGE_INIT   |
           | bf_idx <- 0      |
           +---------+--------+
                     |
                     v
           +------------------+
           |   S_STAGE_RUN    |
           | issue=1          |
           | bf_idx += LANES  |
           +----+--------+----+
                |        |
                |        | bf_idx reached HALF_N-LANES
                |        v
                |   +------------------------------+
                |   | stage == 7 ?                 |
                |   +-----------+------------------+
                |               |
                | yes           | no
                v               v
      +------------------+   +------------------+
      |      S_DONE      |   | stage <- stage+1 |
      | done=1, busy=0   |   | then S_STAGE_INIT|
      +---------+--------+   +------------------+
                |
                v
           +------------------+
           |      S_IDLE      |
           +------------------+
```

- `stage` iterates 0..7 (`LOGN=8`) for a 256-point radix-2 NTT.
- Deterministic issue pattern keeps butterfly scheduling fixed each run.
- Two butterfly operations are issued per cycle (`LANES=2`) to match the dual-port twiddle ROM.
