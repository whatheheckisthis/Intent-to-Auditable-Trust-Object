------------------------------- MODULE OSINTDispatcherRESTLogging -------------------------------
EXTENDS Naturals, Sequences, TLC

(***************************************************************************
High-assurance behavioral model for RESTful logging eventual consistency.
If a command executes, then eventually:
  - a log entry for that command is signed by Ed25519 and transmitted, OR
  - system transitions to Error.
***************************************************************************)

CONSTANTS Commands

VARIABLES
  state,           \* "Ready" | "Executing" | "Logging" | "Error"
  executed,        \* set of executed commands
  signed,          \* set of commands whose logs are Ed25519-signed
  transmitted      \* set of commands whose signed logs are transmitted

States == {"Ready", "Executing", "Logging", "Error"}

Init ==
  /\ state = "Ready"
  /\ executed = {}
  /\ signed = {}
  /\ transmitted = {}

Execute(c) ==
  /\ c \in Commands
  /\ state \in {"Ready", "Executing", "Logging"}
  /\ state' = "Executing"
  /\ executed' = executed \cup {c}
  /\ UNCHANGED <<signed, transmitted>>

SignLogEd25519(c) ==
  /\ c \in executed
  /\ c \notin signed
  /\ state \in {"Executing", "Logging"}
  /\ state' = "Logging"
  /\ signed' = signed \cup {c}
  /\ UNCHANGED <<executed, transmitted>>

TransmitSignedLog(c) ==
  /\ c \in signed
  /\ c \notin transmitted
  /\ state \in {"Executing", "Logging"}
  /\ state' = "Ready"
  /\ transmitted' = transmitted \cup {c}
  /\ UNCHANGED <<executed, signed>>

FailToError ==
  /\ state \in {"Ready", "Executing", "Logging"}
  /\ state' = "Error"
  /\ UNCHANGED <<executed, signed, transmitted>>

StayError ==
  /\ state = "Error"
  /\ UNCHANGED <<state, executed, signed, transmitted>>

Next ==
  \/ \E c \in Commands: Execute(c)
  \/ \E c \in Commands: SignLogEd25519(c)
  \/ \E c \in Commands: TransmitSignedLog(c)
  \/ FailToError
  \/ StayError

Spec ==
  Init /\ [][Next]_<<state, executed, signed, transmitted>>

(***************************************************************************
Safety: transmitted logs must have Ed25519 signatures.
***************************************************************************)
SignedBeforeTransmit == [] (transmitted \subseteq signed)

(***************************************************************************
Liveness: no silent failure.
For any executed command, eventually it is transmitted or we are in Error.
***************************************************************************)
NoSilentFailure ==
  \A c \in Commands: [] (c \in executed => <> (c \in transmitted \/ state = "Error"))

(***************************************************************************
Fairness assumptions so enabled logging and transmission are not starved.
***************************************************************************)
FairSpec ==
  Spec
  /\ \A c \in Commands: WF_<<state, executed, signed, transmitted>>(SignLogEd25519(c))
  /\ \A c \in Commands: WF_<<state, executed, signed, transmitted>>(TransmitSignedLog(c))

THEOREM FairSpec => NoSilentFailure
===============================================================================================
