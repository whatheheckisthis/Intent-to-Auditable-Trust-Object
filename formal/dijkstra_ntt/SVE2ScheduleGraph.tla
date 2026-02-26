---- MODULE SVE2ScheduleGraph ----
EXTENDS Naturals, Sequences

CONSTANTS Instrs, Deps, Hazard
VARIABLES ready, done, schedule

Init ==
  /\ done = {}
  /\ ready = {i \in Instrs : \A d \in Instrs : <<d,i>> \in Deps => d \in done}
  /\ schedule = << >>

CanIssue(i) == i \in Instrs \ done /\ \A d \in Instrs : <<d,i>> \in Deps => d \in done
Cost(i) == Hazard[i]
Pick == CHOOSE i \in {j \in Instrs : CanIssue(j)} : \A k \in {j \in Instrs : CanIssue(j)} : Cost(i) <= Cost(k)

Next ==
  /\ \E i \in Instrs : i = Pick
  /\ done' = done \cup {Pick}
  /\ schedule' = Append(schedule, Pick)
  /\ ready' = ready

Spec == Init /\ [][Next]_<<ready,done,schedule>>

====
