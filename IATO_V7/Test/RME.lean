import IATO.V7.RMEModel

open IATO.V7

def seedState : RmeState :=
  step
    (step default (RmiCall.RealmCreate 7))
    (RmiCall.GranuleDelegate 42)

def validDataState : RmeState :=
  step seedState (RmiCall.DataCreate 7 42)

def invalidDataState : RmeState :=
  step seedState (RmiCall.DataCreate 99 42)

def destroyedState : RmeState :=
  step validDataState (RmiCall.RealmDestroy 7)

def test_valid_data_create : Bool :=
  decide (validDataState.granules 42 = GranuleState.Data 7)

def test_invalid_data_create : Bool :=
  decide (invalidDataState.granules 42 = GranuleState.Delegated)

def test_destroy_scrubs_data : Bool :=
  decide (destroyedState.granules 42 = GranuleState.Delegated)

theorem destroyedState_wf : RmeState.wf destroyedState := by
  unfold destroyedState validDataState seedState
  repeat
    first | apply wf_step | apply wf_default

def test_wf_preserved : Bool := decide destroyedState_wf

def traceState : RmeState :=
  execTrace default
    [ RmiCall.RealmCreate 7
    , RmiCall.GranuleDelegate 42
    , RmiCall.DataCreate 7 42
    , RmiCall.RealmDestroy 7
    ]

def test_trace_exec_matches_manual : Bool :=
  decide (traceState = destroyedState)

theorem traceState_wf : RmeState.wf traceState := by
  unfold traceState
  exact wf_execTrace default wf_default _

def test_trace_wf : Bool := decide traceState_wf

def test_destroy_erases_realm : Bool :=
  decide (7 ∉ traceState.realms)

def main : IO Unit := do
  IO.println s!"test_valid_data_create = {test_valid_data_create}"
  IO.println s!"test_invalid_data_create = {test_invalid_data_create}"
  IO.println s!"test_destroy_scrubs_data = {test_destroy_scrubs_data}"
  IO.println s!"test_wf_preserved = {test_wf_preserved}"
  IO.println s!"test_trace_exec_matches_manual = {test_trace_exec_matches_manual}"
  IO.println s!"test_trace_wf = {test_trace_wf}"
  IO.println s!"test_destroy_erases_realm = {test_destroy_erases_realm}"

  if !(test_valid_data_create &&
      test_invalid_data_create &&
      test_destroy_scrubs_data &&
      test_wf_preserved &&
      test_trace_exec_matches_manual &&
      test_trace_wf &&
      test_destroy_erases_realm) then
    throw <| IO.userError "RME model tests failed"
