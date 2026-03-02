module osint_dispatcher_command_mapping

/**
 * Structural model for OSINT Dispatcher command token -> memory address mappings.
 * Goals:
 * 1) No two command tokens alias to the same memory address.
 * 2) All signatures must originate from the TEE enclave.
 */

sig CommandToken {}
sig MemoryAddress {}

sig Principal {}
one sig Dispatcher, TEEEnclave extends Principal {}

sig Signature {
  signedBy: one Principal,
  forToken: one CommandToken
}

one sig CommandMap {
  mapsTo: CommandToken -> one MemoryAddress
}

fact TotalCommandMapping {
  all t: CommandToken | one CommandMap.mapsTo[t]
}

fact SignaturesOnlyFromTEE {
  all s: Signature | s.signedBy = TEEEnclave
}

assert InjectiveMapping {
  all disj t1, t2: CommandToken |
    CommandMap.mapsTo[t1] != CommandMap.mapsTo[t2]
}

assert SignatureOriginBoundedToTEE {
  all s: Signature | s.signedBy = TEEEnclave
}

check InjectiveMapping for 8 but 8 CommandToken, 8 MemoryAddress
check SignatureOriginBoundedToTEE for 8

pred sampleScenario {
  #CommandToken = 3
  #MemoryAddress = 3
  some Signature
}

run sampleScenario for 6
