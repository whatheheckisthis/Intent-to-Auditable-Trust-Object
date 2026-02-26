---- MODULE SVE2PredicateDiscipline ----
EXTENDS Naturals

CONSTANTS PredicateSource

PublicSources == {"lane_index", "loop_counter", "bounds_mask"}

PredicatesPublicOnly == PredicateSource \subseteq PublicSources

====
