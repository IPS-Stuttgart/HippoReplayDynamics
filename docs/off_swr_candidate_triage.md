# Off-SWR Candidate Triage

`scripts/off_swr_trajectory_discovery.py` aggregates spike-matched off-SWR
scoring runs and now writes a ranked candidate triage layer in addition to the
legacy discovery summaries.

Primary triage outputs:

- `off_swr_candidate_table.csv`: one ranked row per trajectory-family off-SWR
  candidate, with margin, trajectory confidence, entropy diagnostics, nearest
  scored-SWR distance, speed fields when available, decoded geometry fields when
  available, a `candidate_specificity_label`, and the strongest `candidate_tier`
  reached by the window.
- `off_swr_candidate_cluster_table.csv`: session/time clusters of candidate
  windows with median phenotype summaries and counts of interesting versus
  movement/spiking-like candidates.
- `off_swr_candidate_session_summary.csv` and
  `off_swr_candidate_rat_summary.csv`: per-session and per-rat candidate
  phenotype summaries.
- `off_swr_candidate_vs_swr_summary.csv`: compares off-SWR candidates against
  scored real SWR reference windows from the same artifact, including median
  contrasts for family margin, trajectory confidence, spikes, active cells,
  entropy, decoded path length, decoded speed, duration, nearest-SWR distance,
  and animal speed. The `off_swr_vs_swr_interpretation` field reports whether
  candidates look like `A_swr_like_strength`,
  `B_weaker_but_directionally_similar_tail`, or
  `C_mostly_movement_behavioral_decoding_windows`; the last case sets
  `claim_should_narrow`.
- `off_swr_candidate_vs_swr_window_table.csv`: one row per off-SWR candidate
  and scored SWR replay reference window with matched phenotype columns for a
  direct side-by-side contrast.
- `off_swr_candidate_vs_swr_model_distribution.csv`: best trajectory-model
  distributions for SWR reference windows versus off-SWR candidates.
- `off_swr_run_state_stratified_summary.csv`: separates off-SWR immobile
  windows, off-SWR running windows, unknown-speed off-SWR windows, and SWR
  replay reference windows. Each row reports candidate fraction, margin,
  confidence, spikes, active cells, entropy, decoded path length/speed, and
  best-model distribution.
- `off_swr_run_state_specificity_summary.csv`: one-row interpretation of the
  key specificity question. `immobile_off_swr_candidates_present` supports an
  interesting non-SWR replay-like tail; `candidate_signal_concentrated_in_running_windows`
  means the off-SWR claim should narrow because the screen may be dominated by
  movement/place-code decoding.
- `off_swr_nearest_swr_exclusion_summary.csv`: recomputes the off-SWR
  trajectory-family candidate fraction after excluding windows within 100 ms,
  250 ms, 500 ms, and 1 s of known SWR windows. Persistence at 500 ms or 1 s is
  stronger evidence that candidates are not just SWR-edge contamination.
- `off_swr_nearest_swr_specificity_summary.csv`: one-row interpretation of
  whether candidates survive the 500 ms and 1 s exclusion checks. If candidates
  vanish by 500 ms, the off-SWR claim should narrow.
- `off_swr_candidate_tier_threshold_summary.csv`: cumulative selectivity counts
  for weak (`margin >= 5.5`), moderate (`margin >= 20`), strong (`margin >= 50`),
  and extreme (`margin >= 100`) candidate thresholds. It also reports counts
  after immobility filtering and after 500 ms / 1 s nearest-SWR exclusion.
- `off_swr_candidate_tier_session_summary.csv` and
  `off_swr_candidate_tier_rat_summary.csv`: the same cumulative tier counts by
  session and by rat.
- `off_swr_candidate_tier_nearest_swr_exclusion_summary.csv`: tier-by-radius
  survival counts for the 100 ms, 250 ms, 500 ms, and 1 s nearest-SWR exclusion
  filters.
- `off_swr_high_specificity_candidate_table.csv`: strong-tier candidates
  (`margin >= 50`) that survive the 1 s nearest-SWR exclusion, annotated with
  whether they also pass the immobility filter when speed/run-state fields are
  available and the `candidate_specificity_label` filter that excludes
  low-information or ordinary movement/spiking-like audit rows.
- `off_swr_promotion_readiness_summary.csv`: one-row promotion decision. It
  remains exploratory when speed is unavailable, candidates vanish near SWRs, or
  strong candidates are not established in immobile non-SWR windows. It reports
  `ready_for_off_swr_replay_candidate_claim` only when the combined tier,
  nearest-SWR, immobility, and specificity-label filters pass.
- `off_swr_speed_coverage_summary.csv`: one-row metadata coverage audit showing
  whether newly scored off-SWR windows, candidates, and strong candidates have
  position-derived animal-speed fields. Interpret this before promotion status:
  `speed_unavailable` means the artifact still cannot test the immobility
  requirement.
- `off_swr_candidate_specificity_gate_summary.csv`: infrastructure and
  specificity-readiness gates. Missing behavior/LFP phenotype fields are
  reported explicitly and do not silently block table generation.

Promotion funnel:

`scripts/build_off_swr_promotion_funnel.py` joins an off-SWR discovery artifact
with a promoted-candidate exact-core validation artifact. It writes:

- `off_swr_promotion_funnel_summary.csv`: denominator-backed funnel from all
  screened off-SWR windows through weak/moderate/strong/extreme tiers,
  promotion-ready candidates, exact-core validation, and exact-core
  trajectory-confident candidates.
- `off_swr_promotion_funnel_group_summary.csv`: the same funnel counts by rat
  and session.
- `off_swr_promotion_funnel_rejection_summary.csv`: phenotype summaries for
  candidates below the strong tier, high-specificity rows rejected by the
  specificity/immobility filters, and exact-validated promotion-ready rows.
- `off_swr_promotion_funnel_gate_summary.csv`: input and consistency gates,
  including whether exact validation rows match the promotion-ready denominator.

Empirical promotion calibration:

`scripts/calibrate_off_swr_promotion_fdr.py` evaluates the strict promotion rule
against observed running/ordinary movement-spiking controls and shuffled
label/immobility nulls. It writes:

- `off_swr_promotion_null_calibration.csv`: observed control counts and
  permutation-null promotion-count distributions.
- `off_swr_promotion_empirical_fdr_summary.csv`: one-row calibration summary
  with observed promotions, exact validation counts, direct-control false
  promotions, and conservative permutation FDR estimates.
- `off_swr_promotion_threshold_sensitivity.csv`: weak/moderate/strong/extreme
  threshold sensitivity, including whether promoted counts exceed joint-shuffle
  null bounds.
- `off_swr_promotion_null_gate_summary.csv`: required calibration gates.

The triage layer is intended to separate high-priority off-SWR trajectory
candidates from ordinary movement/spiking windows that naturally favor
trajectory models. It does not by itself establish replay-without-ripples
specificity; that requires independent behavior/LFP validation or a stricter
selectivity gate.
