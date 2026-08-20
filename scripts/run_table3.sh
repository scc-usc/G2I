#!/usr/bin/env bash
#
# Reproduce the synthetic rows of Table 3 (intervention evaluation).
#
# Runs every (n_nodes, total_edges) configuration for both synthetic families
# with all three explanation methods (CF-Greedy, CF, CF^2). Results land in
#   results/synthetic_neighbor_feature/*.json
#   results/synthetic_neighbor_only/*.json
# Aggregate them into the table with:
#   python scripts/make_table3.py
#
# Run from the repository root:
#   bash scripts/run_table3.sh
#
# Environment overrides:
#   N=10 BUDGET=5.0 EPSILON=0.1 WORKERS=30 DEVICE=cpu bash scripts/run_table3.sh
#
# Wall-clock warning: the CF and CF^2 baselines dominate the cost. The
# 500-node / 3000-edge configurations take several hours each on a 64-core CPU.

set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-10}"                 # repetitions per configuration (different seeds)
BUDGET="${BUDGET:-5.0}"      # budget B; budget_limit = BUDGET * ln(1/EPSILON)
EPSILON="${EPSILON:-0.1}"    # -> budget_limit = 11.5 with the defaults
WORKERS="${WORKERS:-8}"      # parallel workers for CF / CF^2
DEVICE="${DEVICE:-cpu}"

# (n_nodes, total_edges) pairs, matching the Table 3 row labels N<n>-E<e>
CONFIGS=(
    "100 150"
    "100 400"
    "250 300"
    "250 1500"
    "500 800"
    "500 3000"
)

run_one () {
    local synth_type="$1" m_attrs="$2" cff_mode="$3" n_nodes="$4" total_edges="$5"
    echo ""
    echo "=============================================================="
    echo " ${synth_type}: N${n_nodes}-E${total_edges}-D${m_attrs}"
    echo "=============================================================="
    python run.py \
        --synthetic_type "$synth_type" \
        --n_nodes "$n_nodes" \
        --total_edges "$total_edges" \
        --m_attrs "$m_attrs" \
        --N "$N" \
        --budget "$BUDGET" \
        --epsilon "$EPSILON" \
        --explanation_method cf-greedy cf cff \
        --cf_greedy_edge_mode edge_features \
        --cf_greedy_max_interventions 5 \
        --cf_beta 1 \
        --cff_mode "$cff_mode" \
        --cff_lam 100 \
        --cff_feat_TH 0.7 \
        --cff_edge_TH 0.7 \
        --parallel --num_workers "$WORKERS" \
        --device "$DEVICE"
}

# ---- Neighbor-Feature family (D10) -----------------------------------------
for cfg in "${CONFIGS[@]}"; do
    read -r n_nodes total_edges <<< "$cfg"
    run_one neighbor_feature 10 "edge+feature" "$n_nodes" "$total_edges"
done

# ---- Neighbor-Only family (D6) ---------------------------------------------
for cfg in "${CONFIGS[@]}"; do
    read -r n_nodes total_edges <<< "$cfg"
    run_one neighbor_only 6 "feature" "$n_nodes" "$total_edges"
done

echo ""
echo "All configurations finished. Build the table with:"
echo "  python scripts/make_table3.py"
