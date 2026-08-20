#!/usr/bin/env bash
#
# Emit the command list for the Figure 3 runtime-scalability sweep.
#
# Each line is "<tag>|<command>". Run them however your cluster prefers, e.g.
#
#   bash scripts/gen_scalability_commands.sh
#   cut -d'|' -f2- all_commands_runtime.txt | while read -r c; do eval "$c"; done
#
# or feed the file to GNU parallel / a SLURM array job.
#
# Results land in results/test_results/. Build Figure 3 with:
#   python scripts/plot_scalability.py
#
# Note on the largest sizes: CF and CF^2 do not fit in memory beyond roughly
# 6,400 nodes on a 248 GB machine (Section 4.3.3), so only CF-Greedy is run for
# 25,600 and 102,400 nodes. Those are the missing points in Figure 3.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs results/test_results

COMMANDS_FILE="${COMMANDS_FILE:-all_commands_runtime.txt}"
: > "$COMMANDS_FILE"
JOB_ID=0

# "n_nodes total_edges m_attrs" -- m_attrs is overridden per family below
SYNTH_CONFIGS=(
    "102400 409600"
    "102400 102400"
    "25600 102400"
    "25600 25600"
    "6400 25600"
    "6400 6400"
    "1600 6400"
    "1600 1600"
    "400 1600"
    "400 400"
    "100 400"
    "100 100"
)

for synth_type in neighbor_feature neighbor_only; do
    if [[ "$synth_type" == "neighbor_feature" ]]; then
        M_ATTRS=10
        CFF_MODE="edge+feature"
    else
        M_ATTRS=6
        CFF_MODE="feature"
    fi

    for config in "${SYNTH_CONFIGS[@]}"; do
        read -r n_nodes total_edges <<< "$config"

        # CF / CF^2 exhaust memory above ~6,400 nodes -- CF-Greedy only there.
        if [[ "$n_nodes" == "25600" || "$n_nodes" == "102400" ]]; then
            EXP_METHOD="cf-greedy"
            PARALLEL_ARGS=""
        else
            EXP_METHOD="cf-greedy cf cff"
            PARALLEL_ARGS="--parallel --num_workers ${WORKERS:-30}"
        fi

        METHOD_TAG="${EXP_METHOD// /+}"
        JOB_ID=$((JOB_ID + 1))

        OUTPUT_FILE="results/test_results/runtime_${synth_type}_node${n_nodes}_edge${total_edges}_${METHOD_TAG}.json"

        printf '%s\n' "synthetic_${synth_type}|python run_test_runtime.py \
--synthetic_type $synth_type \
--n_nodes $n_nodes \
--total_edges $total_edges \
--m_attrs $M_ATTRS \
--budget 9 \
--epsilon 0.01 \
--methods $EXP_METHOD \
--cf_greedy_edge_mode edge_features \
--cf_beta 1 \
--cff_mode $CFF_MODE \
--cff_feat_TH 0.7 \
--cff_edge_TH 0.7 \
$PARALLEL_ARGS \
--output_file $OUTPUT_FILE \
--device cpu" >> "$COMMANDS_FILE"
    done
done

echo "Total: $JOB_ID experiments generated in $COMMANDS_FILE"
