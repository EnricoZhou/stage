#!/usr/bin/env bash
#!/bin/sh

# should be submitted from PeerRead dir

OUTPUT_DIR_BASE=../out/PeerRead/real/
SEED=0

#rm -rf ${OUTPUT_DIR_BASE}
mkdir -p ${OUTPUT_DIR_BASE}

# def real():
#     estimates = {}
#     estimates['buzzy'] = dragon_att('../out/PeerRead/real/o_accepted_t_buzzy_title')
#     estimates['theorem_referenced'] = dragon_att('../out/PeerRead/real/o_accepted_t_theorem_referenced')
#     estimate_df = pd.DataFrame(estimates)
#     print(estimate_df.round(2))

declare -a TREATMENTS=(
  #         'title_contains_deep'
  #         'title_contains_neural'
  #         'title_contains_embedding'
  'buzzy_title'
  #         'contains_appendix'
  'theorem_referenced'
  #         'equation_referenced'
  #         'year'
  #         'venue'
)

# export NUM_SPLITS=10
export NUM_SPLITS=2  # 👈 ridotto per test locale, puoi rimettere 10 se vuoi

for TREATMENT in "${TREATMENTS[@]}"; do
  export TREATMENT=${TREATMENT}
  for SPLITi in $(seq 0 $(($NUM_SPLITS - 1))); do
    export SPLIT=${SPLITi}
    NAME=o_accepted_t_${TREATMENT}.split${SPLIT}.seed${SEED}
    export OUTPUT_DIR=${OUTPUT_DIR_BASE}o_accepted_t_${TREATMENT}/split${SPLIT}

    # AGGIUNTO
    mkdir -p ${OUTPUT_DIR}

    echo ">>> Avvio training per trattamento=${TREATMENT}, split=${SPLIT}"

    bash ./PeerRead/submit_scripts/emayhem/paper_experiments/run_real_causal_bert.sh > ${OUTPUT_DIR_BASE}${NAME}.out 2>&1
  done
done
