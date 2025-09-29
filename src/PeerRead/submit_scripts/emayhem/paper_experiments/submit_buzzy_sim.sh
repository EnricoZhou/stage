#!/usr/bin/env bash

OUTPUT_DIR_BASE=../out/PeerRead/buzzy-based-sim/
mkdir -p ${OUTPUT_DIR_BASE}

export BETA0=0.25
export BETA1=1.0
export GAMMA=0.0

export SIMMODE=simple

# export NUM_SPLITS=10
export NUM_SPLITS=3 # 👈 ridotto per test locale, puoi rimettere 10 se vuoi

#declare -a SIMMODES=('simple' 'multiplicative' 'interaction')
declare -a SIMMODES=('simple')

export BETA0=0.25
#todo: in place just to save time by avoiding repeating compute
declare -a BETA1S=(1.0 5.0 25.0)
# declare -a BETA1S=(5.0 25.0)
declare -a GAMMAS=(0.0)

for SIMMODEj in "${SIMMODES[@]}"; do
    export SIMMODE=${SIMMODEj}
    for BETA1j in "${BETA1S[@]}"; do
        export BETA1=${BETA1j}
        for GAMMAj in "${GAMMAS[@]}"; do
            export GAMMA=${GAMMAj}
            for SPLITi in $(seq 0 $(($NUM_SPLITS-1))); do
                export SPLIT=${SPLITi}
                export OUTPUT_DIR=${OUTPUT_DIR_BASE}mode${SIMMODE}/beta0${BETA0}.beta1${BETA1}.gamma${GAMMA}/split${SPLIT}
                NAME=mode${SIMMODE}.beta0${BETA0}.beta1${BETA1}.gamma${GAMMA}.split${SPLIT}

                # 👈 aggiunto: crea la cartella di output per questo split
                mkdir -p ${OUTPUT_DIR}

                echo ">>> Avvio training per trattamento=${TREATMENT}, split=${SPLIT}, beta1=${BETA1}"
                
                bash ./PeerRead/submit_scripts/emayhem/paper_experiments/run_buzzy_sim.sh
            done
        done
    done
done