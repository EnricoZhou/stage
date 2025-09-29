#!/usr/bin/env bash

OUTPUT_DIR_BASE=../out/PeerRead/buzzy-based-sim/
mkdir -p ${OUTPUT_DIR_BASE}

export NUM_SPLITS=2   # 👈 ridotto per test locale, puoi rimettere 10 se vuoi

# Solo simple per test
declare -a SIMMODES=('simple')

export BETA0=0.25
declare -a BETA1S=(1.0)   # 👈 un solo valore per test veloce
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

                # crea la cartella
                mkdir -p ${OUTPUT_DIR}

                echo ">>> Lancio job locale: ${NAME}"

                # 👇 invece di sbatch, lancio direttamente lo script
                # bash ./PeerRead/submit_scripts/emayhem/paper_experiments/run_buzzy_sim.sh > ${OUTPUT_DIR_BASE}${NAME}.out

                OUTFILE=${OUTPUT_DIR_BASE}/${NAME}.out
                bash ./PeerRead/submit_scripts/emayhem/paper_experiments/run_buzzy_sim.sh > "${OUTFILE}" 2>&1

                # controllo se l’output è stato scritto
                if [[ ! -s "${OUTFILE}" ]]; then
                    echo "⚠️  Attenzione: nessun output scritto in ${OUTFILE}"
                fi
            done
        done
    done
done
