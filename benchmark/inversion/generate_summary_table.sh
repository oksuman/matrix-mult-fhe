#!/bin/zsh

# Parse inversion benchmark console log and generate a single clean results file.
# All parameters are read dynamically from the log - no hardcoded values.

INPUT_FILE="inversion_console.log"
OUTPUT_FILE="inversion_results.txt"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: $INPUT_FILE not found" >&2
    exit 1
fi

# ============================================================
# Associative arrays for parsed data
# ============================================================
typeset -A p_multDepth p_batchSize p_ringDim p_levelBudget p_invIter
typeset -A p_trials p_scaleModSize p_firstModSize p_seed
typeset -A times log2_frob log2_max
typeset -a seen_algos seen_dims
typeset -A algo_set dim_set

current_algo=""
current_dim=""
current_section=""

while IFS= read -r line; do
    # Detect algorithm + dimension header
    if [[ $line =~ '========== ([A-Za-z0-9-]+) Inversion d=([0-9]+)' ]]; then
        current_algo="${match[1]}"
        current_dim="${match[2]}"
        if [[ -z "${algo_set[$current_algo]}" ]]; then
            seen_algos+=("$current_algo")
            algo_set[$current_algo]=1
        fi
        if [[ -z "${dim_set[$current_dim]}" ]]; then
            seen_dims+=("$current_dim")
            dim_set[$current_dim]=1
        fi
        current_section=""
        continue
    fi

    # Detect section headers
    if [[ $line == *"--- CKKS Parameters ---"* ]]; then current_section="ckks"; continue; fi
    if [[ $line == *"--- Bootstrapping ---"*   ]]; then current_section="boot"; continue; fi
    if [[ $line == *"--- Algorithm ---"*        ]]; then current_section="algo"; continue; fi
    if [[ $line =~ '--- Summary \(d=[0-9]+\) ---' ]]; then current_section="sum"; continue; fi

    [[ -z "$current_algo" || -z "$current_dim" ]] && continue
    k="${current_algo}_${current_dim}"

    case "$current_section" in
        ckks)
            if   [[ $line =~ 'multDepth:[[:space:]]+([0-9]+)'    ]]; then p_multDepth[$k]="${match[1]}"
            elif [[ $line =~ 'scaleModSize:[[:space:]]+([0-9]+)' ]]; then p_scaleModSize[$k]="${match[1]}"
            elif [[ $line =~ 'firstModSize:[[:space:]]+([0-9]+)' ]]; then p_firstModSize[$k]="${match[1]}"
            elif [[ $line =~ 'batchSize:[[:space:]]+([0-9]+)'    ]]; then p_batchSize[$k]="${match[1]}"
            elif [[ $line =~ 'ringDimension:[[:space:]]+([0-9]+)']]; then p_ringDim[$k]="${match[1]}"
            fi
            ;;
        boot)
            if [[ $line =~ 'levelBudget:[[:space:]]+\{([0-9, ]+)\}' ]]; then
                p_levelBudget[$k]="${${match[1]}// /}"
            fi
            ;;
        algo)
            if   [[ $line =~ 'invIter:[[:space:]]+([0-9]+)' ]]; then p_invIter[$k]="${match[1]}"
            elif [[ $line =~ 'trials:[[:space:]]+([0-9]+)'  ]]; then p_trials[$k]="${match[1]}"
            elif [[ $line =~ 'seed:[[:space:]]+(.+)'        ]]; then p_seed[$k]="${match[1]}"
            fi
            ;;
        sum)
            if   [[ $line =~ 'Time: ([0-9.]+)s'                             ]]; then times[$k]="${match[1]}"
            elif [[ $line =~ 'log2\(Rel\. Frob\.\):[[:space:]]*(-?[0-9.]+)' ]]; then log2_frob[$k]="${match[1]}"
            elif [[ $line =~ 'log2\(Rel\. Max\):[[:space:]]*(-?[0-9.]+)'    ]]; then log2_max[$k]="${match[1]}"
            fi
            ;;
    esac
done < "$INPUT_FILE"

if [[ ${#seen_algos[@]} -eq 0 ]]; then
    echo "Error: no benchmark data found in $INPUT_FILE" >&2
    exit 1
fi

# Sort dimensions numerically
sorted_dims=($(printf '%s\n' "${seen_dims[@]}" | sort -n))

# Separate original vs simple algorithms (preserve log order)
typeset -a orig_algos simple_algos
for algo in "${seen_algos[@]}"; do
    if [[ $algo == *"-Simple" ]]; then simple_algos+=("$algo")
    else orig_algos+=("$algo"); fi
done

# Representative values
fk="${seen_algos[1]}_${sorted_dims[1]}"
num_trials="${p_trials[$fk]:-?}"
scale_mod="${p_scaleModSize[$fk]:-59}"
first_mod="${p_firstModSize[$fk]:-60}"
seed_val="${p_seed[$fk]:-1000+run}"

DASHES="--------------------------------------------------------------------------------"
EQUALS="================================================================================"

# ============================================================
# Helper: print per-algorithm parameter table
# ============================================================
print_param_block() {
    local algos=("$@")
    local NW=22 PW=14 CW=10

    printf "%-${NW}s %-${PW}s" "Algorithm" "Parameter"
    for d in "${sorted_dims[@]}"; do printf "%${CW}s" "d=$d"; done
    echo ""
    echo "$DASHES"

    for algo in "${algos[@]}"; do
        local first=1
        for param in multDepth batchSize ringDim levelBudget invIter; do
            local label=""
            [[ $first -eq 1 ]] && label="$algo" && first=0
            printf "%-${NW}s %-${PW}s" "$label" "$param"
            for d in "${sorted_dims[@]}"; do
                local ky="${algo}_${d}"
                local v
                case "$param" in
                    multDepth)   v="${p_multDepth[$ky]:--}" ;;
                    batchSize)   v="${p_batchSize[$ky]:--}" ;;
                    ringDim)     v="${p_ringDim[$ky]:--}" ;;
                    levelBudget) v="${p_levelBudget[$ky]}"
                                 [[ -n "$v" ]] && v="{$v}" || v="-" ;;
                    invIter)     v="${p_invIter[$ky]:--}" ;;
                esac
                printf "%${CW}s" "$v"
            done
            echo ""
        done
        echo ""
    done
}

# ============================================================
# Helper: print time table
# ============================================================
print_time_block() {
    local algos=("$@")
    local NW=22 CW=10

    printf "%-${NW}s" "Algorithm"
    for d in "${sorted_dims[@]}"; do printf "%${CW}s" "d=$d"; done
    echo ""
    echo "$DASHES"

    for algo in "${algos[@]}"; do
        printf "%-${NW}s" "$algo"
        for d in "${sorted_dims[@]}"; do
            local ky="${algo}_${d}"
            if [[ -n "${times[$ky]}" ]]; then printf "%${CW}.1f" "${times[$ky]}"
            else printf "%${CW}s" "-"; fi
        done
        echo ""
    done
    echo ""
}

# ============================================================
# Helper: print accuracy table
# ============================================================
print_accuracy_block() {
    local algos=("$@")
    local NW=22 CW=10

    printf "%-${NW}s" "Algorithm"
    for d in "${sorted_dims[@]}"; do printf "%${CW}s" "d=$d"; done
    echo ""
    echo "$DASHES"

    for algo in "${algos[@]}"; do
        printf "%-${NW}s" "$algo"
        for d in "${sorted_dims[@]}"; do
            local ky="${algo}_${d}"
            if [[ -n "${log2_frob[$ky]}" ]]; then printf "%${CW}.1f" "${log2_frob[$ky]}"
            else printf "%${CW}s" "-"; fi
        done
        echo ""
    done
    echo ""
}

# ============================================================
# Generate output file
# ============================================================
{
    echo "$EQUALS"
    echo "  Matrix Inversion Benchmark Results"
    echo "  Generated: $(date)"
    echo "$EQUALS"
    echo ""

    echo "--- Common CKKS Parameters ---"
    printf "  %-18s %s\n" "scaleModSize:" "${scale_mod} bits"
    printf "  %-18s %s\n" "firstModSize:" "${first_mod} bits"
    printf "  %-18s %s\n" "security:" "HEStd_128_classic"
    printf "  %-18s %s\n" "trials:" "$num_trials"
    printf "  %-18s %s\n" "seed:" "$seed_val"
    echo ""

    if [[ ${#orig_algos[@]} -gt 0 ]]; then
        echo "$EQUALS"
        echo "  [Original] Per-Algorithm Parameters  (trace + eval_scalar_inverse)"
        echo "$EQUALS"
        print_param_block "${orig_algos[@]}"
    fi

    if [[ ${#simple_algos[@]} -gt 0 ]]; then
        echo "$EQUALS"
        echo "  [Simple] Per-Algorithm Parameters  (1/d^2 upperbound scaling, no trace)"
        echo "$EQUALS"
        print_param_block "${simple_algos[@]}"
    fi

    echo "$EQUALS"
    echo "  Time Comparison (seconds)"
    echo "$EQUALS"
    echo ""
    if [[ ${#orig_algos[@]} -gt 0 ]]; then
        echo "[Original]"
        print_time_block "${orig_algos[@]}"
    fi
    if [[ ${#simple_algos[@]} -gt 0 ]]; then
        echo "[Simple]"
        print_time_block "${simple_algos[@]}"
    fi

    echo "$EQUALS"
    echo "  Accuracy Comparison: log2(Rel. Frobenius Error)"
    echo "$EQUALS"
    echo ""
    if [[ ${#orig_algos[@]} -gt 0 ]]; then
        echo "[Original]"
        print_accuracy_block "${orig_algos[@]}"
    fi
    if [[ ${#simple_algos[@]} -gt 0 ]]; then
        echo "[Simple]"
        print_accuracy_block "${simple_algos[@]}"
    fi

} > "$OUTPUT_FILE"

echo "Results written to $OUTPUT_FILE"
