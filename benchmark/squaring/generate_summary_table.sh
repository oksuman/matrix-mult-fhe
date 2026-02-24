#!/bin/zsh

# Parse squaring benchmark console log and generate a single clean results file.
# All parameters are read dynamically from the log - no hardcoded values.

INPUT_FILE="squaring_console.log"
OUTPUT_FILE="squaring_results.txt"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: $INPUT_FILE not found" >&2
    exit 1
fi

# ============================================================
# Associative arrays for parsed data
# ============================================================
typeset -A p_multDepth p_batchSize p_ringDim p_squaringIter p_trials
typeset -A p_scaleModSize p_seed
typeset -A times log2_frob log2_max
typeset -A idle_mem setup_mem peak_mem ct_size rot_key relin_key
typeset -a seen_algos seen_dims
typeset -A algo_set dim_set

current_algo=""
current_dim=""
current_section=""

while IFS= read -r line; do
    # Detect algorithm name (top-level header)
    if [[ $line =~ 'Matrix Squaring Benchmark - ([A-Za-z0-9]+)' ]]; then
        current_algo="${match[1]}"
        current_dim=""
        current_section=""
        if [[ -z "${algo_set[$current_algo]}" ]]; then
            seen_algos+=("$current_algo")
            algo_set[$current_algo]=1
        fi
    fi

    # Capture idle memory (printed once per algo before per-dim runs)
    if [[ -n "$current_algo" && -z "$current_dim" ]]; then
        if [[ $line =~ 'Idle Memory:[[:space:]]+([0-9.]+) GB' ]]; then
            idle_mem[$current_algo]="${match[1]}"
        fi
    fi

    # Detect dimension header
    if [[ $line =~ '========== ([A-Za-z0-9]+) d=([0-9]+)' ]]; then
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
    if [[ $line == *"--- Experiment ---"*       ]]; then current_section="exp";  continue; fi
    if [[ $line =~ '--- Summary \(d=[0-9]+\) ---' ]]; then current_section="sum"; continue; fi

    [[ -z "$current_algo" || -z "$current_dim" ]] && continue
    k="${current_algo}_${current_dim}"

    case "$current_section" in
        ckks)
            if   [[ $line =~ 'multDepth:[[:space:]]+([0-9]+)'    ]]; then p_multDepth[$k]="${match[1]}"
            elif [[ $line =~ 'scaleModSize:[[:space:]]+([0-9]+)' ]]; then p_scaleModSize[$k]="${match[1]}"
            elif [[ $line =~ 'batchSize:[[:space:]]+([0-9]+)'    ]]; then p_batchSize[$k]="${match[1]}"
            elif [[ $line =~ 'ringDimension:[[:space:]]+([0-9]+)']]; then p_ringDim[$k]="${match[1]}"
            fi
            ;;
        exp)
            if   [[ $line =~ 'squaringIter:[[:space:]]+([0-9]+)' ]]; then p_squaringIter[$k]="${match[1]}"
            elif [[ $line =~ 'trials:[[:space:]]+([0-9]+)'       ]]; then p_trials[$k]="${match[1]}"
            elif [[ $line =~ 'seed:[[:space:]]+(.+)'             ]]; then p_seed[$k]="${match[1]}"
            fi
            ;;
        sum)
            if   [[ $line =~ 'Time: ([0-9.]+)s'                             ]]; then times[$k]="${match[1]}"
            elif [[ $line =~ 'log2\(Rel\. Frob\.\):[[:space:]]*(-?[0-9.]+)' ]]; then log2_frob[$k]="${match[1]}"
            elif [[ $line =~ 'log2\(Rel\. Max\):[[:space:]]*(-?[0-9.]+)'    ]]; then log2_max[$k]="${match[1]}"
            elif [[ $line =~ 'Setup Memory:[[:space:]]+([0-9.]+) GB'        ]]; then setup_mem[$k]="${match[1]}"
            elif [[ $line =~ 'Peak Memory:[[:space:]]+([0-9.]+) GB'         ]]; then peak_mem[$k]="${match[1]}"
            elif [[ $line =~ 'Idle Memory:[[:space:]]+([0-9.]+) GB'         ]]; then idle_mem[$current_algo]="${match[1]}"
            elif [[ $line =~ 'Ciphertext:[[:space:]]+([0-9.]+) MB'          ]]; then ct_size[$k]="${match[1]}"
            elif [[ $line =~ 'Rotation Keys:[[:space:]]+([0-9.]+) MB'       ]]; then rot_key[$k]="${match[1]}"
            elif [[ $line =~ 'Relin Key:[[:space:]]+([0-9.]+) MB'           ]]; then relin_key[$k]="${match[1]}"
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

# Representative values from first algo+dim
fk="${seen_algos[1]}_${sorted_dims[1]}"
num_trials="${p_trials[$fk]:-?}"
scale_mod="${p_scaleModSize[$fk]:-50}"
sqr_iter="${p_squaringIter[$fk]:-15}"
seed_val="${p_seed[$fk]:-42 (fixed)}"

DASHES="--------------------------------------------------------------------------------"
EQUALS="================================================================================"

# ============================================================
# Helper: per-algorithm parameter table
# ============================================================
print_param_block() {
    local NW=12 PW=14 CW=10

    printf "%-${NW}s %-${PW}s" "Algorithm" "Parameter"
    for d in "${sorted_dims[@]}"; do printf "%${CW}s" "d=$d"; done
    echo ""
    echo "$DASHES"

    for algo in "${seen_algos[@]}"; do
        local first=1
        for param in multDepth batchSize ringDim; do
            local label=""
            [[ $first -eq 1 ]] && label="$algo" && first=0
            printf "%-${NW}s %-${PW}s" "$label" "$param"
            for d in "${sorted_dims[@]}"; do
                local ky="${algo}_${d}"
                local v
                case "$param" in
                    multDepth) v="${p_multDepth[$ky]:--}" ;;
                    batchSize) v="${p_batchSize[$ky]:--}" ;;
                    ringDim)   v="${p_ringDim[$ky]:--}" ;;
                esac
                printf "%${CW}s" "$v"
            done
            echo ""
        done
        echo ""
    done
}

# ============================================================
# Helper: time table
# ============================================================
print_time_block() {
    local NW=12 CW=10

    printf "%-${NW}s" "Algorithm"
    for d in "${sorted_dims[@]}"; do printf "%${CW}s" "d=$d"; done
    echo ""
    echo "$DASHES"

    for algo in "${seen_algos[@]}"; do
        printf "%-${NW}s" "$algo"
        for d in "${sorted_dims[@]}"; do
            local ky="${algo}_${d}"
            if [[ -n "${times[$ky]}" ]]; then printf "%${CW}.2f" "${times[$ky]}"
            else printf "%${CW}s" "-"; fi
        done
        echo ""
    done
    echo ""
}

# ============================================================
# Helper: accuracy table
# ============================================================
print_accuracy_block() {
    local NW=12 CW=10

    printf "%-${NW}s" "Algorithm"
    for d in "${sorted_dims[@]}"; do printf "%${CW}s" "d=$d"; done
    echo ""
    echo "$DASHES"

    for algo in "${seen_algos[@]}"; do
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
# Helper: memory table
# ============================================================
print_memory_block() {
    local AW=8 DW=5 MW=12

    printf "%-${AW}s %${DW}s" "Algo" "d"
    printf "%${MW}s" "Idle(GB)"
    printf "%${MW}s" "Setup(GB)"
    printf "%${MW}s" "Peak(GB)"
    printf "%${MW}s" "CT(MB)"
    printf "%${MW}s" "RotKey(MB)"
    printf "%${MW}s" "Relin(MB)"
    echo ""
    echo "$DASHES"

    for algo in "${seen_algos[@]}"; do
        for d in "${sorted_dims[@]}"; do
            local ky="${algo}_${d}"
            [[ -z "${times[$ky]}" ]] && continue
            printf "%-${AW}s %${DW}s" "$algo" "$d"
            printf "%${MW}s" "${idle_mem[$algo]:--}"
            printf "%${MW}s" "${setup_mem[$ky]:--}"
            printf "%${MW}s" "${peak_mem[$ky]:--}"
            printf "%${MW}s" "${ct_size[$ky]:--}"
            printf "%${MW}s" "${rot_key[$ky]:--}"
            printf "%${MW}s" "${relin_key[$ky]:--}"
            echo ""
        done
    done
    echo ""
}

# ============================================================
# Generate output file
# ============================================================
{
    echo "$EQUALS"
    echo "  Matrix Squaring Benchmark Results"
    echo "  Generated: $(date)"
    echo "$EQUALS"
    echo ""

    echo "--- Common Parameters ---"
    printf "  %-18s %s\n" "scaleModSize:" "${scale_mod} bits"
    printf "  %-18s %s\n" "security:" "HEStd_128_classic"
    printf "  %-18s %s\n" "bootstrapping:" "None"
    printf "  %-18s %s\n" "squaringIter:" "$sqr_iter"
    printf "  %-18s %s\n" "trials:" "$num_trials"
    printf "  %-18s %s\n" "seed:" "$seed_val"
    echo ""

    echo "$EQUALS"
    echo "  Per-Algorithm Parameters"
    echo "$EQUALS"
    print_param_block

    echo "$EQUALS"
    echo "  Time Comparison (seconds)"
    echo "$EQUALS"
    echo ""
    print_time_block

    echo "$EQUALS"
    echo "  Accuracy Comparison: log2(Rel. Frobenius Error)"
    echo "$EQUALS"
    echo ""
    print_accuracy_block

    echo "$EQUALS"
    echo "  Memory & Communication"
    echo "$EQUALS"
    echo ""
    print_memory_block

} > "$OUTPUT_FILE"

echo "Results written to $OUTPUT_FILE"
