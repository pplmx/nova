#!/bin/bash
set -e

BUILD_DIR="${1:-$PWD/build}"
COVERAGE_DIR="$BUILD_DIR/coverage"

echo "========================================"
echo "  Nova Coverage Summary"
echo "========================================"
echo "Build directory: $BUILD_DIR"
echo ""

if [ ! -f "$COVERAGE_DIR/lcov/total.info" ]; then
    echo "Error: Coverage data not found. Run generate_coverage.sh first."
    exit 1
fi

# Extract overall coverage.
#
# lcov writes "lines......: 85.0% (85 of 100 lines)", so field 2 already ends
# in a percent sign. Strip it here and let the printf below add the one this
# output wants, otherwise the report reads "85.0%%" and the value no longer
# compares as a plain number.
percent_field() {
    local value
    value=$(grep "$1" "$COVERAGE_DIR/lcov/total.info" 2>/dev/null | tail -1 | awk '{print $2}')
    if [ -z "$value" ]; then
        echo "N/A"
    else
        echo "${value%\%}"
    fi
}

TOTAL_LINES=$(percent_field "lines......:")
TOTAL_BRANCHES=$(percent_field "branches....:")
TOTAL_FUNCTIONS=$(percent_field "functions...:")

echo "## Overall Coverage"
printf "| Metric     | Coverage |\n"
printf "|------------|----------|\n"
printf "| Lines      | %s%% |\n" "$TOTAL_LINES"
printf "| Branches   | %s%% |\n" "$TOTAL_BRANCHES"
printf "| Functions  | %s%% |\n" "$TOTAL_FUNCTIONS"

echo ""
echo "## Coverage by Module"
printf "| Module     | Lines     | Branches  | Functions |\n"
printf "|------------|-----------|-----------|-----------|\n"

MODULES=()
for module_dir in "$PWD"/include/cuda/*/; do
    [ -d "$module_dir" ] || continue
    MODULES+=("$(basename "$module_dir")")
done
for module in "${MODULES[@]}"; do
    if [ -f "$COVERAGE_DIR/module/${module}.info" ]; then
        lines=$(grep "lines......:" "$COVERAGE_DIR/module/${module}.info" 2>/dev/null | tail -1 | awk '{print $2}' | tr -d '%')
        branches=$(grep "branches....:" "$COVERAGE_DIR/module/${module}.info" 2>/dev/null | tail -1 | awk '{print $2}' | tr -d '%')
        funcs=$(grep "functions..." "$COVERAGE_DIR/module/${module}.info" 2>/dev/null | tail -1 | awk '{print $2}' | tr -d '%')
        printf "| %-10s | %9s | %9s | %9s |\n" "$module" "${lines:-N/A}" "${branches:-N/A}" "${funcs:-N/A}"
    fi
done

echo ""
echo "========================================"

# Check minimum coverage threshold.
#
# Compares with awk rather than bc: bc is not installed on ubuntu-latest, so
# the old `| bc -l` piped into a `|| echo 0` fallback, silently reported 0 and
# failed the job on any non-empty report regardless of the actual number.
MIN_COVERAGE="${NOVA_COVERAGE_MIN:-80}"
if [ "$TOTAL_LINES" != "N/A" ]; then
    if awk -v have="$TOTAL_LINES" -v want="$MIN_COVERAGE" 'BEGIN { exit !(have + 0 >= want + 0) }'; then
        echo "Coverage ($TOTAL_LINES%) meets minimum ($MIN_COVERAGE%)"
    else
        echo "Coverage ($TOTAL_LINES%) below minimum ($MIN_COVERAGE%)"
        exit 1
    fi
fi
