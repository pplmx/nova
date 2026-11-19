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

# Extract overall coverage
TOTAL_LINES=$(grep "lines......:" "$COVERAGE_DIR/lcov/total.info" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")
TOTAL_BRANCHES=$(grep "branches....:" "$COVERAGE_DIR/lcov/total.info" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")
TOTAL_FUNCTIONS=$(grep "functions..." "$COVERAGE_DIR/lcov/total.info" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")

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
        lines=$(grep "lines......:" "$COVERAGE_DIR/module/${module}.info" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")
        branches=$(grep "branches....:" "$COVERAGE_DIR/module/${module}.info" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")
        funcs=$(grep "functions..." "$COVERAGE_DIR/module/${module}.info" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")
        printf "| %-10s | %9s | %9s | %9s |\n" "$module" "$lines" "$branches" "$funcs"
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
