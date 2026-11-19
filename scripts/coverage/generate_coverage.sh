#!/bin/bash
set -e

BUILD_DIR="${1:-$PWD/build}"
COVERAGE_DIR="$BUILD_DIR/coverage"

echo "========================================"
echo "  Nova Coverage Report Generator"
echo "========================================"
echo "Build directory: $BUILD_DIR"
echo "Coverage directory: $COVERAGE_DIR"
echo ""

# Check dependencies
if ! command -v lcov &> /dev/null; then
    echo "Error: lcov not found. Install with: apt install lcov"
    exit 1
fi

if ! command -v genhtml &> /dev/null; then
    echo "Error: genhtml not found. Install with: apt install lcov"
    exit 1
fi

# Create output directories
mkdir -p "$COVERAGE_DIR/lcov"
mkdir -p "$COVERAGE_DIR/module"

echo "Capturing coverage data..."
cd "$BUILD_DIR"

# Capture coverage from all .gcda and .gcno files
lcov --capture --directory . \
    --output-file "$COVERAGE_DIR/lcov/app.info" \
    --base-dir . \
    --gcov-tool gcov \
    2>/dev/null || true

# Remove system headers and third-party code
lcov --remove "$COVERAGE_DIR/lcov/app.info" \
    '/usr/*' \
    --output-file "$COVERAGE_DIR/lcov/filtered.info" \
    2>/dev/null || cp "$COVERAGE_DIR/lcov/app.info" "$COVERAGE_DIR/lcov/filtered.info"

# Extract total coverage
lcov --extract "$COVERAGE_DIR/lcov/filtered.info" \
    "$PWD/include/cuda/*" \
    --output-file "$COVERAGE_DIR/lcov/total.info" \
    2>/dev/null || true

# Generate per-module coverage. The module list is derived from the header
# tree rather than hardcoded: the previous hardcoded list pointed at
# include/nova/<module>, a path that has never existed here (the headers live
# under include/cuda), so every lcov --extract below silently produced nothing
# and the per-module tables in the report were always empty.
MODULES=()
for module_dir in "$PWD"/include/cuda/*/; do
    [ -d "$module_dir" ] || continue
    MODULES+=("$(basename "$module_dir")")
done

for module in "${MODULES[@]}"; do
    lcov --extract "$COVERAGE_DIR/lcov/filtered.info" \
        "$PWD/include/cuda/$module/*" \
        --output-file "$COVERAGE_DIR/module/${module}.info" \
        2>/dev/null || true
done

echo "Generating HTML report..."
genhtml "$COVERAGE_DIR/lcov/total.info" \
    --output-directory "$COVERAGE_DIR/index.html" \
    --title "Nova Coverage Report" \
    --show-details \
    --branch-coverage \
    --function-coverage \
    2>/dev/null || true

# Generate module-specific reports
for module in "${MODULES[@]}"; do
    if [ -f "$COVERAGE_DIR/module/${module}.info" ]; then
        genhtml "$COVERAGE_DIR/module/${module}.info" \
            --output-directory "$COVERAGE_DIR/index.html/module/${module}" \
            --title "Nova $module Coverage" \
            --show-details \
            2>/dev/null || true
    fi
done

# Calculate and display summary
TOTAL_LINES=$(grep -o 'lines\.\.\.\.: [0-9.]*' "$COVERAGE_DIR/lcov/total.info" 2>/dev/null | head -1 | grep -o '[0-9.]*' || echo "N/A")
TOTAL_BRANCHES=$(grep -o 'branches\.\.\.\.: [0-9.]*' "$COVERAGE_DIR/lcov/total.info" 2>/dev/null | head -1 | grep -o '[0-9.]*' || echo "N/A")
TOTAL_FUNCTIONS=$(grep -o 'functions\.\.\.\.: [0-9.]*' "$COVERAGE_DIR/lcov/total.info" 2>/dev/null | head -1 | grep -o '[0-9.]*' || echo "N/A")

echo ""
echo "========================================"
echo "  Coverage Summary"
echo "========================================"
echo "Line Coverage:    $TOTAL_LINES%"
echo "Branch Coverage:  $TOTAL_BRANCHES%"
echo "Function Coverage: $TOTAL_FUNCTIONS%"
echo ""
echo "HTML Report: file://$COVERAGE_DIR/index.html/index.html"
echo "========================================"

exit 0
