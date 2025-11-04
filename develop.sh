#!/bin/sh

tmp_requirements_list=$(mktemp)
find . -name "requirements.txt" > "$tmp_requirements_list"

while IFS= read -r line; do
    if command -v pip3 >/dev/null 2>&1; then
        PIP_CMD=pip3
    elif command -v pip >/dev/null 2>&1; then
        PIP_CMD=pip
    else
        echo "Error: pip not found. Please install pip or pip3." >&2
        exit 1
    fi
    $PIP_CMD install -r "$line" >/dev/null 2>&1
done < "$tmp_requirements_list"
rm "$tmp_requirements_list"

found_test=""
test_args=""

for arg in "$@"; do
    if [ "$arg" = "test" ] || [ "$arg" = "-test" ]; then
        found_test="$arg"
        break
    fi
done

if [ -n "$found_test" ]; then
    chmod +x Test/test.sh

    if [ "$found_test" = "test" ]; then
        ./Test/test.sh
    else
        # collect args after '-test'
        after_test=false
        for arg in "$@"; do
            if $after_test; then
                test_args="$test_args $arg"
            elif [ "$arg" = "-test" ]; then
                after_test=true
            fi
        done
        ./Test/test.sh $test_args
    fi
else
    echo "No 'test' or '-test' argument provided."
fi