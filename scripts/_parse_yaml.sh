#!/bin/bash
# Simple flat YAML parser. No dependencies (pure bash + sed).
# Usage: source this file, then call _parse_yaml <file>
# Sets lowercase shell variables from YAML keys.

_parse_yaml() {
    local yaml_file="$1"
    while IFS= read -r line; do
        # skip comments and blank lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// /}" ]] && continue
        # match key: value
        if [[ "$line" =~ ^[[:space:]]*([a-zA-Z_][a-zA-Z0-9_]*)[[:space:]]*:[[:space:]]*(.*) ]]; then
            local key="${BASH_REMATCH[1]}"
            local val="${BASH_REMATCH[2]}"
            # strip inline comment
            val="${val%%#*}"
            # trim trailing whitespace
            val="${val%"${val##*[![:space:]]}"}"
            # strip one layer of surrounding quotes
            val="${val#\"}" ; val="${val%\"}"
            val="${val#\'}" ; val="${val%\'}"
            # assign using printf to preserve special chars in value
            printf -v "$key" '%s' "$val"
        fi
    done < "$yaml_file"
}
