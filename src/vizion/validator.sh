#!/usr/bin/bash
if [ "$#" -ne 2 ]; then
  echo "validator terminated 🍃 | $0 <schemafile> <inputfile>"
  exit 1
fi

schema_file=$1
input_file=$2

# output files
cleaned_version="cleaned_${input_file}"
rejected_version="rejected_${input_file}"

# files initialization
head -n 1 "$input_file" > "$cleaned_version"
head -n 1 "$input_file" > "$rejected_version"

# Awk rule string
awk_rules=""
col_indx=1

# generating awk rule string with dynamic file parsing
while IFS=: read -r header rule; do
  header=$(echo "$header" | tr -d '[:space:]')
  rule=$(echo "$rule" | tr -d '[:space:]')      # <- fixed: was reading $header

  # skip blank lines
  [ -z "$header" ] && continue

  case "$rule" in
    "number")
      # Regex for matching numbers (one or more digits)
      rule_str="\$${col_indx} ~ /^[0-9]+$/"
      ;;
    "float")
      # int or decimal rule
      rule_str="\$${col_indx} ~ /^[0-9]+(\.[0-9]+)?$/"
      ;;
    "string")
      # Cannot be completely empty
      rule_str="\$${col_indx} != \"\""
      ;;
    *)
      if [[ "$rule" == *"|"* ]]; then
        # Enumerated list of allowed values, e.g. "yes|no|maybe"
        rule_str="\$${col_indx} ~ /^(${rule})$/"
      else
        # Unknown or empty rule: don't constrain this column
        rule_str="1"
      fi
      ;;
  esac

  # Building logic of rule string for final iteration
  if [ -z "$awk_rules" ]; then
    awk_rules="$rule_str"
  else
    awk_rules="$awk_rules && $rule_str"
  fi
  ((col_indx++))
done < "${schema_file}"

# Feeding the rule string for the final validation write
tail -n +2 "$input_file" | awk -F',' -v OFS=',' \
  -v out="$cleaned_version" -v rej="$rejected_version" \
  '{
    if ('"$awk_rules"')
      print >> out
    else
      print >> rej
  }'

echo "File validation completed 📑"
