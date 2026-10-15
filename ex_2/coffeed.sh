#!/usr/bin/env bash
# coffeed -- daemon of the department's coffee machine (simulation).
# Writes its log to coffeed.log in the current directory. Start it with
#   bash coffeed.sh &

log=coffeed.log
history_lines=4000
names=(Ada Alan Grace Linus Margaret Dennis Barbara Ken)
drinks=(espresso cappuccino "flat white" "double espresso" "decaf (suspicious)")
first_term=-100
t=0

# Sets $line to the log line for uptime second $1.
make_line() {
    local i=$1 n=$(( $1 / 3 )) msg
    if (( i == 3141 )); then
        msg="ERROR milk reservoir contains orange juice. Who did this?"
    elif (( i % 750 == 0 )); then
        msg="WARNING descaling recommended"
    elif (( i % 3 == 0 )); then
        msg="INFO brewing ${drinks[n % ${#drinks[@]}]} for ${names[n % ${#names[@]}]}"
    elif (( i % 3 == 1 )); then
        msg="INFO grinding beans"
    else
        msg="INFO water temperature $(( 90 + i % 5 )) C"
    fi
    printf -v line '[%02d:%02d:%02d] %s' $(( i / 3600 )) $(( i / 60 % 60 )) $(( i % 60 )) "$msg"
}

say() {
    printf '[%02d:%02d:%02d] %s\n' $(( t / 3600 )) $(( t / 60 % 60 )) $(( t % 60 )) "$1" >> "$log"
}

on_term() {
    if (( SECONDS - first_term <= 10 )); then
        say "INFO second SIGTERM received, shutting down gracefully"
        say "INFO final words: thank you for $t cups. Please descale me."
        exit 0
    fi
    first_term=$SECONDS
    say "WARNING SIGTERM received, but I am brewing! Send SIGTERM again within 10 seconds to confirm."
}
trap on_term TERM

# Uptime so far: write the existing history in one go.
for (( t = 1; t <= history_lines; t++ )); do
    make_line "$t"
    printf '%s\n' "$line"
done > "$log"

echo "coffeed is running, logging to $log" >&2

# From now on, brew one line per second.
for (( ; ; t++ )); do
    sleep 1
    make_line "$t"
    printf '%s\n' "$line" >> "$log"
done
