#compdef g2s

local state

_g2s() {
    local commands server_options
    commands=("server" "enable" "disable" "killall" "help" "-h")
    server_options=("-d" "-To" "-p" "-kod" "-age" "-maxCJ" "-h" "--help")

    if (( CURRENT == 2 )); then
        _describe -t commands 'g2s command' commands
        return
    fi

    if [[ $words[2] == "server" ]]; then
        _describe -t server_options 'g2s server options' server_options
        return
    fi
}

_g2s "$@"
