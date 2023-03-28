_g2s_commands()
{
    local commands="server enable disable killall help -h"
    COMPREPLY=( $(compgen -W "${commands}" -- "${COMP_WORDS[1]}") )
}

_g2s_server_options()
{
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"
    local options="-d -To -p -kod -age -maxCJ -h --help"
    local used_options="$(echo "${COMP_LINE}" | grep -Eo -- '-[a-zA-Z]+')"

    # Remove any options that have already been specified
    for opt in ${used_options}; do
        options=$(echo "${options}" | sed -e "s/${opt} //")
    done

    case "${prev}" in
        -To)
            # Handle the case where the previous word was the -To option
            COMPREPLY=("0")
            return 0
            ;;
        -p)
            # Handle the case where the previous word was the -p option
            COMPREPLY=("8128")
            return 0
            ;;
        -maxCJ)
            # Handle the case where the previous word was the -maxCJ option
            COMPREPLY=("2" "4" "8" "16")
            return 0
            ;;
        -age)
            # Handle the case where the previous word was the -age option
            COMPREPLY=("86400")
            return 0
            ;;
        *)
            if [[ "${prev}" == -* ]]; then
                # Handle the case where the previous word was an option that doesn't require a value
                COMPREPLY=($(compgen -W "${options}" -- "${cur}"))
            elif [[ "${cur}" == -* ]]; then
                # Handle the case where the current word is an option
                COMPREPLY=($(compgen -W "${options}" -- "${cur}"))
            else
                # Handle the case where the current word is not an option
                COMPREPLY=($(compgen -W "${options}" -- "${cur}"))
            fi
            ;;
    esac

    if [ "${#COMPREPLY[@]}" -eq 0 ]; then
        COMPREPLY=("${cur}")
        return 0
    fi
}

_g2s()
{
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"

    case "${prev}" in
        g2s)
            _g2s_commands
            return 0
            ;;
        enable|disable|killall)
            COMPREPLY=()
            return 0
            ;;
        *)
            ;;
    esac

    case "${cur}" in
        g2s)
            _g2s_commands
            return 0
            ;;
    esac


    if [[ "${COMP_WORDS[1]}" == "server" ]]; then
        _g2s_server_options
        return 0
    fi

    COMPREPLY=()
    return 0
}

