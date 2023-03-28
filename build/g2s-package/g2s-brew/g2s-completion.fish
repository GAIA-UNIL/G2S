function __fish_g2s_no_subcommand -d "Test for no subcommand"
    for i in (commandline -opc)
        if contains -- $i server enable disable killall help -h
            return 1
        end
    end
    return 0
end

complete -f -c g2s -n '__fish_g2s_no_subcommand' -a 'server' -d 'Server'
complete -f -c g2s -n '__fish_g2s_no_subcommand' -a 'enable' -d 'Enable'
complete -f -c g2s -n '__fish_g2s_no_subcommand' -a 'disable' -d 'Disable'
complete -f -c g2s -n '__fish_g2s_no_subcommand' -a 'killall' -d 'Killall'
complete -f -c g2s -n '__fish_g2s_no_subcommand' -a 'help' -d 'Help'
complete -f -c g2s -n '__fish_g2s_no_subcommand' -a '-h' -d 'Help'

complete -f -c g2s -n '__fish_seen_subcommand_from server' -a '-d' -d 'Option -d'
complete -f -c g2s -n '__fish_seen_subcommand_from server' -a '-To' -d 'Option -To'
complete -f -c g2s -n '__fish_seen_subcommand_from server' -a '-p' -d 'Option -p'
complete -f -c g2s -n '__fish_seen_subcommand_from server' -a '-kod' -d 'Option -kod'
complete -f -c g2s -n '__fish_seen_subcommand_from server' -a '-age' -d 'Option -age'
complete -f -c g2s -n '__fish_seen_subcommand_from server' -a '-maxCJ' -d 'Option -maxCJ'
complete -f -c g2s -n '__fish_seen_subcommand_from server' -a '-h' -d 'Option -h'
complete -f -c g2s -n '__fish_seen_subcommand_from server' -a '--help' -d 'Option --help'
