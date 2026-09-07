#!/bin/sh
# Acepta ambas invocaciones:
#   docker run imagen --input ... --output ... --method 1
#   docker run imagen python main.py --input ... --output ... --method 1
#
# El enunciado documenta la segunda forma; la primera es la idiomatica de
# Docker. Soportar ambas evita que una diferencia de invocacion invalide
# la evaluacion automatizada.

if [ "$1" = "python" ] || [ "$1" = "python3" ]; then
    shift
    [ "$1" = "main.py" ] && shift
    [ "$1" = "/app/main.py" ] && shift
fi

exec python /app/main.py "$@"
