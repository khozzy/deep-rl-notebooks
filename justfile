notebook:
    uv run jupyter notebook --notebook-dir=notebooks/

test:
    pytest

markmaps:
    #!/usr/bin/env bash
    mkdir -p notebooks/markmaps/html
    for file in notebooks/markmaps/*.mm.md; do
        [ -f "$file" ] && npx markmap-cli --no-open "$file" -o "notebooks/markmaps/html/$(basename "$file" .mm.md).html"
    done