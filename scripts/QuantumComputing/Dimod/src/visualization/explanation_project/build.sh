#!/bin/bash
# Compile the two documents and put the PDFs at the top of the Dimod folder.
#   bash src/visualization/explanation_project/build.sh      (from anywhere)
set -eo pipefail
cd "$(dirname "$0")"
python qpu_check_figure.py > /dev/null 2>&1 || echo "(qpu_check_figure.py not run: matplotlib missing; using the existing png)"
for doc in explanation_project dwave_in_the_code; do
    for _ in 1 2; do                               # twice: table of contents and references
        pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex" > /dev/null
    done
    mv "$doc.pdf" ../../../"$doc.pdf"
    rm -f "$doc".{aux,log,out,toc}
    echo "written: $(cd ../../.. && pwd)/$doc.pdf"
done
