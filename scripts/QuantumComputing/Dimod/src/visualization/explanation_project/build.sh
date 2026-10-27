#!/bin/bash
# Compile explanation_project.tex and put the PDF at the top of the Dimod folder.
#   bash src/visualization/explanation_project/build.sh      (from anywhere)
set -eo pipefail
cd "$(dirname "$0")"
for _ in 1 2; do                                   # twice: table of contents and references
    pdflatex -interaction=nonstopmode -halt-on-error explanation_project.tex > /dev/null
done
mv explanation_project.pdf ../../../explanation_project.pdf
rm -f explanation_project.{aux,log,out,toc}
echo "written: $(cd ../../.. && pwd)/explanation_project.pdf"
