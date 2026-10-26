# Usage (from the repository root):
#   python -m skgame category                    # regenerate category/category.json
#   julia --project=julia/SKGameCategory -e 'using Pkg; Pkg.instantiate()'
#   julia --project=julia/SKGameCategory julia/run_category.jl
using SKGameCategory
C = load_category(joinpath(@__DIR__, "..", "category", "category.json"))
rep = check_category(C)
println("objects: ", rep.n_objects, "  edges: ", rep.n_edges,
        "  math objects: ", rep.n_math_objects, "  math morphisms: ", rep.n_math_morphisms)
println("topological order:\n  ", join(rep.order, "\n  "))
println("\nderivation of Fig17:\n  ", derivation(C, "Fig17OrderParameter"))
println("\nimpacted by dmft_solve: ", impacted_by(C, "dmft_solve"))
write_graphviz(C, joinpath(@__DIR__, "..", "category", "category_catlab.dot"))
