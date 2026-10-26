"""
    SKGameCategory

Catlab.jl bookkeeping of the dependency structure of the SK-game reproduction
(arXiv:2312.12252). Everything is built *automatically* from `category/category.json`,
which the Python side regenerates with `python -m skgame category`.

Two free symmetric monoidal categories are built:

* `OutputCat` — objects: output classes (theory / data / table / benchmark / figure);
  generators: `make_C : D₁ ⊗ … ⊗ D_k → C` (domain `I` when C has no inputs).
* `MathCat`   — objects: mathematical objects of the paper (J, Q, m, S, C, Σ, DMFT, …);
  generators: the computations between them.

and two C-sets (ACSets):

* `DepGraph` — the output dependency graph (vertices = output classes, edges = inputs),
* `Uses`     — the relation output-class ↔ math-morphism it evaluates (a span
  Out ← Use → Mor), used for impact analysis ("if I change `dmft_solve`, which figures
  must be recomputed?").

Main API: `load_category`, `derivation`, `upstream`, `downstream`, `topo_order`,
`impacted_by`, `check_category`, `write_graphviz`.
"""
module SKGameCategory

using Catlab
using JSON3

export load_category, SKCategory, derivation, upstream, downstream, topo_order,
       impacted_by, morphisms_of, check_category, write_graphviz, DepGraph, Uses

# ------------------------------------------------------------------ C-set schemas ----
@present SchDepGraph(FreeSchema) begin
    (V, E)::Ob
    src::Hom(E, V)
    tgt::Hom(E, V)
    Name::AttrType
    vname::Attr(V, Name)
    vkind::Attr(V, Name)
    vref::Attr(V, Name)
end
@acset_type DepGraphGeneric(SchDepGraph, index=[:src, :tgt])
const DepGraph = DepGraphGeneric{String}

@present SchUses(FreeSchema) begin
    (Out, Mor, Use)::Ob
    out::Hom(Use, Out)
    mor::Hom(Use, Mor)
    Name::AttrType
    oname::Attr(Out, Name)
    mname::Attr(Mor, Name)
end
@acset_type UsesGeneric(SchUses, index=[:out, :mor])
const Uses = UsesGeneric{String}

# ------------------------------------------------------------------ container ---------
struct SKCategory
    json::Any
    outputs::Presentation          # OutputCat
    math::Presentation             # MathCat
    graph::DepGraph
    uses::Uses
    vertex::Dict{String,Int}
end

_ob(pres, name::AbstractString) = generator(pres, Symbol(name))

function _tensor(pres, names)
    isempty(names) && return munit(FreeSymmetricMonoidalCategory.Ob)
    obs = [_ob(pres, n) for n in names]
    length(obs) == 1 ? obs[1] : foldl(otimes, obs)
end

function _build_presentation(objs, gens)
    pres = Presentation(FreeSymmetricMonoidalCategory)
    for o in objs
        add_generator!(pres, Ob(FreeSymmetricMonoidalCategory, Symbol(o[:name])))
    end
    for g in gens
        dom = _tensor(pres, String.(g[:dom]))
        cod = _ob(pres, String(g[:codom]))
        add_generator!(pres, Hom(Symbol(g[:name]), dom, cod))
    end
    pres
end

"""
    load_category(path = <repo>/category/category.json) -> SKCategory
"""
function load_category(path::AbstractString=joinpath(@__DIR__, "..", "..", "..", "category", "category.json"))
    J = JSON3.read(read(path, String))
    outputs = _build_presentation(J[:objects], J[:generators])
    math = _build_presentation(J[:math][:objects], J[:math][:generators])

    g = DepGraph()
    vertex = Dict{String,Int}()
    for o in J[:objects]
        vertex[String(o[:name])] = add_part!(g, :V; vname=String(o[:name]), vkind=String(o[:kind]),
                                             vref=String(o[:paper_ref]))
    end
    for h in J[:generators], d in h[:dom]
        add_part!(g, :E; src=vertex[String(d)], tgt=vertex[String(h[:codom])])
    end

    u = Uses()
    oid = Dict{String,Int}(); mid = Dict{String,Int}()
    for o in J[:objects]
        oid[String(o[:name])] = add_part!(u, :Out; oname=String(o[:name]))
    end
    for m in J[:math][:generators]
        mid[String(m[:name])] = add_part!(u, :Mor; mname=String(m[:name]))
    end
    for r in J[:math][:uses]
        add_part!(u, :Use; out=oid[String(r[:output])], mor=mid[String(r[:morphism])])
    end
    SKCategory(J, outputs, math, g, u, vertex)
end

# ------------------------------------------------------------------ queries -----------
_name(C::SKCategory, v::Int) = C.graph[v, :vname]
_inputs(C::SKCategory, v::Int) = C.graph[incident(C.graph, v, :tgt), :src]
_outputs(C::SKCategory, v::Int) = C.graph[incident(C.graph, v, :src), :tgt]

"""Composite morphism `I → C` in the free SMC (`make_C ∘ (deriv(D₁) ⊗ … ⊗ deriv(D_k))`)."""
function derivation(C::SKCategory, name::AbstractString)
    gen = generator(C.outputs, Symbol("make_" * name))
    ins = [_name(C, w) for w in _inputs(C, C.vertex[name])]
    isempty(ins) && return gen
    # order of inputs as declared in the generator's domain
    decl = [String(d) for d in only(filter(h -> String(h[:codom]) == name, collect(C.json[:generators])))[:dom]]
    parts = [derivation(C, d) for d in decl]
    compose(length(parts) == 1 ? parts[1] : foldl(otimes, parts), gen)
end

function _closure(C::SKCategory, name::AbstractString, step)
    seen = Set{Int}(); stack = [C.vertex[name]]
    while !isempty(stack)
        v = pop!(stack)
        for w in step(C, v)
            w in seen || (push!(seen, w); push!(stack, w))
        end
    end
    sort([_name(C, v) for v in seen])
end
upstream(C::SKCategory, name) = _closure(C, name, _inputs)
downstream(C::SKCategory, name) = _closure(C, name, _outputs)

"""Kahn topological order of the output dependency graph (errors on a cycle)."""
function topo_order(C::SKCategory)
    g = C.graph
    indeg = Dict(v => length(incident(g, v, :tgt)) for v in parts(g, :V))
    queue = sort([v for v in parts(g, :V) if indeg[v] == 0])
    out = String[]
    while !isempty(queue)
        v = popfirst!(queue)
        push!(out, _name(C, v))
        for w in _outputs(C, v)
            indeg[w] -= 1
            indeg[w] == 0 && push!(queue, w)
        end
    end
    length(out) == nparts(g, :V) || error("dependency graph has a cycle")
    out
end

"""Math morphisms evaluated by an output class."""
morphisms_of(C::SKCategory, out::AbstractString) =
    sort(unique(C.uses[incident(C.uses, only(incident(C.uses, out, :oname)), :out), [:mor, :mname]]))

"""Every output class that must be recomputed if math morphism `m` changes (direct users + downstream)."""
function impacted_by(C::SKCategory, m::AbstractString)
    ids = incident(C.uses, m, :mname)
    isempty(ids) && error("unknown morphism $m")
    direct = unique(C.uses[incident(C.uses, only(ids), :mor), [:out, :oname]])
    sort(unique(vcat(direct, [x for d in direct for x in downstream(C, d)])))
end

"""Well-formedness: acyclic, every generator well typed, every derivation composable,
every non-figure class feeds something, every figure reachable from I."""
function check_category(C::SKCategory)
    order = topo_order(C)
    for name in order
        f = derivation(C, name)
        dom(f) == munit(FreeSymmetricMonoidalCategory.Ob) || error("derivation of $name does not start at I")
        codom(f) == _ob(C.outputs, name) || error("derivation of $name has wrong codomain")
    end
    orphans = [n for n in order if C.graph[C.vertex[n], :vkind] != "figure" && isempty(downstream(C, n)) &&
               C.graph[C.vertex[n], :vkind] != "table"]
    (; n_objects=length(order), n_edges=nparts(C.graph, :E),
       n_math_objects=length(generators(C.math, :Ob)), n_math_morphisms=length(generators(C.math, :Hom)),
       orphans=orphans, order=order)
end

"""Graphviz DOT of the dependency graph (plain text, no Graphviz install needed)."""
function write_graphviz(C::SKCategory, path::AbstractString)
    col = Dict("theory" => "#cfe2ff", "data" => "#d1e7dd", "figure" => "#fff3cd", "table" => "#e2e3e5",
               "benchmark" => "#f8d7da")
    open(path, "w") do io
        println(io, "digraph SKGame { rankdir=LR; node [shape=box, style=filled];")
        for v in parts(C.graph, :V)
            println(io, "  ", _name(C, v), " [fillcolor=\"", col[C.graph[v, :vkind]], "\"];")
        end
        for e in parts(C.graph, :E)
            println(io, "  ", _name(C, C.graph[e, :src]), " -> ", _name(C, C.graph[e, :tgt]), ";")
        end
        println(io, "}")
    end
    path
end

end # module
