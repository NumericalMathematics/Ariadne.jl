pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add Ariadne to environment stack

using Ariadne
using Documenter
import Documenter.Remotes: GitHub
using Literate
using PlutoStaticHTML
using DocumenterCitations

const NOTEBOOK_DIR = joinpath(@__DIR__, "src", "notebooks")

"""
    build()

Run all Pluto notebooks (".jl" files) in `NOTEBOOK_DIR`.
"""
function build()
    println("Building notebooks in $NOTEBOOK_DIR")
    oopts = OutputOptions(; append_build_context = false)
    output_format = documenter_output
    bopts = BuildOptions(NOTEBOOK_DIR; output_format)
    build_notebooks(bopts, oopts)
    return nothing
end

# Build the notebooks; defaults to true.
if get(ENV, "BUILD_DOCS_NOTEBOOKS", "true") == "true"
    build()
end


DocMeta.setdocmeta!(Ariadne, :DocTestSetup, :(using Ariadne); recursive = true)


##
# Generate examples
##

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR = joinpath(@__DIR__, "src/generated")

examples = [
    "Bratu -- 1D" => "bratu",
    "Bratu -- KernelAbstractions" => "bratu_ka",
    "Simple" => "simple",
    "BVP" => "bvp",
    "Implicit" => "implicit",
    "Implicit -- Spring" => "spring",
    "Implicit -- Heat 1D" => "heat_1D",
    "Implicit -- Heat 1D DG" => "heat_1D_DG",
    "Implicit -- Heat 2D" => "heat_2D",
    "Trixi" => "trixi",
    "Trixi Rosenbrock" => "trixi_rosenbrock",
    "Trixi Rosenbrock Warm Bubble" => "trixi_rosenbrock_warm_bubble",
    "Trixi Rosenbrock Blast Wave" => "trixi_rosenbrock_blast_wave",
]

for (_, name) in examples
    example_filepath = joinpath(EXAMPLES_DIR, string(name, ".jl"))
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter = true)
end

examples = [title => joinpath("generated", string(name, ".md")) for (title, name) in examples]

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(;
    modules = [Ariadne],
    authors = "Valentin Churavy",
    repo = GitHub("vchuravy", "Ariadne.jl"),
    sitename = "Ariadne.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://numericalmathematics.github.io/Ariadne.jl",
        assets = [
            "assets/citations.css",
        ],
        mathengine = MathJax3(),
        size_threshold = 10_000_000,
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => examples,
        "Notebooks" => [
            "Heat 2D" => "notebooks/heat_2d.md",
            "Heat 1D DG" => "notebooks/heat_1D_DG.md",
        ],
    ],
    doctest = true,
    linkcheck = true,
    plugins = [bib]
)

deploydocs(;
    repo = "github.com/NumericalMathematics/Ariadne.jl.git",
    devbranch = "main",
    push_preview = true,
)
