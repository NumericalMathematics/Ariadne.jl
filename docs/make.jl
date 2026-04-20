using Ariadne
using Theseus
using Documenter
import Documenter.Remotes: GitHub
using Literate
using DocumenterCitations

DocMeta.setdocmeta!(Ariadne, :DocTestSetup, :(using Ariadne); recursive = true)
DocMeta.setdocmeta!(Theseus, :DocTestSetup, :(using Theseus); recursive = true)


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
    "Trixi IMEX SSP" => "trixi_imex_ssp",
    "Trixi IMEX ARS" => "trixi_imex_ars",
    "Trixi IMEX Von Karman street" => "trixi_imex_von_karman_street",
]

for (_, name) in examples
    example_filepath = joinpath(EXAMPLES_DIR, string(name, ".jl"))
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter = true)
end

examples = [title => joinpath("generated", string(name, ".md")) for (title, name) in examples]

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(;
    modules = [Ariadne, Theseus],
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
        "Ariadne.jl" => "index.md",
        "Theseus.jl" => "theseus.md",
        "Examples" => examples,
    ],
    doctest = true,
    linkcheck = true,
    plugins = [bib]
)

deploydocs(;
    repo = "github.com/NumericalMathematics/Ariadne.jl.git",
    devbranch = "main",
    # Only push previews if all the relevant environment variables are non-empty.
    push_preview = all(
        !isempty,
        (
            get(ENV, "GITHUB_TOKEN", ""),
            get(ENV, "DOCUMENTER_KEY", ""),
        )
    )
)
