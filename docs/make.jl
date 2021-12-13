using naumer_ICML_2022
using Documenter

DocMeta.setdocmeta!(naumer_ICML_2022, :DocTestSetup, :(using naumer_ICML_2022); recursive=true)

makedocs(;
    modules=[naumer_ICML_2022],
    authors="Helmuth Naumer <hnaumer2@illinois.edu>",
    repo="https://github.com/helmuthn/naumer_ICML_2022.jl/blob/{commit}{path}#{line}",
    sitename="naumer_ICML_2022.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://helmuthn.github.io/naumer_ICML_2022.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="helmuthn/naumer_ICML_2022.jl",
    devbranch="main",
)
