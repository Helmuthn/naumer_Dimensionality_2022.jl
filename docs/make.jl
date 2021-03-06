using naumer_Dimensionality_2022
using Documenter

DocMeta.setdocmeta!(naumer_Dimensionality_2022, :DocTestSetup, :(using naumer_Dimensionality_2022); recursive=true)

makedocs(;
    modules=[naumer_Dimensionality_2022],
    authors="Helmuth Naumer <hnaumer2@illinois.edu>",
    repo="https://github.com/helmuthn/naumer_Dimensionality_2022.jl/blob/{commit}{path}#{line}",
    sitename="naumer_Dimensionality_2022.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://helmuthn.github.io/naumer_Dimensionality_2022.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="Helmuthn/naumer_Dimensionality_2022.jl",
    devbranch="main",
)
