using naumer_ICML_2021
using Documenter

DocMeta.setdocmeta!(naumer_ICML_2021, :DocTestSetup, :(using naumer_ICML_2021); recursive=true)

makedocs(;
    modules=[naumer_ICML_2021],
    authors="Helmuth Naumer <hnaumer2@illinois.edu>",
    repo="https://github.com/helmuthn/naumer_ICML_2021.jl/blob/{commit}{path}#{line}",
    sitename="naumer_ICML_2021.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://helmuthn.github.io/naumer_ICML_2021.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/helmuthn/naumer_ICML_2021.jl",
    devbranch="main",
)
