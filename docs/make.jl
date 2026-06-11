using Documenter
using VariogramAnalysis

makedocs(;
    modules=[VariogramAnalysis],
    authors="msturroc <marcsturrock@protonmail.com> and contributors",
    sitename="VariogramAnalysis.jl",
    format=Documenter.HTML(;
        canonical="https://msturroc.github.io/VariogramAnalysis.jl",
        edit_link="main",
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/msturroc/VariogramAnalysis.jl",
    devbranch="main",
)
