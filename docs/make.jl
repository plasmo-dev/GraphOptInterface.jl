using GraphOptBase
using Documenter

DocMeta.setdocmeta!(GraphOptBase, :DocTestSetup, :(using GraphOptBase); recursive=true)

makedocs(;
    modules=[GraphOptBase],
    authors="jalving <jhjalving@gmail.com> and contributors",
    sitename="GraphOptBase.jl",
    format=Documenter.HTML(;
        canonical="https://jalving.github.io/GraphOptBase.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jalving/GraphOptBase.jl",
    devbranch="main",
)
