using GraphOptInterface
using Documenter

DocMeta.setdocmeta!(
    GraphOptInterface, :DocTestSetup, :(using GraphOptInterface); recursive=true
)

makedocs(;
    modules=[GraphOptInterface],
    authors="Jordan Jalving, Sungho Shin, and contributors",
    sitename="GraphOptInterface.jl",
    format=Documenter.HTML(;
        canonical="https://plasmo-dev.github.io/GraphOptInterface.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/plasmo-dev/GraphOptInterface.jl", devbranch="main")
