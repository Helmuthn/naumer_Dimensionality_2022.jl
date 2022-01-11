using CairoMakie
using CSV

f = CSV.File("HopfSampling.csv")

random_sampling_trace = f.random_sampling_trace .+ eps()
optimal_sampling_trace = f.optimal_sampling_trace .+ eps()
samples = f.samples


noto_sans = "../resources/NotoSans-Regular.ttf"

tickfontsize    = 28
labelfontsize   = 32

f = Figure(font=noto_sans, resolution=(800,600))

ax = Axis(  f[1,1],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="Sample", ylabel = "Tr(Σ)",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
            yscale = log10,
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(10),
            xminorticksvisible = true,
            xminorgridvisible = true,
            xminorticks = IntervalsBetween(10))

l1 = lines!(ax, samples, optimal_sampling_trace, color=:black)
l2 = lines!(ax, samples, random_sampling_trace, color=:black, linestyle=:dash)

Legend(f[1,2], [l1,l2], ["Our Method", "Random"])

ylims!(ax,(1e-3,1))
xlims!(ax,(0,samples[end]))

save("HopfSampling.pdf",f)
