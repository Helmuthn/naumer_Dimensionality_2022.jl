using CairoMakie
using CSV

f = CSV.File("LinearSampling.csv")

random_sampling_trace = f.random_sampling_trace 
optimal_sampling_trace_low = f.optimal_sampling_trace_low
optimal_sampling_trace_medium = f.optimal_sampling_trace_medium
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

l1 = lines!(ax, 1:max_steps, optimal_sampling_trace_low, color=:black)
l2 = lines!(ax, 1:max_steps, random_sampling_trace, color=:black, linestyle=:dash)
l3 = lines!(ax, 1:max_steps, random_sampling_trace ./ 2, color=:black, linestyle = :dot)

Legend(f[1,2], [l1,l2,l3], ["Our Method", "Random", "Optimal CRLB \n    (Estimate)"])

ylims!(ax,(1e-3,1))
xlims!(ax,(0,1000))

save("LinearSampling.pdf",f)
