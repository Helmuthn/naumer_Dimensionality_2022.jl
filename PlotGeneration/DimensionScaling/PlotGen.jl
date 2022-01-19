using CSV
using CairoMakie
using Colors

color_lst = [colorant"#332288" colorant"#555555" colorant"#117733" colorant"#882255"]

f = CSV.File("DimensionScaling.csv")

crlb_trace = f.crlb_trace
random_crlb_trace = f.random_crlb_trace


noto_sans = "../resources/NotoSans-Regular.ttf"

tickfontsize = 24
labelfontsize = 28

f = Figure(resolution = (800,600), font=noto_sans)

ax = Axis(  f[1,1],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="Ambient Dimension", ylabel = "Tr(Σ)",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
#            yscale = log10,
#            xscale = log10,
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(10),
            xminorticksvisible = true,
            xminorgridvisible = true,
            xminorticks = IntervalsBetween(10))

scaling = sum(crlb_trace)/length(crlb_trace)
l1 = lines!(ax, 2 .+ (1:length(crlb_trace)), crlb_trace/scaling,        color = color_lst[4], linewidth=2)
l2 = lines!(ax, 2 .+ (1:length(crlb_trace)), random_crlb_trace/scaling, color = color_lst[1], linewidth=2)

Legend(f[2,1], [l1,l2], ["1D Approximation", "Random Sampling"], orientation=:horizontal, labelsize=labelfontsize)

save("ScalingPlot.pdf",f)
