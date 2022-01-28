using CSV
using CairoMakie
using Colors

color_lst = [colorant"#332288" colorant"#555555" colorant"#117733" colorant"#882255"]

f = CSV.File("SamplingData.csv")

dist1 = f.dist1
dist2 = f.dist2
dist3 = f.dist3

noto_sans = "../resources/NotoSans-Regular.ttf"

tickfontsize = 28
labelfontsize = 34

f = Figure(resolution = (750,450), font=noto_sans)

ax = Axis(  f[1,1],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="Sample Count", ylabel = "E[Min Dist]",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
            yscale = log10,
            xscale = log10,
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(10),
            xminorticksvisible = true,
            xminorgridvisible = true,
            xminorticks = IntervalsBetween(10))

l1 = lines!(ax, 1:length(dist1), dist1, color = color_lst[1], linewidth=3)
l2 = lines!(ax, 1:length(dist2), dist2, color = color_lst[2], linewidth=3)
l3 = lines!(ax, 1:length(dist3), dist3, color = color_lst[3], linewidth=3)

l1elem = LineElement(color=color_lst[1], linewidth=10)
l2elem = LineElement(color=color_lst[2], linewidth=10)
l3elem = LineElement(color=color_lst[3], linewidth=10)

xlims!(ax,(1,length(dist1)))

Legend(f[1,2], [l3elem,l2elem,l1elem], ["3D", "2D", "1D"], labelsize=labelfontsize)

save("SamplingPlot.pdf",f)
