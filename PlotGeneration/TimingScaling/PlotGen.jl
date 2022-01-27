using CSV
using CairoMakie
using Colors

color_lst = [colorant"#332288" colorant"#555555" colorant"#117733" colorant"#882255"]

f = CSV.File("DimensionTiming.csv")

timing = f.timing

f = CSV.File("DimensionTimingDynamic.csv")

timingDynamic = f.timing
dimensions = f.samples

noto_sans = "../resources/NotoSans-Regular.ttf"
noto_sans_bold = "../resources/NotoSans-Bold.ttf"

tickfontsize = 24
labelfontsize = 28

f = Figure(resolution = (1100,500), font=noto_sans)

ax = Axis(  f[1,1],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="Ambient Dimension", ylabel = "Seconds",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
#            yscale = log10,
#            xscale = log10,
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(10),
            xminorticksvisible = true,
            xminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            title="1D Limit Set",
            titlesize=labelfontsize)

l1 = lines!(ax, 2 .+ (1:length(timing)), timing, color = :black, linewidth=2)

ax2 = Axis(  f[1,2],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="Sample Count", ylabel = "Seconds",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
#            yscale = log10,
#            xscale = log10,
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(10),
            xminorticksvisible = true,
            xminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            title="Dynamic Programming",
           titlesize=labelfontsize)

l2 = lines!(ax2, dimensions, timingDynamic, color = :black, linewidth=2)

colgap!(f.layout,100)

Label(f[1,1,TopLeft()], "A", font=noto_sans_bold, textsize = 40, halign=:left, valign=:bottom, padding=(25,0,10,0))
Label(f[1,2,TopLeft()], "B", font=noto_sans_bold, textsize = 40, halign=:left, valign=:bottom, padding=(25,0,10,0))

save("TimingPlot.pdf",f)
