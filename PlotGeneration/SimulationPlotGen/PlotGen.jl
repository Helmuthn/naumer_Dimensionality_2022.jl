using CairoMakie
using CSV
using Colors

color_lst = [colorant"#332288" colorant"#555555" colorant"#117733" colorant"#882255"]
# Blue, Grey, Green, Red, Yellow
# Colorblind friendly color palette

f = CSV.File("StaticSampling/StaticSampling.csv")

static_random_sampling_trace = f.random_sampling_trace 
static_optimal_sampling_trace = f.optimal_sampling_trace

f = CSV.File("LinearSampling/LinearSampling.csv")

Linear_random_sampling_trace = f.random_sampling_trace 
Linear_optimal_sampling_trace = f.optimal_sampling_trace
Linear_approx_sampling_trace  = f.approx_sampling_trace
samples = f.samples

f = CSV.File("HopfSampling/HopfSampling.csv")

Hopf_random_sampling_trace  = f.random_sampling_trace 
Hopf_optimal_sampling_trace = f.optimal_sampling_trace
Hopf_approx_sampling_trace  = f.approx_sampling_trace
Hopf_samples = f.samples


f = CSV.File("LorenzSampling/LorenzSampling.csv")

Lorenz_random_sampling_trace = f.random_sampling_trace 
Lorenz_optimal_sampling_trace = f.optimal_sampling_trace
Lorenz_approx_sampling_trace = f.approx_sampling_trace
Lorenz_samples = f.samples

max_steps = length(samples)


noto_sans = "./resources/NotoSans-Regular.ttf"
noto_sans_bold = "./resources/NotoSans-Bold.ttf"

tickfontsize    = 32
labelfontsize   = 38
basewidth = 2


f = Figure(font=noto_sans, resolution=(1200,1100), figure_padding=40)


ax0 = Axis( f[1,1],
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
            xminorticks = IntervalsBetween(10),
            title="Static Value",
            titlesize=labelfontsize)

l2 = lines!(ax0, 1:max_steps, static_random_sampling_trace, color=color_lst[1], linewidth=basewidth+2)
l1 = lines!(ax0, 1:max_steps, static_optimal_sampling_trace, color=color_lst[3], linewidth=basewidth)

ylims!(ax0,(5e-4,1))
xlims!(ax0,(0,1000))

ax1 = Axis(  f[1,2],
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
            xminorticks = IntervalsBetween(10),
            title="Linear System",
            titlesize=labelfontsize)

ylims!(ax1,(5e-4,1))
xlims!(ax1,(0,1000))

l2 = lines!(ax1, 1:max_steps, Linear_random_sampling_trace, color=color_lst[1],linewidth=basewidth)
l3 = lines!(ax1, 1:max_steps, Linear_random_sampling_trace ./ 2, color=color_lst[2], linewidth=basewidth+2)
l1 = lines!(ax1, 1:max_steps, Linear_optimal_sampling_trace, color=color_lst[3], linewidth=basewidth)
l4 = lines!(ax1, 1:max_steps, Linear_approx_sampling_trace, color=color_lst[4], linewidth=basewidth)


ax2= Axis(  f[2,1],
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
            xminorticks = IntervalsBetween(10),
            title="Hopf Bifurcation",
            titlesize=labelfontsize)


ylims!(ax2,(5e-4,1))
xlims!(ax2,(0,1000))

l2 = lines!(ax2, 1:max_steps, Hopf_random_sampling_trace,      color=color_lst[1], linewidth=basewidth)
l3 = lines!(ax2, 1:max_steps, Hopf_random_sampling_trace ./ 2, color=color_lst[2], linewidth=basewidth+2)
l1 = lines!(ax2, 1:max_steps, Hopf_optimal_sampling_trace,     color=color_lst[3], linewidth=basewidth)
l4 = lines!(ax2, 1:max_steps, Hopf_approx_sampling_trace,      color=color_lst[4], linewidth=basewidth)



ax3= Axis(  f[2,2],
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
            xminorticks = IntervalsBetween(10),
            title="Lorenz System",
            titlesize=labelfontsize)

ylims!(ax3,(3e-3,0.1))
xlims!(ax3,(0,1000))

l2 = lines!(ax3, 1:max_steps, Lorenz_random_sampling_trace,           color=color_lst[1], linewidth=basewidth)
l3 = lines!(ax3, 1:max_steps, Lorenz_random_sampling_trace .* 2 ./ 3, color=color_lst[2], linewidth=basewidth+2)
l1 = lines!(ax3, 1:max_steps, Lorenz_optimal_sampling_trace,          color=color_lst[3], linewidth=basewidth)
l4 = lines!(ax3, 1:max_steps, Lorenz_approx_sampling_trace,           color=color_lst[4], linewidth=basewidth)


Legend(f[3,:], 
       [l4, l1, l2, l3], 
       ["1D Approximation", "Dynamic Programming", "Random Sampling", "Dimension Reduction"], 
       orientation=:horizontal,
       nbanks=2,
       labelsize=36)


colgap!(f.layout,60)

Label(f[1,1,TopLeft()], "A", font=noto_sans_bold, textsize = 40, halign=:left, valign=:bottom, padding=(25,0,10,0))
Label(f[1,2,TopLeft()], "B", font=noto_sans_bold, textsize = 40, halign=:left, valign=:bottom, padding=(25,0,10,0))
Label(f[2,1,TopLeft()], "C", font=noto_sans_bold, textsize = 40, halign=:left, valign=:bottom, padding=(25,0,10,0))
Label(f[2,2,TopLeft()], "D", font=noto_sans_bold, textsize = 40, halign=:left, valign=:bottom, padding=(25,0,10,0))

save("out/FinalSim.pdf",f)
