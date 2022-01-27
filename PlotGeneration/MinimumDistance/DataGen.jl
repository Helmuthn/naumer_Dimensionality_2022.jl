using Random
using naumer_ICML_2022
using CSV

rng = MersenneTwister(4321)

const max_samples = 10000

dataset1 = samplePSD(rng,max_samples,1,1)
dataset2 = samplePSD(rng,max_samples,2,1)
dataset3 = samplePSD(rng,max_samples,3,1)

function minimumDistance(sample, dataset)
    return minimum(sqrt.(sum(abs2, dataset .- sample,dims=(1,2))))
end

function EstimateMinDist(rng, dataset, N, iterations)
    total = 0
    for i in 1:iterations
        mini_dataset = dataset[:,:,randperm(max_samples)[1:(N+1)]]
        chosen_index = rand(1:(N+1))
        chosen_sample = mini_dataset[:,:,chosen_index]
        mini_dataset = cat(mini_dataset[:,:,1:chosen_index-1],mini_dataset[:,:,chosen_index+1:end]; dims=3)
        dist = minimumDistance(chosen_sample, mini_dataset)
        total += dist
    end
    return total/iterations
end


dist1 = zeros(500)
dist2 = zeros(500)
dist3 = zeros(500)

Threads.@threads for i in 1:500
    dist1[i] = EstimateMinDist(rng, dataset1, i, 5000)
    dist2[i] = EstimateMinDist(rng, dataset2, i, 5000)
    dist3[i] = EstimateMinDist(rng, dataset3, i, 5000)
end


CSV.write("SamplingData.csv",(dist1 = dist1, dist2 = dist2, dist3 = dist3))
