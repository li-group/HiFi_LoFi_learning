# How to run this code?
1. Installation: 

In julia do the following:
```julia
using Pkg
Pkg.activate("Gnn_environment")
Pkg.instantiate()
```
2. To generate nominal samples, run generate_base_instance.jl
3. To generate training samples and true output for semisupervised learning,  run generate_instances_variations.jl
4. To train GNNs, run train_selfsupervised.jl and train_semisupervised.jl
5. To run inference, run inference.jl
