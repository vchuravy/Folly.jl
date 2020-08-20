using Folly

function matmul!(A, B, C)
    for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            A[i, j] = zero(eltype(A))
            for k in 1:size(B, 2)
                A[i, j] += B[i, k] * C[k, j]
            end 
        end 
    end 
end

loopnest = Loop(
    :i, [
        Loop(
            :j, [
                :(A[i, j] = zero(eltype(A))),
                Loop(:k, [
                    :(A[i, j] += B[i, k] * C[k, j])
                ])])
        ]
    )
indvars = [
    IndVar(:i, Outer(), :(i += 1), 1, :(size(A, 1))),
    IndVar(:j, Outer(), :(j += 1), 1, :(size(A, 2))),
    IndVar(:k, Inner(), :(k += 1), 1, :(size(B, 2))),
]

matmul2! = Folly.compile(Native(), loopnest, indvars, [:A, :B, :C])
matmul_ka! = Folly.compile(KATarget(CPU()), loopnest, indvars, [:A, :B, :C])

using Test

B = rand(64, 32)
C = rand(32, 64)
A = zeros(64, 64)

matmul!(A, B, C) 
@test A ≈ B*C

A = zeros(64, 64)
matmul2!(A, B, C) 
@test A ≈ B*C

A = zeros(64, 64)
event = matmul_ka!(A, B, C)
wait(event)
@test A ≈ B*C
