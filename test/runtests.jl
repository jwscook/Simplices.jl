using Random, Simplices, Test
using Simplices: Vertex, Simplex, windingnumber, windingangle
using Simplices: centre, assessconvergence, position, value
using Simplices: bestvertex, issortedbyangle, hypervolume

@testset "Simplices tests" begin

  Random.seed!(0)

  @testset "Vertices" begin
    v1 = Vertex([0.0, 1.0], 2.0 - 3.0im)
    v2 = Vertex([0.0, 1.0], 2.0 - 3.0im)
    @test !(v1 == v2)
    @test isequal(v1, v2)
  end

  @testset "Simplex winding number" begin

    irrelevant = [0.0, 0.0]
    @testset "Simplex encloses zero" begin
      v1 = Vertex(irrelevant, 1.0 - im)
      v2 = Vertex(irrelevant, 0.0 + im)
      v3 = Vertex(irrelevant, -1.0 - im)
      encloseszero = Simplex([v1, v2, v3])
      @test isapprox(1, abs(windingangle(encloseszero)) / (2π))
      @test windingnumber(encloseszero) == 1
    end

    @testset "Simplex doesn't enclose zero" begin
      v1 = Vertex(irrelevant, 1.0 + im)
      v2 = Vertex(irrelevant, 2.0 + im)
      v3 = Vertex(irrelevant, 1.0 + im * 2)
      doesntenclosezero = Simplex([v1, v2, v3])
      @test abs(windingangle(doesntenclosezero)) / (2π) < 1.0e-3
      @test windingnumber(doesntenclosezero) == 0
      v1 = Vertex(irrelevant, 1.0 + im)
      v2 = Vertex(irrelevant, 0.0 + 2*im)
      v3 = Vertex(irrelevant, -1.0 + im)
      doesntenclosezero = Simplex([v1, v2, v3])
      @test abs(windingangle(doesntenclosezero)) / (2π) < 1.0e-3
      @test windingnumber(doesntenclosezero) == 0
      v1 = Vertex(irrelevant, 1.0 + 0im)
      v2 = Vertex(irrelevant, 0.0 + im)
      v3 = Vertex(irrelevant, 1.0 + 0im)
      doesntenclosezero = Simplex([v1, v2, v3])
      @test abs(windingangle(doesntenclosezero)) / (2π) < 1.0e-3
      @test windingnumber(doesntenclosezero) == 0
    end
  end

  @testset "Simplex convergence" begin
    @testset "Simplices with identical vertices is converged" begin
      dim = 2
      T = Float64
      U = ComplexF64
      pos = T[1.0, 1.0]
      val = rand(U)
      v1 = Vertex(pos, val)
      v2 = Vertex(pos, val)
      v3 = Vertex(pos, val)
      s = Simplex([v1, v2, v3])

      defaults = Simplices.convergenceconfig(dim, T)
      returncode = assessconvergence(s, defaults)
      @test returncode == :XTOL_REACHED
    end

    @testset "Simplices in a simplex eps apart are converged" begin
      dim = 2
      T = Float64
      U = ComplexF64
      v1 = Vertex([one(T), one(T)], rand(U))
      v2 = Vertex([one(T), one(T) + eps(T)], rand(U))
      v3 = Vertex([one(T) + eps(T), one(T)], rand(U))
      s = Simplex([v1, v2, v3])
      defaults = Simplices.convergenceconfig(dim, T)
      returncode = assessconvergence(s, defaults)
      @test returncode == :XTOL_REACHED
    end

    @testset "Simplices in a chain eps apart are converged" begin
      dim = 2
      T = Float64
      U = ComplexF64
      v1 = Vertex([one(T), one(T)], rand(U))
      v2 = Vertex([one(T), one(T) + eps(T)], rand(U))
      v3 = Vertex([one(T), one(T) + eps(T) + eps(T)], rand(U))
      s = Simplex([v1, v2, v3])
      defaults = Simplices.convergenceconfig(dim, T)
      returncode = assessconvergence(s, defaults)
      @test returncode == :XTOL_REACHED
      @assert all(isapprox(position(v1)[d], position(v2)[d],
                        atol=defaults[:xtol_abs][d],
                        rtol=defaults[:xtol_rel][d]) for d ∈ 1:dim)
      @assert all(isapprox(position(v2)[d], position(v3)[d],
                        atol=defaults[:xtol_abs][d],
                        rtol=defaults[:xtol_rel][d]) for d ∈ 1:dim)
      @assert !all(isapprox(position(v1)[d], position(v3)[d],
                        atol=defaults[:xtol_abs][d],
                        rtol=defaults[:xtol_rel][d]) for d ∈ 1:dim)
      @test returncode == :XTOL_REACHED
    end

    @testset "Simplices vertices are in order" begin
      for _ ∈ 1:10
        dim = 2
        T = Float64
        U = ComplexF64
        v1 = Vertex(rand(dim), rand(U))
        v2 = Vertex(rand(dim), rand(U))
        v3 = Vertex(rand(dim), rand(U))
        s = Simplex([v1, v2, v3])
        p = s.permabs
        @test abs(value(s[p[1]])) < abs(value(s[p[2]]))
        @test abs(value(s[p[2]])) < abs(value(s[p[3]]))
        @test issortedbyangle(s)
      end
    end
  end

  @testset "Shapes" begin
    @testset "centres" begin
      v1 = Vertex([0.0, 0.0], one(ComplexF64))
      v2 = Vertex([1.0, 0.0], one(ComplexF64))
      v3 = Vertex([0.0, 1.0], one(ComplexF64))
      s = Simplex([v1, v2, v3])
      @test all(centre(s) .== [1/3, 1/3])
    end

    @testset "extremas" begin
      s = Simplex(x->im, [1.0, 3.0], 1.0)
      exs = Simplices.extrema(s)
      @test exs[1] == (1.0, 2.0)
      @test exs[2] == (3.0, 4.0)
    end

    @testset "hypervolumes" begin
      x0 = rand(2)
      a, b = rand(2)
      v1 = Vertex(x0 .+ [0.0, 0.0], one(ComplexF64))
      v2 = Vertex(x0 .+ [a, 0.0], one(ComplexF64))
      v3 = Vertex(x0 .+ [0.0, b], one(ComplexF64))
      s = Simplex([v1, v2, v3])
      @test hypervolume(s) ≈ (a * b) / 2
      x0 = rand(3)
      a, b, c = rand(3)
      v1 = Vertex(x0 .+ [0, 0, 0], one(ComplexF64))
      v2 = Vertex(x0 .+ [a, 0, 0], one(ComplexF64))
      v3 = Vertex(x0 .+ [0, b, 0], one(ComplexF64))
      v4 = Vertex(x0 .+ [0, 0, c], one(ComplexF64))
      s = Simplex([v1, v2, v3, v4])
      @test hypervolume(s) ≈ (a * b * c) / 6
    end
  end

  @testset "Partition of hypercubes" begin
    @testset "unity" begin
      for dims in 1:6
        positions = Simplices.partitionunitypositions(dims, Float64)
        ss = [Simplex(x->one(ComplexF64), [p for p in ps]) for ps in positions]
        for i ∈ 1:10
          x = rand(dims)
          xins = [x ∈ s for s in ss]
          @test sum(xins) == 1 # can only exist in 1 simplex
        end
      end
    end
  @testset "random hypercube" begin
      for dims in 1:6
        a = rand(dims) .- 0.5
        b = a .+ rand(dims)
        ss = Simplices.partitionhypercube(x->1, a, b)
        for i ∈ 1:10
          x = a .+ rand(dims) .* (b .- a)
          xins = [x ∈ s for s in ss]
          @test sum(xins) == 1 # can only exist in 1 simplex
        end
      end
    end
   @testset "partition simplices don't overlap" begin
      for dims in 1:4 # anything higher is too expensive
        ss = Simplices.partitionhypercube(x->1, zeros(dims), ones(dims))
        for a in ss, b in ss
          if isequal(a, b)
            @test a ∈ b
          else
            @test !(a ∈ b)
          end
        end
      end
    end
  end

end

