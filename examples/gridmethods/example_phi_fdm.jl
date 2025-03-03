using QuantumBilliards, CairoMakie, LinearAlgebra, Printf, StaticArrays

# One needs to create a phi(x,y) function that acts as a boolean mask (0 or 1 depending whether it is inside the boundary or outside) for the same geometry as the one defined as the <:AbsBilliard struct as. It needs to be externally given as the method itself needs to calculate for wanted cell (i,j) the values there

# RECTANGLE
#=
function rectangle_phi(x, y, width, height)
    dx=abs(x)-width/2
    dy=abs(y)-height/2
    return max(dx,dy) < 0 ? max(dx,dy) : sqrt(max(dx,0)^2+max(dy,0)^2)
end
# Define the rectangle bounds
width=2.0
height=1.0
# Create the signed distance function for this rectangle
phi(x,y)=rectangle_phi(x,y,width,height)
billiard,_=make_rectangle_and_basis(width,height)
fundamental=false
=#

# ELLIPSE
#=
function ellipse_phi(x,y,a,b)
    return (x/a)^2+(y/b)^2-1.0
end
# Define the ellipse parameters
w_e,h_e=2.0,1.0  # Semi-major and semi-minor axes
# Create the signed distance function for this ellipse
phi(x,y)=ellipse_phi(x,y,w_e,h_e)
billiard,_=make_ellipse_and_basis(w_e,h_e)
fundamental=false
=#

# PROSEN
#=
function prosen_phi(x,y,a)
    φ=atan(y,x)+pi
    r=1.0+a*cos(4*φ)
    return x^2+y^2<r^2
end
phi(x,y)=prosen_phi(x,y,0.4)
billiard,_=make_prosen_and_basis(0.4)
fundamental=false
=#

# EQUILATERAL TRIANGLE
#=
function equilateral_triangle_phi(x,y,h,x0=0.0,y0=0.0,rot_angle=0.0)
    p1=SVector(h,0.0) # Right vertex
    angle=2π/3                   
    p2=SVector(h*cos(angle),h*sin(angle)) # Top-left vertex
    p3=SVector(0.0,0.0) # Bottom-left vertex 
    function rotate_point(p, angle)  # Apply rotation if needed
        c,s=cos(angle),sin(angle)
        return SVector(c*p[1]-s*p[2],s*p[1]+c*p[2])
    end
    p1=rotate_point(p1,rot_angle).+SVector(x0,y0)
    p2=rotate_point(p2,rot_angle).+SVector(x0,y0)
    p3=rotate_point(p3,rot_angle).+SVector(x0,y0)
    # Triangle area test to check if (x, y) is inside
    function triangle_area(a,b,c)
        return abs((a[1]*(b[2]-c[2])+b[1]*(c[2]-a[2])+c[1]*(a[2]-b[2]))/2)
    end
    A=triangle_area(p1,p2,p3)
    A1=triangle_area(SVector(x,y),p2,p3)
    A2=triangle_area(p1,SVector(x,y),p3)
    A3=triangle_area(p1,p2,SVector(x,y))
    return abs(A-(A1+A2+A3))<1e-10  # Point inside if area check holds
end
phi(x,y)=equilateral_triangle_phi(x,y,1.0)
billiard,_=make_equilateral_triangle_and_basis(1.0)
fundamental=true
=#

# STADIUM 

function stadium_phi(x,y,half_width,height)
    radius=height # Radius of the semicircles
    # Case 1: Inside the rectangle part
    if abs(x)<=half_width && abs(y)<=radius
        return true
    end
    # Case 2: Inside the semicircles (left or right)
    if abs(x)>half_width
        circle_x=half_width*(x<0 ? -1 : 1)  # Closest semicircle center
        distance_to_circle=sqrt((x-circle_x)^2+y^2)
        return distance_to_circle<=radius  # Inside semicircle
    end
    # Outside the stadium
    return false
end
half_width=1.0 # the full width
radius=1.0
phi(x,y)=stadium_phi(x,y,half_width,radius)
billiard,_=make_stadium_and_basis(half_width,radius=radius)
fundamental=false


# Create the FDM struct
fem=QuantumBilliards.FiniteElementMethod(billiard,300,300;k_max=1000.0,offset_x_symmetric=0.1,offset_y_symmetric=0.1,fundamental=fundamental)
# needed for wavefunction plotting
x_grid,y_grid=fem.x_grid,fem.y_grid
# to see the actual dimension of the Sparse Hamiltonian
println("Interior grid pts: ",fem.Q)
f=Figure()
ax=Axis(f[1,1])
mask=[phi(x,y) for x in x_grid, y in y_grid]
heatmap!(ax,x_grid,y_grid,mask)
save("MASK.png",f)

# Define penalization and stabilization parameters:
γ=5.0 # higher means more strictly enforced BCs but more ill conditioned matrix. One needs to play with this.
σ=1.01 # higher means more strictly enforced BCs but more ill conditioned matrix. One needs to play with this.
# compute the ϕ-FD Hamiltonian:
H_phiFD=phiFD_Hamiltonian(fem,phi,γ,σ)

nev=2000 # number of eigenvalues -> ideally should be 2%-5% of Nx*Ny due to precision problems arising due to discretization for high lying nodes. This should guarantee a relative accuracy of <0.1%
ks,wavefunctions=compute_ϕ_fem_eigenmodes(fem,phi,γ,σ,nev=nev,maxiter=50000,tol=1e-8)
ks=sqrt.(abs.(ks)) # need to transform the E to the k based on m and ℏ
wavefunctions=[abs2.(wf) for wf in wavefunctions] # if wanting to plot the |Ψ(x,y)|^2
fs=QuantumBilliards.plot_wavefunctions(ks,wavefunctions,x_grid,y_grid,billiard,fundamental=fundamental)
for i in eachindex(fs)
    save("φ_$(nameof(typeof(billiard)))_$(i)_package.png",fs[i])
end

# check wavefunctions badness
boundary_mask=compute_boundary(fem.interior_idx,fem.Nx,fem.Ny)
tensions=[compute_boundary_tension(Ψ,boundary_mask) for Ψ in wavefunctions]
f=Figure()
ax=Axis(f[1,1])
scatter!(ax,1:length(tensions),tensions)
save("φ_tensions_$(nameof(typeof(billiard)))_package.png",f)