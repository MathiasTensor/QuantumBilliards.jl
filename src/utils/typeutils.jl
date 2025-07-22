function set_precision(a)
    t=typeof(a)
    return t==Float32 ? Float32(1e-8) : convert(t,1e-16) 
end

"""
    update_field(x::T, fname::Symbol, newval) where T

Return a new instance of struct `typeof(x)` just like `x` but with field `fname` set to `newval`.
"""
function update_field(x::T,fname::Symbol,newval) where T
    names=fieldnames(T)
    idx=findfirst(==(fname), names)
    idx===nothing && throw(ArgumentError("no field $(fname) in type $(T)"))
    values=ntuple(i->i==idx ? newval : getfield(x,names[i]),length(names))
    return T(values...)
end