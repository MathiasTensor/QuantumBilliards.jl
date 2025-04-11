"""
    use_threads(args...)
    
The macro expects either:
(a) A keyword argument "multithreading" followed by a loop expression, or
b) A lone loop expression (in which case multithreading defaults to true).
"""
macro use_threads(args...)
    if length(args)>=2 && args[1] isa Expr && args[1].head==:(=) && args[1].args[1]==:multithreading
        if length(args)!=2
            error("Usage: @use_threads multithreading=[true|false] for ...")
        end
        # Extract the provided multithreading value and the loop expression.
        multithreading_val=args[1].args[2]
        loop_expr=args[2]
        # If the provided value is literally true or false, we branch accordingly at compile time.
        if multithreading_val===true
            return esc(:(@inbounds Threads.@threads $loop_expr))
        elseif multithreading_val===false
            return esc(:(@inbounds $loop_expr))
        else
            # If not a literal (i.e. it's some expression that will be evaluated at runtime),
            # we generate code that conditionally selects the threaded version.
            return esc(quote
                @inbounds begin
                    if $multithreading_val
                        Threads.@threads $loop_expr
                    else
                        $loop_expr
                    end
                end
            end)
        end
    elseif length(args)==1
        # No keyword argument provided. Default behavior is multithreading=true.
        loop_expr=args[1]
        return esc(:(@inbounds Threads.@threads $loop_expr))
    else
        error("Usage: @use_threads [multithreading=[true|false]] for ...")
    end
end