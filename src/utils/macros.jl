"""
    @use_threads(condition, loop)

Macro to conditionally enable multithreading via `Threads.@threads` based on `condition`.
"""
macro use_threads(condition,loop)
    return quote
        if $condition
            Threads.@threads $loop
        else
            $loop
        end
    end
end