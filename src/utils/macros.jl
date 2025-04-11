"""
    @use_threads(condition, loop)

Macro to conditionally enable multithreading via `Threads.@threads` based on `condition`.
"""
macro use_threads(condition,loop)
    quote
        if $(esc(condition))
            Threads.@threads $(esc(loop))
        else
            $(esc(loop))
        end
    end
end